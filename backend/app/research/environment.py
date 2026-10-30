"""
Fatigue-Aware Nudging Research Platform — Environment Module

Implements the health-nudging MDP simulator with habituation dynamics.
Based on the paper: "Fatigue-Aware Nudging: State Augmentation Prevents
Habituation Failures in DP, MC and Q-Learning Health Interventions."

Scientific specification:
- State: (sleep, activity, stress, habituation) each discretized
- Actions: {no_nudge, sleep_nudge, activity_nudge, stress_nudge}
- Habituation attenuates response probability over repeated nudging
- Episode length: 28 days (configurable)
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
from enum import IntEnum


class Action(IntEnum):
    """Nudge actions available to the agent."""
    NO_NUDGE = 0
    SLEEP_NUDGE = 1
    ACTIVITY_NUDGE = 2
    STRESS_NUDGE = 3


class HealthDimension(IntEnum):
    """Health state dimensions."""
    SLEEP = 0
    ACTIVITY = 1
    STRESS = 2


@dataclass
class SimulatorConfig:
    """Configuration for the health-nudging simulator.
    
    Default values follow the paper's experimental setup.
    All probabilities and parameters are documented with their scientific meaning.
    """
    # State space bounds
    health_levels: int = 3        # 0, 1, 2 for each health dimension
    habituation_levels: int = 4   # 0, 1, 2, 3
    
    # Episode settings
    episode_length: int = 28      # days per episode
    discount_factor: float = 0.99
    
    # Response probabilities (base, before habituation attenuation)
    p_respond_sleep: float = 0.6
    p_respond_activity: float = 0.5
    p_respond_stress: float = 0.55
    
    # Habituation dynamics
    p_habituation_increase: float = 0.3   # P(h increases | nudge delivered)
    p_habituation_decay: float = 0.2      # P(h decreases | no nudge)
    habituation_attenuation_rate: float = 0.25  # Response decay per habituation level
    min_response_probability: float = 0.1       # Floor on response probability
    
    # Natural health drift (daily probability of each dimension decreasing)
    p_drift_sleep: float = 0.1
    p_drift_activity: float = 0.1
    p_drift_stress: float = 0.1
    
    # Initial state (None = random)
    initial_sleep: Optional[int] = None
    initial_activity: Optional[int] = None
    initial_stress: Optional[int] = None
    initial_habituation: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SimulatorConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HealthState:
    """Full environment state including habituation."""
    sleep: int = 1       # 0=poor, 1=fair, 2=good
    activity: int = 1    # 0=low, 1=moderate, 2=high
    stress: int = 1      # 0=high_stress, 1=moderate, 2=low_stress
    habituation: int = 0 # 0=none, 1=mild, 2=moderate, 3=severe
    
    def observable_tuple(self) -> Tuple[int, int, int]:
        """Memoryless state representation (no habituation)."""
        return (self.sleep, self.activity, self.stress)
    
    def full_tuple(self) -> Tuple[int, int, int, int]:
        """Fatigue-aware state representation (includes habituation)."""
        return (self.sleep, self.activity, self.stress, self.habituation)
    
    def clone(self) -> 'HealthState':
        return HealthState(self.sleep, self.activity, self.stress, self.habituation)
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)
    
    def health_score(self) -> float:
        """Compute normalized health score: (sleep + activity + stress) / 6.0"""
        return (self.sleep + self.activity + self.stress) / 6.0


@dataclass
class TransitionInfo:
    """Detailed information about a single environment step.
    
    Exposes internals so users can understand why transitions occurred.
    """
    day: int
    state_before: Dict[str, int]
    action: int
    action_name: str
    response_probability: float
    responded: bool
    reward: float
    state_after: Dict[str, int]
    habituation_before: int
    habituation_after: int
    drift_occurred: Dict[str, bool]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NudgingEnvironment:
    """Health-nudging MDP with habituation dynamics.
    
    The environment simulates a person receiving daily health nudges.
    Each day:
    1. Agent observes state and selects an action (nudge type).
    2. If nudge delivered, person responds with probability modulated by habituation.
    3. If response: targeted health dimension improves.
    4. Natural drift may decrease health dimensions.
    5. Habituation updates based on whether a nudge was delivered.
    6. Reward is computed from the resulting health state.
    """
    
    def __init__(self, config: Optional[SimulatorConfig] = None, seed: int = 42):
        self.config = config or SimulatorConfig()
        self.rng = np.random.RandomState(seed)
        self.seed = seed
        self.state: Optional[HealthState] = None
        self.day: int = 0
        self.done: bool = True
        self.episode_history: List[TransitionInfo] = []
        
        # Precompute action-to-dimension mapping
        self._action_to_dim = {
            Action.SLEEP_NUDGE: HealthDimension.SLEEP,
            Action.ACTIVITY_NUDGE: HealthDimension.ACTIVITY,
            Action.STRESS_NUDGE: HealthDimension.STRESS,
        }
        
        # Base response probabilities per nudge type
        self._base_response_prob = {
            Action.SLEEP_NUDGE: self.config.p_respond_sleep,
            Action.ACTIVITY_NUDGE: self.config.p_respond_activity,
            Action.STRESS_NUDGE: self.config.p_respond_stress,
        }
        
        # Drift probabilities per dimension
        self._drift_prob = {
            HealthDimension.SLEEP: self.config.p_drift_sleep,
            HealthDimension.ACTIVITY: self.config.p_drift_activity,
            HealthDimension.STRESS: self.config.p_drift_stress,
        }
    
    def reset(self, initial_state: Optional[HealthState] = None) -> HealthState:
        """Reset the environment to start a new episode.
        
        Args:
            initial_state: If provided, use this as the initial state.
                          Otherwise, use config defaults or random initialization.
        
        Returns:
            The initial health state.
        """
        self.day = 0
        self.done = False
        self.episode_history = []
        
        if initial_state is not None:
            self.state = initial_state.clone()
        else:
            s = self.config.initial_sleep
            a = self.config.initial_activity
            st = self.config.initial_stress
            h = self.config.initial_habituation
            
            if s is None:
                s = self.rng.randint(0, self.config.health_levels)
            if a is None:
                a = self.rng.randint(0, self.config.health_levels)
            if st is None:
                st = self.rng.randint(0, self.config.health_levels)
            
            self.state = HealthState(sleep=s, activity=a, stress=st, habituation=h)
        
        return self.state.clone()
    
    def get_response_probability(self, action: int, habituation: int) -> float:
        """Compute response probability for a given action and habituation level.
        
        P(respond | action, h) = p_base(action) * max(min_p, 1 - attenuation_rate * h)
        
        Args:
            action: The nudge action.
            habituation: Current habituation level.
            
        Returns:
            Response probability in [min_response_probability, p_base].
        """
        if action == Action.NO_NUDGE:
            return 0.0
        
        p_base = self._base_response_prob[Action(action)]
        decay = max(self.config.min_response_probability, 
                    1.0 - self.config.habituation_attenuation_rate * habituation)
        return p_base * decay
    
    def step(self, action: int) -> Tuple[HealthState, float, bool, TransitionInfo]:
        """Execute one day step in the environment.
        
        Args:
            action: The nudge action to take (0-3).
            
        Returns:
            next_state: The new health state.
            reward: The reward for this transition.
            done: Whether the episode has ended.
            info: Detailed transition information.
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action: {action}. Must be 0-3.")
        
        state_before = self.state.to_dict()
        habituation_before = self.state.habituation
        action_enum = Action(action)
        
        # 1. Compute response probability
        response_prob = self.get_response_probability(action, self.state.habituation)
        
        # 2. Determine if person responds
        responded = False
        if action_enum != Action.NO_NUDGE:
            responded = self.rng.random() < response_prob
        
        # 3. Apply nudge effect if responded
        if responded:
            dim = self._action_to_dim[action_enum]
            if dim == HealthDimension.SLEEP:
                self.state.sleep = min(self.config.health_levels - 1, self.state.sleep + 1)
            elif dim == HealthDimension.ACTIVITY:
                self.state.activity = min(self.config.health_levels - 1, self.state.activity + 1)
            elif dim == HealthDimension.STRESS:
                self.state.stress = min(self.config.health_levels - 1, self.state.stress + 1)
        
        # 4. Apply natural drift (each dimension may decrease)
        drift_occurred = {}
        for dim in HealthDimension:
            drift = self.rng.random() < self._drift_prob[dim]
            drift_occurred[dim.name.lower()] = drift
            if drift:
                if dim == HealthDimension.SLEEP:
                    self.state.sleep = max(0, self.state.sleep - 1)
                elif dim == HealthDimension.ACTIVITY:
                    self.state.activity = max(0, self.state.activity - 1)
                elif dim == HealthDimension.STRESS:
                    self.state.stress = max(0, self.state.stress - 1)
        
        # 5. Update habituation
        if action_enum != Action.NO_NUDGE:
            # Nudge delivered: habituation may increase
            if self.rng.random() < self.config.p_habituation_increase:
                self.state.habituation = min(
                    self.config.habituation_levels - 1, 
                    self.state.habituation + 1
                )
        else:
            # No nudge: habituation may decay
            if self.rng.random() < self.config.p_habituation_decay:
                self.state.habituation = max(0, self.state.habituation - 1)
        
        # 6. Compute reward
        reward = self._compute_reward(self.state)
        
        # 7. Advance day
        self.day += 1
        self.done = self.day >= self.config.episode_length
        
        # 8. Record transition
        info = TransitionInfo(
            day=self.day,
            state_before=state_before,
            action=action,
            action_name=action_enum.name,
            response_probability=response_prob,
            responded=responded,
            reward=reward,
            state_after=self.state.to_dict(),
            habituation_before=habituation_before,
            habituation_after=self.state.habituation,
            drift_occurred=drift_occurred,
        )
        self.episode_history.append(info)
        
        return self.state.clone(), reward, self.done, info
    
    def _compute_reward(self, state: HealthState) -> float:
        """Compute the reward from the paper's reward function.
        
        R = (sleep + activity + stress) / 6.0
        
        This normalizes the reward to [0, 1] range where:
        - 0 = all dimensions at worst (0, 0, 0)
        - 1 = all dimensions at best (2, 2, 2)
        """
        return (state.sleep + state.activity + state.stress) / 6.0
    
    def get_all_states(self, include_habituation: bool = True) -> List[Tuple]:
        """Enumerate all possible states.
        
        Args:
            include_habituation: If True, enumerate (s, a, σ, h) tuples.
                               If False, enumerate (s, a, σ) tuples.
        """
        states = []
        H = self.config.health_levels
        if include_habituation:
            for s in range(H):
                for a in range(H):
                    for st in range(H):
                        for h in range(self.config.habituation_levels):
                            states.append((s, a, st, h))
        else:
            for s in range(H):
                for a in range(H):
                    for st in range(H):
                        states.append((s, a, st))
        return states
    
    def get_num_states(self, include_habituation: bool = True) -> int:
        """Get the number of states."""
        H = self.config.health_levels
        n = H ** 3
        if include_habituation:
            n *= self.config.habituation_levels
        return n
    
    def get_num_actions(self) -> int:
        """Get the number of actions."""
        return len(Action)
    
    def get_transition_model(self, include_habituation: bool = True) -> Dict:
        """Build the complete transition model P(s'|s, a) for DP.
        
        Returns a dictionary mapping (state, action) -> list of (probability, next_state, reward).
        
        This uses the analytical transition probabilities rather than sampling.
        For the full fatigue-aware model, this includes habituation transitions.
        """
        states = self.get_all_states(include_habituation)
        n_actions = self.get_num_actions()
        H = self.config.health_levels
        hab_levels = self.config.habituation_levels
        
        model = {}
        
        for state_tuple in states:
            for action in range(n_actions):
                transitions = self._compute_analytical_transitions(
                    state_tuple, action, include_habituation
                )
                model[(state_tuple, action)] = transitions
        
        return model
    
    def _compute_analytical_transitions(
        self, state_tuple: Tuple, action: int, include_habituation: bool
    ) -> List[Tuple[float, Tuple, float]]:
        """Compute all possible transitions from (state, action) with probabilities.
        
        Returns list of (probability, next_state, reward) tuples.
        """
        if include_habituation:
            sleep, activity, stress, habituation = state_tuple
        else:
            sleep, activity, stress = state_tuple
            habituation = 0  # Memoryless assumes no habituation for response calc
        
        H = self.config.health_levels
        action_enum = Action(action)
        
        # Step 1: Response probability
        resp_prob = self.get_response_probability(action, habituation)
        
        # Build all possible outcomes as a list of (probability, next_health, next_hab)
        # We need to enumerate: response (yes/no) × drift (each dim yes/no) × hab change
        
        outcomes = {}
        
        # Response outcomes
        if action_enum == Action.NO_NUDGE:
            response_cases = [(1.0, False)]
        else:
            response_cases = [(resp_prob, True), (1.0 - resp_prob, False)]
        
        for p_resp, responded in response_cases:
            if p_resp < 1e-12:
                continue
                
            # After response, compute health state
            s_after = sleep
            a_after = activity
            st_after = stress
            
            if responded:
                dim = self._action_to_dim.get(action_enum)
                if dim == HealthDimension.SLEEP:
                    s_after = min(H - 1, s_after + 1)
                elif dim == HealthDimension.ACTIVITY:
                    a_after = min(H - 1, a_after + 1)
                elif dim == HealthDimension.STRESS:
                    st_after = min(H - 1, st_after + 1)
            
            # Drift outcomes: each dimension independently drifts or not
            drift_probs = [
                self.config.p_drift_sleep,
                self.config.p_drift_activity,
                self.config.p_drift_stress,
            ]
            
            for drift_mask in range(8):  # 2^3 combinations
                d_s = (drift_mask >> 0) & 1
                d_a = (drift_mask >> 1) & 1
                d_st = (drift_mask >> 2) & 1
                
                p_drift = 1.0
                p_drift *= (drift_probs[0] if d_s else (1 - drift_probs[0]))
                p_drift *= (drift_probs[1] if d_a else (1 - drift_probs[1]))
                p_drift *= (drift_probs[2] if d_st else (1 - drift_probs[2]))
                
                s_final = max(0, s_after - d_s)
                a_final = max(0, a_after - d_a)
                st_final = max(0, st_after - d_st)
                
                # Habituation transitions
                if include_habituation:
                    if action_enum != Action.NO_NUDGE:
                        # May increase
                        p_inc = self.config.p_habituation_increase
                        hab_cases = [
                            (p_inc, min(self.config.habituation_levels - 1, habituation + 1)),
                            (1 - p_inc, habituation),
                        ]
                    else:
                        # May decay
                        p_dec = self.config.p_habituation_decay
                        hab_cases = [
                            (p_dec, max(0, habituation - 1)),
                            (1 - p_dec, habituation),
                        ]
                    
                    for p_hab, h_next in hab_cases:
                        next_state = (s_final, a_final, st_final, h_next)
                        reward = (s_final + a_final + st_final) / 6.0
                        total_p = p_resp * p_drift * p_hab
                        
                        if total_p > 1e-15:
                            key = (next_state, reward)
                            outcomes[key] = outcomes.get(key, 0.0) + total_p
                else:
                    next_state = (s_final, a_final, st_final)
                    reward = (s_final + a_final + st_final) / 6.0
                    total_p = p_resp * p_drift
                    
                    # Still need to marginalize over habituation for memoryless
                    # But since memoryless doesn't track hab, we just use the total
                    if total_p > 1e-15:
                        key = (next_state, reward)
                        outcomes[key] = outcomes.get(key, 0.0) + total_p
        
        result = [(prob, state, rew) for (state, rew), prob in outcomes.items()]
        return result
    
    def state_to_index(self, state_tuple: Tuple, include_habituation: bool = True) -> int:
        """Convert state tuple to a flat index."""
        H = self.config.health_levels
        if include_habituation:
            s, a, st, h = state_tuple
            return ((s * H + a) * H + st) * self.config.habituation_levels + h
        else:
            s, a, st = state_tuple
            return (s * H + a) * H + st
    
    def index_to_state(self, index: int, include_habituation: bool = True) -> Tuple:
        """Convert flat index to state tuple."""
        H = self.config.health_levels
        if include_habituation:
            hab_levels = self.config.habituation_levels
            h = index % hab_levels
            index //= hab_levels
            st = index % H
            index //= H
            a = index % H
            s = index // H
            return (s, a, st, h)
        else:
            st = index % H
            index //= H
            a = index % H
            s = index // H
            return (s, a, st)
