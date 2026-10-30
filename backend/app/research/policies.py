"""
Policy implementations for the Fatigue-Aware Nudging platform.

Implements:
1. Rule-based baseline policy
2. Dynamic Programming (Value Iteration)
3. Monte Carlo Control (first-visit, epsilon-soft)
4. Tabular Q-Learning
5. Oracle DP (with full state access including habituation)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

from .environment import (
    NudgingEnvironment, SimulatorConfig, HealthState, Action, HealthDimension
)


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    value_deltas: List[float] = field(default_factory=list)  # For DP convergence
    exploration_rates: List[float] = field(default_factory=list)
    iterations: int = 0
    converged: bool = False
    convergence_iteration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BasePolicy(ABC):
    """Abstract base class for all policies."""
    
    def __init__(self, name: str, fatigue_aware: bool = False):
        self.name = name
        self.fatigue_aware = fatigue_aware
        self.trained = False
        self.training_metrics = TrainingMetrics()
    
    @abstractmethod
    def select_action(self, state: Tuple) -> int:
        """Select an action given the current state."""
        pass
    
    def get_state_key(self, state: HealthState) -> Tuple:
        """Convert HealthState to the appropriate state representation."""
        if self.fatigue_aware:
            return state.full_tuple()
        return state.observable_tuple()
    
    def get_policy_table(self) -> Dict[Tuple, int]:
        """Return the complete policy mapping state -> action."""
        return {}
    
    def get_value_table(self) -> Dict[Tuple, float]:
        """Return state values if available."""
        return {}
    
    def get_q_table(self) -> Dict[Tuple, Dict[int, float]]:
        """Return Q-values if available."""
        return {}
    
    def get_config(self) -> Dict[str, Any]:
        """Return the policy configuration for serialization."""
        return {
            'name': self.name,
            'fatigue_aware': self.fatigue_aware,
            'trained': self.trained,
        }


class RuleBasedPolicy(BasePolicy):
    """Rule-based baseline policy from the paper.
    
    Strategy:
    - If any health dimension is at its lowest (0), nudge that dimension.
    - Prioritize: sleep > activity > stress.
    - If all dimensions are adequate (>=1), do not nudge.
    - Does not consider habituation (memoryless by design).
    """
    
    def __init__(self):
        super().__init__("Rule-Based", fatigue_aware=False)
        self.trained = True  # Rule-based needs no training
    
    def select_action(self, state: Tuple) -> int:
        """Select action based on simple rules.
        
        state can be (s, a, σ) or (s, a, σ, h) — habituation is ignored.
        """
        sleep, activity, stress = state[0], state[1], state[2]
        
        # Priority: fix the most urgent health dimension
        if sleep == 0:
            return Action.SLEEP_NUDGE
        if activity == 0:
            return Action.ACTIVITY_NUDGE
        if stress == 0:
            return Action.STRESS_NUDGE
        
        # If any dimension is at moderate (1), nudge it (still prioritized)
        if sleep == 1:
            return Action.SLEEP_NUDGE
        if activity == 1:
            return Action.ACTIVITY_NUDGE
        if stress == 1:
            return Action.STRESS_NUDGE
        
        # All dimensions at max — no nudge needed
        return Action.NO_NUDGE
    
    def get_policy_table(self) -> Dict[Tuple, int]:
        env = NudgingEnvironment()
        states = env.get_all_states(include_habituation=False)
        return {s: self.select_action(s) for s in states}


class DynamicProgrammingPolicy(BasePolicy):
    """Dynamic Programming policy using Value Iteration.
    
    Can operate in two modes:
    - Memoryless: uses (s, a, σ) state space, ignores habituation
    - Fatigue-aware: uses (s, a, σ, h) state space
    
    The DP solver uses the analytical transition model.
    """
    
    def __init__(self, fatigue_aware: bool = False, 
                 convergence_tol: float = 1e-6,
                 max_iterations: int = 1000):
        name = "DP (Fatigue-Aware)" if fatigue_aware else "DP (Memoryless)"
        super().__init__(name, fatigue_aware)
        self.convergence_tol = convergence_tol
        self.max_iterations = max_iterations
        self.V: Dict[Tuple, float] = {}
        self.Q: Dict[Tuple, Dict[int, float]] = {}
        self.policy: Dict[Tuple, int] = {}
    
    def train(self, config: SimulatorConfig, seed: int = 42) -> TrainingMetrics:
        """Run value iteration to find the optimal policy.
        
        Args:
            config: Simulator configuration.
            seed: Random seed (deterministic for DP, but needed for consistent API).
        """
        env = NudgingEnvironment(config, seed)
        gamma = config.discount_factor
        
        # Get transition model
        include_hab = self.fatigue_aware
        model = env.get_transition_model(include_habituation=include_hab)
        states = env.get_all_states(include_habituation=include_hab)
        n_actions = env.get_num_actions()
        
        # Initialize values
        self.V = {}
        self.Q = {}
        self.policy = {}
        for s in states:
            self.V[s] = 0.0
            self.Q[s] = {a: 0.0 for a in range(n_actions)}
        
        # Value iteration
        self.training_metrics = TrainingMetrics()
        for iteration in range(self.max_iterations):
            delta = 0.0
            
            for s in states:
                v_old = self.V[s]
                
                # Compute Q-values for each action
                for a in range(n_actions):
                    q_val = 0.0
                    transitions = model.get((s, a), [])
                    for prob, next_state, reward in transitions:
                        q_val += prob * (reward + gamma * self.V.get(next_state, 0.0))
                    self.Q[s][a] = q_val
                
                # Greedy update
                self.V[s] = max(self.Q[s].values())
                delta = max(delta, abs(self.V[s] - v_old))
            
            self.training_metrics.value_deltas.append(delta)
            self.training_metrics.iterations = iteration + 1
            
            if delta < self.convergence_tol:
                self.training_metrics.converged = True
                self.training_metrics.convergence_iteration = iteration + 1
                break
        
        # Extract greedy policy
        for s in states:
            best_action = max(self.Q[s], key=self.Q[s].get)
            self.policy[s] = best_action
        
        self.trained = True
        return self.training_metrics
    
    def select_action(self, state: Tuple) -> int:
        if not self.trained:
            raise RuntimeError("Policy not trained. Call train() first.")
        if state in self.policy:
            return self.policy[state]
        # Fallback for unseen states
        return Action.NO_NUDGE
    
    def get_policy_table(self) -> Dict[Tuple, int]:
        return dict(self.policy)
    
    def get_value_table(self) -> Dict[Tuple, float]:
        return dict(self.V)
    
    def get_q_table(self) -> Dict[Tuple, Dict[int, float]]:
        return {s: dict(q) for s, q in self.Q.items()}
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'convergence_tol': self.convergence_tol,
            'max_iterations': self.max_iterations,
            'converged': self.training_metrics.converged,
            'iterations_run': self.training_metrics.iterations,
        })
        return config


class MonteCarloPolicy(BasePolicy):
    """Monte Carlo Control with epsilon-soft exploration.
    
    First-visit MC method for estimating optimal policies.
    Uses episode-based learning with configurable exploration.
    """
    
    def __init__(self, fatigue_aware: bool = False,
                 epsilon: float = 0.1,
                 n_episodes: int = 50000,
                 epsilon_decay: float = 0.9999,
                 min_epsilon: float = 0.01):
        name = "MC (Fatigue-Aware)" if fatigue_aware else "MC (Memoryless)"
        super().__init__(name, fatigue_aware)
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q: Dict[Tuple, Dict[int, float]] = {}
        self.returns_count: Dict[Tuple, Dict[int, int]] = {}
        self.returns_sum: Dict[Tuple, Dict[int, float]] = {}
        self.policy: Dict[Tuple, int] = {}
    
    def train(self, config: SimulatorConfig, seed: int = 42) -> TrainingMetrics:
        """Train MC control agent via episode sampling."""
        env = NudgingEnvironment(config, seed)
        gamma = config.discount_factor
        n_actions = env.get_num_actions()
        rng = np.random.RandomState(seed + 1000)
        
        # Initialize
        self.Q = {}
        self.returns_count = {}
        self.returns_sum = {}
        self.policy = {}
        states = env.get_all_states(include_habituation=self.fatigue_aware)
        for s in states:
            self.Q[s] = {a: 0.0 for a in range(n_actions)}
            self.returns_count[s] = {a: 0 for a in range(n_actions)}
            self.returns_sum[s] = {a: 0.0 for a in range(n_actions)}
        
        self.training_metrics = TrainingMetrics()
        current_epsilon = self.epsilon
        
        for episode in range(self.n_episodes):
            # Generate episode using epsilon-soft policy
            state = env.reset()
            episode_data = []  # list of (state_key, action, reward)
            
            for t in range(config.episode_length):
                state_key = self._get_state_key_from_health(state)
                
                # Epsilon-greedy action selection
                if rng.random() < current_epsilon:
                    action = rng.randint(0, n_actions)
                else:
                    if state_key in self.Q:
                        action = max(self.Q[state_key], key=self.Q[state_key].get)
                    else:
                        action = rng.randint(0, n_actions)
                
                next_state, reward, done, info = env.step(action)
                episode_data.append((state_key, action, reward))
                state = next_state
                
                if done:
                    break
            
            # Compute returns once, then update only each pair's first visit.
            G = 0.0
            returns = [0.0] * len(episode_data)
            for t in reversed(range(len(episode_data))):
                _, _, reward = episode_data[t]
                G = gamma * G + reward
                returns[t] = G

            visited = set()
            for t, (state_key, action, _) in enumerate(episode_data):
                sa_pair = (state_key, action)
                if sa_pair not in visited:
                    visited.add(sa_pair)
                    
                    if state_key not in self.returns_count:
                        self.Q[state_key] = {a: 0.0 for a in range(n_actions)}
                        self.returns_count[state_key] = {a: 0 for a in range(n_actions)}
                        self.returns_sum[state_key] = {a: 0.0 for a in range(n_actions)}
                    
                    self.returns_count[state_key][action] += 1
                    self.returns_sum[state_key][action] += returns[t]
                    self.Q[state_key][action] = (
                        self.returns_sum[state_key][action] / 
                        self.returns_count[state_key][action]
                    )
            
            # Track episode return
            ep_return = sum(
                reward * (gamma ** t) 
                for t, (_, _, reward) in enumerate(episode_data)
            )
            self.training_metrics.episode_returns.append(ep_return)
            self.training_metrics.episode_lengths.append(len(episode_data))
            self.training_metrics.exploration_rates.append(current_epsilon)
            
            # Decay epsilon
            current_epsilon = max(self.min_epsilon, current_epsilon * self.epsilon_decay)
        
        # Extract greedy policy
        for s in self.Q:
            self.policy[s] = max(self.Q[s], key=self.Q[s].get)
        
        self.training_metrics.iterations = self.n_episodes
        self.trained = True
        return self.training_metrics
    
    def _get_state_key_from_health(self, state: HealthState) -> Tuple:
        if self.fatigue_aware:
            return state.full_tuple()
        return state.observable_tuple()
    
    def select_action(self, state: Tuple) -> int:
        if not self.trained:
            raise RuntimeError("Policy not trained. Call train() first.")
        if state in self.policy:
            return self.policy[state]
        return Action.NO_NUDGE
    
    def get_policy_table(self) -> Dict[Tuple, int]:
        return dict(self.policy)
    
    def get_q_table(self) -> Dict[Tuple, Dict[int, float]]:
        return {s: dict(q) for s, q in self.Q.items()}
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
            'n_episodes': self.n_episodes,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
        })
        return config


class QLearningPolicy(BasePolicy):
    """Tabular Q-Learning with epsilon-greedy exploration.
    
    Off-policy TD control for learning the optimal policy.
    """
    
    def __init__(self, fatigue_aware: bool = False,
                 learning_rate: float = 0.1,
                 epsilon: float = 0.1,
                 n_episodes: int = 50000,
                 epsilon_decay: float = 0.9999,
                 min_epsilon: float = 0.01,
                 lr_decay: float = 1.0):
        name = "Q-Learning (Fatigue-Aware)" if fatigue_aware else "Q-Learning (Memoryless)"
        super().__init__(name, fatigue_aware)
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.n_episodes = n_episodes
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.lr_decay = lr_decay
        self.Q: Dict[Tuple, Dict[int, float]] = {}
        self.policy: Dict[Tuple, int] = {}
    
    def _ensure_state(self, state_key: Tuple, n_actions: int):
        """Initialize Q-values for a state if not yet seen."""
        if state_key not in self.Q:
            self.Q[state_key] = {a: 0.0 for a in range(n_actions)}
    
    def train(self, config: SimulatorConfig, seed: int = 42) -> TrainingMetrics:
        """Train Q-learning agent."""
        env = NudgingEnvironment(config, seed)
        gamma = config.discount_factor
        n_actions = env.get_num_actions()
        rng = np.random.RandomState(seed + 2000)
        
        self.Q = {}
        self.policy = {}
        self.training_metrics = TrainingMetrics()
        current_epsilon = self.epsilon
        current_lr = self.learning_rate
        
        for episode in range(self.n_episodes):
            state = env.reset()
            state_key = self._get_state_key_from_health(state)
            self._ensure_state(state_key, n_actions)
            
            episode_return = 0.0
            
            for t in range(config.episode_length):
                # Epsilon-greedy
                if rng.random() < current_epsilon:
                    action = rng.randint(0, n_actions)
                else:
                    action = max(self.Q[state_key], key=self.Q[state_key].get)
                
                next_state, reward, done, info = env.step(action)
                next_key = self._get_state_key_from_health(next_state)
                self._ensure_state(next_key, n_actions)
                
                # Q-learning update: Q(s,a) += α[r + γ max_a' Q(s',a') - Q(s,a)]
                best_next = max(self.Q[next_key].values())
                td_target = reward + gamma * best_next * (0.0 if done else 1.0)
                td_error = td_target - self.Q[state_key][action]
                self.Q[state_key][action] += current_lr * td_error
                
                episode_return += reward * (gamma ** t)
                state = next_state
                state_key = next_key
                
                if done:
                    break
            
            self.training_metrics.episode_returns.append(episode_return)
            self.training_metrics.episode_lengths.append(t + 1)
            self.training_metrics.exploration_rates.append(current_epsilon)
            
            current_epsilon = max(self.min_epsilon, current_epsilon * self.epsilon_decay)
            current_lr = current_lr * self.lr_decay
        
        # Extract greedy policy
        for s in self.Q:
            self.policy[s] = max(self.Q[s], key=self.Q[s].get)
        
        self.training_metrics.iterations = self.n_episodes
        self.trained = True
        return self.training_metrics
    
    def _get_state_key_from_health(self, state: HealthState) -> Tuple:
        if self.fatigue_aware:
            return state.full_tuple()
        return state.observable_tuple()
    
    def select_action(self, state: Tuple) -> int:
        if not self.trained:
            raise RuntimeError("Policy not trained. Call train() first.")
        if state in self.policy:
            return self.policy[state]
        return Action.NO_NUDGE
    
    def get_policy_table(self) -> Dict[Tuple, int]:
        return dict(self.policy)
    
    def get_q_table(self) -> Dict[Tuple, Dict[int, float]]:
        return {s: dict(q) for s, q in self.Q.items()}
    
    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            'learning_rate': self.learning_rate,
            'epsilon': self.epsilon,
            'n_episodes': self.n_episodes,
            'epsilon_decay': self.epsilon_decay,
            'min_epsilon': self.min_epsilon,
            'lr_decay': self.lr_decay,
        })
        return config


class OracleDPPolicy(DynamicProgrammingPolicy):
    """Oracle DP policy that always uses the full fatigue-aware state.
    
    This serves as an evaluation reference. The oracle has access to the 
    true simulator state including habituation, which a real learner may
    not observe. It represents the best a model-based planner can do with
    complete information.
    """
    
    def __init__(self, convergence_tol: float = 1e-6, max_iterations: int = 1000):
        # Oracle always uses fatigue-aware state
        super().__init__(fatigue_aware=True, convergence_tol=convergence_tol,
                        max_iterations=max_iterations)
        self.name = "Oracle DP"


def create_policy(algorithm: str, fatigue_aware: bool = False, **kwargs) -> BasePolicy:
    """Factory function to create policy instances.
    
    Args:
        algorithm: One of 'rule_based', 'dp', 'mc', 'q_learning', 'oracle_dp'
        fatigue_aware: Whether to use fatigue-aware state representation.
        **kwargs: Algorithm-specific parameters.
    
    Returns:
        Configured policy instance.
    """
    algorithm = algorithm.lower().replace('-', '_').replace(' ', '_')
    
    if algorithm == 'rule_based':
        return RuleBasedPolicy()
    elif algorithm == 'dp':
        return DynamicProgrammingPolicy(
            fatigue_aware=fatigue_aware,
            convergence_tol=kwargs.get('convergence_tol', 1e-6),
            max_iterations=kwargs.get('max_iterations', 1000),
        )
    elif algorithm == 'mc' or algorithm == 'monte_carlo':
        return MonteCarloPolicy(
            fatigue_aware=fatigue_aware,
            epsilon=kwargs.get('epsilon', 0.1),
            n_episodes=kwargs.get('n_episodes', 50000),
            epsilon_decay=kwargs.get('epsilon_decay', 0.9999),
            min_epsilon=kwargs.get('min_epsilon', 0.01),
        )
    elif algorithm == 'q_learning':
        return QLearningPolicy(
            fatigue_aware=fatigue_aware,
            learning_rate=kwargs.get('learning_rate', 0.1),
            epsilon=kwargs.get('epsilon', 0.1),
            n_episodes=kwargs.get('n_episodes', 50000),
            epsilon_decay=kwargs.get('epsilon_decay', 0.9999),
            min_epsilon=kwargs.get('min_epsilon', 0.01),
            lr_decay=kwargs.get('lr_decay', 1.0),
        )
    elif algorithm == 'oracle_dp':
        return OracleDPPolicy(
            convergence_tol=kwargs.get('convergence_tol', 1e-6),
            max_iterations=kwargs.get('max_iterations', 1000),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Choose from: rule_based, dp, mc, q_learning, oracle_dp")
