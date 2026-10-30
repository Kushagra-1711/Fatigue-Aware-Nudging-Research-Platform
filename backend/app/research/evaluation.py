"""
Evaluation module for the Fatigue-Aware Nudging platform.

Evaluates trained policies in the simulator and computes research metrics.
Supports multiple replications with different seeds for statistical validity.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict

from .environment import NudgingEnvironment, SimulatorConfig, HealthState, Action
from .policies import BasePolicy


@dataclass
class EpisodeResult:
    """Result of a single evaluation episode."""
    seed: int
    discounted_return: float
    undiscounted_return: float
    total_reward: float
    mean_health_score: float
    total_nudges: int
    nudge_days_fraction: float
    response_count: int
    response_rate: float
    mean_habituation: float
    final_habituation: int
    habituation_trajectory: List[int]
    health_trajectory: List[Dict[str, int]]
    action_sequence: List[int]
    reward_sequence: List[float]
    response_sequence: List[bool]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReplicationResult:
    """Aggregated result of multiple episodes in one replication."""
    replication_id: int
    seed: int
    n_episodes: int
    mean_discounted_return: float
    std_discounted_return: float
    mean_health_score: float
    std_health_score: float
    mean_nudges: float
    mean_nudge_fraction: float
    mean_response_rate: float
    mean_habituation: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Complete evaluation result across all replications."""
    algorithm: str
    fatigue_aware: bool
    n_replications: int
    n_episodes_per_replication: int
    
    # Aggregate metrics
    mean_discounted_return: float
    ci_95_lower: float
    ci_95_upper: float
    std_discounted_return: float
    
    mean_health_score: float
    mean_nudges_per_episode: float
    mean_nudge_fraction: float
    mean_response_rate: float
    mean_habituation: float
    
    # Per-replication details
    replication_results: List[ReplicationResult] = field(default_factory=list)
    
    # Sample episode for visualization
    sample_episode: Optional[EpisodeResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def evaluate_policy(
    policy: BasePolicy,
    config: SimulatorConfig,
    n_episodes: int = 1000,
    seed: int = 42,
    collect_sample: bool = True,
) -> Tuple[List[EpisodeResult], Optional[EpisodeResult]]:
    """Evaluate a policy over multiple episodes.
    
    Args:
        policy: Trained policy to evaluate.
        config: Simulator configuration.
        n_episodes: Number of evaluation episodes.
        seed: Base random seed for evaluation.
        collect_sample: Whether to collect a detailed sample episode.
    
    Returns:
        List of episode results and optionally a detailed sample episode.
    """
    results = []
    sample = None
    gamma = config.discount_factor
    
    for ep in range(n_episodes):
        ep_seed = seed + ep
        env = NudgingEnvironment(config, ep_seed)
        state = env.reset()
        
        rewards = []
        actions = []
        responses = []
        habituation_traj = [state.habituation]
        health_traj = [state.to_dict()]
        
        total_nudges = 0
        response_count = 0
        total_hab = state.habituation
        
        for t in range(config.episode_length):
            state_key = policy.get_state_key(state)
            action = policy.select_action(state_key)
            
            next_state, reward, done, info = env.step(action)
            
            rewards.append(reward)
            actions.append(action)
            responses.append(info.responded)
            habituation_traj.append(next_state.habituation)
            health_traj.append(next_state.to_dict())
            
            if action != Action.NO_NUDGE:
                total_nudges += 1
                if info.responded:
                    response_count += 1
            
            total_hab += next_state.habituation
            state = next_state
            
            if done:
                break
        
        # Compute metrics
        T = len(rewards)
        discounted_return = sum(r * (gamma ** t) for t, r in enumerate(rewards))
        undiscounted_return = sum(rewards)
        mean_health = np.mean([
            (h['sleep'] + h['activity'] + h['stress']) / 6.0 
            for h in health_traj
        ])
        nudge_fraction = total_nudges / max(T, 1)
        response_rate = response_count / max(total_nudges, 1)
        mean_hab = total_hab / (T + 1)
        
        ep_result = EpisodeResult(
            seed=ep_seed,
            discounted_return=discounted_return,
            undiscounted_return=undiscounted_return,
            total_reward=sum(rewards),
            mean_health_score=float(mean_health),
            total_nudges=total_nudges,
            nudge_days_fraction=nudge_fraction,
            response_count=response_count,
            response_rate=response_rate,
            mean_habituation=float(mean_hab),
            final_habituation=state.habituation,
            habituation_trajectory=habituation_traj,
            health_trajectory=health_traj,
            action_sequence=actions,
            reward_sequence=rewards,
            response_sequence=responses,
        )
        results.append(ep_result)
        
        if collect_sample and ep == 0:
            sample = ep_result
    
    return results, sample


def run_evaluation(
    policy: BasePolicy,
    config: SimulatorConfig,
    n_replications: int = 6,
    n_episodes_per_replication: int = 1000,
    base_seed: int = 42,
) -> EvaluationResult:
    """Run a full evaluation with multiple independent replications.
    
    Following the paper's protocol:
    - 6 independent replications with different seeds
    - 1000 episodes per replication
    - Report mean discounted return with 95% CI
    
    Args:
        policy: Trained policy to evaluate.
        config: Simulator configuration.
        n_replications: Number of independent replications.
        n_episodes_per_replication: Episodes per replication.
        base_seed: Base seed (each replication uses base_seed + rep_id * 10000).
    
    Returns:
        EvaluationResult with aggregate statistics.
    """
    replication_results = []
    all_returns = []
    all_health = []
    all_nudges = []
    all_nudge_frac = []
    all_response_rate = []
    all_hab = []
    sample_episode = None
    
    for rep in range(n_replications):
        rep_seed = base_seed + rep * 10000
        
        episodes, sample = evaluate_policy(
            policy, config, 
            n_episodes=n_episodes_per_replication,
            seed=rep_seed,
            collect_sample=(rep == 0),
        )
        
        if sample is not None and sample_episode is None:
            sample_episode = sample
        
        returns = [e.discounted_return for e in episodes]
        health_scores = [e.mean_health_score for e in episodes]
        nudges = [e.total_nudges for e in episodes]
        nudge_fracs = [e.nudge_days_fraction for e in episodes]
        resp_rates = [e.response_rate for e in episodes]
        habs = [e.mean_habituation for e in episodes]
        
        rep_result = ReplicationResult(
            replication_id=rep,
            seed=rep_seed,
            n_episodes=len(episodes),
            mean_discounted_return=float(np.mean(returns)),
            std_discounted_return=float(np.std(returns)),
            mean_health_score=float(np.mean(health_scores)),
            std_health_score=float(np.std(health_scores)),
            mean_nudges=float(np.mean(nudges)),
            mean_nudge_fraction=float(np.mean(nudge_fracs)),
            mean_response_rate=float(np.mean(resp_rates)),
            mean_habituation=float(np.mean(habs)),
        )
        replication_results.append(rep_result)
        all_returns.extend(returns)
        all_health.extend(health_scores)
        all_nudges.extend(nudges)
        all_nudge_frac.extend(nudge_fracs)
        all_response_rate.extend(resp_rates)
        all_hab.extend(habs)
    
    # Compute aggregate statistics
    rep_means = [r.mean_discounted_return for r in replication_results]
    overall_mean = float(np.mean(rep_means))
    overall_std = float(np.std(rep_means, ddof=1)) if len(rep_means) > 1 else 0.0
    
    # 95% CI using t-distribution approximation
    n = len(rep_means)
    if n > 1:
        # t-value for 95% CI with n-1 degrees of freedom
        from scipy import stats
        t_val = stats.t.ppf(0.975, n - 1)
        se = overall_std / np.sqrt(n)
        ci_lower = overall_mean - t_val * se
        ci_upper = overall_mean + t_val * se
    else:
        ci_lower = overall_mean
        ci_upper = overall_mean
    
    return EvaluationResult(
        algorithm=policy.name,
        fatigue_aware=policy.fatigue_aware,
        n_replications=n_replications,
        n_episodes_per_replication=n_episodes_per_replication,
        mean_discounted_return=overall_mean,
        ci_95_lower=float(ci_lower),
        ci_95_upper=float(ci_upper),
        std_discounted_return=overall_std,
        mean_health_score=float(np.mean(all_health)),
        mean_nudges_per_episode=float(np.mean(all_nudges)),
        mean_nudge_fraction=float(np.mean(all_nudge_frac)),
        mean_response_rate=float(np.mean(all_response_rate)),
        mean_habituation=float(np.mean(all_hab)),
        replication_results=replication_results,
        sample_episode=sample_episode,
    )


def compute_policy_agreement(
    policy_a: BasePolicy,
    policy_b: BasePolicy,
    config: SimulatorConfig,
    include_habituation: bool = True,
) -> Dict[str, Any]:
    """Compute the fraction of states where two policies agree.
    
    Args:
        policy_a: First policy.
        policy_b: Second policy.
        config: Simulator configuration (for state enumeration).
        include_habituation: Whether to include habituation in state enumeration.
    
    Returns:
        Dictionary with agreement statistics.
    """
    env = NudgingEnvironment(config)
    states = env.get_all_states(include_habituation=include_habituation)
    
    agree_count = 0
    disagree_states = []
    
    for s in states:
        action_a = policy_a.select_action(s)
        action_b = policy_b.select_action(s)
        
        if action_a == action_b:
            agree_count += 1
        else:
            disagree_states.append({
                'state': s,
                'action_a': action_a,
                'action_b': action_b,
                'action_a_name': Action(action_a).name,
                'action_b_name': Action(action_b).name,
            })
    
    total = len(states)
    return {
        'total_states': total,
        'agreement_count': agree_count,
        'agreement_rate': agree_count / max(total, 1),
        'n_disagreements': len(disagree_states),
        'sample_disagreements': disagree_states[:20],
    }
