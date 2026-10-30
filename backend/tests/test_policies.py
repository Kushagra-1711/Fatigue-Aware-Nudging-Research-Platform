from backend.app.research import policies
from backend.app.research.environment import HealthState, SimulatorConfig


def test_monte_carlo_uses_first_visit_return(monkeypatch):
    class RepeatedStateEnvironment:
        def __init__(self, config, seed):
            self.config = config
            self.rewards = iter((1.0, 10.0))

        def get_num_actions(self):
            return 4

        def get_all_states(self, include_habituation=True):
            return [(0, 0, 0)]

        def reset(self):
            self.rewards = iter((1.0, 10.0))
            return HealthState(0, 0, 0, 0)

        def step(self, action):
            reward = next(self.rewards)
            return HealthState(0, 0, 0, 0), reward, reward == 10.0, None

    monkeypatch.setattr(policies, "NudgingEnvironment", RepeatedStateEnvironment)
    policy = policies.MonteCarloPolicy(
        fatigue_aware=False,
        epsilon=0.0,
        n_episodes=1,
    )

    metrics = policy.train(SimulatorConfig(discount_factor=0.5), seed=1)

    assert policy.Q[(0, 0, 0)][0] == 6.0
    assert metrics.episode_lengths == [2]
    assert metrics.episode_returns == [6.0]


def test_q_learning_training_is_reproducible():
    config = SimulatorConfig(episode_length=5)
    first = policies.QLearningPolicy(n_episodes=20)
    second = policies.QLearningPolicy(n_episodes=20)

    first.train(config, seed=31)
    second.train(config, seed=31)

    assert first.Q == second.Q
    assert first.policy == second.policy


def test_dynamic_programming_creates_complete_policy():
    config = SimulatorConfig()
    policy = policies.DynamicProgrammingPolicy(max_iterations=20)

    policy.train(config)

    assert len(policy.get_policy_table()) == 27
    assert all(action in range(4) for action in policy.get_policy_table().values())
