# Machine Learning Techniques and Algorithms

This document describes the learning methods implemented in the Fatigue-Aware Nudging Research Platform. It is based on the code in [`backend/app/research/environment.py`](./backend/app/research/environment.py), [`backend/app/research/policies.py`](./backend/app/research/policies.py), and [`backend/app/research/evaluation.py`](./backend/app/research/evaluation.py).

> **Scope and interpretation:** These are tabular reinforcement-learning methods applied to a synthetic simulator. They are not trained on patient records, do not make clinical predictions, and are not validated treatment policies. The research paper referenced by the project was not supplied in the workspace; implementation details and defaults described here should not be treated as independently verified against that paper.

## 1. Problem formulation: a finite, episodic MDP

The simulator casts daily nudge selection as a Markov decision process (MDP). At each day, a policy receives a state representation, selects an action, and receives a reward after the synthetic environment transitions to its next state.

### Full environment state

The simulator's internal state is:

`x = (sleep, activity, stress, habituation)`

With the defaults:

- Sleep, activity, and stress each have three ordinal levels: `0`, `1`, and `2`.
- Stress is oriented so that a larger value means lower stress and better health.
- Habituation has four levels: `0` through `3`.
- The full state space contains `3 × 3 × 3 × 4 = 108` states.

The **memoryless** learner observes only `(sleep, activity, stress)`, giving 27 states. The **fatigue-aware** learner observes the full four-component state, giving 108 states. Both use the same simulator; the distinction is what the policy uses as its state key. The simulator still evolves habituation in memoryless runs, even though a memoryless policy cannot observe it.

### Actions

There are four discrete actions:

| ID | Action | Effect |
|---:|---|---|
| 0 | No nudge | Delivers no intervention. Habituation may decay. |
| 1 | Sleep nudge | May improve sleep by one level after a response. |
| 2 | Activity nudge | May improve activity by one level after a response. |
| 3 | Stress-management nudge | May improve the stress score by one level after a response. |

Health improvements are capped at the maximum health level, and natural drift is capped at zero.

### Stochastic transition and reward

For a delivered nudge, the response probability is:

`P(response | action, h) = p_base(action) × max(min_response_probability, 1 − attenuation_rate × h)`

The default base response probabilities are 0.60 for sleep, 0.50 for activity, and 0.55 for stress. At habituation level `h`, the default attenuation factor is `max(0.1, 1 − 0.25h)`. No-nudge has zero response probability.

Each day proceeds in this order:

1. Determine whether a delivered nudge receives a response.
2. If the person responds, improve the targeted health dimension by one.
3. Independently sample natural downward drift for sleep, activity, and stress.
4. If a nudge was delivered, habituation may increase; otherwise, it may decay.
5. Compute reward from the resulting health state.

The reward is the normalized post-transition health score:

`r = (sleep + activity + stress) / 6`

With three health levels, the reward ranges from 0 to 1. The default discount factor is `γ = 0.99` and the default simulation episode lasts 28 days. These values and probabilities are configurable.

## 2. State representations and the habituation question

The project compares policies with different information:

- A memoryless policy maps health states to actions and cannot condition directly on current habituation.
- A fatigue-aware policy can choose different actions at the same health levels depending on habituation.

This makes fatigue-aware Q-tables larger: four action values per health-only state versus four per health-and-habituation state. The larger representation can capture the effect of habituation on response probability, but needs to estimate more state-action pairs when learned from samples.

There is an important implementation detail for interpreting the DP comparison: when the DP solver is configured as memoryless, its analytical transition model uses habituation level zero when computing response probability and omits habituation from successor states. It does **not** explicitly maintain a probability distribution over hidden habituation levels. Meanwhile, sampled memoryless MC and Q-learning episodes run in the full simulator, where hidden habituation still changes and affects responses. Thus, the memoryless DP planning model and the sampled memoryless environment do not represent hidden habituation in exactly the same way.

## 3. Algorithms implemented

### 3.1 Rule-based baseline

The rule-based policy is a deterministic hand-coded baseline; it does not learn from data or simulator episodes. It checks dimensions in a fixed priority order: sleep, then activity, then stress.

1. Nudge the first dimension at level 0.
2. If none is at 0, nudge the first dimension at level 1.
3. If all dimensions are at level 2, choose no nudge.

It always ignores habituation. This provides a simple reference policy, not a learned or optimized policy.

### 3.2 Dynamic Programming (DP): model-based value iteration

DP plans using an analytical transition model rather than learning transition probabilities from sampled experience. The environment enumerates possible response, health-drift, and (for fatigue-aware planning) habituation outcomes to produce transition probabilities for each state-action pair.

Value iteration repeatedly computes the expected discounted action value:

`Q(s,a) = Σ_s' P(s' | s,a) [r(s,a,s') + γ V(s')]`

Then it updates the state value and ultimately extracts a greedy policy:

`V(s) = max_a Q(s,a)`  
`π(s) = argmax_a Q(s,a)`

The implementation updates state values in place during each sweep. It stops when the largest value change in a sweep is below `convergence_tol`, or when `max_iterations` is reached. The defaults are `1e-6` and 1,000 iterations. Training metrics include per-sweep value deltas and whether the tolerance was reached.

Because the value update always includes `γV(s')` and does not model a remaining-day counter or terminal transition, this solver is a stationary discounted-MDP planner. It is not explicitly finite-horizon dynamic programming for the 28-day episode. DP itself does not need random exploration; its seed is accepted for API consistency.

### 3.3 Monte Carlo (MC) control: first-visit action-value estimation

Monte Carlo control learns from complete sampled episodes. It maintains a table `Q(s,a)` and estimates action values from the discounted returns observed after taking an action in a state.

The training procedure is:

1. Initialize action values and return counts/sums to zero for the configured state space.
2. Generate an episode using epsilon-greedy action selection. With probability `ε`, choose a random action; otherwise choose an action with the largest current Q-value.
3. Work backward through the episode to compute the discounted return for each time step:

   `G_t = r_(t+1) + γG_(t+1)`

4. For each state-action pair, update its average return only at its **first occurrence in that episode**:

   `Q(s,a) = cumulative_sum_of_first_visit_returns / first_visit_count`

5. Reduce epsilon after each episode, bounded below by `min_epsilon`.
6. After training, make the policy greedy with respect to the learned Q-table.

Defaults in the policy class are 50,000 training episodes, `ε = 0.1`, epsilon decay `0.9999`, and minimum epsilon `0.01`. Experiment requests may override the training episode count and algorithm settings. Monte Carlo updates happen after complete episodes; it does not use a one-step temporal-difference bootstrap.

### 3.4 Q-learning: tabular off-policy temporal-difference control

Q-learning learns action values incrementally from each transition. It uses epsilon-greedy exploration during training and applies the off-policy update:

`Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`

For a terminal transition, the next-state bootstrap is set to zero:

`target = r` if the episode is done; otherwise `target = r + γ max_a' Q(s',a')`.

The implementation:

1. Starts with unseen state-action values set to zero.
2. Uses epsilon-greedy action selection.
3. Steps the stochastic simulator and observes reward and next state.
4. Updates the chosen state-action value using the temporal-difference target.
5. Decays epsilon and, optionally, the learning rate after each episode.
6. Extracts a greedy policy from the learned table at the end of training.

Class defaults are learning rate `α = 0.1`, `ε = 0.1`, 50,000 episodes, epsilon decay `0.9999`, minimum epsilon `0.01`, and learning-rate decay `1.0`. A learning-rate decay of 1.0 leaves the rate constant. The learning rate is decayed once per episode, rather than once per transition.

### 3.5 Oracle DP: full-state planning reference

Oracle DP uses the same value-iteration implementation as DP but always includes habituation in its state. It has access to the simulator's full state and analytical transition model. It is therefore a model-based, full-information reference for this synthetic environment—not an independently measured real-world upper bound, and not a separate learning algorithm.

### Summary

| Method | Learning/planning style | Uses sampled episodes to improve policy? | State values/action values |
|---|---|---:|---|
| Rule-based | Fixed deterministic rules | No | None |
| DP | Model-based value iteration | No | `V(s)` and `Q(s,a)` |
| Monte Carlo control | Model-free, first-visit returns | Yes | `Q(s,a)` |
| Q-learning | Model-free, temporal-difference control | Yes | `Q(s,a)` |
| Oracle DP | Full-state model-based value iteration | No | `V(s)` and `Q(s,a)` |

“Model-free” here means the learner's update does not require the analytical transition table; it still interacts with the project's simulator during training.

## 4. Training, exploration, and reproducibility

Training is run by the experiment service. It builds a simulator configuration and selected policy, trains DP/MC/Q-learning where applicable, then evaluates the resulting policy. The rule-based policy is already marked trained and skips training.

For stochastic learners, environment and action-selection randomness use separate NumPy random generators. Given the same code, configuration, and seeds, training is designed to be repeatable:

- Training environment uses the experiment base seed.
- MC action-selection randomness uses `base_seed + 1000`.
- Q-learning action-selection randomness uses `base_seed + 2000`.
- Evaluation begins at `base_seed + 100000`; replication `i` uses `base_seed + 100000 + i × 10000`, and its episode `j` uses that replication seed plus `j`.

The policy is trained once per experiment and evaluated across the replications; replications use different evaluation seeds rather than independently retraining the policy. Reproducibility is implementation- and configuration-specific and should not be interpreted as evidence that the underlying assumptions are scientifically validated.

## 5. Evaluation metrics

For every evaluation episode, the evaluator records:

- **Discounted return:** `Σ_t γ^t r_t`.
- **Undiscounted return / total reward:** `Σ_t r_t`.
- **Mean health score:** average normalized health, including the initial health state in the trajectory.
- **Total nudges:** number of days with an action other than no-nudge.
- **Nudge-day fraction:** nudged days divided by episode length.
- **Response rate:** successful responses divided by delivered nudges; it is zero when no nudges were delivered.
- **Mean habituation:** average habituation across the initial state and each subsequent state.
- **Sample traces:** action, reward, response, health, and habituation sequences for a representative episode.

Across replications, the reported mean discounted return is the mean of the replication-level mean returns. The 95% confidence interval is computed from those replication means using a Student's t critical value and standard error. With only one replication, the interval collapses to the observed mean. Other aggregate metrics are means over the collected episode-level measures.

The defaults in the evaluation module are six replications with 1,000 episodes each. API/UI experiment requests can select smaller values for shorter runs. The confidence interval quantifies variation across the configured replications; it does not account for model misspecification, alternative environments, hyperparameter selection, or real-world uncertainty.

## 6. Important limitations when interpreting algorithm results

- The environment and all transitions are synthetic and stochastic.
- No clinical data is used, and no policy is clinically validated.
- The referenced paper was not provided, so the equations/defaults are not independently paper-verified.
- Tabular state-action methods do not generalize beyond the discretized state/action tables.
- MC and Q-learning require sufficient exploration and training coverage; unvisited state-action pairs retain default values.
- Memoryless sampled policies do not observe habituation, even though it affects their environment's transition probabilities.
- Memoryless DP assumes habituation zero in its planning model instead of marginalizing over hidden habituation.
- DP's stationary discounted value iteration does not explicitly account for the finite episode horizon.
- Oracle DP is an idealized planner with complete simulator knowledge, not a real-world benchmark.
- A comparison is informative only when the simulator configuration, algorithm settings, training budget, evaluation protocol, and seeds are considered. The application summarizes selected runs but does not guarantee all of these conditions are matched.

For setup and user-facing flows, see [`application_walkthrough.md`](./application_walkthrough.md). For system and environment architecture, see [`ARCHITECTURE.md`](./ARCHITECTURE.md) and [`README.md`](./README.md).
