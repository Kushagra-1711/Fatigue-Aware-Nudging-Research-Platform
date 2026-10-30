# Application Walkthrough

This guide walks through the Fatigue-Aware Nudging Research Platform, from starting the application to reviewing and exporting experiment results.

> **Research-use notice:** The platform is a synthetic simulator. It contains no patient records, is not clinically validated, and must not be used for treatment or health-care decisions. Its assumptions have not been independently verified against the referenced research paper.

## 1. Start the application

For local development, start the backend and frontend in separate PowerShell terminals from the project root.

Backend:

```powershell
cd .\backend
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Frontend:

```powershell
cd .\frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The header reports whether the frontend can reach the API. If setting up a fresh environment, see the installation instructions in [README.md](./README.md). The API's interactive documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## 2. Explore the overview

The **Overview** page summarizes saved experiments and highlights the latest and best observed runs. Its metrics are loaded from the local API and database. Use **Configure experiment** to begin a repeatable research run.

## 3. Run a single episode

Open **Simulator** to inspect one episode's daily trajectory:

1. Choose an episode duration (1–365 days) and a non-negative random seed.
2. Select a policy: Rule-based, Dynamic programming (DP), Monte Carlo control (MC), Q-learning, or Oracle DP.
3. Choose whether the learner uses a fatigue-aware state where supported.
4. Select **Run episode**.

The results show the discounted return, health and habituation chart, and a daily trace of actions, response probabilities, responses, rewards, and habituation changes. Reusing the same seed and settings makes this stochastic episode reproducible within the implementation.

## 4. Configure an experiment

Open **Experiments** and create a named run:

- Select an algorithm and choose whether habituation is included in its state representation.
- Set evaluation episodes per replication, number of independent replications, and a base seed.
- Set training episodes for MC or Q-learning. This setting does not affect the rule-based policy.
- Select **Create experiment** to submit it to the backend.

The UI shows the new run's ID and directs you to history. Experiments run in local backend worker threads; their status is persisted and can be refreshed from the history page. Large workloads may take time.

## 5. Review history, compare, and export

In **History & compare**, refresh the experiment list to see current statuses. Open an experiment name to view its details. Select at least two runs and choose **Compare selected** to chart their mean discounted returns.

Use the **CSV** or **JSON** actions on a run to download its data. CSV contains replication-level aggregates; JSON includes the saved result structure, including aggregate metrics and sample episode data.

Comparisons are summaries of the selected runs, not a guarantee of a controlled comparison. Before interpreting differences, check that the model configuration, algorithm settings, evaluation setup, and seeds are appropriate and comparable.

## 6. Inspect results and policies

The experiment detail page includes run status, mean discounted return and its 95% confidence interval, health score, response rate, mean habituation, and available sample episode or training charts.

Open **Policy inspection**, select a completed run, and choose **Load policy** to browse the saved state-to-action mappings and response probabilities. Habituation appears as **Unobserved** for policies whose state representation does not include it.

## 7. Understand the synthetic model

The environment represents sleep, activity, and stress on three ordinal levels, plus habituation on four levels. A memoryless learner sees 27 health states; a fatigue-aware learner can observe 108 health-and-habituation states. Available actions are no nudge, sleep nudge, activity nudge, and stress-management nudge.

For a delivered nudge, response probability depends on the nudge type and is attenuated as habituation increases. A response may improve the targeted health dimension; health dimensions can also experience natural downward drift. Habituation may rise after a nudge or decay on a day without one. The default episode is 28 days, and the post-transition reward is the normalized health score.

The **Methodology** page summarizes implemented model defaults and limitations. For full model and policy details, see [ARCHITECTURE.md](./ARCHITECTURE.md) and [README.md](./README.md).

## 8. Interpret outputs with care

Results describe the behavior of this implementation under its configured synthetic environment; they are not clinical evidence or claims about real-world outcomes. A fixed seed supports repeatability for the same configuration, but changing the seed, policy, training settings, or environment configuration can change results. The referenced research paper was not included in the workspace, so the model defaults and equations have not been independently checked against it.
