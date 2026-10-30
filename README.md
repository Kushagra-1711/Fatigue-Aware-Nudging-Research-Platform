# Fatigue-Aware Nudging Research Platform

A full-stack research application for simulating health-related nudges and studying how habituation affects policy decisions. The platform combines a configurable stochastic environment, five policy implementations, repeatable evaluation, a persistence-backed API, and an interactive dashboard.

> **Scope:** This is a synthetic research simulator. It contains no patient records, is not clinically validated, and must not be used to make treatment or health-care decisions.

## Project at a glance

- **Environment:** daily transitions over sleep, activity, stress, and habituation.
- **Actions:** no nudge, sleep nudge, activity nudge, or stress-management nudge.
- **Policies:** rule-based, Dynamic Programming (DP), Monte Carlo (MC), Q-learning, and Oracle DP.
- **Experiments:** configurable training and evaluation, multiple replications, recorded results, and policy tables.
- **Dashboard:** episode simulation, experiment creation and history, comparisons, result charts, and policy inspection.
- **Backend:** Python, FastAPI, Pydantic, SQLAlchemy, and SQLite.
- **Frontend:** React, TypeScript, Vite, and Recharts.

## Research question

The project explores how representing habituation in an agent's state changes its nudge policy. Memoryless policies observe health only; fatigue-aware policies can also observe the current habituation level. Results can be compared across the same synthetic environment and evaluation protocol.

## Environment and model

### State and actions

Each health dimension has three ordinal levels by default:

| Variable | Values | Meaning |
|---|---|---|
| Sleep | 0–2 | Poor to good sleep |
| Activity | 0–2 | Low to high activity |
| Stress | 0–2 | High to low stress (higher is better) |
| Habituation | 0–3 | None to severe |

The **memoryless** state representation is `(sleep, activity, stress)`, with 27 possible states under the defaults. The **fatigue-aware** representation is `(sleep, activity, stress, habituation)`, with 108 states.

| Action | ID |
|---|---:|
| No nudge | 0 |
| Sleep nudge | 1 |
| Activity nudge | 2 |
| Stress-management nudge | 3 |

### Daily transition sequence

For each day in an episode, the environment:

1. Receives a policy action.
2. Samples whether the person responds to a delivered nudge, using the nudge-specific response probability adjusted for current habituation.
3. Improves the targeted health dimension by one level if a response occurs, capped at the maximum level.
4. Samples independent natural downward drift for each health dimension.
5. Updates habituation: a delivered nudge may increase it; no nudge may allow it to decay.
6. Calculates reward from the resulting health state and advances the episode.

The response probability for a nudge is:

`p_base(action) × max(min_response_probability, 1 − habituation_attenuation_rate × habituation)`

The no-nudge action has zero response probability. Nudge delivery does not guarantee a response.

### Reward and default configuration

The default reward is the post-transition health score:

`reward = (sleep + activity + stress) / 6`

With three levels per dimension, the score ranges from 0 to 1. Higher stress values represent lower stress and therefore contribute positively to the score.

| Parameter | Default |
|---|---:|
| Episode length | 28 days |
| Discount factor (`γ`) | 0.99 |
| Health levels | 3 |
| Habituation levels | 4 |
| Base response probability: sleep | 0.60 |
| Base response probability: activity | 0.50 |
| Base response probability: stress | 0.55 |
| Habituation increase probability after a nudge | 0.30 |
| Habituation decay probability after no nudge | 0.20 |
| Response attenuation per habituation level | 0.25 |
| Minimum response attenuation factor | 0.10 |
| Natural drift probability per health dimension | 0.10 |
| Initial health state | Random unless configured |
| Initial habituation | 0 |

These values can be overridden for simulations and experiments. The validated API configuration supports health-level counts from 2–5, habituation-level counts from 2–8, and episode lengths from 1–365 days.

## Policies

| Policy | Description |
|---|---|
| **Rule-based** | Uses a fixed health-priority rule: nudges the first low or moderate dimension in sleep, activity, stress order; otherwise chooses no nudge. It ignores habituation and does not require training. |
| **DP** | Uses value iteration over an analytical transition model. It can plan over the memoryless or fatigue-aware state representation. |
| **Monte Carlo control** | Learns action values from sampled episodes using first-visit returns and an epsilon-soft exploration policy. |
| **Q-learning** | Learns tabular action values with epsilon-greedy exploration and temporal-difference updates. |
| **Oracle DP** | Runs DP with full access to habituation and serves as a full-state planning reference. |

The MC and Q-learning policies expose settings for episode count, exploration rate and decay; Q-learning also exposes learning-rate settings. DP exposes its convergence tolerance and maximum iteration count. The policy factory and API accept these settings through algorithm configuration.

## Evaluation and reproducibility

The evaluator runs a trained policy over episodes grouped into independent replications. Each episode uses a deterministic seed derived from its replication seed. The default research protocol is six replications with 1,000 evaluation episodes per replication; API and UI limits allow smaller runs for quick checks.

Reported measures include:

- Mean discounted return and its 95% confidence interval.
- Standard deviation of discounted return across replication means.
- Mean health score.
- Mean nudges per episode and fraction of days with a nudge.
- Response rate among delivered nudges.
- Mean habituation.
- Per-replication statistics and a sample episode trace.

Training and evaluation use separate seed offsets for reproducibility. Using the same simulator configuration, algorithm settings, and base seed makes a run reproducible within the implementation. Stochastic MC/Q-learning results can differ when those inputs change.

## Application workflow

1. **Overview:** check API connectivity and review persisted experiment summaries.
2. **Simulator:** run one episode with a selected policy and seed; inspect its daily transitions, action, response, reward, health, and habituation.
3. **Experiments:** submit a named run with an algorithm, state representation, training/evaluation settings, replication count, and base seed. The backend validates the request and queues execution in a local thread pool.
4. **History & compare:** refresh experiment status, inspect summaries, compare selected runs, or export results.
5. **Experiment details:** view aggregate metrics, a sample episode, and available training traces.
6. **Policy inspection:** browse a completed experiment's state-to-action table and state-specific response probabilities.
7. **Methodology:** review the model outline and scope notes.

Experiment configuration, lifecycle status, logs, evaluation results, training metrics, and policy mappings are stored in SQLite. The frontend history page refreshes statuses on request; it does not use live push updates.

## Repository layout

```text
.
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI routes
│   │   ├── core/                # Application settings
│   │   ├── research/            # Environment, policies, evaluation
│   │   ├── services/            # Experiment execution and persistence workflows
│   │   ├── main.py              # FastAPI application factory
│   │   ├── models.py            # SQLAlchemy models and database setup
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── examples/                # Example experiment outputs/configuration
│   ├── tests/                   # Backend tests
│   └── requirements.txt
├── frontend/
│   ├── src/                     # React application, API client, styles, tests
│   ├── package.json
│   └── vite.config.ts
├── ARCHITECTURE.md              # Model and architecture notes
├── docker-compose.yml
└── README.md
```

## Requirements

- Python 3.11 or newer
- Node.js 20.19 or newer (or 22.12 or newer) and npm
- Optional: Docker and Docker Compose

## Run locally

Start the backend and frontend in separate PowerShell terminals from the project root.

### Backend

```powershell
cd .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`; interactive API documentation is at `http://localhost:8000/docs`. Database tables are created when the application starts. By default, the SQLite database and exports directory are relative to the backend process working directory.

### Frontend

```powershell
cd .\frontend
npm ci
npm run dev
```

Open `http://localhost:5173`. Vite proxies `/api` requests to `http://127.0.0.1:8000` by default, so local development does not require browser CORS configuration.

To point the proxy to another backend, set `VITE_PROXY_TARGET` before starting Vite. To have the frontend call an API directly instead, set `VITE_API_URL` and add the frontend origin to the backend's `CORS_ORIGINS`.

### Run with Docker Compose

From the project root:

```powershell
docker compose up --build
```

The dashboard is served at `http://localhost:5173`, and the API at `http://localhost:8000`. Compose stores the SQLite database and exports in the persistent `research-data` volume. Stop the services with `Ctrl+C`; run `docker compose down` to remove the containers while retaining the named volume.

## Configuration

Backend settings are read from process environment variables. `.env.example` lists the supported settings; the application reads environment variables directly, so load them into the process environment before launching the backend.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Backend bind address |
| `PORT` | `8000` | Backend port |
| `DEBUG` | `true` | Debug logging |
| `DATABASE_URL` | `sqlite:///./fatigue_nudging.db` | SQLAlchemy database URL |
| `CORS_ORIGINS` | Localhost development origins | Comma-separated allowed frontend origins |
| `MAX_EXPERIMENT_STEPS` | `5000000` | Maximum estimated experiment workload |
| `MAX_CONCURRENT_JOBS` | `3` | Concurrent local experiment workers |
| `EXPORT_DIR` | `./exports` | Export directory |
| `VITE_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite development proxy destination |
| `VITE_API_URL` | Empty | Optional direct API base URL for the frontend |

The request schema sets the input bounds, including up to 200,000 training episodes, 5,000 evaluation episodes per replication, 20 replications, and 365 days per episode. These schema bounds are separate from the similarly named `MAX_*` example values in `.env.example`; the runtime workload and concurrency settings are `MAX_EXPERIMENT_STEPS` and `MAX_CONCURRENT_JOBS`.

## API

All application routes are prefixed with `/api`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | API health and version |
| `GET` | `/config/defaults` | Default simulator and algorithm settings |
| `GET` | `/algorithms` | Available policy identifiers |
| `GET` | `/dashboard` | Experiment counts and latest/best run summaries |
| `POST` | `/simulations` | Run one episode and return its step trace |
| `POST` | `/experiments` | Create and queue an experiment |
| `GET` | `/experiments` | List experiments (`limit` and `offset` supported) |
| `GET` | `/experiments/{id}` | Read experiment status and summary |
| `GET` | `/experiments/{id}/results` | Read evaluation results and training metrics |
| `GET` | `/experiments/{id}/logs` | Read experiment execution logs |
| `GET` | `/experiments/{id}/policy` | Read the persisted policy table |
| `POST` | `/experiments/{id}/rerun` | Queue a run using the saved configuration, optionally with a new seed |
| `GET` | `/experiments/{id}/export?format=json\|csv` | Export experiment results |
| `POST` | `/compare` | Compare summaries for two or more experiment IDs |

Request bodies and query parameters are validated by Pydantic schemas. The API enforces bounds on configuration and estimated experiment workload. See `/docs` for interactive request and response schemas.

## Data and example outputs

- The default local database is `backend/fatigue_nudging.db` when the server is started from `backend`.
- Local export files are written under `backend/exports` by default.
- Docker Compose uses a named volume for persistent database and export data.
- `backend/examples/` contains example output/configuration files, including a recorded rule-based run. They illustrate the platform's result format; new runs should be created for current configurations.

CSV export contains replication-level aggregates. JSON export includes the full saved result structure, including aggregate metrics and sample episode data.

## Tests and production build

From the project root, run the backend tests:

```powershell
python -m pytest backend/tests -q
```

Run frontend tests and build:

```powershell
cd .\frontend
npm ci
npm test
npm run build
```

The backend tests cover first-visit Monte Carlo return updates, deterministic Q-learning, DP policy construction, and a SQLite-backed API workflow. Frontend tests cover application navigation and experiment submission.

## Operational notes and limitations

- Experiment workers run in local threads. There is no distributed worker, process recovery, or cancellation endpoint.
- Experiment history is stored in SQLite by default. The application does not configure PostgreSQL migrations.
- Comparisons summarize the selected runs; users should make sure configurations and evaluation conditions are appropriate for the comparison.
- The simulator is synthetic and has no patient records or clinical validation. Outputs are for research and software experimentation only, not individual health recommendations.
