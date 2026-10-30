# Fatigue-Aware Nudging Research Platform — Architecture

## 1. Research Objective

Demonstrate that **fatigue-aware state augmentation prevents habituation failures** in adaptive health-nudging policies using Dynamic Programming, Monte Carlo, and Q-Learning.

## 2. Scientific Model

### State Space
- **Sleep quality** s ∈ {0, 1, 2} (poor, fair, good)
- **Activity level** a ∈ {0, 1, 2} (low, moderate, high)
- **Stress level** σ ∈ {0, 1, 2} (high, moderate, low)  [inverted: 0=high stress]
- **Habituation** h ∈ {0, 1, 2, 3} (none, mild, moderate, severe)

**Memoryless state**: (s, a, σ) — 27 states
**Fatigue-aware state**: (s, a, σ, h) — 108 states

### Action Space
- 0: No nudge
- 1: Sleep nudge
- 2: Activity nudge
- 3: Stress-management nudge

### Response Probability
P(respond | action, habituation) = p_base × decay(h)
- p_base depends on the nudge type (e.g., 0.6 for sleep, 0.5 for activity, 0.55 for stress)
- decay(h) = max(0.1, 1.0 - 0.25 * h)
- No-nudge action: no response
- Response determines whether the targeted health dimension improves

### Habituation Update
- If nudge delivered: h ← min(3, h + 1) with probability p_habit_increase (e.g., 0.3)
- If no nudge: h ← max(0, h - 1) with probability p_habit_decay (e.g., 0.2)
- Habituation increases with repeated nudging, recovers with rest

### Health State Transitions
- If response to nudge: targeted dimension improves by 1 (capped at 2)
- Natural drift: each dimension may decrease by 1 with probability p_drift (e.g., 0.1)
- Dimensions evolve semi-independently

### Reward Function
R(s, a, σ) = (s + a + (2-σ_inverted)) / 6.0  [normalized 0-1]
Or equivalently: R = (sleep + activity + stress_good) / 6.0
Where stress_good maps stress so higher = better

### Discount Factor
γ = 0.99

### Episode Length
T = 28 days

### Evaluation
- 6 independent replications with different random seeds
- 1000 evaluation episodes per replication
- Report mean discounted return with 95% CI

### Paper-Reported Values (Table reference)
| Policy | Memoryless Return | Fatigue-Aware Return |
|--------|------------------|---------------------|
| Rule-based | ~9.00 | ~9.00 |
| Oracle DP | ~9.47 | ~9.47 |
| DP | ~9.74 | ~improved |
| Monte Carlo | ~9.91 | ~improved |
| Q-learning | ~9.87 | ~improved |

> **Note**: These are approximate values from the prompt. The actual paper PDF was not
> found in the workspace. The implementation faithfully follows the specification
> provided but acknowledges this limitation.

## 3. System Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend (React + TS + Vite)  │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │
│  │Dash  │ │Sim   │ │Algos │ │Train │ │Policy│ │
│  │board │ │ulator│ │Comp  │ │Curves│ │Insp  │ │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │
│  ┌──────┐ ┌──────┐ ┌──────┐                    │
│  │Expt  │ │Result│ │Hist  │                    │
│  │Config│ │Repro │ │ory   │                    │
│  └──────┘ └──────┘ └──────┘                    │
└─────────────────┬───────────────────────────────┘
                  │ REST API (JSON)
┌─────────────────┴───────────────────────────────┐
│               Backend (FastAPI + Python)         │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐ │
│  │API Routes│  │Services  │  │Research Engine │ │
│  │         │  │          │  │ ┌────────────┐ │ │
│  │/sim     │  │simulation│  │ │Environment │ │ │
│  │/expt    │  │experiment│  │ │Habituation │ │ │
│  │/config  │  │evaluation│  │ │Reward      │ │ │
│  │/export  │  │export    │  │ │Policies    │ │ │
│  └─────────┘  └──────────┘  │ │DP/MC/QL   │ │ │
│                             │ │Evaluation  │ │ │
│  ┌──────────────────────┐   │ └────────────┘ │ │
│  │  SQLite + SQLAlchemy │   └────────────────┘ │
│  └──────────────────────┘                       │
└─────────────────────────────────────────────────┘
```

## 4. Key Design Decisions

1. **Deterministic seeds**: All random operations use numpy RandomState with explicit seeds
2. **Separation of concerns**: Simulator is pure Python, no API dependencies
3. **Service boundary**: API routes delegate experiment persistence and execution to `ExperimentService`; a separate repository abstraction is not currently implemented
4. **Job execution**: Background threads for long experiments with status tracking
5. **Export formats**: CSV and JSON with full metadata
6. **Memoryless vs Fatigue-aware**: Same simulator, different state representations for learner

## 5. Implemented application status

The workspace now includes a FastAPI application factory (`backend/app/main.py`), SQLite initialization, background experiment execution, a Vite/React research dashboard, and integration tests. API routes are documented through FastAPI's `/docs`. Frontend pages cover overview, single episode simulation, experiment creation/history/comparison, policy inspection, and methodology notes.

The research paper PDF was not supplied in the workspace. The simulator defaults/equations above are retained from the existing implementation and must not be described as independently paper-verified. The UI and README disclose this limitation. The result values in the build prompt are not treated as generated or validated research outcomes.

Known implementation gaps: jobs are local threads without persisted worker recovery or cancellation; experiment comparison does not yet enforce matched configurations; policy inspection does not calculate oracle agreement or expected action values; no database migration framework is configured. These features require follow-up work before a production deployment or a verified reproduction.
