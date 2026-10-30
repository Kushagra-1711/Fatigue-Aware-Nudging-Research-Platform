"""
FastAPI application routes for the Fatigue-Aware Nudging Research Platform.
"""

import json
import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from ..schemas import (
    SimulationRequest, SimulationResponse, StepInfo,
    ExperimentCreateRequest, ExperimentSummary, ExperimentDetail,
    ExperimentResults, PolicyInspection, PolicyEntry,
    ComparisonRequest, ComparisonResponse, ComparisonEntry,
    HealthResponse, DefaultsResponse, ErrorResponse,
    SimulatorConfigSchema, AlgorithmConfigSchema, AlgorithmType,
)
from ..research.environment import (
    NudgingEnvironment, SimulatorConfig, HealthState, Action
)
from ..research.policies import create_policy
from ..research.evaluation import evaluate_policy, compute_policy_agreement
from ..core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

# Will be set by main.py
experiment_service = None


def get_experiment_service():
    if experiment_service is None:
        raise HTTPException(500, "Experiment service not initialized")
    return experiment_service


# ============================================================
# Health & Config
# ============================================================

@router.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        database="connected",
    )


@router.get("/api/config/defaults", response_model=DefaultsResponse)
async def get_defaults():
    """Get default configuration values."""
    return DefaultsResponse(
        simulator=SimulatorConfigSchema(),
        algorithm=AlgorithmConfigSchema(),
        algorithms=[
            {
                "id": "rule_based",
                "name": "Rule-Based Baseline",
                "description": "Simple rule-based policy that nudges based on health state thresholds.",
                "requires_training": False,
                "supports_fatigue_aware": False,
            },
            {
                "id": "dp",
                "name": "Dynamic Programming",
                "description": "Value iteration using the analytical transition model.",
                "requires_training": True,
                "supports_fatigue_aware": True,
            },
            {
                "id": "mc",
                "name": "Monte Carlo Control",
                "description": "First-visit MC control with epsilon-soft exploration.",
                "requires_training": True,
                "supports_fatigue_aware": True,
            },
            {
                "id": "q_learning",
                "name": "Q-Learning",
                "description": "Tabular Q-learning with epsilon-greedy exploration.",
                "requires_training": True,
                "supports_fatigue_aware": True,
            },
            {
                "id": "oracle_dp",
                "name": "Oracle DP",
                "description": "DP with full state access (including habituation). Evaluation reference only.",
                "requires_training": True,
                "supports_fatigue_aware": True,
            },
        ],
    )


@router.get("/api/algorithms")
async def list_algorithms():
    """List available algorithms."""
    return {
        "algorithms": [
            {"id": "rule_based", "name": "Rule-Based", "requires_training": False},
            {"id": "dp", "name": "Dynamic Programming", "requires_training": True},
            {"id": "mc", "name": "Monte Carlo", "requires_training": True},
            {"id": "q_learning", "name": "Q-Learning", "requires_training": True},
            {"id": "oracle_dp", "name": "Oracle DP", "requires_training": True},
        ]
    }


# ============================================================
# Simulations (single episode)
# ============================================================

@router.post("/api/simulations", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run a single simulation episode and return step-by-step results."""
    try:
        # Build config
        sim_config = SimulatorConfig(**request.config.model_dump())
        
        # Create policy
        algo_kwargs = request.algorithm_config.model_dump()
        policy = create_policy(
            request.algorithm.value,
            fatigue_aware=request.fatigue_aware,
            **algo_kwargs,
        )
        
        # Train if needed
        if hasattr(policy, 'train') and not policy.trained:
            estimated_steps = getattr(policy, "n_episodes", 0) * sim_config.episode_length
            if estimated_steps > settings.MAX_EXPERIMENT_STEPS:
                raise HTTPException(
                    422,
                    "Training workload exceeds the configured execution limit",
                )
            policy.train(sim_config, seed=request.seed)
        
        # Run episode
        env = NudgingEnvironment(sim_config, seed=request.seed + 99999)
        state = env.reset()
        
        steps = []
        total_reward = 0
        total_nudges = 0
        response_count = 0
        discounted_return = 0
        gamma = sim_config.discount_factor
        total_hab = state.habituation
        health_scores = [state.health_score()]
        
        for t in range(sim_config.episode_length):
            state_key = policy.get_state_key(state)
            action = policy.select_action(state_key)
            
            next_state, reward, done, info = env.step(action)
            
            steps.append(StepInfo(
                day=info.day,
                state_before=info.state_before,
                action=info.action,
                action_name=info.action_name,
                response_probability=round(info.response_probability, 4),
                responded=info.responded,
                reward=round(info.reward, 4),
                state_after=info.state_after,
                habituation_before=info.habituation_before,
                habituation_after=info.habituation_after,
            ))
            
            total_reward += reward
            discounted_return += reward * (gamma ** t)
            if action != Action.NO_NUDGE:
                total_nudges += 1
                if info.responded:
                    response_count += 1
            total_hab += next_state.habituation
            health_scores.append(next_state.health_score())
            state = next_state
            
            if done:
                break
        
        T = len(steps)
        return SimulationResponse(
            seed=request.seed,
            algorithm=policy.name,
            fatigue_aware=request.fatigue_aware,
            episode_length=T,
            discounted_return=round(discounted_return, 4),
            total_reward=round(total_reward, 4),
            mean_health_score=round(sum(health_scores) / len(health_scores), 4),
            total_nudges=total_nudges,
            response_rate=round(response_count / max(total_nudges, 1), 4),
            mean_habituation=round(total_hab / (T + 1), 4),
            steps=steps,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("Single simulation failed unexpectedly")
        raise HTTPException(500, "Simulation failed unexpectedly")


# ============================================================
# Experiments
# ============================================================

@router.post("/api/experiments")
async def create_experiment(request: ExperimentCreateRequest):
    """Create and start a new experiment."""
    svc = get_experiment_service()
    training_episodes = (
        request.n_training_episodes or request.algorithm_config.n_episodes
        if request.algorithm in {AlgorithmType.MONTE_CARLO, AlgorithmType.Q_LEARNING}
        else 0
    )
    estimated_steps = (
        training_episodes
        + request.n_eval_episodes * request.n_replications
    ) * request.simulator_config.episode_length
    if estimated_steps > settings.MAX_EXPERIMENT_STEPS:
        raise HTTPException(
            422,
            "Estimated experiment workload exceeds the configured execution limit",
        )
    try:
        data = request.model_dump()
        data['simulator_config'] = request.simulator_config.model_dump()
        data['algorithm_config'] = request.algorithm_config.model_dump()
        data['algorithm'] = request.algorithm.value
        
        result = svc.create_experiment(data)
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        logger.exception("Experiment creation failed unexpectedly")
        raise HTTPException(500, "Experiment creation failed unexpectedly")


@router.get("/api/experiments")
async def list_experiments(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all experiments."""
    svc = get_experiment_service()
    experiments = svc.list_experiments(limit=limit, offset=offset)
    total = svc.get_experiment_count()
    return {"experiments": experiments, "total": total}


@router.get("/api/experiments/{experiment_id}")
async def get_experiment(experiment_id: int):
    """Get experiment details."""
    svc = get_experiment_service()
    experiment = svc.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return experiment


@router.get("/api/experiments/{experiment_id}/results")
async def get_experiment_results(experiment_id: int):
    """Get full experiment results including evaluation data."""
    svc = get_experiment_service()
    results = svc.get_experiment_results(experiment_id)
    if results is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return results


@router.get("/api/experiments/{experiment_id}/logs")
async def get_experiment_logs(experiment_id: int):
    """Get experiment execution logs."""
    svc = get_experiment_service()
    experiment = svc.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    logs = svc.get_experiment_logs(experiment_id)
    return {"logs": logs}


@router.get("/api/experiments/{experiment_id}/policy")
async def get_experiment_policy(experiment_id: int):
    """Get the learned policy table."""
    svc = get_experiment_service()
    policy = svc.get_experiment_policy(experiment_id)
    if policy is None:
        raise HTTPException(404, f"Policy for experiment {experiment_id} not found")
    return policy


@router.post("/api/experiments/{experiment_id}/rerun")
async def rerun_experiment(experiment_id: int, new_seed: Optional[int] = None):
    """Re-run an experiment with same or new seed."""
    svc = get_experiment_service()
    try:
        result = svc.rerun_experiment(experiment_id, new_seed)
        return result
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(400, str(e))


@router.get("/api/experiments/{experiment_id}/export")
async def export_experiment(
    experiment_id: int,
    format: str = Query(default="json", pattern="^(json|csv)$"),
):
    """Export experiment results."""
    svc = get_experiment_service()
    export = svc.export_results(experiment_id, format)
    if export is None:
        raise HTTPException(404, f"Experiment {experiment_id} not found")
    return export


# ============================================================
# Dashboard
# ============================================================

@router.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard summary."""
    svc = get_experiment_service()
    return svc.get_dashboard_stats()


# ============================================================
# Comparison
# ============================================================

@router.post("/api/compare")
async def compare_experiments(request: ComparisonRequest):
    """Compare multiple experiments."""
    svc = get_experiment_service()
    entries = []
    
    for exp_id in request.experiment_ids:
        exp = svc.get_experiment(exp_id)
        if exp is None:
            raise HTTPException(404, f"Experiment {exp_id} not found")
        
        entries.append(ComparisonEntry(
            id=exp['id'],
            name=exp['name'],
            algorithm=exp['algorithm'],
            fatigue_aware=exp['fatigue_aware'],
            mean_discounted_return=exp.get('mean_discounted_return'),
            ci_95_lower=exp.get('ci_95_lower'),
            ci_95_upper=exp.get('ci_95_upper'),
            mean_health_score=exp.get('mean_health_score'),
            mean_response_rate=exp.get('mean_response_rate'),
            mean_habituation=exp.get('mean_habituation'),
            mean_nudges=exp.get('mean_nudges'),
        ))
    
    return ComparisonResponse(experiments=entries)
