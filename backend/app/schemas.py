"""
Pydantic schemas for API request/response validation.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator
from datetime import datetime


# ============================================================
# Enums
# ============================================================

class AlgorithmType(str, Enum):
    RULE_BASED = "rule_based"
    DP = "dp"
    MONTE_CARLO = "mc"
    Q_LEARNING = "q_learning"
    ORACLE_DP = "oracle_dp"


class ExperimentStatusEnum(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================
# Simulator Config
# ============================================================

class SimulatorConfigSchema(BaseModel):
    """Simulator configuration matching the research paper."""
    health_levels: int = Field(default=3, ge=2, le=5)
    habituation_levels: int = Field(default=4, ge=2, le=8)
    episode_length: int = Field(default=28, ge=1, le=365)
    discount_factor: float = Field(default=0.99, ge=0.0, le=1.0)
    
    p_respond_sleep: float = Field(default=0.6, ge=0.0, le=1.0)
    p_respond_activity: float = Field(default=0.5, ge=0.0, le=1.0)
    p_respond_stress: float = Field(default=0.55, ge=0.0, le=1.0)
    
    p_habituation_increase: float = Field(default=0.3, ge=0.0, le=1.0)
    p_habituation_decay: float = Field(default=0.2, ge=0.0, le=1.0)
    habituation_attenuation_rate: float = Field(default=0.25, ge=0.0, le=1.0)
    min_response_probability: float = Field(default=0.1, ge=0.0, le=1.0)
    
    p_drift_sleep: float = Field(default=0.1, ge=0.0, le=1.0)
    p_drift_activity: float = Field(default=0.1, ge=0.0, le=1.0)
    p_drift_stress: float = Field(default=0.1, ge=0.0, le=1.0)
    
    initial_sleep: Optional[int] = Field(default=None, ge=0)
    initial_activity: Optional[int] = Field(default=None, ge=0)
    initial_stress: Optional[int] = Field(default=None, ge=0)
    initial_habituation: int = Field(default=0, ge=0)

    @model_validator(mode='after')
    def validate_initial_state(self):
        for value in (self.initial_sleep, self.initial_activity, self.initial_stress):
            if value is not None and value >= self.health_levels:
                raise ValueError("Initial health values must be below health_levels")
        if self.initial_habituation >= self.habituation_levels:
            raise ValueError("initial_habituation must be below habituation_levels")
        return self


class AlgorithmConfigSchema(BaseModel):
    """Algorithm-specific configuration."""
    # DP parameters
    convergence_tol: float = Field(default=1e-6, ge=1e-12, le=1.0)
    max_iterations: int = Field(default=1000, ge=1, le=10000)
    
    # MC & Q-Learning parameters
    epsilon: float = Field(default=0.1, ge=0.0, le=1.0)
    n_episodes: int = Field(default=50000, ge=100, le=200000)
    epsilon_decay: float = Field(default=0.9999, ge=0.9, le=1.0)
    min_epsilon: float = Field(default=0.01, ge=0.0, le=1.0)
    
    # Q-Learning specific
    learning_rate: float = Field(default=0.1, ge=0.001, le=1.0)
    lr_decay: float = Field(default=1.0, ge=0.9, le=1.0)


# ============================================================
# Simulation Requests/Responses
# ============================================================

class SimulationRequest(BaseModel):
    """Request to run a single simulation episode."""
    config: SimulatorConfigSchema = Field(default_factory=SimulatorConfigSchema)
    seed: int = Field(default=42, ge=0)
    algorithm: AlgorithmType = AlgorithmType.RULE_BASED
    fatigue_aware: bool = False
    algorithm_config: AlgorithmConfigSchema = Field(default_factory=AlgorithmConfigSchema)


class StepInfo(BaseModel):
    """Information about a single simulation step."""
    day: int
    state_before: dict[str, int]
    action: int
    action_name: str
    response_probability: float
    responded: bool
    reward: float
    state_after: dict[str, int]
    habituation_before: int
    habituation_after: int


class SimulationResponse(BaseModel):
    """Response from a simulation run."""
    seed: int
    algorithm: str
    fatigue_aware: bool
    episode_length: int
    discounted_return: float
    total_reward: float
    mean_health_score: float
    total_nudges: int
    response_rate: float
    mean_habituation: float
    steps: list[StepInfo]


# ============================================================
# Experiment Requests/Responses
# ============================================================

class ExperimentCreateRequest(BaseModel):
    """Request to create and run an experiment."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    algorithm: AlgorithmType
    fatigue_aware: bool = False
    simulator_config: SimulatorConfigSchema = Field(default_factory=SimulatorConfigSchema)
    algorithm_config: AlgorithmConfigSchema = Field(default_factory=AlgorithmConfigSchema)
    n_training_episodes: Optional[int] = Field(default=None, ge=100, le=200000)
    n_eval_episodes: int = Field(default=1000, ge=10, le=5000)
    n_replications: int = Field(default=6, ge=1, le=20)
    base_seed: int = Field(default=42, ge=0)

    @model_validator(mode='after')
    def set_training_episodes(self):
        if self.n_training_episodes is None and self.algorithm in [
            AlgorithmType.MONTE_CARLO,
            AlgorithmType.Q_LEARNING,
        ]:
            self.n_training_episodes = 50000
        return self


class ExperimentSummary(BaseModel):
    """Summary of an experiment for list views."""
    id: int
    name: str
    description: Optional[str]
    status: str
    algorithm: str
    fatigue_aware: bool
    mean_discounted_return: Optional[float]
    ci_95_lower: Optional[float]
    ci_95_upper: Optional[float]
    mean_health_score: Optional[float]
    created_at: Optional[str]
    completed_at: Optional[str]


class ExperimentDetail(BaseModel):
    """Full experiment details."""
    id: int
    name: str
    description: Optional[str]
    status: str
    error_message: Optional[str]
    algorithm: str
    fatigue_aware: bool
    simulator_config: Optional[dict[str, Any]]
    algorithm_config: Optional[dict[str, Any]]
    n_training_episodes: Optional[int]
    n_eval_episodes: int
    n_replications: int
    base_seed: int
    mean_discounted_return: Optional[float]
    ci_95_lower: Optional[float]
    ci_95_upper: Optional[float]
    mean_health_score: Optional[float]
    mean_response_rate: Optional[float]
    mean_habituation: Optional[float]
    mean_nudges: Optional[float]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    version: Optional[str]


class ExperimentResults(BaseModel):
    """Full experiment results including evaluation data."""
    experiment: ExperimentDetail
    evaluation: Optional[dict[str, Any]] = None
    training_metrics: Optional[dict[str, Any]] = None
    sample_episode: Optional[dict[str, Any]] = None


class PolicyEntry(BaseModel):
    """A single entry in the policy table."""
    state: list[int]
    action: int
    action_name: str
    q_values: Optional[dict[str, float]] = None
    response_probability: Optional[float] = None


class PolicyInspection(BaseModel):
    """Policy inspection response."""
    algorithm: str
    fatigue_aware: bool
    total_states: int
    entries: list[PolicyEntry]


# ============================================================
# Comparison
# ============================================================

class ComparisonRequest(BaseModel):
    """Request to compare multiple experiment results."""
    experiment_ids: list[int] = Field(..., min_length=2)


class ComparisonEntry(BaseModel):
    """A single entry in a comparison."""
    id: int
    name: str
    algorithm: str
    fatigue_aware: bool
    mean_discounted_return: Optional[float]
    ci_95_lower: Optional[float]
    ci_95_upper: Optional[float]
    mean_health_score: Optional[float]
    mean_response_rate: Optional[float]
    mean_habituation: Optional[float]
    mean_nudges: Optional[float]


class ComparisonResponse(BaseModel):
    """Response for experiment comparison."""
    experiments: list[ComparisonEntry]


# ============================================================
# General
# ============================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str


class DefaultsResponse(BaseModel):
    """Default configuration response."""
    simulator: SimulatorConfigSchema
    algorithm: AlgorithmConfigSchema
    algorithms: list[dict[str, Any]]


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_type: Optional[str] = None
