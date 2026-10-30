"""
SQLAlchemy database models for experiment persistence.
"""

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, make_url
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.orm.session import Session
from enum import Enum as PyEnum


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


class Base(DeclarativeBase):
    pass


class ExperimentStatus(str, PyEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentModel(Base):
    """Persisted experiment with configuration and results."""
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(
        String(50), default=ExperimentStatus.QUEUED.value, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Configuration (JSON)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    fatigue_aware: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    simulator_config: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    algorithm_config: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Training parameters
    n_training_episodes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_eval_episodes: Mapped[int] = mapped_column(Integer, default=1000, nullable=False)
    n_replications: Mapped[int] = mapped_column(Integer, default=6, nullable=False)
    base_seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)

    # Results (JSON)
    evaluation_results: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    training_metrics: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    policy_table: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Aggregate metrics (denormalized for quick access)
    mean_discounted_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_95_lower: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_95_upper: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_health_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_response_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_habituation: Mapped[float | None] = mapped_column(Float, nullable=True)
    mean_nudges: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Version tracking
    version: Mapped[str] = mapped_column(String(50), default="1.0.0", nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status,
            'error_message': self.error_message,
            'algorithm': self.algorithm,
            'fatigue_aware': self.fatigue_aware,
            'simulator_config': json.loads(self.simulator_config) if self.simulator_config else None,
            'algorithm_config': json.loads(self.algorithm_config) if self.algorithm_config else None,
            'n_training_episodes': self.n_training_episodes,
            'n_eval_episodes': self.n_eval_episodes,
            'n_replications': self.n_replications,
            'base_seed': self.base_seed,
            'mean_discounted_return': self.mean_discounted_return,
            'ci_95_lower': self.ci_95_lower,
            'ci_95_upper': self.ci_95_upper,
            'mean_health_score': self.mean_health_score,
            'mean_response_rate': self.mean_response_rate,
            'mean_habituation': self.mean_habituation,
            'mean_nudges': self.mean_nudges,
            'created_at': _datetime_to_utc_iso(self.created_at),
            'started_at': _datetime_to_utc_iso(self.started_at),
            'completed_at': _datetime_to_utc_iso(self.completed_at),
            'version': self.version,
        }


class LogEntryModel(Base):
    """Log entries for experiment execution."""
    __tablename__ = "log_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"), nullable=False, index=True
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utc_now, nullable=False
    )
    level: Mapped[str] = mapped_column(String(20), default="INFO", nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)


# Database setup
def get_engine(database_url: str | None = None) -> Engine:
    """Create database engine."""
    from .core.config import settings

    url = database_url or settings.DATABASE_URL
    is_sqlite = make_url(url).get_backend_name() == "sqlite"
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    return create_engine(
        url,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=not is_sqlite,
    )


def create_tables(engine: Engine | None = None) -> Engine:
    """Create all database tables."""
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    """Get a session factory."""
    if engine is None:
        engine = get_engine()
    return sessionmaker(bind=engine, expire_on_commit=False)
