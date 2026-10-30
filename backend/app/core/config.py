"""
Application configuration loaded from environment variables.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./fatigue_nudging.db"
    
    # CORS
    CORS_ORIGINS: list[str] | None = None
    
    # Experiment limits
    MAX_TRAINING_EPISODES: int = 200000
    MAX_EVAL_EPISODES: int = 5000
    MAX_REPLICATIONS: int = 20
    MAX_EPISODE_LENGTH: int = 365
    MAX_EXPERIMENT_STEPS: int = 5000000
    
    # Background jobs
    MAX_CONCURRENT_JOBS: int = 3
    
    # Export
    EXPORT_DIR: str = "./exports"
    
    def __post_init__(self):
        if self.CORS_ORIGINS is None:
            self.CORS_ORIGINS = [
                "http://localhost:5173",
                "http://localhost:3000",
                "http://127.0.0.1:5173",
                "http://127.0.0.1:3000",
            ]
        
        # Ensure export directory exists
        os.makedirs(self.EXPORT_DIR, exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Load settings from environment variables."""
        return cls(
            HOST=os.getenv("HOST", "0.0.0.0"),
            PORT=int(os.getenv("PORT", "8000")),
            DEBUG=os.getenv("DEBUG", "true").lower() == "true",
            DATABASE_URL=os.getenv("DATABASE_URL", "sqlite:///./fatigue_nudging.db"),
            CORS_ORIGINS=(
                [
                    origin.strip().rstrip("/")
                    for origin in os.getenv("CORS_ORIGINS", "").split(",")
                    if origin.strip()
                ]
                if os.getenv("CORS_ORIGINS")
                else None
            ),
            MAX_TRAINING_EPISODES=int(os.getenv("MAX_TRAINING_EPISODES", "200000")),
            MAX_EVAL_EPISODES=int(os.getenv("MAX_EVAL_EPISODES", "5000")),
            MAX_REPLICATIONS=int(os.getenv("MAX_REPLICATIONS", "20")),
            MAX_EPISODE_LENGTH=int(os.getenv("MAX_EPISODE_LENGTH", "365")),
            MAX_EXPERIMENT_STEPS=int(os.getenv("MAX_EXPERIMENT_STEPS", "5000000")),
            MAX_CONCURRENT_JOBS=int(os.getenv("MAX_CONCURRENT_JOBS", "3")),
            EXPORT_DIR=os.getenv("EXPORT_DIR", "./exports"),
        )


settings = Settings.from_env()
