"""FastAPI application entry point."""

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import models
from .api import routes
from .core.config import settings
from .services.experiment_service import ExperimentService


def create_app(database_url: Optional[str] = None) -> FastAPI:
    """Create the API and initialize its database."""
    engine = models.get_engine(database_url)
    models.create_tables(engine)
    routes.experiment_service = ExperimentService(models.get_session_factory(engine))

    application = FastAPI(
        title="Fatigue-Aware Nudging Research Platform",
        description=(
            "Research simulator for health nudging policies. "
            "This synthetic environment is not a clinical decision-support system."
        ),
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(routes.router)

    logging.basicConfig(
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return application


app = create_app()
