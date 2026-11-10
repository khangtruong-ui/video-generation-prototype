"""API package exposing FastAPI application and schemas."""

from video_gen.api.app import app, create_app
from video_gen.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatusResponse,
)

__all__ = [
    "app",
    "create_app",
    "GenerateRequest",
    "GenerateResponse",
    "JobStatusResponse",
    "HealthResponse",
]
