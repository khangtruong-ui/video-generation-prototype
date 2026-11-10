"""FastAPI Application factory with CORS, lifecycle, and OpenAPI metadata."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from video_gen.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI video generation server."""
    app = FastAPI(
        title="Long Video Generation Prototype API",
        description=(
            "Production-ready API for generating continuous long videos (10s up to minutes) "
            "from prompts and images using SVD-XT, autoregressive chunk chaining, and L4 GPU acceleration."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Allow cross-origin requests (for Colab frontends or remote web apps)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.get("/", tags=["Root"])
    def root():
        return JSONResponse(
            content={
                "name": "Long Video Generation Prototype API",
                "version": "0.1.0",
                "docs": "/docs",
                "health": "/api/v1/health",
                "endpoints": {
                    "generate": "POST /api/v1/generate",
                    "generate_upload": "POST /api/v1/generate/upload",
                    "get_job": "GET /api/v1/jobs/{job_id}",
                    "download_video": "GET /api/v1/videos/{video_id}",
                },
            }
        )

    return app


app = create_app()
