"""Pydantic request and response schemas for Video Generation API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Payload for video generation request."""
    prompt: Optional[str] = Field(
        default=None,
        description="Text prompt guiding scene and motion (direct text).",
        examples=["Cinematic drone flying over majestic misty mountains at sunset, 4k photorealistic"]
    )
    prompt_file: Optional[str] = Field(
        default=None,
        description="Path or filename of a .txt file containing the prompt script."
    )
    image: Optional[str] = Field(
        default=None,
        description="Conditioning image URL (http/https), local file path, or base64 data URI. Defaults to internet landscape if omitted."
    )
    images: Optional[List[str]] = Field(
        default=None,
        description="List of multiple image URLs or paths for multi-scene storyboard video generation."
    )
    duration_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=600.0,
        description="Target duration in seconds (default 10s, supports up to minutes)."
    )
    fps: int = Field(
        default=16,
        ge=1,
        le=60,
        description="Chunk generation frames per second (default 16 for Wan2.1).",
    )
    target_fps: Optional[int] = Field(
        default=60,
        ge=1,
        le=240,
        description="Target high framerate achieved via GPU optical-flow interpolation (e.g. 60 fps).",
    )
    interpolate_factor: Optional[int] = Field(
        default=None,
        ge=1,
        le=16,
        description="Frame upsampling multiplier (e.g. 2x, 4x) if target_fps is omitted.",
    )
    interpolate_method: Optional[str] = Field(
        default="auto",
        description="Interpolation engine: 'auto', 'raft' (deep optical flow), or 'blend' (GPU tensor).",
    )
    motion_bucket_id: Optional[int] = Field(
        default=None,
        ge=1,
        le=255,
        description="Motion magnitude (1-255). If omitted, automatically inferred from prompt.",
    )
    num_inference_steps: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Denoising steps per chunk (default 30 for Wan on L4).",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )
    async_mode: bool = Field(
        default=False,
        description="If True, executes asynchronously in background and returns job ID immediately.",
    )
    pipeline_type: Optional[str] = Field(
        default=None,
        description="Pipeline engine: 'wan' (default Wan2.1 14B on GPU), 'svd_xt', 'ltx_video', or 'mock'.",
    )


class GenerateResponse(BaseModel):
    """Response returned upon video generation or job dispatch."""
    video_id: str
    status: str  # "completed", "processing", "failed"
    download_url: Optional[str] = None
    stream_url: Optional[str] = None
    duration_seconds: Optional[float] = None
    fps: Optional[int] = None
    num_frames: Optional[int] = None
    num_chunks: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Job status query response."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress_percent: float = 0.0
    message: str = ""
    download_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """System health and hardware status."""
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    vram_allocated_gb: float = 0.0
    vram_reserved_gb: float = 0.0
    vram_total_gb: Optional[float] = None
    pipeline_type: str
    pipeline_loaded: bool
    default_config: Dict[str, Any]
