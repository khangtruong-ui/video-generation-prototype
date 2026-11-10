"""FastAPI REST routes for video generation, file streaming, and job status."""

import asyncio
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from video_gen.api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    JobStatusResponse,
)
from video_gen.config import VideoConfig
from video_gen.generator import LongVideoGenerator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Video Generation"])

# Global job registry: job_id -> dict
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

# Global generator instance cache: pipeline_type -> LongVideoGenerator
_GENERATORS: Dict[str, LongVideoGenerator] = {}


def get_generator(pipeline_type: Optional[str] = None) -> LongVideoGenerator:
    """Retrieve or initialize generator instance."""
    p_type = pipeline_type or ("svd_xt" if torch.cuda.is_available() else "mock")
    if p_type not in _GENERATORS:
        if p_type == "mock":
            cfg = VideoConfig.for_mock()
        else:
            cfg = VideoConfig.for_l4_gpu()
        _GENERATORS[p_type] = LongVideoGenerator(config=cfg)
    return _GENERATORS[p_type]


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Retrieve service health, GPU memory allocation, and pipeline status."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    vram_alloc = torch.cuda.memory_allocated(0) / (1024**3) if gpu_available else 0.0
    vram_res = torch.cuda.memory_reserved(0) / (1024**3) if gpu_available else 0.0
    vram_total = (
        torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_available
        else None
    )

    p_type = "svd_xt" if gpu_available else "mock"
    gen = _GENERATORS.get(p_type)
    loaded = gen.pipeline.is_loaded() if gen else False

    cfg = VideoConfig.for_l4_gpu()
    return HealthResponse(
        status="healthy",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_allocated_gb=round(vram_alloc, 3),
        vram_reserved_gb=round(vram_res, 3),
        vram_total_gb=round(vram_total, 2) if vram_total else None,
        pipeline_type=p_type,
        pipeline_loaded=loaded,
        default_config={
            "duration_seconds": cfg.generation.duration_seconds,
            "fps": cfg.generation.fps,
            "width": cfg.generation.width,
            "height": cfg.generation.height,
            "motion_bucket_id": cfg.generation.motion_bucket_id,
            "num_inference_steps": cfg.generation.num_inference_steps,
        },
    )


def _run_generation_task(job_id: str, req: GenerateRequest, output_dir: Path):
    """Background task runner for async generation."""
    try:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["message"] = "Starting generation..."

        gen = get_generator(req.pipeline_type)

        def progress_cb(prog: Dict[str, Any]):
            with JOBS_LOCK:
                if job_id in JOBS:
                    JOBS[job_id]["progress_percent"] = prog.get("overall_percent", 0.0)
                    JOBS[job_id]["message"] = prog.get("message", "")

        out_path = output_dir / f"{job_id}.mp4"
        res = gen.generate(
            prompt=req.prompt,
            prompt_file=req.prompt_file,
            image=req.image,
            images=req.images,
            duration_seconds=req.duration_seconds,
            fps=req.fps,
            motion_bucket_id=req.motion_bucket_id,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
            output_path=out_path,
            progress_callback=progress_cb,
        )

        with JOBS_LOCK:
            JOBS[job_id]["status"] = "completed"
            JOBS[job_id]["progress_percent"] = 100.0
            JOBS[job_id]["message"] = "Completed successfully"
            JOBS[job_id]["video_path"] = str(res.video_path)
            JOBS[job_id]["download_url"] = f"/api/v1/videos/{job_id}"
            JOBS[job_id]["metadata"] = res.metadata

    except Exception as exc:
        logger.exception("Error in background generation job %s: %s", job_id, exc)
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = str(exc)
            JOBS[job_id]["message"] = f"Failed: {exc}"


@router.post("/generate", response_model=GenerateResponse)
def generate_video(req: GenerateRequest, background_tasks: BackgroundTasks):
    """Generate a long video from prompt and image(s).

    Supports synchronous return or asynchronous background processing via `async_mode=True`.
    """
    job_id = f"gen_{uuid.uuid4().hex[:12]}"
    output_dir = Path("./outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if req.async_mode:
        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "pending",
                "progress_percent": 0.0,
                "message": "Queued for generation",
                "download_url": None,
                "metadata": None,
                "error": None,
            }
        background_tasks.add_task(_run_generation_task, job_id, req, output_dir)
        return GenerateResponse(
            video_id=job_id,
            status="processing",
            download_url=f"/api/v1/videos/{job_id}",
            stream_url=f"/api/v1/videos/{job_id}/stream",
            metadata={"job_id": job_id, "mode": "async"},
        )

    # Synchronous mode:
    try:
        gen = get_generator(req.pipeline_type)
        out_path = output_dir / f"{job_id}.mp4"
        res = gen.generate(
            prompt=req.prompt,
            prompt_file=req.prompt_file,
            image=req.image,
            images=req.images,
            duration_seconds=req.duration_seconds,
            fps=req.fps,
            motion_bucket_id=req.motion_bucket_id,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
            output_path=out_path,
        )

        with JOBS_LOCK:
            JOBS[job_id] = {
                "job_id": job_id,
                "status": "completed",
                "video_path": str(res.video_path),
                "download_url": f"/api/v1/videos/{job_id}",
                "metadata": res.metadata,
            }

        return GenerateResponse(
            video_id=job_id,
            status="completed",
            download_url=f"/api/v1/videos/{job_id}",
            stream_url=f"/api/v1/videos/{job_id}/stream",
            duration_seconds=res.duration_seconds,
            fps=res.fps,
            num_frames=res.num_frames,
            num_chunks=res.metadata.get("num_chunks"),
            metadata=res.metadata,
        )
    except Exception as exc:
        logger.exception("Synchronous generation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {exc}",
        )


@router.post("/generate/upload", response_model=GenerateResponse)
async def generate_video_upload(
    prompt: Optional[str] = Form(None),
    prompt_file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    duration_seconds: float = Form(10.0),
    fps: int = Form(7),
    motion_bucket_id: Optional[int] = Form(None),
    num_inference_steps: Optional[int] = Form(None),
    seed: Optional[int] = Form(None),
    pipeline_type: Optional[str] = Form(None),
):
    """Generate a video accepting multipart file uploads for prompt script and images."""
    upload_dir = Path("./outputs/uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Handle uploaded prompt file
    resolved_prompt_text = prompt
    if prompt_file is not None and prompt_file.filename:
        prompt_bytes = await prompt_file.read()
        resolved_prompt_text = prompt_bytes.decode("utf-8")

    # Handle uploaded image file
    image_bytes = None
    if image is not None and image.filename:
        image_bytes = await image.read()

    req = GenerateRequest(
        prompt=resolved_prompt_text,
        image=image_bytes if image_bytes else None,
        duration_seconds=duration_seconds,
        fps=fps,
        motion_bucket_id=motion_bucket_id,
        num_inference_steps=num_inference_steps,
        seed=seed,
        pipeline_type=pipeline_type,
        async_mode=False,
    )

    # Use generate_video helper
    bg = BackgroundTasks()
    return generate_video(req, background_tasks=bg)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    """Query progress or result of an asynchronous generation job."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress_percent=job.get("progress_percent", 0.0),
        message=job.get("message", ""),
        download_url=job.get("download_url"),
        metadata=job.get("metadata"),
        error=job.get("error"),
    )


@router.get("/videos/{video_id}")
def download_video(video_id: str):
    """Download the generated MP4 video file."""
    # Strip extension if passed
    v_id = video_id[:-4] if video_id.endswith(".mp4") else video_id

    # Check job registry first
    with JOBS_LOCK:
        job = JOBS.get(v_id)
        if job and "video_path" in job:
            path = Path(job["video_path"])
            if path.exists():
                return FileResponse(
                    str(path),
                    media_type="video/mp4",
                    filename=f"{v_id}.mp4",
                )

    # Check outputs directory
    potential_file = Path("./outputs") / f"{v_id}.mp4"
    if potential_file.exists():
        return FileResponse(
            str(potential_file),
            media_type="video/mp4",
            filename=f"{v_id}.mp4",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Video '{video_id}' not found.",
    )


@router.get("/videos/{video_id}/stream")
def stream_video(video_id: str):
    """Stream video file inline in browser."""
    return download_video(video_id)
