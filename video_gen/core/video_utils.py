"""Video encoding, decoding, frame interpolation, and metadata utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import imageio
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def save_video(
    frames: List[Image.Image],
    output_path: Union[str, Path],
    fps: int = 7,
    codec: str = "libx264",
    quality: int = 8,
) -> Path:
    """Encode a sequence of PIL Images to MP4 using imageio and ffmpeg.

    Ensures yuv420p pixel format for universal browser & device playback compatibility.

    Args:
        frames: List of PIL RGB Images.
        output_path: Target path for the output .mp4 file.
        fps: Playback framerate.
        codec: Video codec (default libx264).
        quality: Video compression quality (1-10, where 10 is highest).

    Returns:
        Path to the saved video file.
    """
    if not frames:
        raise ValueError("Cannot save video with empty frames list")

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert PIL Images to numpy arrays
    np_frames = [np.array(f.convert("RGB")) for f in frames]

    # imageio ffmpeg writer with yuv420p and macro_block_size=None to prevent unneeded resizing
    writer = imageio.get_writer(
        str(out_file),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec=codec,
        quality=quality,
        pixelformat="yuv420p",
        macro_block_size=None,
        output_params=["-movflags", "faststart"],  # Web-optimized MP4 header
    )

    try:
        for frame in np_frames:
            writer.append_data(frame)
    finally:
        writer.close()

    if not out_file.exists() or out_file.stat().st_size == 0:
        raise RuntimeError(f"Video file was not successfully generated at {out_file}")

    file_size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(
        "Successfully saved video (%d frames, %d fps, %.2fMB) to: %s",
        len(frames),
        fps,
        file_size_mb,
        out_file,
    )
    return out_file


def interpolate_frames(
    frames: List[Image.Image],
    factor: Optional[int] = None,
    target_fps: Optional[int] = None,
    source_fps: int = 16,
    method: str = "auto",
    device: Optional[str] = None,
) -> List[Image.Image]:
    """Upsample framerate by interpolating intermediate frames.

    Uses deep learning optical flow (RAFT) or GPU tensor blending to achieve high-FPS video.

    Args:
        frames: Sequence of PIL images.
        factor: Integer upsampling factor (e.g. 2 for 2x fps).
        target_fps: Target high framerate (e.g. 30, 60 fps).
        source_fps: Input native playback framerate (default 16).
        method: "auto", "raft" (deep optical flow), or "blend" (GPU tensor blend).
        device: "cuda" or "cpu" (defaults to CUDA if available).

    Returns:
        List of interpolated PIL images.
    """
    if len(frames) < 2:
        return list(frames)

    # Backward compatible default
    effective_factor = factor
    if effective_factor is None and target_fps is None:
        effective_factor = 2

    from video_gen.core.interpolator import GPUFrameInterpolator

    interpolator = GPUFrameInterpolator(device=device)
    result_frames, _ = interpolator.interpolate_sequence(
        frames=frames,
        factor=effective_factor,
        target_fps=target_fps,
        source_fps=source_fps,
        method=method,
    )
    return result_frames


def read_video_metadata(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract metadata (duration, fps, frame count, resolution) from a video file."""
    path_str = str(video_path)
    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {path_str}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = (frame_count / fps) if fps > 0 else 0.0
    return {
        "path": path_str,
        "width": width,
        "height": height,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "duration_seconds": round(duration, 2),
    }
