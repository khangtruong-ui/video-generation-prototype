"""Video encoding, decoding, frame interpolation, and metadata utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Union
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
    factor: int = 2,
) -> List[Image.Image]:
    """Upsample framerate by interpolating intermediate frames via linear frame blending.

    For factor=2: duplicates frame count and doubles smoothness (e.g. 7 fps -> 14 fps).
    """
    if factor <= 1 or len(frames) < 2:
        return list(frames)

    interpolated: List[Image.Image] = []
    for i in range(len(frames) - 1):
        f0 = np.array(frames[i], dtype=np.float32)
        f1 = np.array(frames[i + 1], dtype=np.float32)

        for step in range(factor):
            weight = step / factor
            blended = (1.0 - weight) * f0 + weight * f1
            interpolated.append(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)))

    # Append the last frame
    interpolated.append(frames[-1])
    return interpolated


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
