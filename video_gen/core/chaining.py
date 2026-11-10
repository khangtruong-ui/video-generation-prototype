"""Autoregressive long-video chunk chaining, temporal blending, and storyboard planning."""

import logging
import math
from typing import Any, Dict, List
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def calculate_chunks_plan(
    duration_seconds: float,
    fps: int = 7,
    num_frames_per_chunk: int = 25,
    transition_frames: int = 2,
) -> Dict[str, Any]:
    """Calculate the required chunks, overlap frames, and expected duration.

    Args:
        duration_seconds: Target total video duration in seconds (e.g. 10.0, 30.0, 60.0).
        fps: Playback frames per second (e.g. 7).
        num_frames_per_chunk: Frames produced by model per chunk (25 for SVD-XT).
        transition_frames: Number of overlapping frames cross-dissolved between adjacent chunks.

    Returns:
        Dictionary containing planning metrics (num_chunks, total_frames, actual_duration).
    """
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if num_frames_per_chunk <= 0:
        raise ValueError("num_frames_per_chunk must be positive")
    if transition_frames < 0 or transition_frames >= num_frames_per_chunk:
        raise ValueError("transition_frames must be >= 0 and < num_frames_per_chunk")

    target_total_frames = max(1, int(round(duration_seconds * fps)))

    if target_total_frames <= num_frames_per_chunk:
        num_chunks = 1
        total_frames = num_frames_per_chunk
    else:
        effective_per_next_chunk = num_frames_per_chunk - transition_frames
        needed_extra_frames = target_total_frames - num_frames_per_chunk
        extra_chunks = int(math.ceil(needed_extra_frames / effective_per_next_chunk))
        num_chunks = 1 + extra_chunks
        total_frames = num_frames_per_chunk + extra_chunks * effective_per_next_chunk

    actual_duration = total_frames / fps

    plan = {
        "target_duration_seconds": duration_seconds,
        "actual_duration_seconds": round(actual_duration, 2),
        "fps": fps,
        "num_chunks": num_chunks,
        "num_frames_per_chunk": num_frames_per_chunk,
        "transition_frames": transition_frames,
        "target_total_frames": target_total_frames,
        "total_stitched_frames": total_frames,
    }
    logger.info(
        "Chaining plan: target=%.1fs -> %d chunks of %d frames with %d overlap => %d frames (%.2fs)",
        duration_seconds,
        num_chunks,
        num_frames_per_chunk,
        transition_frames,
        total_frames,
        actual_duration,
    )
    return plan


def cross_dissolve_blend(
    chunk_a: List[Image.Image],
    chunk_b: List[Image.Image],
    overlap_frames: int = 2,
) -> List[Image.Image]:
    """Blend the end of chunk_a with the beginning of chunk_b using linear cross-dissolve.

    Args:
        chunk_a: First video chunk (list of PIL Images).
        chunk_b: Second video chunk (list of PIL Images).
        overlap_frames: Number of frames to overlap and blend.

    Returns:
        Combined list of PIL Images with smooth transition.
    """
    if not chunk_a:
        return list(chunk_b)
    if not chunk_b:
        return list(chunk_a)

    overlap = min(overlap_frames, len(chunk_a), len(chunk_b))

    if overlap == 0:
        return list(chunk_a) + list(chunk_b)

    # Main non-overlapping body of chunk_a
    result: List[Image.Image] = list(chunk_a[:-overlap])

    # Blend overlapping region
    for i in range(overlap):
        alpha = (i + 1) / (overlap + 1)  # Progressive blend from 0 to 1
        frame_a = chunk_a[-overlap + i]
        frame_b = chunk_b[i]

        arr_a = np.array(frame_a, dtype=np.float32)
        arr_b = np.array(frame_b, dtype=np.float32)

        blended_arr = (1.0 - alpha) * arr_a + alpha * arr_b
        blended_frame = Image.fromarray(np.clip(blended_arr, 0, 255).astype(np.uint8))
        result.append(blended_frame)

    # Append remaining non-overlapping body of chunk_b
    result.extend(chunk_b[overlap:])
    return result


def chain_chunks(
    chunks: List[List[Image.Image]],
    overlap_frames: int = 2,
) -> List[Image.Image]:
    """Sequentially stitch multiple video chunks into a single seamless frame sequence."""
    if not chunks:
        return []
    if len(chunks) == 1:
        return list(chunks[0])

    stitched: List[Image.Image] = list(chunks[0])
    for next_chunk in chunks[1:]:
        stitched = cross_dissolve_blend(stitched, next_chunk, overlap_frames=overlap_frames)

    return stitched


def distribute_images_to_chunks(
    images: List[Image.Image],
    num_chunks: int,
) -> List[Image.Image]:
    """Distribute input storyboard images across chunks.

    If 1 image is provided, it is returned as initial anchor.
    If multiple images are provided, they are spaced evenly across chunks.
    """
    if not images:
        raise ValueError("Images list cannot be empty")

    if len(images) == 1:
        return [images[0]] * num_chunks

    # Evenly distribute multiple keyframe images across the number of chunks
    assigned: List[Image.Image] = []
    num_images = len(images)
    for c in range(num_chunks):
        img_idx = min(int((c / num_chunks) * num_images), num_images - 1)
        assigned.append(images[img_idx])

    return assigned
