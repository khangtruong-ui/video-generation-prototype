"""Long video orchestrator: chaining, conditioning, and generation pipeline."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image
import torch

from video_gen.config import VideoConfig
from video_gen.core.chaining import (
    calculate_chunks_plan,
    chain_chunks,
    distribute_images_to_chunks,
)
from video_gen.core.image_loader import ImageInput, load_images
from video_gen.core.prompt_loader import infer_motion_bucket_id, load_prompt
from video_gen.core.video_utils import interpolate_frames, save_video
from video_gen.pipelines import BaseVideoPipeline, get_pipeline

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Encapsulates the output of long video generation."""
    video_path: Path
    frames: List[Image.Image] = field(repr=False)
    duration_seconds: float
    fps: int
    num_frames: int
    width: int
    height: int
    metadata: Dict[str, Any]

    def save(self, path: Union[str, Path]) -> Path:
        """Save or copy the video to a new destination."""
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        return save_video(self.frames, target, fps=self.fps)

    def display(
        self,
        width: int = 640,
        height: Optional[int] = None,
        autoplay: bool = True,
        loop: bool = True,
        include_download: bool = True,
        title: Optional[str] = None,
    ):
        """Helper to render video in an active Jupyter/Colab notebook with HTML5 player and download button."""
        from video_gen.colab import display_video
        return display_video(
            self,
            width=width,
            height=height,
            autoplay=autoplay,
            loop=loop,
            include_download=include_download,
            title=title,
        )


class LongVideoGenerator:
    """Master generator producing long videos (10s up to minutes) via chunk chaining."""

    def __init__(
        self,
        config: Optional[VideoConfig] = None,
        pipeline: Optional[BaseVideoPipeline] = None,
    ):
        self.config = config or VideoConfig.for_l4_gpu()
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = get_pipeline(
                pipeline_type=self.config.generation.pipeline_type,
                config=self.config,
            )

    def generate(
        self,
        prompt: Optional[str] = None,
        prompt_file: Optional[Union[str, Path]] = None,
        image: Optional[ImageInput] = None,
        images: Optional[List[ImageInput]] = None,
        duration_seconds: Optional[float] = None,
        fps: Optional[int] = None,
        target_fps: Optional[int] = None,
        interpolate_factor: Optional[int] = None,
        interpolate_method: Optional[str] = None,
        motion_bucket_id: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        noise_aug_strength: Optional[float] = None,
        seed: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GenerationResult:
        """Generate a long video from prompt and conditioning image(s).

        Args:
            prompt: Text prompt describing the scene/motion.
            prompt_file: Path to a .txt file containing the prompt.
            image: Single conditioning image (URL, path, base64, or PIL Image).
            images: Multiple conditioning images for multi-scene storyboard.
            duration_seconds: Desired video length (default: config value, e.g. 10.0s).
            fps: Desired chunk generation framerate (default: config value, e.g. 16).
            target_fps: Desired final output framerate after high-FPS GPU interpolation (e.g. 60).
            interpolate_factor: Integer frame multiplier (e.g. 2, 4) if not using target_fps.
            interpolate_method: Interpolation mode ("auto", "raft" for optical flow, "blend" for GPU tensor).
            motion_bucket_id: Motion intensity index (1-255).
            num_inference_steps: Denoising steps per chunk.
            noise_aug_strength: Conditioning noise level (default 0.02).
            seed: RNG seed for reproducible generation.
            output_path: Destination path for output MP4 file.
            progress_callback: Optional callback receiving progress dict.

        Returns:
            GenerationResult containing video path, frames, and metadata.
        """
        start_time = time.time()

        # 1. Resolve prompt
        prompt_text = load_prompt(prompt=prompt, prompt_file=prompt_file)

        # 2. Determine motion bucket
        base_motion = motion_bucket_id if motion_bucket_id is not None else self.config.generation.motion_bucket_id
        final_motion = infer_motion_bucket_id(prompt_text, base_bucket=base_motion)

        # 3. Resolve generation settings
        target_duration = duration_seconds or self.config.generation.duration_seconds
        base_fps = fps or self.config.generation.fps
        steps = num_inference_steps or self.config.generation.num_inference_steps
        noise_aug = noise_aug_strength if noise_aug_strength is not None else self.config.generation.noise_aug_strength
        chosen_seed = seed if seed is not None else self.config.generation.seed

        resolution = (self.config.generation.width, self.config.generation.height)

        # 4. Resolve input image(s)
        if images is not None and len(images) > 0:
            loaded_images = load_images(images, target_size=resolution)
        elif image is not None:
            loaded_images = load_images([image], target_size=resolution)
        else:
            # Fallback to default internet image
            loaded_images = load_images([None], target_size=resolution)

        # 5. Plan temporal chunks
        plan = calculate_chunks_plan(
            duration_seconds=target_duration,
            fps=base_fps,
            num_frames_per_chunk=self.config.generation.num_frames_per_chunk,
            transition_frames=self.config.generation.transition_frames,
        )
        num_chunks = plan["num_chunks"]

        # Distribute storyboard images if multiple are provided
        storyboard_images = distribute_images_to_chunks(loaded_images, num_chunks=num_chunks)

        logger.info(
            "Starting generation: duration=%.1fs, chunks=%d, fps=%d, resolution=%dx%d",
            target_duration,
            num_chunks,
            base_fps,
            resolution[0],
            resolution[1],
        )

        generated_chunks: List[List[Image.Image]] = []

        # 6. Autoregressive chunk generation loop
        for chunk_idx in range(num_chunks):
            chunk_start = time.time()

            # Determine conditioning image:
            if chunk_idx == 0:
                cond_image = storyboard_images[0]
            elif len(loaded_images) > 1 and storyboard_images[chunk_idx] != storyboard_images[chunk_idx - 1]:
                # Switch to next storyboard keyframe image
                cond_image = storyboard_images[chunk_idx]
            else:
                # Autoregressive mode: condition on last frame of previous chunk!
                cond_image = generated_chunks[-1][-1]

            def chunk_step_reporter(step: int, total_steps: int, msg: str):
                if progress_callback:
                    overall_pct = ((chunk_idx + (step / max(1, total_steps))) / num_chunks) * 100
                    progress_callback({
                        "chunk_index": chunk_idx,
                        "total_chunks": num_chunks,
                        "chunk_step": step,
                        "chunk_total_steps": total_steps,
                        "overall_percent": round(overall_pct, 1),
                        "message": f"Chunk {chunk_idx + 1}/{num_chunks}: {msg}",
                    })

            chunk_seed = (chosen_seed + chunk_idx * 1337) if chosen_seed is not None else None

            # Generate chunk frames
            chunk_frames = self.pipeline.generate_chunk(
                image=cond_image,
                prompt=prompt_text,
                num_frames=self.config.generation.num_frames_per_chunk,
                fps=base_fps,
                motion_bucket_id=final_motion,
                noise_aug_strength=noise_aug,
                num_inference_steps=steps,
                min_guidance_scale=self.config.generation.min_guidance_scale,
                max_guidance_scale=self.config.generation.max_guidance_scale,
                seed=chunk_seed,
                callback=chunk_step_reporter,
            )

            generated_chunks.append(chunk_frames)
            chunk_time = time.time() - chunk_start
            logger.info("Chunk %d/%d completed in %.2fs", chunk_idx + 1, num_chunks, chunk_time)

        # 7. Stitch chunks together with cross-dissolve transitions
        stitched_frames = chain_chunks(
            generated_chunks,
            overlap_frames=self.config.generation.transition_frames,
        )

        # 8. High-FPS frame interpolation
        eff_target_fps = target_fps if target_fps is not None else self.config.generation.target_fps
        eff_factor = interpolate_factor if interpolate_factor is not None else self.config.generation.interpolate_factor
        eff_method = interpolate_method or getattr(self.config.generation, "interpolate_method", "auto")
        interp_device = "cuda" if torch.cuda.is_available() else self.pipeline.device

        final_fps = base_fps
        interpolated = False

        if eff_target_fps is not None and eff_target_fps > base_fps:
            logger.info("Interpolating frames from %d fps to high %d fps using method=%s on %s...", base_fps, eff_target_fps, eff_method, interp_device)
            if progress_callback:
                progress_callback({
                    "chunk_index": num_chunks,
                    "total_chunks": num_chunks,
                    "overall_percent": 95.0,
                    "message": f"Interpolating to {eff_target_fps} FPS using GPU ({eff_method})...",
                })
            stitched_frames = interpolate_frames(
                stitched_frames,
                target_fps=eff_target_fps,
                source_fps=base_fps,
                method=eff_method,
                device=interp_device,
            )
            final_fps = eff_target_fps
            interpolated = True
        elif eff_factor is not None and eff_factor > 1:
            logger.info("Interpolating frames with factor %dx from %d fps to %d fps on %s...", eff_factor, base_fps, base_fps * eff_factor, interp_device)
            if progress_callback:
                progress_callback({
                    "chunk_index": num_chunks,
                    "total_chunks": num_chunks,
                    "overall_percent": 95.0,
                    "message": f"Interpolating {eff_factor}x factor using GPU ({eff_method})...",
                })
            stitched_frames = interpolate_frames(
                stitched_frames,
                factor=eff_factor,
                source_fps=base_fps,
                method=eff_method,
                device=interp_device,
            )
            final_fps = base_fps * eff_factor
            interpolated = True

        # 9. Save to output MP4
        if output_path is not None:
            out_file = Path(output_path).resolve()
        else:
            out_dir = Path(self.config.generation.output_dir).resolve()
            out_dir.mkdir(parents=True, exist_ok=True)
            video_id = f"video_{uuid.uuid4().hex[:10]}"
            out_file = out_dir / f"{video_id}.mp4"

        saved_path = save_video(stitched_frames, out_file, fps=final_fps)
        total_time = time.time() - start_time
        actual_duration = len(stitched_frames) / final_fps

        if progress_callback:
            progress_callback({
                "chunk_index": num_chunks,
                "total_chunks": num_chunks,
                "overall_percent": 100.0,
                "message": "Video generation and encoding complete!",
                "video_path": str(saved_path),
            })

        metadata = {
            "prompt": prompt_text,
            "motion_bucket_id": final_motion,
            "target_duration_seconds": target_duration,
            "actual_duration_seconds": round(actual_duration, 2),
            "fps": final_fps,
            "source_fps": base_fps,
            "interpolated": interpolated,
            "interpolate_method": eff_method if interpolated else None,
            "num_frames": len(stitched_frames),
            "num_chunks": num_chunks,
            "num_inference_steps": steps,
            "seed": chosen_seed,
            "pipeline": self.pipeline.name,
            "device": self.pipeline.device,
            "execution_time_seconds": round(total_time, 2),
        }

        logger.info(
            "Video generation finished: %s (%.2fs, %d frames, took %.1fs)",
            saved_path,
            actual_duration,
            len(stitched_frames),
            total_time,
        )

        return GenerationResult(
            video_path=saved_path,
            frames=stitched_frames,
            duration_seconds=actual_duration,
            fps=final_fps,
            num_frames=len(stitched_frames),
            width=resolution[0],
            height=resolution[1],
            metadata=metadata,
        )
