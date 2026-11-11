"""Configuration classes and factory presets for video generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Literal, Union
import torch

Union_Path = Union[str, Path]


@dataclass
class DeviceConfig:
    """Hardware and execution device configuration."""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype: str = (
        "bfloat16"
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else ("float16" if torch.cuda.is_available() else "float32")
    )
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_model_cpu_offload: bool = True
    decode_chunk_size: int = 4

    def get_torch_dtype(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping.get(self.torch_dtype, torch.bfloat16 if torch.cuda.is_available() else torch.float32)


@dataclass
class GenerationConfig:
    """Parameters controlling video generation, chaining, and frame interpolation."""
    # Duration and temporal structure
    duration_seconds: float = 10.0
    fps: int = 16  # Native Wan framerate
    target_fps: Optional[int] = 60  # High-FPS GPU frame interpolation target
    interpolate_factor: int = 1  # Integer multiplier (1 = native fps, 2 = 2x, etc.)
    interpolate_method: str = "auto"  # "auto", "raft" (deep optical flow), or "blend"
    num_frames_per_chunk: int = 81  # Wan native 81 frames per chunk
    transition_frames: int = 4  # Smooth cross-dissolve overlap between chunks

    # Frame geometry (Wan2.1 480P widescreen 16:9)
    width: int = 832
    height: int = 480

    # Model inference settings
    model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    pipeline_type: Literal["wan", "wan_i2v", "svd_xt", "ltx_video", "mock"] = "wan"
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 5.0
    motion_bucket_id: int = 127
    noise_aug_strength: float = 0.02
    seed: Optional[int] = None

    # Output storage
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    output_format: str = "mp4"

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.width, self.height)


@dataclass
class VideoConfig:
    """Master configuration wrapping device and generation settings."""
    device: DeviceConfig = field(default_factory=DeviceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @classmethod
    def for_l4_gpu(
        cls,
        duration_seconds: float = 10.0,
        fps: int = 16,
        target_fps: Optional[int] = 60,
        interpolate_method: str = "auto",
        width: int = 832,
        height: int = 480,
        num_inference_steps: int = 30,
        guidance_scale: float = 5.0,
        model_id: str = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        seed: Optional[int] = None,
        output_dir: Union_Path = Path("./outputs"),
    ) -> "VideoConfig":
        """Pre-configured optimal settings for NVIDIA L4 GPU (24GB VRAM) using Wan2.1 14B.

        Key L4 optimizations:
        - Default model: Wan2.1-I2V-14B-480P-Diffusers (14B parameter foundation model)
        - bfloat16 precision native to Ada Lovelace / NVIDIA L4 tensor cores
        - Model CPU offload enabled so 14B model fits and runs with full capacity in 24GB VRAM
        - 3D VAE slicing & tiling enabled to minimize memory usage
        - High-fps GPU optical flow frame interpolation (16 fps -> 60 fps)
        """
        device = DeviceConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16",
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_model_cpu_offload=True,
            decode_chunk_size=4,
        )
        generation = GenerationConfig(
            duration_seconds=duration_seconds,
            fps=fps,
            target_fps=target_fps,
            interpolate_factor=1,
            interpolate_method=interpolate_method,
            num_frames_per_chunk=81,
            transition_frames=4,
            width=width,
            height=height,
            model_id=model_id,
            pipeline_type="wan",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            min_guidance_scale=1.0,
            max_guidance_scale=guidance_scale,
            motion_bucket_id=127,
            noise_aug_strength=0.02,
            seed=seed,
            output_dir=Path(output_dir),
        )
        return cls(device=device, generation=generation)

    @classmethod
    def for_svd_xt(
        cls,
        duration_seconds: float = 10.0,
        fps: int = 7,
        target_fps: Optional[int] = None,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 25,
        motion_bucket_id: int = 127,
        seed: Optional[int] = None,
        output_dir: Union_Path = Path("./outputs"),
    ) -> "VideoConfig":
        """Preset for Stable Video Diffusion (SVD-XT) pipeline."""
        device = DeviceConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="float16",
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_model_cpu_offload=False,
            decode_chunk_size=4,
        )
        generation = GenerationConfig(
            duration_seconds=duration_seconds,
            fps=fps,
            target_fps=target_fps,
            num_frames_per_chunk=25,
            transition_frames=2,
            width=width,
            height=height,
            model_id="stabilityai/stable-video-diffusion-img2vid-xt",
            pipeline_type="svd_xt",
            num_inference_steps=num_inference_steps,
            min_guidance_scale=1.0,
            max_guidance_scale=3.0,
            motion_bucket_id=motion_bucket_id,
            noise_aug_strength=0.02,
            seed=seed,
            output_dir=Path(output_dir),
        )
        return cls(device=device, generation=generation)

    @classmethod
    def for_ltx(
        cls,
        duration_seconds: float = 10.0,
        fps: int = 25,
        target_fps: Optional[int] = None,
        width: int = 704,
        height: int = 480,
        num_inference_steps: int = 25,
        seed: Optional[int] = None,
        output_dir: Union_Path = Path("./outputs"),
    ) -> "VideoConfig":
        """Preset for LTX-Video DiT pipeline."""
        device = DeviceConfig(
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype="bfloat16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "float16",
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_model_cpu_offload=True,
            decode_chunk_size=4,
        )
        generation = GenerationConfig(
            duration_seconds=duration_seconds,
            fps=fps,
            target_fps=target_fps,
            num_frames_per_chunk=25,
            transition_frames=2,
            width=width,
            height=height,
            model_id="Lightricks/LTX-Video",
            pipeline_type="ltx_video",
            num_inference_steps=num_inference_steps,
            min_guidance_scale=1.0,
            max_guidance_scale=3.0,
            seed=seed,
            output_dir=Path(output_dir),
        )
        return cls(device=device, generation=generation)

    @classmethod
    def for_mock(
        cls,
        duration_seconds: float = 4.0,
        fps: int = 7,
        target_fps: Optional[int] = None,
        width: int = 512,
        height: int = 288,
        output_dir: Union_Path = Path("./outputs"),
    ) -> "VideoConfig":
        """Fast mock configuration for lightweight unit testing, CI, and CPU debugging."""
        device = DeviceConfig(
            device="cpu",
            torch_dtype="float32",
            enable_vae_slicing=False,
            enable_vae_tiling=False,
            enable_model_cpu_offload=False,
            decode_chunk_size=8,
        )
        generation = GenerationConfig(
            duration_seconds=duration_seconds,
            fps=fps,
            target_fps=target_fps,
            num_frames_per_chunk=10,
            transition_frames=1,
            width=width,
            height=height,
            pipeline_type="mock",
            num_inference_steps=2,
            output_dir=Path(output_dir),
        )
        return cls(device=device, generation=generation)

    @classmethod
    def for_modal(
        cls,
        duration_seconds: float = 10.0,
        gpu_type: str = "L4",
    ) -> "VideoConfig":
        """Configuration preset for running on Modal cloud workers."""
        config = cls.for_l4_gpu(duration_seconds=duration_seconds)
        return config
