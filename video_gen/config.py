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
    torch_dtype: str = "float16" if torch.cuda.is_available() else "float32"
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_model_cpu_offload: bool = False
    decode_chunk_size: int = 4

    def get_torch_dtype(self) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(self.torch_dtype, torch.float16)


@dataclass
class GenerationConfig:
    """Parameters controlling video generation and chaining."""
    # Duration and temporal structure
    duration_seconds: float = 10.0
    fps: int = 7
    num_frames_per_chunk: int = 25
    transition_frames: int = 2
    interpolate_factor: int = 1  # 1 = native fps, 2 = 2x frame interpolation

    # Frame geometry
    width: int = 1024
    height: int = 576

    # Model inference settings
    model_id: str = "stabilityai/stable-video-diffusion-img2vid-xt"
    pipeline_type: Literal["svd_xt", "mock"] = "svd_xt"
    num_inference_steps: int = 25
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 3.0
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
        fps: int = 7,
        width: int = 1024,
        height: int = 576,
        num_inference_steps: int = 25,
        motion_bucket_id: int = 127,
        seed: Optional[int] = None,
        output_dir: Union_Path = Path("./outputs"),
    ) -> "VideoConfig":
        """Pre-configured optimal settings for NVIDIA L4 GPU (24GB VRAM).

        Key L4 optimizations:
        - fp16 precision for tensor cores
        - VAE slicing & tiling enabled to minimize peak VRAM
        - decode_chunk_size=8 for low memory VAE decoding
        - Fits comfortably in ~6-8GB VRAM with headroom for batching and long chaining
        """
        device = DeviceConfig(
            device="cuda",
            torch_dtype="float16",
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_model_cpu_offload=False,
            decode_chunk_size=4,
        )
        generation = GenerationConfig(
            duration_seconds=duration_seconds,
            fps=fps,
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
    def for_mock(
        cls,
        duration_seconds: float = 4.0,
        fps: int = 7,
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


