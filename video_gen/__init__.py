"""Long Video Generation Prototype.

Extensible long video generation package using Stable Video Diffusion XT,
autoregressive chunk chaining, and FastAPI service endpoints.
"""

from video_gen.config import DeviceConfig, GenerationConfig, VideoConfig
from video_gen.generator import GenerationResult, LongVideoGenerator
from video_gen.core.interpolator import GPUFrameInterpolator
from video_gen.core.video_utils import interpolate_frames
from video_gen.api.app import app, create_app
from video_gen.colab import display_video, download_video, is_colab

__version__ = "0.2.0"

__all__ = [
    "VideoConfig",
    "DeviceConfig",
    "GenerationConfig",
    "LongVideoGenerator",
    "GenerationResult",
    "GPUFrameInterpolator",
    "interpolate_frames",
    "app",
    "create_app",
    "display_video",
    "download_video",
    "is_colab",
    "__version__",
]
