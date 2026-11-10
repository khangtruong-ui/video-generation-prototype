"""Local runner executing long video generation on the local machine/GPU."""

import logging
from typing import Any, Callable, Dict, Optional
from video_gen.config import VideoConfig
from video_gen.generator import GenerationResult, LongVideoGenerator

logger = logging.getLogger(__name__)


class LocalRunner:
    """Manages local in-process video generation tasks on L4 GPU."""

    def __init__(self, config: Optional[VideoConfig] = None):
        self.config = config or VideoConfig.for_l4_gpu()
        self.generator = LongVideoGenerator(config=self.config)

    def run(self, **kwargs: Any) -> GenerationResult:
        """Run generation locally."""
        logger.info("Executing video generation on local backend (%s)...", self.config.device.device)
        return self.generator.generate(**kwargs)
