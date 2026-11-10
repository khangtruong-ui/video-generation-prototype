"""Pipeline registry and factory loader."""

from typing import Dict, Type
from video_gen.config import VideoConfig
from video_gen.pipelines.base import BaseVideoPipeline
from video_gen.pipelines.mock_pipeline import MockVideoPipeline
from video_gen.pipelines.svd_pipeline import SVDPipeline

PIPELINE_REGISTRY: Dict[str, Type[BaseVideoPipeline]] = {
    "svd_xt": SVDPipeline,
    "mock": MockVideoPipeline,
}


def get_pipeline(pipeline_type: str = "svd_xt", config: VideoConfig = None) -> BaseVideoPipeline:
    """Instantiate a video generation pipeline by name."""
    cfg = config or VideoConfig.for_l4_gpu()
    key = pipeline_type.lower()
    if key not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline_type '{pipeline_type}'. Available options: {list(PIPELINE_REGISTRY.keys())}"
        )

    pipeline_cls = PIPELINE_REGISTRY[key]
    if pipeline_cls is SVDPipeline:
        return SVDPipeline(model_id=cfg.generation.model_id, config=cfg)
    elif pipeline_cls is MockVideoPipeline:
        return MockVideoPipeline(name="mock", device=cfg.device.device)
    else:
        return pipeline_cls()


__all__ = [
    "BaseVideoPipeline",
    "SVDPipeline",
    "MockVideoPipeline",
    "PIPELINE_REGISTRY",
    "get_pipeline",
]
