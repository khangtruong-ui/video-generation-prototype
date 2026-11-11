"""Pipeline registry and factory loader."""

from typing import Dict, Type
from video_gen.config import VideoConfig
from video_gen.pipelines.base import BaseVideoPipeline
from video_gen.pipelines.mock_pipeline import MockVideoPipeline
from video_gen.pipelines.svd_pipeline import SVDPipeline
from video_gen.pipelines.ltx_pipeline import LTXVideoPipeline
from video_gen.pipelines.wan_pipeline import WanVideoPipeline

PIPELINE_REGISTRY: Dict[str, Type[BaseVideoPipeline]] = {
    "wan": WanVideoPipeline,
    "wan_i2v": WanVideoPipeline,
    "svd_xt": SVDPipeline,
    "ltx_video": LTXVideoPipeline,
    "mock": MockVideoPipeline,
}


def get_pipeline(pipeline_type: str = "wan", config: VideoConfig = None) -> BaseVideoPipeline:
    """Instantiate a video generation pipeline by name."""
    cfg = config or VideoConfig.for_l4_gpu()
    key = pipeline_type.lower()
    if key not in PIPELINE_REGISTRY:
        raise ValueError(
            f"Unknown pipeline_type '{pipeline_type}'. Available options: {list(PIPELINE_REGISTRY.keys())}"
        )

    pipeline_cls = PIPELINE_REGISTRY[key]
    if pipeline_cls is WanVideoPipeline:
        return WanVideoPipeline(model_id=cfg.generation.model_id, config=cfg)
    elif pipeline_cls is SVDPipeline:
        return SVDPipeline(model_id=cfg.generation.model_id, config=cfg)
    elif pipeline_cls is LTXVideoPipeline:
        return LTXVideoPipeline(config=cfg)
    elif pipeline_cls is MockVideoPipeline:
        return MockVideoPipeline(name="mock", device=cfg.device.device)
    else:
        return pipeline_cls()


__all__ = [
    "BaseVideoPipeline",
    "WanVideoPipeline",
    "SVDPipeline",
    "LTXVideoPipeline",
    "MockVideoPipeline",
    "PIPELINE_REGISTRY",
    "get_pipeline",
]
