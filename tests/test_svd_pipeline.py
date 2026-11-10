"""Tests for Stable Video Diffusion pipeline (CUDA only)."""

import pytest
import torch
from PIL import Image
from video_gen.config import VideoConfig
from video_gen.pipelines.svd_pipeline import SVDPipeline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available")
def test_svd_pipeline_initialization():
    cfg = VideoConfig.for_l4_gpu()
    pipeline = SVDPipeline(config=cfg)
    assert pipeline.name == "svd_xt"
    assert pipeline.device.startswith("cuda")
    assert pipeline.is_loaded() is False
