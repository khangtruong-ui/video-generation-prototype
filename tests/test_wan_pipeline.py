"""Tests for Wan2.1 Image-to-Video foundation pipeline."""

from unittest.mock import MagicMock
from PIL import Image
import numpy as np
import pytest
import torch

from video_gen.config import VideoConfig
from video_gen.pipelines import get_pipeline, PIPELINE_REGISTRY
from video_gen.pipelines.wan_pipeline import WanVideoPipeline


def test_wan_pipeline_registry():
    assert "wan" in PIPELINE_REGISTRY
    assert "wan_i2v" in PIPELINE_REGISTRY
    pipe = get_pipeline("wan")
    assert isinstance(pipe, WanVideoPipeline)
    assert pipe.name == "wan"
    assert "Wan2.1-I2V-14B-480P" in pipe.model_id


def test_wan_pipeline_initialization():
    cfg = VideoConfig.for_l4_gpu()
    pipeline = WanVideoPipeline(config=cfg)
    assert pipeline.name == "wan"
    assert pipeline.is_loaded() is False
    assert pipeline.device in ("cuda", "cpu")


def test_wan_pipeline_mocked_generate_chunk():
    cfg = VideoConfig.for_mock(width=160, height=96)
    pipeline = WanVideoPipeline(config=cfg)

    # Mock the inner diffusers pipeline output
    mock_frames = [np.zeros((96, 160, 3), dtype=np.uint8) for _ in range(17)]
    mock_pipe_output = MagicMock()
    mock_pipe_output.frames = [mock_frames]

    mock_pipe = MagicMock()
    mock_pipe.return_value = mock_pipe_output
    pipeline.pipe = mock_pipe

    assert pipeline.is_loaded() is True

    input_img = Image.new("RGB", (160, 96), color=(20, 40, 60))
    progress_calls = []

    def cb(step, total, msg):
        progress_calls.append((step, total, msg))

    frames = pipeline.generate_chunk(
        image=input_img,
        prompt="A peaceful cinematic sunrise",
        num_frames=17,
        num_inference_steps=5,
        callback=cb,
    )

    assert len(frames) == 17
    assert isinstance(frames[0], Image.Image)
    assert frames[0].size == (160, 96)

    # Test unload
    pipeline.unload()
    assert pipeline.is_loaded() is False


def test_diffusers_wan_prompt_clean_no_name_error():
    import diffusers.pipelines.wan.pipeline_wan_i2v as wan_mod
    # Verify that basic_clean and prompt_clean do not raise NameError: name 'ftfy' is not defined
    cleaned = wan_mod.prompt_clean("A magical sunset with &amp; special characters")
    assert "sunset" in cleaned
