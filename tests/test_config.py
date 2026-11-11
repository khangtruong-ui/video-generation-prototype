"""Tests for configuration classes and presets."""

from pathlib import Path
import torch
from video_gen.config import DeviceConfig, GenerationConfig, VideoConfig


def test_device_config_defaults():
    cfg = DeviceConfig()
    assert cfg.torch_dtype in ["bfloat16", "float16", "float32"]
    assert cfg.decode_chunk_size == 4
    assert cfg.enable_vae_slicing is True
    assert cfg.enable_vae_tiling is True
    assert isinstance(cfg.get_torch_dtype(), torch.dtype)


def test_generation_config_defaults():
    cfg = GenerationConfig()
    assert cfg.duration_seconds == 10.0
    assert cfg.fps == 16
    assert cfg.target_fps == 60
    assert cfg.num_frames_per_chunk == 81
    assert cfg.resolution == (832, 480)
    assert cfg.pipeline_type == "wan"
    assert "Wan2.1-I2V-14B-480P" in cfg.model_id


def test_video_config_for_l4_gpu():
    cfg = VideoConfig.for_l4_gpu(duration_seconds=15.0, fps=16)
    assert cfg.device.device == ("cuda" if torch.cuda.is_available() else "cpu")
    assert cfg.device.torch_dtype in ["bfloat16", "float16"]
    assert cfg.device.enable_vae_slicing is True
    assert cfg.device.enable_vae_tiling is True
    assert cfg.device.enable_model_cpu_offload is True
    assert cfg.generation.duration_seconds == 15.0
    assert cfg.generation.fps == 16
    assert cfg.generation.target_fps == 60
    assert cfg.generation.pipeline_type == "wan"
    assert cfg.generation.num_frames_per_chunk == 81
    assert "Wan2.1-I2V-14B-480P" in cfg.generation.model_id


def test_video_config_for_svd_xt():
    cfg = VideoConfig.for_svd_xt(duration_seconds=10.0, fps=7)
    assert cfg.generation.pipeline_type == "svd_xt"
    assert cfg.generation.fps == 7
    assert cfg.generation.num_frames_per_chunk == 25
    assert cfg.generation.resolution == (1024, 576)


def test_video_config_for_ltx():
    cfg = VideoConfig.for_ltx(duration_seconds=10.0, fps=25)
    assert cfg.generation.pipeline_type == "ltx_video"
    assert cfg.generation.fps == 25
    assert cfg.generation.resolution == (704, 480)


def test_video_config_for_mock(tmp_path):
    cfg = VideoConfig.for_mock(duration_seconds=3.0, output_dir=tmp_path)
    assert cfg.device.device == "cpu"
    assert cfg.generation.pipeline_type == "mock"
    assert cfg.generation.duration_seconds == 3.0
