"""Pytest fixtures for video generation tests."""

import pytest
from PIL import Image
from pathlib import Path
import tempfile
import shutil

from video_gen.config import VideoConfig
from video_gen.generator import LongVideoGenerator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_pil_image():
    """Create a sample 512x288 RGB test image."""
    return Image.new("RGB", (512, 288), color=(100, 150, 200))


@pytest.fixture
def mock_generator(temp_dir):
    """Instantiate a LongVideoGenerator with MockVideoPipeline."""
    cfg = VideoConfig.for_mock(duration_seconds=2.0, fps=5, width=256, height=144, output_dir=temp_dir)
    return LongVideoGenerator(config=cfg)
