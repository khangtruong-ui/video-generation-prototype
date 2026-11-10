"""Tests for Colab notebook utilities and video display."""

from pathlib import Path
import pytest
from PIL import Image

from video_gen.colab import display_video, download_video, is_colab, _format_bytes, _resolve_video_path
from video_gen.generator import GenerationResult


def test_format_bytes():
    assert _format_bytes(500) == "500.0 B"
    assert _format_bytes(1024) == "1.0 KB"
    assert _format_bytes(1024 * 1024 * 2.5) == "2.5 MB"


def test_resolve_video_path(tmp_path):
    mp4_file = tmp_path / "sample.mp4"
    mp4_file.touch()

    # From string
    assert _resolve_video_path(str(mp4_file)) == mp4_file.resolve()
    # From Path
    assert _resolve_video_path(mp4_file) == mp4_file.resolve()

    # From GenerationResult
    result = GenerationResult(
        video_path=mp4_file,
        frames=[],
        duration_seconds=10.0,
        fps=7,
        num_frames=71,
        width=1024,
        height=576,
        metadata={},
    )
    assert _resolve_video_path(result) == mp4_file.resolve()

    # Invalid type
    with pytest.raises(TypeError):
        _resolve_video_path(12345)


def test_display_video_file_not_found():
    with pytest.raises(FileNotFoundError):
        display_video("/non/existent/path/video.mp4")


def test_display_video_with_real_video(tmp_path):
    # Create dummy mp4 file
    mp4_file = tmp_path / "test_render.mp4"
    mp4_file.write_bytes(b"dummy_mp4_bytes_for_testing")

    result = GenerationResult(
        video_path=mp4_file,
        frames=[Image.new("RGB", (64, 64))],
        duration_seconds=10.0,
        fps=7,
        num_frames=71,
        width=1024,
        height=576,
        metadata={"prompt": "flying drone"},
    )

    rendered = display_video(result, width=480, include_download=True)
    html_data = getattr(rendered, "data", str(rendered))

    # Verify key elements
    assert "video/mp4;base64," in html_data
    assert "Download Video" in html_data
    assert "10.00s" in html_data
    assert "7 fps" in html_data
    assert "1024×576" in html_data

    # Test GenerationResult.display() wrapper
    res_disp = result.display(width=320)
    res_html = getattr(res_disp, "data", str(res_disp))
    assert "Download Video" in res_html


def test_download_video_non_existent():
    with pytest.raises(FileNotFoundError):
        download_video("/non/existent/video.mp4")


def test_download_video_local(tmp_path):
    mp4_file = tmp_path / "test.mp4"
    mp4_file.write_bytes(b"test")

    # In standard pytest environment (or Colab), verify it doesn't crash
    val = download_video(mp4_file)
    assert isinstance(val, bool)


def test_is_colab_boolean():
    assert isinstance(is_colab(), bool)
