"""Tests for video saving, metadata extraction, and frame interpolation."""

from pathlib import Path
from PIL import Image
from video_gen.core.video_utils import (
    interpolate_frames,
    read_video_metadata,
    save_video,
)


def test_save_video_and_read_metadata(tmp_path):
    frames = [Image.new("RGB", (160, 90), color=(i * 10, i * 15, i * 20)) for i in range(14)]
    out_file = tmp_path / "test_out.mp4"

    saved_path = save_video(frames, out_file, fps=7)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    meta = read_video_metadata(saved_path)
    assert meta["width"] == 160
    assert meta["height"] == 90
    assert meta["fps"] == 7.0
    assert meta["frame_count"] == 14
    assert meta["duration_seconds"] == 2.0


def test_interpolate_frames():
    frames = [Image.new("RGB", (10, 10), color=(0, 0, 0)), Image.new("RGB", (10, 10), color=(100, 100, 100))]
    # 2 frames with factor=2 -> (2 - 1)*2 + 1 = 3 frames
    interpolated = interpolate_frames(frames, factor=2)
    assert len(interpolated) == 3
