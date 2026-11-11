"""Tests for GPU deep learning optical flow and tensor frame interpolation."""

from pathlib import Path
from PIL import Image
import pytest
import torch

from video_gen.core.interpolator import GPUFrameInterpolator
from video_gen.core.video_utils import interpolate_frames, read_video_metadata, save_video


def test_interpolator_initialization():
    interpolator = GPUFrameInterpolator()
    assert interpolator.device in ("cuda", "cpu")
    assert interpolator.model_name == "raft_small"


def test_warp_flow_identity():
    # An image warped with zero flow should be identical to the original image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img = torch.rand(1, 3, 64, 64, device=device)
    zero_flow = torch.zeros(1, 2, 64, 64, device=device)

    warped = GPUFrameInterpolator.warp_flow(img, zero_flow)
    assert torch.allclose(img, warped, atol=1e-4)


def test_interpolate_pair_blend():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    interpolator = GPUFrameInterpolator(device=device)

    f0 = torch.zeros(1, 3, 32, 32, device=device)
    f1 = torch.ones(1, 3, 32, 32, device=device)

    intermediates = interpolator.interpolate_pair(f0, f1, alphas=[0.25, 0.5, 0.75], method="blend")
    assert len(intermediates) == 3
    assert torch.allclose(intermediates[0], torch.full_like(intermediates[0], 0.25), atol=1e-3)
    assert torch.allclose(intermediates[1], torch.full_like(intermediates[1], 0.50), atol=1e-3)
    assert torch.allclose(intermediates[2], torch.full_like(intermediates[2], 0.75), atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for RAFT optical flow test")
def test_interpolate_pair_raft():
    device = "cuda"
    interpolator = GPUFrameInterpolator(device=device)

    f0 = torch.rand(1, 3, 32, 32, device=device)
    f1 = torch.rand(1, 3, 32, 32, device=device)

    intermediates = interpolator.interpolate_pair(f0, f1, alphas=[0.5], method="raft")
    assert len(intermediates) == 1
    assert intermediates[0].shape == (1, 3, 32, 32)
    assert intermediates[0].min() >= 0.0
    assert intermediates[0].max() <= 1.0


def test_interpolate_sequence_by_factor():
    frames = [
        Image.new("RGB", (64, 64), color=(0, 0, 0)),
        Image.new("RGB", (64, 64), color=(128, 128, 128)),
        Image.new("RGB", (64, 64), color=(255, 255, 255)),
    ]
    interpolator = GPUFrameInterpolator()

    # 3 frames with factor=2 -> (3 - 1)*2 + 1 = 5 frames
    result_frames, final_fps = interpolator.interpolate_sequence(frames, factor=2, source_fps=15, method="blend")
    assert len(result_frames) == 5
    assert final_fps == 30


def test_interpolate_sequence_by_target_fps():
    # 5 frames at 16 fps = 4/16 = 0.25 seconds
    # At target 60 fps: 0.25 * 60 + 1 = 16 frames
    frames = [Image.new("RGB", (64, 64), color=(i * 30, i * 40, i * 50)) for i in range(5)]
    interpolator = GPUFrameInterpolator()

    result_frames, final_fps = interpolator.interpolate_sequence(frames, target_fps=60, source_fps=16, method="blend")
    assert final_fps == 60
    assert len(result_frames) == 16


def test_interpolate_frames_end_to_end(tmp_path):
    frames = [Image.new("RGB", (80, 48), color=(i * 20, 50, 100)) for i in range(4)]
    # Upsample from 16 fps to 30 fps
    interpolated = interpolate_frames(frames, target_fps=30, source_fps=16, method="blend")
    assert len(interpolated) > len(frames)

    # Save and verify metadata
    out_file = tmp_path / "high_fps_test.mp4"
    saved = save_video(interpolated, out_file, fps=30)
    assert saved.exists()

    meta = read_video_metadata(saved)
    assert meta["fps"] == 30.0
    assert meta["frame_count"] == len(interpolated)
