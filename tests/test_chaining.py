"""Tests for chunk calculation, cross-dissolve blending, and storyboard distribution."""

from PIL import Image
from video_gen.core.chaining import (
    calculate_chunks_plan,
    chain_chunks,
    cross_dissolve_blend,
    distribute_images_to_chunks,
)


def test_calculate_chunks_plan_short():
    # 3.5s at 7fps = 25 frames -> 1 chunk
    plan = calculate_chunks_plan(duration_seconds=3.5, fps=7, num_frames_per_chunk=25, transition_frames=2)
    assert plan["num_chunks"] == 1
    assert plan["total_stitched_frames"] == 25


def test_calculate_chunks_plan_10s():
    # 10s at 7fps = 70 frames target -> 3 chunks with 2 frames overlap = 25 + 23 + 23 = 71 frames (10.14s)
    plan = calculate_chunks_plan(duration_seconds=10.0, fps=7, num_frames_per_chunk=25, transition_frames=2)
    assert plan["num_chunks"] == 3
    assert plan["total_stitched_frames"] == 71
    assert plan["actual_duration_seconds"] > 10.0


def test_calculate_chunks_plan_long():
    # 60s at 7fps = 420 frames target
    plan = calculate_chunks_plan(duration_seconds=60.0, fps=7, num_frames_per_chunk=25, transition_frames=2)
    assert plan["num_chunks"] > 15
    assert plan["total_stitched_frames"] >= 420


def test_cross_dissolve_blend():
    chunk_a = [Image.new("RGB", (20, 20), color=(100, 100, 100)) for _ in range(5)]
    chunk_b = [Image.new("RGB", (20, 20), color=(200, 200, 200)) for _ in range(5)]

    # 5 frames + 5 frames with 2 overlap = (5 - 2) + 2 blended + (5 - 2) = 3 + 2 + 3 = 8 frames
    blended = cross_dissolve_blend(chunk_a, chunk_b, overlap_frames=2)
    assert len(blended) == 8


def test_chain_chunks():
    c1 = [Image.new("RGB", (10, 10), color=(50, 50, 50)) for _ in range(10)]
    c2 = [Image.new("RGB", (10, 10), color=(100, 100, 100)) for _ in range(10)]
    c3 = [Image.new("RGB", (10, 10), color=(150, 150, 150)) for _ in range(10)]

    # 10 + 9 + 9 = 28 frames with 1 overlap
    stitched = chain_chunks([c1, c2, c3], overlap_frames=1)
    assert len(stitched) == 28


def test_distribute_images_to_chunks():
    img1 = Image.new("RGB", (10, 10), color=(1, 1, 1))
    img2 = Image.new("RGB", (10, 10), color=(2, 2, 2))

    distributed = distribute_images_to_chunks([img1, img2], num_chunks=4)
    assert len(distributed) == 4
    assert distributed[0] == img1
    assert distributed[-1] == img2
