"""Tests for MockVideoPipeline."""

from PIL import Image
from video_gen.pipelines.mock_pipeline import MockVideoPipeline


def test_mock_pipeline_generation():
    pipeline = MockVideoPipeline(name="mock", device="cpu")
    pipeline.load()
    assert pipeline.is_loaded() is True

    test_img = Image.new("RGB", (128, 72), color=(80, 120, 160))

    progress_reports = []
    def callback(step, total, msg):
        progress_reports.append((step, total, msg))

    frames = pipeline.generate_chunk(
        image=test_img,
        num_frames=8,
        fps=5,
        motion_bucket_id=127,
        callback=callback,
    )

    assert len(frames) == 8
    assert frames[0].size == (128, 72)
    assert len(progress_reports) == 8
    assert progress_reports[-1][0] == 8

    pipeline.unload()
    assert pipeline.is_loaded() is False
