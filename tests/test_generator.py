"""Tests for master LongVideoGenerator."""

from PIL import Image
from video_gen.config import VideoConfig
from video_gen.generator import LongVideoGenerator


def test_long_video_generator_mock(tmp_path):
    # Test generating a 4-second video with mock pipeline
    cfg = VideoConfig.for_mock(duration_seconds=4.0, fps=5, width=128, height=72, output_dir=tmp_path)
    generator = LongVideoGenerator(config=cfg)

    input_img = Image.new("RGB", (128, 72), color=(50, 100, 150))
    prompt = "A serene camera pan across a peaceful valley"

    progress_events = []
    def on_progress(p):
        progress_events.append(p)

    out_file = tmp_path / "long_test.mp4"
    result = generator.generate(
        prompt=prompt,
        image=input_img,
        duration_seconds=4.0,
        fps=5,
        output_path=out_file,
        progress_callback=on_progress,
    )

    assert result.video_path.exists()
    assert result.video_path.stat().st_size > 0
    assert result.duration_seconds >= 3.5
    assert len(result.frames) >= 15
    assert result.metadata["num_chunks"] >= 2
    assert len(progress_events) > 0
    assert progress_events[-1]["overall_percent"] == 100.0


def test_long_video_generator_multi_image(tmp_path):
    # Test multi-image storyboard generation
    cfg = VideoConfig.for_mock(duration_seconds=3.0, fps=5, width=128, height=72, output_dir=tmp_path)
    generator = LongVideoGenerator(config=cfg)

    img1 = Image.new("RGB", (128, 72), color=(200, 50, 50))
    img2 = Image.new("RGB", (128, 72), color=(50, 200, 50))

    result = generator.generate(
        prompt="Transition between two distinct landscapes",
        images=[img1, img2],
        duration_seconds=3.0,
        fps=5,
        output_path=tmp_path / "storyboard.mp4",
    )

    assert result.video_path.exists()
    assert len(result.frames) > 0
