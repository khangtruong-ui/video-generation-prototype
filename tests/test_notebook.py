"""Test notebook execution end-to-end."""

import json
from pathlib import Path
import nbformat
import pytest


def test_notebook_structure_and_validity():
    nb_path = Path("/content/video-generation-prototype/examples/video_generation_demo.ipynb")
    assert nb_path.exists()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    assert len(nb.cells) >= 8

    # Verify key code blocks exist
    code_sources = [c.source for c in nb.cells if c.cell_type == "code"]
    combined = "\n".join(code_sources)

    assert "LongVideoGenerator" in combined
    assert "VideoConfig" in combined
    assert "display_video" in combined
    assert "download_video" in combined
    assert "image_url" in combined
    assert "prompt" in combined
    assert "duration_seconds" in combined
    assert "fps" in combined


def test_notebook_cell_execution(tmp_path):
    # Execute the core workflow from the notebook using mock pipeline
    import torch
    from video_gen import LongVideoGenerator, VideoConfig, display_video, is_colab
    from video_gen.core.image_loader import load_image

    prompt = "Cinematic aerial camera gliding smoothly over majestic snow-capped mountain peaks"
    # Create test image
    test_img = tmp_path / "cond_image.jpg"
    from PIL import Image
    Image.new("RGB", (256, 144), color=(30, 80, 150)).save(test_img)

    duration_seconds = 4.0
    fps = 7

    config = VideoConfig.for_mock(
        duration_seconds=duration_seconds,
        fps=fps,
        output_dir=tmp_path,
    )
    generator = LongVideoGenerator(config=config)

    progress_events = []
    result = generator.generate(
        prompt=prompt,
        image=str(test_img),
        duration_seconds=duration_seconds,
        fps=fps,
        output_path=tmp_path / "test_out.mp4",
        progress_callback=lambda p: progress_events.append(p),
    )

    assert result.video_path.exists()
    assert result.duration_seconds > 0
    assert len(progress_events) > 0

    # Test display_video helper
    rendered = display_video(
        result,
        width=640,
        autoplay=True,
        loop=True,
        include_download=True,
    )
    html_data = getattr(rendered, "data", str(rendered))
    assert "Download Video" in html_data
    assert "video/mp4;base64," in html_data
