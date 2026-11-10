"""Tests for image loading, base64 decoding, resizing, and fallbacks."""

import base64
import io
from pathlib import Path
from PIL import Image
from video_gen.core.image_loader import (
    create_procedural_fallback_image,
    load_image,
    load_images,
    resize_and_crop,
)


def test_create_procedural_fallback_image():
    img = create_procedural_fallback_image(size=(320, 180))
    assert isinstance(img, Image.Image)
    assert img.size == (320, 180)
    assert img.mode == "RGB"


def test_resize_and_crop():
    # Square image resized to 16:9
    orig = Image.new("RGB", (400, 400), color=(255, 0, 0))
    resized = resize_and_crop(orig, target_size=(160, 90))
    assert resized.size == (160, 90)


def test_load_image_from_pil():
    orig = Image.new("RGB", (100, 100), color=(0, 255, 0))
    loaded = load_image(orig, target_size=(50, 50))
    assert loaded.size == (50, 50)


def test_load_image_from_file(tmp_path):
    img_path = tmp_path / "test.png"
    Image.new("RGB", (200, 200), color=(10, 20, 30)).save(img_path)

    loaded = load_image(img_path, target_size=(100, 100))
    assert loaded.size == (100, 100)


def test_load_image_from_base64():
    img = Image.new("RGB", (80, 80), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64_str}"

    loaded = load_image(data_uri, target_size=(40, 40))
    assert loaded.size == (40, 40)


def test_load_images_multiple():
    img1 = Image.new("RGB", (100, 100), color=(1, 2, 3))
    img2 = Image.new("RGB", (100, 100), color=(4, 5, 6))

    loaded_list = load_images([img1, img2], target_size=(60, 60))
    assert len(loaded_list) == 2
    assert loaded_list[0].size == (60, 60)
    assert loaded_list[1].size == (60, 60)
