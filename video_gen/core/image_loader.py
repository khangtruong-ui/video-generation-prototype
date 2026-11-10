"""Robust image loading, validation, and preprocessing."""

import base64
import io
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw
import requests

logger = logging.getLogger(__name__)

# Curated high-resolution internet default images (nature & cinematic landscapes)
DEFAULT_INTERNET_IMAGE_URL = (
    "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=1024&q=80"
)

ImageInput = Union[str, Path, bytes, Image.Image, None]


def create_procedural_fallback_image(size: Tuple[int, int] = (1024, 576)) -> Image.Image:
    """Generate a high-quality procedural gradient image if internet is unreachable."""
    width, height = size
    # Create smooth sunrise/sunset gradient
    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for y in range(height):
        ratio = y / height
        # Sky: twilight blue to amber sunset
        r = int(20 + 210 * ratio)
        g = int(30 + 120 * ratio)
        b = int(70 + 60 * (1 - ratio))
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Add subtle sun circle
    sun_x, sun_y = width // 2, int(height * 0.55)
    sun_radius = int(min(width, height) * 0.12)
    draw.ellipse(
        [
            (sun_x - sun_radius, sun_y - sun_radius),
            (sun_x + sun_radius, sun_y + sun_radius),
        ],
        fill=(255, 230, 160),
    )
    return image


def resize_and_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    """Resize and center-crop image to target dimensions while maintaining aspect ratio."""
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    if (orig_w, orig_h) == (target_w, target_h):
        return image

    orig_aspect = orig_w / orig_h
    target_aspect = target_w / target_h

    if orig_aspect > target_aspect:
        # Source is wider: scale by height, crop sides
        new_h = target_h
        new_w = int(target_h * orig_aspect)
    else:
        # Source is taller: scale by width, crop top/bottom
        new_w = target_w
        new_h = int(target_w / orig_aspect)

    resized = image.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    return resized.crop((left, top, right, bottom))


def load_image_from_url(url: str, timeout: int = 15) -> Image.Image:
    """Download image from HTTP/HTTPS URL with proper User-Agent headers."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


def load_image_from_base64(data_str: str) -> Image.Image:
    """Decode base64 encoded image string or data URI."""
    if "," in data_str:
        data_str = data_str.split(",", 1)[1]
    image_bytes = base64.b64decode(data_str)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_image(
    source: ImageInput = None,
    target_size: Optional[Tuple[int, int]] = (1024, 576),
) -> Image.Image:
    """Load an image from URL, local file, base64, bytes, or fallback.

    Args:
        source: Image source (URL, path, base64 data URI, bytes, PIL Image, or None).
        target_size: Optional (width, height) to resize and center-crop to.

    Returns:
        PIL.Image.Image in RGB mode.
    """
    image: Optional[Image.Image] = None

    if source is None or source == "":
        logger.info("No image specified. Fetching default internet image: %s", DEFAULT_INTERNET_IMAGE_URL)
        try:
            image = load_image_from_url(DEFAULT_INTERNET_IMAGE_URL)
        except Exception as exc:
            logger.warning("Failed to fetch default internet image: %s. Using procedural fallback.", exc)
            target = target_size or (1024, 576)
            image = create_procedural_fallback_image(target)

    elif isinstance(source, Image.Image):
        image = source.convert("RGB")

    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source)).convert("RGB")

    elif isinstance(source, (str, Path)):
        str_src = str(source).strip()
        if str_src.startswith("http://") or str_src.startswith("https://"):
            try:
                image = load_image_from_url(str_src)
            except Exception as exc:
                logger.error("Failed to download image from URL %s: %s", str_src, exc)
                raise ValueError(f"Could not load image from URL '{str_src}': {exc}") from exc
        elif str_src.startswith("data:image/") or ";base64," in str_src:
            image = load_image_from_base64(str_src)
        else:
            path = Path(str_src)
            if path.exists() and path.is_file():
                image = Image.open(path).convert("RGB")
            else:
                # Check if it could be raw base64 string
                try:
                    image = load_image_from_base64(str_src)
                except Exception:
                    raise FileNotFoundError(f"Image file or valid source not found: {str_src}")

    if image is None:
        target = target_size or (1024, 576)
        image = create_procedural_fallback_image(target)

    if target_size is not None:
        image = resize_and_crop(image, target_size)

    return image


def load_images(
    sources: Union[List[ImageInput], ImageInput],
    target_size: Optional[Tuple[int, int]] = (1024, 576),
) -> List[Image.Image]:
    """Load a list of images or a single image source into a list of PIL Images."""
    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    if len(sources) == 0:
        sources = [None]

    loaded = [load_image(s, target_size=target_size) for s in sources]
    return loaded
