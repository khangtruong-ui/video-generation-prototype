"""Google Colab & Jupyter notebook utilities for video rendering and downloading."""

import base64
import logging
from pathlib import Path
from typing import Any, Optional, Union
import uuid

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Return True if executing inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _format_bytes(num_bytes: int) -> str:
    """Format file size in human-readable units."""
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def _resolve_video_path(video: Any) -> Path:
    """Extract and validate Path from string, Path, or GenerationResult."""
    if hasattr(video, "video_path"):
        path = Path(video.video_path)
    elif isinstance(video, (str, Path)):
        path = Path(video)
    else:
        raise TypeError(f"Expected str, Path, or GenerationResult, got {type(video).__name__}")

    return path.resolve()


def download_video(video: Union[str, Path, Any]) -> bool:
    """Download video to user disk using Google Colab files API.

    Args:
        video: Video file path (str, Path) or GenerationResult object.

    Returns:
        bool: True if downloaded via Google Colab, False if not running in Colab.
    """
    path = _resolve_video_path(video)
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found at: {path}")

    if is_colab():
        try:
            from google.colab import files
            files.download(str(path))
            return True
        except Exception as exc:
            logger.warning("google.colab.files.download failed: %s", exc)
            return False
    else:
        logger.info("Not running in Google Colab. Video file is stored locally at: %s", path)
        return False


def display_video(
    video: Union[str, Path, Any],
    width: int = 640,
    height: Optional[int] = None,
    autoplay: bool = True,
    loop: bool = True,
    include_download: bool = True,
    title: Optional[str] = None,
) -> Optional[Any]:
    """Display video in Google Colab / Jupyter notebook with HTML5 player and download button.

    Features:
    - HTML5 video player with responsive card styling
    - Instant browser download button (data URI / blob)
    - Google Colab native download API trigger button when running in Colab
    - Automatic extraction of video duration, fps, frame count, resolution if GenerationResult is passed

    Args:
        video: Video file path (str, Path) or GenerationResult instance.
        width: Player width in pixels (default: 640).
        height: Player height in pixels (optional).
        autoplay: Whether video should autoplay (default: True).
        loop: Whether video should loop playback (default: True).
        include_download: Whether to render download buttons (default: True).
        title: Optional custom header title.

    Returns:
        IPython.display.HTML object if IPython is available, else raw HTML string.
    """
    path = _resolve_video_path(video)
    if not path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {path}")

    file_bytes = path.read_bytes()
    b64_str = base64.b64encode(file_bytes).decode("utf-8")
    file_size_str = _format_bytes(len(file_bytes))
    filename = path.name

    # Extract metadata badges if available
    metadata_badges = []
    if hasattr(video, "duration_seconds") and video.duration_seconds:
        metadata_badges.append(f"{video.duration_seconds:.2f}s")
    if hasattr(video, "fps") and video.fps:
        metadata_badges.append(f"{video.fps} fps")
    if hasattr(video, "num_frames") and video.num_frames:
        metadata_badges.append(f"{video.num_frames} frames")
    if hasattr(video, "width") and hasattr(video, "height") and video.width and video.height:
        metadata_badges.append(f"{video.width}×{video.height}")

    meta_html = ""
    if metadata_badges:
        badge_elements = "".join(
            f'<span style="background: rgba(128,128,128,0.15); padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500;">{b}</span>'
            for b in metadata_badges
        )
        meta_html = f'<div style="display: flex; gap: 6px; align-items: center; flex-wrap: wrap;">{badge_elements}</div>'

    header_title = title or f"🎬 {filename}"
    autoplay_attr = "autoplay" if autoplay else ""
    loop_attr = "loop" if loop else ""
    height_style = f"max-height: {height}px;" if height else ""

    # Unique ID for JS bindings
    widget_id = f"vid_{uuid.uuid4().hex[:8]}"

    # Setup Colab Files API callback if running in Colab
    colab_button_html = ""
    if is_colab() and include_download:
        try:
            from google.colab import output

            callback_name = f"colab_dl_{widget_id}"

            def _on_colab_download():
                from google.colab import files
                files.download(str(path))

            output.register_callback(callback_name, _on_colab_download)
            colab_button_html = f"""
            <button onclick="google.colab.kernel.invokeFunction('{callback_name}', [], {{}})"
                    style="background: #34a853; color: white; border: none; padding: 7px 14px; border-radius: 6px; font-size: 12px; font-weight: 500; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.15); transition: background 0.2s;"
                    onmouseover="this.style.background='#2d9249'"
                    onmouseout="this.style.background='#34a853'"
                    title="Download using Google Colab files.download API">
                💾 Colab Download
            </button>
            """
        except Exception as exc:
            logger.debug("Could not register Colab callback: %s", exc)

    download_section_html = ""
    if include_download:
        download_section_html = f"""
        <div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; justify-content: space-between;">
            <div style="display: flex; flex-wrap: wrap; gap: 8px; align-items: center;">
                <a href="data:video/mp4;base64,{b64_str}" download="{filename}"
                   style="background: #1a73e8; color: #ffffff !important; text-decoration: none; padding: 7px 14px; border-radius: 6px; font-size: 12px; font-weight: 500; display: inline-flex; align-items: center; gap: 6px; box-shadow: 0 1px 3px rgba(26,115,232,0.3); transition: background 0.2s;"
                   onmouseover="this.style.background='#1557b0'"
                   onmouseout="this.style.background='#1a73e8'"
                   title="Direct download to browser">
                    📥 Download Video ({file_size_str})
                </a>
                {colab_button_html}
            </div>
            <span style="font-size: 11px; opacity: 0.7; font-family: monospace;">{filename} ({file_size_str})</span>
        </div>
        """

    html_code = f"""
    <div id="{widget_id}" style="max-width: {width}px; margin: 16px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <div style="background: rgba(128,128,128,0.06); border: 1px solid rgba(128,128,128,0.22); border-radius: 10px; padding: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 6px;">
                <span style="font-weight: 600; font-size: 14px;">{header_title}</span>
                {meta_html}
            </div>
            <video width="100%" controls {autoplay_attr} {loop_attr} style="border-radius: 6px; display: block; background: #000; {height_style}">
                <source src="data:video/mp4;base64,{b64_str}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            {download_section_html}
        </div>
    </div>
    """

    try:
        from IPython.display import HTML, display
        rendered = HTML(html_code)
        display(rendered)
        return rendered
    except ImportError:
        logger.warning("IPython is not installed; returning raw HTML string.")
        return html_code
