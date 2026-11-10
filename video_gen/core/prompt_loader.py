"""Prompt ingestion, file loading, and motion dynamics analysis."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = "Cinematic aerial camera gliding smoothly over majestic natural scenery, ultra-detailed, photorealistic 8k, warm sunset lighting."

# Keywords mapped to motion intensity
HIGH_MOTION_KEYWORDS = {
    "fast", "speed", "racing", "storm", "running", "running", "flying", "blast",
    "explosion", "action", "wave", "crashing", "turbulent", "windy", "rushing",
    "cascade", "tornado", "waterfall", "hyperlapse", "timelapse", "rapid"
}

LOW_MOTION_KEYWORDS = {
    "still", "calm", "slow", "serene", "peaceful", "subtle", "gentle", "static",
    "soft", "quiet", "breeze", "floating", "portrait", "meditative", "ambient"
}


def load_prompt(
    prompt: Optional[str] = None,
    prompt_file: Optional[Union[str, Path]] = None,
    default: str = DEFAULT_PROMPT,
) -> str:
    """Load text prompt from direct string or a text file.

    Args:
        prompt: Direct text prompt string.
        prompt_file: Path to a .txt file containing the prompt.
        default: Fallback prompt if both are omitted.

    Returns:
        Cleaned prompt string.
    """
    if prompt_file is not None:
        file_path = Path(prompt_file)
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found at: {file_path}")
        text = file_path.read_text(encoding="utf-8").strip()
        if text:
            logger.info("Loaded prompt from file '%s' (%d chars)", file_path, len(text))
            return text

    if prompt is not None and prompt.strip():
        return prompt.strip()

    logger.info("No prompt provided. Using default cinematic prompt.")
    return default


def parse_storyboard_prompts(text: str) -> List[str]:
    """Parse a multi-scene storyboard script into individual scene prompts.

    Supports formats like:
    - Blank line separated paragraphs
    - "Scene 1: ... \n Scene 2: ..."
    - Numbered lines "1. ... \n 2. ..."
    """
    cleaned = text.strip()
    if not cleaned:
        return [DEFAULT_PROMPT]

    # Check for "Scene \d+:" or numbered lines
    scene_pattern = r"(?:^|\n)\s*(?:Scene\s*\d+:?|\d+[\.\)])\s*"
    splits = [s.strip() for s in re.split(scene_pattern, cleaned, flags=re.IGNORECASE) if s.strip()]

    if len(splits) > 1:
        return splits

    # Check for double newline separated scenes
    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs

    return [cleaned]


def infer_motion_bucket_id(prompt: str, base_bucket: int = 127) -> int:
    """Analyze prompt semantics to recommend a motion_bucket_id between 1 and 255.

    - High motion prompts: 180 - 220
    - Subtle/slow prompts: 40 - 80
    - Neutral/balanced: base_bucket (typically 127)
    """
    words = set(re.findall(r"\b\w+\b", prompt.lower()))

    high_matches = words.intersection(HIGH_MOTION_KEYWORDS)
    low_matches = words.intersection(LOW_MOTION_KEYWORDS)

    if len(high_matches) > len(low_matches):
        chosen = min(220, base_bucket + 40 + len(high_matches) * 10)
        logger.debug("Detected high-motion keywords %s -> motion_bucket_id=%d", high_matches, chosen)
        return chosen
    elif len(low_matches) > len(high_matches):
        chosen = max(40, base_bucket - 50 - len(low_matches) * 10)
        logger.debug("Detected low-motion keywords %s -> motion_bucket_id=%d", low_matches, chosen)
        return chosen

    return base_bucket
