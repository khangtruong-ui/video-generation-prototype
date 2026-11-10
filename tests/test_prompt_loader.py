"""Tests for prompt loading, script parsing, and motion analysis."""

from pathlib import Path
from video_gen.core.prompt_loader import (
    DEFAULT_PROMPT,
    infer_motion_bucket_id,
    load_prompt,
    parse_storyboard_prompts,
)


def test_load_prompt_direct():
    text = "A serene sunset over a calm lake"
    res = load_prompt(prompt=text)
    assert res == text


def test_load_prompt_default():
    res = load_prompt()
    assert res == DEFAULT_PROMPT


def test_load_prompt_from_file(tmp_path):
    prompt_file = tmp_path / "script.txt"
    prompt_file.write_text("Cinematic hyperlapse through neon cyber city streets.")

    res = load_prompt(prompt_file=prompt_file)
    assert res == "Cinematic hyperlapse through neon cyber city streets."


def test_parse_storyboard_prompts():
    script = """
    Scene 1: Drone shot flying over snow-covered mountains.
    Scene 2: Transition into a lush pine forest in morning mist.
    Scene 3: Reveal a crystal-clear mountain lake reflecting the sunrise.
    """
    scenes = parse_storyboard_prompts(script)
    assert len(scenes) == 3
    assert "mountains" in scenes[0]
    assert "forest" in scenes[1]
    assert "lake" in scenes[2]


def test_infer_motion_bucket_id():
    high_motion_prompt = "A high speed racing car rushing through an explosive storm"
    low_motion_prompt = "A peaceful still portrait of a calm serene person sleeping softly"
    neutral_prompt = "An artistic painting of a castle on a hill"

    high_bucket = infer_motion_bucket_id(high_motion_prompt, base_bucket=127)
    low_bucket = infer_motion_bucket_id(low_motion_prompt, base_bucket=127)
    neutral_bucket = infer_motion_bucket_id(neutral_prompt, base_bucket=127)

    assert high_bucket > 127
    assert low_bucket < 127
    assert neutral_bucket == 127
