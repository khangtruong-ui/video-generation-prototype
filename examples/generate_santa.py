"""Generate Santa Claus demo video on NVIDIA L4 GPU."""

import logging
from pathlib import Path
from video_gen.config import VideoConfig
from video_gen.generator import LongVideoGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("🎅 Generating Santa Claus demo video on NVIDIA L4 GPU...")

    # Configure generator for a short 5-second video (2 chained chunks)
    config = VideoConfig.for_l4_gpu(
        duration_seconds=5.0,
        fps=7,
        num_inference_steps=20,
        motion_bucket_id=110,
        seed=1225,  # Christmas seed 🎅
    )
    generator = LongVideoGenerator(config=config)

    santa_image_path = Path("outputs/santa_claus.jpg")
    if not santa_image_path.exists():
        raise FileNotFoundError(f"Santa Claus image not found at {santa_image_path}")

    prompt = (
        "Santa Claus smiling warmly and waving merrily in a magical Christmas winter wonderland, "
        "gentle falling snowflakes, festive glowing lights in background, cinematic 8k photorealistic"
    )

    def on_progress(p):
        pct = p.get("overall_percent", 0.0)
        msg = p.get("message", "")
        print(f"[{pct:5.1f}%] {msg}")

    result = generator.generate(
        prompt=prompt,
        image=str(santa_image_path),
        duration_seconds=5.0,
        fps=7,
        output_path="outputs/santa_claus_demo.mp4",
        progress_callback=on_progress,
    )

    print("\n🎅 Santa Claus Demo Video Generation Complete!")
    print(f"Saved to: {result.video_path}")
    print(f"Duration: {result.duration_seconds:.2f}s ({result.num_frames} frames @ {result.fps} fps)")
    print(f"File size: {result.video_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
