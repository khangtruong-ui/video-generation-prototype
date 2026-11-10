"""Sample script demonstrating 10-second long video generation on NVIDIA L4 GPU."""

import logging
from pathlib import Path
from video_gen.config import VideoConfig
from video_gen.generator import LongVideoGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("🎬 Initializing Long Video Generator for NVIDIA L4 GPU...")
    config = VideoConfig.for_l4_gpu(
        duration_seconds=10.0,
        fps=7,
        num_inference_steps=20,  # 20 steps for fast sample generation
        motion_bucket_id=127,
        seed=42,
    )

    generator = LongVideoGenerator(config=config)

    prompt = (
        "A breathtaking cinematic drone flight over a serene alpine mountain valley, "
        "gliding smoothly over reflective lakes and golden hour sunset lighting, 8k photorealistic."
    )

    # Use internet default landscape image or user-provided image
    image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=1024&q=80"

    print(f"Generating 10-second video with prompt: '{prompt}'...")

    def on_progress(p):
        pct = p.get("overall_percent", 0.0)
        msg = p.get("message", "")
        print(f"[{pct:5.1f}%] {msg}")

    result = generator.generate(
        prompt=prompt,
        image=image_url,
        duration_seconds=10.0,
        fps=7,
        output_path="outputs/sample_10s_demo.mp4",
        progress_callback=on_progress,
    )

    print("\n✅ Generation Complete!")
    print(f"Saved to: {result.video_path}")
    print(f"Duration: {result.duration_seconds:.2f} seconds ({result.num_frames} frames)")
    print(f"Chunks:   {result.metadata['num_chunks']}")
    print(f"Time:     {result.metadata['execution_time_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
