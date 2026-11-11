"""Sample script demonstrating 10-second long video generation on NVIDIA L4 GPU.

Uses default Wan2.1 14B Image-to-Video foundation model with high-FPS GPU optical-flow interpolation.
"""

import logging
from pathlib import Path
from video_gen.config import VideoConfig
from video_gen.generator import LongVideoGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("🎬 Initializing Long Video Generator for NVIDIA L4 GPU...")
    print("Default model: Wan2.1 14B DiT (Wan-AI/Wan2.1-I2V-14B-480P-Diffusers)")
    print("GPU Interpolation: Deep Learning RAFT Optical Flow -> 60 FPS output")

    # Configure generator with Wan2.1 and high-FPS GPU interpolation
    config = VideoConfig.for_l4_gpu(
        duration_seconds=10.0,
        fps=16,
        target_fps=60,  # Interpolate to high 60 FPS
        interpolate_method="auto",
        num_inference_steps=30,
        seed=42,
    )

    generator = LongVideoGenerator(config=config)

    prompt = (
        "A breathtaking cinematic drone flight over a serene alpine mountain valley, "
        "gliding smoothly over reflective lakes and golden hour sunset lighting, 8k photorealistic."
    )

    # Use internet default landscape image or user-provided image
    image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=832&q=80"

    print(f"Generating 10-second video with prompt: '{prompt}'...")

    def on_progress(p):
        pct = p.get("overall_percent", 0.0)
        msg = p.get("message", "")
        print(f"[{pct:5.1f}%] {msg}")

    result = generator.generate(
        prompt=prompt,
        image=image_url,
        duration_seconds=10.0,
        fps=16,
        target_fps=60,
        output_path="outputs/sample_10s_demo.mp4",
        progress_callback=on_progress,
    )

    print("\n✅ Generation Complete!")
    print(f"Saved to:  {result.video_path}")
    print(f"Duration:  {result.duration_seconds:.2f} seconds ({result.num_frames} frames @ {result.fps} fps)")
    print(f"Pipeline:  {result.metadata['pipeline']}")
    print(f"Chunks:    {result.metadata['num_chunks']}")
    print(f"Smooth:    {'High-FPS GPU interpolated' if result.metadata.get('interpolated') else 'Native'}")
    print(f"Exec Time: {result.metadata['execution_time_seconds']:.2f} seconds")


if __name__ == "__main__":
    main()
