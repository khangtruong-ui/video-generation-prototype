"""Build interactive Colab demo notebooks for the repository."""

from pathlib import Path
import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            """# 🎬 Long Video Generation Demo in Google Colab
### Continuous 10-Second Video Generation with Stable Video Diffusion XT (SVD-XT) & L4 GPU

This notebook demonstrates how to generate high-fidelity, continuous long videos (default **10.0 seconds** at **7 fps**, scaleable to minutes) using the **`video-generation-prototype`** API:
1. **Conditioning Ingestion**: Text prompt describing motion + conditioning image from a web URL, local path, or curated landscape.
2. **Autoregressive Temporal Chaining**: Generates sequential 25-frame chunks with cross-dissolve frame blending to eliminate seam cuts.
3. **NVIDIA L4 Optimization**: fp16 precision, chunked latent VAE decoding (`decode_chunk_size=4`), VAE slicing and tiling.
4. **Colab Video Display & Download API**: In-notebook HTML5 responsive player with instant direct download and Colab Files API integration.
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 1. Environment & GPU Verification
# Verify GPU availability (optimized for NVIDIA L4 24GB VRAM)
import os
import sys
import subprocess
import torch

print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🚀 GPU Detected: {device_name} ({vram_gb:.1f} GB VRAM)")
else:
    print("⚠️ No GPU detected. Running in CPU mode (Mock pipeline will be used).")
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 2. Import Dependencies & video_gen API
import sys
from pathlib import Path

# Ensure video-generation-prototype repository is in Python path
repo_dir = Path("/content/video-generation-prototype")
if repo_dir.exists() and str(repo_dir) not in sys.path:
    sys.path.insert(0, str(repo_dir))

# Import video_gen components
from video_gen import (
    VideoConfig,
    LongVideoGenerator,
    display_video,
    download_video,
    is_colab,
)
from video_gen.core.image_loader import load_image

print(f"✅ Loaded video_gen successfully! Running in Colab: {is_colab()}")
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 3. Video Generation Parameters { run: "auto" }

# @markdown ### 📝 Text Prompt
# @markdown Describe the camera movement, action, and environment scene:
prompt = "Cinematic aerial camera gliding smoothly over majestic snow-capped mountain peaks at golden hour sunset, 8k photorealistic"  # @param {type:"string"}

# @markdown ### 🖼️ Conditioning Image Source
# @markdown Provide a web URL, a local path, or leave as default Unsplash landscape URL:
image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=1024&q=80"  # @param {type:"string"}

# @markdown ### ⏱️ Video Duration & Dynamics
# @markdown Default is 10.0 seconds (3 chained chunks of 25 frames, 71 total frames):
duration_seconds = 10.0  # @param {type:"number"}

# @markdown Frame rate (fps) - 7 fps is native and low-fps optimal for SVD-XT:
fps = 7  # @param {type:"integer"}

# @markdown Motion bucket ID (lower = subtle motion, higher = vigorous dynamic motion):
motion_bucket_id = 127  # @param {type:"slider", min:50, max:200, step:1}

# @markdown Diffusion denoising inference steps (20-25 recommended for fast generation):
num_inference_steps = 25  # @param {type:"slider", min:15, max:40, step:5}

# @markdown ### ⚙️ Execution Pipeline
# @markdown Choose `svd_xt` (real GPU diffusion) or `mock` (instant test):
pipeline_type = "svd_xt"  # @param ["svd_xt", "mock"]

print(f"Target Video Plan: {duration_seconds}s @ {fps} fps ({int(duration_seconds * fps)} target frames)")
print(f"Prompt: {prompt}")
print(f"Image Source: {image_url}")
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 4. Preview Conditioning Image
# Load and preview conditioning image (from link, local path, or default)
cond_image = load_image(image_url)
print(f"Loaded conditioning image: {cond_image.size[0]}×{cond_image.size[1]} RGB")

# Display preview thumbnail
preview_img = cond_image.copy()
preview_img.thumbnail((640, 360))
display(preview_img)
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 5. Generate Long Video (Chained Autoregressive Diffusion)
# Setup configuration based on pipeline selection
if pipeline_type == "svd_xt" and torch.cuda.is_available():
    print("🎬 Initializing LongVideoGenerator with SVD-XT pipeline on NVIDIA L4 GPU...")
    config = VideoConfig.for_l4_gpu(
        duration_seconds=duration_seconds,
        fps=fps,
        num_inference_steps=num_inference_steps,
        motion_bucket_id=motion_bucket_id,
        seed=42,
    )
else:
    print("⚡ Initializing LongVideoGenerator with fast Mock pipeline...")
    config = VideoConfig.for_mock(
        duration_seconds=duration_seconds,
        fps=fps,
    )

generator = LongVideoGenerator(config=config)

# Progress callback to show step-by-step chaining progress
def on_progress(p):
    pct = p.get("overall_percent", 0.0)
    chunk = p.get("chunk_index", 0)
    total_chunks = p.get("total_chunks", 0)
    msg = p.get("message", "")
    print(f"[{pct:5.1f}%] Chunk {chunk}/{total_chunks}: {msg}")

print(f"Starting video generation ({duration_seconds}s at {fps} fps)...")
result = generator.generate(
    prompt=prompt,
    image=image_url,
    duration_seconds=duration_seconds,
    fps=fps,
    progress_callback=on_progress,
)

print("\\n" + "=" * 50)
print("🎉 Video Generation Finished!")
print(f"📁 Video Path: {result.video_path}")
print(f"⏱️ Duration:   {result.duration_seconds:.2f} seconds ({result.num_frames} frames)")
print(f"🎞️ Resolution: {result.width}×{result.height}")
print(f"🧩 Chunks:     {result.metadata['num_chunks']}")
print(f"⚡ Time taken: {result.metadata['execution_time_seconds']:.2f}s")
print("=" * 50)
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 6. Display Generated Video with Download Button
# Provided Colab API to display the video player and download buttons
display_video(
    result,
    width=720,
    autoplay=True,
    loop=True,
    include_download=True,
    title=f"🎬 10s Generated Video: {prompt[:45]}...",
)
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 7. Programmatic Colab Download (Optional)
# In addition to the "Download Video" button above, you can also trigger
# download directly using the Colab Files API:
from video_gen.colab import download_video

# Uncomment below to trigger instant download prompt:
# download_video(result.video_path)
print(f"Video file located at: {result.video_path}")
print("Click the 'Download Video' or 'Colab Download' button in the player card above to save locally!")
"""
        ),
        nbf.v4.new_code_cell(
            """# @title 8. (Alternative) Generate via FastAPI REST Service
# The repository also provides a production-grade FastAPI REST server.
# Here is how to query the REST API directly:
import requests

try:
    health = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=3).json()
    print("✅ REST API is online:", health)
    print("You can submit jobs via POST http://127.0.0.1:8000/api/v1/generate")
except Exception as exc:
    print(f"💡 REST server check: {exc}")
    print("You can run the server with: !video-gen serve --port 8000 &")
"""
        ),
    ]

    nb.cells = cells
    return nb


def main():
    nb = build_notebook()
    output_paths = [
        Path("/content/video-generation-prototype/examples/video_generation_demo.ipynb"),
        Path("/content/video-generation-prototype/VideoGenerationTest.ipynb"),
        Path("/content/VideoGenerationTest.ipynb"),
    ]

    for p in output_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print(f"Successfully generated: {p}")


if __name__ == "__main__":
    main()
