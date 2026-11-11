"""Build interactive Colab demo notebooks for the repository."""

from pathlib import Path
import nbformat as nbf


def build_notebook():
    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell(
            """# 🎬 Long Video Generation Demo in Google Colab
### Continuous Long Video Generation with Wan2.1 14B & High-FPS GPU Interpolation (NVIDIA L4 GPU)

This notebook demonstrates how to generate high-fidelity, continuous long videos (default **10.0 seconds** with **Wan2.1 14B Image-to-Video foundation model**, interpolated to smooth **60 FPS** via deep optical flow) using the **`video-generation-prototype`** API:
1. **Wan2.1 14B Foundation Model**: Flagship open DiT video foundation model by Alibaba running in **Float 8-bit (`float8_e4m3fn`)** native to Ada Lovelace / NVIDIA L4 GPU (with automatic fallback to 16-bit `bfloat16`/`float16` if FP8 is unsupported) and sequential model CPU offload.
2. **Deep Learning Optical Flow Frame Interpolation**: GPU-accelerated motion-compensated bidirectional warping (RAFT) upscaling native 16 fps to silky smooth 60 fps.
3. **Autoregressive Temporal Chaining**: Generates sequential 81-frame chunks with cross-dissolve frame blending to eliminate seam cuts.
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
    bf16 = torch.cuda.is_bf16_supported()
    from video_gen.config import is_float8_supported, get_optimal_torch_dtype
    fp8 = is_float8_supported()
    opt_dtype = get_optimal_torch_dtype()
    print(f"🚀 GPU Detected: {device_name} ({vram_gb:.1f} GB VRAM, BF16={bf16}, FP8={fp8})")
    print(f"⚡ Default Precision: {opt_dtype} (FP8 with automatic 16-bit fallback)")
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
    GPUFrameInterpolator,
    interpolate_frames,
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
image_url = "https://images.unsplash.com/photo-1506744038136-46273834b3fb?w=832&q=80"  # @param {type:"string"}

# @markdown ### ⏱️ Video Duration & Dynamics
# @markdown Desired video duration in seconds:
duration_seconds = 10.0  # @param {type:"number"}

# @markdown Native chunk framerate (default 16 for Wan2.1):
fps = 16  # @param {type:"integer"}

# @markdown Target high FPS via GPU Optical-Flow interpolation (e.g. 60 fps):
target_fps = 60  # @param {type:"slider", min:16, max:120, step:1}

# @markdown Diffusion denoising inference steps:
num_inference_steps = 30  # @param {type:"slider", min:15, max:50, step:5}

# @markdown ### ⚙️ Execution Pipeline & Precision
# @markdown Choose `wan` (Wan2.1 14B on GPU), `svd_xt`, `ltx_video`, or `mock` (instant test):
pipeline_type = "wan"  # @param ["wan", "svd_xt", "ltx_video", "mock"]

# @markdown Model precision: Float 8-bit (`auto` defaults to `float8_e4m3fn` on L4 GPU with 16-bit fallback):
precision = "auto"  # @param ["auto", "float8_e4m3fn", "bfloat16", "float16"]

print(f"Target Video Plan: {duration_seconds}s @ native {fps} fps -> High-FPS output: {target_fps} fps (precision: {precision})")
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
            """# @title 5. Generate Long Video (Chained Autoregressive Diffusion + GPU Interpolation)
# Setup configuration based on pipeline selection
if pipeline_type == "mock" or not torch.cuda.is_available():
    print("⚡ Initializing LongVideoGenerator with fast Mock pipeline...")
    config = VideoConfig.for_mock(
        duration_seconds=duration_seconds,
        fps=fps,
        target_fps=target_fps,
    )
elif pipeline_type == "svd_xt":
    print("🎬 Initializing SVD-XT pipeline...")
    config = VideoConfig.for_svd_xt(
        duration_seconds=duration_seconds,
        fps=7,
        target_fps=target_fps,
        num_inference_steps=num_inference_steps,
        seed=42,
    )
elif pipeline_type == "ltx_video":
    print("🎬 Initializing LTX-Video DiT pipeline...")
    config = VideoConfig.for_ltx(
        duration_seconds=duration_seconds,
        fps=25,
        target_fps=target_fps,
        num_inference_steps=num_inference_steps,
        seed=42,
    )
else:
    print("🎬 Initializing LongVideoGenerator with Wan2.1 14B Image-to-Video pipeline on NVIDIA L4 GPU...")
    torch_dtype = None if precision == "auto" else precision
    config = VideoConfig.for_l4_gpu(
        duration_seconds=duration_seconds,
        fps=fps,
        target_fps=target_fps,
        num_inference_steps=num_inference_steps,
        torch_dtype=torch_dtype,
        seed=42,
    )

generator = LongVideoGenerator(config=config)

# Progress callback to show step-by-step chaining and interpolation progress
def on_progress(p):
    pct = p.get("overall_percent", 0.0)
    chunk = p.get("chunk_index", 0)
    total_chunks = p.get("total_chunks", 0)
    msg = p.get("message", "")
    print(f"[{pct:5.1f}%] Chunk {chunk}/{total_chunks}: {msg}")

print(f"Starting video generation ({duration_seconds}s at {fps} fps -> target {target_fps} fps)...")
result = generator.generate(
    prompt=prompt,
    image=image_url,
    duration_seconds=duration_seconds,
    fps=fps,
    target_fps=target_fps,
    progress_callback=on_progress,
)

print("\\n" + "=" * 50)
print("🎉 Video Generation Finished!")
print(f"📁 Video Path: {result.video_path}")
print(f"⏱️ Duration:   {result.duration_seconds:.2f} seconds ({result.num_frames} frames @ {result.fps} fps)")
print(f"🎞️ Resolution: {result.width}×{result.height}")
print(f"🧩 Chunks:     {result.metadata['num_chunks']}")
print(f"🚀 Smooth:     {'High-FPS GPU interpolated' if result.metadata.get('interpolated') else 'Native'}")
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
    title=f"🎬 {result.fps} FPS Generated Video: {prompt[:45]}...",
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
