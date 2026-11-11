"""Command line interface for video generation, server hosting, and diagnostics."""

import argparse
import sys
from pathlib import Path
import torch

from video_gen.config import VideoConfig
from video_gen.generator import LongVideoGenerator


def cmd_generate(args):
    """Handle the generate command."""
    print("=" * 60)
    print("🎬 Long Video Generation Prototype (Wan2.1 + GPU Interpolation)")
    print("=" * 60)

    pipeline_type = "mock" if args.mock else (args.pipeline or "wan")

    if pipeline_type == "mock":
        config = VideoConfig.for_mock(
            duration_seconds=args.duration,
            fps=args.fps,
            target_fps=args.target_fps,
        )
        print("Using: Mock Pipeline (fast procedural testing)")
    elif pipeline_type == "svd_xt":
        config = VideoConfig.for_svd_xt(
            duration_seconds=args.duration,
            fps=args.fps or 7,
            target_fps=args.target_fps,
            num_inference_steps=args.steps,
            motion_bucket_id=args.motion_bucket,
            seed=args.seed,
        )
        print("Using: SVD-XT Pipeline")
    elif pipeline_type == "ltx_video":
        config = VideoConfig.for_ltx(
            duration_seconds=args.duration,
            fps=args.fps or 25,
            target_fps=args.target_fps,
            num_inference_steps=args.steps,
            seed=args.seed,
        )
        print("Using: LTX-Video DiT Pipeline")
    else:
        # Default: Wan2.1 14B Image-to-Video foundation model
        config = VideoConfig.for_l4_gpu(
            duration_seconds=args.duration,
            fps=args.fps or 16,
            target_fps=args.target_fps,
            interpolate_method=args.interpolate_method,
            num_inference_steps=args.steps or 30,
            seed=args.seed,
        )
        print("Using: Wan2.1 14B Pipeline on NVIDIA L4 GPU (24GB VRAM)")

    generator = LongVideoGenerator(config=config)

    def print_progress(info):
        pct = info.get("overall_percent", 0.0)
        msg = info.get("message", "")
        print(f"[{pct:5.1f}%] {msg}")

    print(f"Target duration: {args.duration}s | Native FPS: {config.generation.fps} | Output FPS: {args.target_fps or config.generation.fps}")
    if args.prompt:
        print(f"Prompt: {args.prompt[:80]}...")
    if args.prompt_file:
        print(f"Prompt file: {args.prompt_file}")
    if args.image:
        print(f"Image: {args.image}")
    if args.images:
        print(f"Images: {args.images}")

    result = generator.generate(
        prompt=args.prompt,
        prompt_file=args.prompt_file,
        image=args.image,
        images=args.images,
        duration_seconds=args.duration,
        fps=config.generation.fps,
        target_fps=args.target_fps,
        interpolate_factor=args.interpolate_factor,
        interpolate_method=args.interpolate_method,
        motion_bucket_id=args.motion_bucket,
        num_inference_steps=args.steps,
        seed=args.seed,
        output_path=args.output,
        progress_callback=print_progress,
    )

    print("=" * 60)
    print(f"✅ Video generated successfully!")
    print(f"Saved to:  {result.video_path}")
    print(f"Duration:  {result.duration_seconds:.2f}s ({result.num_frames} frames @ {result.fps} fps)")
    print(f"Pipeline:  {result.metadata.get('pipeline')}")
    print(f"Chunks:    {result.metadata.get('num_chunks')}")
    print(f"Smooth:    {'High-FPS GPU interpolated' if result.metadata.get('interpolated') else 'Native framerate'}")
    print(f"Exec Time: {result.metadata.get('execution_time_seconds')}s")
    print("=" * 60)


def cmd_serve(args):
    """Launch the FastAPI uvicorn server."""
    import uvicorn
    print(f"🚀 Launching Video Generation API server on {args.host}:{args.port}...")
    print(f"Swagger documentation: http://{args.host}:{args.port}/docs")
    uvicorn.run(
        "video_gen.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


def cmd_info(args):
    """Print system diagnostics, GPU memory, and pipeline caches."""
    print("=" * 60)
    print("🔍 System Diagnostics & GPU Acceleration")
    print("=" * 60)
    print(f"Python:       {sys.version.split()[0]}")
    print(f"PyTorch:      {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")

    if cuda_avail:
        print(f"Device Name:  {torch.cuda.get_device_name(0)}")
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total VRAM:   {total_vram:.2f} GB")
        alloc_vram = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"Alloc VRAM:   {alloc_vram:.2f} GB")
        print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")
    else:
        print("Running in CPU-only mode.")

    # Check Optical flow model support
    try:
        from torchvision.models.optical_flow import raft_small
        print("RAFT Optical Flow: Supported & Available")
    except ImportError:
        print("RAFT Optical Flow: Not installed")

    # Check Wan cache
    wan_cache = Path("/root/.cache/huggingface/hub/models--Wan-AI--Wan2.1-I2V-14B-480P-Diffusers")
    print(f"Wan2.1 Cache: {'Found' if wan_cache.exists() else 'Not cached'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="video-gen",
        description="CLI for generating continuous long videos with Wan2.1 and high-FPS GPU interpolation.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate a video from prompt and image(s)")
    gen_parser.add_argument("--prompt", type=str, default=None, help="Scene prompt description")
    gen_parser.add_argument("--prompt-file", type=str, default=None, help="Path to .txt prompt file")
    gen_parser.add_argument("--image", type=str, default=None, help="Image URL, path, or base64")
    gen_parser.add_argument("--images", nargs="+", type=str, default=None, help="List of images for storyboard")
    gen_parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds (default: 10.0)")
    gen_parser.add_argument("--fps", type=int, default=16, help="Native chunk generation framerate (default: 16)")
    gen_parser.add_argument("--target-fps", type=int, default=60, help="Target high framerate via GPU interpolation (default: 60)")
    gen_parser.add_argument("--interpolate-factor", type=int, default=None, help="Integer frame multiplier (e.g. 2, 4)")
    gen_parser.add_argument("--interpolate-method", type=str, default="auto", choices=["auto", "raft", "blend"], help="Interpolation method (default: auto)")
    gen_parser.add_argument("--steps", type=int, default=30, help="Inference steps per chunk (default: 30)")
    gen_parser.add_argument("--motion-bucket", type=int, default=127, help="Motion bucket index (1-255)")
    gen_parser.add_argument("--pipeline", type=str, default="wan", choices=["wan", "svd_xt", "ltx_video", "mock"], help="Pipeline engine (default: wan)")
    gen_parser.add_argument("--seed", type=int, default=None, help="Reproducibility seed")
    gen_parser.add_argument("--output", type=str, default=None, help="Output MP4 file path")
    gen_parser.add_argument("--mock", action="store_true", help="Use fast procedural mock pipeline for testing")
    gen_parser.set_defaults(func=cmd_generate)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI REST server")
    serve_parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve_parser.set_defaults(func=cmd_serve)

    # Info command
    info_parser = subparsers.add_parser("info", help="Display GPU and environment information")
    info_parser.set_defaults(func=cmd_info)

    args = parser.parse_args()
    if args.subcommand is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
