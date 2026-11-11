"""GPU-accelerated deep optical flow and tensor frame interpolation for high-FPS video."""

import logging
import math
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GPUFrameInterpolator:
    """High-performance frame interpolator utilizing GPU tensor cores and deep optical flow.

    Supports:
    1. Deep Learning Optical Flow (RAFT: Recurrent All-Pairs Field Transforms)
       - Motion-compensated bidirectional warping with zero ghosting
       - Executed fully on CUDA with FP16/BF16/FP32
    2. Vectorized Batched GPU Tensor Blending
       - Extreme speed for real-time or ultra-high frame count scenarios
    3. Arbitrary High Target Framerates (e.g., 16 fps -> 30 fps, 60 fps, 120 fps)
       or Integer Upsampling Factors (2x, 4x, 8x).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "raft_small",
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.torch_dtype = torch_dtype or (torch.float32 if self.device == "cpu" else torch.float32)
        self._raft_model: Optional[torch.nn.Module] = None

    def _load_raft_model(self) -> Optional[torch.nn.Module]:
        """Lazy loader for RAFT optical flow model."""
        if self._raft_model is not None:
            return self._raft_model

        try:
            from torchvision.models.optical_flow import (
                raft_small,
                raft_large,
                Raft_Small_Weights,
                Raft_Large_Weights,
            )

            logger.info("Loading RAFT optical flow model (%s) onto %s...", self.model_name, self.device)
            if self.model_name == "raft_large":
                model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            else:
                model = raft_small(weights=Raft_Small_Weights.DEFAULT)

            model = model.to(device=self.device, dtype=self.torch_dtype).eval()
            self._raft_model = model
            logger.info("RAFT model loaded successfully.")
            return self._raft_model
        except Exception as exc:
            logger.warning("Could not initialize RAFT model (%s): %s. Falling back to GPU tensor blending.", self.model_name, exc)
            return None

    @staticmethod
    def warp_flow(img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Warp an image tensor (B, C, H, W) according to optical flow displacement (B, 2, H, W)."""
        B, C, H, W = img.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=img.device, dtype=img.dtype),
            torch.arange(0, W, device=img.device, dtype=img.dtype),
            indexing="ij",
        )
        gx = grid_x + flow[:, 0, :, :]
        gy = grid_y + flow[:, 1, :, :]

        # Normalize grid coordinates to [-1, 1] required by grid_sample
        gx_norm = 2.0 * gx / max(W - 1, 1) - 1.0
        gy_norm = 2.0 * gy / max(H - 1, 1) - 1.0
        grid = torch.stack((gx_norm, gy_norm), dim=-1)

        return F.grid_sample(
            img,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

    def _compute_flow(
        self,
        f0: torch.Tensor,
        f1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute bidirectional optical flows F_{0->1} and F_{1->0} using RAFT.

        Args:
            f0: First frame tensor (1, 3, H, W) in range [0, 255].
            f1: Second frame tensor (1, 3, H, W) in range [0, 255].

        Returns:
            Tuple of (flow_0_to_1, flow_1_to_0), each of shape (1, 2, H, W).
        """
        model = self._load_raft_model()
        if model is None:
            raise RuntimeError("RAFT model is not available.")

        _, _, H, W = f0.shape
        # RAFT requires height and width divisible by 8
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8

        if pad_h > 0 or pad_w > 0:
            f0_pad = F.pad(f0, (0, pad_w, 0, pad_h), mode="replicate")
            f1_pad = F.pad(f1, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            f0_pad, f1_pad = f0, f1

        with torch.no_grad():
            flow_01 = model(f0_pad, f1_pad)[-1]
            flow_10 = model(f1_pad, f0_pad)[-1]

        if pad_h > 0 or pad_w > 0:
            flow_01 = flow_01[:, :, :H, :W]
            flow_10 = flow_10[:, :, :H, :W]

        return flow_01, flow_10

    def interpolate_pair(
        self,
        f0: torch.Tensor,
        f1: torch.Tensor,
        alphas: List[float],
        method: str = "auto",
    ) -> List[torch.Tensor]:
        """Interpolate intermediate frames between f0 and f1 at fractional points alphas in (0, 1).

        Args:
            f0: First frame tensor (1, 3, H, W) in [0, 1].
            f1: Second frame tensor (1, 3, H, W) in [0, 1].
            alphas: Fractional timestamps between 0 and 1.
            method: "auto", "raft", or "blend".

        Returns:
            List of intermediate tensors of shape (1, 3, H, W) in [0, 1].
        """
        use_raft = (method in ("auto", "raft")) and (self._load_raft_model() is not None)

        if use_raft:
            try:
                # Compute bidirectional flow once on GPU
                flow_01, flow_10 = self._compute_flow(f0 * 255.0, f1 * 255.0)

                intermediates = []
                for a in alphas:
                    # Bidirectional motion-compensated warping
                    w0 = self.warp_flow(f0, -a * flow_01)
                    w1 = self.warp_flow(f1, -(1.0 - a) * flow_10)
                    blended = (1.0 - a) * w0 + a * w1
                    intermediates.append(torch.clamp(blended, 0.0, 1.0))
                return intermediates
            except Exception as exc:
                logger.warning("Optical flow interpolation encountered an issue: %s. Falling back to tensor blend.", exc)

        # Fast GPU tensor linear blending fallback
        intermediates = []
        for a in alphas:
            blended = (1.0 - a) * f0 + a * f1
            intermediates.append(torch.clamp(blended, 0.0, 1.0))
        return intermediates

    def interpolate_sequence(
        self,
        frames: List[Image.Image],
        factor: Optional[int] = None,
        target_fps: Optional[int] = None,
        source_fps: int = 16,
        method: str = "auto",
    ) -> Tuple[List[Image.Image], int]:
        """Interpolate an entire sequence of PIL frames to a high framerate.

        Args:
            frames: Input sequence of PIL RGB images.
            factor: Integer upsampling factor (e.g. 2 for 2x fps).
            target_fps: Desired high target FPS (e.g. 30, 60, 120 fps).
            source_fps: Original video playback framerate.
            method: "auto", "raft", or "blend".

        Returns:
            Tuple of (interpolated_frames_list, final_fps).
        """
        if len(frames) < 2:
            final_fps = target_fps or (source_fps * (factor or 1))
            return list(frames), final_fps

        # Resolve target fps and mapping
        if target_fps is not None and target_fps > source_fps:
            duration = (len(frames) - 1) / source_fps
            target_count = max(len(frames), int(round(duration * target_fps)) + 1)
            final_fps = target_fps
        elif factor is not None and factor > 1:
            target_count = (len(frames) - 1) * factor + 1
            final_fps = source_fps * factor
        else:
            return list(frames), source_fps

        logger.info(
            "Interpolating %d frames from %d fps to %d fps (method=%s, device=%s)...",
            len(frames),
            source_fps,
            final_fps,
            method,
            self.device,
        )

        # Pre-convert frames to tensors on target device
        tensors = []
        for f in frames:
            np_arr = np.array(f.convert("RGB"), dtype=np.float32) / 255.0
            t = torch.from_numpy(np_arr).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.torch_dtype)
            tensors.append(t)

        N = len(frames)
        # Compute query timestamps across timeline
        # For each target frame k in [0, target_count - 1]:
        # timestamp = k / final_fps
        # source_time = timestamp * source_fps
        # interval index i = min(floor(source_time), N - 2)
        # alpha = source_time - i

        interval_queries: List[List[Tuple[int, float]]] = [[] for _ in range(N - 1)]

        for k in range(target_count):
            tau = k / final_fps
            s = tau * source_fps
            i = min(int(math.floor(s)), N - 2)
            alpha = s - i
            # Bound alpha to [0, 1]
            alpha = max(0.0, min(1.0, alpha))
            interval_queries[i].append((k, alpha))

        output_tensors: List[Optional[torch.Tensor]] = [None] * target_count

        for i in range(N - 1):
            queries = interval_queries[i]
            if not queries:
                continue

            f0 = tensors[i]
            f1 = tensors[i + 1]

            # Separate boundary queries (alpha == 0 or alpha == 1) from intermediate
            intermediate_alphas = []
            intermediate_indices = []

            for k_idx, alpha in queries:
                if abs(alpha) < 1e-5:
                    output_tensors[k_idx] = f0
                elif abs(alpha - 1.0) < 1e-5:
                    output_tensors[k_idx] = f1
                else:
                    intermediate_alphas.append(alpha)
                    intermediate_indices.append(k_idx)

            if intermediate_alphas:
                generated_intermediates = self.interpolate_pair(
                    f0,
                    f1,
                    intermediate_alphas,
                    method=method,
                )
                for out_idx, tensor_res in zip(intermediate_indices, generated_intermediates):
                    output_tensors[out_idx] = tensor_res

        # Ensure last frame is placed
        if output_tensors[-1] is None:
            output_tensors[-1] = tensors[-1]

        # Convert back to PIL images
        result_pil: List[Image.Image] = []
        for out_t in output_tensors:
            if out_t is None:
                continue
            arr = out_t.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            result_pil.append(Image.fromarray(arr))

        logger.info(
            "Successfully interpolated to %d frames at %d fps (smooth playback).",
            len(result_pil),
            final_fps,
        )
        return result_pil, final_fps
