# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Depth Anything 3 API module.

This module provides the main API for Depth Anything 3, including model loading,
inference, and export capabilities. It supports both single and nested model architectures.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image

from depth_anything_3.cfg import create_object, load_config
from depth_anything_3.registry import MODEL_REGISTRY
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.export import export
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.io.input_processor import InputProcessor
from depth_anything_3.utils.io.output_processor import OutputProcessor
from depth_anything_3.utils.logger import logger
from depth_anything_3.utils.pose_align import align_poses_umeyama

torch.backends.cudnn.benchmark = False
# logger.info("CUDNN Benchmark Disabled")

SAFETENSORS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"


class DepthAnything3(nn.Module, PyTorchModelHubMixin):
    """
    Depth Anything 3 main API class.

    This class provides a high-level interface for depth estimation using Depth Anything 3.
    It supports both single and nested model architectures with metric scaling capabilities.

    Features:
    - Hugging Face Hub integration via PyTorchModelHubMixin
    - Support for multiple model presets (vitb, vitg, nested variants)
    - Automatic mixed precision inference
    - Export capabilities for various formats (GLB, PLY, NPZ, etc.)
    - Camera pose estimation and metric depth scaling

    Usage:
        # Load from Hugging Face Hub
        model = DepthAnything3.from_pretrained("huggingface/model-name")

        # Or create with specific preset
        model = DepthAnything3(preset="vitg")

        # Run inference
        prediction = model.inference(images, export_dir="output", export_format="glb")
    """

    _commit_hash: str | None = None  # Set by mixin when loading from Hub

    def __init__(self, model_name: str = "da3-large", **kwargs):
        """
        Initialize DepthAnything3 with specified preset.

        Args:
        model_name: The name of the model preset to use.
                    Examples: 'da3-giant', 'da3-large', 'da3metric-large', 'da3nested-giant-large'.
        **kwargs: Additional keyword arguments (currently unused).
        """
        super().__init__()
        self.model_name = model_name

        # Build the underlying network
        self.config = load_config(MODEL_REGISTRY[self.model_name])
        self.model = create_object(self.config)
        self.model.eval()

        # Initialize processors
        self.input_processor = InputProcessor()
        self.output_processor = OutputProcessor()
        self._normalize_torch = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Device management (set by user)
        self.device = None

    @torch.inference_mode()
    def forward(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        export_feat_layers: list[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            image: Input batch with shape ``(B, N, 3, H, W)`` on the model device.
            extrinsics: Optional camera extrinsics with shape ``(B, N, 4, 4)``.
            intrinsics: Optional camera intrinsics with shape ``(B, N, 3, 3)``.
            export_feat_layers: Layer indices to return intermediate features for.
            infer_gs: Enable Gaussian Splatting branch.
            use_ray_pose: Use ray-based pose estimation instead of camera decoder.
            ref_view_strategy: Strategy for selecting reference view from multiple views.

        Returns:
            Dictionary containing model predictions
        """
        # Determine optimal autocast dtype
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.no_grad():
            with torch.autocast(device_type=image.device.type, dtype=autocast_dtype):
                return self.model(
                    image, extrinsics, intrinsics, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
                )

    def inference(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        align_to_input_ext_scale: bool = True,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        render_exts: np.ndarray | None = None,
        render_ixts: np.ndarray | None = None,
        render_hw: tuple[int, int] | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_dir: str | None = None,
        export_format: str = "mini_npz",
        export_feat_layers: Sequence[int] | None = None,
        # GLB export parameters
        conf_thresh_percentile: float = 40.0,
        num_max_points: int = 1_000_000,
        show_cameras: bool = True,
        # Feat_vis export parameters
        feat_vis_fps: int = 15,
        # Other export parameters, e.g., gs_ply, gs_video
        export_kwargs: Optional[dict] = {},
    ) -> Prediction:
        """
        Run inference on input images.

        Args:
            image: List of input images (numpy arrays, PIL Images, or file paths)
            extrinsics: Camera extrinsics (N, 4, 4)
            intrinsics: Camera intrinsics (N, 3, 3)
            align_to_input_ext_scale: whether to align the input pose scale to the prediction
            infer_gs: Enable the 3D Gaussian branch (needed for `gs_ply`/`gs_video` exports)
            use_ray_pose: Use ray-based pose estimation instead of camera decoder (default: False)
            ref_view_strategy: Strategy for selecting reference view from multiple views.
                Options: "first", "middle", "saddle_balanced", "saddle_sim_range".
                Default: "saddle_balanced". For single view input (S ≤ 2), no reordering is performed.
            render_exts: Optional render extrinsics for Gaussian video export
            render_ixts: Optional render intrinsics for Gaussian video export
            render_hw: Optional render resolution for Gaussian video export
            process_res: Processing resolution
            process_res_method: Resize method for processing
            export_dir: Directory to export results
            export_format: Export format (mini_npz, npz, glb, ply, gs, gs_video)
            export_feat_layers: Layer indices to export intermediate features from
            conf_thresh_percentile: [GLB] Lower percentile for adaptive confidence threshold (default: 40.0) # noqa: E501
            num_max_points: [GLB] Maximum number of points in the point cloud (default: 1,000,000)
            show_cameras: [GLB] Show camera wireframes in the exported scene (default: True)
            feat_vis_fps: [FEAT_VIS] Frame rate for output video (default: 15)
            export_kwargs: additional arguments to export functions.

        Returns:
            Prediction object containing depth maps and camera parameters
        """
        if "gs" in export_format:
            assert infer_gs, "must set `infer_gs=True` to perform gs-related export."

        if "colmap" in export_format:
            assert isinstance(image[0], str), "`image` must be image paths for COLMAP export."

        # Preprocess images
        imgs_cpu, extrinsics, intrinsics = self._preprocess_inputs(
            image, extrinsics, intrinsics, process_res, process_res_method
        )

        # Prepare tensors for model
        imgs, ex_t, in_t = self._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)

        # Normalize extrinsics
        ex_t_norm = self._normalize_extrinsics(ex_t.clone() if ex_t is not None else None)

        # Run model forward pass
        export_feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        raw_output = self._run_model_forward(
            imgs, ex_t_norm, in_t, export_feat_layers, infer_gs, use_ray_pose, ref_view_strategy
        )

        # Convert raw output to prediction
        prediction = self._convert_to_prediction(raw_output)

        # Align prediction to extrinsincs
        prediction = self._align_to_input_extrinsics_intrinsics(
            extrinsics, intrinsics, prediction, align_to_input_ext_scale
        )

        # Add processed images for visualization
        prediction = self._add_processed_images(prediction, imgs_cpu)

        # Export if requested
        if export_dir is not None:

            if "gs" in export_format:
                if infer_gs and "gs_video" not in export_format:
                    export_format = f"{export_format}-gs_video"
                if "gs_video" in export_format:
                    if "gs_video" not in export_kwargs:
                        export_kwargs["gs_video"] = {}
                    export_kwargs["gs_video"].update(
                        {
                            "extrinsics": render_exts,
                            "intrinsics": render_ixts,
                            "out_image_hw": render_hw,
                        }
                    )
            # Add GLB export parameters
            if "glb" in export_format:
                if "glb" not in export_kwargs:
                    export_kwargs["glb"] = {}
                export_kwargs["glb"].update(
                    {
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "num_max_points": num_max_points,
                        "show_cameras": show_cameras,
                    }
                )
            # Add Feat_vis export parameters
            if "feat_vis" in export_format:
                if "feat_vis" not in export_kwargs:
                    export_kwargs["feat_vis"] = {}
                export_kwargs["feat_vis"].update(
                    {
                        "fps": feat_vis_fps,
                    }
                )
            # Add COLMAP export parameters
            if "colmap" in export_format:
                if "colmap" not in export_kwargs:
                    export_kwargs["colmap"] = {}
                export_kwargs["colmap"].update(
                    {
                        "image_paths": image,
                        "conf_thresh_percentile": conf_thresh_percentile,
                        "process_res_method": process_res_method,
                    }
                )
            self._export_results(prediction, export_format, export_dir, **export_kwargs)

        return prediction

    @torch.inference_mode()
    def inference_torch(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None = None,
        intrinsics: torch.Tensor | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        export_feat_layers: Sequence[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        """
        GPU-only torch inference path that keeps tensors on device.

        Args:
            image: Input tensor in RGB with shape (N, H, W, C) or (N, C, H, W).
                   C can be 3 or 4 (alpha channel is ignored). Integer inputs are
                   treated as [0, 255]; floating inputs are assumed to be [0, 1].
            extrinsics: Optional camera extrinsics with shape (N, 4, 4).
            intrinsics: Optional camera intrinsics with shape (N, 3, 3).
            process_res: Processing resolution.
            process_res_method: One of:
                "upper_bound_resize", "upper_bound_crop",
                "lower_bound_resize", "lower_bound_crop".
            export_feat_layers: Optional feature layers for model forward.
            infer_gs: Enable Gaussian Splatting branch.
            use_ray_pose: Use ray-based pose estimation.
            ref_view_strategy: Reference view selection strategy.

        Returns:
            A dictionary of torch tensors with singleton scene/view dimensions removed:
                - "depth": (N, H', W') at minimum
                - optional model outputs such as "depth_conf", "sky", "extrinsics", "intrinsics"
        """
        model_device = self._get_model_device()
        if model_device.type != "cuda":
            raise RuntimeError(
                "inference_torch is GPU-only. Move model to CUDA first, e.g. model.to('cuda')."
            )

        # Keep API compatibility for extrinsics but ignore it in this training-oriented path for now.
        extrinsics = None

        imgs, ex_t, in_t = self._preprocess_inputs_torch(
            image=image,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            process_res=process_res,
            process_res_method=process_res_method,
            device=model_device,
        )

        # Use B=N, S=1 so each sample is treated as an independent single-view scene.
        imgs = imgs[:, None]  # (N, 1, 3, H', W')
        ex_t = ex_t[:, None] if ex_t is not None else None
        in_t = in_t[:, None] if in_t is not None else None
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else []

        raw_output = self.forward(
            imgs,
            ex_t,
            in_t,
            feat_layers,
            infer_gs,
            use_ray_pose,
            ref_view_strategy,
        )

        return self._postprocess_model_output_torch(raw_output)

    def _preprocess_inputs(
        self,
        image: list[np.ndarray | Image.Image | str],
        extrinsics: np.ndarray | None = None,
        intrinsics: np.ndarray | None = None,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Preprocess input images using input processor."""
        start_time = time.time()
        imgs_cpu, extrinsics, intrinsics = self.input_processor(
            image,
            extrinsics.copy() if extrinsics is not None else None,
            intrinsics.copy() if intrinsics is not None else None,
            process_res,
            process_res_method,
        )
        end_time = time.time()
        logger.info(
            "Processed Images Done taking",
            end_time - start_time,
            "seconds. Shape: ",
            imgs_cpu.shape,
        )
        return imgs_cpu, extrinsics, intrinsics

    def _preprocess_inputs_torch(
        self,
        image: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        process_res: int,
        process_res_method: str,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Torch/GPU preprocessing for same-shape batched images."""
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for image, got {type(image)}.")
        if image.dim() != 4:
            raise ValueError(
                "Expected image with shape (N,H,W,C) or (N,C,H,W), "
                f"got {tuple(image.shape)}."
            )
        if process_res <= 0:
            raise ValueError(f"process_res must be positive, got {process_res}.")

        imgs = image.to(device=device, non_blocking=True)
        imgs = self._to_nchw_rgb(imgs)
        imgs = imgs.float()
        if not torch.is_floating_point(image):
            imgs = imgs / 255.0

        n, _, orig_h, orig_w = imgs.shape
        ex_t, in_t = self._validate_camera_tensors(extrinsics, intrinsics, n, device)

        resized_h, resized_w = self._compute_boundary_resize_shape(
            orig_h, orig_w, process_res, process_res_method
        )
        if resized_h != orig_h or resized_w != orig_w:
            imgs = self._resize_images(imgs, resized_h, resized_w)
            in_t = self._resize_intrinsics(in_t, orig_w, orig_h, resized_w, resized_h)

        if process_res_method.endswith("resize"):
            div_h = self._nearest_multiple(resized_h, self.input_processor.PATCH_SIZE)
            div_w = self._nearest_multiple(resized_w, self.input_processor.PATCH_SIZE)
            div_h = max(1, div_h)
            div_w = max(1, div_w)
            if div_h != resized_h or div_w != resized_w:
                imgs = self._resize_images(imgs, div_h, div_w)
                in_t = self._resize_intrinsics(in_t, resized_w, resized_h, div_w, div_h)
        elif process_res_method.endswith("crop"):
            patch = self.input_processor.PATCH_SIZE
            div_h = (resized_h // patch) * patch
            div_w = (resized_w // patch) * patch
            if div_h <= 0 or div_w <= 0:
                raise ValueError(
                    f"Image size too small for patch size {patch}: {(resized_h, resized_w)}."
                )
            if div_h != resized_h or div_w != resized_w:
                crop_top = (resized_h - div_h) // 2
                crop_left = (resized_w - div_w) // 2
                imgs = imgs[:, :, crop_top : crop_top + div_h, crop_left : crop_left + div_w]
                in_t = self._crop_intrinsics(in_t, crop_left, crop_top)
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        imgs = self._normalize_torch(imgs)

        return imgs.contiguous(), ex_t, in_t

    def _postprocess_model_output_torch(
        self, model_output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Normalize known tensor outputs without applying broad shape assumptions."""
        out: dict[str, torch.Tensor] = {}
        batch_squeeze_keys = {"depth", "depth_conf", "sky", "extrinsics", "intrinsics"}
        map_like_keys = {"depth", "depth_conf", "sky"}
        for key, value in model_output.items():
            if isinstance(value, torch.Tensor):
                tensor = value
                if key in batch_squeeze_keys and tensor.dim() > 1 and tensor.shape[1] == 1:
                    tensor = tensor.squeeze(1)
                if key in batch_squeeze_keys and tensor.dim() > 0 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                if key in map_like_keys:
                    if tensor.dim() == 4 and tensor.shape[-1] == 1:
                        tensor = tensor.squeeze(-1)
                    if tensor.dim() == 4 and tensor.shape[1] == 1:
                        tensor = tensor.squeeze(1)
                out[key] = tensor
        return out

    def _to_nchw_rgb(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image tensor to RGB NCHW."""
        if image.shape[-1] in (3, 4):
            # NHWC -> NCHW
            image = image.permute(0, 3, 1, 2)
        elif image.shape[1] not in (3, 4):
            raise ValueError(
                "Expected channels in last dim (3/4) or second dim (3/4), "
                f"got shape {tuple(image.shape)}."
            )

        if image.shape[1] == 4:
            image = image[:, :3]
        elif image.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB image, got shape {tuple(image.shape)}.")
        return image

    def _validate_camera_tensors(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Validate and move optional camera tensors to target device."""
        ex_t = None
        if extrinsics is not None:
            if not isinstance(extrinsics, torch.Tensor):
                raise TypeError(f"Expected extrinsics as torch.Tensor, got {type(extrinsics)}.")
            if extrinsics.shape != (batch_size, 4, 4):
                raise ValueError(
                    f"Expected extrinsics shape {(batch_size, 4, 4)}, got {tuple(extrinsics.shape)}."
                )
            ex_t = extrinsics.to(device=device, non_blocking=True).float()

        in_t = None
        if intrinsics is not None:
            if not isinstance(intrinsics, torch.Tensor):
                raise TypeError(f"Expected intrinsics as torch.Tensor, got {type(intrinsics)}.")
            if intrinsics.shape != (batch_size, 3, 3):
                raise ValueError(
                    f"Expected intrinsics shape {(batch_size, 3, 3)}, got {tuple(intrinsics.shape)}."
                )
            in_t = intrinsics.to(device=device, non_blocking=True).float()

        return ex_t, in_t

    def _compute_boundary_resize_shape(
        self, height: int, width: int, process_res: int, process_res_method: str
    ) -> tuple[int, int]:
        """Compute resized image shape based on process_res_method."""
        if process_res_method.startswith("upper_bound_"):
            denom = max(height, width)
        elif process_res_method.startswith("lower_bound_"):
            denom = min(height, width)
        else:
            raise ValueError(f"Unsupported process_res_method: {process_res_method}")

        if denom <= 0:
            raise ValueError(f"Invalid image shape {(height, width)}.")

        scale = process_res / float(denom)
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        return new_h, new_w

    def _resize_images(self, imgs: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
        """Resize image batch with interpolation matching CPU path intent."""
        in_h, in_w = imgs.shape[-2], imgs.shape[-1]
        upscale = out_h > in_h or out_w > in_w
        if upscale:
            return F.interpolate(
                imgs, size=(out_h, out_w), mode="bicubic", align_corners=False
            )
        return F.interpolate(imgs, size=(out_h, out_w), mode="area")

    def _resize_intrinsics(
        self,
        intrinsics: torch.Tensor | None,
        in_w: int,
        in_h: int,
        out_w: int,
        out_h: int,
    ) -> torch.Tensor | None:
        """Apply resize transform to intrinsics as done in InputProcessor._resize_ixt."""
        if intrinsics is None:
            return None
        k = intrinsics.clone()
        k[:, 0, :] *= out_w / float(in_w)
        k[:, 1, :] *= out_h / float(in_h)
        return k

    def _crop_intrinsics(
        self, intrinsics: torch.Tensor | None, crop_left: int, crop_top: int
    ) -> torch.Tensor | None:
        """Apply center-crop transform to intrinsics as done in InputProcessor._crop_ixt."""
        if intrinsics is None:
            return None
        k = intrinsics.clone()
        k[:, 0, 2] -= crop_left
        k[:, 1, 2] -= crop_top
        return k

    @staticmethod
    def _nearest_multiple(value: int, patch: int) -> int:
        """Round to nearest patch multiple (ties round up), matching InputProcessor logic."""
        down = (value // patch) * patch
        up = down + patch
        return up if abs(up - value) <= abs(value - down) else down

    def _prepare_model_inputs(
        self,
        imgs_cpu: torch.Tensor,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Prepare tensors for model input."""
        device = self._get_model_device()

        # Move images to model device
        imgs = imgs_cpu.to(device, non_blocking=True)[None].float()

        # Convert camera parameters to tensors
        ex_t = (
            extrinsics.to(device, non_blocking=True)[None].float()
            if extrinsics is not None
            else None
        )
        in_t = (
            intrinsics.to(device, non_blocking=True)[None].float()
            if intrinsics is not None
            else None
        )

        return imgs, ex_t, in_t

    def _normalize_extrinsics(self, ex_t: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize extrinsics"""
        if ex_t is None:
            return None
        transform = affine_inverse(ex_t[:, :1])
        ex_t_norm = ex_t @ transform
        c2ws = affine_inverse(ex_t_norm)
        translations = c2ws[..., :3, 3]
        dists = translations.norm(dim=-1)
        median_dist = torch.median(dists)
        median_dist = torch.clamp(median_dist, min=1e-1)
        ex_t_norm[..., :3, 3] = ex_t_norm[..., :3, 3] / median_dist
        return ex_t_norm

    def _align_to_input_extrinsics_intrinsics(
        self,
        extrinsics: torch.Tensor | None,
        intrinsics: torch.Tensor | None,
        prediction: Prediction,
        align_to_input_ext_scale: bool = True,
        ransac_view_thresh: int = 10,
    ) -> Prediction:
        """Align depth map to input extrinsics"""
        if extrinsics is None:
            return prediction
        prediction.intrinsics = intrinsics.numpy()
        _, _, scale, aligned_extrinsics = align_poses_umeyama(
            prediction.extrinsics,
            extrinsics.numpy(),
            ransac=len(extrinsics) >= ransac_view_thresh,
            return_aligned=True,
            random_state=42,
        )
        if align_to_input_ext_scale:
            prediction.extrinsics = extrinsics[..., :3, :].numpy()
            prediction.depth /= scale
        else:
            prediction.extrinsics = aligned_extrinsics
        return prediction

    def _run_model_forward(
        self,
        imgs: torch.Tensor,
        ex_t: torch.Tensor | None,
        in_t: torch.Tensor | None,
        export_feat_layers: Sequence[int] | None = None,
        infer_gs: bool = False,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
    ) -> dict[str, torch.Tensor]:
        """Run model forward pass."""
        device = imgs.device
        need_sync = device.type == "cuda"
        if need_sync:
            torch.cuda.synchronize(device)
        start_time = time.time()
        feat_layers = list(export_feat_layers) if export_feat_layers is not None else None
        output = self.forward(imgs, ex_t, in_t, feat_layers, infer_gs, use_ray_pose, ref_view_strategy)
        if need_sync:
            torch.cuda.synchronize(device)
        end_time = time.time()
        logger.info(f"Model Forward Pass Done. Time: {end_time - start_time} seconds")
        return output

    def _convert_to_prediction(self, raw_output: dict[str, torch.Tensor]) -> Prediction:
        """Convert raw model output to Prediction object."""
        start_time = time.time()
        output = self.output_processor(raw_output)
        end_time = time.time()
        logger.info(f"Conversion to Prediction Done. Time: {end_time - start_time} seconds")
        return output

    def _add_processed_images(self, prediction: Prediction, imgs_cpu: torch.Tensor) -> Prediction:
        """Add processed images to prediction for visualization."""
        # Convert from (N, 3, H, W) to (N, H, W, 3) and denormalize
        processed_imgs = imgs_cpu.permute(0, 2, 3, 1).cpu().numpy()  # (N, H, W, 3)

        # Denormalize from ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        processed_imgs = processed_imgs * std + mean
        processed_imgs = np.clip(processed_imgs, 0, 1)
        processed_imgs = (processed_imgs * 255).astype(np.uint8)

        prediction.processed_images = processed_imgs
        return prediction

    def _export_results(
        self, prediction: Prediction, export_format: str, export_dir: str, **kwargs
    ) -> None:
        """Export results to specified format and directory."""
        start_time = time.time()
        export(prediction, export_format, export_dir, **kwargs)
        end_time = time.time()
        logger.info(f"Export Results Done. Time: {end_time - start_time} seconds")

    def _get_model_device(self) -> torch.device:
        """
        Get the device where the model is located.

        Returns:
            Device where the model parameters are located

        Raises:
            ValueError: If no tensors are found in the model
        """
        if self.device is not None:
            return self.device

        # Find device from parameters
        for param in self.parameters():
            self.device = param.device
            return param.device

        # Find device from buffers
        for buffer in self.buffers():
            self.device = buffer.device
            return buffer.device

        raise ValueError("No tensor found in model")
