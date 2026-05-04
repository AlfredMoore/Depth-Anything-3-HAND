"""
General DA3 inference utility shared across HAND and dex rl workflows.

Design constraints:
- No config-source fallback: YAML must contain top-level `da3_cfg`.
- No import fallback: depth_anything_3 must be importable via installed package path.
- No runtime fallback: compile path is explicit and controlled by config.
- Two explicit inference entrypoints:
  - infer_chunked(...): HAND-style chunked dispatch
  - infer_no_chunk(...): dex-rl style batch_size=1 dispatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from depth_anything_3.api import DepthAnything3


DEFAULT_DA3_MODEL = "depth-anything/DA3-BASE"
_VALID_PROCESS_RES_METHODS = {
    "upper_bound_resize",
    "upper_bound_crop",
    "lower_bound_resize",
    "lower_bound_crop",
}


def _require_keys(raw: dict[str, Any], required: tuple[str, ...], scope: str) -> None:
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"Missing required keys in {scope}: {missing}")


@dataclass(frozen=True)
class CompileConfig:
    enabled: bool = False
    backend: str = "inductor"
    fullgraph: bool = False
    dynamic: bool = False

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "CompileConfig":
        _require_keys(
            raw,
            (
                "enabled",
                "backend",
                "fullgraph",
                "dynamic",
            ),
            "da3_cfg.compile",
        )
        cfg = CompileConfig(
            enabled=bool(raw["enabled"]),
            backend=str(raw["backend"]),
            fullgraph=bool(raw["fullgraph"]),
            dynamic=bool(raw["dynamic"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.backend.strip():
            raise ValueError("da3_cfg.compile.backend cannot be empty")


@dataclass(frozen=True)
class DA3Config:
    model: str = DEFAULT_DA3_MODEL
    device: str = "cuda"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    chunk_size: int = 64

    focal: float | None = None
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None

    cache_dir: str = "models/da3"
    compile: CompileConfig = field(default_factory=CompileConfig)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "DA3Config":
        if not isinstance(raw, dict):
            raise TypeError(f"da3_cfg must be dict, got {type(raw)}")

        _require_keys(
            raw,
            (
                "model",
                "device",
                "process_res",
                "process_res_method",
                "chunk_size",
                "focal",
                "fx",
                "fy",
                "cx",
                "cy",
                "cache_dir",
                "compile",
            ),
            "da3_cfg",
        )

        if raw["model"] is None:
            raise ValueError("da3_cfg.model is explicitly null. Set a concrete model string.")
        model = str(raw["model"]).strip()
        if not model:
            raise ValueError("da3_cfg.model cannot be empty")

        process_res_method = str(raw["process_res_method"])
        if process_res_method not in _VALID_PROCESS_RES_METHODS:
            raise ValueError(
                f"Unsupported process_res_method: {process_res_method}. "
                f"Expected one of {sorted(_VALID_PROCESS_RES_METHODS)}"
            )

        compile_raw = raw["compile"]
        if compile_raw is None:
            raise ValueError("da3_cfg.compile cannot be null")
        if not isinstance(compile_raw, dict):
            raise TypeError(f"da3_cfg.compile must be dict, got {type(compile_raw)}")

        cfg = DA3Config(
            model=model,
            device=str(raw["device"]),
            process_res=int(raw["process_res"]),
            process_res_method=process_res_method,
            chunk_size=int(raw["chunk_size"]),
            focal=(None if raw["focal"] is None else float(raw["focal"])),
            fx=(None if raw["fx"] is None else float(raw["fx"])),
            fy=(None if raw["fy"] is None else float(raw["fy"])),
            cx=(None if raw["cx"] is None else float(raw["cx"])),
            cy=(None if raw["cy"] is None else float(raw["cy"])),
            cache_dir=str(raw["cache_dir"]),
            compile=CompileConfig.from_dict(compile_raw),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.process_res <= 0:
            raise ValueError(f"process_res must be positive, got {self.process_res}")
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        cache_dir_path = Path(self.cache_dir).expanduser()
        if not cache_dir_path.is_absolute():
            raise ValueError(
                "da3_cfg.cache_dir must be an absolute path. "
                f"Got: {self.cache_dir}"
            )


def load_da3_cfg_from_yaml(yaml_path: str | Path) -> DA3Config:
    """Load DA3 config from YAML top-level key `da3_cfg` (no fallback schema)."""
    path = Path(yaml_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping, got {type(payload)}")
    if "da3_cfg" not in payload:
        raise KeyError(f"Missing top-level key `da3_cfg` in YAML: {path}")

    return DA3Config.from_dict(payload["da3_cfg"])


class _DA3CompileWrapper(torch.nn.Module):
    """Wrap DA3 forward path only (preprocessing stays outside torch.compile)."""

    def __init__(
        self,
        da3_model: DepthAnything3,
        is_metric_model: bool,
        focal: float | None,
    ):
        super().__init__()
        self.da3_model = da3_model
        self.is_metric_model = bool(is_metric_model)
        self.focal = None if focal is None else float(focal)

    def forward(self, preprocessed_nchw: torch.Tensor) -> torch.Tensor:
        imgs = preprocessed_nchw[:, None]  # (N, 1, 3, H, W)
        outputs = self.da3_model.forward(
            imgs,
            None,
            None,
            [],
            False,
            False,
            "saddle_balanced",
        )
        depth = outputs["depth"]
        if depth.dim() > 1 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        if depth.dim() > 0 and depth.shape[0] == 1:
            depth = depth.squeeze(0)
        if depth.dim() == 4 and depth.shape[-1] == 1:
            depth = depth.squeeze(-1)
        if depth.dim() == 4 and depth.shape[1] == 1:
            depth = depth.squeeze(1)
        if self.is_metric_model and self.focal is not None:
            depth = self.focal * depth / 300.0
        return depth


class DA3Inference:
    """Unified DA3 inference runner for HAND (chunked) and dex rl (non-chunked)."""

    def __init__(self, config: DA3Config):
        self.cfg = config
        self.device = torch.device(config.device)
        self.model_name = config.model
        self.process_res = config.process_res
        self.process_res_method = config.process_res_method
        self.chunk_size = config.chunk_size
        self.focal = config.focal
        self._is_metric_model = "metric" in self.model_name.lower()

        self._intrinsics_np: np.ndarray | None = None
        self._intrinsics_torch: torch.Tensor | None = None
        if (
            config.fx is not None
            and config.fy is not None
            and config.cx is not None
            and config.cy is not None
        ):
            self._intrinsics_np = np.array(
                [
                    [config.fx, 0.0, config.cx],
                    [0.0, config.fy, config.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )[None]
            self._intrinsics_torch = torch.from_numpy(self._intrinsics_np).to(
                device=self.device, dtype=torch.float32
            )

        cache_dir = Path(config.cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = DepthAnything3.from_pretrained(self.model_name, cache_dir=str(cache_dir))
        self.model = self.model.to(device=self.device).eval()
        self.model.device = self.device

        self._compiled_runner: torch.nn.Module | None = None
        if self.cfg.compile.enabled:
            self._compiled_runner = self._build_torch_compile_runner()

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "DA3Inference":
        return DA3Inference(load_da3_cfg_from_yaml(yaml_path))

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "DA3Inference":
        return DA3Inference(DA3Config.from_dict(raw))

    def _build_torch_compile_runner(self) -> torch.nn.Module:
        wrapper = _DA3CompileWrapper(
            da3_model=self.model,
            is_metric_model=self._is_metric_model,
            focal=self.focal,
        ).to(self.device).eval()
        return torch.compile(
            wrapper,
            backend=self.cfg.compile.backend,
            fullgraph=self.cfg.compile.fullgraph,
            dynamic=self.cfg.compile.dynamic,
        )

    def _validate_rgb_tensor(self, rgb: torch.Tensor) -> None:
        if not isinstance(rgb, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(rgb)}")
        if rgb.dim() != 4:
            raise ValueError(
                "Expected RGB tensor with shape (N,H,W,3) or (N,3,H,W), "
                f"got {tuple(rgb.shape)}"
            )

        if rgb.shape[-1] not in (3, 4) and rgb.shape[1] not in (3, 4):
            raise ValueError(
                "RGB tensor must have channel dim equal to 3/4 in either dim=1 or dim=-1, "
                f"got shape {tuple(rgb.shape)}"
            )

    def _prepare_rgb_tensor(self, rgb: torch.Tensor) -> torch.Tensor:
        self._validate_rgb_tensor(rgb)
        rgb_t = rgb.to(device=self.device, non_blocking=True)

        if not torch.is_floating_point(rgb_t):
            rgb_t = rgb_t.float() / 255.0
        else:
            rgb_t = rgb_t.float()

        return rgb_t.contiguous()

    def _expand_intrinsics_torch(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self._intrinsics_torch is None:
            return None

        intrinsics = self._intrinsics_torch
        if intrinsics.device != device:
            intrinsics = intrinsics.to(device=device, dtype=torch.float32)
            self._intrinsics_torch = intrinsics

        if intrinsics.shape[0] == 1:
            return intrinsics.expand(batch_size, -1, -1)
        if intrinsics.shape[0] == batch_size:
            return intrinsics

        raise ValueError(
            "intrinsics batch dimension mismatch: "
            f"intrinsics.shape[0]={intrinsics.shape[0]}, batch_size={batch_size}"
        )

    @torch.inference_mode()
    def infer(self, bgr: np.ndarray, process_res_method: str | None = None) -> np.ndarray:
        method = self.process_res_method if process_res_method is None else process_res_method
        if method not in _VALID_PROCESS_RES_METHODS:
            raise ValueError(f"Unsupported process_res_method: {method}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        prediction = self.model.inference(
            [pil_image],
            intrinsics=self._intrinsics_np,
            process_res=self.process_res,
            process_res_method=method,
        )
        depth = prediction.depth[0]
        if self._is_metric_model and self.focal is not None:
            depth = self.focal * depth / 300.0
        return depth

    @torch.inference_mode()
    def infer_rgb(self, rgb: np.ndarray, process_res_method: str | None = None) -> np.ndarray:
        method = self.process_res_method if process_res_method is None else process_res_method
        if method not in _VALID_PROCESS_RES_METHODS:
            raise ValueError(f"Unsupported process_res_method: {method}")
        
        prediction = self.model.inference(
            [rgb],
            intrinsics=self._intrinsics_np,
            process_res=self.process_res,
            process_res_method=method,
        )
        depth = prediction.depth[0]
        if self._is_metric_model and self.focal is not None:
            depth = self.focal * depth / 300.0
        return depth

    @torch.inference_mode()
    def infer_torch_batched(
        self,
        rgb: torch.Tensor,
        *,
        batch_size: int,
        process_res_method: str | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        method = self.process_res_method if process_res_method is None else process_res_method
        if method not in _VALID_PROCESS_RES_METHODS:
            raise ValueError(f"Unsupported process_res_method: {method}")

        rgb_t = self._prepare_rgb_tensor(rgb)
        total = rgb_t.shape[0]

        # Current policy: do not handle non-divisible tail batch automatically.
        if total % batch_size != 0:
            raise ValueError(
                f"Input batch size {total} is not divisible by batch_size {batch_size}. "
                "Tail-batch handling is intentionally disabled."
            )

        out_chunks: list[torch.Tensor] = []
        for start in range(0, total, batch_size):
            end = start + batch_size
            chunk = rgb_t[start:end]
            out_chunks.append(self._infer_one_batch(chunk, method))

        return torch.cat(out_chunks, dim=0)

    @torch.inference_mode()
    def infer_chunked(self, rgb: torch.Tensor, process_res_method: str | None = None) -> torch.Tensor:
        return self.infer_torch_batched(
            rgb,
            batch_size=self.chunk_size,
            process_res_method=process_res_method,
        )

    @torch.inference_mode()
    def infer_no_chunk(self, rgb: torch.Tensor, process_res_method: str | None = None) -> torch.Tensor:
        return self.infer_torch_batched(
            rgb,
            batch_size=1,
            process_res_method=process_res_method,
        )

    @torch.inference_mode()
    def _infer_one_batch(self, rgb_chunk: torch.Tensor, process_res_method: str) -> torch.Tensor:
        if self._compiled_runner is not None:
            if process_res_method != self.process_res_method:
                raise ValueError(
                    "Compiled runner was built with "
                    f"process_res_method={self.process_res_method}, "
                    f"but got {process_res_method}."
                )

            imgs, _, _ = self.model._preprocess_inputs_torch(
                image=rgb_chunk,
                extrinsics=None,
                intrinsics=None,
                process_res=self.process_res,
                process_res_method=process_res_method,
                device=self.device,
            )
            depth = self._compiled_runner(imgs)
            if not isinstance(depth, torch.Tensor):
                raise TypeError(f"Compiled runner output must be torch.Tensor, got {type(depth)}")
            return depth

        intrinsics_t = self._expand_intrinsics_torch(rgb_chunk.shape[0], rgb_chunk.device)
        outputs = self.model.inference_torch(
            image=rgb_chunk,
            device=self.device,
            intrinsics=intrinsics_t,
            process_res=self.process_res,
            process_res_method=process_res_method,
        )
        depth = outputs["depth"]
        if self._is_metric_model and self.focal is not None:
            depth = self.focal * depth / 300.0
        return depth

    def export_runtime_summary(self) -> dict[str, Any]:
        compile_cfg = self.cfg.compile
        return {
            "model": self.model_name,
            "device": str(self.device),
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "chunk_size": self.chunk_size,
            "compile_enabled": compile_cfg.enabled,
            "compile_backend": compile_cfg.backend,
            "compile_fullgraph": compile_cfg.fullgraph,
            "compile_dynamic": compile_cfg.dynamic,
            "strict_mode": {
                "no_cfg_fallback": True,
                "no_import_fallback": True,
                "no_runtime_fallback": True,
            },
        }

    def export_runtime_summary_json(self) -> str:
        return json.dumps(self.export_runtime_summary(), indent=2, sort_keys=True)
