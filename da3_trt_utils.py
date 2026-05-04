"""
General DA3 inference utility shared across HAND and dex rl workflows.

Design constraints:
- No config-source fallback: YAML must contain top-level `da3_cfg`.
- No import fallback: depth_anything_3 must be importable via installed package path.
- No engine fallback: TensorRT path either loads/compiles successfully or raises.
- Two explicit inference entrypoints:
  - infer_chunked(...): HAND-style chunked dispatch
  - infer_no_chunk(...): dex-rl style batch_size=1 dispatch
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import hashlib
import json

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


_DA3_UTILS_FILE = Path(__file__).resolve()
_DA3_REPO_ROOT = _DA3_UTILS_FILE.parent
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
class TRTCompileConfig:
    enabled: bool = False
    precision: str = "fp16"  # fp16 | fp32
    min_block_size: int = 1
    workspace_size: int = 0
    max_aux_streams: int | None = None
    optimization_level: int | None = None
    require_full_compilation: bool = True
    pass_through_build_failures: bool = True

    use_dynamic_batch: bool = False
    batch_size: int = 1
    min_batch_size: int = 1
    opt_batch_size: int = 1
    max_batch_size: int = 1

    input_height: int = 240
    input_width: int = 320
    engine_dir: str = "models/da3/trt_engines"

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "TRTCompileConfig":
        _require_keys(
            raw,
            (
                "enabled",
                "precision",
                "min_block_size",
                "workspace_size",
                "max_aux_streams",
                "optimization_level",
                "require_full_compilation",
                "pass_through_build_failures",
                "use_dynamic_batch",
                "batch_size",
                "min_batch_size",
                "opt_batch_size",
                "max_batch_size",
                "input_height",
                "input_width",
                "engine_dir",
            ),
            "da3_cfg.trt",
        )
        cfg = TRTCompileConfig(
            enabled=bool(raw["enabled"]),
            precision=str(raw["precision"]).lower(),
            min_block_size=int(raw["min_block_size"]),
            workspace_size=int(raw["workspace_size"]),
            max_aux_streams=(None if raw["max_aux_streams"] is None else int(raw["max_aux_streams"])),
            optimization_level=(
                None if raw["optimization_level"] is None else int(raw["optimization_level"])
            ),
            require_full_compilation=bool(raw["require_full_compilation"]),
            pass_through_build_failures=bool(raw["pass_through_build_failures"]),
            use_dynamic_batch=bool(raw["use_dynamic_batch"]),
            batch_size=int(raw["batch_size"]),
            min_batch_size=int(raw["min_batch_size"]),
            opt_batch_size=int(raw["opt_batch_size"]),
            max_batch_size=int(raw["max_batch_size"]),
            input_height=int(raw["input_height"]),
            input_width=int(raw["input_width"]),
            engine_dir=str(raw["engine_dir"]),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.precision not in {"fp16", "fp32"}:
            raise ValueError(f"Unsupported trt.precision: {self.precision}")
        if self.min_block_size <= 0:
            raise ValueError(f"trt.min_block_size must be positive, got {self.min_block_size}")
        if self.workspace_size < 0:
            raise ValueError(f"trt.workspace_size must be >= 0, got {self.workspace_size}")
        if self.input_height <= 0 or self.input_width <= 0:
            raise ValueError(
                "trt.input_height and trt.input_width must be positive, "
                f"got ({self.input_height}, {self.input_width})"
            )
        engine_dir_path = Path(self.engine_dir).expanduser()
        if not engine_dir_path.is_absolute():
            raise ValueError(
                "da3_cfg.trt.engine_dir must be an absolute path. "
                f"Got: {self.engine_dir}"
            )

        if self.use_dynamic_batch:
            if not (1 <= self.min_batch_size <= self.opt_batch_size <= self.max_batch_size):
                raise ValueError(
                    "Dynamic batch profile must satisfy "
                    "1 <= min_batch_size <= opt_batch_size <= max_batch_size"
                )
        else:
            if self.batch_size <= 0:
                raise ValueError(f"trt.batch_size must be positive, got {self.batch_size}")


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
    trt: TRTCompileConfig = field(default_factory=TRTCompileConfig)

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
                "trt",
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

        trt_raw = raw["trt"]
        if trt_raw is None:
            raise ValueError("da3_cfg.trt cannot be null")
        if not isinstance(trt_raw, dict):
            raise TypeError(f"da3_cfg.trt must be dict, got {type(trt_raw)}")

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
            trt=TRTCompileConfig.from_dict(trt_raw),
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


class _DA3TRTWrapper(torch.nn.Module):
    """Wrap DepthAnything3.inference_torch(image=...) into a single-input module."""

    def __init__(
        self,
        da3_model: DepthAnything3,
        process_res: int,
        process_res_method: str,
        intrinsics_torch: torch.Tensor | None,
        is_metric_model: bool,
        focal: float | None,
    ):
        super().__init__()
        self.da3_model = da3_model
        self.process_res = int(process_res)
        self.process_res_method = process_res_method
        self.is_metric_model = bool(is_metric_model)
        self.focal = None if focal is None else float(focal)

        if intrinsics_torch is None:
            intrinsics_torch = torch.empty((0, 3, 3), dtype=torch.float32)
        else:
            intrinsics_torch = intrinsics_torch.detach().to(dtype=torch.float32)
        self.register_buffer("intrinsics_torch", intrinsics_torch, persistent=False)

    def _expand_intrinsics(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.intrinsics_torch.numel() == 0:
            return None
        intrinsics = self.intrinsics_torch
        if intrinsics.device != device:
            intrinsics = intrinsics.to(device=device, dtype=torch.float32)

        if intrinsics.shape[0] == 1:
            return intrinsics.expand(batch_size, -1, -1)
        if intrinsics.shape[0] == batch_size:
            return intrinsics
        raise ValueError(
            "intrinsics batch dimension mismatch: "
            f"intrinsics.shape[0]={intrinsics.shape[0]}, batch_size={batch_size}"
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        intrinsics = self._expand_intrinsics(image.shape[0], image.device)
        outputs = self.da3_model.inference_torch(
            image=image,
            intrinsics=intrinsics,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
        )
        depth = outputs["depth"]
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

        self._trt_runner: torch.nn.Module | None = None
        self._trt_engine_path: Path | None = None
        if self.cfg.trt.enabled:
            self._trt_runner, self._trt_engine_path = self._build_or_load_trt_runner()

    @staticmethod
    def from_yaml(yaml_path: str | Path) -> "DA3Inference":
        return DA3Inference(load_da3_cfg_from_yaml(yaml_path))

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "DA3Inference":
        return DA3Inference(DA3Config.from_dict(raw))

    def _sanitize_model_for_filename(self) -> str:
        return self.model_name.replace("/", "--").replace(".", "_")

    def _engine_basename(self) -> str:
        trt = self.cfg.trt
        batch_tag = (
            f"dynb{trt.min_batch_size}-{trt.opt_batch_size}-{trt.max_batch_size}"
            if trt.use_dynamic_batch
            else f"b{trt.batch_size}"
        )
        stem = (
            f"{self._sanitize_model_for_filename()}"
            f"_res{self.process_res}_{self.process_res_method}"
            f"_h{trt.input_height}_w{trt.input_width}_{batch_tag}_{trt.precision}"
        )
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        return f"{stem}_{digest}.ep"

    def _get_engine_path(self) -> Path:
        trt = self.cfg.trt
        engine_dir = Path(trt.engine_dir).expanduser().resolve()
        engine_dir.mkdir(parents=True, exist_ok=True)
        return engine_dir / self._engine_basename()

    def _trt_precision_set(self) -> set[torch.dtype]:
        if self.cfg.trt.precision == "fp16":
            return {torch.float16}
        if self.cfg.trt.precision == "fp32":
            return {torch.float32}
        raise ValueError(f"Unsupported TRT precision: {self.cfg.trt.precision}")

    def _trt_input_spec(self, torch_tensorrt: Any) -> list[Any]:
        trt = self.cfg.trt
        shape_hwc = (trt.input_height, trt.input_width, 3)

        if trt.use_dynamic_batch:
            return [
                torch_tensorrt.Input(
                    min_shape=(trt.min_batch_size, *shape_hwc),
                    opt_shape=(trt.opt_batch_size, *shape_hwc),
                    max_shape=(trt.max_batch_size, *shape_hwc),
                    dtype=torch.float32,
                    format=torch.contiguous_format,
                )
            ]

        return [
            torch_tensorrt.Input(
                shape=(trt.batch_size, *shape_hwc),
                dtype=torch.float32,
                format=torch.contiguous_format,
            )
        ]

    def _unwrap_loaded_trt_module(self, loaded: Any) -> torch.nn.Module:
        if hasattr(loaded, "module") and callable(loaded.module):
            mod = loaded.module()
            if not isinstance(mod, torch.nn.Module):
                raise TypeError(f"torch_tensorrt.load(...).module() returned {type(mod)}")
            return mod
        if isinstance(loaded, torch.nn.Module):
            return loaded
        raise TypeError(f"Unsupported object type from torch_tensorrt.load: {type(loaded)}")

    def _build_or_load_trt_runner(self) -> tuple[torch.nn.Module, Path]:
        try:
            import torch_tensorrt
        except ImportError as exc:
            raise ImportError(
                "TRT compile requested (da3_cfg.trt.enabled=true), but torch_tensorrt is not installed"
            ) from exc

        engine_path = self._get_engine_path()
        input_specs = self._trt_input_spec(torch_tensorrt)

        if engine_path.exists():
            loaded = torch_tensorrt.load(str(engine_path))
            runner = self._unwrap_loaded_trt_module(loaded).to(self.device).eval()
            return runner, engine_path

        wrapper = _DA3TRTWrapper(
            da3_model=self.model,
            process_res=self.process_res,
            process_res_method=self.process_res_method,
            intrinsics_torch=self._intrinsics_torch,
            is_metric_model=self._is_metric_model,
            focal=self.focal,
        ).to(self.device).eval()

        compile_kwargs: dict[str, Any] = {
            "ir": "dynamo",
            "arg_inputs": input_specs,
            "enabled_precisions": self._trt_precision_set(),
            "min_block_size": self.cfg.trt.min_block_size,
            "workspace_size": self.cfg.trt.workspace_size,
            "require_full_compilation": self.cfg.trt.require_full_compilation,
            "pass_through_build_failures": self.cfg.trt.pass_through_build_failures,
            "use_python_runtime": False,
        }
        if self.cfg.trt.max_aux_streams is not None:
            compile_kwargs["max_aux_streams"] = self.cfg.trt.max_aux_streams
        if self.cfg.trt.optimization_level is not None:
            compile_kwargs["optimization_level"] = self.cfg.trt.optimization_level

        trt_gm = torch_tensorrt.compile(wrapper, **compile_kwargs)
        torch_tensorrt.save(trt_gm, str(engine_path), arg_inputs=input_specs)
        loaded = torch_tensorrt.load(str(engine_path))
        runner = self._unwrap_loaded_trt_module(loaded).to(self.device).eval()
        return runner, engine_path

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
        if self._trt_runner is not None:
            depth = self._trt_runner(rgb_chunk)
            if not isinstance(depth, torch.Tensor):
                raise TypeError(f"TRT runner output must be torch.Tensor, got {type(depth)}")
            return depth

        intrinsics_t = self._expand_intrinsics_torch(rgb_chunk.shape[0], rgb_chunk.device)
        outputs = self.model.inference_torch(
            image=rgb_chunk,
            intrinsics=intrinsics_t,
            process_res=self.process_res,
            process_res_method=process_res_method,
        )
        depth = outputs["depth"]
        if self._is_metric_model and self.focal is not None:
            depth = self.focal * depth / 300.0
        return depth

    def export_runtime_summary(self) -> dict[str, Any]:
        trt = self.cfg.trt
        return {
            "model": self.model_name,
            "device": str(self.device),
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "chunk_size": self.chunk_size,
            "trt_enabled": trt.enabled,
            "trt_engine_path": None if self._trt_engine_path is None else str(self._trt_engine_path),
            "trt_precision": trt.precision,
            "trt_dynamic_batch": trt.use_dynamic_batch,
            "trt_profile": {
                "batch_size": trt.batch_size,
                "min_batch_size": trt.min_batch_size,
                "opt_batch_size": trt.opt_batch_size,
                "max_batch_size": trt.max_batch_size,
                "input_height": trt.input_height,
                "input_width": trt.input_width,
            },
            "strict_mode": {
                "no_cfg_fallback": True,
                "no_import_fallback": True,
                "no_engine_fallback": True,
            },
        }

    def export_runtime_summary_json(self) -> str:
        return json.dumps(self.export_runtime_summary(), indent=2, sort_keys=True)
