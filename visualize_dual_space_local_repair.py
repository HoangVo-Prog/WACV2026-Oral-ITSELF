#!/usr/bin/env python3
"""Local repair diagnostics for visual and text prototype spaces.

Raw cross-modal embeddings are not directly mixed. Visual-space translated
evidence is built only through ``text_to_image``; text-space translated evidence
is built only through ``image_to_text``.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from plot_ambiguity_rate_compare import (  # type: ignore
        SplitData,
        build_eval_transforms,
        build_repo_model,
        candidate_state_keys,
        checkpoint_state_dict,
        default_model_args,
        image_dataset_class,
        load_split_data,
        parse_img_size,
        text_dataset_class,
        torch_load_checkpoint,
        validate_split_data,
    )
except ModuleNotFoundError:
    from scripts.plot_ambiguity_rate_compare import (  # type: ignore
        SplitData,
        build_eval_transforms,
        build_repo_model,
        candidate_state_keys,
        checkpoint_state_dict,
        default_model_args,
        image_dataset_class,
        load_split_data,
        parse_img_size,
        text_dataset_class,
        torch_load_checkpoint,
        validate_split_data,
    )



TITLE = "Local Embedding Repair around Identity-Level Ambiguity"
MEMORY_NAMES = (
    "image_prototypes",
    "text_prototypes",
    "text_to_image",
    "image_to_text",
    "proto_pids",
    "initialized",
)
PROTOTYPE_KEY_TERMS = ("proto", "prototype", "memory", "bank", "pid", "pbt", "text_to", "image_to")
POS_COLOR = "#1B9E77"
HARD_NEG_COLOR = "#D95F02"
PROTO_EDGE = "#222222"
TRANSLATED_COLOR = "#4C78A8"
POS_MOTION_COLOR = "#007A55"
NEG_MOTION_COLOR = "#C33A2C"


@dataclass
class PrototypeBankData:
    image_prototypes: torch.Tensor
    text_prototypes: torch.Tensor
    text_to_image: torch.Tensor
    image_to_text: torch.Tensor
    proto_pids: torch.Tensor
    initialized: bool
    source_path: str
    key_map: Dict[str, str]
    config: Dict[str, Any]
    tensor_summary: Dict[str, List[int]]

    @property
    def dim(self) -> int:
        return int(self.image_prototypes.shape[1])

    @property
    def total(self) -> int:
        return int(self.image_prototypes.shape[0])

    @property
    def unique_pids(self) -> List[int]:
        return sorted(int(x) for x in torch.unique(self.proto_pids).cpu().tolist())

    @property
    def prototypes_per_id(self) -> int:
        cfg_k = scalar_int(self.config.get("prototypes_per_id")) or scalar_int(self.config.get("prototype_per_id"))
        if cfg_k and cfg_k > 0 and self.total % cfg_k == 0:
            return int(cfg_k)
        counts = Counter(int(x) for x in self.proto_pids.cpu().tolist())
        if counts:
            return int(max(counts.values()))
        return 1

    @property
    def num_classes(self) -> int:
        k = max(self.prototypes_per_id, 1)
        if self.total % k == 0:
            return int(self.total // k)
        return max(int(self.proto_pids.max().item()) + 1, len(self.unique_pids), 1)


@dataclass
class CheckpointLoadReport:
    path: str
    loaded: int
    skipped_memory: int
    skipped_missing: int
    skipped_shape: int
    skipped_non_tensor: int
    loaded_keys: List[str] = field(default_factory=list)
    missing_examples: List[Dict[str, Any]] = field(default_factory=list)
    shape_examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "loaded": self.loaded,
            "skipped_memory": self.skipped_memory,
            "skipped_missing": self.skipped_missing,
            "skipped_shape": self.skipped_shape,
            "skipped_non_tensor": self.skipped_non_tensor,
            "loaded_keys": self.loaded_keys,
            "missing_examples": self.missing_examples,
            "shape_examples": self.shape_examples,
        }


@dataclass
class ProjectedBundle:
    text_features: torch.Tensor
    image_features: torch.Tensor
    query_pids: torch.Tensor
    gallery_pids: torch.Tensor
    feature_source: str
    projection_source: str


@dataclass
class RetrievalBundle:
    text_global: Optional[torch.Tensor]
    image_global: Optional[torch.Tensor]
    text_grab: Optional[torch.Tensor]
    image_grab: Optional[torch.Tensor]
    query_pids: torch.Tensor
    gallery_pids: torch.Tensor
    mode: str
    alpha: float

    def similarity(self) -> torch.Tensor:
        if self.mode == "global":
            assert self.text_global is not None and self.image_global is not None
            return self.text_global @ self.image_global.t()
        if self.mode == "grab":
            assert self.text_grab is not None and self.image_grab is not None
            return self.text_grab @ self.image_grab.t()
        if self.mode == "global+grab":
            assert self.text_global is not None and self.image_global is not None
            assert self.text_grab is not None and self.image_grab is not None
            return self.alpha * (self.text_global @ self.image_global.t()) + (1.0 - self.alpha) * (
                self.text_grab @ self.image_grab.t()
            )
        if self.mode == "itself":
            assert self.text_global is not None and self.image_global is not None
            assert self.text_grab is not None and self.image_grab is not None
            s_global = self.text_global @ self.image_global.t()
            s_grab = self.text_grab @ self.image_grab.t()
            return self.alpha * s_grab + (1.0 - self.alpha) * s_global
        raise ValueError(f"Unknown retrieval mode: {self.mode}")


@dataclass
class RowContext:
    key: str
    label: str
    projected: ProjectedBundle
    bank: PrototypeBankData
    branch: torch.nn.Module
    bank_source: str
    projection_decision: str
    shared_iapr_anchors: bool


@dataclass
class PointSpec:
    vector: torch.Tensor
    kind: str
    label: str
    marker: str
    color: str
    edgecolor: str = PROTO_EDGE
    size: float = 54.0
    alpha: float = 0.9
    hollow: bool = False
    annotate: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PanelSpec:
    title: str
    points: List[PointSpec]
    arrows: List["ArrowSpec"] = field(default_factory=list)
    metric_text: str = ""


@dataclass
class ArrowSpec:
    start_label: str
    end_label: str
    color: str
    linestyle: str = "-"
    linewidth: float = 1.5
    alpha: float = 0.85
    label: str = ""


@dataclass
class SharedProjectionResult:
    coords_by_label: Dict[str, np.ndarray]
    limits: Tuple[float, float, float, float]
    metadata: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--config", default="", help="Optional path to configs.yaml; defaults are used when omitted.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--host_ckpt", required=True)
    parser.add_argument("--iapr_ckpt", required=True)
    parser.add_argument("--prototype_ckpt", default="")
    parser.add_argument(
        "--prototype_projector_ckpt",
        required=True,
        help="Prototype projector/branch checkpoint, e.g. best_prototype_branch.pth. Pass --iapr_ckpt if projectors are inside it.",
    )
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_cases", type=int, default=5)
    parser.add_argument(
        "--top_pos",
        type=int,
        default=2,
        help="Legacy candidate-selection positive count; plotted positives now use all gallery items with pid == query_pid.",
    )
    parser.add_argument(
        "--top_hard_neg",
        type=int,
        default=2,
        help="Number of hard negative samples to plot per panel.",
    )
    parser.add_argument(
        "--prototype_per_id",
        type=int,
        default=2,
        help="Number of positive prototypes to plot per identity in the IAPR panels.",
    )
    parser.add_argument(
        "--prototype_hard_k",
        type=int,
        default=2,
        help="Number of hard negative prototypes to plot in the IAPR panels.",
    )
    parser.add_argument("--projection", default="pca", choices=["pca", "mds", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--retrieval_branch", default="auto", choices=["auto", "global", "grab", "global+grab", "itself"])
    parser.add_argument(
        "--retrieval_alpha",
        type=float,
        default=0.68,
        help="Legacy global+grab weight: alpha*s_global + (1-alpha)*s_grab when --retrieval_branch global+grab is used.",
    )
    parser.add_argument(
        "--itself",
        action="store_true",
        help="Use ITSELF retrieval/inference scoring for Host/IAPR: lambda*s_grab + (1-lambda)*s_global.",
    )
    parser.add_argument(
        "--itself_lambda",
        "--lambda",
        dest="itself_lambda",
        type=float,
        default=0.68,
        help="ITSELF retrieval weight lambda for s_grab in lambda*s_grab + (1-lambda)*s_global.",
    )
    parser.add_argument("--host_margin_max", type=float, default=0.0)
    parser.add_argument("--min_delta_margin", type=float, default=0.0)
    parser.add_argument("--min_iapr_margin", type=float, default=0.0)
    parser.add_argument("--max_candidate_queries", type=int, default=0)
    parser.add_argument("--pretrain_choice", default="ViT-B/16")
    parser.add_argument("--img_size", default="384,128", help='Input image size as "height,width" when --config is omitted.')
    parser.add_argument("--stride_size", type=int, default=16)
    parser.add_argument("--text_length", type=int, default=77)
    parser.add_argument("--select_ratio", type=float, default=0.4)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def scalar_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base / out
    return out.resolve()


def edict_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items()}
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}



def load_config_file(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyyaml is required to read --config YAML files.") from exc
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Config file must contain a mapping, got {type(data)!r}: {path}")
    return dict(data)
def json_default(value: Any) -> Any:

    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=json_default)
        file.write("\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA was requested but is not available; using CPU.")
        return torch.device("cpu")
    return device


def module_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback


def normalize_tensor(tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return F.normalize(tensor.float(), p=2, dim=dim)


def flatten_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if torch.is_tensor(obj):
        out[prefix.rstrip(".") or "tensor"] = obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            out.update(flatten_tensors(value, f"{prefix}{key}."))
    return out


def checkpoint_config(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, Mapping):
        return {}
    for key in ("prototype_config", "config", "args"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}



def available_prototype_tensors(flat: Mapping[str, torch.Tensor]) -> str:
    rows = []
    for key, value in sorted(flat.items()):
        if any(term in key.lower() for term in PROTOTYPE_KEY_TERMS):
            rows.append(f"  {key}: {tuple(value.shape)}")
    if not rows:
        rows = [f"  {key}: {tuple(value.shape)}" for key, value in sorted(flat.items())[:80]]
    return "\n".join(rows) if rows else "  <no tensors found>"


def choose_memory_tensor(flat: Mapping[str, torch.Tensor], name: str) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    aliases = [
        name,
        f"prototype_bank.{name}",
        f"memory.{name}",
        f"prototype_branch.memory.{name}",
        f"model.prototype_branch.memory.{name}",
        f"module.prototype_branch.memory.{name}",
        f"prototype_branch.prototype_bank.{name}",
    ]
    for alias in aliases:
        if alias in flat:
            return alias, flat[alias]
    matches = sorted(key for key in flat if key.endswith(f".{name}"))
    if matches:
        return matches[0], flat[matches[0]]
    return None, None


def reshape_memory_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    value = value.detach().cpu()
    if name == "initialized":
        return torch.as_tensor(bool(value.item())) if value.numel() == 1 else torch.tensor(True)
    if name == "proto_pids":
        return value.long().reshape(-1)
    if value.ndim == 3:
        return value.reshape(-1, value.shape[-1]).float()
    if value.ndim != 2:
        raise ValueError(f"Prototype memory tensor {name!r} must be 2D or 3D, got shape={tuple(value.shape)}.")
    return value.float()


def infer_proto_pids_from_config(cfg: Mapping[str, Any], total: int) -> Optional[torch.Tensor]:
    num_classes = scalar_int(cfg.get("num_classes"))
    k = scalar_int(cfg.get("prototypes_per_id")) or scalar_int(cfg.get("prototype_per_id"))
    if num_classes and k and num_classes > 0 and k > 0 and num_classes * k == total:
        return torch.arange(num_classes).repeat_interleave(k).long()
    return None


def load_prototype_bank(path: Path, required: bool = True) -> Optional[PrototypeBankData]:
    raw = torch_load_checkpoint(path)
    flat = flatten_tensors(raw)
    cfg = checkpoint_config(raw)
    summary = {key: list(value.shape) for key, value in flat.items()}
    selected: Dict[str, str] = {}
    tensors: Dict[str, torch.Tensor] = {}

    for name in MEMORY_NAMES:
        key, value = choose_memory_tensor(flat, name)
        if value is None:
            if name == "initialized" and isinstance(raw, Mapping) and bool(raw.get("prototype_ready", False)):
                selected[name] = "prototype_ready"
                tensors[name] = torch.tensor(True)
                continue
            if name == "initialized":
                selected[name] = "inferred_true"
                tensors[name] = torch.tensor(True)
                continue
            if name == "proto_pids":
                total_tensor = tensors.get("image_prototypes")
                if total_tensor is None:
                    total_tensor = tensors.get("text_prototypes")
                total = int(total_tensor.shape[0]) if total_tensor is not None else 0
                inferred = infer_proto_pids_from_config(cfg, total)
                if inferred is not None:
                    selected[name] = "inferred_from_prototype_config"
                    tensors[name] = inferred
                    continue
            if not required:
                return None
            raise RuntimeError(
                f"Missing prototype memory tensor {name!r} in {path}.\n"
                f"Available prototype-related tensors:\n{available_prototype_tensors(flat)}"
            )
        selected[name] = str(key)
        tensors[name] = reshape_memory_tensor(name, value)

    image = normalize_tensor(tensors["image_prototypes"], dim=1)
    text = normalize_tensor(tensors["text_prototypes"], dim=1)
    text_to_image = normalize_tensor(tensors["text_to_image"], dim=1)
    image_to_text = normalize_tensor(tensors["image_to_text"], dim=1)
    proto_pids = tensors["proto_pids"].long().reshape(-1)
    initialized = bool(tensors["initialized"].bool().item()) if tensors["initialized"].numel() == 1 else True

    first_shape = tuple(image.shape)
    for name, tensor in {
        "text_prototypes": text,
        "text_to_image": text_to_image,
        "image_to_text": image_to_text,
    }.items():
        if tuple(tensor.shape) != first_shape:
            raise RuntimeError(f"Prototype memory shape mismatch in {path}: {name} has {tuple(tensor.shape)}, expected {first_shape}.")
    if int(proto_pids.numel()) != int(image.shape[0]):
        raise RuntimeError(f"Prototype pid count mismatch in {path}: proto_pids={proto_pids.numel()}, prototypes={image.shape[0]}.")

    return PrototypeBankData(
        image_prototypes=image,
        text_prototypes=text,
        text_to_image=text_to_image,
        image_to_text=image_to_text,
        proto_pids=proto_pids,
        initialized=initialized,
        source_path=str(path),
        key_map=selected,
        config=dict(cfg),
        tensor_summary=summary,
    )


def is_memory_key(key: str) -> bool:
    norm = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "model.", "net.", "network."):
            if norm.startswith(prefix):
                norm = norm[len(prefix):]
                changed = True
    return norm.startswith("prototype_branch.memory.") or norm.startswith("memory.")



def checkpoint_state_for_model_loading(raw: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(raw, Mapping) and isinstance(raw.get("prototype_branch"), Mapping):
        prefixed: Dict[str, torch.Tensor] = {}
        for key, value in raw["prototype_branch"].items():
            key = str(key)
            model_key = key if key.startswith("prototype_branch.") else f"prototype_branch.{key}"
            prefixed[model_key] = value
        return prefixed
    return checkpoint_state_dict(raw)
def load_checkpoint_into_model(model: torch.nn.Module, path: Path, require_update: bool = True) -> CheckpointLoadReport:
    raw = torch_load_checkpoint(path)
    state = checkpoint_state_for_model_loading(raw)
    model_state = model.state_dict()
    update: MutableMapping[str, torch.Tensor] = {}
    skipped_memory = skipped_missing = skipped_shape = skipped_non_tensor = 0
    missing_examples: List[Dict[str, Any]] = []
    shape_examples: List[Dict[str, Any]] = []

    for raw_key, value in state.items():
        raw_key = str(raw_key)
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            continue
        if is_memory_key(raw_key):
            skipped_memory += 1
            continue
        target = next((candidate for candidate in candidate_state_keys(raw_key) if candidate in model_state), None)
        if target is None:
            skipped_missing += 1
            if len(missing_examples) < 30:
                missing_examples.append({"checkpoint_key": raw_key, "checkpoint_shape": list(value.shape)})
            continue
        if target.startswith("prototype_branch.memory."):
            skipped_memory += 1
            continue
        if tuple(model_state[target].shape) != tuple(value.shape):
            skipped_shape += 1
            if len(shape_examples) < 30:
                shape_examples.append(
                    {
                        "checkpoint_key": raw_key,
                        "model_key": target,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(model_state[target].shape),
                    }
                )
            continue
        update[target] = value.detach().clone()

    if require_update and not update:
        raise RuntimeError(f"No compatible model tensors found in checkpoint: {path}")
    if update:
        model_state.update(update)
        model.load_state_dict(model_state, strict=True)
    return CheckpointLoadReport(
        path=str(path),
        loaded=len(update),
        skipped_memory=skipped_memory,
        skipped_missing=skipped_missing,
        skipped_shape=skipped_shape,
        skipped_non_tensor=skipped_non_tensor,
        loaded_keys=sorted(update.keys()),
        missing_examples=missing_examples,
        shape_examples=shape_examples,
    )


def load_memory_into_branch(branch: torch.nn.Module, bank: PrototypeBankData) -> Dict[str, List[int]]:
    memory = getattr(branch, "memory", None)
    if memory is None:
        raise RuntimeError("Prototype branch has no memory module.")
    expected = memory.state_dict()
    state = {
        "image_prototypes": bank.image_prototypes,
        "text_prototypes": bank.text_prototypes,
        "text_to_image": bank.text_to_image,
        "image_to_text": bank.image_to_text,
        "proto_pids": bank.proto_pids,
        "initialized": torch.tensor(bank.initialized),
    }
    shape_report: Dict[str, List[int]] = {}
    for name, expected_tensor in expected.items():
        value = state[name]
        if tuple(value.shape) != tuple(expected_tensor.shape):
            raise RuntimeError(
                f"Prototype bank/model mismatch for {name}: bank shape={tuple(value.shape)}, "
                f"model shape={tuple(expected_tensor.shape)}. Check prototype_dim/prototype_per_id/num_classes."
            )
        state[name] = value.to(dtype=expected_tensor.dtype)
        shape_report[name] = list(value.shape)
    memory.load_state_dict(state, strict=True)
    return shape_report


def projector_param_names(branch: torch.nn.Module) -> Tuple[List[str], List[str]]:
    image_names = [f"prototype_branch.image_projector.{name}" for name, _ in branch.image_projector.named_parameters()]
    text_names = [f"prototype_branch.text_projector.{name}" for name, _ in branch.text_projector.named_parameters()]
    return image_names, text_names


def projector_weights_loaded(branch: torch.nn.Module, report: CheckpointLoadReport) -> bool:
    image_names, text_names = projector_param_names(branch)
    loaded = set(report.loaded_keys)
    image_ok = not image_names or any(name in loaded for name in image_names)
    text_ok = not text_names or any(name in loaded for name in text_names)
    return bool(image_ok and text_ok)


def config_model_args(cli: argparse.Namespace, bank: PrototypeBankData) -> SimpleNamespace:
    cfg = default_model_args()
    if cli.config:
        cfg.update(edict_to_dict(load_config_file(resolve_path(cli.config))))
    cfg["dataset_name"] = cli.dataset_name
    cfg["root_dir"] = str(resolve_path(cli.data_root))
    cfg["training"] = False
    cfg["batch_size"] = int(cli.batch_size)
    cfg["test_batch_size"] = int(cli.batch_size)
    cfg["num_workers"] = int(cli.num_workers)
    if not cli.config:
        cfg["pretrain_choice"] = cli.pretrain_choice
        cfg["stride_size"] = int(cli.stride_size)
        cfg["select_ratio"] = float(cli.select_ratio)
        cfg["text_length"] = int(cli.text_length)
    cfg["img_aug"] = False
    cfg["txt_aug"] = False
    cfg["track_train_diagnostics"] = False
    cfg["prototype"] = True
    cfg["use_loss_id"] = False
    cfg["prototype_dim"] = int(bank.dim)
    cfg["prototype_per_id"] = int(bank.prototypes_per_id)
    cfg["prototype_momentum"] = float(bank.config.get("momentum", cfg.get("prototype_momentum", 0.2)))
    if bank.config.get("projector_mode") is not None:
        cfg["prototype_projector"] = str(bank.config.get("projector_mode"))
    use_itself_retrieval = bool(getattr(cli, "itself", False) or getattr(cli, "retrieval_branch", "auto") == "itself")
    wants_local_retrieval = bool(use_itself_retrieval or getattr(cli, "retrieval_branch", "auto") in {"grab", "global+grab"})
    cfg["only_global"] = not wants_local_retrieval
    cfg["prototype_feature"] = "global"
    cfg["return_all"] = bool(use_itself_retrieval)
    cfg["modify_k"] = bool(use_itself_retrieval)
    cfg["topk_type"] = "custom" if use_itself_retrieval else "mean"
    cfg["average_attn_weights"] = True
    cfg["loss_names"] = "tal+cid"
    cfg["track_train_diagnostics"] = False
    cfg["text_length"] = int(cfg.get("text_length", cli.text_length))
    return SimpleNamespace(**cfg)


def build_loaded_model(
    label: str,
    ckpt_path: Path,
    model_args: SimpleNamespace,
    num_classes: int,
    device: torch.device,
    optional_branch_path: Optional[Path] = None,
) -> Tuple[torch.nn.Module, CheckpointLoadReport, Optional[CheckpointLoadReport]]:
    run_args = SimpleNamespace(**vars(model_args))
    model = build_repo_model(run_args, num_classes=num_classes)
    report = load_checkpoint_into_model(model, ckpt_path, require_update=True)
    aux_report: Optional[CheckpointLoadReport] = None
    branch = getattr(model, "prototype_branch", None)
    if optional_branch_path is not None and branch is not None:
        aux_report = load_checkpoint_into_model(model, optional_branch_path, require_update=False)
        if aux_report.loaded:
            print(f"[{label}] loaded {aux_report.loaded} compatible auxiliary tensors from {optional_branch_path}")
    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model, report, aux_report


def build_vanilla_model(
    label: str,
    model_args: SimpleNamespace,
    num_classes: int,
    device: torch.device,
) -> torch.nn.Module:
    run_args = SimpleNamespace(**vars(model_args))
    model = build_repo_model(run_args, num_classes=num_classes)
    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    print(f"[{label}] no fine-tuned checkpoint loaded")
    return model


def feature_tensor(output: Any, kind: str) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"model.encode_{kind}(...) returned {type(output)!r}, expected a tensor.")
    return output.float()


def has_local_layers(model: torch.nn.Module) -> bool:
    return hasattr(model, "visul_emb_layer") and hasattr(model, "texual_emb_layer")


def require_local_layers(model: torch.nn.Module, label: str) -> None:
    if not has_local_layers(model):
        raise RuntimeError(
            f"{label} needs local GRAB features for prototype-space projection, but the built model has no "
            "visul_emb_layer/texual_emb_layer. Check prototype_feature/only_global architecture settings."
        )


def resolve_retrieval_mode(cli: argparse.Namespace, model: torch.nn.Module, model_args: SimpleNamespace) -> str:
    if bool(getattr(cli, "itself", False)):
        if cli.retrieval_branch not in ("auto", "itself"):
            print(f"[Retrieval] overriding --retrieval_branch={cli.retrieval_branch} to itself because --itself was set")
        mode = "itself"
    elif cli.retrieval_branch != "auto":
        mode = cli.retrieval_branch
    elif bool(getattr(model_args, "only_global", False)) or not has_local_layers(model):
        mode = "global"
    else:
        mode = "global+grab"

    if bool(getattr(model_args, "only_global", False)) and mode != "global":
        print(f"[Retrieval] overriding mode={mode} to global because only_global=True")
        return "global"
    if mode in ("grab", "global+grab", "itself"):
        require_local_layers(model, "Retrieval scoring")
    return mode



@torch.inference_mode()
def extract_text_kind(
    model: torch.nn.Module,
    split_data: SplitData,
    text_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    kind: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    TextDataset = text_dataset_class()
    text_set = TextDataset(split_data.caption_pids, split_data.captions, text_length=text_length)
    loader = DataLoader(text_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []
    model.eval()
    for pid, tokens in tqdm(loader, desc=desc):
        tokens = tokens.to(device, non_blocking=True)
        if kind == "global":
            feats = feature_tensor(model.encode_text(tokens), "text")
        elif kind == "grab":
            require_local_layers(model, desc)
            feats = feature_tensor(model.encode_text_grab(tokens), "text_grab")
        else:
            raise ValueError(f"Unknown text feature kind: {kind}")
        features.append(F.normalize(feats.float(), p=2, dim=1).cpu())
        pids.append(pid.view(-1).cpu().long())
    if not features:
        raise RuntimeError("No text features were extracted.")
    return torch.cat(features, dim=0), torch.cat(pids, dim=0).long()


@torch.inference_mode()
def extract_image_kind(
    model: torch.nn.Module,
    split_data: SplitData,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    kind: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ImageDataset = image_dataset_class()
    image_set = ImageDataset(split_data.image_pids, split_data.img_paths, transform=build_eval_transforms(img_size))
    loader = DataLoader(image_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []
    model.eval()
    for pid, images in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        if kind == "global":
            feats = feature_tensor(model.encode_image(images), "image")
        elif kind == "grab":
            require_local_layers(model, desc)
            feats = feature_tensor(model.encode_image_grab(images), "image_grab")
        else:
            raise ValueError(f"Unknown image feature kind: {kind}")
        features.append(F.normalize(feats.float(), p=2, dim=1).cpu())
        pids.append(pid.view(-1).cpu().long())
    if not features:
        raise RuntimeError("No image features were extracted.")
    return torch.cat(features, dim=0), torch.cat(pids, dim=0).long()


@torch.inference_mode()
def extract_retrieval_bundle(
    label: str,
    model: torch.nn.Module,
    split_data: SplitData,
    model_args: SimpleNamespace,
    mode: str,
    alpha: float,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> RetrievalBundle:
    text_global = image_global = text_grab = image_grab = None
    query_pids: Optional[torch.Tensor] = None
    gallery_pids: Optional[torch.Tensor] = None
    img_size = parse_img_size(model_args.img_size)
    text_length = int(model_args.text_length)
    if mode in ("global", "global+grab", "itself"):
        text_global, query_pids = extract_text_kind(model, split_data, text_length, batch_size, num_workers, device, "global", f"{label} text global")
        image_global, gallery_pids = extract_image_kind(model, split_data, img_size, batch_size, num_workers, device, "global", f"{label} image global")
    if mode in ("grab", "global+grab", "itself"):
        text_grab, query_pids_grab = extract_text_kind(model, split_data, text_length, batch_size, num_workers, device, "grab", f"{label} text GRAB")
        image_grab, gallery_pids_grab = extract_image_kind(model, split_data, img_size, batch_size, num_workers, device, "grab", f"{label} image GRAB")
        if query_pids is None:
            query_pids = query_pids_grab
        elif not torch.equal(query_pids, query_pids_grab):
            raise RuntimeError(f"{label} global and GRAB query pid arrays differ.")
        if gallery_pids is None:
            gallery_pids = gallery_pids_grab
        elif not torch.equal(gallery_pids, gallery_pids_grab):
            raise RuntimeError(f"{label} global and GRAB gallery pid arrays differ.")
    assert query_pids is not None and gallery_pids is not None
    return RetrievalBundle(text_global, image_global, text_grab, image_grab, query_pids, gallery_pids, mode, alpha)


def project_with_module(projector: torch.nn.Module, features: torch.Tensor, fallback_device: torch.device) -> torch.Tensor:
    projector.float()
    device = module_device(projector, fallback_device)
    projected = projector(features.to(device=device, dtype=torch.float32, non_blocking=True))
    return F.normalize(projected.float(), p=2, dim=1)


@torch.inference_mode()
def extract_projected_text(
    raw_model: torch.nn.Module,
    branch: torch.nn.Module,
    split_data: SplitData,
    text_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    feature_kind: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    TextDataset = text_dataset_class()
    text_set = TextDataset(split_data.caption_pids, split_data.captions, text_length=text_length)
    loader = DataLoader(text_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []
    for pid, tokens in tqdm(loader, desc=desc):
        tokens = tokens.to(device, non_blocking=True)
        if feature_kind == "local":
            require_local_layers(raw_model, desc)
            raw = feature_tensor(raw_model.encode_text_grab(tokens), "text_grab")
        else:
            raw = feature_tensor(raw_model.encode_text(tokens), "text")
        projected = project_with_module(branch.text_projector, raw, device)
        features.append(projected.cpu())
        pids.append(pid.view(-1).cpu().long())
    if not features:
        raise RuntimeError("No projected text features were extracted.")
    return torch.cat(features, dim=0), torch.cat(pids, dim=0).long()


@torch.inference_mode()
def extract_projected_image(
    raw_model: torch.nn.Module,
    branch: torch.nn.Module,
    split_data: SplitData,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    feature_kind: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ImageDataset = image_dataset_class()
    image_set = ImageDataset(split_data.image_pids, split_data.img_paths, transform=build_eval_transforms(img_size))
    loader = DataLoader(image_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []
    for pid, images in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        if feature_kind == "local":
            require_local_layers(raw_model, desc)
            raw = feature_tensor(raw_model.encode_image_grab(images), "image_grab")
        else:
            raw = feature_tensor(raw_model.encode_image(images), "image")
        projected = project_with_module(branch.image_projector, raw, device)
        features.append(projected.cpu())
        pids.append(pid.view(-1).cpu().long())
    if not features:
        raise RuntimeError("No projected image features were extracted.")
    return torch.cat(features, dim=0), torch.cat(pids, dim=0).long()


@torch.inference_mode()
def extract_projected_bundle(
    label: str,
    raw_model: torch.nn.Module,
    branch: torch.nn.Module,
    split_data: SplitData,
    model_args: SimpleNamespace,
    projection_source: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> ProjectedBundle:
    feature_kind = "local" if bool(getattr(branch, "use_local", False)) else "global"
    text_features, query_pids = extract_projected_text(
        raw_model,
        branch,
        split_data,
        int(model_args.text_length),
        batch_size,
        num_workers,
        device,
        feature_kind,
        f"{label} projected text ({feature_kind})",
    )
    image_features, gallery_pids = extract_projected_image(
        raw_model,
        branch,
        split_data,
        parse_img_size(model_args.img_size),
        batch_size,
        num_workers,
        device,
        feature_kind,
        f"{label} projected image ({feature_kind})",
    )
    return ProjectedBundle(text_features, image_features, query_pids, gallery_pids, feature_kind, projection_source)


def topk_from_mask(scores: torch.Tensor, mask: torch.Tensor, k: int) -> List[int]:
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0 or k <= 0:
        return []
    k = min(int(k), int(indices.numel()))
    local_scores = scores[indices]
    order = torch.argsort(local_scores, descending=True)[:k]
    return [int(indices[int(i)].item()) for i in order]


def topk_all(scores: torch.Tensor, k: int) -> List[int]:
    if scores.numel() == 0 or k <= 0:
        return []
    k = min(int(k), int(scores.numel()))
    order = torch.argsort(scores, descending=True)[:k]
    return [int(index.item()) for index in order]


def score_extrema(scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> Tuple[float, float, float, int, int]:
    pos_scores = scores.masked_fill(~pos_mask, float("-inf"))
    neg_scores = scores.masked_fill(~neg_mask, float("-inf"))
    pos_score, pos_idx = torch.max(pos_scores, dim=0)
    neg_score, neg_idx = torch.max(neg_scores, dim=0)
    margin = pos_score - neg_score
    return float(pos_score.item()), float(neg_score.item()), float(margin.item()), int(pos_idx.item()), int(neg_idx.item())


def pid_has_slots(bank: PrototypeBankData, pid: int) -> bool:
    return bool(torch.any(bank.proto_pids.eq(int(pid))).item())


def pid_slot_indices(bank: PrototypeBankData, pid: int) -> List[Tuple[int, int]]:
    idxs = torch.nonzero(bank.proto_pids.eq(int(pid)), as_tuple=False).flatten().cpu().tolist()
    return [(slot, int(index)) for slot, index in enumerate(idxs)]


def assign_to_pid_slots(feature: torch.Tensor, bank_vectors: torch.Tensor, bank: PrototypeBankData, pid: int) -> Dict[str, Any]:
    slots = pid_slot_indices(bank, pid)
    if not slots:
        raise KeyError(f"No prototype slots are available for pid={pid} in {bank.source_path}")
    global_indices = torch.tensor([global_idx for _slot, global_idx in slots], dtype=torch.long)
    local_vectors = F.normalize(bank_vectors[global_indices].float(), p=2, dim=1)
    feature = F.normalize(feature.detach().float().view(1, -1), p=2, dim=1)
    sims = (feature @ local_vectors.t()).flatten()
    best = int(torch.argmax(sims).item())
    return {
        "pid": int(pid),
        "local_slot": int(slots[best][0]),
        "global_index": int(slots[best][1]),
        "similarity": float(sims[best].item()),
        "slot_similarities": [float(x) for x in sims.cpu().tolist()],
    }


def truncate_text(text: str, chars: int = 150) -> str:
    clean = " ".join(str(text).split())
    if len(clean) <= chars:
        return clean
    return clean[: max(chars - 3, 1)].rstrip() + "..."



def vector_unique_count(vectors: np.ndarray) -> int:
    if vectors.size == 0:
        return 0
    rounded = np.round(vectors.astype(np.float64), decimals=8)
    return int(np.unique(rounded, axis=0).shape[0])


def local_project(points: Sequence[PointSpec], method: str) -> np.ndarray:
    vectors = torch.stack([p.vector.detach().float().cpu().view(-1) for p in points], dim=0)
    vectors = F.normalize(vectors, p=2, dim=1).numpy().astype(np.float64)
    if vector_unique_count(vectors) < 3:
        raise ValueError("too few unique points for a stable local 2D projection")
    if method == "umap":
        try:
            import umap  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("--projection umap was requested, but umap is not installed in this environment.") from exc
        n_neighbors = min(8, max(2, vectors.shape[0] - 1))
        projected = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42).fit_transform(vectors)
        return np.asarray(projected, dtype=np.float64)
    if method == "mds":
        cosine_distance = np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)
        squared = cosine_distance ** 2
        n = vectors.shape[0]
        centering = np.eye(n) - np.ones((n, n), dtype=np.float64) / float(n)
        gram = -0.5 * centering @ squared @ centering
        eigvals, eigvecs = np.linalg.eigh(gram)
        order = np.argsort(eigvals)[::-1][:2]
        scales = np.sqrt(np.maximum(eigvals[order], 0.0))
        projected = eigvecs[:, order] * scales.reshape(1, -1)
        if projected.shape[1] < 2:
            projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])), mode="constant")
        return np.asarray(projected[:, :2], dtype=np.float64)
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    try:
        from sklearn.decomposition import PCA  # type: ignore

        projected = PCA(n_components=2).fit_transform(vectors)
    except Exception:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].T
        projected = centered @ components
    if projected.shape[1] < 2:
        projected = np.pad(projected, ((0, 0), (0, 2 - projected.shape[1])), mode="constant")
    return np.asarray(projected[:, :2], dtype=np.float64)


def unique_points_by_label(points: Sequence[PointSpec]) -> List[PointSpec]:
    unique: List[PointSpec] = []
    vectors_by_label: Dict[str, torch.Tensor] = {}
    for point in points:
        vector = point.vector.detach().float().cpu().view(-1)
        previous = vectors_by_label.get(point.label)
        if previous is not None:
            if previous.shape != vector.shape or not torch.allclose(previous, vector, rtol=1e-4, atol=1e-5):
                raise ValueError(f"point label {point.label!r} is reused for different vectors")
            continue
        vectors_by_label[point.label] = vector
        unique.append(point)
    return unique


def coordinate_limits(coords: np.ndarray) -> Tuple[float, float, float, float]:
    if coords.size == 0:
        return -1.0, 1.0, -1.0, 1.0
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = span * 0.16
    return (
        float(mins[0] - pad[0]),
        float(maxs[0] + pad[0]),
        float(mins[1] - pad[1]),
        float(maxs[1] + pad[1]),
    )


def apply_limits(ax: Any, limits: Tuple[float, float, float, float]) -> None:
    ax.set_xlim(float(limits[0]), float(limits[1]))
    ax.set_ylim(float(limits[2]), float(limits[3]))


def fit_shared_projection(points: Sequence[PointSpec], method: str, scope: str) -> SharedProjectionResult:
    unique_points = unique_points_by_label(points)
    coords = local_project(unique_points, method)
    coords_by_label = {point.label: coords[index] for index, point in enumerate(unique_points)}
    raw_vectors = np.stack([point.vector.detach().float().cpu().numpy() for point in unique_points], axis=0)
    return SharedProjectionResult(
        coords_by_label=coords_by_label,
        limits=coordinate_limits(coords),
        metadata={
            "scope": scope,
            "method": method,
            "num_points": int(len(unique_points)),
            "unique_points": int(vector_unique_count(raw_vectors)),
            "point_labels": [point.label for point in unique_points],
            "axis_limits": {
                "x": [float(coordinate_limits(coords)[0]), float(coordinate_limits(coords)[1])],
                "y": [float(coordinate_limits(coords)[2]), float(coordinate_limits(coords)[3])],
            },
        },
    )


def scatter_point(ax: Any, point: PointSpec, xy: np.ndarray, alpha_override: Optional[float] = None) -> None:
    alpha = float(point.alpha if alpha_override is None else alpha_override)
    if point.marker == "x":
        ax.scatter([xy[0]], [xy[1]], marker="x", s=point.size, c=point.color, linewidths=1.45, alpha=alpha, zorder=3)
    elif point.hollow:
        ax.scatter(
            [xy[0]],
            [xy[1]],
            marker=point.marker,
            s=point.size,
            facecolors="none",
            edgecolors=point.color,
            linewidths=1.25,
            alpha=alpha,
            zorder=3,
        )
    else:
        ax.scatter(
            [xy[0]],
            [xy[1]],
            marker=point.marker,
            s=point.size,
            c=point.color,
            edgecolors=point.edgecolor if point.marker not in {"*"} else "black",
            linewidths=0.7,
            alpha=alpha,
            zorder=3,
        )
    if point.annotate:
        ax.annotate(
            point.annotate,
            (xy[0], xy[1]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.0,
            color="#333333",
            zorder=4,
        )


def compute_overlap_alphas(points: Sequence[PointSpec], coords: np.ndarray, limits: Tuple[float, float, float, float]) -> List[float]:
    alphas = [float(point.alpha) for point in points]
    if len(points) < 2 or coords.size == 0:
        return alphas
    x_span = max(float(limits[1] - limits[0]), 1e-6)
    y_span = max(float(limits[3] - limits[2]), 1e-6)
    scaled = coords.astype(np.float64).copy()
    scaled[:, 0] = (scaled[:, 0] - float(limits[0])) / x_span
    scaled[:, 1] = (scaled[:, 1] - float(limits[2])) / y_span
    close_counts = [0 for _ in points]
    threshold = 0.045
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = float(np.linalg.norm(scaled[i] - scaled[j]))
            if distance < threshold:
                close_counts[i] += 1
                close_counts[j] += 1
    adjusted: List[float] = []
    for point, alpha, count in zip(points, alphas, close_counts):
        if count <= 0:
            adjusted.append(alpha)
            continue
        role = str(point.metadata.get("role", ""))
        if point.metadata.get("ghost_reference") == "vanilla":
            adjusted.append(max(0.12, min(alpha, 0.22)))
        elif role in {"positive_sample", "hard_negative_sample"}:
            adjusted.append(max(0.16, min(alpha, alpha * max(0.38, 1.0 - 0.16 * count))))
        elif bool(point.metadata.get("motion_annotation")):
            adjusted.append(max(0.36, min(alpha, alpha * max(0.70, 1.0 - 0.08 * count))))
        elif role in {"translated_centroid", "positive_prototype", "negative_prototype"} or "centroid" in point.kind:
            adjusted.append(max(0.54, alpha * max(0.72, 1.0 - 0.08 * count)))
        else:
            adjusted.append(max(0.46, alpha * max(0.60, 1.0 - 0.13 * count)))
    return adjusted


def draw_arrow(ax: Any, arrow: ArrowSpec, coords_by_label: Mapping[str, np.ndarray]) -> None:
    if arrow.start_label not in coords_by_label or arrow.end_label not in coords_by_label:
        return
    start = coords_by_label[arrow.start_label]
    end = coords_by_label[arrow.end_label]
    ax.annotate(
        "",
        xy=(float(end[0]), float(end[1])),
        xytext=(float(start[0]), float(start[1])),
        arrowprops={
            "arrowstyle": "->",
            "color": arrow.color,
            "lw": arrow.linewidth,
            "linestyle": arrow.linestyle,
            "alpha": arrow.alpha,
            "shrinkA": 4,
            "shrinkB": 5,
        },
        zorder=1,
    )


def panel_points(panels: Sequence[PanelSpec]) -> List[PointSpec]:
    points: List[PointSpec] = []
    for panel in panels:
        points.extend(panel.points)
    return points


def draw_panel_with_shared_projection(ax: Any, panel: PanelSpec, projection: SharedProjectionResult) -> Dict[str, Any]:
    coords_for_panel: List[np.ndarray] = []
    for point in panel.points:
        if point.label not in projection.coords_by_label:
            raise KeyError(f"point {point.label!r} is missing from the shared projection")
        coords_for_panel.append(projection.coords_by_label[point.label])
    coords = np.asarray(coords_for_panel, dtype=np.float64) if coords_for_panel else np.zeros((0, 2), dtype=np.float64)
    coords_by_label = {point.label: projection.coords_by_label[point.label] for point in panel.points}
    overlap_alphas = compute_overlap_alphas(panel.points, coords, projection.limits)
    for arrow in panel.arrows:
        draw_arrow(ax, arrow, coords_by_label)
    for point, xy, alpha in zip(panel.points, coords, overlap_alphas):
        scatter_point(ax, point, xy, alpha_override=alpha)
    ax.set_title(panel.title, fontsize=9.4, fontweight="bold", pad=3.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    apply_limits(ax, projection.limits)
    return {
        "title": panel.title,
        "num_points": int(len(panel.points)),
        "point_labels": [point.label for point in panel.points],
        "arrows": [arrow.__dict__ for arrow in panel.arrows],
        "metric_text": panel.metric_text,
    }


def legend_handles() -> Tuple[List[Any], List[str]]:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", label="positive samples", markerfacecolor=POS_COLOR, markeredgecolor=POS_COLOR, markersize=5.2, alpha=0.58),
        Line2D([0], [0], marker="x", color=HARD_NEG_COLOR, label="hard negative samples", markersize=5.7, linestyle="None", markeredgewidth=1.4, alpha=0.78),
        Line2D([0], [0], marker="*", color="w", label="translated centroid (IAPR)", markerfacecolor=TRANSLATED_COLOR, markeredgecolor="#111111", markersize=8.8),
        Line2D([0], [0], marker="D", color="w", label="positive prototypes (IAPR)", markerfacecolor="none", markeredgecolor=POS_COLOR, markersize=6.0),
        Line2D([0], [0], marker="^", color="w", label="negative prototypes (IAPR)", markerfacecolor="none", markeredgecolor=HARD_NEG_COLOR, markersize=6.0),
        Line2D([0], [0], marker="o", color="#999999", label="group centroid marker", markerfacecolor="#FFFFFF", markeredgecolor="#777777", markersize=5.4, alpha=0.72),
        Line2D([0], [0], color=POS_MOTION_COLOR, label="Vanilla -> Host positive group", linestyle="--", linewidth=1.35),
        Line2D([0], [0], color=POS_MOTION_COLOR, label="Vanilla -> IAPR positive group", linestyle=":", linewidth=1.55),
        Line2D([0], [0], color=NEG_MOTION_COLOR, label="Vanilla -> Host hard-negative group", linestyle="-.", linewidth=1.35),
        Line2D([0], [0], color=NEG_MOTION_COLOR, label="Vanilla -> IAPR hard-negative group", linestyle="-", linewidth=1.20),
    ]
    return handles, [h.get_label() for h in handles]

def footer_metric_line(model_label: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"{model_label}: "
        f"pos={float(metrics['positive_score']):.3f}  "
        f"neg={float(metrics['hard_negative_score']):.3f}  "
        f"m={float(metrics['margin']):+.3f}"
    )


def draw_footer_metrics(fig: Any, metric_grid: Any, metrics_by_model: Mapping[str, Mapping[str, Any]]) -> List[str]:
    order = [("vanilla", "Vanilla CLIP"), ("host", "Host"), ("iapr", "IAPR")]
    lines: List[str] = []
    for col, (key, label) in enumerate(order):
        ax = fig.add_subplot(metric_grid[0, col])
        ax.axis("off")
        line = footer_metric_line(label, metrics_by_model[key])
        lines.append(line)
        ax.text(
            0.5,
            0.52,
            line,
            ha="center",
            va="center",
            fontsize=6.9,
            family="monospace",
            color="#333333",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FAFAFA", "edgecolor": "#DDDDDD", "alpha": 0.82},
        )
    return lines


def draw_footer_legend(fig: Any, legend_spec: Any) -> None:
    ax = fig.add_subplot(legend_spec)
    ax.axis("off")
    handles, labels_text = legend_handles()
    ax.legend(
        handles,
        labels_text,
        loc="center",
        ncol=5,
        frameon=False,
        fontsize=6.3,
        handlelength=1.35,
        handletextpad=0.38,
        columnspacing=0.95,
        borderpad=0.0,
        labelspacing=0.15,
    )


def normalized_centroid(vectors: torch.Tensor) -> torch.Tensor:
    if vectors.numel() == 0:
        raise ValueError("cannot compute a centroid from an empty tensor")
    if vectors.ndim == 1:
        vectors = vectors.view(1, -1)
    centroid = vectors.detach().float().mean(dim=0, keepdim=True)
    return F.normalize(centroid, p=2, dim=1).squeeze(0).cpu()


def feature_centroid(features: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if not indices:
        raise ValueError("cannot compute a centroid without selected sample indices")
    return normalized_centroid(features[[int(index) for index in indices]])


def retrieval_metrics_for_query(
    sim: torch.Tensor,
    query_index: int,
    query_pid: int,
    gallery_pids: torch.Tensor,
) -> Dict[str, Any]:
    pos_mask = gallery_pids.eq(int(query_pid))
    neg_mask = ~pos_mask
    pos, neg, margin, pos_idx, neg_idx = score_extrema(sim[int(query_index)], pos_mask, neg_mask)
    return {
        "positive_score": float(pos),
        "hard_negative_score": float(neg),
        "margin": float(margin),
        "best_positive_index": int(pos_idx),
        "hard_negative_index": int(neg_idx),
        "hard_negative_pid": int(gallery_pids[int(neg_idx)].item()),
    }


def rank1_summary(sim: torch.Tensor, query_pids: torch.Tensor, gallery_pids: torch.Tensor) -> Dict[str, Any]:
    if sim.ndim != 2:
        raise ValueError(f"R1 expects a 2D similarity matrix, got shape={tuple(sim.shape)}")
    if int(sim.shape[0]) != int(query_pids.numel()):
        raise ValueError("R1 similarity/query pid count mismatch")
    if int(sim.shape[1]) != int(gallery_pids.numel()):
        raise ValueError("R1 similarity/gallery pid count mismatch")
    if sim.numel() == 0 or int(query_pids.numel()) == 0:
        return {"R1": 0.0, "r1_percent": 0.0, "hits": 0, "num_queries": 0}
    top_indices = torch.argmax(sim.float().cpu(), dim=1)
    top_pids = gallery_pids.cpu().long()[top_indices]
    hits = top_pids.eq(query_pids.cpu().long())
    hit_count = int(hits.sum().item())
    num_queries = int(hits.numel())
    r1 = 100.0 * float(hit_count) / float(max(num_queries, 1))
    return {"R1": r1, "r1_percent": r1, "hits": hit_count, "num_queries": num_queries}


def print_rank1_summaries(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    print("[R1 before plotting]")
    for key in ("vanilla", "host", "iapr"):
        row = metrics[key]
        print(f"  {key.capitalize():<8} R1={float(row['R1']):6.2f}% ({int(row['hits'])}/{int(row['num_queries'])})")


def model_context_sequence(
    vanilla_ctx: RowContext,
    host_ctx: RowContext,
    iapr_ctx: RowContext,
) -> List[Tuple[str, str, RowContext]]:
    return [
        ("vanilla", "Vanilla CLIP", vanilla_ctx),
        ("host", "Host", host_ctx),
        ("iapr", "IAPR", iapr_ctx),
    ]


def collect_positive_image_indices(gallery_pids: torch.Tensor, pid: int) -> List[int]:
    return [int(index.item()) for index in torch.nonzero(gallery_pids.eq(int(pid)), as_tuple=False).flatten()]


def collect_positive_text_indices(text_pids: torch.Tensor, pid: int) -> List[int]:
    return [int(index.item()) for index in torch.nonzero(text_pids.eq(int(pid)), as_tuple=False).flatten()]


def select_hard_negative_image_indices(
    query_index: int,
    query_pid: int,
    gallery_pids: torch.Tensor,
    host_similarity: torch.Tensor,
    top_k: int,
) -> List[int]:
    mask = ~gallery_pids.eq(int(query_pid))
    return topk_from_mask(host_similarity[int(query_index)], mask, int(top_k))


def select_hard_negative_text_indices(
    text_features: torch.Tensor,
    text_pids: torch.Tensor,
    query_pid: int,
    reference: torch.Tensor,
    top_k: int,
) -> Tuple[List[int], torch.Tensor]:
    reference = F.normalize(reference.detach().float().view(1, -1), p=2, dim=1).cpu()
    features = F.normalize(text_features.detach().float().cpu(), p=2, dim=1)
    scores = (features @ reference.t()).flatten()
    mask = ~text_pids.cpu().long().eq(int(query_pid))
    return topk_from_mask(scores, mask, int(top_k)), scores


def prototype_slot_record(bank: PrototypeBankData, global_index: int, similarity: Optional[float] = None) -> Dict[str, Any]:
    global_index = int(global_index)
    pid = int(bank.proto_pids[global_index].item())
    local_slot = 0
    for slot, slot_global in pid_slot_indices(bank, pid):
        if int(slot_global) == global_index:
            local_slot = int(slot)
            break
    record: Dict[str, Any] = {"pid": pid, "local_slot": local_slot, "global_index": global_index}
    if similarity is not None:
        record["similarity_to_translated_centroid"] = float(similarity)
    return record


def positive_prototype_records(bank: PrototypeBankData, query_pid: int, max_slots: int) -> List[Dict[str, Any]]:
    slots = pid_slot_indices(bank, int(query_pid))[: max(int(max_slots), 0)]
    return [prototype_slot_record(bank, global_idx) for _local_slot, global_idx in slots]


def select_hard_negative_prototypes(
    reference: torch.Tensor,
    prototype_vectors: torch.Tensor,
    bank: PrototypeBankData,
    query_pid: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    if int(top_k) <= 0:
        return []
    reference_n = F.normalize(reference.detach().float().view(1, -1), p=2, dim=1).cpu()
    prototypes_n = F.normalize(prototype_vectors.detach().float().cpu(), p=2, dim=1)
    scores = (prototypes_n @ reference_n.t()).flatten()
    mask = ~bank.proto_pids.cpu().long().eq(int(query_pid))
    indices = topk_from_mask(scores, mask, int(top_k))
    return [prototype_slot_record(bank, index, float(scores[int(index)].item())) for index in indices]


def build_text_translated_centroid(
    iapr_ctx: RowContext,
    bank: PrototypeBankData,
    query_pid: int,
    positive_text_indices: Sequence[int],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # Text evidence enters visual space only through text prototype assignment and text_to_image translation.
    anchors: List[torch.Tensor] = []
    assignments: Dict[str, Any] = {}
    for text_index in positive_text_indices:
        assignment = assign_to_pid_slots(iapr_ctx.projected.text_features[int(text_index)], bank.text_prototypes, bank, int(query_pid))
        assignments[str(int(text_index))] = assignment
        anchors.append(bank.text_to_image[int(assignment["global_index"])])
    if not anchors:
        raise ValueError("no positive text samples are available for the text-translated visual centroid")
    return normalized_centroid(torch.stack(anchors, dim=0)), assignments


def build_image_translated_centroid(
    iapr_ctx: RowContext,
    bank: PrototypeBankData,
    query_pid: int,
    positive_image_indices: Sequence[int],
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # Image evidence enters text space only through image prototype assignment and image_to_text translation.
    anchors: List[torch.Tensor] = []
    assignments: Dict[str, Any] = {}
    for image_index in positive_image_indices:
        assignment = assign_to_pid_slots(iapr_ctx.projected.image_features[int(image_index)], bank.image_prototypes, bank, int(query_pid))
        assignments[str(int(image_index))] = assignment
        anchors.append(bank.image_to_text[int(assignment["global_index"])])
    if not anchors:
        raise ValueError("no positive image samples are available for the image-translated text centroid")
    return normalized_centroid(torch.stack(anchors, dim=0)), assignments


def sample_point(
    vector: torch.Tensor,
    space_key: str,
    model_key: str,
    group: str,
    sample_index: int,
    pid: int,
    rank: int,
) -> PointSpec:
    is_positive = group == "positive"
    index_key = "gallery_index" if space_key == "visual" else "text_index"
    return PointSpec(
        vector,
        f"{group}_{space_key}_sample",
        f"{space_key}_{model_key}_{group}_sample_{rank}_{sample_index}",
        "o" if is_positive else "x",
        POS_COLOR if is_positive else HARD_NEG_COLOR,
        edgecolor=POS_COLOR if is_positive else HARD_NEG_COLOR,
        size=34 if is_positive else 45,
        alpha=0.48 if is_positive else 0.66,
        metadata={
            "role": "positive_sample" if is_positive else "hard_negative_sample",
            "group": group,
            index_key: int(sample_index),
            "pid": int(pid),
            "rank": int(rank),
        },
    )


def motion_centroid_point(vector: torch.Tensor, space_key: str, model_key: str, group: str, ghost: bool = False) -> PointSpec:
    is_positive = group == "positive"
    label_model = "vanilla" if ghost else model_key
    return PointSpec(
        vector,
        f"{group}_{space_key}_group_centroid",
        f"{space_key}_{label_model}_{group}_group_centroid",
        "o" if is_positive else "s",
        "#8F8F8F" if ghost else (POS_MOTION_COLOR if is_positive else NEG_MOTION_COLOR),
        edgecolor="#8F8F8F" if ghost else (POS_MOTION_COLOR if is_positive else NEG_MOTION_COLOR),
        size=76 if is_positive else 84,
        alpha=0.20 if ghost else 0.48,
        hollow=True,
        annotate="" if ghost else ("P" if is_positive else "HN"),
        metadata={
            "role": "motion_positive_group_centroid" if is_positive else "motion_hard_negative_group_centroid",
            "group": group,
            "model": label_model,
            "motion_annotation": True,
            "ghost_reference": "vanilla" if ghost else None,
        },
    )


def translated_centroid_point(vector: torch.Tensor, space_key: str) -> PointSpec:
    kind = "text_translated_centroid" if space_key == "visual" else "image_translated_centroid"
    return PointSpec(
        vector,
        kind,
        f"{space_key}_iapr_translated_centroid",
        "*",
        TRANSLATED_COLOR,
        size=170,
        alpha=0.92,
        annotate="translated",
        metadata={"role": "translated_centroid", "model": "iapr", "space": space_key},
    )


def prototype_point(
    vector: torch.Tensor,
    space_key: str,
    positive: bool,
    record: Mapping[str, Any],
    annotate: str = "",
) -> PointSpec:
    role = "positive_prototype" if positive else "negative_prototype"
    proto_space = "visual" if space_key == "visual" else "text"
    return PointSpec(
        vector,
        f"{role}_{proto_space}",
        f"{space_key}_iapr_{role}_pid_{int(record['pid'])}_slot_{int(record['local_slot'])}_global_{int(record['global_index'])}",
        "D" if positive else "^",
        POS_COLOR if positive else HARD_NEG_COLOR,
        size=72 if positive else 76,
        alpha=0.88,
        hollow=True,
        annotate=annotate,
        metadata={"role": role, **dict(record)},
    )


def motion_arrows(space_key: str, target_model: str) -> List[ArrowSpec]:
    is_host = target_model == "host"
    return [
        ArrowSpec(
            f"{space_key}_vanilla_positive_group_centroid",
            f"{space_key}_{target_model}_positive_group_centroid",
            POS_MOTION_COLOR,
            linestyle="--" if is_host else ":",
            linewidth=1.35 if is_host else 1.55,
            alpha=0.72,
            label=f"Vanilla -> {'Host' if is_host else 'IAPR'} positive group",
        ),
        ArrowSpec(
            f"{space_key}_vanilla_hard_negative_group_centroid",
            f"{space_key}_{target_model}_hard_negative_group_centroid",
            NEG_MOTION_COLOR,
            linestyle="-." if is_host else "-",
            linewidth=1.35 if is_host else 1.20,
            alpha=0.72,
            label=f"Vanilla -> {'Host' if is_host else 'IAPR'} hard-negative group",
        ),
    ]


def motion_metadata(space_key: str, panels: Sequence[PanelSpec]) -> Dict[str, Any]:
    tracks: List[Dict[str, Any]] = []
    for panel in panels:
        for arrow in panel.arrows:
            tracks.append(
                {
                    "label": arrow.label,
                    "start_label": arrow.start_label,
                    "end_label": arrow.end_label,
                    "color": arrow.color,
                    "linestyle": arrow.linestyle,
                }
            )
    return {
        "space": space_key,
        "centroid_definition": "L2-normalized mean of the plotted sample group embeddings in each model panel.",
        "tracks": tracks,
    }


def build_visual_space_panels(
    vanilla_ctx: RowContext,
    host_ctx: RowContext,
    iapr_ctx: RowContext,
    query_index: int,
    query_pid: int,
    positive_image_indices: Sequence[int],
    hard_negative_image_indices: Sequence[int],
    positive_text_indices: Sequence[int],
    gallery_pids: torch.Tensor,
    max_prototypes_per_id: int,
    prototype_hard_k: int,
) -> Tuple[List[PanelSpec], Dict[str, Any]]:
    bank = iapr_ctx.bank
    text_translated_centroid, text_assignments = build_text_translated_centroid(iapr_ctx, bank, query_pid, positive_text_indices)
    positive_proto_records = positive_prototype_records(bank, query_pid, max_prototypes_per_id)
    negative_proto_records = select_hard_negative_prototypes(text_translated_centroid, bank.image_prototypes, bank, query_pid, prototype_hard_k)

    centroids: Dict[str, Dict[str, torch.Tensor]] = {}
    sample_records: Dict[str, Any] = {}
    for model_key, _display, ctx in model_context_sequence(vanilla_ctx, host_ctx, iapr_ctx):
        centroids[model_key] = {
            "positive": feature_centroid(ctx.projected.image_features, positive_image_indices),
            "hard_negative": feature_centroid(ctx.projected.image_features, hard_negative_image_indices),
        }
        sample_records[model_key] = {
            "positive": [
                {"gallery_index": int(index), "pid": int(gallery_pids[int(index)].item())}
                for index in positive_image_indices
            ],
            "hard_negative": [
                {"gallery_index": int(index), "pid": int(gallery_pids[int(index)].item())}
                for index in hard_negative_image_indices
            ],
        }

    def model_points(model_key: str, ctx: RowContext) -> List[PointSpec]:
        points: List[PointSpec] = []
        for rank, image_index in enumerate(positive_image_indices, start=1):
            points.append(
                sample_point(
                    ctx.projected.image_features[int(image_index)],
                    "visual",
                    model_key,
                    "positive",
                    int(image_index),
                    int(gallery_pids[int(image_index)].item()),
                    rank,
                )
            )
        for rank, image_index in enumerate(hard_negative_image_indices, start=1):
            points.append(
                sample_point(
                    ctx.projected.image_features[int(image_index)],
                    "visual",
                    model_key,
                    "hard_negative",
                    int(image_index),
                    int(gallery_pids[int(image_index)].item()),
                    rank,
                )
            )
        return points

    semantic_iapr_points: List[PointSpec] = [translated_centroid_point(text_translated_centroid, "visual")]
    for idx, record in enumerate(positive_proto_records):
        semantic_iapr_points.append(prototype_point(bank.image_prototypes[int(record["global_index"])], "visual", True, record, annotate="P+" if idx == 0 else ""))
    for idx, record in enumerate(negative_proto_records):
        semantic_iapr_points.append(prototype_point(bank.image_prototypes[int(record["global_index"])], "visual", False, record, annotate="P-" if idx == 0 else ""))

    vanilla_points = model_points("vanilla", vanilla_ctx) + [
        motion_centroid_point(centroids["vanilla"]["positive"], "visual", "vanilla", "positive"),
        motion_centroid_point(centroids["vanilla"]["hard_negative"], "visual", "vanilla", "hard_negative"),
    ]
    host_points = model_points("host", host_ctx) + [
        motion_centroid_point(centroids["vanilla"]["positive"], "visual", "vanilla", "positive", ghost=True),
        motion_centroid_point(centroids["vanilla"]["hard_negative"], "visual", "vanilla", "hard_negative", ghost=True),
        motion_centroid_point(centroids["host"]["positive"], "visual", "host", "positive"),
        motion_centroid_point(centroids["host"]["hard_negative"], "visual", "host", "hard_negative"),
    ]
    iapr_points = model_points("iapr", iapr_ctx) + [
        motion_centroid_point(centroids["vanilla"]["positive"], "visual", "vanilla", "positive", ghost=True),
        motion_centroid_point(centroids["vanilla"]["hard_negative"], "visual", "vanilla", "hard_negative", ghost=True),
        motion_centroid_point(centroids["iapr"]["positive"], "visual", "iapr", "positive"),
        motion_centroid_point(centroids["iapr"]["hard_negative"], "visual", "iapr", "hard_negative"),
    ] + semantic_iapr_points

    panels = [
        PanelSpec("Vanilla CLIP", vanilla_points),
        PanelSpec("Host", host_points, arrows=motion_arrows("visual", "host")),
        PanelSpec("IAPR", iapr_points, arrows=motion_arrows("visual", "iapr")),
    ]
    metadata = {
        "space": "visual",
        "layout": "1x3",
        "positive_samples": {
            "rule": "all gallery images whose pid == query_pid",
            "indices": [int(index) for index in positive_image_indices],
            "count": int(len(positive_image_indices)),
        },
        "hard_negative_samples": {
            "rule": "shared sample set selected once from the Host image-retrieval ranking; positives are excluded before taking top_hard_neg",
            "shared_across_models": True,
            "top_hard_neg": int(len(hard_negative_image_indices)),
            "indices": [int(index) for index in hard_negative_image_indices],
            "pids": [int(gallery_pids[int(index)].item()) for index in hard_negative_image_indices],
        },
        "translated_centroid": {
            "name": "text-translated centroid",
            "plotted_in": "IAPR visual panel only",
            "construction": "all positive text samples -> IAPR text projector -> assign within query pid text_prototypes -> text_to_image -> mean -> L2 normalize",
            "positive_text_indices": [int(index) for index in positive_text_indices],
            "assignments_by_text_index": text_assignments,
        },
        "prototypes": {
            "positive_visual_prototypes": positive_proto_records,
            "negative_visual_prototypes": negative_proto_records,
            "negative_ranking": "top prototype_hard_k non-query image_prototypes by cosine similarity to the text-translated centroid",
        },
        "panel_semantics": {
            "vanilla": "positive image samples + shared hard negative image samples; centroid markers are motion annotations only",
            "host": "positive image samples + shared hard negative image samples; centroid markers/arrows are motion annotations only",
            "iapr": "positive image samples + shared hard negative image samples + text-translated centroid + visual prototypes",
        },
        "samples_by_model": sample_records,
        "group_motion": motion_metadata("visual", panels),
        "conceptual_guardrail": "No raw text embedding is plotted directly in visual space; translated text evidence is represented through IAPR text_to_image only.",
    }
    return panels, metadata


def build_text_space_panels(
    vanilla_ctx: RowContext,
    host_ctx: RowContext,
    iapr_ctx: RowContext,
    query_index: int,
    query_pid: int,
    positive_text_indices: Sequence[int],
    positive_image_indices: Sequence[int],
    max_prototypes_per_id: int,
    prototype_hard_k: int,
    top_hard_negative: int,
) -> Tuple[List[PanelSpec], Dict[str, Any]]:
    bank = iapr_ctx.bank
    image_translated_centroid, image_assignments = build_image_translated_centroid(iapr_ctx, bank, query_pid, positive_image_indices)
    positive_proto_records = positive_prototype_records(bank, query_pid, max_prototypes_per_id)
    negative_proto_records = select_hard_negative_prototypes(image_translated_centroid, bank.text_prototypes, bank, query_pid, prototype_hard_k)

    hard_negative_indices: Dict[str, List[int]] = {}
    hard_negative_scores: Dict[str, torch.Tensor] = {}
    reference_metadata: Dict[str, Any] = {}
    for model_key, display, ctx in model_context_sequence(vanilla_ctx, host_ctx, iapr_ctx):
        if model_key == "iapr":
            reference = image_translated_centroid
            reference_name = "image-translated centroid"
        else:
            reference = ctx.projected.text_features[int(query_index)]
            reference_name = "model query-side text representation"
        indices, scores = select_hard_negative_text_indices(ctx.projected.text_features, ctx.projected.query_pids, query_pid, reference, top_hard_negative)
        hard_negative_indices[model_key] = indices
        hard_negative_scores[model_key] = scores
        reference_metadata[model_key] = {
            "display_label": display,
            "reference": reference_name,
            "selection_rule": "rank all text samples in this model text space by cosine similarity to the reference; exclude query_pid positives; take top_hard_neg",
            "selected_indices": [int(index) for index in indices],
            "selected_pids": [int(ctx.projected.query_pids[int(index)].item()) for index in indices],
            "selected_scores": [float(scores[int(index)].item()) for index in indices],
        }

    centroids: Dict[str, Dict[str, torch.Tensor]] = {}
    sample_records: Dict[str, Any] = {}
    for model_key, _display, ctx in model_context_sequence(vanilla_ctx, host_ctx, iapr_ctx):
        centroids[model_key] = {
            "positive": feature_centroid(ctx.projected.text_features, positive_text_indices),
            "hard_negative": feature_centroid(ctx.projected.text_features, hard_negative_indices[model_key]),
        }
        sample_records[model_key] = {
            "positive": [
                {"text_index": int(index), "pid": int(ctx.projected.query_pids[int(index)].item())}
                for index in positive_text_indices
            ],
            "hard_negative": [
                {"text_index": int(index), "pid": int(ctx.projected.query_pids[int(index)].item()), "score": float(hard_negative_scores[model_key][int(index)].item())}
                for index in hard_negative_indices[model_key]
            ],
        }

    def model_points(model_key: str, ctx: RowContext) -> List[PointSpec]:
        points: List[PointSpec] = []
        for rank, text_index in enumerate(positive_text_indices, start=1):
            points.append(
                sample_point(
                    ctx.projected.text_features[int(text_index)],
                    "text",
                    model_key,
                    "positive",
                    int(text_index),
                    int(ctx.projected.query_pids[int(text_index)].item()),
                    rank,
                )
            )
        for rank, text_index in enumerate(hard_negative_indices[model_key], start=1):
            points.append(
                sample_point(
                    ctx.projected.text_features[int(text_index)],
                    "text",
                    model_key,
                    "hard_negative",
                    int(text_index),
                    int(ctx.projected.query_pids[int(text_index)].item()),
                    rank,
                )
            )
        return points

    semantic_iapr_points: List[PointSpec] = [translated_centroid_point(image_translated_centroid, "text")]
    for idx, record in enumerate(positive_proto_records):
        semantic_iapr_points.append(prototype_point(bank.text_prototypes[int(record["global_index"])], "text", True, record, annotate="P+" if idx == 0 else ""))
    for idx, record in enumerate(negative_proto_records):
        semantic_iapr_points.append(prototype_point(bank.text_prototypes[int(record["global_index"])], "text", False, record, annotate="P-" if idx == 0 else ""))

    vanilla_points = model_points("vanilla", vanilla_ctx) + [
        motion_centroid_point(centroids["vanilla"]["positive"], "text", "vanilla", "positive"),
        motion_centroid_point(centroids["vanilla"]["hard_negative"], "text", "vanilla", "hard_negative"),
    ]
    host_points = model_points("host", host_ctx) + [
        motion_centroid_point(centroids["vanilla"]["positive"], "text", "vanilla", "positive", ghost=True),
        motion_centroid_point(centroids["vanilla"]["hard_negative"], "text", "vanilla", "hard_negative", ghost=True),
        motion_centroid_point(centroids["host"]["positive"], "text", "host", "positive"),
        motion_centroid_point(centroids["host"]["hard_negative"], "text", "host", "hard_negative"),
    ]
    iapr_points = model_points("iapr", iapr_ctx) + [
        motion_centroid_point(centroids["vanilla"]["positive"], "text", "vanilla", "positive", ghost=True),
        motion_centroid_point(centroids["vanilla"]["hard_negative"], "text", "vanilla", "hard_negative", ghost=True),
        motion_centroid_point(centroids["iapr"]["positive"], "text", "iapr", "positive"),
        motion_centroid_point(centroids["iapr"]["hard_negative"], "text", "iapr", "hard_negative"),
    ] + semantic_iapr_points

    panels = [
        PanelSpec("Vanilla CLIP", vanilla_points),
        PanelSpec("Host", host_points, arrows=motion_arrows("text", "host")),
        PanelSpec("IAPR", iapr_points, arrows=motion_arrows("text", "iapr")),
    ]
    metadata = {
        "space": "text",
        "layout": "1x3",
        "positive_samples": {
            "rule": "all text/caption samples whose pid == query_pid",
            "indices": [int(index) for index in positive_text_indices],
            "count": int(len(positive_text_indices)),
        },
        "hard_negative_samples": {
            "rule": "panel-specific symmetric text-space diagnostic ranking; positives are excluded before taking top_hard_neg",
            "shared_across_models": False,
            "top_hard_neg": int(top_hard_negative),
            "by_model": reference_metadata,
            "note": "Vanilla/Host use the model query-side text representation; IAPR uses the image-translated centroid.",
        },
        "translated_centroid": {
            "name": "image-translated centroid",
            "plotted_in": "IAPR text panel only",
            "construction": "all positive image samples -> IAPR image projector -> assign within query pid image_prototypes -> image_to_text -> mean -> L2 normalize",
            "positive_image_indices": [int(index) for index in positive_image_indices],
            "assignments_by_image_index": image_assignments,
        },
        "prototypes": {
            "positive_text_prototypes": positive_proto_records,
            "negative_text_prototypes": negative_proto_records,
            "negative_ranking": "top prototype_hard_k non-query text_prototypes by cosine similarity to the image-translated centroid",
        },
        "panel_semantics": {
            "vanilla": "positive text samples + panel-specific hard negative text samples; centroid markers are motion annotations only",
            "host": "positive text samples + panel-specific hard negative text samples; centroid markers/arrows are motion annotations only",
            "iapr": "positive text samples + panel-specific hard negative text samples + image-translated centroid + text prototypes",
        },
        "samples_by_model": sample_records,
        "group_motion": motion_metadata("text", panels),
        "conceptual_guardrail": "No raw image embedding is plotted directly in text space; translated image evidence is represented through IAPR image_to_text only.",
    }
    return panels, metadata

def image_record(
    index: int,
    split_data: SplitData,
    gallery_pids: torch.Tensor,
    sim_host: torch.Tensor,
    sim_iapr: torch.Tensor,
    query_index: int,
    sim_vanilla: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    record = {
        "gallery_index": int(index),
        "pid": int(gallery_pids[index].item()),
        "path": str(split_data.img_paths[index]),
        "host_similarity": float(sim_host[query_index, index].item()),
        "iapr_similarity": float(sim_iapr[query_index, index].item()),
    }
    if sim_vanilla is not None:
        record["vanilla_similarity"] = float(sim_vanilla[query_index, index].item())
    return record


def text_record(index: int, split_data: SplitData, text_pids: torch.Tensor) -> Dict[str, Any]:
    return {
        "text_index": int(index),
        "pid": int(text_pids[int(index)].item()),
        "caption": str(split_data.captions[int(index)]),
    }


def passes_prototype_pid_filter(bank: PrototypeBankData, query_pid: int, gallery_pids: torch.Tensor, indices: Sequence[int]) -> bool:
    if not pid_has_slots(bank, query_pid):
        return False
    return all(pid_has_slots(bank, int(gallery_pids[index].item())) for index in indices)


def select_candidates(
    sim_host: torch.Tensor,
    sim_iapr: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    split_data: SplitData,
    bank: PrototypeBankData,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidates: List[Dict[str, Any]] = []
    counters: Dict[str, int] = defaultdict(int)
    total = int(sim_host.shape[0])
    if args.max_candidate_queries > 0:
        total = min(total, int(args.max_candidate_queries))
    for query_index in tqdm(range(total), desc="Selecting repair candidates"):
        query_pid = int(query_pids[query_index].item())
        pos_mask = gallery_pids.eq(query_pid)
        neg_mask = ~pos_mask
        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            counters["missing_positive_or_negative"] += 1
            continue
        if not pid_has_slots(bank, query_pid):
            counters["query_pid_missing_from_prototype_bank"] += 1
            continue

        host_pos, host_neg, host_margin, host_pos_idx, host_neg_idx = score_extrema(sim_host[query_index], pos_mask, neg_mask)
        iapr_pos, iapr_neg, iapr_margin, iapr_pos_idx, iapr_neg_idx = score_extrema(sim_iapr[query_index], pos_mask, neg_mask)
        delta_margin = iapr_margin - host_margin
        if delta_margin <= float(args.min_delta_margin):
            counters["delta_margin_too_small"] += 1
            continue

        hard_negative_pid = int(gallery_pids[host_neg_idx].item())
        if hard_negative_pid == query_pid:
            counters["host_hard_negative_pid_matches_query"] += 1
            continue
        if not pid_has_slots(bank, hard_negative_pid):
            counters["host_hard_negative_pid_missing_from_prototype_bank"] += 1
            continue

        pos_scores = torch.maximum(sim_host[query_index], sim_iapr[query_index])
        selected_positive = topk_from_mask(pos_scores, pos_mask, args.top_pos)
        hard_negative_mask = gallery_pids.eq(hard_negative_pid)
        selected_negative = topk_from_mask(sim_host[query_index], hard_negative_mask, args.top_hard_neg)
        selected_positive = [idx for idx in selected_positive if Path(split_data.img_paths[idx]).is_file()]
        selected_negative = [idx for idx in selected_negative if Path(split_data.img_paths[idx]).is_file()]
        if not selected_positive or not selected_negative:
            counters["no_selected_positive_or_negative_after_filter"] += 1
            continue
        if not passes_prototype_pid_filter(bank, query_pid, gallery_pids, selected_positive + selected_negative):
            counters["selected_pid_missing_from_bank"] += 1
            continue

        preferred_host_low = host_margin < float(args.host_margin_max)
        preferred_iapr_positive = iapr_margin > float(args.min_iapr_margin)
        fully_repaired = bool(host_margin < 0.0 and iapr_margin > 0.0)
        preferred_repair_case = bool(preferred_host_low and preferred_iapr_positive)
        if not preferred_host_low:
            counters["fallback_host_margin_not_low"] += 1
        if not preferred_iapr_positive:
            counters["fallback_iapr_margin_not_high_enough"] += 1

        candidates.append(
            {
                "query_index": int(query_index),
                "query_pid": query_pid,
                "query_text": split_data.captions[query_index],
                "host_pos_score": host_pos,
                "host_neg_score": host_neg,
                "host_margin": host_margin,
                "host_best_pos_index": host_pos_idx,
                "host_hard_neg_index": host_neg_idx,
                "iapr_pos_score": iapr_pos,
                "iapr_neg_score": iapr_neg,
                "iapr_margin": iapr_margin,
                "iapr_best_pos_index": iapr_pos_idx,
                "iapr_hard_neg_index": iapr_neg_idx,
                "delta_margin": delta_margin,
                "hard_negative_pid": hard_negative_pid,
                "fully_repaired": fully_repaired,
                "preferred_repair_case": preferred_repair_case,
                "repair_status": "fully_repaired" if fully_repaired else "improved_but_not_fully_repaired",
                "selected_positive": selected_positive,
                "selected_negative": selected_negative,
                "iapr_top_retrieved": topk_all(sim_iapr[query_index], 2),
            }
        )
    candidates.sort(
        key=lambda row: (
            bool(row["preferred_repair_case"]),
            bool(row["fully_repaired"]),
            float(row["delta_margin"]),
            float(row["iapr_margin"]),
            -float(row["host_margin"]),
        ),
        reverse=True,
    )
    return candidates, dict(counters)


def render_space_figure(
    case_id: int,
    space_key: str,
    panels: Sequence[PanelSpec],
    candidate: Mapping[str, Any],
    retrieval_metrics_by_model: Mapping[str, Mapping[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    scope = (
        f"{space_key} space: one shared 2D projection fit over Vanilla CLIP, Host, and IAPR "
        "panel points for within-space comparability"
    )
    projection = fit_shared_projection(panel_points(panels), args.projection, scope)

    fig = plt.figure(figsize=(13.3, 5.7), constrained_layout=False)
    outer = GridSpec(
        3,
        1,
        figure=fig,
        height_ratios=[4.7, 0.46, 0.72],
        hspace=0.075,
        top=0.835,
        bottom=0.055,
        left=0.035,
        right=0.990,
    )
    panel_grid = outer[0].subgridspec(1, 3, wspace=0.045)
    metric_grid = outer[1].subgridspec(1, 3, wspace=0.06)

    projection_meta: Dict[str, Any] = dict(projection.metadata)
    projection_meta["panels"] = []
    for col, panel in enumerate(panels):
        ax = fig.add_subplot(panel_grid[0, col])
        projection_meta["panels"].append(draw_panel_with_shared_projection(ax, panel, projection))

    footer_metric_lines = draw_footer_metrics(fig, metric_grid, retrieval_metrics_by_model)
    draw_footer_legend(fig, outer[2])

    query_index = int(candidate["query_index"])
    query_pid = int(candidate["query_pid"])
    status = "fully repaired" if bool(candidate.get("fully_repaired")) else "improved, not fully repaired"
    vanilla_margin = float(retrieval_metrics_by_model["vanilla"]["margin"])
    host_margin = float(retrieval_metrics_by_model["host"]["margin"])
    iapr_margin = float(retrieval_metrics_by_model["iapr"]["margin"])
    delta_iapr_host = iapr_margin - host_margin
    title_space = "Visual Space" if space_key == "visual" else "Text Space"
    fig.suptitle(
        f"{title_space}: Vanilla CLIP / Host / IAPR | q={query_index}, pid={query_pid} | "
        f"m V={vanilla_margin:+.3f}, H={host_margin:+.3f}, I={iapr_margin:+.3f}, dIH={delta_iapr_host:+.3f} | {status}\n"
        f"{truncate_text(str(candidate['query_text']), chars=150)}",
        fontsize=9.1,
        fontweight="bold",
        y=0.970,
    )

    pdf_path = output_dir / f"case_{case_id:03d}_{space_key}_space.pdf"
    png_path = output_dir / f"case_{case_id:03d}_{space_key}_space.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    return {
        "output_pdf": str(pdf_path),
        "output_png": str(png_path),
        "footer_metric_lines": footer_metric_lines,
        "projection": projection_meta,
    }


def render_case(
    case_id: int,
    candidate: Mapping[str, Any],
    vanilla_ctx: RowContext,
    host_ctx: RowContext,
    iapr_ctx: RowContext,
    split_data: SplitData,
    gallery_pids: torch.Tensor,
    sim_vanilla: torch.Tensor,
    sim_host: torch.Tensor,
    sim_iapr: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    """Render separate 1x3 visual-space and text-space diagnostic figures."""
    try:
        import matplotlib

        matplotlib.use("Agg")
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to draw the local repair figures.") from exc

    query_index = int(candidate["query_index"])
    query_pid = int(candidate["query_pid"])

    positive_image_indices = collect_positive_image_indices(gallery_pids, query_pid)
    positive_text_indices = collect_positive_text_indices(iapr_ctx.projected.query_pids, query_pid)
    hard_negative_image_indices = select_hard_negative_image_indices(
        query_index,
        query_pid,
        gallery_pids,
        sim_host,
        int(args.top_hard_neg),
    )
    if not positive_image_indices:
        raise ValueError(f"no positive gallery images found for query_pid={query_pid}")
    if not positive_text_indices:
        raise ValueError(f"no positive text samples found for query_pid={query_pid}")
    if not hard_negative_image_indices:
        raise ValueError(f"no hard negative gallery images found for query_pid={query_pid}")

    retrieval_metrics_by_model = {
        "vanilla": retrieval_metrics_for_query(sim_vanilla, query_index, query_pid, gallery_pids),
        "host": retrieval_metrics_for_query(sim_host, query_index, query_pid, gallery_pids),
        "iapr": retrieval_metrics_for_query(sim_iapr, query_index, query_pid, gallery_pids),
    }

    visual_panels, visual_meta = build_visual_space_panels(
        vanilla_ctx=vanilla_ctx,
        host_ctx=host_ctx,
        iapr_ctx=iapr_ctx,
        query_index=query_index,
        query_pid=query_pid,
        positive_image_indices=positive_image_indices,
        hard_negative_image_indices=hard_negative_image_indices,
        positive_text_indices=positive_text_indices,
        gallery_pids=gallery_pids,
        max_prototypes_per_id=int(args.prototype_per_id),
        prototype_hard_k=int(args.prototype_hard_k),
    )
    text_panels, text_meta = build_text_space_panels(
        vanilla_ctx=vanilla_ctx,
        host_ctx=host_ctx,
        iapr_ctx=iapr_ctx,
        query_index=query_index,
        query_pid=query_pid,
        positive_text_indices=positive_text_indices,
        positive_image_indices=positive_image_indices,
        max_prototypes_per_id=int(args.prototype_per_id),
        prototype_hard_k=int(args.prototype_hard_k),
        top_hard_negative=int(args.top_hard_neg),
    )

    visual_render_meta = render_space_figure(case_id, "visual", visual_panels, candidate, retrieval_metrics_by_model, args, output_dir)
    text_render_meta = render_space_figure(case_id, "text", text_panels, candidate, retrieval_metrics_by_model, args, output_dir)

    vanilla_margin = float(retrieval_metrics_by_model["vanilla"]["margin"])
    host_margin = float(retrieval_metrics_by_model["host"]["margin"])
    iapr_margin = float(retrieval_metrics_by_model["iapr"]["margin"])
    delta_iapr_host = iapr_margin - host_margin
    status = "fully repaired" if bool(candidate.get("fully_repaired")) else "improved, not fully repaired"

    hard_negative_text_records_by_model = {
        key: [text_record(int(index), split_data, iapr_ctx.projected.query_pids) for index in value.get("selected_indices", [])]
        for key, value in text_meta["hard_negative_samples"]["by_model"].items()
    }

    metadata = {
        "query_index": query_index,
        "query_text": str(candidate["query_text"]),
        "query_pid": query_pid,
        "rendering_layout": "separate_1x3_visual_and_text_figures",
        "repair_status": str(candidate.get("repair_status", status.replace(" ", "_"))),
        "fully_repaired": bool(candidate.get("fully_repaired")),
        "preferred_repair_case": bool(candidate.get("preferred_repair_case")),
        "vanilla_margin": vanilla_margin,
        "host_margin": host_margin,
        "iapr_margin": iapr_margin,
        "delta_margin": delta_iapr_host,
        "delta_iapr_host_margin": delta_iapr_host,
        "vanilla_positive_score": float(retrieval_metrics_by_model["vanilla"]["positive_score"]),
        "vanilla_hard_negative_score": float(retrieval_metrics_by_model["vanilla"]["hard_negative_score"]),
        "host_positive_score": float(retrieval_metrics_by_model["host"]["positive_score"]),
        "host_hard_negative_score": float(retrieval_metrics_by_model["host"]["hard_negative_score"]),
        "iapr_positive_score": float(retrieval_metrics_by_model["iapr"]["positive_score"]),
        "iapr_hard_negative_score": float(retrieval_metrics_by_model["iapr"]["hard_negative_score"]),
        "retrieval_metrics": retrieval_metrics_by_model,
        "plot_controls": {
            "top_hard_neg_samples": int(args.top_hard_neg),
            "prototype_per_id": int(args.prototype_per_id),
            "prototype_hard_k": int(args.prototype_hard_k),
            "top_pos_case_selection_only": int(args.top_pos),
            "itself_enabled_for_host_iapr": bool(args.itself or args.retrieval_branch == "itself"),
            "itself_lambda": float(args.itself_lambda),
        },
        "plotted_samples": {
            "positive_images": [image_record(idx, split_data, gallery_pids, sim_host, sim_iapr, query_index, sim_vanilla=sim_vanilla) for idx in positive_image_indices],
            "hard_negative_images": [image_record(idx, split_data, gallery_pids, sim_host, sim_iapr, query_index, sim_vanilla=sim_vanilla) for idx in hard_negative_image_indices],
            "positive_texts": [text_record(idx, split_data, iapr_ctx.projected.query_pids) for idx in positive_text_indices],
            "hard_negative_texts_by_model": hard_negative_text_records_by_model,
        },
        "candidate_selection": {
            "note": "Case selection is preserved from the earlier Host-vs-IAPR margin diagnostic; plotted positive samples are no longer limited by top_pos.",
            "host_case_hard_negative_pid": int(candidate["hard_negative_pid"]),
            "case_selection_positive_indices": [int(x) for x in candidate.get("selected_positive", [])],
            "case_selection_negative_indices": [int(x) for x in candidate.get("selected_negative", [])],
            "iapr_top_retrieved_images": [
                image_record(int(idx), split_data, gallery_pids, sim_host, sim_iapr, query_index, sim_vanilla=sim_vanilla)
                for idx in [int(x) for x in candidate.get("iapr_top_retrieved", [])]
            ],
        },
        "similarities_used_for_ranking": {
            "margin_formula": "max_positive_similarity - max_hard_negative_similarity",
            "vanilla_best_positive_index": int(retrieval_metrics_by_model["vanilla"]["best_positive_index"]),
            "vanilla_hard_negative_index": int(retrieval_metrics_by_model["vanilla"]["hard_negative_index"]),
            "host_best_positive_index": int(candidate["host_best_pos_index"]),
            "host_hard_negative_index": int(candidate["host_hard_neg_index"]),
            "iapr_best_positive_index": int(candidate["iapr_best_pos_index"]),
            "iapr_hard_negative_index": int(candidate["iapr_hard_neg_index"]),
        },
        "visual_space": {
            **visual_meta,
            **visual_render_meta,
        },
        "text_space": {
            **text_meta,
            **text_render_meta,
        },
        "output_files": {
            "visual_space": {"pdf": visual_render_meta["output_pdf"], "png": visual_render_meta["output_png"]},
            "text_space": {"pdf": text_render_meta["output_pdf"], "png": text_render_meta["output_png"]},
        },
        "projection": {
            "method": args.projection,
            "layout": "separate 1x3 figures",
            "fit_scope": {
                "visual_space": "one shared fit over Vanilla + Host + IAPR visual-space points only",
                "text_space": "one shared fit over Vanilla + Host + IAPR text-space points only",
            },
        },
        "diagnostic_anchor_space": {
            "projectors": "IAPR projection heads are used for Vanilla, Host, and IAPR projected bundles",
            "prototype_bank": "IAPR prototype bank is used for IAPR-panel prototypes and translated-centroid construction",
            "visual_space_translation": "positive text evidence is translated to visual space only through text_to_image",
            "text_space_translation": "positive image evidence is translated to text space only through image_to_text",
        },
        "conceptual_guardrail": (
            "Raw cross-modal embeddings are not directly mixed. The visual-space translated centroid is built through text_to_image, "
            "and the text-space translated centroid is built through image_to_text."
        ),
    }
    metadata_path = output_dir / f"case_{case_id:03d}_metadata.json"
    save_json(metadata, metadata_path)
    metadata["metadata_path"] = str(metadata_path)
    return metadata

def print_load_report(label: str, report: CheckpointLoadReport) -> None:
    print(
        f"[{label}] loaded={report.loaded} missing={report.skipped_missing} "
        f"shape={report.skipped_shape} memory_skipped={report.skipped_memory} non_tensor={report.skipped_non_tensor}"
    )


def cleanup_model(model: Optional[torch.nn.Module], device: torch.device) -> None:
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()



def main() -> None:
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("--num_cases must be positive.")
    if args.top_pos <= 0:
        raise ValueError("--top_pos must be positive.")
    if args.top_hard_neg <= 0:
        raise ValueError("--top_hard_neg must be positive.")
    if args.prototype_per_id <= 0:
        raise ValueError("--prototype_per_id must be positive.")
    if args.prototype_hard_k < 0:
        raise ValueError("--prototype_hard_k must be non-negative.")
    if not (0.0 <= float(args.retrieval_alpha) <= 1.0):
        raise ValueError("--retrieval_alpha must be in [0, 1].")
    if not (0.0 <= float(args.itself_lambda) <= 1.0):
        raise ValueError("--itself_lambda/--lambda must be in [0, 1].")
    use_itself_scoring = bool(args.itself or args.retrieval_branch == "itself")

    set_seed(int(args.seed))
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = resolve_path(args.config) if args.config else None
    data_root = resolve_path(args.data_root)
    host_ckpt = resolve_path(args.host_ckpt)
    iapr_ckpt = resolve_path(args.iapr_ckpt)
    prototype_path = resolve_path(args.prototype_ckpt) if args.prototype_ckpt else iapr_ckpt
    prototype_projector_path = resolve_path(args.prototype_projector_ckpt)
    path_checks = [("host_ckpt", host_ckpt), ("iapr_ckpt", iapr_ckpt), ("prototype source", prototype_path), ("prototype_projector_ckpt", prototype_projector_path)]
    if config_path is not None:
        path_checks.insert(0, ("config", config_path))
    for label, path in path_checks:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    device = resolve_device(args.device)
    print(TITLE)
    print(f"[Device] {device}")
    print(f"[Prototype] loading IAPR memory from {prototype_path}")
    try:
        iapr_bank = load_prototype_bank(prototype_path, required=True)
    except RuntimeError as exc:
        if not args.prototype_ckpt:
            raise RuntimeError(
                f"Could not load prototype memory from --iapr_ckpt ({iapr_ckpt}). Pass --prototype_ckpt "
                "pointing to best_prototype_bank.pth or another checkpoint containing prototype memory."
            ) from exc
        raise
    assert iapr_bank is not None
    print(
        f"[Prototype] dim={iapr_bank.dim} total={iapr_bank.total} "
        f"k={iapr_bank.prototypes_per_id} unique_pids={len(iapr_bank.unique_pids)}"
    )
    if args.prototype_per_id != iapr_bank.prototypes_per_id:
        print(
            f"[Prototype] bank has {iapr_bank.prototypes_per_id} slots per id; "
            f"--prototype_per_id={args.prototype_per_id} controls plotted slots only."
        )

    split_data = load_split_data(args.dataset_name, data_root, args.split)
    validate_split_data(split_data, args.split)
    print(f"[Dataset] {args.dataset_name} split={args.split} queries={len(split_data.captions)} gallery={len(split_data.img_paths)}")

    model_args = config_model_args(args, iapr_bank)
    num_classes = int(iapr_bank.num_classes)
    print(
        f"[Model] num_classes={num_classes} only_global={model_args.only_global} prototype_feature={model_args.prototype_feature} "
        f"return_all={model_args.return_all} topk_type={model_args.topk_type} modify_k={model_args.modify_k} "
        f"prototype_dim={model_args.prototype_dim} prototype_per_id={model_args.prototype_per_id}"
    )

    iapr_model, iapr_report, iapr_aux_report = build_loaded_model(
        "IAPR", iapr_ckpt, model_args, num_classes, device, optional_branch_path=prototype_projector_path
    )
    iapr_branch = getattr(iapr_model, "prototype_branch", None)
    if iapr_branch is None:
        raise RuntimeError("IAPR model was built without a prototype branch. Check prototype/architecture CLI settings.")
    if not projector_weights_loaded(iapr_branch, iapr_report) and not (iapr_aux_report and projector_weights_loaded(iapr_branch, iapr_aux_report)):
        image_names, text_names = projector_param_names(iapr_branch)
        if image_names or text_names:
            raise RuntimeError(
                "IAPR projection heads were not loaded from the provided checkpoints. Expected keys like "
                f"{(image_names + text_names)[:8]}. If your checkpoint uses different names, edit candidate_state_keys() "
                "in scripts/plot_ambiguity_rate_compare.py or pass the correct --prototype_projector_ckpt."
            )
    iapr_memory_shapes = load_memory_into_branch(iapr_branch, iapr_bank)
    print_load_report("IAPR", iapr_report)
    if iapr_aux_report is not None:
        print_load_report("IAPR auxiliary", iapr_aux_report)

    host_model, host_report, host_aux_report = build_loaded_model("Host", host_ckpt, model_args, num_classes, device)
    host_branch = getattr(host_model, "prototype_branch", None)
    if host_branch is None:
        raise RuntimeError("Host diagnostic model was built without a prototype branch. Check prototype/architecture CLI settings.")
    print_load_report("Host", host_report)
    if host_aux_report is not None:
        print_load_report("Host auxiliary", host_aux_report)

    vanilla_model = build_vanilla_model("Vanilla CLIP", model_args, num_classes, device)

    host_bank = load_prototype_bank(host_ckpt, required=False)
    host_own_bank_available = False
    host_bank_reason = "host checkpoint has no compatible prototype memory"
    if host_bank is not None:
        try:
            host_projectors_ok = projector_weights_loaded(host_branch, host_report)
            if not host_projectors_ok:
                host_bank_reason = "host prototype memory exists but compatible projection heads were not loaded"
            else:
                host_own_bank_available = True
                host_bank_reason = "host checkpoint contains compatible prototype memory and projectors, ignored for the shared 3-way diagnostic"
        except Exception as exc:
            host_own_bank_available = False
            host_bank_reason = f"host prototype memory was found but not usable: {exc}"

    shared_iapr_anchors = True
    print(f"[Diagnostic anchors] shared IAPR projectors + shared IAPR bank for Vanilla/Host/IAPR ({host_bank_reason})")

    host_retrieval_mode = resolve_retrieval_mode(args, host_model, model_args)
    iapr_retrieval_mode = resolve_retrieval_mode(args, iapr_model, model_args)
    if use_itself_scoring:
        vanilla_retrieval_mode = "global"
        print("[Retrieval] Vanilla CLIP remains global-only; --itself is applied to Host/IAPR checkpoints.")
    else:
        vanilla_retrieval_mode = resolve_retrieval_mode(args, vanilla_model, model_args)
    if host_retrieval_mode != iapr_retrieval_mode:
        raise RuntimeError(
            "Host and IAPR retrieval modes differ: "
            f"Host={host_retrieval_mode}, IAPR={iapr_retrieval_mode}"
        )
    if not use_itself_scoring and host_retrieval_mode != vanilla_retrieval_mode:
        raise RuntimeError(
            "Retrieval modes differ: "
            f"Vanilla={vanilla_retrieval_mode}, Host={host_retrieval_mode}, IAPR={iapr_retrieval_mode}"
        )
    retrieval_score_weight = float(args.itself_lambda) if host_retrieval_mode == "itself" else float(args.retrieval_alpha)
    if host_retrieval_mode == "itself":
        print(
            f"[Retrieval] Vanilla={vanilla_retrieval_mode} Host/IAPR=itself "
            f"lambda={retrieval_score_weight:.3f}; score=lambda*s_grab+(1-lambda)*s_global"
        )
    else:
        print(f"[Retrieval] mode={host_retrieval_mode} alpha={retrieval_score_weight:.3f}")

    vanilla_retrieval = extract_retrieval_bundle(
        "Vanilla", vanilla_model, split_data, model_args, vanilla_retrieval_mode, retrieval_score_weight, args.batch_size, args.num_workers, device
    )
    host_retrieval = extract_retrieval_bundle(
        "Host", host_model, split_data, model_args, host_retrieval_mode, retrieval_score_weight, args.batch_size, args.num_workers, device
    )
    iapr_retrieval = extract_retrieval_bundle(
        "IAPR", iapr_model, split_data, model_args, iapr_retrieval_mode, retrieval_score_weight, args.batch_size, args.num_workers, device
    )
    if not torch.equal(vanilla_retrieval.query_pids, host_retrieval.query_pids):
        raise RuntimeError("Vanilla and Host query pids differ.")
    if not torch.equal(vanilla_retrieval.gallery_pids, host_retrieval.gallery_pids):
        raise RuntimeError("Vanilla and Host gallery pids differ.")
    if not torch.equal(host_retrieval.query_pids, iapr_retrieval.query_pids):
        raise RuntimeError("Host and IAPR query pids differ.")
    if not torch.equal(host_retrieval.gallery_pids, iapr_retrieval.gallery_pids):
        raise RuntimeError("Host and IAPR gallery pids differ.")
    sim_vanilla = vanilla_retrieval.similarity().cpu()
    sim_host = host_retrieval.similarity().cpu()
    sim_iapr = iapr_retrieval.similarity().cpu()
    rank1_before_plotting = {
        "vanilla": rank1_summary(sim_vanilla, vanilla_retrieval.query_pids, vanilla_retrieval.gallery_pids),
        "host": rank1_summary(sim_host, host_retrieval.query_pids, host_retrieval.gallery_pids),
        "iapr": rank1_summary(sim_iapr, iapr_retrieval.query_pids, iapr_retrieval.gallery_pids),
    }
    print_rank1_summaries(rank1_before_plotting)

    diagnostic_projection_decision = "IAPR projection heads and IAPR prototype bank used as the common diagnostic space for Vanilla, Host, and IAPR features"

    vanilla_projected = extract_projected_bundle(
        "Vanilla",
        vanilla_model,
        iapr_branch,
        split_data,
        model_args,
        "iapr_diagnostic_projector",
        args.batch_size,
        args.num_workers,
        device,
    )
    host_projected = extract_projected_bundle(
        "Host",
        host_model,
        iapr_branch,
        split_data,
        model_args,
        "iapr_diagnostic_projector",
        args.batch_size,
        args.num_workers,
        device,
    )
    iapr_projected = extract_projected_bundle("IAPR", iapr_model, iapr_branch, split_data, model_args, "iapr_diagnostic_projector", args.batch_size, args.num_workers, device)
    if not torch.equal(vanilla_projected.query_pids, vanilla_retrieval.query_pids):
        raise RuntimeError("Vanilla projected query pids differ from retrieval query pids.")
    if not torch.equal(vanilla_projected.gallery_pids, vanilla_retrieval.gallery_pids):
        raise RuntimeError("Vanilla projected gallery pids differ from retrieval gallery pids.")
    if not torch.equal(host_projected.query_pids, host_retrieval.query_pids):
        raise RuntimeError("Host projected query pids differ from retrieval query pids.")
    if not torch.equal(host_projected.gallery_pids, host_retrieval.gallery_pids):
        raise RuntimeError("Host projected gallery pids differ from retrieval gallery pids.")
    if not torch.equal(iapr_projected.query_pids, iapr_retrieval.query_pids):
        raise RuntimeError("IAPR projected query pids differ from retrieval query pids.")
    if not torch.equal(iapr_projected.gallery_pids, iapr_retrieval.gallery_pids):
        raise RuntimeError("IAPR projected gallery pids differ from retrieval gallery pids.")

    vanilla_ctx = RowContext(
        key="vanilla",
        label="Vanilla CLIP",
        projected=vanilla_projected,
        bank=iapr_bank,
        branch=iapr_branch,
        bank_source=iapr_bank.source_path,
        projection_decision="Vanilla CLIP features extracted without fine-tuned checkpoint, then projected by IAPR projection heads",
        shared_iapr_anchors=True,
    )
    host_ctx = RowContext(
        key="host",
        label="HOST",
        projected=host_projected,
        bank=iapr_bank,
        branch=iapr_branch,
        bank_source=iapr_bank.source_path,
        projection_decision="Host features projected by IAPR projection heads for shared diagnostic comparability",
        shared_iapr_anchors=shared_iapr_anchors,
    )
    iapr_ctx = RowContext(
        key="iapr",
        label="IAPR",
        projected=iapr_projected,
        bank=iapr_bank,
        branch=iapr_branch,
        bank_source=iapr_bank.source_path,
        projection_decision="IAPR features projected by IAPR projection heads with the shared IAPR prototype bank",
        shared_iapr_anchors=shared_iapr_anchors,
    )

    candidates, selection_counters = select_candidates(sim_host, sim_iapr, host_retrieval.query_pids, host_retrieval.gallery_pids, split_data, iapr_bank, args)
    print(f"[Selection] candidate queries={len(candidates)} counters={selection_counters}")
    if not candidates:
        proto_range = (min(iapr_bank.unique_pids), max(iapr_bank.unique_pids)) if iapr_bank.unique_pids else (None, None)
        split_pids = sorted(set(int(x) for x in host_retrieval.query_pids.tolist()) | set(int(x) for x in host_retrieval.gallery_pids.tolist()))
        split_range = (min(split_pids), max(split_pids)) if split_pids else (None, None)
        raise RuntimeError(
            "No visualization candidates survived filtering. This often means the prototype bank does not cover "
            f"the requested split identities. prototype_pid_range={proto_range}, split_pid_range={split_range}, "
            f"selection_counters={selection_counters}."
        )

    selected_cases: List[Dict[str, Any]] = []
    skipped_render = 0
    for candidate in candidates:
        if len(selected_cases) >= int(args.num_cases):
            break
        try:
            metadata = render_case(
                len(selected_cases),
                candidate,
                vanilla_ctx,
                host_ctx,
                iapr_ctx,
                split_data,
                host_retrieval.gallery_pids,
                sim_vanilla,
                sim_host,
                sim_iapr,
                args,
                output_dir,
            )
        except (ValueError, KeyError) as exc:
            skipped_render += 1
            print(f"[Render skip] q={candidate['query_index']} reason={exc}")
            continue
        selected_cases.append(metadata)
        print(f"[Saved] case_{len(selected_cases) - 1:03d} q={metadata['query_index']} delta={metadata['delta_margin']:.4f}")

    if not selected_cases:
        raise RuntimeError(f"No cases could be rendered after projection checks; skipped_render={skipped_render}.")

    summary = {
        "title": TITLE,
        "config": str(config_path) if config_path is not None else None,
        "dataset_name": args.dataset_name,
        "data_root": str(data_root),
        "split": args.split,
        "vanilla_ckpt": None,
        "vanilla_definition": "repo model architecture with the selected config/pretrain choice and no fine-tuned checkpoint loaded",
        "host_ckpt": str(host_ckpt),
        "iapr_ckpt": str(iapr_ckpt),
        "prototype_ckpt": str(prototype_path),
        "prototype_projector_ckpt": str(prototype_projector_path),
        "rendering_layout": "separate_1x3_visual_and_text_figures",
        "num_cases_requested": int(args.num_cases),
        "num_cases_rendered": int(len(selected_cases)),
        "skipped_render_cases": int(skipped_render),
        "plot_controls": {
            "top_pos_case_selection_only": int(args.top_pos),
            "top_hard_neg_samples": int(args.top_hard_neg),
            "prototype_per_id": int(args.prototype_per_id),
            "prototype_hard_k": int(args.prototype_hard_k),
            "itself_enabled_for_host_iapr": bool(use_itself_scoring),
            "itself_lambda": float(args.itself_lambda),
        },
        "output_naming": {
            "visual_space": "case_XXX_visual_space.{pdf,png}",
            "text_space": "case_XXX_text_space.{pdf,png}",
            "metadata": "case_XXX_metadata.json",
        },
        "retrieval_scoring": {
            "mode": host_retrieval_mode,
            "vanilla_mode": vanilla_retrieval_mode,
            "host_mode": host_retrieval_mode,
            "iapr_mode": iapr_retrieval_mode,
            "itself_enabled_for_host_iapr": bool(use_itself_scoring),
            "itself_lambda": float(args.itself_lambda),
            "itself_formula": "lambda*s_grab + (1-lambda)*s_global",
            "legacy_global_grab_alpha": float(args.retrieval_alpha),
            "score_weight_used": float(retrieval_score_weight),
            "host_margin_max": float(args.host_margin_max),
            "min_delta_margin": float(args.min_delta_margin),
            "min_iapr_margin": float(args.min_iapr_margin),
            "margin_formula": "max_positive_similarity - max_hard_negative_similarity",
            "note": "When --itself is set, Host/IAPR retrieval uses the ITSELF score and Vanilla CLIP remains global-only.",
        },
        "rank1_before_plotting": rank1_before_plotting,
        "model_architecture": {
            "only_global": bool(model_args.only_global),
            "prototype_feature": str(model_args.prototype_feature),
            "return_all": bool(model_args.return_all),
            "topk_type": str(model_args.topk_type),
            "modify_k": bool(model_args.modify_k),
            "note": (
                "Prototype-space diagnostic projections remain global; when ITSELF retrieval is enabled the model is built with only_global=False, return_all=True, topk_type=custom, and modify_k=True."
            ),
        },
        "shared_iapr_prototype_anchors_for_diagnostic": bool(shared_iapr_anchors),
        "diagnostic_anchor_space": {
            "projectors": "IAPR projection heads are used for Vanilla, Host, and IAPR projected bundles",
            "prototype_bank": "IAPR prototype bank is used for IAPR-panel prototypes and translated-centroid construction",
            "projection_decision": diagnostic_projection_decision,
            "projection_fit_scope": {
                "visual_space": "one shared 2D fit over Vanilla + Host + IAPR visual-space points only",
                "text_space": "one shared 2D fit over Vanilla + Host + IAPR text-space points only",
            },
            "visual_space_translation": "positive text evidence is translated to visual space only through text_to_image",
            "text_space_translation": "positive image evidence is translated to text space only through image_to_text",
        },
        "host_projection_decision": "Host features projected by IAPR projection heads for shared diagnostic comparability",
        "host_own_bank_available_but_unused": bool(host_own_bank_available),
        "host_bank_reason": host_bank_reason,
        "prototype_memory": {
            "source_path": iapr_bank.source_path,
            "dim": iapr_bank.dim,
            "total": iapr_bank.total,
            "prototypes_per_id": iapr_bank.prototypes_per_id,
            "num_classes": iapr_bank.num_classes,
            "key_map": iapr_bank.key_map,
            "config": iapr_bank.config,
        },
        "iapr_memory_loaded_into_branch_shapes": iapr_memory_shapes,
        "checkpoint_load_reports": {
            "vanilla": {"path": None, "loaded": 0, "note": "no fine-tuned checkpoint loaded"},
            "host": host_report.to_json(),
            "host_auxiliary": host_aux_report.to_json() if host_aux_report else None,
            "iapr": iapr_report.to_json(),
            "iapr_auxiliary": iapr_aux_report.to_json() if iapr_aux_report else None,
        },
        "selection_counters": selection_counters,
        "cases": selected_cases,
        "conceptual_guardrail": (
            "Raw cross-modal embeddings are not directly mixed. The visual-space translated centroid is built through text_to_image, "
            "and the text-space translated centroid is built through image_to_text."
        ),
        "key_mapping_notes": {
            "raw_visual_prototypes": "image_prototypes",
            "raw_text_prototypes": "text_prototypes",
            "text_to_visual_translated_prototypes": "text_to_image",
            "visual_to_text_translated_prototypes": "image_to_text",
            "prototype_identity_ids": "proto_pids",
            "mapping_function_to_edit": "choose_memory_tensor() in this script; candidate_state_keys() in scripts/plot_ambiguity_rate_compare.py",
        },
    }
    save_json(summary, output_dir / "summary.json")
    print(f"[Summary] wrote {output_dir / 'summary.json'}")
    cleanup_model(vanilla_model, device)
    cleanup_model(host_model, device)
    cleanup_model(iapr_model, device)


if __name__ == "__main__":
    main()
