#!/usr/bin/env python3
"""Dual-space local embedding repair diagnostic for IAPR/TBPS results.

Raw cross-modal embeddings are not directly mixed. In the visual-side panel,
text evidence is represented through text-to-visual translated prototypes. In
the text-side panel, image evidence is represented through visual-to-text
translated prototypes.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

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



TITLE = "Dual-Space Local Embedding Repair around Identity-Level Ambiguity"
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
NEG_COLORS = ["#D95F02", "#E15759", "#B07AA1", "#A6761D", "#CC6677"]
QUERY_COLOR = "#111111"
PROTO_EDGE = "#222222"


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
        raise ValueError(f"Unknown retrieval mode: {self.mode}")


@dataclass
class RowContext:
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
    parser.add_argument("--top_pos", type=int, default=3)
    parser.add_argument("--top_hard_neg", type=int, default=5)
    parser.add_argument("--prototype_per_id", type=int, default=2)
    parser.add_argument("--projection", default="pca", choices=["pca", "umap"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--retrieval_branch", default="auto", choices=["auto", "global", "grab", "global+grab"])
    parser.add_argument("--retrieval_alpha", type=float, default=0.68)
    parser.add_argument("--host_margin_max", type=float, default=0.05)
    parser.add_argument("--min_delta_margin", type=float, default=0.0)
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
    cfg["only_global"] = True
    cfg["prototype_feature"] = "global"
    cfg["return_all"] = False
    cfg["modify_k"] = False
    cfg["topk_type"] = "mean"
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
    if bool(getattr(model_args, "only_global", False)):
        if cli.retrieval_branch not in ("auto", "global"):
            print(f"[Retrieval] overriding --retrieval_branch={cli.retrieval_branch} to global because only_global=True")
        return "global"
    if cli.retrieval_branch != "auto":
        mode = cli.retrieval_branch
    elif bool(getattr(model_args, "only_global", False)) or not has_local_layers(model):
        mode = "global"
    else:
        mode = "global+grab"
    if mode in ("grab", "global+grab"):
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
    if mode in ("global", "global+grab"):
        text_global, query_pids = extract_text_kind(model, split_data, text_length, batch_size, num_workers, device, "global", f"{label} text global")
        image_global, gallery_pids = extract_image_kind(model, split_data, img_size, batch_size, num_workers, device, "global", f"{label} image global")
    if mode in ("grab", "global+grab"):
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


def choose_slots_for_pid(bank: PrototypeBankData, pid: int, required_global: Iterable[int], max_slots: int) -> List[Tuple[int, int]]:
    all_slots = pid_slot_indices(bank, pid)
    if not all_slots:
        return []
    selected: List[Tuple[int, int]] = []
    used = set()
    by_global = {global_idx: (slot, global_idx) for slot, global_idx in all_slots}
    for global_idx in [int(x) for x in required_global]:
        if global_idx in by_global and global_idx not in used:
            selected.append(by_global[global_idx])
            used.add(global_idx)
    for slot, global_idx in all_slots:
        if len(selected) >= max(int(max_slots), 1):
            break
        if global_idx not in used:
            selected.append((slot, global_idx))
            used.add(global_idx)
    return selected[: max(int(max_slots), 1)]


def hard_negative_color(pid: int, neg_pids: Sequence[int]) -> str:
    ordered: List[int] = []
    for item in neg_pids:
        if int(item) not in ordered:
            ordered.append(int(item))
    try:
        idx = ordered.index(int(pid))
    except ValueError:
        idx = 0
    return NEG_COLORS[idx % len(NEG_COLORS)]


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


def pad_limits(ax: Any, coords: np.ndarray) -> None:
    if coords.size == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    pad = span * 0.16
    ax.set_xlim(float(mins[0] - pad[0]), float(maxs[0] + pad[0]))
    ax.set_ylim(float(mins[1] - pad[1]), float(maxs[1] + pad[1]))


def scatter_point(ax: Any, point: PointSpec, xy: np.ndarray) -> None:
    if point.marker == "x":
        ax.scatter([xy[0]], [xy[1]], marker="x", s=point.size, c=point.color, linewidths=1.5, alpha=point.alpha)
    elif point.hollow:
        ax.scatter(
            [xy[0]],
            [xy[1]],
            marker=point.marker,
            s=point.size,
            facecolors="none",
            edgecolors=point.color,
            linewidths=1.3,
            alpha=point.alpha,
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
            alpha=point.alpha,
        )
    if point.annotate:
        ax.annotate(point.annotate, (xy[0], xy[1]), xytext=(4, 4), textcoords="offset points", fontsize=6.6)


def draw_panel(ax: Any, panel: PanelSpec, method: str) -> Dict[str, Any]:
    coords = local_project(panel.points, method)
    for point, xy in zip(panel.points, coords):
        scatter_point(ax, point, xy)
    ax.set_title(panel.title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    pad_limits(ax, coords)
    raw_vectors = np.stack([p.vector.detach().float().cpu().numpy() for p in panel.points], axis=0)
    return {
        "num_points": int(len(panel.points)),
        "unique_points": int(vector_unique_count(raw_vectors)),
        "point_labels": [p.label for p in panel.points],
    }


def legend_handles() -> Tuple[List[Any], List[str]]:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="*", color="w", label="query anchor", markerfacecolor=QUERY_COLOR, markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="positive image/anchor", markerfacecolor=POS_COLOR, markeredgecolor=PROTO_EDGE, markersize=6),
        Line2D([0], [0], marker="x", color=NEG_COLORS[0], label="hard-negative image/anchor", markersize=7, linestyle="None"),
        Line2D([0], [0], marker="D", color="w", label="positive raw prototype", markerfacecolor="none", markeredgecolor=POS_COLOR, markersize=6),
        Line2D([0], [0], marker="^", color="w", label="hard-negative raw prototype", markerfacecolor="none", markeredgecolor=NEG_COLORS[0], markersize=6),
    ]
    return handles, [h.get_label() for h in handles]


def build_row_panels(
    ctx: RowContext,
    query_index: int,
    query_pid: int,
    selected_positive: Sequence[int],
    selected_negative: Sequence[int],
    gallery_pids: torch.Tensor,
    max_prototypes_per_id: int,
) -> Tuple[PanelSpec, PanelSpec, Dict[str, Any]]:
    bank = ctx.bank
    q_text = ctx.projected.text_features[query_index]
    query_text_assignment = assign_to_pid_slots(q_text, bank.text_prototypes, bank, query_pid)
    q_anchor_visual = bank.text_to_image[query_text_assignment["global_index"]]

    image_assignments: Dict[int, Dict[str, Any]] = {}
    required_by_pid: Dict[int, List[int]] = defaultdict(list)
    required_by_pid[int(query_pid)].append(int(query_text_assignment["global_index"]))
    for image_index in list(selected_positive) + list(selected_negative):
        pid = int(gallery_pids[image_index].item())
        assign = assign_to_pid_slots(ctx.projected.image_features[image_index], bank.image_prototypes, bank, pid)
        image_assignments[int(image_index)] = assign
        required_by_pid[pid].append(int(assign["global_index"]))

    neg_pids = [int(gallery_pids[i].item()) for i in selected_negative]
    pids_to_plot = [int(query_pid)] + [pid for pid in neg_pids if pid != int(query_pid)]
    pids_to_plot = list(dict.fromkeys(pids_to_plot))

    selected_slots: Dict[int, List[Tuple[int, int]]] = {}
    for pid in pids_to_plot:
        selected_slots[pid] = choose_slots_for_pid(bank, pid, required_by_pid.get(pid, []), max_prototypes_per_id)

    visual_points: List[PointSpec] = [
        PointSpec(q_anchor_visual, "query_t2v_anchor", "query t->v", "*", QUERY_COLOR, size=150, annotate="q t->v")
    ]
    for image_index in selected_positive:
        visual_points.append(
            PointSpec(
                ctx.projected.image_features[image_index],
                "positive_image_embedding",
                f"pos image {image_index}",
                "o",
                POS_COLOR,
                size=54,
                metadata={"gallery_index": int(image_index), "pid": int(gallery_pids[image_index].item())},
            )
        )
    for image_index in selected_negative:
        pid = int(gallery_pids[image_index].item())
        visual_points.append(
            PointSpec(
                ctx.projected.image_features[image_index],
                "hard_negative_image_embedding",
                f"neg image {image_index}",
                "x",
                hard_negative_color(pid, neg_pids),
                size=58,
                metadata={"gallery_index": int(image_index), "pid": pid},
            )
        )
    for pid, slots in selected_slots.items():
        is_pos = int(pid) == int(query_pid)
        for local_slot, global_idx in slots:
            color = POS_COLOR if is_pos else hard_negative_color(pid, neg_pids)
            visual_points.append(
                PointSpec(
                    bank.image_prototypes[global_idx],
                    "positive_visual_prototype" if is_pos else "hard_negative_visual_prototype",
                    f"visual proto pid={pid} r={local_slot}",
                    "D" if is_pos else "^",
                    color,
                    size=62,
                    hollow=not is_pos,
                    annotate=f"P+ r={local_slot}" if is_pos else f"P- {pid} r={local_slot}",
                    metadata={"pid": int(pid), "local_slot": int(local_slot), "global_index": int(global_idx)},
                )
            )

    text_points: List[PointSpec] = [
        PointSpec(q_text, "query_text_embedding", "query text", "*", QUERY_COLOR, size=150, annotate="q text")
    ]
    for image_index in selected_positive:
        assign = image_assignments[int(image_index)]
        text_points.append(
            PointSpec(
                bank.image_to_text[int(assign["global_index"])],
                "positive_v2t_anchor",
                f"pos image-mode anchor {image_index}",
                "o",
                POS_COLOR,
                size=54,
                metadata={"gallery_index": int(image_index), "assignment": assign},
            )
        )
    for image_index in selected_negative:
        pid = int(gallery_pids[image_index].item())
        assign = image_assignments[int(image_index)]
        text_points.append(
            PointSpec(
                bank.image_to_text[int(assign["global_index"])],
                "hard_negative_v2t_anchor",
                f"neg image-mode anchor {image_index}",
                "x",
                hard_negative_color(pid, neg_pids),
                size=58,
                metadata={"gallery_index": int(image_index), "pid": pid, "assignment": assign},
            )
        )
    for pid, slots in selected_slots.items():
        is_pos = int(pid) == int(query_pid)
        for local_slot, global_idx in slots:
            color = POS_COLOR if is_pos else hard_negative_color(pid, neg_pids)
            text_points.append(
                PointSpec(
                    bank.text_prototypes[global_idx],
                    "positive_text_prototype" if is_pos else "hard_negative_text_prototype",
                    f"text proto pid={pid} r={local_slot}",
                    "D" if is_pos else "^",
                    color,
                    size=62,
                    hollow=not is_pos,
                    annotate=f"P+ r={local_slot}" if is_pos else f"P- {pid} r={local_slot}",
                    metadata={"pid": int(pid), "local_slot": int(local_slot), "global_index": int(global_idx)},
                )
            )

    metadata = {
        "row_label": ctx.label,
        "bank_source": ctx.bank_source,
        "projection_decision": ctx.projection_decision,
        "shared_iapr_anchors": bool(ctx.shared_iapr_anchors),
        "prototype_feature_source": ctx.projected.feature_source,
        "query_text_assignment_slot": query_text_assignment,
        "image_visual_assignment_slots": {str(k): v for k, v in sorted(image_assignments.items())},
        "plotted_prototype_slots": {
            str(pid): [
                {"local_slot": int(local_slot), "global_index": int(global_idx)}
                for local_slot, global_idx in slots
            ]
            for pid, slots in selected_slots.items()
        },
    }
    return PanelSpec(f"{ctx.label} visual-side space", visual_points), PanelSpec(f"{ctx.label} text-side space", text_points), metadata


def image_record(index: int, split_data: SplitData, gallery_pids: torch.Tensor, sim_host: torch.Tensor, sim_iapr: torch.Tensor, query_index: int) -> Dict[str, Any]:
    return {
        "gallery_index": int(index),
        "pid": int(gallery_pids[index].item()),
        "path": str(split_data.img_paths[index]),
        "host_similarity": float(sim_host[query_index, index].item()),
        "iapr_similarity": float(sim_iapr[query_index, index].item()),
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
        if host_margin > float(args.host_margin_max):
            counters["host_margin_not_low"] += 1
            continue
        if delta_margin <= float(args.min_delta_margin):
            counters["delta_margin_too_small"] += 1
            continue

        pos_scores = torch.maximum(sim_host[query_index], sim_iapr[query_index])
        selected_positive = topk_from_mask(pos_scores, pos_mask, args.top_pos)
        hard_negative_pool = topk_from_mask(sim_host[query_index], neg_mask, max(args.top_hard_neg * 5, args.top_hard_neg))
        selected_negative: List[int] = []
        for index in hard_negative_pool:
            pid = int(gallery_pids[index].item())
            if not pid_has_slots(bank, pid):
                counters["hard_negative_pid_missing_from_prototype_bank"] += 1
                continue
            if not Path(split_data.img_paths[index]).is_file():
                counters["hard_negative_path_missing"] += 1
                continue
            selected_negative.append(int(index))
            if len(selected_negative) >= int(args.top_hard_neg):
                break
        selected_positive = [idx for idx in selected_positive if Path(split_data.img_paths[idx]).is_file()]
        if not selected_positive or not selected_negative:
            counters["no_selected_positive_or_negative_after_filter"] += 1
            continue
        if not passes_prototype_pid_filter(bank, query_pid, gallery_pids, selected_positive + selected_negative):
            counters["selected_pid_missing_from_bank"] += 1
            continue

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
                "selected_positive": selected_positive,
                "selected_negative": selected_negative,
            }
        )
    candidates.sort(key=lambda row: (float(row["delta_margin"]), -float(row["host_margin"])), reverse=True)
    return candidates, dict(counters)


def render_case(
    case_id: int,
    candidate: Mapping[str, Any],
    host_ctx: RowContext,
    iapr_ctx: RowContext,
    split_data: SplitData,
    gallery_pids: torch.Tensor,
    sim_host: torch.Tensor,
    sim_iapr: torch.Tensor,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to draw the dual-space repair figure.") from exc

    query_index = int(candidate["query_index"])
    query_pid = int(candidate["query_pid"])
    selected_positive = [int(x) for x in candidate["selected_positive"]]
    selected_negative = [int(x) for x in candidate["selected_negative"]]

    host_visual, host_text, host_meta = build_row_panels(host_ctx, query_index, query_pid, selected_positive, selected_negative, gallery_pids, args.prototype_per_id)
    iapr_visual, iapr_text, iapr_meta = build_row_panels(iapr_ctx, query_index, query_pid, selected_positive, selected_negative, gallery_pids, args.prototype_per_id)

    panels = [host_visual, host_text, iapr_visual, iapr_text]
    panel_projection_meta: Dict[str, Any] = {}
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    axes_flat = axes.flatten()
    labels = ["a", "b", "c", "d"]
    for axis, label, panel in zip(axes_flat, labels, panels):
        projection_meta = draw_panel(axis, panel, args.projection)
        axis.text(0.01, 0.99, f"({label})", transform=axis.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")
        panel_projection_meta[label] = projection_meta

    title_text = truncate_text(str(candidate["query_text"]), chars=135)
    wrapped = "\n".join(textwrap.wrap(title_text, width=96, break_long_words=False))
    fig.suptitle(
        f"{TITLE}\n"
        f"q={query_index} pid={query_pid} | host margin={float(candidate['host_margin']):.4f}, "
        f"IAPR margin={float(candidate['iapr_margin']):.4f}, delta={float(candidate['delta_margin']):.4f}\n"
        f"{wrapped}",
        fontsize=11,
    )
    handles, labels_text = legend_handles()
    fig.legend(handles, labels_text, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))

    pdf_path = output_dir / f"case_{case_id:03d}_dual_space_repair.pdf"
    png_path = output_dir / f"case_{case_id:03d}_dual_space_repair.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "query_index": query_index,
        "query_text": str(candidate["query_text"]),
        "query_pid": query_pid,
        "host_margin": float(candidate["host_margin"]),
        "iapr_margin": float(candidate["iapr_margin"]),
        "delta_margin": float(candidate["delta_margin"]),
        "host_positive_score": float(candidate["host_pos_score"]),
        "host_hard_negative_score": float(candidate["host_neg_score"]),
        "iapr_positive_score": float(candidate["iapr_pos_score"]),
        "iapr_hard_negative_score": float(candidate["iapr_neg_score"]),
        "selected_positive_images": [image_record(idx, split_data, gallery_pids, sim_host, sim_iapr, query_index) for idx in selected_positive],
        "selected_hard_negative_images": [image_record(idx, split_data, gallery_pids, sim_host, sim_iapr, query_index) for idx in selected_negative],
        "hard_negative_pids": [int(gallery_pids[idx].item()) for idx in selected_negative],
        "selected_prototype_slots": {"host_row": host_meta, "iapr_row": iapr_meta},
        "similarities_used_for_ranking": {
            "margin_formula": "max_positive_similarity - max_hard_negative_similarity",
            "host_best_positive_index": int(candidate["host_best_pos_index"]),
            "host_hard_negative_index": int(candidate["host_hard_neg_index"]),
            "iapr_best_positive_index": int(candidate["iapr_best_pos_index"]),
            "iapr_hard_negative_index": int(candidate["iapr_hard_neg_index"]),
        },
        "projection": {"method": args.projection, "fit_scope": "separate local fit per panel", "panels": panel_projection_meta},
        "output_pdf": str(pdf_path),
        "output_png": str(png_path),
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
    if not (0.0 <= float(args.retrieval_alpha) <= 1.0):
        raise ValueError("--retrieval_alpha must be in [0, 1].")

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

    host_bank = load_prototype_bank(host_ckpt, required=False)
    host_uses_own_bank = False
    host_bank_reason = "host checkpoint has no compatible prototype memory"
    if host_bank is not None:
        try:
            host_projectors_ok = projector_weights_loaded(host_branch, host_report)
            if not host_projectors_ok:
                host_bank_reason = "host prototype memory exists but compatible projection heads were not loaded"
            else:
                load_memory_into_branch(host_branch, host_bank)
                host_uses_own_bank = True
                host_bank_reason = "host checkpoint contains compatible prototype memory and projectors"
        except Exception as exc:
            host_uses_own_bank = False
            host_bank_reason = f"host prototype memory was found but not usable: {exc}"

    shared_iapr_anchors = not host_uses_own_bank
    print(f"[Host prototype anchors] {'own host bank' if host_uses_own_bank else 'shared IAPR bank'} ({host_bank_reason})")

    host_retrieval_mode = resolve_retrieval_mode(args, host_model, model_args)
    iapr_retrieval_mode = resolve_retrieval_mode(args, iapr_model, model_args)
    if host_retrieval_mode != iapr_retrieval_mode:
        raise RuntimeError(f"Host and IAPR retrieval modes differ: {host_retrieval_mode} vs {iapr_retrieval_mode}")
    print(f"[Retrieval] mode={host_retrieval_mode} alpha={args.retrieval_alpha}")

    host_retrieval = extract_retrieval_bundle(
        "Host", host_model, split_data, model_args, host_retrieval_mode, args.retrieval_alpha, args.batch_size, args.num_workers, device
    )
    iapr_retrieval = extract_retrieval_bundle(
        "IAPR", iapr_model, split_data, model_args, iapr_retrieval_mode, args.retrieval_alpha, args.batch_size, args.num_workers, device
    )
    if not torch.equal(host_retrieval.query_pids, iapr_retrieval.query_pids):
        raise RuntimeError("Host and IAPR query pids differ.")
    if not torch.equal(host_retrieval.gallery_pids, iapr_retrieval.gallery_pids):
        raise RuntimeError("Host and IAPR gallery pids differ.")
    sim_host = host_retrieval.similarity().cpu()
    sim_iapr = iapr_retrieval.similarity().cpu()

    projection_branch_for_host = host_branch if host_uses_own_bank else iapr_branch
    bank_for_host = host_bank if host_uses_own_bank and host_bank is not None else iapr_bank
    projection_decision = (
        "host checkpoint prototype projectors and memory"
        if host_uses_own_bank
        else "IAPR projection heads and IAPR prototype bank used as fixed diagnostic anchors for host features"
    )

    host_projected = extract_projected_bundle(
        "Host",
        host_model,
        projection_branch_for_host,
        split_data,
        model_args,
        "host" if host_uses_own_bank else "iapr_diagnostic_projector",
        args.batch_size,
        args.num_workers,
        device,
    )
    iapr_projected = extract_projected_bundle("IAPR", iapr_model, iapr_branch, split_data, model_args, "iapr", args.batch_size, args.num_workers, device)
    if not torch.equal(host_projected.query_pids, host_retrieval.query_pids):
        raise RuntimeError("Host projected query pids differ from retrieval query pids.")
    if not torch.equal(iapr_projected.gallery_pids, iapr_retrieval.gallery_pids):
        raise RuntimeError("IAPR projected gallery pids differ from retrieval gallery pids.")

    host_ctx = RowContext(
        label="HOST",
        projected=host_projected,
        bank=bank_for_host,
        branch=projection_branch_for_host,
        bank_source=bank_for_host.source_path,
        projection_decision=projection_decision,
        shared_iapr_anchors=shared_iapr_anchors,
    )
    iapr_ctx = RowContext(
        label="IAPR",
        projected=iapr_projected,
        bank=iapr_bank,
        branch=iapr_branch,
        bank_source=iapr_bank.source_path,
        projection_decision="IAPR projection heads and IAPR prototype bank",
        shared_iapr_anchors=False,
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
                host_ctx,
                iapr_ctx,
                split_data,
                host_retrieval.gallery_pids,
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
        "host_ckpt": str(host_ckpt),
        "iapr_ckpt": str(iapr_ckpt),
        "prototype_ckpt": str(prototype_path),
        "prototype_projector_ckpt": str(prototype_projector_path),
        "num_cases_requested": int(args.num_cases),
        "num_cases_rendered": int(len(selected_cases)),
        "skipped_render_cases": int(skipped_render),
        "retrieval_scoring": {
            "mode": host_retrieval_mode,
            "alpha": float(args.retrieval_alpha),
            "margin_formula": "max_positive_similarity - max_hard_negative_similarity",
            "note": "This follows the repository margin diagnostic convention.",
        },
        "model_architecture": {
            "only_global": bool(model_args.only_global),
            "prototype_feature": str(model_args.prototype_feature),
            "note": "Backbone/prototype-space feature extraction is forced to global-only in this diagnostic.",
        },
        "shared_iapr_prototype_anchors_for_diagnostic": bool(shared_iapr_anchors),
        "host_projection_decision": projection_decision,
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
            "host": host_report.to_json(),
            "host_auxiliary": host_aux_report.to_json() if host_aux_report else None,
            "iapr": iapr_report.to_json(),
            "iapr_auxiliary": iapr_aux_report.to_json() if iapr_aux_report else None,
        },
        "selection_counters": selection_counters,
        "cases": selected_cases,
        "conceptual_guardrail": (
            "Raw cross-modal embeddings are not directly mixed. In the visual-side panel, text evidence is represented "
            "through text-to-visual translated prototypes. In the text-side panel, image evidence is represented "
            "through visual-to-text translated prototypes."
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
    cleanup_model(host_model, device)
    cleanup_model(iapr_model, device)


if __name__ == "__main__":
    main()
