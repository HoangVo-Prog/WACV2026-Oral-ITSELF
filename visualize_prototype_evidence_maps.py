#!/usr/bin/env python3
"""Prototype-Conditioned Evidence Maps.

Probe the trained prototype/IAPR branch with the final saved EMA prototype
memory. This script does not recompute clusters, reassign samples, retrieve
nearest whole images/captions, or use cosine nearest-neighbor explanations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from PIL import Image, ImageDraw, ImageFile, ImageOps
    PIL_AVAILABLE = True
except ModuleNotFoundError:
    Image = ImageDraw = ImageFile = ImageOps = None
    PIL_AVAILABLE = False

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if ImageFile is not None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

from visualize_prototype_modes import (  # noqa: E402
    Logger, PrototypeBank, Sample, apply_pid_shift, candidate_keys,
    checkpoint_state_dict, ckpt_config, device_from_arg, font, infer_k,
    load_annotations, load_prototypes, make_tokenizer, overview,
    preprocess_image, resolve_path, save_json, scalar_int, split_alignment,
    tokenize_text, torch_load, wrap,
)

TITLE = "Prototype-Conditioned Evidence Maps"
FIGURE_CAPTION = (
    "Prototype-conditioned evidence maps. We probe the trained prototype branch using the final "
    "EMA-updated prototype bank without recomputing clusters or reassigning samples. For each "
    "image-text pair, we visualize the attention mass induced by each identity-owned prototype "
    "slot over visual regions and text tokens. Different slots highlight complementary localized "
    "evidence, such as upper-body attributes, carried objects, or lower-body appearance."
)
CSV_FIELDS = [
    "pid", "sample_index", "image_path", "caption", "prototype_slot", "prototype_index",
    "gate", "visual_entropy", "text_entropy", "top_visual_regions", "top_tokens",
    "evidence_score", "figure_path",
]


@dataclass
class PairSample:
    split_index: int
    sample_index: int
    pid: int
    image_path: str
    caption: str
    caption_index: int


@dataclass
class ScoreRecord:
    pair_pos: int
    pid: int
    sample_index: int
    contribution_score: float
    attention_concentration: float
    inter_prototype_diversity: float
    visual_entropy: float
    text_entropy: float
    evidence_score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Render {TITLE}.")
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--prototype_ckpt", required=True)
    p.add_argument("--prototype_branch_ckpt", default="", help="Path to best_prototype_branch.pth containing prototype projector/branch weights. If omitted, the script infers it next to --prototype_ckpt or --model_ckpt.")
    p.add_argument("--projector_ckpt", default="", help="Alias for --prototype_branch_ckpt.")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_name", default="auto", choices=["auto", "RSTPReid", "CUHK-PEDES", "ICFG-PEDES"])
    p.add_argument("--train_anno", default="")
    p.add_argument("--val_anno", default="")
    p.add_argument("--test_anno", default="")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_pairs", type=int, default=12)
    p.add_argument("--ids", nargs="*", type=int, default=None)
    p.add_argument("--sample_indices", nargs="*", type=int, default=None)
    p.add_argument("--prototype_per_id", type=int, default=None, help="Number of identity-owned slots to render per sample.")
    p.add_argument("--max_prototypes_per_id", type=int, default=None)
    p.add_argument("--top_tokens", type=int, default=8)
    p.add_argument("--image_size", type=int, default=160)
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_raw_tensors", action="store_true")
    p.add_argument("--save_raw_heads", action="store_true", help="Save per-head CLIP attentions in raw_evidence_tensors.pt.")
    p.add_argument("--debug_shapes", action="store_true")
    p.add_argument("--force_split", action="store_true")
    p.add_argument("--max_rows", type=int, default=12)
    p.add_argument("--pretrain_choice", default="ViT-B/16")
    p.add_argument("--model_img_size", nargs=2, type=int, default=[384, 128], metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--stride_size", type=int, default=16)
    p.add_argument("--text_length", type=int, default=77)
    p.add_argument("--select_ratio", type=float, default=0.4)
    p.add_argument("--prototype_feature", default="auto", choices=["auto", "local", "global"])
    p.add_argument("--prototype_projector", default="default")
    p.add_argument("--prototype_dim", type=int, default=None)
    p.add_argument("--no_pbt", action="store_true")
    p.add_argument("--only_global", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--return_all", action="store_true")
    p.add_argument("--topk_type", default="mean", choices=["mean", "std", "layer_index", "custom", "last"])
    p.add_argument("--layer_index", type=int, default=-1)
    return p.parse_args()


def as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def bank_k(bank: PrototypeBank) -> int:
    return infer_k(bank.proto_pids)


def bank_num_classes(bank: PrototypeBank) -> int:
    k = bank_k(bank)
    n = int(bank.primary.shape[0])
    cfg_classes = scalar_int(bank.config.get("num_classes"))
    if cfg_classes and cfg_classes * k == n:
        return cfg_classes
    if k > 0 and n % k == 0:
        return n // k
    return max(int(bank.proto_pids.max().item()) + 1 if bank.proto_pids.numel() else 1, 1)


def render_slot_count(args: argparse.Namespace, bank: PrototypeBank, logger: Logger) -> int:
    k = bank_k(bank)
    requested = args.max_prototypes_per_id or args.prototype_per_id or k
    if requested <= 0:
        raise ValueError("Requested prototype slot count must be positive.")
    if requested > k:
        logger.log(f"WARNING: requested {requested} slots per identity, but bank has {k}; rendering {k}.")
        requested = k
    if args.prototype_per_id is not None and args.prototype_per_id != k:
        logger.log(f"Prototype bank has {k} slots per identity; --prototype_per_id={args.prototype_per_id} is treated as a render request, not a memory reshape.")
    return int(requested)


def make_model_args(args: argparse.Namespace, bank: PrototypeBank) -> SimpleNamespace:
    cfg = bank.config
    cfg_use_local = as_bool(cfg.get("use_local"))
    cfg_feature_dim = scalar_int(cfg.get("feature_dim"))
    if args.prototype_feature != "auto":
        feature = args.prototype_feature
    elif cfg_use_local is True or cfg_feature_dim == 4096:
        feature = "local"
    elif cfg_use_local is False:
        feature = "global"
    else:
        feature = "auto"
    only_global = bool(args.only_global) if args.only_global is not None else False
    if feature == "local":
        only_global = False
    return SimpleNamespace(
        loss_names="tal+cid", pretrain_choice=args.pretrain_choice, img_size=tuple(args.model_img_size),
        stride_size=args.stride_size, temperature=0.02, prototype=True, use_loss_id=False,
        no_pbt=bool(args.no_pbt or as_bool(cfg.get("no_pbt")) is True), prototype_feature=feature,
        prototype_dim=int(args.prototype_dim or scalar_int(cfg.get("prototype_dim")) or bank.dim),
        prototype_per_id=bank_k(bank), prototype_projector=str(cfg.get("projector_mode", args.prototype_projector)),
        prototype_residual_scale=float(cfg.get("prototype_residual_scale", 0.1)), prototype_kmeans_iters=0,
        prototype_warmup_epochs=0, prototype_tau=0.05, prototype_hard_k=16, prototype_id_weight=0.2,
        prototype_momentum=float(cfg.get("momentum", 0.2)), only_global=only_global, select_ratio=args.select_ratio,
        return_all=bool(args.return_all), topk_type="mean" if args.topk_type == "last" else args.topk_type,
        layer_index=args.layer_index, average_attn_weights=True, modify_k=False, track_train_diagnostics=False,
        training=False, txt_aug=False, img_aug=False, num_workers=args.num_workers, batch_size=args.batch_size,
        test_batch_size=args.batch_size, dataset_name=args.dataset_name if args.dataset_name != "auto" else "custom",
        root_dir=str(resolve_path(args.dataset_root)),
    )


def is_memory_key(key: str) -> bool:
    norm = str(key)
    for prefix in ("module.", "model.", "net.", "network."):
        while norm.startswith(prefix):
            norm = norm[len(prefix):]
    return norm.startswith("prototype_branch.memory.") or norm.startswith("memory.")


def load_model_checkpoint_full(model: torch.nn.Module, path: Path, logger: Logger) -> Dict[str, int]:
    state = checkpoint_state_dict(torch_load(path))
    model_state = model.state_dict()
    update: MutableMapping[str, torch.Tensor] = {}
    skipped_memory = skipped_missing = skipped_shape = skipped_non_tensor = 0
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            continue
        if is_memory_key(str(raw_key)):
            skipped_memory += 1
            continue
        target = next((c for c in candidate_keys(str(raw_key)) if c in model_state), None)
        if target is None:
            skipped_missing += 1
            continue
        if target.startswith("prototype_branch.memory."):
            skipped_memory += 1
            continue
        if model_state[target].shape != value.shape:
            skipped_shape += 1
            continue
        update[target] = value.detach().clone()
    if not update:
        raise RuntimeError(f"No compatible model tensors found in {path}")
    model_state.update(update)
    model.load_state_dict(model_state, strict=True)
    stats = {"loaded": len(update), "skipped_memory": skipped_memory, "skipped_missing": skipped_missing, "skipped_shape": skipped_shape, "skipped_non_tensor": skipped_non_tensor}
    logger.log(f"Loaded model checkpoint tensors: {stats['loaded']} (prototype_memory_skipped={skipped_memory}, missing={skipped_missing}, shape={skipped_shape}, non_tensor={skipped_non_tensor})")
    return stats


def flatten_state(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if torch.is_tensor(obj):
        out[prefix.rstrip(".") or "tensor"] = obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            out.update(flatten_state(value, f"{prefix}{key}."))
    return out


def choose_memory_tensor(flat: Mapping[str, torch.Tensor], name: str) -> Tuple[Optional[str], Optional[torch.Tensor]]:
    aliases = [name, f"prototype_bank.{name}", f"memory.{name}", f"prototype_branch.memory.{name}", f"model.prototype_branch.memory.{name}", f"module.prototype_branch.memory.{name}"]
    for alias in aliases:
        if alias in flat:
            return alias, flat[alias]
    matches = sorted(k for k in flat if k.endswith(f".{name}"))
    return (matches[0], flat[matches[0]]) if matches else (None, None)


def load_memory_exact(model: torch.nn.Module, path: Path, logger: Logger) -> Dict[str, str]:
    branch = getattr(model, "prototype_branch", None)
    if branch is None:
        raise RuntimeError("Built model has no prototype branch; refusing to visualize prototype evidence.")
    raw = torch_load(path)
    flat = flatten_state(raw)
    selected: Dict[str, str] = {}
    state: Dict[str, torch.Tensor] = {}
    for name, expected in branch.memory.state_dict().items():
        key, value = choose_memory_tensor(flat, name)
        if value is None:
            ready = bool(raw.get("prototype_ready", False)) if isinstance(raw, Mapping) else False
            if name == "initialized" and ready:
                key, value = "prototype_ready", torch.tensor(True)
            else:
                available = "\n".join(f"  {k}: {tuple(v.shape)}" for k, v in sorted(flat.items()))
                raise RuntimeError(f"Missing prototype memory tensor {name!r} in {path}. Available tensors:\n{available}")
        if tuple(value.shape) != tuple(expected.shape):
            raise RuntimeError(f"Prototype bank/model dimension mismatch for {name}: checkpoint shape={tuple(value.shape)}, model shape={tuple(expected.shape)}.")
        state[name] = value.detach().clone().to(dtype=expected.dtype)
        selected[name] = str(key)
    branch.memory.load_state_dict(state, strict=True)
    logger.log(f"Selected prototype tensor key: {selected.get('image_prototypes')}")
    logger.log(f"Selected text prototype tensor key: {selected.get('text_prototypes')}")
    logger.log(f"Selected PID mapping key: {selected.get('proto_pids')}")
    logger.log(f"Prototype memory initialized: {branch.memory.is_ready()}")
    return selected

def _unique_paths(paths: Sequence[Path]) -> List[Path]:
    out: List[Path] = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def infer_prototype_branch_ckpt(args: argparse.Namespace, logger: Logger) -> Path:
    explicit = str(args.prototype_branch_ckpt or args.projector_ckpt or "").strip()
    if explicit:
        path = resolve_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Prototype branch/projector checkpoint not found: {path}")
        logger.log(f"Prototype branch/projector checkpoint: {path}")
        return path

    prototype_path = resolve_path(args.prototype_ckpt)
    model_path = resolve_path(args.model_ckpt)
    candidates: List[Path] = []

    for source in (prototype_path, model_path):
        name = source.name
        for old, new in (
            ("best_prototype_bank", "best_prototype_branch"),
            ("prototype_bank", "prototype_branch"),
        ):
            if old in name:
                candidates.append(source.with_name(name.replace(old, new)))
        if source.stem == "best":
            candidates.append(source.with_name("best_prototype_branch" + source.suffix))
        candidates.append(source.with_name(source.stem + "_prototype_branch" + source.suffix))

    candidates.append(prototype_path.parent / "best_prototype_branch.pth")
    candidates.append(model_path.parent / "best_prototype_branch.pth")

    candidates = _unique_paths([path for path in candidates if path.name != prototype_path.name])
    for path in candidates:
        if path.is_file():
            logger.log(f"Inferred prototype branch/projector checkpoint: {path}")
            return path

    tried = "\n".join(f"  {path}" for path in candidates)
    raise FileNotFoundError(
        "Could not infer best_prototype_branch.pth. This diagnostic must load the "
        "saved prototype projector/branch checkpoint; pass --prototype_branch_ckpt explicitly.\n"
        f"Tried:\n{tried}"
    )


def merge_branch_config_into_bank(bank: PrototypeBank, raw_branch: Any, logger: Logger) -> Dict[str, Any]:
    cfg = ckpt_config(raw_branch)
    if not cfg:
        logger.log("WARNING: prototype branch checkpoint has no prototype_config metadata; using prototype bank config only.")
        return {}
    override_keys = {
        "feature_dim", "prototype_dim", "projector_mode", "prototype_residual_scale",
        "no_pbt", "use_local", "momentum",
    }
    applied: Dict[str, Any] = {}
    for key in sorted(override_keys):
        if key not in cfg or cfg[key] is None:
            continue
        old = bank.config.get(key)
        new = cfg[key]
        if old is not None and str(old) != str(new):
            logger.log(f"WARNING: prototype bank config {key}={old!r} differs from branch checkpoint {key}={new!r}; using branch value for model build.")
        bank.config[key] = new
        applied[key] = new
    logger.log(f"Prototype branch config used for projector build: {applied}")
    return applied


def prototype_branch_state_dict(raw: Any) -> Tuple[Mapping[str, Any], str]:
    if not isinstance(raw, Mapping):
        raise ValueError("Prototype branch checkpoint must be a mapping.")
    branch_state = raw.get("prototype_branch")
    if isinstance(branch_state, Mapping):
        return branch_state, "prototype_branch"
    for key in ("state_dict", "branch", "model", "model_state_dict", "net", "network", "module"):
        value = raw.get(key)
        if isinstance(value, Mapping):
            return value, key
    return raw, "top_level"


def branch_candidate_keys(key: str) -> List[str]:
    out: List[str] = []

    def add(value: str) -> None:
        if value and value not in out:
            out.append(value)

    for candidate in candidate_keys(key):
        add(candidate)
        for prefix in ("prototype_branch.", "module.prototype_branch.", "model.prototype_branch."):
            if candidate.startswith(prefix):
                add(candidate[len(prefix):])
        if not candidate.startswith("prototype_branch."):
            add(f"prototype_branch.{candidate}")
    return out


def load_prototype_branch_checkpoint(model: torch.nn.Module, path: Path, logger: Logger, raw: Optional[Any] = None) -> Dict[str, Any]:
    branch = getattr(model, "prototype_branch", None)
    if branch is None:
        raise RuntimeError("Built model has no prototype branch; cannot load prototype projector checkpoint.")
    raw = torch_load(path) if raw is None else raw
    state, source_key = prototype_branch_state_dict(raw)
    branch_state = branch.state_dict()
    update: MutableMapping[str, torch.Tensor] = {}
    loaded_projector_keys: List[str] = []
    skipped_memory = skipped_missing = skipped_shape = skipped_non_tensor = 0
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            continue
        if is_memory_key(str(raw_key)):
            skipped_memory += 1
            continue
        target = next((c for c in branch_candidate_keys(str(raw_key)) if c in branch_state), None)
        if target is None:
            skipped_missing += 1
            continue
        if target.startswith("memory."):
            skipped_memory += 1
            continue
        if tuple(branch_state[target].shape) != tuple(value.shape):
            skipped_shape += 1
            continue
        update[target] = value.detach().clone().to(dtype=branch_state[target].dtype)
        if target.startswith(("image_projector", "text_projector")):
            loaded_projector_keys.append(target)

    expected_branch_keys = [key for key in branch_state if not key.startswith("memory.")]
    expected_projector_keys = [key for key in expected_branch_keys if key.startswith(("image_projector", "text_projector"))]
    if expected_branch_keys and not update:
        available = "\n".join(str(k) for k in state.keys())
        raise RuntimeError(
            f"No compatible prototype branch/projector tensors found in {path}. "
            f"Expected branch keys like {expected_branch_keys[:8]}; available keys:\n{available}"
        )
    if expected_projector_keys and not loaded_projector_keys:
        raise RuntimeError(
            f"Prototype branch checkpoint {path} did not load any image/text projector tensors. "
            f"Expected projector keys like {expected_projector_keys[:8]}."
        )

    branch_state.update(update)
    branch.load_state_dict(branch_state, strict=True)
    stats = {
        "path": str(path),
        "source_key": source_key,
        "loaded": len(update),
        "loaded_projector": len(loaded_projector_keys),
        "loaded_projector_keys": loaded_projector_keys,
        "skipped_memory": skipped_memory,
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_non_tensor": skipped_non_tensor,
    }
    logger.log(
        "Loaded prototype branch/projector tensors: "
        f"{stats['loaded']} from {path} "
        f"(projector={stats['loaded_projector']}, source={source_key}, "
        f"memory_skipped={skipped_memory}, missing={skipped_missing}, "
        f"shape={skipped_shape}, non_tensor={skipped_non_tensor})"
    )
    if loaded_projector_keys:
        logger.log("Loaded projector keys: " + ", ".join(loaded_projector_keys[:12]))
    return stats

def build_evidence_model(args: argparse.Namespace, bank: PrototypeBank, device: torch.device, logger: Logger, prototype_branch_ckpt: Path, raw_branch: Any) -> Tuple[torch.nn.Module, SimpleNamespace, Dict[str, Any]]:
    from model import build_model as repo_build_model
    ns = make_model_args(args, bank)
    num_classes = bank_num_classes(bank)
    logger.log(f"Building model num_classes={num_classes}, only_global={ns.only_global}, prototype_enabled={ns.prototype}, prototype_feature={ns.prototype_feature}")
    model = repo_build_model(ns, num_classes=num_classes)
    model_stats = load_model_checkpoint_full(model, resolve_path(args.model_ckpt), logger)
    branch_stats = load_prototype_branch_checkpoint(model, prototype_branch_ckpt, logger, raw_branch)
    memory_keys = load_memory_exact(model, resolve_path(args.prototype_ckpt), logger)
    branch = model.prototype_branch
    logger.log(f"Prototype branch enabled: {branch is not None}; use_local={branch.use_local}; feature_dim={branch.feature_dim}; prototype_dim={branch.prototype_dim}")
    logger.log(f"Prototype memory shape: image={tuple(branch.memory.image_prototypes.shape)}, text={tuple(branch.memory.text_prototypes.shape)}, proto_pids={tuple(branch.memory.proto_pids.shape)}")
    if int(branch.prototype_dim) != int(branch.memory.dim):
        raise RuntimeError(f"Prototype branch dim {branch.prototype_dim} does not match loaded memory dim {branch.memory.dim}.")
    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model, ns, {"model_load": model_stats, "prototype_branch_load": branch_stats, "memory_keys": memory_keys}


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[PairSample], img_size: Tuple[int, int], text_length: int):
        self.pairs = list(pairs)
        self.img_size = img_size
        self.text_length = int(text_length)
        self.tokenizer = make_tokenizer()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        image = preprocess_image(pair.image_path, self.img_size)
        caption_ids = tokenize_text(pair.caption, self.tokenizer, self.text_length)
        return index, pair.sample_index, int(pair.pid), image, caption_ids


def pair_collate(rows):
    pair_pos, sample_index, pids, images, captions = zip(*rows)
    return {
        "pair_pos": torch.tensor(pair_pos, dtype=torch.long),
        "sample_index": torch.tensor(sample_index, dtype=torch.long),
        "pids": torch.tensor(pids, dtype=torch.long),
        "images": torch.stack(images, dim=0),
        "caption_ids": torch.stack(captions, dim=0),
    }


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def build_pairs(samples: Sequence[Sample], bank: PrototypeBank, ids: Optional[Sequence[int]], sample_indices: Optional[Sequence[int]], logger: Logger) -> List[PairSample]:
    proto_pids = {int(x) for x in bank.proto_pids.tolist()}
    wanted_ids = set(int(x) for x in ids) if ids else None
    wanted_samples = set(int(x) for x in sample_indices) if sample_indices else None
    pairs: List[PairSample] = []
    skipped_pid = skipped_caption = 0
    for split_index, sample in enumerate(samples):
        if wanted_samples is not None and int(sample.raw_index) not in wanted_samples and split_index not in wanted_samples:
            continue
        if wanted_ids is not None and int(sample.pid) not in wanted_ids:
            continue
        if int(sample.pid) not in proto_pids:
            skipped_pid += 1
            continue
        caption_index, caption = 0, ""
        for ci, value in enumerate(sample.captions):
            if str(value).strip():
                caption_index, caption = ci, str(value)
                break
        if not caption:
            skipped_caption += 1
        pairs.append(PairSample(split_index, int(sample.raw_index), int(sample.pid), sample.image_path, caption, caption_index))
    if wanted_samples is not None:
        order = {int(v): i for i, v in enumerate(sample_indices or [])}
        pairs.sort(key=lambda p: order.get(p.sample_index, order.get(p.split_index, 10**9)))
    logger.log(f"Candidate image-text pairs: {len(pairs)} (skipped_no_prototype_pid={skipped_pid}, empty_caption={skipped_caption})")
    return pairs


def detach_cpu(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return [detach_cpu(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(detach_cpu(x) for x in obj)
    if isinstance(obj, Mapping):
        return {k: detach_cpu(v) for k, v in obj.items()}
    return obj


def shape_summary(obj: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in obj.items():
        if torch.is_tensor(value):
            out[key] = list(value.shape)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            out[key] = [list(x.shape) for x in value]
        elif isinstance(value, tuple):
            out[key] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out


def normalized_entropy(weights: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    w = weights.detach().float().clone()
    if mask is not None:
        mask = mask.detach().bool()
        w = w * mask.to(dtype=w.dtype)
        count = int(mask.sum().item())
    else:
        count = int(w.numel())
    total = float(w.sum().item())
    if count <= 1 or total <= 1e-12:
        return 0.0
    p = (w / total).clamp_min(1e-12)
    ent = float(-(p * p.log()).sum().item())
    return max(0.0, min(1.0, ent / math.log(count)))


def js_divergence(p: torch.Tensor, q: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    p = p.detach().float().clone()
    q = q.detach().float().clone()
    if mask is not None:
        mask = mask.detach().bool()
        p = p * mask.to(dtype=p.dtype)
        q = q * mask.to(dtype=q.dtype)
    if float(p.sum().item()) <= 1e-12 or float(q.sum().item()) <= 1e-12:
        return 0.0
    p = (p / p.sum().clamp_min(1e-12)).clamp_min(1e-12)
    q = (q / q.sum().clamp_min(1e-12)).clamp_min(1e-12)
    m = 0.5 * (p + q)
    value = 0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum())
    return max(0.0, min(1.0, float((value / math.log(2.0)).item())))


def pairwise_diversity(matrix: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    vals = [js_divergence(matrix[i], matrix[j], mask=mask) for i in range(matrix.shape[0]) for j in range(i + 1, matrix.shape[0])]
    return float(np.mean(vals)) if vals else 0.0


def score_one(evidence: Mapping[str, Any], batch_index: int, pair_pos: int, pair: PairSample) -> ScoreRecord:
    slot_mask = evidence["prototype_slot_mask"][batch_index].bool()
    visual = evidence["visual_attention"][batch_index][slot_mask]
    text = evidence["text_attention"][batch_index][slot_mask]
    visual_mask = evidence.get("visual_token_mask", torch.ones(visual.shape[-1], dtype=torch.bool))[batch_index].bool()
    text_mask = evidence.get("text_token_mask", torch.ones(text.shape[-1], dtype=torch.bool))[batch_index].bool()
    contribution = evidence.get("contribution")
    contribution_score = safe_float(contribution[batch_index][slot_mask].max().item()) if torch.is_tensor(contribution) and slot_mask.any() else 0.0
    visual_entropy = float(np.mean([normalized_entropy(visual[k], visual_mask) for k in range(visual.shape[0])])) if visual.shape[0] else 1.0
    text_entropy = float(np.mean([normalized_entropy(text[k], text_mask) for k in range(text.shape[0])])) if text.shape[0] else 1.0
    concentration = max(0.0, 1.0 - 0.5 * (visual_entropy + text_entropy))
    diversity = 0.5 * (pairwise_diversity(visual, visual_mask) + pairwise_diversity(text, text_mask))
    evidence_score = float(contribution_score + concentration + diversity)
    return ScoreRecord(int(pair_pos), int(pair.pid), int(pair.sample_index), contribution_score, concentration, diversity, visual_entropy, text_entropy, evidence_score)


@torch.inference_mode()
def score_pairs(model: torch.nn.Module, pairs: Sequence[PairSample], args: argparse.Namespace, render_k: int, device: torch.device, logger: Logger) -> Tuple[List[ScoreRecord], Dict[str, Any]]:
    loader = DataLoader(PairDataset(pairs, tuple(args.model_img_size), args.text_length), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=pair_collate)
    records: List[ScoreRecord] = []
    first_shapes: Dict[str, Any] = {}
    for batch in tqdm(loader, desc="Scoring evidence pairs"):
        evidence = detach_cpu(model.collect_prototype_evidence(move_batch(batch, device), max_prototypes_per_id=render_k, include_raw_heads=False))
        if not first_shapes:
            first_shapes = shape_summary(evidence)
            if args.debug_shapes:
                logger.log("Evidence tensor shapes: " + json.dumps(first_shapes, sort_keys=True, default=str))
        if "visual_attention" not in evidence or "text_attention" not in evidence:
            raise RuntimeError("Prototype evidence extraction failed: expected visual_attention and text_attention tensors were missing. This script does not fall back to cosine nearest-neighbor retrieval.")
        for row, pair_pos in enumerate(batch["pair_pos"].tolist()):
            records.append(score_one(evidence, row, int(pair_pos), pairs[int(pair_pos)]))
    logger.log(f"Scored {len(records)} candidate pairs with contribution + concentration + diversity.")
    return records, first_shapes


def select_records(records: Sequence[ScoreRecord], args: argparse.Namespace) -> List[ScoreRecord]:
    if args.sample_indices:
        order = {int(v): i for i, v in enumerate(args.sample_indices)}
        return sorted(records, key=lambda r: order.get(r.sample_index, 10**9))
    return sorted(records, key=lambda r: (r.evidence_score, r.attention_concentration, r.inter_prototype_diversity), reverse=True)[: max(0, int(args.num_pairs))]


def image_display_size(args: argparse.Namespace) -> Tuple[int, int]:
    width = max(80, int(args.image_size))
    h, w = int(args.model_img_size[0]), int(args.model_img_size[1])
    return width, max(120, int(round(width * h / max(w, 1))))


def load_render_image(path: str, size: Tuple[int, int]) -> Image.Image:
    try:
        image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        return image.resize(size, Image.BICUBIC)
    except Exception:
        canvas = Image.new("RGB", size, "#eef1f5")
        ImageDraw.Draw(canvas).text((10, size[1] // 2), "missing image", fill="#9a1c1c", font=font(13, True))
        return canvas


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    lo = float(np.nanmin(values)) if values.size else 0.0
    hi = float(np.nanmax(values)) if values.size else 0.0
    return np.zeros_like(values, dtype=np.float32) if hi - lo <= 1e-12 else (values - lo) / (hi - lo)


def heatmap_overlay(base: Image.Image, weights: torch.Tensor, grid: Tuple[int, int], alpha: float) -> Image.Image:
    gh, gw = int(grid[0]), int(grid[1])
    if gh <= 0 or gw <= 0 or gh * gw != int(weights.numel()):
        raise ValueError("Spatial grid is not compatible with evidence length.")
    heat = normalize_array(weights.detach().float().reshape(gh, gw).cpu().numpy())
    heat_img = Image.fromarray(np.uint8(heat * 255), mode="L").resize(base.size, Image.BICUBIC)
    heat_np = np.asarray(heat_img).astype(np.float32) / 255.0
    base_np = np.asarray(base).astype(np.float32)
    color = np.zeros_like(base_np)
    color[..., 0] = 255.0
    color[..., 1] = 80.0 + 120.0 * (1.0 - heat_np)
    color[..., 2] = 30.0
    a = np.clip(float(alpha), 0.0, 1.0) * heat_np[..., None]
    return Image.fromarray(np.uint8(np.clip(base_np * (1.0 - a) + color * a, 0, 255)))


def evidence_bar_plot(weights: torch.Tensor, size: Tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    vals = weights.detach().float().cpu().numpy()
    vals = vals / max(float(vals.sum()), 1e-12)
    top = np.argsort(-vals)[: min(12, len(vals))]
    left, right, y = 18, size[0] - 18, 18
    bar_h = max(8, (size[1] - 36) // max(len(top), 1) - 4)
    for rank, idx in enumerate(top):
        score = float(vals[idx])
        width = int((right - left) * score / max(float(vals[top[0]]), 1e-12))
        draw.rectangle([left, y, left + width, y + bar_h], fill="#ef7d32" if rank == 0 else "#f4b183")
        draw.rectangle([left, y, right, y + bar_h], outline="#d1d5db")
        draw.text((left, y + bar_h + 1), f"region {int(idx)}  {score:.3f}", fill="#374151", font=font(11))
        y += bar_h + 18
    return canvas


def top_visual_regions(weights: torch.Tensor, grid: Tuple[int, int], top_n: int = 4) -> List[str]:
    vals = weights.detach().float().cpu()
    if vals.numel() == 0:
        return []
    gh, gw = int(grid[0]), int(grid[1])
    out = []
    for idx in torch.topk(vals, k=min(top_n, vals.numel())).indices.tolist():
        score = float(vals[idx].item())
        if gh > 0 and gw > 0 and gh * gw == vals.numel():
            row, col = divmod(int(idx), gw)
            band = "upper" if row < gh / 3 else ("middle" if row < 2 * gh / 3 else "lower")
            out.append(f"{band} patch r{row + 1}c{col + 1}:{score:.3f}")
        else:
            out.append(f"region {int(idx)}:{score:.3f}")
    return out


def token_labels_from_ids(token_ids: torch.Tensor, tokenizer: Any) -> List[str]:
    labels: List[str] = []
    encoder = getattr(tokenizer, "encoder", {}) if tokenizer is not None else {}
    special = {0, encoder.get("<|startoftext|>", -1), encoder.get("<|endoftext|>", -1), encoder.get("<|mask|>", -1)}
    for raw in token_ids.detach().cpu().tolist():
        idx = int(raw)
        if idx in special:
            labels.append("")
            continue
        try:
            text = tokenizer.decode([idx]).strip() if tokenizer is not None else str(idx)
        except Exception:
            text = str(idx)
        labels.append(text.replace("\n", " ").strip())
    return labels


def top_token_summary(labels: Sequence[str], scores: torch.Tensor, top_n: int) -> List[str]:
    vals = scores.detach().float().cpu()
    pairs = [(i, float(vals[i].item())) for i, label in enumerate(labels) if label]
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [f"{labels[i]}:{score:.3f}" for i, score in pairs[: max(0, int(top_n))]]


def color_for_score(score: float) -> Tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    base = np.array([255, 255, 255], dtype=np.float32)
    hot = np.array([239, 125, 50], dtype=np.float32)
    color = base * (1.0 - score) + hot * score
    return tuple(int(x) for x in color)


def draw_token_strip(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], labels: Sequence[str], scores: torch.Tensor, top_n: int) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, fill="#ffffff", outline="#d1d5db")
    arr = scores.detach().float().cpu().numpy()
    max_score = max(float(np.nanmax(arr)) if arr.size else 0.0, 1e-12)
    token_font, small_font = font(12), font(11)
    x, y, line_h, drawn = x0 + 8, y0 + 10, 25, 0
    for label, score in zip(labels, arr):
        if not label:
            continue
        label = label[:18]
        norm = math.sqrt(max(float(score), 0.0) / max_score)
        bbox = draw.textbbox((0, 0), label, font=token_font)
        tw = min(max(18, bbox[2] - bbox[0] + 12), x1 - x0 - 16)
        if x + tw > x1 - 8:
            x, y = x0 + 8, y + line_h
        if y + line_h > y1 - 24:
            draw.text((x, y), "...", fill="#6b7280", font=token_font)
            break
        draw.rounded_rectangle([x, y, x + tw, y + 19], radius=3, fill=color_for_score(norm), outline="#e5e7eb")
        draw.text((x + 6, y + 3), label, fill="#111827", font=token_font)
        x += tw + 4
        drawn += 1
    if drawn == 0:
        draw.text((x0 + 10, y0 + 10), "No caption tokens", fill="#6b7280", font=token_font)
    draw.text((x0 + 8, y1 - 18), "Top: " + "; ".join(top_token_summary(labels, scores, top_n)), fill="#374151", font=small_font)


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, width: int, max_lines: int, fill: str, fnt) -> int:
    x, y = xy
    for line in wrap(text, max(12, width // 8), max_lines):
        draw.text((x, y), line, fill=fill, font=fnt)
        y += int(getattr(fnt, "size", 12) * 1.35)
    return y


def render_pair_figure(pair: PairSample, evidence: Mapping[str, Any], batch_index: int, score: ScoreRecord, args: argparse.Namespace, out_path: Path, tokenizer: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    display_size = image_display_size(args)
    base_image = load_render_image(pair.image_path, display_size)
    slot_mask = evidence["prototype_slot_mask"][batch_index].bool()
    proto_indices = evidence["prototype_indices"][batch_index][slot_mask].tolist()
    visual = evidence["visual_attention"][batch_index][slot_mask]
    text = evidence["text_attention"][batch_index][slot_mask]
    contribution = evidence.get("contribution")
    contribution = contribution[batch_index][slot_mask] if torch.is_tensor(contribution) else torch.zeros(len(proto_indices))
    gate_available = bool(evidence.get("gate_available", False))
    gate = evidence.get("gate")
    gate_values = gate[batch_index][slot_mask] if torch.is_tensor(gate) else None
    visual_mask = evidence.get("visual_token_mask", torch.ones(visual.shape[-1], dtype=torch.bool))[batch_index].bool()
    text_mask = evidence.get("text_token_mask", torch.ones(text.shape[-1], dtype=torch.bool))[batch_index].bool()
    grid = tuple(evidence.get("image_grid", (0, 0)))
    grid_ok = int(grid[0]) > 0 and int(grid[1]) > 0 and int(grid[0]) * int(grid[1]) == int(visual.shape[-1])
    k = len(proto_indices)
    margin, gap = 22, 14
    image_w, image_h = display_size
    heat_w = image_w
    text_w = max(360, image_w * 2)
    header_h, cell_h, footer_h = 112, image_h + 112, 64
    page_w = margin * 2 + image_w + k * heat_w + text_w + gap * (k + 1)
    page_h = margin * 2 + header_h + cell_h + footer_h
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    title_f, sub_f, head_f, small_f = font(24, True), font(14), font(15, True), font(12)
    draw.text((margin, margin), TITLE, fill="#111827", font=title_f)
    draw.text((margin, margin + 34), "Final EMA prototype bank; no k-means, reassignment, or whole-sample nearest-neighbor retrieval.", fill="#374151", font=sub_f)
    draw_wrapped(draw, (margin, margin + 56), f"PID {pair.pid} / sample {pair.sample_index}: {pair.caption}", page_w - 2 * margin, 2, "#4b5563", small_f)
    y0, x = margin + header_h, margin
    draw.text((x, y0), "Original image", fill="#111827", font=head_f)
    page.paste(base_image, (x, y0 + 24))
    draw.text((x, y0 + 30 + image_h), f"score={score.evidence_score:.3f}", fill="#374151", font=small_f)
    x += image_w + gap
    rows: List[Dict[str, Any]] = []
    figure_slots: List[Dict[str, Any]] = []
    token_ids = tokenize_text(pair.caption, tokenizer, args.text_length)
    labels = token_labels_from_ids(token_ids, tokenizer)
    aggregate_text = text.max(dim=0).values if text.numel() else torch.zeros(args.text_length)
    for slot, proto_idx in enumerate(proto_indices):
        weights = visual[slot]
        visual_entropy = normalized_entropy(weights, visual_mask)
        text_entropy = normalized_entropy(text[slot], text_mask)
        rendered = heatmap_overlay(base_image, weights, grid, args.alpha) if grid_ok else evidence_bar_plot(weights, display_size)
        label_value = safe_float(gate_values[slot].item()) if gate_values is not None else safe_float(contribution[slot].item())
        label_name = "gate" if gate_available else "contribution"
        draw.text((x, y0), f"Prototype {slot}", fill="#111827", font=head_f)
        draw.text((x, y0 + 19), f"{label_name}={label_value:.3f}  entropy={visual_entropy:.3f}", fill="#374151", font=small_f)
        page.paste(rendered, (x, y0 + 44))
        top_regions = top_visual_regions(weights, grid, top_n=4)
        draw_wrapped(draw, (x, y0 + 50 + image_h), "Top visual: " + "; ".join(top_regions), heat_w, 3, "#374151", small_f)
        top_tokens = top_token_summary(labels, text[slot], args.top_tokens)
        row = {
            "pid": int(pair.pid), "sample_index": int(pair.sample_index), "image_path": pair.image_path,
            "caption": pair.caption, "prototype_slot": int(slot), "prototype_index": int(proto_idx),
            "gate": f"{label_value:.6f}", "visual_entropy": f"{visual_entropy:.6f}", "text_entropy": f"{text_entropy:.6f}",
            "top_visual_regions": "; ".join(top_regions), "top_tokens": "; ".join(top_tokens),
            "evidence_score": f"{score.evidence_score:.6f}", "figure_path": str(out_path),
        }
        rows.append(row)
        figure_slots.append(dict(row))
        x += heat_w + gap
    draw.text((x, y0), "Text token evidence", fill="#111827", font=head_f)
    draw.text((x, y0 + 19), "max over rendered prototype slots", fill="#374151", font=small_f)
    draw_token_strip(draw, (x, y0 + 44, x + text_w, y0 + 44 + image_h), labels, aggregate_text, args.top_tokens)
    draw_wrapped(draw, (x, y0 + 52 + image_h), "Caption: " + pair.caption, text_w, 3, "#374151", small_f)
    draw_wrapped(draw, (margin, page_h - footer_h), FIGURE_CAPTION, page_w - 2 * margin, 2, "#4b5563", font(11))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)
    meta = {"pid": int(pair.pid), "sample_index": int(pair.sample_index), "figure_path": str(out_path), "evidence_score": float(score.evidence_score), "prototype_slots": figure_slots}
    return rows, meta


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def single_item_evidence(evidence: Mapping[str, Any], batch_index: int, batch_size: int) -> Dict[str, Any]:
    item: Dict[str, Any] = {}
    for key, value in evidence.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
            item[key] = value[batch_index].detach().cpu()
        elif isinstance(value, list) and value and torch.is_tensor(value[0]) and value[0].ndim > 0 and value[0].shape[0] == batch_size:
            item[key] = [v[batch_index].detach().cpu() for v in value]
        else:
            item[key] = detach_cpu(value)
    return item


@torch.inference_mode()
def render_selected(model: torch.nn.Module, selected_pairs: Sequence[PairSample], selected_scores: Sequence[ScoreRecord], args: argparse.Namespace, render_k: int, outdir: Path, device: torch.device, logger: Logger) -> Tuple[List[Dict[str, Any]], List[Path], List[Dict[str, Any]], List[Dict[str, Any]]]:
    loader = DataLoader(PairDataset(selected_pairs, tuple(args.model_img_size), args.text_length), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", collate_fn=pair_collate)
    tokenizer = make_tokenizer()
    summary_rows: List[Dict[str, Any]] = []
    figure_paths: List[Path] = []
    figure_meta: List[Dict[str, Any]] = []
    raw_items: List[Dict[str, Any]] = []
    for batch in tqdm(loader, desc="Rendering selected evidence"):
        evidence = detach_cpu(model.collect_prototype_evidence(move_batch(batch, device), max_prototypes_per_id=render_k, include_raw_heads=bool(args.save_raw_heads)))
        batch_size = int(batch["pair_pos"].numel())
        for row, local_pos in enumerate(batch["pair_pos"].tolist()):
            pair = selected_pairs[int(local_pos)]
            score = selected_scores[int(local_pos)]
            out_path = outdir / f"prototype_evidence_pid_{int(pair.pid):04d}_sample_{int(pair.sample_index):04d}.png"
            rows, meta = render_pair_figure(pair, evidence, row, score, args, out_path, tokenizer)
            summary_rows.extend(rows)
            figure_paths.append(out_path)
            figure_meta.append(meta)
            logger.log(f"Saved {out_path}")
            if args.save_raw_tensors:
                item = single_item_evidence(evidence, row, batch_size)
                item.update({"pid": int(pair.pid), "sample_index": int(pair.sample_index), "image_path": pair.image_path, "caption": pair.caption, "figure_path": str(out_path)})
                raw_items.append(item)
    return summary_rows, figure_paths, figure_meta, raw_items


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    outdir = resolve_path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir)
    try:
        logger.log(TITLE)
        logger.log("Hard checks: no k-means, no dataset reassignment, no whole-sample cosine NN retrieval.")
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for rendering evidence maps.")
        save_json(vars(args), outdir / "used_config.json")
        samples_by_split = load_annotations(args, logger)
        bank = load_prototypes(resolve_path(args.prototype_ckpt), logger)
        prototype_branch_ckpt = infer_prototype_branch_ckpt(args, logger)
        raw_branch_ckpt = torch_load(prototype_branch_ckpt)
        branch_config_used = merge_branch_config_into_bank(bank, raw_branch_ckpt, logger)
        k = bank_k(bank)
        logger.log(f"Number of prototypes per identity in bank: {k}")
        render_k = render_slot_count(args, bank, logger)
        logger.log(f"Rendering prototype slots per identity: {render_k}")
        align = split_alignment(samples_by_split, bank, args.split, args.force_split, logger)
        save_json(align, outdir / "split_alignment_report.json")
        used_split = str(align["used_split"])
        if not samples_by_split.get(used_split):
            raise RuntimeError(f"No samples available for split {used_split!r}.")
        if int(align["overlap_counts"].get(used_split, 0)) <= 0:
            raise RuntimeError(f"Selected split {used_split!r} has no PID overlap with the prototype bank. Use --split train or inspect split_alignment_report.json.")
        pid_shift = int(align.get("used_pid_shift", 0))
        samples = apply_pid_shift(samples_by_split[used_split], pid_shift)
        selection_ids = args.ids
        if selection_ids and pid_shift:
            selection_ids = [int(pid) + pid_shift for pid in selection_ids]
            logger.log(f"Interpreting requested --ids after pid_shift={pid_shift}: {selection_ids}")
        device = device_from_arg(args.device)
        logger.log(f"Using device: {device}")
        model, model_ns, load_stats = build_evidence_model(args, bank, device, logger, prototype_branch_ckpt, raw_branch_ckpt)
        branch = model.prototype_branch
        if branch is None or not branch.is_ready():
            raise RuntimeError("Prototype branch is not active after checkpoint loading.")
        pairs = build_pairs(samples, bank, selection_ids, args.sample_indices, logger)
        if not pairs:
            raise RuntimeError("No candidate image-text pairs overlap with the loaded prototype PIDs.")
        records, evidence_shapes = score_pairs(model, pairs, args, render_k, device, logger)
        selected_scores = select_records(records, args)
        if not selected_scores:
            raise RuntimeError("No pairs selected for rendering.")
        selected_pairs = [pairs[r.pair_pos] for r in selected_scores]
        logger.log("Selected samples: " + ", ".join(f"pid={p.pid}/sample={p.sample_index}" for p in selected_pairs))
        rows, fig_paths, fig_meta, raw_items = render_selected(model, selected_pairs, selected_scores, args, render_k, outdir, device, logger)
        overview_path = outdir / "prototype_evidence_overview.png"
        overview(fig_paths, overview_path, args.max_rows)
        if overview_path.is_file():
            logger.log(f"Saved {overview_path}")
        csv_path = outdir / "evidence_summary.csv"
        save_csv(rows, csv_path)
        logger.log(f"Saved {csv_path}")
        visual_bank_key, text_bank_key = branch._prototype_banks_for_loss()[2:]
        metadata = {
            "title": TITLE,
            "figure_caption": FIGURE_CAPTION,
            "requested_split": args.split,
            "used_split": used_split,
            "pid_shift": pid_shift,
            "prototype_source": bank.source_path,
            "prototype_branch_source": str(prototype_branch_ckpt),
            "prototype_branch_config_used": branch_config_used,
            "prototype_dim": bank.dim,
            "bank_prototypes_per_id": k,
            "rendered_prototypes_per_id": render_k,
            "prototype_branch_enabled": branch is not None,
            "prototype_branch_ready": bool(branch.is_ready()),
            "prototype_feature_source": "local" if getattr(branch, "use_local", False) else "global",
            "gate_available": False,
            "gate_note": "No learned prototype gate exists in this implementation; contribution is diagnostic slot softmax similarity.",
            "visual_bank_key_used_by_branch": visual_bank_key,
            "text_bank_key_used_by_branch": text_bank_key,
            "model_args": vars(model_ns),
            "load_stats": load_stats,
            "alignment": align,
            "evidence_shapes": evidence_shapes,
            "scoring": {
                "formula": "contribution_score + attention_concentration + inter_prototype_diversity",
                "contribution_score": "max diagnostic identity-slot contribution; no learned gate is present",
                "attention_concentration": "1 - mean normalized entropy over visual/text evidence",
                "inter_prototype_diversity": "mean pairwise JS divergence over visual/text evidence maps",
            },
            "selected_figures": fig_meta,
            "summary_rows": rows,
            "hard_prohibitions_observed": {
                "cosine_nearest_whole_samples": False,
                "kmeans_recomputed": False,
                "dataset_reassigned_to_prototypes": False,
                "prototype_false_model": False,
                "forced_only_global": bool(model_ns.only_global) and model_ns.prototype_feature != "local",
            },
        }
        save_json(metadata, outdir / "figure_metadata.json")
        logger.log(f"Saved {outdir / 'figure_metadata.json'}")
        if args.save_raw_tensors:
            raw_path = outdir / "raw_evidence_tensors.pt"
            torch.save({"items": raw_items, "metadata": metadata}, raw_path)
            logger.log(f"Saved {raw_path}")
        logger.log("Done.")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
