
#!/usr/bin/env python3
"""Figure 1: Prototype Mode Discovery.

README usage
============
Visualize identity-owned prototype slots by showing nearest images and captions
for each selected identity and prototype slot.
This script is intentionally global/CLIP-only: it builds the model with
only_global=True, prototype=False, and never calls encode_image_grab /
encode_text_grab or prototype projectors.

Example:
python visualize_prototype_modes.py \
  --model_ckpt /path/to/model.pth \
  --prototype_ckpt /path/to/best_prototype_bank.pth \
  --dataset_root /path/to/data_or_dataset_root \
  --split test \
  --output_dir /path/to/output/prototype_vis \
  --num_ids 8 --topk_images 3 --topk_texts 3

The annotation paths are optional. If they are not provided, --dataset_root is
inspected using the repository dataset layouts:
  RSTPReid/data_captions.json
  CUHK-PEDES/reid_raw.json
  ICFG-PEDES/ICFG-PEDES.json
You may pass either the dataset directory itself or its parent data directory.
If that parent directory contains multiple known datasets, pass --dataset_name
or let the script infer it from checkpoint/output paths when possible.

If --split test is requested but the prototype bank is train-aligned, the script
prints a warning, switches true Prototype Mode Discovery to the aligned split,
and optionally exports simple test-ID sheets marked as NOT prototype figures.

Adapt PROTOTYPE_KEY_HINTS and PID_KEY_HINTS if your checkpoint key names differ.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import matplotlib
    matplotlib.use("Agg")
except ModuleNotFoundError:
    matplotlib = None
import numpy as np
import torch
import torch.nn.functional as F
try:
    from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps
    PIL_AVAILABLE = True
except ModuleNotFoundError:
    Image = ImageDraw = ImageFile = ImageFont = ImageOps = None
    PIL_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if ImageFile is not None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True

PID_KEYS = ["id", "pid", "person_id", "identity", "identity_id", "label"]
IMAGE_KEYS = ["img_path", "file_path", "image_path", "path", "filename", "image", "img"]
CAPTION_KEYS = ["captions", "caption", "text", "texts", "description", "descriptions"]
DEFAULT_ANNOTATIONS = ["data_captions.json", "reid_raw.json", "ICFG-PEDES.json", "annotations.json", "annotation.json"]
DATASET_LAYOUTS = [
    {"name": "RSTPReid", "dirname": "RSTPReid", "annotation": "data_captions.json", "image_dir": "imgs"},
    {"name": "CUHK-PEDES", "dirname": "CUHK-PEDES", "annotation": "reid_raw.json", "image_dir": "imgs"},
    {"name": "ICFG-PEDES", "dirname": "ICFG-PEDES", "annotation": "ICFG-PEDES.json", "image_dir": "imgs"},
]
SPLIT_ANNOTATION_CANDIDATES = {
    "train": ["train.json", "train_anno.json", "train_annotation.json", "train_captions.json"],
    "val": ["val.json", "valid.json", "validation.json", "val_anno.json", "val_annotation.json", "val_captions.json"],
    "test": ["test.json", "test_anno.json", "test_annotation.json", "test_captions.json"],
}
# Adapt these hints if your prototype checkpoint uses different key names.
PROTOTYPE_KEY_HINTS = {
    "visual": ["image_prototypes", "visual_prototypes", "img_prototypes", "text_to_image", "visual_bank", "image_bank", "prototypes"],
    "text": ["text_prototypes", "caption_prototypes", "txt_prototypes", "image_to_text", "text_bank", "caption_bank"],
}
PID_KEY_HINTS = ["proto_pids", "prototype_pids", "pids", "pid", "ids", "identity_ids"]
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
CLIP_STATE_KEY_PREFIXES = (
    "visual.",
    "transformer.",
    "token_embedding.",
    "ln_final.",
    "positional_embedding",
    "text_projection",
    "logit_scale",
)

@dataclass
class Sample:
    image_path: str
    pid: int
    captions: List[str]
    split: str
    raw_index: int

@dataclass
class AnnotationSource:
    split: Optional[str]
    path: Path
    dataset_dir: Path
    image_dir: Optional[Path]
    dataset_name: str

@dataclass
class TextItem:
    pid: int
    sample_index: int
    caption_index: int
    text: str

@dataclass
class PrototypeBank:
    visual: Optional[torch.Tensor]
    text: Optional[torch.Tensor]
    proto_pids: torch.Tensor
    config: Dict[str, Any]
    tensor_summary: Dict[str, List[int]]
    candidate_keys: Dict[str, List[str]]
    assignment_stats: Dict[int, float]
    source_path: str
    @property
    def primary(self) -> torch.Tensor:
        if self.visual is not None:
            return self.visual
        if self.text is not None:
            return self.text
        raise RuntimeError("No usable prototype tensor loaded.")
    @property
    def dim(self) -> int:
        return int(self.primary.shape[-1])

@dataclass
class EmbeddingBundle:
    image_embeddings: torch.Tensor
    image_sample_indices: List[int]
    text_embeddings: torch.Tensor
    text_items: List[TextItem]
    feature_kind: str
    used_projector: bool

class Logger:
    def __init__(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        self.path = output_dir / "logs.txt"
        self.file = self.path.open("w", encoding="utf-8")
    def log(self, msg: str) -> None:
        print(msg)
        self.file.write(msg + "\n")
        self.file.flush()
    def close(self) -> None:
        self.file.close()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw Figure 1: Prototype Mode Discovery.")
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--prototype_ckpt", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_name", default="auto", choices=["auto", "RSTPReid", "CUHK-PEDES", "ICFG-PEDES"])
    p.add_argument("--train_anno", default="")
    p.add_argument("--val_anno", default="")
    p.add_argument("--test_anno", default="")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_ids", type=int, default=8)
    p.add_argument("--ids", nargs="*", type=int, default=None)
    p.add_argument("--topk_images", type=int, default=3)
    p.add_argument("--topk_texts", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--save_contact_sheet", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--image_size", type=int, default=160)
    p.add_argument("--max_rows", type=int, default=4)
    p.add_argument("--pretrain_choice", default="ViT-B/16")
    p.add_argument("--model_img_size", nargs=2, type=int, default=[384, 128], metavar=("HEIGHT", "WIDTH"))
    p.add_argument("--stride_size", type=int, default=16)
    p.add_argument("--text_length", type=int, default=77)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--only_global", action="store_true", default=True, help="Force CLIP/global inference. This is always enabled for this figure.")
    p.add_argument("--prototype_feature", default="global", choices=["auto", "local", "global"], help="Kept for checkpoint metadata compatibility; embeddings are always global when --only_global is true.")
    p.add_argument("--prototype_projector", default="default")
    p.add_argument("--prototype_dim", type=int, default=None)
    p.add_argument("--prototype_per_id", type=int, default=None)
    p.add_argument("--select_ratio", type=float, default=0.4)
    p.add_argument("--max_test_sheet_images", type=int, default=12)
    p.add_argument("--force_split", action="store_true")
    return p.parse_args()

def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    out = Path(path).expanduser()
    if not out.is_absolute():
        out = base / out
    return out.resolve()

def torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")

def save_json(data: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.write("\n")

def safe_stem(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_") or "item"

def scalar_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None

def first_key(row: Mapping[str, Any], keys: Sequence[str]) -> Optional[str]:
    lower = {str(k).lower(): str(k) for k in row.keys()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    for key in keys:
        for actual in row.keys():
            if key.lower() in str(actual).lower():
                return str(actual)
    return None

def records_from_json(raw: Any, split: str) -> List[Mapping[str, Any]]:
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, Mapping)]
    if isinstance(raw, Mapping):
        for key in (split, "annotations", "annos", "data", "items", "samples"):
            if isinstance(raw.get(key), list):
                return [r for r in raw[key] if isinstance(r, Mapping)]
        rows = []
        for value in raw.values():
            if isinstance(value, list):
                rows.extend(r for r in value if isinstance(r, Mapping))
        if rows:
            return rows
    raise ValueError("Annotation JSON must be a list or a mapping containing lists.")

def captions_from_row(row: Mapping[str, Any]) -> List[str]:
    key = first_key(row, CAPTION_KEYS)
    if key is None:
        return []
    val = row[key]
    if isinstance(val, str):
        return [val]
    if isinstance(val, Sequence):
        return [str(v) for v in val if v is not None]
    return [str(val)]

def resolve_image(raw: str, dataset_root: Path, anno_dir: Path, image_dir: Optional[Path] = None) -> Path:
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p.resolve()
    candidates: List[Path] = []
    if image_dir is not None:
        candidates.extend([image_dir / p, image_dir / p.name])
    candidates.extend([dataset_root / p, dataset_root / "imgs" / p, dataset_root / "images" / p, anno_dir / p, anno_dir / "imgs" / p, anno_dir / "images" / p])
    for c in candidates:
        if c.exists():
            return c.resolve()
    return candidates[0].resolve()

def load_annotation(path: Path, dataset_root: Path, split: str, logger: Logger, image_dir: Optional[Path] = None) -> List[Sample]:
    raw = json.load(path.open("r", encoding="utf-8"))
    rows = records_from_json(raw, split)
    out, skipped = [], 0
    for row in rows:
        row_split = str(row.get("split", split))
        if row.get("split") is not None:
            if split == "val" and row_split in ("train", "test"):
                continue
            if split != "val" and row_split != split:
                continue
        pid_key = first_key(row, PID_KEYS)
        img_key = first_key(row, IMAGE_KEYS)
        if pid_key is None or img_key is None:
            skipped += 1
            continue
        pid = scalar_int(row.get(pid_key))
        if pid is None:
            skipped += 1
            continue
        out.append(Sample(str(resolve_image(str(row[img_key]), dataset_root, path.parent, image_dir)), pid, captions_from_row(row), split, len(out)))
    logger.log(f"[{split}] loaded {len(out)} samples from {path} (dataset_dir={dataset_root}, image_dir={image_dir or 'auto'}, skipped={skipped})")
    return out

def layout_candidates(root: Path, dataset_name: str) -> List[AnnotationSource]:
    wanted = None if dataset_name == "auto" else dataset_name
    candidates: List[AnnotationSource] = []
    for layout in DATASET_LAYOUTS:
        if wanted is not None and layout["name"] != wanted:
            continue
        dirs = [root]
        if root.name != layout["dirname"]:
            dirs.append(root / layout["dirname"])
        for dataset_dir in dirs:
            anno = dataset_dir / layout["annotation"]
            image_dir = dataset_dir / layout["image_dir"]
            if anno.is_file():
                candidates.append(AnnotationSource(None, anno.resolve(), dataset_dir.resolve(), image_dir.resolve() if image_dir.is_dir() else None, layout["name"]))
    return candidates

def generic_annotation_candidates(root: Path) -> List[AnnotationSource]:
    bases = [root]
    if root.is_dir():
        bases.extend(p for p in root.iterdir() if p.is_dir())
    out: List[AnnotationSource] = []
    for base in bases:
        for name in DEFAULT_ANNOTATIONS:
            p = base / name
            if p.is_file():
                img_dir = base / "imgs"
                out.append(AnnotationSource(None, p.resolve(), base.resolve(), img_dir.resolve() if img_dir.is_dir() else None, "custom"))
    return out

def split_file_candidates(root: Path) -> List[AnnotationSource]:
    bases = [root]
    for sub in ("annotations", "annos"):
        if (root / sub).is_dir():
            bases.append(root / sub)
    sources: List[AnnotationSource] = []
    for split, names in SPLIT_ANNOTATION_CANDIDATES.items():
        for base in bases:
            for name in names:
                p = base / name
                if p.is_file():
                    img_dir = root / "imgs"
                    sources.append(AnnotationSource(split, p.resolve(), root.resolve(), img_dir.resolve() if img_dir.is_dir() else None, "custom-split-files"))
                    break
            if any(s.split == split for s in sources):
                break
    return sources

def detect_annotation_sources(root: Path, dataset_name: str, logger: Logger) -> List[AnnotationSource]:
    if not root.exists():
        raise FileNotFoundError(root)
    sources = layout_candidates(root, dataset_name)
    if not sources and dataset_name == "auto":
        sources = generic_annotation_candidates(root)
    if not sources:
        sources = split_file_candidates(root)
    if not sources:
        known = ", ".join(f"{x['dirname']}/{x['annotation']}" for x in DATASET_LAYOUTS)
        raise FileNotFoundError(f"No annotation JSON found under {root}. Expected one of: {known}, or train/test/val JSON files.")

    single_file_sources = [s for s in sources if s.split is None]
    if single_file_sources:
        if len(single_file_sources) > 1:
            described = "; ".join(f"{s.dataset_name}: {s.path}" for s in single_file_sources)
            if dataset_name == "auto":
                raise RuntimeError(f"Multiple dataset annotations detected under {root}: {described}. Pass --dataset_name RSTPReid, CUHK-PEDES, or ICFG-PEDES.")
        src = single_file_sources[0]
        logger.log(f"Auto-detected dataset annotation: dataset={src.dataset_name}, annotation={src.path}, image_dir={src.image_dir or 'auto'}")
        return [src]

    logger.log("Auto-detected split annotation files: " + ", ".join(f"{s.split}={s.path}" for s in sources))
    return sources

def norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())

def infer_dataset_name_from_context(args: argparse.Namespace) -> Optional[str]:
    haystack = norm_name(" ".join([args.dataset_root, args.model_ckpt, args.prototype_ckpt, args.output_dir]))
    matches = []
    for layout in DATASET_LAYOUTS:
        if norm_name(layout["name"]) in haystack or norm_name(layout["dirname"]) in haystack:
            matches.append(layout["name"])
    return matches[0] if len(set(matches)) == 1 else None

def load_annotations(args: argparse.Namespace, logger: Logger) -> Dict[str, List[Sample]]:
    root = resolve_path(args.dataset_root)
    out = {"train": [], "val": [], "test": []}
    given = {"train": args.train_anno, "val": args.val_anno, "test": args.test_anno}
    used_any = False
    for split, raw_path in given.items():
        if raw_path:
            path = resolve_path(raw_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            out[split] = load_annotation(path, root, split, logger)
            used_any = True
    if not used_any:
        dataset_name = args.dataset_name
        if dataset_name == "auto":
            inferred = infer_dataset_name_from_context(args)
            if inferred:
                dataset_name = inferred
                logger.log(f"Inferred dataset_name={dataset_name} from CLI paths.")
        sources = detect_annotation_sources(root, dataset_name, logger)
        if len(sources) == 1 and sources[0].split is None:
            src = sources[0]
            for split in out:
                out[split] = load_annotation(src.path, src.dataset_dir, split, logger, src.image_dir)
        else:
            for src in sources:
                if src.split in out:
                    out[src.split] = load_annotation(src.path, src.dataset_dir, src.split, logger, src.image_dir)
    return out

def group_by_pid(samples: Sequence[Sample]) -> Dict[int, List[int]]:
    g: Dict[int, List[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        g[int(s.pid)].append(i)
    return dict(g)

def pid_set(samples: Sequence[Sample], shift: int = 0) -> set[int]:
    return {int(s.pid) + int(shift) for s in samples}

def candidate_pid_shifts(samples: Sequence[Sample], proto_pids: set[int]) -> List[int]:
    sample_pids = {int(s.pid) for s in samples}
    shifts = {0, -1, 1}
    if sample_pids and proto_pids:
        shifts.add(min(proto_pids) - min(sample_pids))
        shifts.add(max(proto_pids) - max(sample_pids))
    return sorted(shifts, key=lambda x: (abs(x), x))

def best_pid_shift(samples: Sequence[Sample], proto_pids: set[int]) -> Tuple[int, int]:
    best_shift, best_overlap = 0, 0
    for shift in candidate_pid_shifts(samples, proto_pids):
        overlap = len(pid_set(samples, shift) & proto_pids)
        if overlap > best_overlap:
            best_shift, best_overlap = shift, overlap
    return best_shift, best_overlap

def apply_pid_shift(samples: Sequence[Sample], shift: int) -> List[Sample]:
    if int(shift) == 0:
        return list(samples)
    return [Sample(s.image_path, int(s.pid) + int(shift), list(s.captions), s.split, s.raw_index) for s in samples]

def flatten_tensors(obj: Any, prefix: str = "") -> Dict[str, torch.Tensor]:
    out = {}
    if torch.is_tensor(obj):
        out[prefix.rstrip(".") or "tensor"] = obj
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(flatten_tensors(v, f"{prefix}{k}."))
    return out

def ckpt_config(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, Mapping):
        for key in ("prototype_config", "config", "args"):
            if isinstance(raw.get(key), Mapping):
                return dict(raw[key])
    return {}

def name_score(name: str, hints: Sequence[str]) -> int:
    lo, score = name.lower(), 0
    for i, h in enumerate(hints):
        if h.lower() in lo:
            score += 100 - i
    if any(b in lo for b in ("optimizer", "scheduler", "classifier", "projector")):
        score -= 80
    return score

def choose_proto(tensors: Mapping[str, torch.Tensor], hints: Sequence[str]) -> Tuple[Optional[str], Optional[torch.Tensor], List[str]]:
    cands, names = [], []
    for k, v in tensors.items():
        if torch.is_tensor(v) and v.ndim in (2, 3) and v.shape[-1] >= 16:
            s = name_score(k, hints)
            if s > 0:
                cands.append((s, k, v.detach().float().cpu()))
                names.append(f"{k}{tuple(v.shape)}")
    if not cands:
        return None, None, names
    cands.sort(key=lambda x: (x[0], x[2].numel()), reverse=True)
    _, key, t = cands[0]
    if t.ndim == 3:
        t = t.reshape(-1, t.shape[-1])
    return key, t.contiguous(), names

def infer_k(proto_pids: torch.Tensor) -> int:
    counts = Counter(int(x) for x in proto_pids.tolist())
    return max(1, int(round(float(np.median(list(counts.values())))))) if counts else 1

def choose_pids(tensors: Mapping[str, torch.Tensor], n: int, cfg: Mapping[str, Any]) -> Tuple[torch.Tensor, List[str]]:
    names = []
    for k, v in tensors.items():
        if torch.is_tensor(v) and v.ndim == 1 and v.numel() == n and any(h in k.lower() for h in PID_KEY_HINTS):
            names.append(f"{k}{tuple(v.shape)}")
            return v.detach().cpu().long(), names
    num_classes = scalar_int(cfg.get("num_classes"))
    k = scalar_int(cfg.get("prototypes_per_id")) or scalar_int(cfg.get("prototype_per_id"))
    if num_classes and k and num_classes * k == n:
        return torch.arange(num_classes).repeat_interleave(k).long(), names
    return torch.arange(n).long(), names

def assignment_stats(tensors: Mapping[str, torch.Tensor]) -> Dict[int, float]:
    stats = defaultdict(float)
    for k, v in tensors.items():
        if not torch.is_tensor(v) or not any(h in k.lower() for h in ("assign", "usage", "count")):
            continue
        flat = v.detach().cpu().float().flatten()
        if flat.numel() > 200000:
            continue
        for i, val in enumerate(flat.tolist()):
            if val > 0:
                stats[int(i)] += float(val)
    return dict(stats)

def inspect_checkpoint(path: Path, logger: Logger) -> Dict[str, Any]:
    raw = torch_load(path)
    tensors = flatten_tensors(raw)
    summary = {k: list(v.shape) for k, v in tensors.items()}
    logger.log(f"Prototype checkpoint: {path}")
    logger.log(f"Found {len(tensors)} tensor entries.")
    for k, s in sorted(summary.items()):
        if any(h in k.lower() for h in ("proto", "memory", "bank", "pid")):
            logger.log(f"  {k}: {s}")
    return {"raw": raw, "tensors": tensors, "summary": summary}

def load_prototypes(path: Path, logger: Logger) -> PrototypeBank:
    data = inspect_checkpoint(path, logger)
    tensors, raw = data["tensors"], data["raw"]
    cfg = ckpt_config(raw)
    vk, visual, vc = choose_proto(tensors, PROTOTYPE_KEY_HINTS["visual"])
    tk, text, tc = choose_proto(tensors, PROTOTYPE_KEY_HINTS["text"])
    if visual is None and text is None:
        all_keys = "\n".join(f"  {k}: {v}" for k, v in sorted(data["summary"].items()))
        raise RuntimeError("No prototype tensor found. Adapt PROTOTYPE_KEY_HINTS.\n" + all_keys)
    n = int((visual if visual is not None else text).shape[0])
    proto_pids, pc = choose_pids(tensors, n, cfg)
    logger.log(f"Selected visual prototype tensor: {vk or 'none'}")
    logger.log(f"Selected text prototype tensor: {tk or 'none'}")
    logger.log(f"Candidate visual prototype keys: {vc or 'none'}")
    logger.log(f"Candidate text prototype keys: {tc or 'none'}")
    logger.log(f"PID mapping keys: {pc or 'none; inferred from config/order'}")
    modality = "multimodal" if visual is not None and text is not None else ("visual" if visual is not None else "text")
    logger.log(f"Prototype modality inferred as: {modality}")
    logger.log(f"Prototype count={n}, dim={(visual if visual is not None else text).shape[-1]}, unique_pids={len(set(proto_pids.tolist()))}")
    logger.log(f"Prototype config: {json.dumps(cfg, sort_keys=True, default=str) if cfg else 'not found'}")
    return PrototypeBank(
        F.normalize(visual.float(), p=2, dim=1) if visual is not None else None,
        F.normalize(text.float(), p=2, dim=1) if text is not None else None,
        proto_pids.long(), dict(cfg), data["summary"], {"visual": vc, "text": tc, "pid": pc}, assignment_stats(tensors), str(path)
    )

def model_args(args: argparse.Namespace, bank: PrototypeBank, train_samples: Sequence[Sample]) -> SimpleNamespace:
    cfg = bank.config
    proto_dim = args.prototype_dim or scalar_int(cfg.get("prototype_dim")) or bank.dim
    proto_k = args.prototype_per_id or scalar_int(cfg.get("prototypes_per_id")) or scalar_int(cfg.get("prototype_per_id")) or infer_k(bank.proto_pids)
    return SimpleNamespace(
        loss_names="tal+cid", pretrain_choice=args.pretrain_choice, img_size=tuple(args.model_img_size), stride_size=args.stride_size,
        temperature=0.02, prototype=False, use_loss_id=False, no_pbt=False,
        prototype_feature="global", prototype_dim=int(proto_dim), prototype_per_id=int(proto_k),
        prototype_projector=str(cfg.get("projector_mode", args.prototype_projector)), prototype_residual_scale=float(cfg.get("prototype_residual_scale", 0.1)),
        prototype_kmeans_iters=20, prototype_warmup_epochs=0, prototype_tau=0.05, prototype_hard_k=16,
        prototype_id_weight=0.2, prototype_momentum=float(cfg.get("momentum", 0.2)), only_global=True,
        select_ratio=args.select_ratio, return_all=False, topk_type="mean", layer_index=-1, average_attn_weights=True, modify_k=False,
        track_train_diagnostics=False, training=False, txt_aug=False, img_aug=False, num_workers=args.num_workers,
        batch_size=args.batch_size, test_batch_size=args.batch_size, dataset_name=args.dataset_name if args.dataset_name != "auto" else "custom", root_dir=str(resolve_path(args.dataset_root))
    )

def checkpoint_state_dict(ckpt: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(ckpt, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "net", "network", "module"):
            if isinstance(ckpt.get(key), Mapping):
                return ckpt[key]
        return ckpt
    raise ValueError("Model checkpoint must be a mapping.")

def strip_prefixes(key: str, prefixes: Sequence[str]) -> str:
    changed, out = True, key
    while changed:
        changed = False
        for p in prefixes:
            if out.startswith(p):
                out = out[len(p):]
                changed = True
    return out

def candidate_keys(key: str) -> List[str]:
    raw, out = str(key), []
    def add(x: str):
        if x and x not in out:
            out.append(x)
    add(raw)
    no_mod = strip_prefixes(raw, ("module.",)); add(no_mod)
    no_wrap = strip_prefixes(no_mod, ("model.", "net.", "network.")); add(no_wrap)
    norm = strip_prefixes(raw, ("module.", "model.", "net.", "network.")); add(norm)
    for c in (no_wrap, norm):
        add(c[len("base_model."):] if c.startswith("base_model.") else f"base_model.{c}")
    if no_mod.startswith("model."):
        add(no_mod[len("model."):])
    return out

def is_clip_global_key(raw_key: str) -> bool:
    for candidate in candidate_keys(raw_key):
        key = strip_prefixes(candidate, ("base_model.",))
        if key == "logit_scale" or key.startswith(CLIP_STATE_KEY_PREFIXES):
            return True
    return False

def load_model_checkpoint(model: torch.nn.Module, path: Path, logger: Logger) -> Dict[str, int]:
    state = checkpoint_state_dict(torch_load(path))
    model_state = model.state_dict()
    update: MutableMapping[str, torch.Tensor] = {}
    miss = shape = non_tensor = non_global = 0
    for raw_key, value in state.items():
        if not torch.is_tensor(value):
            non_tensor += 1; continue
        if not is_clip_global_key(str(raw_key)):
            non_global += 1; continue
        target = next((c for c in candidate_keys(str(raw_key)) if c in model_state), None)
        if target is None:
            miss += 1; continue
        if model_state[target].shape != value.shape:
            shape += 1; continue
        update[target] = value.detach().clone()
    if not update:
        raise RuntimeError(f"No compatible model tensors found in {path}")
    model_state.update(update)
    model.load_state_dict(model_state, strict=True)
    stats = {"loaded": len(update), "skipped_missing": miss, "skipped_shape": shape, "skipped_non_tensor": non_tensor, "skipped_non_global": non_global}
    logger.log(
        "Loaded checkpoint tensors: "
        f"{stats['loaded']} (missing={stats['skipped_missing']}, "
        f"shape={stats['skipped_shape']}, non_tensor={stats['skipped_non_tensor']}, "
        f"ignored_non_global={stats['skipped_non_global']})"
    )
    return stats

def build_model(args: argparse.Namespace, bank: PrototypeBank, train_samples: Sequence[Sample], device: torch.device, logger: Logger) -> torch.nn.Module:
    from model import build_model as repo_build_model
    nclasses = max(len({s.pid for s in train_samples}), len(set(bank.proto_pids.tolist())), 1)
    ns = model_args(args, bank, train_samples)
    logger.log(f"Building global/CLIP model num_classes={nclasses}, only_global={ns.only_global}, prototype_enabled={ns.prototype}")
    model = repo_build_model(ns, num_classes=nclasses)
    load_model_checkpoint(model, resolve_path(args.model_ckpt), logger)
    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model

def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(arg)
    return torch.device("cpu") if dev.type == "cuda" and not torch.cuda.is_available() else dev

class ImageSampleDataset(Dataset):
    def __init__(self, samples: Sequence[Sample], indices: Sequence[int], img_size: Tuple[int, int]):
        self.samples, self.indices, self.img_size = samples, list(indices), img_size
    def __len__(self) -> int:
        return len(self.indices)
    def __getitem__(self, i: int) -> Tuple[int, int, torch.Tensor]:
        idx = self.indices[i]
        s = self.samples[idx]
        return idx, int(s.pid), preprocess_image(s.image_path, self.img_size)

class TextItemDataset(Dataset):
    def __init__(self, items: Sequence[TextItem], text_length: int):
        self.items, self.text_length, self.tokenizer = list(items), text_length, None
    def __len__(self) -> int:
        return len(self.items)
    def __getitem__(self, i: int) -> Tuple[int, int, torch.Tensor]:
        if self.tokenizer is None:
            self.tokenizer = make_tokenizer()
        item = self.items[i]
        return i, int(item.pid), tokenize_text(item.text, self.tokenizer, self.text_length)

def preprocess_image(path: str, img_size: Tuple[int, int]) -> torch.Tensor:
    h, w = int(img_size[0]), int(img_size[1])
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img).resize((w, h), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - CLIP_MEAN) / CLIP_STD
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))

def make_tokenizer() -> Any:
    try:
        from utils.simple_tokenizer import SimpleTokenizer
        return SimpleTokenizer()
    except ModuleNotFoundError:
        return None

def tokenize_text(text: str, tokenizer: Any, length: int) -> torch.Tensor:
    if tokenizer is not None:
        try:
            from datasets.bases import tokenize as repo_tokenize
            return repo_tokenize(text, tokenizer=tokenizer, text_length=length, truncate=True)
        except ModuleNotFoundError:
            pass
    from model.clip_model import tokenize as clip_tokenize
    return clip_tokenize([text], context_length=length, truncate=True)[0]

def text_items_for(samples: Sequence[Sample], indices: Sequence[int]) -> List[TextItem]:
    out = []
    for idx in indices:
        s = samples[idx]
        for ci, cap in enumerate(s.captions):
            out.append(TextItem(s.pid, idx, ci, cap))
    return out

def feature_kind(model: torch.nn.Module, bank: PrototypeBank) -> str:
    return "global"

def cache_name(args: argparse.Namespace, split: str, bank: PrototypeBank, samples: Sequence[Sample], kind: str) -> str:
    pid_digest = hashlib.sha1(",".join(str(int(s.pid)) for s in samples).encode()).hexdigest()[:12]
    payload = {"model": str(resolve_path(args.model_ckpt)), "mtime": os.path.getmtime(resolve_path(args.model_ckpt)), "proto": str(resolve_path(args.prototype_ckpt)), "split": split, "kind": kind, "dim": bank.dim, "n": len(samples), "pid_digest": pid_digest, "img": list(args.model_img_size), "text": args.text_length}
    return f"{split}_{kind}_{hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]}.pt"

@torch.inference_mode()
def encode_images(model: torch.nn.Module, loader: DataLoader, device: torch.device, kind: str) -> Tuple[torch.Tensor, List[int]]:
    feats, idxs = [], []
    for sample_idx, _pid, images in tqdm(loader, desc="Encoding images"):
        images = images.to(device, non_blocking=True)
        out = model.encode_image(images).float()
        feats.append(out.cpu()); idxs.extend(int(x) for x in sample_idx.tolist())
    return torch.cat(feats, dim=0), idxs

@torch.inference_mode()
def encode_texts(model: torch.nn.Module, loader: DataLoader, device: torch.device, kind: str) -> Tuple[torch.Tensor, List[int]]:
    feats, idxs = [], []
    for text_idx, _pid, tokens in tqdm(loader, desc="Encoding captions"):
        tokens = tokens.to(device, non_blocking=True)
        out = model.encode_text(tokens).float()
        feats.append(out.cpu()); idxs.extend(int(x) for x in text_idx.tolist())
    return (torch.cat(feats, dim=0), idxs) if feats else (torch.empty(0, 1), [])

@torch.inference_mode()
def project_if_needed(model: torch.nn.Module, img: torch.Tensor, txt: torch.Tensor, kind: str, batch: int) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    return img, txt, False

def build_embeddings(model: torch.nn.Module, samples: Sequence[Sample], split: str, bank: PrototypeBank, args: argparse.Namespace, outdir: Path, device: torch.device, logger: Logger) -> EmbeddingBundle:
    kind = feature_kind(model, bank)
    cache_dir = outdir / "embedding_cache"; cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / cache_name(args, split, bank, samples, kind)
    if cache.is_file():
        logger.log(f"Loading embedding cache: {cache}")
        p = torch_load(cache)
        return EmbeddingBundle(p["image_embeddings"], list(p["image_sample_indices"]), p["text_embeddings"], [TextItem(**x) for x in p["text_items"]], p.get("feature_kind", kind), bool(p.get("used_projector", False)))
    indices = list(range(len(samples)))
    text_items = text_items_for(samples, indices)
    iloader = DataLoader(ImageSampleDataset(samples, indices, tuple(args.model_img_size)), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    img_feat, img_indices = encode_images(model, iloader, device, kind)
    if text_items:
        tloader = DataLoader(TextItemDataset(text_items, args.text_length), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == "cuda")
        txt_feat, order = encode_texts(model, tloader, device, kind)
        text_items = [text_items[i] for i in order]
    else:
        txt_feat = torch.empty(0, img_feat.shape[1])
    img_feat, txt_feat, used_proj = project_if_needed(model, img_feat, txt_feat, kind, args.batch_size)
    img_feat = F.normalize(img_feat.float(), p=2, dim=1).cpu()
    txt_feat = F.normalize(txt_feat.float(), p=2, dim=1).cpu() if txt_feat.numel() else txt_feat.cpu()
    torch.save({"image_embeddings": img_feat, "image_sample_indices": img_indices, "text_embeddings": txt_feat, "text_items": [x.__dict__ for x in text_items], "feature_kind": kind, "used_projector": used_proj}, cache)
    logger.log(f"Saved embedding cache: {cache}")
    return EmbeddingBundle(img_feat, img_indices, txt_feat, text_items, kind, used_proj)

def split_alignment(samples_by_split: Mapping[str, Sequence[Sample]], bank: PrototypeBank, requested: str, force: bool, logger: Logger) -> Dict[str, Any]:
    proto_pids = set(int(x) for x in bank.proto_pids.tolist())
    report = {"requested_split": requested, "prototype_pid_count": len(proto_pids), "split_pid_counts": {}, "overlap_counts": {}, "pid_shifts": {}, "used_split": requested, "used_pid_shift": 0, "switched": False, "warning": ""}
    for split, samples in samples_by_split.items():
        shift, overlap = best_pid_shift(samples, proto_pids)
        report["split_pid_counts"][split] = len({s.pid for s in samples})
        report["overlap_counts"][split] = int(overlap)
        report["pid_shifts"][split] = int(shift)
        logger.log(f"Prototype/split alignment candidate: split={split} raw_pids={report['split_pid_counts'][split]} best_overlap={overlap} pid_shift={shift}")
    if force:
        report["used_pid_shift"] = int(report["pid_shifts"].get(requested, 0))
        logger.log(f"Prototype/split alignment: --force_split uses requested split={requested}, pid_shift={report['used_pid_shift']}, overlap={report['overlap_counts'].get(requested, 0)}")
        return report

    requested_overlap = int(report["overlap_counts"].get(requested, 0))
    train_overlap = int(report["overlap_counts"].get("train", 0))
    if requested == "test" and train_overlap > 0 and train_overlap >= requested_overlap:
        report.update({
            "used_split": "train",
            "used_pid_shift": int(report["pid_shifts"].get("train", 0)),
            "switched": requested != "train",
            "warning": (
                "Prototype bank appears train-aligned. True Prototype Mode Discovery will use split 'train' "
                f"(train_overlap={train_overlap}, requested_test_overlap={requested_overlap}). Use --force_split only if this checkpoint was built for test IDs."
            ),
        })
        logger.log("WARNING: " + report["warning"])
        return report

    if requested_overlap > 0:
        report["used_pid_shift"] = int(report["pid_shifts"].get(requested, 0))
        logger.log(f"Prototype/split alignment: using requested split={requested}, pid_shift={report['used_pid_shift']}, overlap={requested_overlap}")
        return report

    best = max(("train", "val", "test"), key=lambda s: int(report["overlap_counts"].get(s, 0)))
    if report["overlap_counts"].get(best, 0) > 0:
        report.update({"used_split": best, "used_pid_shift": int(report["pid_shifts"].get(best, 0)), "switched": True, "warning": f"Requested split {requested!r} has no prototype PID overlap. Using aligned split {best!r} for true Prototype Mode Discovery."})
        logger.log("WARNING: " + report["warning"])
    else:
        report["warning"] = "No split has PID overlap with the prototype bank."
        logger.log("WARNING: " + report["warning"])
    return report

def choose_ids(samples: Sequence[Sample], bank: PrototypeBank, args: argparse.Namespace, logger: Logger) -> List[int]:
    grouped = group_by_pid(samples)
    proto_pids = set(int(x) for x in bank.proto_pids.tolist())
    if args.ids:
        selected = [pid for pid in args.ids if pid in grouped and pid in proto_pids]
        missing = [pid for pid in args.ids if pid not in selected]
        if missing: logger.log(f"WARNING: requested ids unavailable for prototype visualization: {missing}")
        return selected[:args.num_ids]
    min_s = max(args.topk_images, 3)
    cands = [pid for pid, idxs in grouped.items() if len(idxs) >= min_s and pid in proto_pids] or [pid for pid in grouped if pid in proto_pids]
    rng = random.Random(args.seed)
    def score(pid: int) -> Tuple[int, int, float]:
        return (int((bank.proto_pids == pid).sum().item()), len(grouped[pid]), float(bank.assignment_stats.get(pid, 0.0)))
    cands.sort(key=score, reverse=True)
    pool = cands[:max(args.num_ids * 4, args.num_ids)]
    rng.shuffle(pool)
    out = sorted(pool[:args.num_ids], key=score, reverse=True)
    logger.log(f"Selected identity IDs: {out}")
    return out

def slots_for_pid(bank: PrototypeBank, pid: int) -> List[Tuple[int, int]]:
    idxs = (bank.proto_pids == int(pid)).nonzero(as_tuple=False).flatten().tolist()
    return [(slot, int(idx)) for slot, idx in enumerate(idxs)]

def topk_proto(proto: torch.Tensor, emb: torch.Tensor, positions: Sequence[int], k: int) -> List[Tuple[int, float]]:
    if k <= 0 or not positions or emb.numel() == 0: return []
    pos = torch.tensor(list(positions), dtype=torch.long)
    sims = emb[pos] @ F.normalize(proto.float(), p=2, dim=0)
    vals, loc = torch.topk(sims, k=min(k, sims.numel()), largest=True)
    return [(int(pos[int(i)].item()), float(v.item())) for v, i in zip(vals, loc)]

def retrieve_topk_for_prototype(proto: torch.Tensor, emb: torch.Tensor, positions: Sequence[int], k: int) -> List[Tuple[int, float]]:
    return topk_proto(proto, emb, positions, k)

STOPWORDS = {"the", "and", "with", "wearing", "person", "man", "woman", "shirt", "pants", "body", "image", "photo", "this", "that", "has", "have", "standing", "walking"}
def auto_tag(captions: Sequence[str]) -> str:
    words = []
    for cap in captions:
        words += [w for w in re.findall(r"[A-Za-z][A-Za-z-]+", cap.lower()) if len(w) > 2 and w not in STOPWORDS]
    if not words: return "visual mode"
    return " / ".join([w for w, _ in Counter(words).most_common(2)])

def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    paths = []
    if os.name == "nt":
        paths += [Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf")]
    paths += [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")]
    for p in paths:
        if p.exists(): return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()

def wrap(text: str, width: int, max_lines: int) -> List[str]:
    lines = textwrap.wrap(str(text), width=width, break_long_words=False, replace_whitespace=True) or [""]
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]; lines[-1] = lines[-1].rstrip(".") + "..."
    return lines

def thumb(path: str, size: int) -> Image.Image:
    canvas = Image.new("RGB", (size, size * 3 // 2), "#eef1f5")
    try:
        img = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        img = ImageOps.contain(img, canvas.size)
        canvas.paste(img, ((canvas.width - img.width) // 2, (canvas.height - img.height) // 2))
    except Exception:
        d = ImageDraw.Draw(canvas); d.text((10, canvas.height // 2), "missing image", fill="#9a1c1c", font=font(13, True))
    return canvas

def render_identity_figure(pid: int, samples: Sequence[Sample], bundle: EmbeddingBundle, bank: PrototypeBank, out: Path, topk_images: int, topk_texts: int, image_size: int) -> List[Dict[str, Any]]:
    grouped = group_by_pid(samples)
    sample_idxs = grouped.get(pid, [])
    pos_by_sample = {si: pos for pos, si in enumerate(bundle.image_sample_indices)}
    img_positions = [pos_by_sample[i] for i in sample_idxs if i in pos_by_sample]
    txt_positions = [i for i, item in enumerate(bundle.text_items) if item.pid == pid]
    slots = slots_for_pid(bank, pid)
    if not slots: return []
    title_f, sub_f, slot_f, small_f, cap_f = font(24, True), font(16), font(18, True), font(13), font(12)
    block_w = max(image_size * max(topk_images, 1) + 34, 360)
    header_h, caption_h, margin, gap = 88, max(92, topk_texts * 36 + 24), 24, 18
    block_h = 44 + image_size * 3 // 2 + caption_h
    page = Image.new("RGB", (margin * 2 + len(slots) * block_w + max(0, len(slots)-1) * gap, margin * 2 + header_h + block_h), "white")
    d = ImageDraw.Draw(page)
    rep = next((samples[i].captions[0] for i in sample_idxs if samples[i].captions), "")
    d.text((margin, margin), f"Identity {pid}: Prototype Mode Discovery", fill="#111827", font=title_f)
    for li, line in enumerate(wrap(rep, 120, 2)): d.text((margin, margin + 34 + li * 20), line, fill="#4b5563", font=sub_f)
    rows = []
    y0 = margin + header_h
    for b, (slot, proto_idx) in enumerate(slots):
        x0 = margin + b * (block_w + gap)
        d.rounded_rectangle([x0, y0, x0 + block_w, y0 + block_h], radius=6, outline="#d8dee6", fill="#ffffff")
        vproto = bank.visual[proto_idx] if bank.visual is not None else bank.primary[proto_idx]
        tproto = bank.text[proto_idx] if bank.text is not None else vproto
        top_img = topk_proto(vproto, bundle.image_embeddings, img_positions, topk_images)
        top_txt = topk_proto(tproto, bundle.text_embeddings, txt_positions, topk_texts)
        caps = [bundle.text_items[p].text for p, _ in top_txt]
        tag = auto_tag(caps)
        d.text((x0 + 14, y0 + 12), f"Prototype {slot}: {tag}", fill="#111827", font=slot_f)
        ix, iy = x0 + 14, y0 + 44
        paths = []
        for rank, (pos, score) in enumerate(top_img, start=1):
            s = samples[bundle.image_sample_indices[pos]]; paths.append(s.image_path)
            im = thumb(s.image_path, image_size); page.paste(im, (ix, iy))
            d.text((ix + 4, iy + 4), f"#{rank} {score:.2f}", fill="#111827", font=small_f)
            ix += image_size + 8
        ty = iy + image_size * 3 // 2 + 12
        if caps:
            for ci, cap in enumerate(caps[:topk_texts], start=1):
                for line in wrap(f"T{ci}: {cap}", max(32, block_w // 8), 2):
                    d.text((x0 + 14, ty), line, fill="#374151", font=cap_f); ty += 15
                ty += 4
        else:
            d.text((x0 + 14, ty), "No captions available.", fill="#6b7280", font=cap_f)
        rows.append({"pid": pid, "prototype_slot": slot, "prototype_index": proto_idx, "selected_image_paths": ";".join(paths), "selected_caption_indices": ";".join(f"{bundle.text_items[p].sample_index}:{bundle.text_items[p].caption_index}" for p, _ in top_txt), "selected_text_snippets": " || ".join(caps), "auto_mode_tag": tag})
    out.parent.mkdir(parents=True, exist_ok=True); page.save(out)
    return rows

def overview(paths: Sequence[Path], out: Path, max_rows: int) -> None:
    imgs = [Image.open(p).convert("RGB") for p in paths[:max_rows] if p.is_file()]
    if not imgs: return
    w, h = max(i.width for i in imgs), sum(i.height for i in imgs)
    canvas = Image.new("RGB", (w, h), "white"); y = 0
    for im in imgs:
        canvas.paste(im, ((w - im.width)//2, y)); y += im.height
    canvas.save(out)

def render_overview_figure(paths: Sequence[Path], out: Path, max_rows: int) -> None:
    overview(paths, out, max_rows)

def render_test_sheet(pid: int, samples: Sequence[Sample], out: Path, image_size: int, max_images: int) -> None:
    idxs = group_by_pid(samples).get(pid, [])[:max_images]
    if not idxs: return
    cols, margin, header = min(4, len(idxs)), 24, 62
    cell_w, cell_h = image_size + 18, image_size * 3 // 2 + 78
    page = Image.new("RGB", (margin*2 + cols*cell_w, margin*2 + header + math.ceil(len(idxs)/cols)*cell_h), "white")
    d = ImageDraw.Draw(page); d.text((margin, margin), f"Test identity {pid}: sample sheet (not prototype-owned)", fill="#111827", font=font(22, True))
    for r, si in enumerate(idxs):
        s = samples[si]; x = margin + (r % cols) * cell_w; y = margin + header + (r // cols) * cell_h
        im = thumb(s.image_path, image_size); page.paste(im, (x, y)); ty = y + im.height + 8
        cap = s.captions[0] if s.captions else Path(s.image_path).name
        for line in wrap(cap, 22, 3): d.text((x, ty), line, fill="#374151", font=font(12)); ty += 14
    out.parent.mkdir(parents=True, exist_ok=True); page.save(out)

def save_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    cols = ["pid", "prototype_slot", "prototype_index", "selected_image_paths", "selected_caption_indices", "selected_text_snippets", "auto_mode_tag"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    outdir = resolve_path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(outdir)
    try:
        logger.log("Prototype Mode Discovery visualization")
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for image loading/rendering. Install it in this environment with `pip install pillow`.")
        if not args.only_global:
            raise RuntimeError("This Figure 1 tool is intentionally global/CLIP-only. Do not use --no-only_global for prototype mode discovery.")
        save_json(vars(args), outdir / "used_config.json")
        samples_by_split = load_annotations(args, logger)
        bank = load_prototypes(resolve_path(args.prototype_ckpt), logger)
        align = split_alignment(samples_by_split, bank, args.split, args.force_split, logger)
        save_json(align, outdir / "split_alignment_report.json")
        used_split = str(align["used_split"])
        if not samples_by_split.get(used_split): raise RuntimeError(f"No samples for split {used_split}")
        device = device_from_arg(args.device); logger.log(f"Using device: {device}")
        model = build_model(args, bank, samples_by_split.get("train", []), device, logger)
        pid_shift = int(align.get("used_pid_shift", 0))
        samples = apply_pid_shift(samples_by_split[used_split], pid_shift)
        if pid_shift:
            logger.log(f"Applied pid_shift={pid_shift} to split={used_split} so annotation IDs align with prototype IDs.")
        bundle = build_embeddings(model, samples, used_split, bank, args, outdir, device, logger)
        logger.log(f"Embedding feature_kind={bundle.feature_kind}, used_projector={bundle.used_projector}, dim={bundle.image_embeddings.shape[1]}")
        if bundle.image_embeddings.shape[1] != bank.dim:
            raise RuntimeError(
                f"Global/CLIP embedding dim {bundle.image_embeddings.shape[1]} does not match prototype dim {bank.dim}. "
                "This script does not use ITSELF/local/grab inference. Use a global prototype bank/checkpoint, "
                "or rebuild/export the prototype bank in CLIP/global feature space."
            )
        selection_args = SimpleNamespace(**vars(args))
        if args.ids and pid_shift:
            selection_args.ids = [int(pid) + pid_shift for pid in args.ids]
            logger.log(f"Interpreting requested --ids in prototype PID space after shift: {selection_args.ids}")
        ids = choose_ids(samples, bank, selection_args, logger)
        if not ids and args.ids:
            logger.log("WARNING: requested --ids do not map to the prototype-aligned split; falling back to automatic ID selection for the true prototype figure.")
            fallback_args = SimpleNamespace(**vars(selection_args)); fallback_args.ids = None
            ids = choose_ids(samples, bank, fallback_args, logger)
        if not ids: raise RuntimeError("No identity IDs could be selected for true Prototype Mode Discovery.")
        summary, id_paths = [], []
        for pid in ids:
            path = outdir / f"prototype_modes_pid_{int(pid):04d}.png"
            rows = render_identity_figure(pid, samples, bundle, bank, path, args.topk_images, args.topk_texts, args.image_size)
            if rows:
                summary.extend(rows); id_paths.append(path); logger.log(f"Saved {path}")
        if args.save_contact_sheet:
            ov = outdir / "prototype_modes_overview.png"; overview(id_paths, ov, args.max_rows); logger.log(f"Saved {ov}")
        save_csv(summary, outdir / "nearest_neighbors_summary.csv"); logger.log(f"Saved {outdir / 'nearest_neighbors_summary.csv'}")
        if args.split == "test" and align.get("switched") and samples_by_split.get("test"):
            test_samples = samples_by_split["test"]; test_ids = args.ids if args.ids else list(group_by_pid(test_samples).keys())[:args.num_ids]
            for pid in test_ids[:args.num_ids]:
                p = outdir / f"test_id_sheet_pid_{int(pid):04d}.png"; render_test_sheet(int(pid), test_samples, p, args.image_size, args.max_test_sheet_images)
                if p.is_file(): logger.log(f"Saved {p} (NOT a prototype figure)")
        save_json({"requested_split": args.split, "used_split": used_split, "selected_ids": ids, "prototype_dim": bank.dim, "feature_kind": bundle.feature_kind, "used_projector": bundle.used_projector, "prototype_source": bank.source_path, "alignment": align}, outdir / "figure_metadata.json")
        logger.log("Done.")
    finally:
        logger.close()

if __name__ == "__main__":
    main()
