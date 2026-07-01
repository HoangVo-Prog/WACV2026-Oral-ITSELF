#!/usr/bin/env python3
"""Compare identity-level image/text embeddings for two TBPS checkpoints.

The script follows the loading and inference conventions used by
scripts/plot_ambiguity_rate_compare.py, then selects confusion-guided identity
groups from the baseline prototype space and visualizes Baseline vs IAPR in 2D.

Example:

python scripts/plot_identity_embedding_compare.py \
--dataset_name RSTPReid \
--baseline_checkpoint /path/to/baseline/best.pth \
--iapr_checkpoint /path/to/iapr/best.pth \
--output_dir outputs/identity_embedding_compare/rstpreid \
--reducer tsne \
--plot_prototypes
"""

from __future__ import annotations

import argparse
import csv
import gc
import inspect
import json
import math
import random
import sys
from dataclasses import dataclass
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

_DATASET_FACTORIES: Optional[Dict[str, type]] = None


GENERIC_DATASET_CONFIGS = {
    "CUHK-PEDES": {
        "dataset_dir": "CUHK-PEDES",
        "annotation_files": ["reid_raw.json"],
        "image_dirs": ["imgs"],
        "path_keys": ["file_path"],
    },
    "ICFG-PEDES": {
        "dataset_dir": "ICFG-PEDES",
        "annotation_files": ["ICFG-PEDES.json"],
        "image_dirs": ["imgs"],
        "path_keys": ["file_path"],
    },
    "RSTPReid": {
        "dataset_dir": "RSTPReid",
        "annotation_files": ["data_captions.json"],
        "image_dirs": ["imgs"],
        "path_keys": ["img_path"],
    },
}

DEFAULT_ANNOTATION_FILES = [
    "data_captions.json",
    "reid_raw.json",
    "ICFG-PEDES.json",
    "annotations.json",
    "annotation.json",
]
DEFAULT_IMAGE_DIRS = ["imgs", "images", "image", ""]
DEFAULT_PATH_KEYS = ["img_path", "file_path", "image_path", "path", "filename", "image"]
DEFAULT_PID_KEYS = ["id", "pid", "person_id", "identity", "identity_id", "label"]
DEFAULT_CAPTION_KEYS = ["captions", "caption", "text", "description"]
CONFIG_FILE_NAMES = ("configs.yaml", "config.yaml", "config.yml", "args.yaml", "args.yml")
CHECKPOINT_CONFIG_KEYS = ("args", "config", "cfg", "model_args", "train_args", "training_args")

GROUP_SCORE_COLUMNS = [
    "group_rank",
    "score",
    "identity_ids",
    "num_identities",
    "num_samples",
    "C_baseline",
    "C_iapr",
    "delta_compactness",
    "S_baseline",
    "S_iapr",
    "delta_separation",
    "M_baseline",
    "M_iapr",
    "delta_margin",
    "A_baseline",
    "A_iapr",
    "delta_ambiguity",
    "imbalance_penalty",
]


@dataclass
class SplitData:
    image_pids: List[int]
    img_paths: List[str]
    caption_pids: List[int]
    captions: List[str]
    num_train_ids: int


@dataclass
class EmbeddingBundle:
    features: np.ndarray
    pids: np.ndarray
    modalities: np.ndarray
    source_indices: np.ndarray
    checkpoint: str
    load_stats: Dict[str, Any]
    inference: Dict[str, Any]


@dataclass
class PrototypeTable:
    identity_ids: np.ndarray
    prototypes: np.ndarray
    counts: Dict[int, int]
    id_to_index: Dict[int, int]


def runtime_dependency_error(context: str, exc: ModuleNotFoundError) -> RuntimeError:
    error = RuntimeError(
        f"Could not import project runtime dependency while {context}. "
        "Install the repository dependencies before running inference."
    )
    error.__cause__ = exc
    return error


def dataset_factories() -> Dict[str, type]:
    global _DATASET_FACTORIES
    if _DATASET_FACTORIES is None:
        try:
            from datasets.cuhkpedes import CUHKPEDES
            from datasets.icfgpedes import ICFGPEDES
            from datasets.rstpreid import RSTPReid
        except ModuleNotFoundError as exc:
            raise runtime_dependency_error("loading dataset classes", exc)
        _DATASET_FACTORIES = {
            "CUHK-PEDES": CUHKPEDES,
            "ICFG-PEDES": ICFGPEDES,
            "RSTPReid": RSTPReid,
        }
    return _DATASET_FACTORIES


def text_dataset_class() -> type:
    try:
        from datasets.bases import TextDataset
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading the text dataset wrapper", exc)
    return TextDataset


def image_dataset_class() -> type:
    try:
        from datasets.bases import ImageDataset
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading the image dataset wrapper", exc)
    return ImageDataset


def build_eval_transforms(img_size: Tuple[int, int]) -> Any:
    try:
        from datasets.build import build_transforms
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading image transforms", exc)
    return build_transforms(img_size=img_size, is_train=False)


def build_repo_model(args: SimpleNamespace, num_classes: int) -> torch.nn.Module:
    try:
        from model import build_model
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("building the model", exc)
    return build_model(args, num_classes=num_classes)


def evaluator_class() -> type:
    try:
        from utils.metrics import Evaluator
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading the standard retrieval evaluator", exc)
    return Evaluator


def load_train_config(path: Path) -> Dict[str, Any]:
    try:
        from utils.iotools import load_train_configs
    except ModuleNotFoundError as exc:
        raise runtime_dependency_error("loading saved train config", exc)
    return config_to_dict(load_train_configs(str(path)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confusion-guided identity embedding visualization for Baseline vs IAPR checkpoints."
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=sorted(GENERIC_DATASET_CONFIGS),
        help="Dataset name.",
    )
    parser.add_argument(
        "--dataset_root",
        "--data_root",
        "--root_dir",
        dest="dataset_root",
        default=None,
        help="Dataset folder or parent folder. If omitted, root_dir is read from the configs.",
    )
    parser.add_argument("--split", default="test", choices=["test"], help="Dataset split to visualize. Test only.")
    parser.add_argument("--baseline_config", default=None, help="Optional baseline configs.yaml override.")
    parser.add_argument("--iapr_config", default=None, help="Optional IAPR configs.yaml override.")
    parser.add_argument("--baseline_checkpoint", "--baseline_ckpt", dest="baseline_checkpoint", required=True)
    parser.add_argument("--iapr_checkpoint", "--ours_checkpoint", "--iapr_ckpt", dest="iapr_checkpoint", required=True)
    parser.add_argument("--baseline_name", default="Baseline", help="Left subplot title.")
    parser.add_argument("--iapr_name", default="IAPR", help="Right subplot title.")
    parser.add_argument("--output_dir", default="outputs/identity_embedding_compare", help="Directory to save outputs.")

    parser.add_argument("--group_size", type=int, default=15, help="Number of identities per group.")
    parser.add_argument("--min_samples_per_id", type=int, default=4, help="Minimum image+text embeddings per identity.")
    parser.add_argument("--max_samples_per_id", type=int, default=20, help="Visualization sample cap per identity.")
    parser.add_argument("--num_candidate_groups", type=int, default=200, help="Maximum confusion-guided groups to score.")
    parser.add_argument("--top_groups", type=int, default=1, help="Number of top-scoring groups to export.")
    parser.add_argument("--reducer", choices=["pca", "tsne"], default="tsne", help="2D reducer.")
    parser.add_argument(
        "--pca_before_tsne",
        dest="pca_before_tsne",
        action="store_true",
        default=True,
        help="Reduce to 50D with PCA before t-SNE when possible. Enabled by default.",
    )
    parser.add_argument(
        "--no_pca_before_tsne",
        dest="pca_before_tsne",
        action="store_false",
        help="Disable PCA preprocessing before t-SNE.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ambiguity_eps", type=float, default=0.01, help="Margin threshold for ambiguous samples.")
    parser.add_argument("--plot_prototypes", action="store_true", help="Plot identity prototypes as star markers.")
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Accepted for explicitness; the group score CSV is always saved.",
    )
    parser.add_argument("--save_npz", action="store_true", help="Save selected embeddings and 2D coordinates.")

    parser.add_argument("--w_margin", type=float, default=1.0)
    parser.add_argument("--w_ambiguity", type=float, default=1.0)
    parser.add_argument("--w_separation", type=float, default=1.0)
    parser.add_argument("--w_compactness", type=float, default=1.0)
    parser.add_argument("--w_baseline_ambiguity", type=float, default=0.25)
    parser.add_argument("--w_imbalance", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for feature extraction.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", default="cuda", help='Device, e.g. "cuda" or "cpu".')
    parser.add_argument("--img_size", default=None, help='Override input image size as "height,width".')
    parser.add_argument("--text_length", type=int, default=None, help="Override tokenized text length.")
    parser.add_argument("--pretrain_choice", default=None, help="Override CLIP backbone choice.")
    parser.add_argument(
        "--model_type",
        default="itself",
        choices=["clip", "itself"],
        help="Default inference/model mode for both checkpoints.",
    )
    parser.add_argument("--baseline_model_type", default=None, choices=["clip", "itself"])
    parser.add_argument("--iapr_model_type", default=None, choices=["clip", "itself"])
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    return parser.parse_args()


def resolve_path(path: str | Path, base: Path = REPO_ROOT) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = base / resolved
    return resolved.resolve()


def parse_img_size(value: Any) -> Tuple[int, int]:
    if value is None:
        return (384, 128)
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, str):
        cleaned = value.strip().strip("()[]")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Could not parse --img_size from value: {value!r}")
        return (int(parts[0]), int(parts[1]))
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Could not parse --img_size from value: {value!r}")


def config_to_dict(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return {key: getattr(config, key) for key in dir(config) if not key.startswith("_")}


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return str(value)


def checkpoint_config_dict(checkpoint_path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch_load_checkpoint(checkpoint_path)
    except Exception as exc:
        print(f"[Config] Warning: could not inspect checkpoint metadata for {checkpoint_path}: {exc}")
        return {}
    if not isinstance(checkpoint, Mapping):
        return {}
    for key in CHECKPOINT_CONFIG_KEYS:
        value = checkpoint.get(key)
        if value is None:
            continue
        cfg = config_to_dict(value)
        if cfg:
            return cfg
    return {}


def infer_config_path_from_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    search_dirs: List[Path] = []
    for parent in [checkpoint_path.parent] + list(checkpoint_path.parents):
        if parent in search_dirs:
            continue
        search_dirs.append(parent)
        if parent == REPO_ROOT or len(search_dirs) >= 8:
            break

    for directory in search_dirs:
        for filename in CONFIG_FILE_NAMES:
            candidate = directory / filename
            if candidate.is_file():
                return candidate.resolve()
    return None


def resolve_config_for_checkpoint(
    label: str,
    explicit_config: Optional[str],
    checkpoint_path: Path,
) -> Tuple[Dict[str, Any], Optional[Path], str]:
    if explicit_config:
        config_path = resolve_path(explicit_config)
        if not config_path.is_file():
            raise FileNotFoundError(f"{label} config file not found: {config_path}")
        return load_train_config(config_path), config_path, "manual"

    inferred_path = infer_config_path_from_checkpoint(checkpoint_path)
    if inferred_path is not None:
        return load_train_config(inferred_path), inferred_path, "checkpoint_nearby_configs.yaml"

    checkpoint_cfg = checkpoint_config_dict(checkpoint_path)
    if checkpoint_cfg:
        return checkpoint_cfg, None, "checkpoint_metadata"

    return {}, None, "defaults_cli"


def default_model_args() -> Dict[str, Any]:
    return {
        "tau": 0.015,
        "select_ratio": 0.4,
        "margin": 0.1,
        "lambda1_weight": 0.5,
        "lambda2_weight": 3.5,
        "local_rank": 0,
        "output_dir": "run_logs",
        "name": "ITSELF",
        "run_time": "",
        "seed": 1,
        "deterministic": True,
        "log_period": 20,
        "eval_period": 1,
        "val_dataset": "test",
        "resume": False,
        "resume_ckpt_file": "",
        "finetune": "",
        "finetune_clip": "",
        "pretrain": "",
        "nohup": False,
        "nohup_log_dir": "logs",
        "wandb": False,
        "wandb_project": "ITSELF",
        "wandb_entity": "",
        "wandb_name": "",
        "wandb_mode": "disabled",
        "wandb_tags": [],
        "pretrain_choice": "ViT-B/16",
        "temperature": 0.02,
        "img_aug": False,
        "txt_aug": False,
        "loss_names": "tal+cid",
        "prototype": False,
        "use_loss_id": False,
        "no_pbt": False,
        "prototype_feature": "auto",
        "prototype_projector": "default",
        "prototype_residual_scale": 0.1,
        "prototype_per_id": 2,
        "prototype_dim": 512,
        "prototype_kmeans_iters": 20,
        "prototype_warmup_epochs": 0,
        "prototype_tau": 0.05,
        "prototype_hard_k": 16,
        "prototype_id_weight": 0.2,
        "prototype_momentum": 0.2,
        "img_size": (384, 128),
        "stride_size": 16,
        "text_length": 77,
        "vocab_size": 49408,
        "optimizer": "Adam",
        "lr": 1e-5,
        "bias_lr_factor": 2.0,
        "lr_factor": 5.0,
        "prototype_lr": None,
        "momentum": 0.9,
        "weight_decay": 4e-5,
        "weight_decay_bias": 0.0,
        "alpha": 0.9,
        "beta": 0.999,
        "num_epoch": 60,
        "lr_total_epochs": None,
        "milestones": (45, 50),
        "gamma": 0.1,
        "warmup_factor": 0.1,
        "warmup_epochs": 5,
        "warmup_method": "linear",
        "lrscheduler": "cosine",
        "target_lr": 0,
        "power": 0.9,
        "early_stop_patience": 0,
        "early_stop_min_delta": 0.0,
        "dataset_name": "RSTPReid",
        "sampler": "identity",
        "num_instance": 2,
        "root_dir": "data",
        "batch_size": 128,
        "test_batch_size": 128,
        "num_workers": 4,
        "training": False,
        "only_global": False,
        "return_all": False,
        "topk_type": "mean",
        "layer_index": -1,
        "average_attn_weights": True,
        "modify_k": False,
        "track_train_diagnostics": False,
    }


def build_model_args(
    cli_args: argparse.Namespace,
    loaded_config: Mapping[str, Any],
    model_type: str,
) -> SimpleNamespace:
    cfg = default_model_args()
    cfg.update(config_to_dict(loaded_config))

    cfg["dataset_name"] = cli_args.dataset_name
    if cli_args.dataset_root:
        cfg["root_dir"] = str(resolve_path(cli_args.dataset_root))
    elif cfg.get("root_dir"):
        cfg["root_dir"] = str(resolve_path(cfg["root_dir"]))
    else:
        raise ValueError("--dataset_root is required when the config does not provide root_dir.")

    cfg["training"] = False
    cfg["batch_size"] = int(cli_args.batch_size)
    cfg["test_batch_size"] = int(cli_args.batch_size)
    cfg["num_workers"] = int(cli_args.num_workers)
    cfg["img_size"] = parse_img_size(cli_args.img_size if cli_args.img_size is not None else cfg.get("img_size"))
    if cli_args.text_length is not None:
        cfg["text_length"] = int(cli_args.text_length)
    if cli_args.pretrain_choice is not None:
        cfg["pretrain_choice"] = cli_args.pretrain_choice
    cfg["model_type"] = model_type
    cfg["only_global"] = model_type == "clip"
    cfg["selected_inference"] = None
    cfg.setdefault("return_all", False)
    cfg.setdefault("topk_type", "mean")
    cfg.setdefault("layer_index", -1)
    cfg.setdefault("average_attn_weights", True)
    cfg.setdefault("modify_k", False)
    cfg.setdefault("track_train_diagnostics", False)
    cfg.setdefault("no_ira", False)
    cfg.setdefault("no_ira_mode", "hard")
    cfg.setdefault("no_iopm", False)
    cfg.setdefault("prototype_lr", None)
    return SimpleNamespace(**cfg)


def dataset_root_for_repo_class(factory: type, dataset_root: Path) -> Path:
    dataset_dir = getattr(factory, "dataset_dir", None)
    if dataset_dir and dataset_root.name.lower() == str(dataset_dir).lower():
        return dataset_root.parent
    return dataset_root


def validate_split_data(split_data: SplitData, split: str) -> None:
    if len(split_data.image_pids) != len(split_data.img_paths):
        raise ValueError(
            f"Image labels and image paths differ in length for split {split}: "
            f"{len(split_data.image_pids)} labels vs {len(split_data.img_paths)} paths."
        )
    if len(split_data.caption_pids) != len(split_data.captions):
        raise ValueError(
            f"Caption labels and captions differ in length for split {split}: "
            f"{len(split_data.caption_pids)} labels vs {len(split_data.captions)} captions."
        )
    if not split_data.image_pids:
        raise ValueError(f"No gallery images found for split {split}.")
    if not split_data.caption_pids:
        raise ValueError(f"No text queries found for split {split}.")
    if any(pid is None for pid in split_data.image_pids) or any(pid is None for pid in split_data.caption_pids):
        raise ValueError(
            f"Identity labels are required to visualize identity embeddings, but split {split} "
            "contains missing image or caption labels."
        )


def split_data_from_repo_dataset(dataset: Any, split: str) -> SplitData:
    train_ids = getattr(dataset, "train_id_container", set())
    num_train_ids = len(train_ids)

    if split == "train":
        if not hasattr(dataset, "train"):
            raise ValueError("The dataset object has no train split.")
        image_key_to_index: Dict[Tuple[int, str], int] = {}
        image_pids: List[int] = []
        img_paths: List[str] = []
        caption_pids: List[int] = []
        captions: List[str] = []

        for row in dataset.train:
            if len(row) != 4:
                raise ValueError("Expected train rows to be (pid, image_id, img_path, caption).")
            pid, _image_id, img_path, caption = row
            pid = int(pid)
            img_path = str(img_path)
            key = (pid, img_path)
            if key not in image_key_to_index:
                image_key_to_index[key] = len(img_paths)
                image_pids.append(pid)
                img_paths.append(img_path)
            caption_pids.append(pid)
            captions.append(str(caption))

        return SplitData(image_pids, img_paths, caption_pids, captions, num_train_ids)

    if not hasattr(dataset, split):
        raise ValueError(f"The dataset object has no {split!r} split.")
    split_obj = getattr(dataset, split)
    required = ["image_pids", "img_paths", "caption_pids", "captions"]
    missing = [key for key in required if key not in split_obj]
    if missing:
        raise ValueError(f"Split {split} is missing required fields: {missing}")

    return SplitData(
        image_pids=[int(pid) for pid in split_obj["image_pids"]],
        img_paths=[str(path) for path in split_obj["img_paths"]],
        caption_pids=[int(pid) for pid in split_obj["caption_pids"]],
        captions=[str(caption) for caption in split_obj["captions"]],
        num_train_ids=num_train_ids,
    )


def locate_generic_dataset_dir(dataset_name: str, dataset_root: Path) -> Path:
    cfg = GENERIC_DATASET_CONFIGS.get(dataset_name, {})
    dataset_dir = cfg.get("dataset_dir", dataset_name)
    if dataset_root.name.lower() == str(dataset_dir).lower():
        return dataset_root
    candidate = dataset_root / str(dataset_dir)
    if candidate.is_dir():
        return candidate
    return dataset_root


def locate_first_existing(base_dir: Path, names: Iterable[str], kind: str) -> Path:
    for name in names:
        candidate = base_dir / name if name else base_dir
        if kind == "file" and candidate.is_file():
            return candidate
        if kind == "dir" and candidate.is_dir():
            return candidate
    candidates = ", ".join(str(base_dir / name) for name in names if name)
    raise FileNotFoundError(f"Could not find expected {kind} under {base_dir}. Tried: {candidates}")


def annotation_pid(dataset_name: str, split: str, anno: Mapping[str, Any]) -> int:
    for key in DEFAULT_PID_KEYS:
        if key in anno and anno[key] is not None:
            pid = int(anno[key])
            if dataset_name == "CUHK-PEDES" and split == "train":
                pid -= 1
            return pid
    raise ValueError("Identity labels are required, but an annotation has no pid/id field.")


def annotation_captions(anno: Mapping[str, Any]) -> List[str]:
    for key in DEFAULT_CAPTION_KEYS:
        if key not in anno:
            continue
        value = anno[key]
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence):
            return [str(item) for item in value]
    raise ValueError("A text query field is required, but an annotation has no captions/caption/text field.")


def annotation_image_path(anno: Mapping[str, Any], dataset_dir: Path, image_dir: Path, path_keys: Sequence[str]) -> str:
    rel_path: Optional[str] = None
    for key in path_keys:
        if key in anno and anno[key]:
            rel_path = str(anno[key])
            break
    if rel_path is None:
        raise ValueError(f"An image path field is required, but an annotation has none of: {list(path_keys)}")

    candidate = Path(rel_path).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())

    img_candidate = image_dir / candidate
    if img_candidate.exists():
        return str(img_candidate.resolve())

    dataset_candidate = dataset_dir / candidate
    if dataset_candidate.exists():
        return str(dataset_candidate.resolve())

    return str(img_candidate.resolve())


def split_annotations(raw: Any, split: str) -> List[Mapping[str, Any]]:
    if isinstance(raw, Mapping):
        if split in raw and isinstance(raw[split], Sequence):
            return list(raw[split])
        for key in ("annotations", "annos", "data"):
            if key in raw and isinstance(raw[key], Sequence):
                raw = raw[key]
                break
    if not isinstance(raw, Sequence):
        raise ValueError("Annotation JSON must be a list or a dict containing split lists.")

    annos = []
    for anno in raw:
        if not isinstance(anno, Mapping):
            continue
        anno_split = anno.get("split")
        if anno_split == split:
            annos.append(anno)
        elif split == "val" and anno_split not in ("train", "test"):
            annos.append(anno)
    return annos


def generic_split_data(dataset_name: str, dataset_root: Path, split: str) -> SplitData:
    cfg = GENERIC_DATASET_CONFIGS.get(dataset_name, {})
    dataset_dir = locate_generic_dataset_dir(dataset_name, dataset_root)
    annotation_files = list(cfg.get("annotation_files", [])) + DEFAULT_ANNOTATION_FILES
    image_dirs = list(cfg.get("image_dirs", [])) + DEFAULT_IMAGE_DIRS
    path_keys = list(cfg.get("path_keys", [])) + DEFAULT_PATH_KEYS

    annotation_path = locate_first_existing(dataset_dir, annotation_files, "file")
    image_dir = locate_first_existing(dataset_dir, image_dirs, "dir")

    with annotation_path.open("r", encoding="utf-8") as file:
        raw_annos = json.load(file)

    selected_annos = split_annotations(raw_annos, split)
    if not selected_annos:
        raise ValueError(f"No annotations found for split {split} in {annotation_path}.")

    all_annos_for_train = split_annotations(raw_annos, "train")
    train_pids = {
        annotation_pid(dataset_name, "train", anno)
        for anno in all_annos_for_train
        if isinstance(anno, Mapping)
    }

    image_key_to_index: Dict[Tuple[int, str], int] = {}
    image_pids: List[int] = []
    img_paths: List[str] = []
    caption_pids: List[int] = []
    captions: List[str] = []

    for anno in selected_annos:
        pid = annotation_pid(dataset_name, split, anno)
        img_path = annotation_image_path(anno, dataset_dir, image_dir, path_keys)
        image_key = (pid, img_path)
        if image_key not in image_key_to_index:
            image_key_to_index[image_key] = len(img_paths)
            image_pids.append(pid)
            img_paths.append(img_path)
        for caption in annotation_captions(anno):
            caption_pids.append(pid)
            captions.append(caption)

    num_train_ids = len(train_pids) if train_pids else len(set(image_pids) | set(caption_pids))
    return SplitData(image_pids, img_paths, caption_pids, captions, num_train_ids)


def load_split_data(dataset_name: str, dataset_root: Path, split: str) -> SplitData:
    factory = dataset_factories().get(dataset_name)
    if factory is not None:
        root_for_factory = dataset_root_for_repo_class(factory, dataset_root)
        dataset = factory(root=str(root_for_factory), verbose=False)
        split_data = split_data_from_repo_dataset(dataset, split)
    else:
        split_data = generic_split_data(dataset_name, dataset_root, split)

    validate_split_data(split_data, split)
    return split_data


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "net", "network", "module"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping or contain a model state mapping.")
    return checkpoint


def strip_repeated_prefixes(key: str, prefixes: Sequence[str]) -> str:
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                changed = True
    return stripped


def candidate_state_keys(key: str) -> List[str]:
    raw = str(key)
    candidates: List[str] = []

    def add(candidate: str) -> None:
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(raw)
    no_module = strip_repeated_prefixes(raw, ("module.",))
    add(no_module)
    no_wrapper = strip_repeated_prefixes(no_module, ("model.", "net.", "network."))
    add(no_wrapper)
    normalized = strip_repeated_prefixes(raw, ("module.", "model.", "net.", "network."))
    add(normalized)
    for candidate in (no_wrapper, normalized):
        if candidate.startswith("base_model."):
            add(candidate[len("base_model."):])
        else:
            add(f"base_model.{candidate}")
    if no_module.startswith("model."):
        add(no_module[len("model."):])
    return candidates


def torch_load_checkpoint(checkpoint_path: Path) -> Any:
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def load_checkpoint_for_inference(model: torch.nn.Module, checkpoint_path: Path) -> Dict[str, int]:
    checkpoint = torch_load_checkpoint(checkpoint_path)
    loaded_state = checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    update_state: MutableMapping[str, torch.Tensor] = {}
    skipped_missing = 0
    skipped_shape = 0
    skipped_non_tensor = 0

    for raw_key, value in loaded_state.items():
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            continue

        target_key = None
        for candidate in candidate_state_keys(str(raw_key)):
            if candidate in model_state:
                target_key = candidate
                break

        if target_key is None:
            skipped_missing += 1
            continue
        if model_state[target_key].shape != value.shape:
            skipped_shape += 1
            continue
        update_state[target_key] = value.detach().clone()

    if not update_state:
        raise RuntimeError(f"No compatible tensors found in checkpoint: {checkpoint_path}")

    model_state.update(update_state)
    model.load_state_dict(model_state, strict=True)
    return {
        "loaded": len(update_state),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "skipped_non_tensor": skipped_non_tensor,
    }


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA was requested but is not available; using CPU.")
        return torch.device("cpu")
    return device


def feature_tensor(output: Any, kind: str) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise RuntimeError(f"model.encode_{kind}(...) returned {type(output)!r}, expected a tensor.")
    return output


def call_model_encoder(encoder: Any, tensor: torch.Tensor, kind: str) -> torch.Tensor:
    params = inspect.signature(encoder).parameters
    output = encoder(tensor, 0) if len(params) >= 2 else encoder(tensor)
    return feature_tensor(output, kind)


@torch.inference_mode()
def extract_text_features_from_encoder(
    model: torch.nn.Module,
    split_data: SplitData,
    text_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    encoder_name: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder = getattr(model, encoder_name, None)
    if encoder is None:
        raise RuntimeError(f"Model does not provide {encoder_name}; choose --model_type clip for global-only checkpoints.")

    TextDataset = text_dataset_class()
    text_set = TextDataset(split_data.caption_pids, split_data.captions, text_length=text_length)
    loader = DataLoader(
        text_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []

    model.eval()
    for pid, tokens in tqdm(loader, desc=desc):
        tokens = tokens.to(device, non_blocking=True)
        feats = call_model_encoder(encoder, tokens, encoder_name).float()
        features.append(feats.cpu())
        pids.append(pid.view(-1).cpu().long())

    if not features:
        raise RuntimeError(f"No text features were extracted from {encoder_name}.")

    text_features = F.normalize(torch.cat(features, dim=0), p=2, dim=1)
    text_pids = torch.cat(pids, dim=0).long()
    return text_features, text_pids


@torch.inference_mode()
def extract_image_features_from_encoder(
    model: torch.nn.Module,
    split_data: SplitData,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    encoder_name: str,
    desc: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoder = getattr(model, encoder_name, None)
    if encoder is None:
        raise RuntimeError(f"Model does not provide {encoder_name}; choose --model_type clip for global-only checkpoints.")

    ImageDataset = image_dataset_class()
    transform = build_eval_transforms(img_size)
    image_set = ImageDataset(split_data.image_pids, split_data.img_paths, transform=transform)
    loader = DataLoader(
        image_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    features: List[torch.Tensor] = []
    pids: List[torch.Tensor] = []

    model.eval()
    for pid, images in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        feats = call_model_encoder(encoder, images, encoder_name).float()
        features.append(feats.cpu())
        pids.append(pid.view(-1).cpu().long())

    if not features:
        raise RuntimeError(f"No image features were extracted from {encoder_name}.")

    image_features = F.normalize(torch.cat(features, dim=0), p=2, dim=1)
    image_pids = torch.cat(pids, dim=0).long()
    return image_features, image_pids


def extract_text_features(
    model: torch.nn.Module,
    split_data: SplitData,
    text_length: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return extract_text_features_from_encoder(
        model,
        split_data,
        text_length=text_length,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_text",
        desc="Extracting text features",
    )


def extract_image_features(
    model: torch.nn.Module,
    split_data: SplitData,
    img_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return extract_image_features_from_encoder(
        model,
        split_data,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_image",
        desc="Extracting image features",
    )


def ensure_same_pids(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.shape != right.shape or not torch.equal(left, right):
        raise RuntimeError(f"{label} identity order changed between global and GRAB feature extraction.")


ITSELF_ABLATION_CANDIDATES: List[Tuple[str, float]] = [
    ("global", 1.0),
    ("grab", 0.0),
    ("global+grab(0.1)", 0.1),
    ("global+grab(0.2)", 0.2),
    ("global+grab(0.3)", 0.3),
    ("global+grab(0.4)", 0.4),
    ("global+grab(0.5)", 0.5),
    ("global+grab(0.6)", 0.6),
    ("global+grab(0.7)", 0.7),
    ("global+grab(0.8)", 0.8),
    ("global+grab(0.9)", 0.9),
    ("global+grab(0.68)", 0.68),
    ("global+grab(0.32)", 0.32),
]


def clip_inference_metadata() -> Dict[str, Any]:
    return {
        "model_type": "clip",
        "inference_mode": "clip_global",
        "ablation_task": "global-t2i",
        "global_weight": 1.0,
        "grab_weight": 0.0,
        "selection_source": "clip_global",
    }


def normalize_ablation_task(task: str) -> str:
    task = str(task).strip()
    for suffix in ("-t2i", "-i2t"):
        if task.endswith(suffix):
            return task[: -len(suffix)]
    return task


def itself_inference_metadata(
    ablation_name: str,
    global_weight: float,
    selection_source: str,
    selected_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    grab_weight = 1.0 - float(global_weight)
    metadata: Dict[str, Any] = {
        "model_type": "itself",
        "inference_mode": "itself_best_ablation",
        "ablation_task": f"{ablation_name}-t2i",
        "global_weight": float(global_weight),
        "grab_weight": float(grab_weight),
        "formula": "global_weight * s_global + grab_weight * s_grab",
        "selection_source": selection_source,
    }
    if selected_metrics:
        metadata["selected_metrics"] = {
            str(key): float(value)
            for key, value in selected_metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    return metadata


def itself_inference_metadata_from_task(
    task: str,
    selection_source: str,
    selected_metrics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    ablation_name = normalize_ablation_task(task)
    for candidate_name, global_weight in ITSELF_ABLATION_CANDIDATES:
        if ablation_name == candidate_name:
            return itself_inference_metadata(
                candidate_name,
                global_weight,
                selection_source=selection_source,
                selected_metrics=selected_metrics,
            )
    valid = ", ".join(f"{name}-t2i" for name, _ in ITSELF_ABLATION_CANDIDATES)
    raise RuntimeError(
        f"ITSELF evaluator selected unsupported ablation task {task!r}. "
        f"Expected one of: {valid}."
    )


def selected_inference_from_standard_eval(
    model_args: SimpleNamespace,
    best_metrics: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    model_type = getattr(model_args, "model_type", "itself")
    if model_type == "clip":
        return clip_inference_metadata()
    if model_type != "itself":
        raise ValueError(f"Unsupported model_type: {model_type!r}")

    task = best_metrics.get("task") if best_metrics else None
    if task:
        return itself_inference_metadata_from_task(
            str(task),
            selection_source="standard_test_eval_best_r1",
            selected_metrics=best_metrics,
        )
    return None


def selected_itself_inference_from_args(model_args: SimpleNamespace) -> Optional[Dict[str, Any]]:
    selection = getattr(model_args, "selected_inference", None)
    if not selection:
        return None
    if not isinstance(selection, Mapping):
        raise RuntimeError(f"selected_inference must be a mapping, got {type(selection)!r}.")

    task = selection.get("ablation_task") or selection.get("task")
    selected_metrics = selection.get("selected_metrics") if isinstance(selection.get("selected_metrics"), Mapping) else None
    if task:
        metadata = itself_inference_metadata_from_task(
            str(task),
            selection_source=str(selection.get("selection_source", "stored_selection")),
            selected_metrics=selected_metrics,
        )
    elif "global_weight" in selection:
        global_weight = float(selection["global_weight"])
        if not math.isfinite(global_weight) or global_weight < 0.0 or global_weight > 1.0:
            raise RuntimeError(f"Stored ITSELF global_weight must be in [0, 1], got {global_weight!r}.")
        metadata = itself_inference_metadata(
            f"global+grab({global_weight:g})",
            global_weight,
            selection_source=str(selection.get("selection_source", "stored_selection")),
            selected_metrics=selected_metrics,
        )
    else:
        raise RuntimeError(
            "ITSELF selected_inference is missing an ablation_task or global_weight. "
            "Run the test-set verification step before identity visualization."
        )

    return metadata


def combine_itself_similarity(sim_global: torch.Tensor, sim_grab: torch.Tensor, global_weight: float) -> torch.Tensor:
    grab_weight = 1.0 - float(global_weight)
    return float(global_weight) * sim_global + grab_weight * sim_grab


def retrieval_metrics_from_similarity(
    sim: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
) -> Dict[str, float]:
    if sim.ndim != 2:
        raise ValueError(f"Expected a 2D similarity matrix, got shape {tuple(sim.shape)}.")
    if sim.shape[0] != query_pids.numel() or sim.shape[1] != gallery_pids.numel():
        raise ValueError("Similarity shape does not match query/gallery pid counts.")

    indices = torch.argsort(sim, dim=1, descending=True)
    pred_labels = gallery_pids[indices.cpu()]
    matches = pred_labels.eq(query_pids.view(-1, 1))
    num_rel = matches.sum(1)
    valid = num_rel > 0
    if not bool(valid.any()):
        raise RuntimeError("No query has a positive gallery match; cannot compute retrieval metrics.")

    matches = matches[valid]
    num_rel = num_rel[valid]
    max_rank = min(10, matches.shape[1])
    cmc = matches[:, :max_rank].cumsum(1)
    cmc[cmc > 1] = 1
    cmc = cmc.float().mean(0) * 100.0

    cumulative = matches.cumsum(1)
    ranks = torch.arange(1, matches.shape[1] + 1, dtype=torch.float32).view(1, -1)
    precision_at_rank = cumulative.float() / ranks
    ap = (precision_at_rank * matches.float()).sum(1) / num_rel.float()
    positive_positions = [int(row.nonzero(as_tuple=False)[-1].item()) for row in matches]
    minp_values = [
        cumulative[row_idx, pos] / (pos + 1.0)
        for row_idx, pos in enumerate(positive_positions)
    ]
    mINP = torch.stack(minp_values).float().mean() * 100.0
    mAP = ap.mean() * 100.0

    def recall_at(rank: int) -> float:
        index = min(rank - 1, max_rank - 1)
        return float(cmc[index].item())

    r1 = recall_at(1)
    r5 = recall_at(5)
    r10 = recall_at(10)
    return {
        "R1": r1,
        "R5": r5,
        "R10": r10,
        "mAP": float(mAP.item()),
        "mINP": float(mINP.item()),
        "rSum": r1 + r5 + r10,
        "num_queries_used": int(valid.sum().item()),
    }


def select_best_itself_ablation_from_sims(
    sim_global: torch.Tensor,
    sim_grab: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
) -> Dict[str, Any]:
    best_metadata: Optional[Dict[str, Any]] = None
    best_r1 = float("-inf")
    for ablation_name, global_weight in ITSELF_ABLATION_CANDIDATES:
        candidate_sim = combine_itself_similarity(sim_global, sim_grab, global_weight)
        metrics = retrieval_metrics_from_similarity(candidate_sim, query_pids, gallery_pids)
        if metrics["R1"] >= best_r1:
            best_metadata = itself_inference_metadata(
                ablation_name,
                global_weight,
                selection_source="local_similarity_sweep_best_r1",
                selected_metrics=metrics,
            )
            best_r1 = metrics["R1"]

    if best_metadata is None:
        raise RuntimeError("Could not select an ITSELF ablation combo from similarity matrices.")
    return best_metadata


def compute_inference_similarity(
    model: torch.nn.Module,
    split_data: SplitData,
    model_args: SimpleNamespace,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    text_features, query_pids = extract_text_features(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    image_features, gallery_pids = extract_image_features(
        model,
        split_data,
        img_size=parse_img_size(model_args.img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    sim_global = text_features @ image_features.t()

    model_type = getattr(model_args, "model_type", "itself")
    if model_type == "clip":
        return sim_global, query_pids, gallery_pids, clip_inference_metadata()

    if model_type != "itself":
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    if not hasattr(model, "encode_text_grab") or not hasattr(model, "encode_image_grab"):
        raise RuntimeError("ITSELF inference requires encode_text_grab/encode_image_grab; use --model_type clip for CLIP-only checkpoints.")

    text_grab_features, grab_query_pids = extract_text_features_from_encoder(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_text_grab",
        desc="Extracting GRAB text features",
    )
    image_grab_features, grab_gallery_pids = extract_image_features_from_encoder(
        model,
        split_data,
        img_size=parse_img_size(model_args.img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_image_grab",
        desc="Extracting GRAB image features",
    )
    ensure_same_pids(query_pids, grab_query_pids, "Query")
    ensure_same_pids(gallery_pids, grab_gallery_pids, "Gallery")

    sim_grab = text_grab_features @ image_grab_features.t()
    metadata = selected_itself_inference_from_args(model_args)
    if metadata is None:
        print(
            "Warning: no ITSELF ablation selection was attached; "
            "selecting the best combo from the current split."
        )
        metadata = select_best_itself_ablation_from_sims(
            sim_global,
            sim_grab,
            query_pids,
            gallery_pids,
        )
    sim = combine_itself_similarity(sim_global, sim_grab, float(metadata["global_weight"]))
    return sim, query_pids, gallery_pids, metadata


def combine_itself_embeddings(
    global_features: torch.Tensor,
    grab_features: torch.Tensor,
    global_weight: float,
) -> torch.Tensor:
    global_weight = float(global_weight)
    grab_weight = 1.0 - global_weight
    if global_weight < 0.0 or grab_weight < 0.0:
        raise ValueError(f"ITSELF global_weight must be in [0, 1], got {global_weight}.")

    parts: List[torch.Tensor] = []
    if global_weight > 0.0:
        parts.append(math.sqrt(global_weight) * global_features)
    if grab_weight > 0.0:
        parts.append(math.sqrt(grab_weight) * grab_features)
    if not parts:
        raise RuntimeError("No embedding branch remained after applying ITSELF weights.")
    combined = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    return F.normalize(combined, p=2, dim=1)


def build_embedding_bundle(
    image_features: torch.Tensor,
    image_pids: torch.Tensor,
    text_features: torch.Tensor,
    text_pids: torch.Tensor,
    checkpoint: Path,
    load_stats: Mapping[str, Any],
    inference: Mapping[str, Any],
) -> EmbeddingBundle:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("Image and text feature tensors must both be 2D.")
    if image_features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            "Image and text embeddings have different dimensions "
            f"({image_features.shape[1]} vs {text_features.shape[1]}), so a shared identity prototype space cannot be built."
        )

    features = torch.cat([image_features, text_features], dim=0).cpu().numpy().astype(np.float32)
    features = l2_normalize(features)
    pids = np.concatenate([image_pids.cpu().numpy(), text_pids.cpu().numpy()]).astype(np.int64)
    modalities = np.array(["image"] * int(image_pids.numel()) + ["text"] * int(text_pids.numel()))
    source_indices = np.concatenate([
        np.arange(int(image_pids.numel()), dtype=np.int64),
        np.arange(int(text_pids.numel()), dtype=np.int64),
    ])
    return EmbeddingBundle(
        features=features,
        pids=pids,
        modalities=modalities,
        source_indices=source_indices,
        checkpoint=str(checkpoint),
        load_stats=dict(load_stats),
        inference=dict(inference),
    )


def extract_embedding_bundle(
    model: torch.nn.Module,
    split_data: SplitData,
    model_args: SimpleNamespace,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    checkpoint: Path,
    load_stats: Mapping[str, Any],
) -> EmbeddingBundle:
    text_global, text_pids = extract_text_features(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )
    image_global, image_pids = extract_image_features(
        model,
        split_data,
        img_size=parse_img_size(model_args.img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    model_type = getattr(model_args, "model_type", "itself")
    if model_type == "clip":
        return build_embedding_bundle(
            image_global,
            image_pids,
            text_global,
            text_pids,
            checkpoint,
            load_stats,
            clip_inference_metadata(),
        )

    if model_type != "itself":
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    if not hasattr(model, "encode_text_grab") or not hasattr(model, "encode_image_grab"):
        raise RuntimeError("ITSELF embedding extraction requires encode_text_grab/encode_image_grab.")

    text_grab, grab_text_pids = extract_text_features_from_encoder(
        model,
        split_data,
        text_length=int(model_args.text_length),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_text_grab",
        desc="Extracting GRAB text features",
    )
    image_grab, grab_image_pids = extract_image_features_from_encoder(
        model,
        split_data,
        img_size=parse_img_size(model_args.img_size),
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        encoder_name="encode_image_grab",
        desc="Extracting GRAB image features",
    )
    ensure_same_pids(text_pids, grab_text_pids, "Query")
    ensure_same_pids(image_pids, grab_image_pids, "Gallery")

    sim_global = text_global @ image_global.t()
    sim_grab = text_grab @ image_grab.t()
    inference = selected_itself_inference_from_args(model_args)
    if inference is None:
        print(
            "Warning: no ITSELF ablation selection was attached; "
            "selecting the best combo from the current split."
        )
        inference = select_best_itself_ablation_from_sims(sim_global, sim_grab, text_pids, image_pids)
    global_weight = float(inference["global_weight"])
    text_features = combine_itself_embeddings(text_global, text_grab, global_weight)
    image_features = combine_itself_embeddings(image_global, image_grab, global_weight)
    return build_embedding_bundle(
        image_features,
        image_pids,
        text_features,
        text_pids,
        checkpoint,
        load_stats,
        inference,
    )


def load_model_for_checkpoint(
    checkpoint_path: str | Path,
    model_args: SimpleNamespace,
    split_data: SplitData,
    device: torch.device,
) -> Tuple[torch.nn.Module, SimpleNamespace, Dict[str, int], Path]:
    checkpoint = resolve_path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")

    run_args = SimpleNamespace(**vars(model_args))
    num_classes = max(int(split_data.num_train_ids), 1)
    model = build_repo_model(run_args, num_classes=num_classes)
    load_stats = load_checkpoint_for_inference(model, checkpoint)

    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model, run_args, load_stats, checkpoint


def load_embeddings_for_checkpoint(
    label: str,
    checkpoint_path: str | Path,
    model_args: SimpleNamespace,
    split_data: SplitData,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> EmbeddingBundle:
    model: Optional[torch.nn.Module] = None
    try:
        model, run_args, load_stats, checkpoint = load_model_for_checkpoint(
            checkpoint_path,
            model_args,
            split_data,
            device,
        )
        print_load_stats(label, {"load_stats": load_stats})
        bundle = extract_embedding_bundle(
            model,
            split_data,
            run_args,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            checkpoint=checkpoint,
            load_stats=load_stats,
        )
        return bundle
    finally:
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def build_standard_eval_loaders(
    split_data: SplitData,
    model_args: SimpleNamespace,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[DataLoader, DataLoader]:
    ImageDataset = image_dataset_class()
    TextDataset = text_dataset_class()
    transform = build_eval_transforms(parse_img_size(model_args.img_size))
    image_set = ImageDataset(split_data.image_pids, split_data.img_paths, transform=transform)
    text_set = TextDataset(split_data.caption_pids, split_data.captions, text_length=int(model_args.text_length))
    image_loader = DataLoader(
        image_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    text_loader = DataLoader(
        text_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    return image_loader, text_loader


def build_standard_evaluator(image_loader: DataLoader, text_loader: DataLoader, model_args: SimpleNamespace) -> Any:
    Evaluator = evaluator_class()
    params = inspect.signature(Evaluator).parameters
    if "args" in params or len(params) >= 3:
        return Evaluator(image_loader, text_loader, model_args)
    return Evaluator(image_loader, text_loader)


def numeric_mapping(mapping: Mapping[str, Any]) -> Dict[str, float]:
    numeric: Dict[str, float] = {}
    for key, value in mapping.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric[str(key)] = float(value)
    return numeric


def run_standard_retrieval_eval(evaluator: Any, model: torch.nn.Module) -> Tuple[Dict[str, float], Dict[str, Any]]:
    if hasattr(evaluator, "eval_metrics"):
        metrics = evaluator.eval_metrics(model)
        return numeric_mapping(metrics), {}

    eval_params = inspect.signature(evaluator.eval).parameters
    kwargs: Dict[str, Any] = {}
    if "return_metrics" in eval_params:
        kwargs["return_metrics"] = True
    result = evaluator.eval(model, **kwargs)

    if isinstance(result, tuple):
        top1 = result[0]
        metrics = result[1] if len(result) > 1 and isinstance(result[1], Mapping) else {"R1": top1}
        best_metrics = result[2] if len(result) > 2 and isinstance(result[2], Mapping) else {}
        return numeric_mapping(metrics), dict(best_metrics)

    return {"R1": float(result)}, {}


def split_metric_key(key: str) -> Tuple[str, str]:
    tail = key.rsplit("/", 1)[-1]
    if "_" in tail:
        task, metric = key.rsplit("_", 1)
        return task, metric
    if "/" in key:
        task, metric = key.rsplit("/", 1)
        return task, metric
    return "retrieval", key


def grouped_retrieval_metrics(metrics: Mapping[str, float]) -> List[Tuple[str, Dict[str, float]]]:
    wanted = {"R1", "R5", "R10", "mAP", "mINP", "rSum"}
    groups: Dict[str, Dict[str, float]] = {}
    order: List[str] = []
    for key, value in metrics.items():
        task, metric = split_metric_key(str(key))
        if metric not in wanted:
            continue
        if task not in groups:
            groups[task] = {}
            order.append(task)
        groups[task][metric] = float(value)
    return [(task, groups[task]) for task in order]


def print_retrieval_metrics(label: str, metrics: Mapping[str, float], best_metrics: Mapping[str, Any]) -> None:
    print(f"[{label}] Standard test-set retrieval metrics:")
    metric_order = [("R1", "R@1"), ("R5", "R@5"), ("R10", "R@10"), ("mAP", "mAP"), ("mINP", "mINP"), ("rSum", "rSum")]
    if best_metrics:
        best_parts = []
        for key, display in metric_order:
            if key in best_metrics:
                best_parts.append(f"{display}={float(best_metrics[key]):.2f}")
        task = best_metrics.get("task", "best")
        if best_parts:
            print(f"  best ({task}): " + ", ".join(best_parts))

    groups = grouped_retrieval_metrics(metrics)
    if not groups:
        print(f"  No standard R@/mAP metrics were returned. Raw metric keys: {sorted(metrics)}")
        return

    for task, values in groups:
        parts = [f"{display}={values[key]:.2f}" for key, display in metric_order if key in values]
        print(f"  {task}: " + ", ".join(parts))


def print_selected_inference_metrics(label: str, metrics: Mapping[str, float], inference: Mapping[str, Any]) -> None:
    mode = inference.get("inference_mode", "selected")
    if inference.get("model_type") == "itself":
        task = inference.get("ablation_task", "best-t2i")
        global_weight = float(inference.get("global_weight", 0.0))
        grab_weight = float(inference.get("grab_weight", 0.0))
        mode = f"{mode} ({task}, global={global_weight:.4g}, grab={grab_weight:.4g})"
    parts = [
        f"R@1={metrics['R1']:.2f}",
        f"R@5={metrics['R5']:.2f}",
        f"R@10={metrics['R10']:.2f}",
        f"mAP={metrics['mAP']:.2f}",
        f"mINP={metrics['mINP']:.2f}",
        f"rSum={metrics['rSum']:.2f}",
    ]
    print(f"[{label}] Selected inference metrics for visualization ({mode}): " + ", ".join(parts))


def evaluate_checkpoint_on_test_split(
    label: str,
    checkpoint_path: str | Path,
    model_args: SimpleNamespace,
    test_split_data: SplitData,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Dict[str, Any]:
    model: Optional[torch.nn.Module] = None
    try:
        model, run_args, load_stats, checkpoint = load_model_for_checkpoint(
            checkpoint_path,
            model_args,
            test_split_data,
            device,
        )
        print_load_stats(label, {"load_stats": load_stats})
        image_loader, text_loader = build_standard_eval_loaders(
            test_split_data,
            run_args,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        evaluator = build_standard_evaluator(image_loader, text_loader, run_args)
        metrics, best_metrics = run_standard_retrieval_eval(evaluator, model)
        print_retrieval_metrics(label, metrics, best_metrics)
        selected_inference = selected_inference_from_standard_eval(run_args, best_metrics)
        if selected_inference is not None:
            run_args.selected_inference = selected_inference
        selected_sim, selected_query_pids, selected_gallery_pids, inference = compute_inference_similarity(
            model,
            test_split_data,
            run_args,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        selected_metrics = retrieval_metrics_from_similarity(
            selected_sim,
            selected_query_pids,
            selected_gallery_pids,
        )
        print_selected_inference_metrics(label, selected_metrics, inference)
        return {
            "checkpoint": str(checkpoint),
            "load_stats": load_stats,
            "retrieval_metrics": metrics,
            "best_retrieval_metrics": dict(best_metrics),
            "selected_inference": inference,
            "selected_retrieval_metrics": selected_metrics,
        }
    finally:
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def l2_normalize(array: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norm, eps)


def compute_identity_prototypes(features: np.ndarray, pids: np.ndarray) -> PrototypeTable:
    identity_ids = np.array(sorted(int(pid) for pid in np.unique(pids)), dtype=np.int64)
    prototypes: List[np.ndarray] = []
    counts: Dict[int, int] = {}
    for identity_id in identity_ids:
        mask = pids == identity_id
        counts[int(identity_id)] = int(np.sum(mask))
        proto = np.mean(features[mask], axis=0, dtype=np.float64)
        prototypes.append(proto.astype(np.float32))
    proto_array = l2_normalize(np.stack(prototypes, axis=0))
    id_to_index = {int(identity_id): index for index, identity_id in enumerate(identity_ids)}
    return PrototypeTable(identity_ids=identity_ids, prototypes=proto_array, counts=counts, id_to_index=id_to_index)


def ensure_bundle_alignment(baseline: EmbeddingBundle, iapr: EmbeddingBundle) -> None:
    if baseline.pids.shape != iapr.pids.shape or not np.array_equal(baseline.pids, iapr.pids):
        raise RuntimeError("Baseline and IAPR bundles do not have the same identity-label order.")
    if baseline.modalities.shape != iapr.modalities.shape or not np.array_equal(baseline.modalities, iapr.modalities):
        raise RuntimeError("Baseline and IAPR bundles do not have the same modality order.")
    if not np.array_equal(baseline.source_indices, iapr.source_indices):
        raise RuntimeError("Baseline and IAPR bundles do not have the same source-index order.")


def compute_margins_to_prototypes(
    features: np.ndarray,
    pids: np.ndarray,
    identity_ids: Sequence[int],
    prototypes: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    if len(identity_ids) != prototypes.shape[0]:
        raise ValueError("identity_ids and prototypes must have the same length.")
    if len(identity_ids) < 2:
        raise ValueError("At least two identities are required to compute hard-negative margins.")
    id_to_col = {int(identity_id): index for index, identity_id in enumerate(identity_ids)}
    own_cols = np.array([id_to_col[int(pid)] for pid in pids], dtype=np.int64)
    margins = np.empty(features.shape[0], dtype=np.float32)

    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        sims = features[start:end] @ prototypes.T
        local_own_cols = own_cols[start:end]
        rows = np.arange(end - start)
        own = sims[rows, local_own_cols]
        sims_for_neg = sims.copy()
        sims_for_neg[rows, local_own_cols] = -np.inf
        hard_neg = np.max(sims_for_neg, axis=1)
        margins[start:end] = own - hard_neg

    return margins


def identity_ambiguity_scores(
    bundle: EmbeddingBundle,
    table: PrototypeTable,
    valid_ids: Sequence[int],
    eps: float,
) -> Dict[int, Dict[str, float]]:
    valid_ids = [int(identity_id) for identity_id in valid_ids]
    table_indices = [table.id_to_index[identity_id] for identity_id in valid_ids]
    valid_prototypes = table.prototypes[table_indices]
    mask = np.isin(bundle.pids, np.array(valid_ids, dtype=np.int64))
    valid_features = bundle.features[mask]
    valid_pids = bundle.pids[mask]
    margins = compute_margins_to_prototypes(valid_features, valid_pids, valid_ids, valid_prototypes)

    scores: Dict[int, Dict[str, float]] = {}
    for identity_id in valid_ids:
        id_mask = valid_pids == identity_id
        id_margins = margins[id_mask]
        scores[identity_id] = {
            "ambiguous_rate": float(np.mean(id_margins < eps) * 100.0),
            "mean_margin": float(np.mean(id_margins)),
            "count": float(np.sum(id_mask)),
        }
    return scores


def build_confusion_neighbors(
    table: PrototypeTable,
    valid_ids: Sequence[int],
) -> Dict[int, List[int]]:
    valid_ids = [int(identity_id) for identity_id in valid_ids]
    table_indices = [table.id_to_index[identity_id] for identity_id in valid_ids]
    prototypes = table.prototypes[table_indices]
    sim = prototypes @ prototypes.T
    np.fill_diagonal(sim, -np.inf)
    neighbors: Dict[int, List[int]] = {}
    for row, identity_id in enumerate(valid_ids):
        order = np.argsort(-sim[row])
        neighbors[identity_id] = [valid_ids[int(index)] for index in order if np.isfinite(sim[row, int(index)])]
    return neighbors


def generate_candidate_groups(
    baseline_bundle: EmbeddingBundle,
    baseline_table: PrototypeTable,
    group_size: int,
    min_samples_per_id: int,
    num_candidate_groups: int,
    ambiguity_eps: float,
) -> Tuple[List[Tuple[int, ...]], Dict[int, Dict[str, float]]]:
    valid_ids = [
        int(identity_id)
        for identity_id in baseline_table.identity_ids
        if baseline_table.counts[int(identity_id)] >= min_samples_per_id
    ]
    if len(valid_ids) < group_size:
        raise RuntimeError(
            f"Only {len(valid_ids)} identities have at least {min_samples_per_id} embeddings; "
            f"cannot build groups of size {group_size}."
        )

    ambiguity = identity_ambiguity_scores(baseline_bundle, baseline_table, valid_ids, ambiguity_eps)
    neighbors = build_confusion_neighbors(baseline_table, valid_ids)
    seed_order = sorted(
        valid_ids,
        key=lambda identity_id: (
            -ambiguity[identity_id]["ambiguous_rate"],
            ambiguity[identity_id]["mean_margin"],
            -ambiguity[identity_id]["count"],
            identity_id,
        ),
    )

    groups: List[Tuple[int, ...]] = []
    seen = set()
    for seed in seed_order:
        group = [seed] + neighbors[seed][: group_size - 1]
        if len(group) != group_size:
            continue
        key = tuple(sorted(group))
        if key in seen:
            continue
        seen.add(key)
        groups.append(tuple(group))
        if len(groups) >= num_candidate_groups:
            break
    return groups, ambiguity


def compute_group_metrics(
    bundle: EmbeddingBundle,
    table: PrototypeTable,
    group_ids: Sequence[int],
    ambiguity_eps: float,
) -> Dict[str, float]:
    group_ids = [int(identity_id) for identity_id in group_ids]
    if len(group_ids) < 2:
        raise ValueError("Group metrics require at least two identities.")
    proto_indices = [table.id_to_index[identity_id] for identity_id in group_ids]
    prototypes = table.prototypes[proto_indices]
    mask = np.isin(bundle.pids, np.array(group_ids, dtype=np.int64))
    features = bundle.features[mask]
    pids = bundle.pids[mask]
    if features.size == 0:
        raise RuntimeError("Group has no samples.")

    id_to_col = {identity_id: index for index, identity_id in enumerate(group_ids)}
    own_cols = np.array([id_to_col[int(pid)] for pid in pids], dtype=np.int64)
    sims = features @ prototypes.T
    rows = np.arange(features.shape[0])
    own = sims[rows, own_cols]
    sims_for_neg = sims.copy()
    sims_for_neg[rows, own_cols] = -np.inf
    hard_neg = np.max(sims_for_neg, axis=1)
    margins = own - hard_neg

    compactness = float(np.mean(1.0 - own))
    proto_sim = prototypes @ prototypes.T
    distances = 1.0 - proto_sim
    np.fill_diagonal(distances, np.inf)
    separation = float(np.mean(np.min(distances, axis=1)))
    margin = float(np.mean(margins))
    ambiguity = float(np.mean(margins < ambiguity_eps) * 100.0)
    counts = np.array([np.sum(pids == identity_id) for identity_id in group_ids], dtype=np.float64)
    imbalance_penalty = float(np.std(counts) / max(float(np.mean(counts)), 1e-12))

    return {
        "compactness": compactness,
        "separation": separation,
        "margin": margin,
        "ambiguity": ambiguity,
        "imbalance_penalty": imbalance_penalty,
        "num_samples": float(features.shape[0]),
    }


def score_candidate_groups(
    candidate_groups: Sequence[Sequence[int]],
    baseline_bundle: EmbeddingBundle,
    iapr_bundle: EmbeddingBundle,
    baseline_table: PrototypeTable,
    iapr_table: PrototypeTable,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for group in tqdm(candidate_groups, desc="Scoring candidate groups"):
        group_ids = [int(identity_id) for identity_id in group]
        if any(identity_id not in iapr_table.id_to_index for identity_id in group_ids):
            continue
        baseline = compute_group_metrics(baseline_bundle, baseline_table, group_ids, args.ambiguity_eps)
        iapr = compute_group_metrics(iapr_bundle, iapr_table, group_ids, args.ambiguity_eps)
        delta_compactness = baseline["compactness"] - iapr["compactness"]
        delta_separation = iapr["separation"] - baseline["separation"]
        delta_margin = iapr["margin"] - baseline["margin"]
        delta_ambiguity = baseline["ambiguity"] - iapr["ambiguity"]
        imbalance_penalty = baseline["imbalance_penalty"]
        score = (
            args.w_margin * delta_margin
            + args.w_ambiguity * delta_ambiguity
            + args.w_separation * delta_separation
            + args.w_compactness * delta_compactness
            + args.w_baseline_ambiguity * baseline["ambiguity"]
            - args.w_imbalance * imbalance_penalty
        )
        scored.append(
            {
                "score": float(score),
                "identity_ids": "|".join(str(identity_id) for identity_id in group_ids),
                "group_ids": tuple(group_ids),
                "num_identities": int(len(group_ids)),
                "num_samples": int(baseline["num_samples"]),
                "C_baseline": baseline["compactness"],
                "C_iapr": iapr["compactness"],
                "delta_compactness": delta_compactness,
                "S_baseline": baseline["separation"],
                "S_iapr": iapr["separation"],
                "delta_separation": delta_separation,
                "M_baseline": baseline["margin"],
                "M_iapr": iapr["margin"],
                "delta_margin": delta_margin,
                "A_baseline": baseline["ambiguity"],
                "A_iapr": iapr["ambiguity"],
                "delta_ambiguity": delta_ambiguity,
                "imbalance_penalty": imbalance_penalty,
            }
        )

    scored.sort(key=lambda row: float(row["score"]), reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["group_rank"] = rank
    return scored


def reduce_embeddings_2d(
    features: np.ndarray,
    reducer: str,
    seed: int,
    pca_before_tsne: bool,
) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {features.shape}.")
    if features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if features.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)

    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for PCA/t-SNE visualization.") from exc

    data = np.asarray(features, dtype=np.float32)
    if reducer == "pca":
        n_components = min(2, data.shape[0], data.shape[1])
        coords = PCA(n_components=n_components, random_state=seed).fit_transform(data)
        if coords.shape[1] == 1:
            coords = np.concatenate([coords, np.zeros_like(coords)], axis=1)
        return coords.astype(np.float32)

    if reducer != "tsne":
        raise ValueError(f"Unsupported reducer: {reducer!r}")

    if pca_before_tsne and data.shape[1] > 50 and data.shape[0] > 2:
        n_components = min(50, data.shape[1], data.shape[0] - 1)
        data = PCA(n_components=n_components, random_state=seed).fit_transform(data).astype(np.float32)

    perplexity = min(30.0, max(1.0, (data.shape[0] - 1) / 3.0))
    tsne_kwargs: Dict[str, Any] = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": seed,
        "init": "pca" if data.shape[1] >= 2 and data.shape[0] > 2 else "random",
        "learning_rate": "auto",
    }
    tsne_params = inspect.signature(TSNE).parameters
    if "max_iter" in tsne_params:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    coords = TSNE(**tsne_kwargs).fit_transform(data)
    return coords.astype(np.float32)


def select_visual_indices(
    bundle: EmbeddingBundle,
    group_ids: Sequence[int],
    max_samples_per_id: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    for identity_id in group_ids:
        id_indices = np.flatnonzero(bundle.pids == int(identity_id))
        if len(id_indices) <= max_samples_per_id:
            selected.extend(int(index) for index in id_indices)
            continue

        image_indices = id_indices[bundle.modalities[id_indices] == "image"]
        text_indices = id_indices[bundle.modalities[id_indices] == "text"]
        image_take = min(len(image_indices), max_samples_per_id // 2)
        text_take = min(len(text_indices), max_samples_per_id - image_take)
        chosen: List[int] = []
        if image_take > 0:
            chosen.extend(int(index) for index in rng.choice(image_indices, size=image_take, replace=False))
        if text_take > 0:
            chosen.extend(int(index) for index in rng.choice(text_indices, size=text_take, replace=False))

        remaining_budget = max_samples_per_id - len(chosen)
        if remaining_budget > 0:
            remaining = np.array([index for index in id_indices if int(index) not in set(chosen)], dtype=np.int64)
            if len(remaining) > 0:
                fill_take = min(len(remaining), remaining_budget)
                chosen.extend(int(index) for index in rng.choice(remaining, size=fill_take, replace=False))
        selected.extend(sorted(chosen))
    return np.array(selected, dtype=np.int64)


def prepare_reduced_coordinates(
    bundle: EmbeddingBundle,
    table: PrototypeTable,
    group_ids: Sequence[int],
    sample_indices: np.ndarray,
    reducer: str,
    seed: int,
    pca_before_tsne: bool,
    plot_prototypes: bool,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    sample_features = bundle.features[sample_indices]
    proto_features = None
    all_features = sample_features
    if plot_prototypes:
        proto_indices = [table.id_to_index[int(identity_id)] for identity_id in group_ids]
        proto_features = table.prototypes[proto_indices]
        all_features = np.vstack([sample_features, proto_features])
    coords = reduce_embeddings_2d(all_features, reducer, seed, pca_before_tsne)
    sample_coords = coords[: sample_features.shape[0]]
    proto_coords = coords[sample_features.shape[0] :] if plot_prototypes else None
    return sample_coords, proto_coords


def color_palette(num_colors: int) -> List[Any]:
    import matplotlib.pyplot as plt

    if num_colors <= 20:
        cmap = plt.get_cmap("tab20")
        return [cmap(index) for index in range(num_colors)]
    cmap = plt.get_cmap("hsv")
    return [cmap(index / max(num_colors, 1)) for index in range(num_colors)]


def plot_group_comparison(
    baseline_bundle: EmbeddingBundle,
    iapr_bundle: EmbeddingBundle,
    baseline_table: PrototypeTable,
    iapr_table: PrototypeTable,
    group_row: Mapping[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Path, Dict[str, Any]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for visualization.") from exc

    group_ids = [int(identity_id) for identity_id in group_row["group_ids"]]
    ordered_ids = sorted(group_ids)
    sample_indices = select_visual_indices(
        baseline_bundle,
        ordered_ids,
        max_samples_per_id=int(args.max_samples_per_id),
        seed=int(args.seed) + int(group_row["group_rank"]),
    )
    anonymous_labels = {identity_id: f"ID-{index + 1:02d}" for index, identity_id in enumerate(ordered_ids)}

    baseline_xy, baseline_proto_xy = prepare_reduced_coordinates(
        baseline_bundle,
        baseline_table,
        ordered_ids,
        sample_indices,
        args.reducer,
        int(args.seed),
        bool(args.pca_before_tsne),
        bool(args.plot_prototypes),
    )
    iapr_xy, iapr_proto_xy = prepare_reduced_coordinates(
        iapr_bundle,
        iapr_table,
        ordered_ids,
        sample_indices,
        args.reducer,
        int(args.seed),
        bool(args.pca_before_tsne),
        bool(args.plot_prototypes),
    )

    colors = color_palette(len(ordered_ids))
    color_map = {identity_id: colors[index] for index, identity_id in enumerate(ordered_ids)}
    selected_pids = baseline_bundle.pids[sample_indices]
    selected_modalities = baseline_bundle.modalities[sample_indices]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5), sharex=False, sharey=False)
    subplot_specs = [
        (axes[0], args.baseline_name, baseline_xy, baseline_proto_xy, baseline_bundle),
        (axes[1], args.iapr_name, iapr_xy, iapr_proto_xy, iapr_bundle),
    ]
    marker_for_modality = {"image": "o", "text": "^"}

    for ax, title, coords, proto_coords, _bundle in subplot_specs:
        for identity_id in ordered_ids:
            for modality, marker in marker_for_modality.items():
                mask = (selected_pids == identity_id) & (selected_modalities == modality)
                if not np.any(mask):
                    continue
                ax.scatter(
                    coords[mask, 0],
                    coords[mask, 1],
                    s=30 if modality == "image" else 36,
                    marker=marker,
                    color=color_map[identity_id],
                    alpha=0.78,
                    edgecolors="black",
                    linewidths=0.25,
                )
        if args.plot_prototypes and proto_coords is not None:
            for index, identity_id in enumerate(ordered_ids):
                ax.scatter(
                    proto_coords[index, 0],
                    proto_coords[index, 1],
                    s=150,
                    marker="*",
                    color=color_map[identity_id],
                    edgecolors="black",
                    linewidths=0.6,
                    zorder=5,
                )
        ax.set_title(str(title), fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{args.reducer.upper()}-1")
        ax.set_ylabel(f"{args.reducer.upper()}-2")
        ax.grid(True, alpha=0.22, linewidth=0.6)
        ax.tick_params(labelsize=8)

    identity_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            label=anonymous_labels[identity_id],
            markerfacecolor=color_map[identity_id],
            markeredgecolor="black",
            markersize=7,
        )
        for identity_id in ordered_ids
    ]
    modality_handles = [
        Line2D([0], [0], marker="o", color="black", label="Image", linestyle="None", markersize=7),
        Line2D([0], [0], marker="^", color="black", label="Text", linestyle="None", markersize=7),
    ]
    if args.plot_prototypes:
        modality_handles.append(Line2D([0], [0], marker="*", color="black", label="Prototype", linestyle="None", markersize=10))

    fig.legend(
        handles=identity_handles + modality_handles,
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        f"{args.dataset_name} group {group_row['group_rank']} | score={float(group_row['score']):.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.95])
    output_path = output_dir / f"{args.dataset_name}_identity_vis_group{int(group_row['group_rank'])}_{args.reducer}.png"
    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)

    coord_payload = {
        "sample_indices": sample_indices,
        "baseline_xy": baseline_xy,
        "iapr_xy": iapr_xy,
        "baseline_proto_xy": baseline_proto_xy,
        "iapr_proto_xy": iapr_proto_xy,
        "ordered_ids": np.array(ordered_ids, dtype=np.int64),
        "anonymous_labels": np.array([anonymous_labels[identity_id] for identity_id in ordered_ids]),
    }
    return output_path, coord_payload


def save_group_npz(
    path: Path,
    baseline_bundle: EmbeddingBundle,
    iapr_bundle: EmbeddingBundle,
    coord_payload: Mapping[str, Any],
    group_row: Mapping[str, Any],
) -> None:
    sample_indices = np.asarray(coord_payload["sample_indices"], dtype=np.int64)
    baseline_proto_xy = coord_payload["baseline_proto_xy"]
    iapr_proto_xy = coord_payload["iapr_proto_xy"]
    np.savez_compressed(
        path,
        identity_ids=np.asarray(coord_payload["ordered_ids"], dtype=np.int64),
        anonymous_labels=np.asarray(coord_payload["anonymous_labels"]),
        selected_indices=sample_indices,
        labels=baseline_bundle.pids[sample_indices],
        modalities=baseline_bundle.modalities[sample_indices],
        source_indices=baseline_bundle.source_indices[sample_indices],
        baseline_embeddings=baseline_bundle.features[sample_indices],
        iapr_embeddings=iapr_bundle.features[sample_indices],
        baseline_xy=np.asarray(coord_payload["baseline_xy"], dtype=np.float32),
        iapr_xy=np.asarray(coord_payload["iapr_xy"], dtype=np.float32),
        baseline_prototype_xy=np.asarray(baseline_proto_xy if baseline_proto_xy is not None else np.zeros((0, 2)), dtype=np.float32),
        iapr_prototype_xy=np.asarray(iapr_proto_xy if iapr_proto_xy is not None else np.zeros((0, 2)), dtype=np.float32),
        metrics_json=np.array(json.dumps({key: value for key, value in group_row.items() if key != "group_ids"}, default=str)),
    )


def save_csv(rows: Sequence[Mapping[str, Any]], path: Path, columns: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=json_default)
        file.write("\n")


def evaluation_summary_row(
    label: str,
    config_path: Optional[Path],
    config_source: str,
    metadata: Mapping[str, Any],
) -> Dict[str, Any]:
    best = metadata.get("best_retrieval_metrics", {})
    selected = metadata.get("selected_retrieval_metrics", {})
    inference = metadata.get("selected_inference", {})
    return {
        "label": label,
        "checkpoint": metadata.get("checkpoint", ""),
        "config": str(config_path) if config_path is not None else "",
        "config_source": config_source,
        "best_task": best.get("task", ""),
        "best_R1": best.get("R1", ""),
        "best_R5": best.get("R5", ""),
        "best_R10": best.get("R10", ""),
        "best_mAP": best.get("mAP", ""),
        "best_mINP": best.get("mINP", ""),
        "best_rSum": best.get("rSum", ""),
        "selected_task": inference.get("ablation_task", ""),
        "selected_global_weight": inference.get("global_weight", ""),
        "selected_grab_weight": inference.get("grab_weight", ""),
        "selected_R1": selected.get("R1", ""),
        "selected_R5": selected.get("R5", ""),
        "selected_R10": selected.get("R10", ""),
        "selected_mAP": selected.get("mAP", ""),
        "selected_mINP": selected.get("mINP", ""),
        "selected_rSum": selected.get("rSum", ""),
        "loaded_tensors": metadata.get("load_stats", {}).get("loaded", ""),
        "skipped_missing": metadata.get("load_stats", {}).get("skipped_missing", ""),
        "skipped_shape": metadata.get("load_stats", {}).get("skipped_shape", ""),
        "skipped_non_tensor": metadata.get("load_stats", {}).get("skipped_non_tensor", ""),
    }


def save_test_evaluation_outputs(
    output_dir: Path,
    dataset_name: str,
    baseline_meta: Mapping[str, Any],
    iapr_meta: Mapping[str, Any],
    baseline_config: Optional[Path],
    iapr_config: Optional[Path],
    baseline_config_source: str,
    iapr_config_source: str,
) -> Tuple[Path, Path]:
    rows = [
        evaluation_summary_row("Baseline", baseline_config, baseline_config_source, baseline_meta),
        evaluation_summary_row("IAPR", iapr_config, iapr_config_source, iapr_meta),
    ]
    csv_path = output_dir / f"{dataset_name}_identity_vis_test_eval_summary.csv"
    json_path = output_dir / f"{dataset_name}_identity_vis_test_eval_summary.json"
    save_csv(rows, csv_path, list(rows[0].keys()))
    save_json(
        {
            "dataset_name": dataset_name,
            "split": "test",
            "baseline": baseline_meta,
            "iapr": iapr_meta,
            "summary_rows": rows,
        },
        json_path,
    )
    return csv_path, json_path


def print_load_stats(label: str, metadata: Mapping[str, Any]) -> None:
    stats = metadata.get("load_stats", {})
    print(
        f"[{label}] Loaded checkpoint tensors: "
        f"{stats.get('loaded', 0)} loaded, "
        f"{stats.get('skipped_missing', 0)} missing/extra keys, "
        f"{stats.get('skipped_shape', 0)} shape-mismatch, "
        f"{stats.get('skipped_non_tensor', 0)} non-tensor entries skipped."
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    if args.split != "test":
        raise ValueError("Identity embedding visualization is intentionally test-only; use --split test.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.group_size < 2:
        raise ValueError("--group_size must be at least 2.")
    if args.min_samples_per_id <= 0:
        raise ValueError("--min_samples_per_id must be positive.")
    if args.max_samples_per_id <= 0:
        raise ValueError("--max_samples_per_id must be positive.")
    if args.num_candidate_groups <= 0:
        raise ValueError("--num_candidate_groups must be positive.")
    if args.top_groups <= 0:
        raise ValueError("--top_groups must be positive.")
    if not math.isfinite(float(args.ambiguity_eps)):
        raise ValueError("--ambiguity_eps must be finite.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    for weight_name in (
        "w_margin",
        "w_ambiguity",
        "w_separation",
        "w_compactness",
        "w_baseline_ambiguity",
        "w_imbalance",
    ):
        if not math.isfinite(float(getattr(args, weight_name))):
            raise ValueError(f"--{weight_name} must be finite.")


def shared_dataset_root(
    args: argparse.Namespace,
    baseline_args: SimpleNamespace,
    iapr_args: SimpleNamespace,
    baseline_config_source: str,
    iapr_config_source: str,
) -> Path:
    if args.dataset_root:
        root = resolve_path(args.dataset_root)
    else:
        baseline_root_value = getattr(baseline_args, "root_dir", None)
        iapr_root_value = getattr(iapr_args, "root_dir", None)
        if baseline_config_source != "defaults_cli" and baseline_root_value:
            root = resolve_path(baseline_root_value)
        elif iapr_config_source != "defaults_cli" and iapr_root_value:
            root = resolve_path(iapr_root_value)
        else:
            root = resolve_path(baseline_root_value or iapr_root_value)
    baseline_root = resolve_path(getattr(baseline_args, "root_dir", root))
    iapr_root = resolve_path(getattr(iapr_args, "root_dir", root))
    if baseline_root != iapr_root:
        print(f"[Config] Warning: baseline root_dir={baseline_root} differs from IAPR root_dir={iapr_root}; using {root}.")
    baseline_args.root_dir = str(root)
    iapr_args.root_dir = str(root)
    return root


def harmonize_eval_settings(baseline_args: SimpleNamespace, iapr_args: SimpleNamespace) -> None:
    shared_keys = ("dataset_name", "root_dir", "img_size", "text_length", "batch_size", "test_batch_size", "num_workers")
    for key in shared_keys:
        baseline_value = getattr(baseline_args, key, None)
        iapr_value = getattr(iapr_args, key, None)
        if baseline_value != iapr_value:
            print(
                f"[Config] Warning: baseline {key}={baseline_value!r} differs from "
                f"IAPR {key}={iapr_value!r}; using baseline value for both."
            )
        setattr(iapr_args, key, baseline_value)

    baseline_model_type = getattr(baseline_args, "model_type", "itself")
    iapr_model_type = getattr(iapr_args, "model_type", "itself")
    if baseline_model_type != iapr_model_type:
        raise ValueError(
            "Baseline and IAPR must use the same model_type/inference path for a fair comparison. "
            f"Got baseline={baseline_model_type!r}, IAPR={iapr_model_type!r}."
        )
    baseline_args.only_global = baseline_model_type == "clip"
    iapr_args.only_global = iapr_model_type == "clip"


def print_selected_group(row: Mapping[str, Any]) -> None:
    print(
        f"[Selected group {row['group_rank']}] score={float(row['score']):.6f}, "
        f"ids={row['identity_ids']}, samples={row['num_samples']}, "
        f"delta_margin={float(row['delta_margin']):.6f}, "
        f"delta_ambiguity={float(row['delta_ambiguity']):.2f}, "
        f"delta_separation={float(row['delta_separation']):.6f}, "
        f"delta_compactness={float(row['delta_compactness']):.6f}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(int(args.seed))

    baseline_checkpoint = resolve_path(args.baseline_checkpoint)
    iapr_checkpoint = resolve_path(args.iapr_checkpoint)
    if not baseline_checkpoint.is_file():
        raise FileNotFoundError(f"Baseline checkpoint file not found: {baseline_checkpoint}")
    if not iapr_checkpoint.is_file():
        raise FileNotFoundError(f"IAPR checkpoint file not found: {iapr_checkpoint}")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_cfg, baseline_config, baseline_config_source = resolve_config_for_checkpoint(
        "Baseline",
        args.baseline_config,
        baseline_checkpoint,
    )
    iapr_cfg, iapr_config, iapr_config_source = resolve_config_for_checkpoint(
        "IAPR",
        args.iapr_config,
        iapr_checkpoint,
    )

    baseline_model_type = args.baseline_model_type or args.model_type
    iapr_model_type = args.iapr_model_type or args.model_type
    baseline_args = build_model_args(args, baseline_cfg, baseline_model_type)
    iapr_args = build_model_args(args, iapr_cfg, iapr_model_type)
    dataset_root = shared_dataset_root(
        args,
        baseline_args,
        iapr_args,
        baseline_config_source,
        iapr_config_source,
    )
    harmonize_eval_settings(baseline_args, iapr_args)
    device = resolve_device(args.device)

    print(
        f"[Config] Baseline config: "
        f"{baseline_config if baseline_config is not None else 'defaults + CLI'} "
        f"(source={baseline_config_source})"
    )
    print(
        f"[Config] IAPR config: "
        f"{iapr_config if iapr_config is not None else 'defaults + CLI'} "
        f"(source={iapr_config_source})"
    )
    print(f"[Dataset] Loading {args.dataset_name} split=test from {dataset_root}")
    print(f"[Checkpoint] Baseline: {baseline_checkpoint}")
    print(f"[Checkpoint] IAPR: {iapr_checkpoint}")

    split_data = load_split_data(args.dataset_name, dataset_root, "test")
    print(
        f"[Verification] Running mandatory standard retrieval evaluation on test split: "
        f"queries={len(split_data.captions)} gallery={len(split_data.img_paths)}"
    )
    baseline_eval_meta = evaluate_checkpoint_on_test_split(
        "Baseline",
        baseline_checkpoint,
        baseline_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    iapr_eval_meta = evaluate_checkpoint_on_test_split(
        "IAPR",
        iapr_checkpoint,
        iapr_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    baseline_args.selected_inference = baseline_eval_meta.get("selected_inference")
    iapr_args.selected_inference = iapr_eval_meta.get("selected_inference")
    eval_csv_path, eval_json_path = save_test_evaluation_outputs(
        output_dir,
        args.dataset_name,
        baseline_eval_meta,
        iapr_eval_meta,
        baseline_config,
        iapr_config,
        baseline_config_source,
        iapr_config_source,
    )
    print(f"[Output] saved test evaluation CSV: {eval_csv_path}")
    print(f"[Output] saved test evaluation JSON: {eval_json_path}")

    validate_split_data(split_data, "test")
    print(
        f"[Dataset] {args.dataset_name} split=test "
        f"images={len(split_data.img_paths)} captions={len(split_data.captions)} "
        f"train_ids={split_data.num_train_ids}"
    )

    print("[Baseline] Extracting normalized image/text embeddings.")
    baseline_bundle = load_embeddings_for_checkpoint(
        "Baseline",
        baseline_checkpoint,
        baseline_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    print("[IAPR] Extracting normalized image/text embeddings.")
    iapr_bundle = load_embeddings_for_checkpoint(
        "IAPR",
        iapr_checkpoint,
        iapr_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    ensure_bundle_alignment(baseline_bundle, iapr_bundle)

    baseline_table = compute_identity_prototypes(baseline_bundle.features, baseline_bundle.pids)
    iapr_table = compute_identity_prototypes(iapr_bundle.features, iapr_bundle.pids)
    print(
        f"[Embeddings] baseline shape={baseline_bundle.features.shape}, "
        f"IAPR shape={iapr_bundle.features.shape}, identities={len(baseline_table.identity_ids)}"
    )
    print(f"[Inference] Baseline: {baseline_bundle.inference}")
    print(f"[Inference] IAPR: {iapr_bundle.inference}")

    candidate_groups, ambiguity = generate_candidate_groups(
        baseline_bundle,
        baseline_table,
        group_size=int(args.group_size),
        min_samples_per_id=int(args.min_samples_per_id),
        num_candidate_groups=int(args.num_candidate_groups),
        ambiguity_eps=float(args.ambiguity_eps),
    )
    if not candidate_groups:
        raise RuntimeError("No candidate groups were generated from the baseline confusion graph.")
    avg_seed_ambiguity = float(np.mean([value["ambiguous_rate"] for value in ambiguity.values()]))
    print(
        f"[Candidates] generated={len(candidate_groups)} "
        f"from valid_identities={len(ambiguity)} average_baseline_ambiguity={avg_seed_ambiguity:.2f}%"
    )

    scored_groups = score_candidate_groups(
        candidate_groups,
        baseline_bundle,
        iapr_bundle,
        baseline_table,
        iapr_table,
        args,
    )
    if not scored_groups:
        raise RuntimeError("No candidate groups could be scored.")
    top_groups = scored_groups[: min(int(args.top_groups), len(scored_groups))]

    csv_path = output_dir / f"{args.dataset_name}_identity_vis_group_scores.csv"
    save_csv(scored_groups, csv_path, GROUP_SCORE_COLUMNS)
    print(f"[Output] saved group score CSV: {csv_path}")

    for row in top_groups:
        print_selected_group(row)
        figure_path, coord_payload = plot_group_comparison(
            baseline_bundle,
            iapr_bundle,
            baseline_table,
            iapr_table,
            row,
            args,
            output_dir,
        )
        print(f"[Output] saved figure: {figure_path}")
        if args.save_npz:
            npz_path = output_dir / f"{args.dataset_name}_identity_vis_group{int(row['group_rank'])}_{args.reducer}.npz"
            save_group_npz(npz_path, baseline_bundle, iapr_bundle, coord_payload, row)
            print(f"[Output] saved NPZ: {npz_path}")


if __name__ == "__main__":
    main()
