#!/usr/bin/env python3
"""Visualize local embedding repair around ambiguous text queries.

This utility compares the final retrieval embeddings used by the standard
text-image retrieval path for a host checkpoint and a Host+IAPR checkpoint.
It does not use prototype memories, projection heads, prototype-space vectors,
or prototype logits.

Example:

python tools/plot_local_embedding_repair.py \
  --host_ckpt /path/to/host/best.pth \
  --iapr_ckpt /path/to/iapr/best.pth \
  --data_root /path/to/dataset/RSTPReid \
  --dataset_name RSTPReid \
  --split test \
  --output_dir outputs/local_repair_rstp \
  --top_pos 5 \
  --top_neg 10 \
  --only_ambiguous true \
  --ambiguous_eps 0.01 \
  --only_improved true \
  --min_margin_gain 0.0
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_ambiguity_rate_compare import (  # noqa: E402
    build_model_args,
    build_repo_model,
    extract_image_features,
    extract_text_features,
    load_checkpoint_for_inference,
    load_split_data,
    parse_img_size,
    resolve_device,
    resolve_path,
    validate_split_data,
)


SUMMARY_COLUMNS = [
    "query_index",
    "query_pid",
    "query_text",
    "host_pos_score",
    "host_neg_score",
    "host_margin",
    "iapr_pos_score",
    "iapr_neg_score",
    "iapr_margin",
    "margin_gain",
    "selected_positive_indices",
    "selected_hard_negative_indices",
    "output_png",
    "output_pdf",
]


def str2bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot shared-PCA local retrieval neighborhoods before/after Host+IAPR."
    )
    parser.add_argument("--host_ckpt", required=True, help="Path to the host/backbone checkpoint.")
    parser.add_argument("--iapr_ckpt", required=True, help="Path to the Host+IAPR checkpoint.")
    parser.add_argument("--data_root", required=True, help="Dataset folder or parent folder containing the dataset.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=["CUHK-PEDES", "ICFG-PEDES", "RSTPReid", "PAB"],
        help="Dataset name.",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split.")
    parser.add_argument("--output_dir", required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=256, help="Embedding extraction batch size.")
    parser.add_argument("--device", default="cuda", help='Device, e.g. "cuda" or "cpu".')
    parser.add_argument("--top_pos", type=int, default=5, help="Number of positive gallery images to show.")
    parser.add_argument("--top_neg", type=int, default=10, help="Number of hard negatives per checkpoint to union.")
    parser.add_argument("--max_queries", type=int, default=-1, help="Maximum plotted queries; <=0 plots all after filtering.")
    parser.add_argument("--only_ambiguous", type=str2bool, default=False, help="Only plot queries with host_margin < ambiguous_eps.")
    parser.add_argument("--ambiguous_eps", type=float, default=0.01, help="Ambiguous-query threshold on host margin.")
    parser.add_argument("--only_improved", type=str2bool, default=False, help="Only plot queries with margin_gain > min_margin_gain.")
    parser.add_argument("--min_margin_gain", type=float, default=0.0, help="Minimum margin gain for --only_improved.")
    parser.add_argument(
        "--sort_by",
        default="margin_gain",
        choices=[
            "margin_gain",
            "host_margin",
            "iapr_margin",
            "host_neg_score",
            "iapr_neg_score",
            "query_index",
        ],
        help="Criterion used before applying --max_queries. Default selects largest margin repair first.",
    )
    parser.add_argument("--sort_desc", type=str2bool, default=True, help="Sort descending before applying --max_queries.")
    parser.add_argument(
        "--no_sort",
        type=str2bool,
        default=False,
        help="Keep dataset order instead of sorting before plotting.",
    )
    parser.add_argument("--save_pdf", type=str2bool, default=True, help="Write per-query PDF figures.")
    parser.add_argument("--save_png", type=str2bool, default=True, help="Write per-query PNG figures.")
    parser.add_argument("--cache_embeddings", type=str2bool, default=True, help="Save extracted embeddings under output_dir/cache.")
    parser.add_argument("--projection", default="pca", choices=["pca"], help="2D projection method.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--img_size", default="384,128", help='Input image size as "height,width".')
    parser.add_argument("--text_length", type=int, default=77, help="Tokenized text length.")
    parser.add_argument("--pretrain_choice", default="ViT-B/16", help="CLIP backbone choice used by build_model(...).")
    parser.add_argument("--host_name", default="Host", help="Left-panel title.")
    parser.add_argument("--iapr_name", default="Host + IAPR", help="Right-panel title.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output DPI.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative.")
    if args.top_pos < 0:
        raise ValueError("--top_pos must be non-negative.")
    if args.top_neg < 0:
        raise ValueError("--top_neg must be non-negative.")
    if args.text_length <= 0:
        raise ValueError("--text_length must be positive.")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive.")
    if not args.save_png and not args.save_pdf:
        print("Warning: both --save_png and --save_pdf are false; only CSV/cache files will be written.")


def compatibility_model_args(args: argparse.Namespace) -> SimpleNamespace:
    compat = SimpleNamespace(
        dataset_name=args.dataset_name,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        text_length=args.text_length,
        pretrain_choice=args.pretrain_choice,
    )
    return build_model_args(compat)


@torch.no_grad()
def extract_embeddings_for_checkpoint(
    checkpoint_path: Path,
    model_args: SimpleNamespace,
    split_data: Any,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    model: Optional[torch.nn.Module] = None
    try:
        run_args = SimpleNamespace(**vars(model_args))
        num_classes = max(int(split_data.num_train_ids), 1)
        model = build_repo_model(run_args, num_classes=num_classes)
        load_stats = load_checkpoint_for_inference(model, checkpoint_path)
        model.to(device)
        if device.type == "cpu":
            model.float()
        model.eval()

        text_features, query_pids = extract_text_features(
            model,
            split_data,
            text_length=int(run_args.text_length),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        image_features, gallery_pids = extract_image_features(
            model,
            split_data,
            img_size=parse_img_size(run_args.img_size),
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )

        text_features = F.normalize(text_features.float(), p=2, dim=1).cpu()
        image_features = F.normalize(image_features.float(), p=2, dim=1).cpu()
        return text_features, image_features, query_pids.cpu().long(), gallery_pids.cpu().long(), load_stats
    finally:
        model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def save_embedding_cache(
    cache_dir: Path,
    host_text: torch.Tensor,
    host_image: torch.Tensor,
    iapr_text: torch.Tensor,
    iapr_image: torch.Tensor,
    query_pids: torch.Tensor,
    gallery_pids: torch.Tensor,
    query_texts: Optional[Sequence[str]],
    gallery_paths: Optional[Sequence[str]],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "host_text.npy", host_text.numpy())
    np.save(cache_dir / "host_image.npy", host_image.numpy())
    np.save(cache_dir / "iapr_text.npy", iapr_text.numpy())
    np.save(cache_dir / "iapr_image.npy", iapr_image.numpy())
    np.save(cache_dir / "query_pids.npy", query_pids.numpy())
    np.save(cache_dir / "gallery_pids.npy", gallery_pids.numpy())
    if query_texts is not None:
        (cache_dir / "query_texts.json").write_text(json.dumps(list(query_texts), ensure_ascii=False, indent=2), encoding="utf-8")
    if gallery_paths is not None:
        (cache_dir / "gallery_paths.json").write_text(json.dumps(list(gallery_paths), ensure_ascii=False, indent=2), encoding="utf-8")


def topk_from_mask(scores: torch.Tensor, mask: torch.Tensor, k: int) -> List[int]:
    indices = torch.nonzero(mask, as_tuple=False).view(-1)
    if indices.numel() == 0 or k <= 0:
        return []
    k = min(int(k), int(indices.numel()))
    local_scores = scores[indices]
    order = torch.argsort(local_scores, descending=True)[:k]
    return [int(indices[i].item()) for i in order]


def score_extrema(scores: torch.Tensor, pos_mask: torch.Tensor, neg_mask: torch.Tensor) -> Tuple[float, float, float]:
    pos_score = scores.masked_fill(~pos_mask, float("-inf")).max()
    neg_score = scores.masked_fill(~neg_mask, float("-inf")).max()
    margin = pos_score - neg_score
    return float(pos_score.item()), float(neg_score.item()), float(margin.item())


def unique_preserve_order(indices: Iterable[int]) -> List[int]:
    seen = set()
    result = []
    for index in indices:
        index = int(index)
        if index in seen:
            continue
        seen.add(index)
        result.append(index)
    return result


def fit_project_2d(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}.")
    if features.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if features.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float64)

    try:
        from sklearn.decomposition import PCA

        projected = PCA(n_components=2).fit_transform(features)
    except Exception:
        centered = features - features.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:2].T
        projected = centered @ components

    if projected.shape[1] == 1:
        projected = np.concatenate([projected, np.zeros((projected.shape[0], 1), dtype=projected.dtype)], axis=1)
    return np.asarray(projected[:, :2], dtype=np.float64)


def pad_limits(ax: Any, coords: np.ndarray) -> None:
    if coords.size == 0:
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        return
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    x_pad = max((x_max - x_min) * 0.12, 1e-3)
    y_pad = max((y_max - y_min) * 0.12, 1e-3)
    ax.set_xlim(float(x_min - x_pad), float(x_max + x_pad))
    ax.set_ylim(float(y_min - y_pad), float(y_max + y_pad))


def draw_panel(
    ax: Any,
    title: str,
    coords: np.ndarray,
    selected_indices: Sequence[int],
    selected_positive: Sequence[int],
    selected_negative: Sequence[int],
    margin: float,
) -> None:
    text_xy = coords[0]
    image_xy = coords[1:]
    pos_set = {int(index) for index in selected_positive}
    neg_set = {int(index) for index in selected_negative}
    pos_xy = np.array([image_xy[i] for i, index in enumerate(selected_indices) if int(index) in pos_set])
    neg_xy = np.array([image_xy[i] for i, index in enumerate(selected_indices) if int(index) in neg_set])

    if len(pos_xy):
        ax.scatter(pos_xy[:, 0], pos_xy[:, 1], marker="o", s=46, alpha=0.88, label="positive")
    if len(neg_xy):
        ax.scatter(neg_xy[:, 0], neg_xy[:, 1], marker="x", s=54, linewidths=1.5, alpha=0.88, label="hard negative")
    ax.scatter([text_xy[0]], [text_xy[1]], marker="*", s=150, edgecolors="black", linewidths=0.6, label="text query")
    ax.set_title(f"{title}\nmargin={margin:.4f}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_query_neighborhood(
    output_base: Path,
    query_index: int,
    query_pid: int,
    selected_indices: Sequence[int],
    selected_positive: Sequence[int],
    selected_negative: Sequence[int],
    host_text: torch.Tensor,
    host_images: torch.Tensor,
    iapr_text: torch.Tensor,
    iapr_images: torch.Tensor,
    host_margin: float,
    iapr_margin: float,
    host_name: str,
    iapr_name: str,
    save_png: bool,
    save_pdf: bool,
    dpi: int,
) -> Tuple[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    local_features = torch.cat([
        host_text.view(1, -1),
        host_images,
        iapr_text.view(1, -1),
        iapr_images,
    ], dim=0).numpy()
    projected = fit_project_2d(local_features)
    panel_n = 1 + len(selected_indices)
    host_coords = projected[:panel_n]
    iapr_coords = projected[panel_n:]
    all_coords = np.vstack([host_coords, iapr_coords])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), constrained_layout=True)
    draw_panel(axes[0], host_name, host_coords, selected_indices, selected_positive, selected_negative, host_margin)
    draw_panel(axes[1], iapr_name, iapr_coords, selected_indices, selected_positive, selected_negative, iapr_margin)
    for ax in axes:
        pad_limits(ax, all_coords)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))

    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    written_png = ""
    written_pdf = ""
    if save_png:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        written_png = str(png_path)
    if save_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
        written_pdf = str(pdf_path)
    plt.close(fig)
    return written_png, written_pdf


def row_filename(query_index: int, pid: int, host_margin: float, iapr_margin: float) -> str:
    return f"query_{query_index:05d}_pid_{pid}_hostm_{host_margin:.4f}_iaprm_{iapr_margin:.4f}"


def join_indices(indices: Sequence[int]) -> str:
    return ";".join(str(int(index)) for index in indices)


def write_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_top_margin_gains(rows: Sequence[Mapping[str, Any]], limit: int = 10) -> None:
    sorted_rows = sorted(rows, key=lambda row: float(row["margin_gain"]), reverse=True)
    print("\nTop queries by margin_gain:")
    if not sorted_rows:
        print("  no plotted queries")
        return
    for row in sorted_rows[:limit]:
        print(
            f"  q={int(row['query_index']):05d} pid={row['query_pid']} "
            f"gain={float(row['margin_gain']):.4f} "
            f"host_m={float(row['host_margin']):.4f} iapr_m={float(row['iapr_margin']):.4f}"
        )


def selected_query_passes_filters(args: argparse.Namespace, host_margin: float, margin_gain: float) -> bool:
    if args.only_ambiguous and not (host_margin < float(args.ambiguous_eps)):
        return False
    if args.only_improved and not (margin_gain > float(args.min_margin_gain)):
        return False
    return True


def sorted_candidates(candidates: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> List[Mapping[str, Any]]:
    if args.no_sort:
        return list(candidates)
    sort_key = args.sort_by

    def value(row: Mapping[str, Any]) -> Tuple[float, int]:
        primary = float(row[sort_key])
        return primary, -int(row["query_index"])

    return sorted(candidates, key=value, reverse=bool(args.sort_desc))


def main() -> None:
    args = parse_args()
    validate_args(args)

    host_ckpt = resolve_path(args.host_ckpt)
    iapr_ckpt = resolve_path(args.iapr_ckpt)
    data_root = resolve_path(args.data_root)
    output_dir = resolve_path(args.output_dir)
    figures_dir = output_dir / "figures"
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not host_ckpt.is_file():
        raise FileNotFoundError(f"Host checkpoint not found: {host_ckpt}")
    if not iapr_ckpt.is_file():
        raise FileNotFoundError(f"Host+IAPR checkpoint not found: {iapr_ckpt}")

    device = resolve_device(args.device)
    model_args = compatibility_model_args(args)
    split_data = load_split_data(args.dataset_name, data_root, args.split)
    validate_split_data(split_data, args.split)
    query_texts = list(getattr(split_data, "captions", [])) if hasattr(split_data, "captions") else []
    gallery_paths = list(getattr(split_data, "img_paths", [])) if hasattr(split_data, "img_paths") else []

    print(
        f"[Dataset] {args.dataset_name} split={args.split} "
        f"queries={len(split_data.captions)} gallery={len(split_data.img_paths)}"
    )
    print(f"[Host] Extracting retrieval embeddings from {host_ckpt}")
    host_text, host_image, query_pids, gallery_pids, host_stats = extract_embeddings_for_checkpoint(
        host_ckpt,
        model_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    print(
        f"[Host] loaded={host_stats.get('loaded', 0)} "
        f"missing={host_stats.get('skipped_missing', 0)} shape={host_stats.get('skipped_shape', 0)}"
    )

    print(f"[Host+IAPR] Extracting retrieval embeddings from {iapr_ckpt}")
    iapr_text, iapr_image, query_pids_iapr, gallery_pids_iapr, iapr_stats = extract_embeddings_for_checkpoint(
        iapr_ckpt,
        model_args,
        split_data,
        device,
        args.batch_size,
        args.num_workers,
    )
    print(
        f"[Host+IAPR] loaded={iapr_stats.get('loaded', 0)} "
        f"missing={iapr_stats.get('skipped_missing', 0)} shape={iapr_stats.get('skipped_shape', 0)}"
    )

    if not torch.equal(query_pids, query_pids_iapr):
        raise RuntimeError("Host and Host+IAPR query pid arrays differ.")
    if not torch.equal(gallery_pids, gallery_pids_iapr):
        raise RuntimeError("Host and Host+IAPR gallery pid arrays differ.")

    if args.cache_embeddings:
        save_embedding_cache(
            cache_dir,
            host_text,
            host_image,
            iapr_text,
            iapr_image,
            query_pids,
            gallery_pids,
            query_texts,
            gallery_paths,
        )
        print(f"[Cache] wrote embeddings to {cache_dir}")

    sim_host = host_text @ host_image.t()
    sim_iapr = iapr_text @ iapr_image.t()

    candidates: List[Dict[str, Any]] = []
    skipped = 0
    for query_index in tqdm(range(sim_host.shape[0]), desc="Scoring query candidates"):
        query_pid = int(query_pids[query_index].item())
        pos_mask = gallery_pids.eq(query_pid)
        neg_mask = ~pos_mask
        if not bool(pos_mask.any()) or not bool(neg_mask.any()):
            skipped += 1
            continue

        host_pos_score, host_neg_score, host_margin = score_extrema(sim_host[query_index], pos_mask, neg_mask)
        iapr_pos_score, iapr_neg_score, iapr_margin = score_extrema(sim_iapr[query_index], pos_mask, neg_mask)
        margin_gain = iapr_margin - host_margin
        if not selected_query_passes_filters(args, host_margin, margin_gain):
            continue

        combined_pos_scores = torch.maximum(sim_host[query_index], sim_iapr[query_index])
        selected_positive = topk_from_mask(combined_pos_scores, pos_mask, args.top_pos)
        host_negatives = topk_from_mask(sim_host[query_index], neg_mask, args.top_neg)
        iapr_negatives = topk_from_mask(sim_iapr[query_index], neg_mask, args.top_neg)
        selected_negative = unique_preserve_order(host_negatives + iapr_negatives)
        selected_gallery = unique_preserve_order(selected_positive + selected_negative)
        if not selected_gallery:
            skipped += 1
            continue

        candidates.append(
            {
                "query_index": int(query_index),
                "query_pid": query_pid,
                "host_pos_score": host_pos_score,
                "host_neg_score": host_neg_score,
                "host_margin": host_margin,
                "iapr_pos_score": iapr_pos_score,
                "iapr_neg_score": iapr_neg_score,
                "iapr_margin": iapr_margin,
                "margin_gain": margin_gain,
                "selected_positive": selected_positive,
                "selected_negative": selected_negative,
                "selected_gallery": selected_gallery,
            }
        )

    ordered_candidates = sorted_candidates(candidates, args)
    if args.max_queries > 0:
        ordered_candidates = ordered_candidates[:int(args.max_queries)]
    print(
        f"[Selection] candidates={len(candidates)} plotted={len(ordered_candidates)} "
        f"sort={'dataset_order' if args.no_sort else args.sort_by} desc={args.sort_desc}"
    )

    rows: List[Dict[str, Any]] = []
    for candidate in tqdm(ordered_candidates, desc="Plotting selected neighborhoods"):
        query_index = int(candidate["query_index"])
        query_pid = int(candidate["query_pid"])
        host_pos_score = float(candidate["host_pos_score"])
        host_neg_score = float(candidate["host_neg_score"])
        host_margin = float(candidate["host_margin"])
        iapr_pos_score = float(candidate["iapr_pos_score"])
        iapr_neg_score = float(candidate["iapr_neg_score"])
        iapr_margin = float(candidate["iapr_margin"])
        margin_gain = float(candidate["margin_gain"])
        selected_positive = list(candidate["selected_positive"])
        selected_negative = list(candidate["selected_negative"])
        selected_gallery = list(candidate["selected_gallery"])

        base_name = row_filename(query_index, query_pid, host_margin, iapr_margin)
        output_base = figures_dir / base_name
        png_path, pdf_path = plot_query_neighborhood(
            output_base,
            query_index,
            query_pid,
            selected_gallery,
            selected_positive,
            selected_negative,
            host_text[query_index],
            host_image[selected_gallery],
            iapr_text[query_index],
            iapr_image[selected_gallery],
            host_margin,
            iapr_margin,
            args.host_name,
            args.iapr_name,
            args.save_png,
            args.save_pdf,
            args.dpi,
        )

        rows.append(
            {
                "query_index": int(query_index),
                "query_pid": query_pid,
                "query_text": query_texts[query_index] if query_index < len(query_texts) else "",
                "host_pos_score": f"{host_pos_score:.8f}",
                "host_neg_score": f"{host_neg_score:.8f}",
                "host_margin": f"{host_margin:.8f}",
                "iapr_pos_score": f"{iapr_pos_score:.8f}",
                "iapr_neg_score": f"{iapr_neg_score:.8f}",
                "iapr_margin": f"{iapr_margin:.8f}",
                "margin_gain": f"{margin_gain:.8f}",
                "selected_positive_indices": join_indices(selected_positive),
                "selected_hard_negative_indices": join_indices(selected_negative),
                "output_png": png_path,
                "output_pdf": pdf_path,
            }
        )

    summary_path = output_dir / "summary.csv"
    sorted_path = output_dir / "summary_sorted_by_margin_gain.csv"
    write_summary_csv(rows, summary_path)
    sorted_rows = sorted(rows, key=lambda row: float(row["margin_gain"]), reverse=True)
    write_summary_csv(sorted_rows, sorted_path)

    print(f"[Summary] wrote {summary_path}")
    print(f"[Summary] wrote {sorted_path}")
    print_top_margin_gains(rows, limit=10)
    print(f"\nPlotted queries: {len(rows)}")
    if skipped:
        print(f"Skipped queries without usable local neighborhoods: {skipped}")


if __name__ == "__main__":
    main()
