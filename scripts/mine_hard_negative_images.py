import argparse
import csv
import html
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import build_model
from model.clip_model import tokenize as clip_tokenize


ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = {
    "CUHK-PEDES": {
        "dataset_dir": "CUHK-PEDES",
        "annotation_file": "reid_raw.json",
        "image_dir": "imgs",
        "path_key": "file_path",
    },
    "ICFG-PEDES": {
        "dataset_dir": "ICFG-PEDES",
        "annotation_file": "ICFG-PEDES.json",
        "image_dir": "imgs",
        "path_key": "file_path",
    },
    "RSTPReid": {
        "dataset_dir": "RSTPReid",
        "annotation_file": "data_captions.json",
        "image_dir": "imgs",
        "path_key": "img_path",
    },
}


def read_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    return Image.open(img_path).convert("RGB")


def build_eval_transform(img_size):
    height, width = img_size
    return T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


class ImagePathDataset(Dataset):
    def __init__(self, image_rows, transform):
        self.image_rows = image_rows
        self.transform = transform

    def __len__(self):
        return len(self.image_rows)

    def __getitem__(self, index):
        row = self.image_rows[index]
        image = read_image(row["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "pid": row["pid"],
            "image_index": row["image_index"],
            "image_path": row["image_path"],
        }


def collate_image_batch(batch):
    return {
        "images": torch.stack([item["image"] for item in batch], dim=0),
        "pids": torch.tensor([item["pid"] for item in batch], dtype=torch.long),
        "image_indices": torch.tensor([item["image_index"] for item in batch], dtype=torch.long),
        "image_paths": [item["image_path"] for item in batch],
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Mine hard-negative images for visualization. The script loads a dataset split "
            "and a CLIP/ITSELF checkpoint, then writes positive images and ranked negative "
            "images for up to K captions per identity."
        )
    )
    parser.add_argument("--dataset-name", choices=sorted(DATASETS), default="RSTPReid")
    parser.add_argument("--root-dir", default="data", help="Dataset root containing RSTPReid/CUHK-PEDES/etc.")
    parser.add_argument(
        "--input-jsonl",
        default="",
        help="Render outputs from an existing JSONL file and skip model/dataset mining.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "val", "test"],
        help="Dataset splits to mine. Each split is mined against its own image gallery.",
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Optional ITSELF checkpoint. If omitted, the OpenAI CLIP backbone from --pretrain-choice is used.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default="",
        help="Optional baseline checkpoint for two-checkpoint comparison mode.",
    )
    parser.add_argument(
        "--best-checkpoint",
        default="",
        help="Optional best checkpoint for two-checkpoint comparison mode.",
    )
    parser.add_argument("--baseline-name", default="baseline", help="Display/output name for the baseline checkpoint.")
    parser.add_argument("--best-name", default="best", help="Display/output name for the best checkpoint.")
    parser.add_argument("--pretrain-choice", default="ViT-B/16")
    parser.add_argument("--img-size", nargs=2, type=int, default=[384, 128], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--stride-size", type=int, default=16)
    parser.add_argument("--text-length", type=int, default=77)
    parser.add_argument("--select-ratio", type=float, default=0.4)
    parser.add_argument(
        "--feature",
        choices=["global", "grab", "ensemble"],
        default="global",
        help="Feature used for caption-image similarity. ensemble = alpha*global + (1-alpha)*grab.",
    )
    parser.add_argument("--ensemble-alpha", type=float, default=0.5)
    parser.add_argument(
        "--captions-per-id",
        type=int,
        default=2,
        help="Number of captions to output per identity per split. Use 0 for all captions.",
    )
    parser.add_argument("--caption-selection", choices=["first", "random"], default="first")
    parser.add_argument("--hard-k", type=int, default=20, help="Number of hard negative images per row.")
    parser.add_argument(
        "--allow-repeat-negative-id",
        action="store_true",
        help="Allow multiple hard negatives from the same negative identity.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hard_negative_visualization",
        help="Directory for jsonl/csv/html outputs.",
    )
    parser.add_argument("--output-prefix", default="", help="Optional filename prefix.")
    parser.add_argument("--image-batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--html-max-rows",
        type=int,
        default=500,
        help="Maximum rows rendered into HTML. Use 0 to render all rows.",
    )
    parser.add_argument(
        "--html-positive-limit",
        type=int,
        default=5,
        help="Maximum positive thumbnails per HTML row. JSONL still contains all positives.",
    )
    parser.add_argument("--no-html", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--grid-image", action="store_true", help="Write PNG contact-sheet image pages.")
    parser.add_argument(
        "--grid-rows-per-page",
        type=int,
        default=50,
        help="Rows per PNG page. Use 0 to render all rows into one very tall image.",
    )
    parser.add_argument("--grid-width", type=int, default=2200, help="PNG page width in pixels.")
    parser.add_argument("--grid-thumb-width", type=int, default=120)
    parser.add_argument("--grid-thumb-height", type=int, default=180)
    parser.add_argument(
        "--grid-positive-limit",
        type=int,
        default=5,
        help="Maximum positive thumbnails per PNG row. Use 0 to show all positives.",
    )
    parser.add_argument(
        "--compare-ranks",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="R@K thresholds used in baseline-vs-best comparison plots.",
    )
    parser.add_argument(
        "--compare-top-m",
        type=int,
        default=50,
        help="Number of test captions to plot where best improves baseline at R@1.",
    )
    parser.add_argument(
        "--compare-split",
        default="test",
        choices=["train", "val", "test"],
        help="Split used for the baseline-vs-best comparison plot.",
    )
    parser.add_argument(
        "--compare-rows-per-page",
        type=int,
        default=20,
        help="Rows per baseline-vs-best comparison PNG page. Use 0 for one tall image.",
    )
    parser.add_argument(
        "--no-compare-plot",
        action="store_true",
        help="Disable automatic baseline-vs-best comparison PNG/CSV/JSONL outputs.",
    )
    return parser.parse_args()


def build_model_args(args):
    return SimpleNamespace(
        loss_names="tal+cid",
        pretrain_choice=args.pretrain_choice,
        img_size=tuple(args.img_size),
        stride_size=args.stride_size,
        temperature=0.02,
        prototype=False,
        use_loss_id=False,
        prototype_feature="auto",
        prototype_dim=512,
        prototype_per_id=2,
        prototype_projector="default",
        prototype_residual_scale=0.1,
        prototype_kmeans_iters=20,
        prototype_warmup_epochs=0,
        prototype_tau=0.05,
        prototype_hard_k=16,
        prototype_id_weight=0.2,
        prototype_momentum=0.2,
        no_pbt=False,
        only_global=args.feature == "global",
        select_ratio=args.select_ratio,
        return_all=False,
        topk_type="mean",
        layer_index=-1,
        average_attn_weights=True,
        modify_k=False,
    )


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def strip_module_prefix(key):
    return key[7:] if key.startswith("module.") else key


def load_checkpoint_for_inference(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded_state = checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    update_state = {}
    skipped_missing = 0
    skipped_shape = 0

    for raw_key, value in loaded_state.items():
        key = strip_module_prefix(raw_key)
        candidate_keys = [key]
        if not key.startswith("base_model."):
            candidate_keys.append(f"base_model.{key}")

        target_key = None
        for candidate in candidate_keys:
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
    }


def load_dataset_annotations(dataset_name, root_dir):
    config = DATASETS[dataset_name]
    root_path = Path(root_dir).expanduser()
    if not root_path.is_absolute():
        root_path = REPO_ROOT / root_path
    root_path = root_path.resolve()

    dataset_dir = root_path / config["dataset_dir"]
    img_dir = dataset_dir / config["image_dir"]
    annotation_path = dataset_dir / config["annotation_file"]

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        annos = json.load(file)

    splits = {"train": [], "val": [], "test": []}
    for anno in annos:
        split = anno.get("split")
        if split == "train":
            splits["train"].append(anno)
        elif split == "test":
            splits["test"].append(anno)
        else:
            splits["val"].append(anno)

    train_pids = {
        annotation_pid(dataset_name, "train", anno)
        for anno in splits["train"]
    }
    return {
        "img_dir": str(img_dir),
        "path_key": config["path_key"],
        "splits": splits,
        "num_train_ids": len(train_pids),
    }


def split_annotations(dataset, split):
    return dataset["splits"][split]


def annotation_pid(dataset_name, split, anno):
    pid = int(anno["id"])
    if dataset_name == "CUHK-PEDES" and split == "train":
        pid -= 1
    return pid


def annotation_image_path(dataset, anno):
    rel_path = anno.get(dataset["path_key"])
    if rel_path is None:
        raise KeyError(f"Annotation has no path key: {dataset['path_key']}")
    return str((Path(dataset["img_dir"]) / rel_path).resolve())


def collect_split_rows(dataset, dataset_name, split):
    annos = split_annotations(dataset, split)
    caption_rows = []
    image_rows = []
    image_key_to_index = {}
    pid_to_image_indices = defaultdict(list)

    for image_id, anno in enumerate(annos):
        pid = annotation_pid(dataset_name, split, anno)
        image_path = annotation_image_path(dataset, anno)
        image_key = (pid, image_path)
        if image_key not in image_key_to_index:
            image_index = len(image_rows)
            image_key_to_index[image_key] = image_index
            image_rows.append({
                "pid": pid,
                "image_index": image_index,
                "image_id": image_id,
                "image_path": image_path,
            })
            pid_to_image_indices[pid].append(image_index)
        else:
            image_index = image_key_to_index[image_key]

        for caption_id, caption in enumerate(anno["captions"]):
            caption_rows.append({
                "pid": pid,
                "caption": caption,
                "caption_index": len(caption_rows),
                "caption_id_within_image": caption_id,
                "image_id": image_id,
                "paired_image_index": image_index,
                "paired_positive_image_path": image_path,
            })

    return caption_rows, image_rows, dict(pid_to_image_indices)


def select_caption_rows(caption_rows, captions_per_id, mode, seed):
    grouped = defaultdict(list)
    for row in caption_rows:
        grouped[row["pid"]].append(row)

    rng = random.Random(seed)
    selected = []
    for pid in sorted(grouped):
        rows = list(grouped[pid])
        if mode == "random":
            rng.shuffle(rows)
        if captions_per_id > 0:
            rows = rows[:captions_per_id]
        for rank, row in enumerate(rows):
            copied = dict(row)
            copied["caption_rank_within_pid"] = rank
            selected.append(copied)
    return selected


@torch.no_grad()
def encode_image_bank(model, image_rows, transform, args, device):
    loader = DataLoader(
        ImagePathDataset(image_rows, transform),
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_image_batch,
        pin_memory=device.type == "cuda",
    )

    global_feats = []
    grab_feats = []
    image_pids = []
    image_indices = []
    image_paths = []

    for batch in tqdm(loader, desc="Encoding images", leave=False):
        images = batch["images"].to(device, non_blocking=True)
        if args.feature in ("global", "ensemble"):
            feats = model.encode_image(images).float()
            global_feats.append(F.normalize(feats, p=2, dim=1).cpu())
        if args.feature in ("grab", "ensemble"):
            feats = model.encode_image_grab(images).float()
            grab_feats.append(F.normalize(feats, p=2, dim=1).cpu())
        image_pids.append(batch["pids"])
        image_indices.append(batch["image_indices"])
        image_paths.extend(batch["image_paths"])

    bank = {
        "image_pids": torch.cat(image_pids, dim=0).long(),
        "image_indices": torch.cat(image_indices, dim=0).long(),
        "image_paths": image_paths,
    }
    if global_feats:
        bank["global_feats"] = torch.cat(global_feats, dim=0)
    if grab_feats:
        bank["grab_feats"] = torch.cat(grab_feats, dim=0)
    return bank


@torch.no_grad()
def encode_text_features(model, captions, args, device):
    tokens = clip_tokenize(captions, context_length=args.text_length, truncate=True).to(device)
    features = {}
    if args.feature in ("global", "ensemble"):
        feats = model.encode_text(tokens).float()
        features["global_feats"] = F.normalize(feats, p=2, dim=1).cpu()
    if args.feature in ("grab", "ensemble"):
        feats = model.encode_text_grab(tokens).float()
        features["grab_feats"] = F.normalize(feats, p=2, dim=1).cpu()
    return features


def similarity_matrix(text_features, image_bank, args):
    if args.feature == "global":
        return text_features["global_feats"] @ image_bank["global_feats"].t()
    if args.feature == "grab":
        return text_features["grab_feats"] @ image_bank["grab_feats"].t()
    global_sims = text_features["global_feats"] @ image_bank["global_feats"].t()
    grab_sims = text_features["grab_feats"] @ image_bank["grab_feats"].t()
    alpha = float(args.ensemble_alpha)
    return alpha * global_sims + (1.0 - alpha) * grab_sims


def clean_compare_ranks(args):
    ranks = []
    for rank in getattr(args, "compare_ranks", [1, 5, 10]):
        rank = int(rank)
        if rank > 0:
            ranks.append(rank)
    ranks = sorted(set(ranks))
    return ranks or [1]


def retrieval_details(scores, image_bank, query_pid, paired_index, top_k, recall_ks):
    image_pids = image_bank["image_pids"]
    sorted_indices = torch.argsort(scores, descending=True)
    query_pid = int(query_pid)
    paired_index = int(paired_index)
    top_k = max(0, int(top_k))
    top_items = []
    first_positive_rank = None

    for rank, index in enumerate(sorted_indices.tolist(), start=1):
        score = float(scores[index].item())
        pid = int(image_pids[index].item())
        is_positive = pid == query_pid
        if rank <= top_k:
            top_items.append({
                "rank": rank,
                "pid": pid,
                "image_index": int(image_bank["image_indices"][index].item()),
                "image_path": image_bank["image_paths"][index],
                "similarity": score,
                "is_positive": is_positive,
                "is_paired_positive": int(index) == paired_index,
            })
        if is_positive and first_positive_rank is None:
            first_positive_rank = rank
        if first_positive_rank is not None and rank >= top_k:
            break

    recall_hits = {
        f"R{int(k)}": bool(first_positive_rank is not None and first_positive_rank <= int(k))
        for k in recall_ks
    }
    rank1 = top_items[0] if top_items else None
    return {
        "first_positive_rank": first_positive_rank,
        "recall_hits": recall_hits,
        "rank1_pid": rank1["pid"] if rank1 else None,
        "rank1_image_path": rank1["image_path"] if rank1 else None,
        "rank1_similarity": rank1["similarity"] if rank1 else None,
        "rank1_is_positive": bool(rank1 and rank1["is_positive"]),
        "top_retrieved_images": top_items,
    }


def recall_summary(rows, ranks):
    summary = {"num_queries": len(rows)}
    for rank in ranks:
        key = f"R{int(rank)}"
        hits = [bool(row.get("recall_hits", {}).get(key)) for row in rows]
        summary[key] = float(sum(hits) * 100.0 / len(hits)) if hits else 0.0
    return summary


def ranked_positive_images(scores, image_bank, positive_indices, paired_index):
    positives = []
    for image_index in positive_indices:
        score = float(scores[image_index].item())
        positives.append({
            "image_index": int(image_bank["image_indices"][image_index].item()),
            "image_path": image_bank["image_paths"][image_index],
            "similarity": score,
            "is_paired_positive": int(image_index) == int(paired_index),
        })
    positives.sort(key=lambda item: item["similarity"], reverse=True)
    return positives


def select_hard_negative_images(scores, image_bank, query_pid, hard_k, unique_negative_pid):
    if hard_k <= 0:
        return []

    image_pids = image_bank["image_pids"]
    masked_scores = scores.clone()
    masked_scores[image_pids.eq(int(query_pid))] = float("-inf")

    if not torch.isfinite(masked_scores).any():
        return []

    hard_negatives = []
    seen_pids = set()
    sorted_indices = torch.argsort(masked_scores, descending=True)

    for index in sorted_indices.tolist():
        score = float(masked_scores[index].item())
        if not math.isfinite(score):
            break
        neg_pid = int(image_pids[index].item())
        if unique_negative_pid and neg_pid in seen_pids:
            continue
        seen_pids.add(neg_pid)
        hard_negatives.append({
            "rank": len(hard_negatives) + 1,
            "pid": neg_pid,
            "image_index": int(image_bank["image_indices"][index].item()),
            "image_path": image_bank["image_paths"][index],
            "similarity": score,
        })
        if len(hard_negatives) >= hard_k:
            break
    return hard_negatives


def mine_split(dataset, dataset_name, split, model, transform, args, device, run_name="", top_retrieval_k=0, recall_ks=None):
    recall_ks = recall_ks or []
    caption_rows, image_rows, pid_to_image_indices = collect_split_rows(dataset, dataset_name, split)
    selected_caption_rows = select_caption_rows(
        caption_rows,
        captions_per_id=args.captions_per_id,
        mode=args.caption_selection,
        seed=args.seed,
    )

    log_name = f"{run_name}:" if run_name else ""
    print(
        f"[{log_name}{split}] images={len(image_rows)} captions={len(caption_rows)} "
        f"selected_captions={len(selected_caption_rows)} identities={len(pid_to_image_indices)}"
    )
    image_bank = encode_image_bank(model, image_rows, transform, args, device)

    rows = []
    unique_negative_pid = not args.allow_repeat_negative_id
    for start in tqdm(range(0, len(selected_caption_rows), args.text_batch_size), desc=f"Mining {split}"):
        batch_rows = selected_caption_rows[start:start + args.text_batch_size]
        captions = [row["caption"] for row in batch_rows]
        text_features = encode_text_features(model, captions, args, device)
        sims = similarity_matrix(text_features, image_bank, args)

        for row_offset, caption_row in enumerate(batch_rows):
            scores = sims[row_offset]
            pid = int(caption_row["pid"])
            positive_indices = pid_to_image_indices.get(pid, [])
            positives = ranked_positive_images(
                scores,
                image_bank,
                positive_indices,
                paired_index=caption_row["paired_image_index"],
            )
            hard_negatives = select_hard_negative_images(
                scores,
                image_bank,
                query_pid=pid,
                hard_k=args.hard_k,
                unique_negative_pid=unique_negative_pid,
            )
            row_data = {
                "dataset": dataset_name,
                "split": split,
                "feature": args.feature,
                "ensemble_alpha": float(args.ensemble_alpha) if args.feature == "ensemble" else None,
                "pid": pid,
                "caption_index": int(caption_row["caption_index"]),
                "caption_rank_within_pid": int(caption_row["caption_rank_within_pid"]),
                "image_id": int(caption_row["image_id"]),
                "caption_id_within_image": int(caption_row["caption_id_within_image"]),
                "caption": caption_row["caption"],
                "paired_positive_image_path": caption_row["paired_positive_image_path"],
                "positive_images": positives,
                "hard_negative_images": hard_negatives,
            }
            if run_name:
                row_data["checkpoint_name"] = run_name
            if top_retrieval_k > 0 or recall_ks:
                row_data.update(retrieval_details(
                    scores,
                    image_bank,
                    query_pid=pid,
                    paired_index=caption_row["paired_image_index"],
                    top_k=top_retrieval_k,
                    recall_ks=recall_ks,
                ))
            rows.append(row_data)
    return rows


def output_paths(args, split, run_name=""):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{args.dataset_name}_{split}_{args.feature}"
    if run_name:
        prefix = f"{prefix}_{run_name}"
    return {
        "jsonl": output_dir / f"{prefix}.jsonl",
        "csv": output_dir / f"{prefix}.csv",
        "html": output_dir / f"{prefix}.html",
        "grid": output_dir / f"{prefix}_grid.png",
    }


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    fieldnames = [
        "dataset",
        "split",
        "checkpoint_name",
        "feature",
        "pid",
        "caption_index",
        "caption_rank_within_pid",
        "caption",
        "paired_positive_image_path",
        "first_positive_rank",
        "rank1_pid",
        "rank1_image_path",
        "rank1_similarity",
        "rank1_is_positive",
        "positive_image_paths",
        "positive_scores",
        "hard_negative_pids",
        "hard_negative_image_paths",
        "hard_negative_scores",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            positives = row["positive_images"]
            negatives = row["hard_negative_images"]
            writer.writerow({
                "dataset": row["dataset"],
                "split": row["split"],
                "checkpoint_name": row.get("checkpoint_name", ""),
                "feature": row["feature"],
                "pid": row["pid"],
                "caption_index": row["caption_index"],
                "caption_rank_within_pid": row["caption_rank_within_pid"],
                "caption": row["caption"],
                "paired_positive_image_path": row["paired_positive_image_path"],
                "first_positive_rank": row.get("first_positive_rank", ""),
                "rank1_pid": row.get("rank1_pid", ""),
                "rank1_image_path": row.get("rank1_image_path", ""),
                "rank1_similarity": f"{row['rank1_similarity']:.6f}" if row.get("rank1_similarity") is not None else "",
                "rank1_is_positive": row.get("rank1_is_positive", ""),
                "positive_image_paths": ";".join(item["image_path"] for item in positives),
                "positive_scores": ";".join(f"{item['similarity']:.6f}" for item in positives),
                "hard_negative_pids": ";".join(str(item["pid"]) for item in negatives),
                "hard_negative_image_paths": ";".join(item["image_path"] for item in negatives),
                "hard_negative_scores": ";".join(f"{item['similarity']:.6f}" for item in negatives),
            })


def display_path(path):
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def html_image_src(image_path, html_path):
    image_path = Path(image_path).expanduser().resolve()
    html_dir = Path(html_path).expanduser().resolve().parent
    try:
        rel_path = os.path.relpath(str(image_path), str(html_dir))
    except ValueError:
        return image_path.as_uri()
    return quote(rel_path.replace(os.sep, "/"), safe="/")


def render_image_card(item, label, html_path):
    src = html.escape(html_image_src(item["image_path"], html_path), quote=True)
    path = html.escape(display_path(item["image_path"]))
    score = html.escape(f"{item['similarity']:.4f}")
    pid = html.escape(str(item.get("pid", "")))
    extra = f"<div>pid: {pid}</div>" if pid else ""
    paired = " paired" if item.get("is_paired_positive") else ""
    return (
        f'<figure class="thumb{paired}">'
        f'<img src="{src}" loading="lazy" alt="{html.escape(label)}">'
        f"<figcaption><strong>{html.escape(label)}</strong>{extra}<div>sim: {score}</div>"
        f'<div class="path">{path}</div></figcaption></figure>'
    )


def write_html(rows, path, max_rows, positive_limit):
    visible_rows = rows if max_rows == 0 else rows[:max_rows]
    blocks = []
    for row in visible_rows:
        positives = row["positive_images"]
        if positive_limit > 0:
            positives = positives[:positive_limit]
        positive_cards = "\n".join(render_image_card(item, "positive", path) for item in positives)
        negative_cards = "\n".join(
            render_image_card(item, f"negative #{item['rank']}", path)
            for item in row["hard_negative_images"]
        )
        blocks.append(
            '<section class="row">'
            f'<header><div class="meta">split={html.escape(row["split"])} '
            f'pid={row["pid"]} caption_index={row["caption_index"]}</div>'
            f'<p>{html.escape(row["caption"])}</p></header>'
            '<h3>Positive images</h3>'
            f'<div class="grid">{positive_cards}</div>'
            '<h3>Hard negative images</h3>'
            f'<div class="grid">{negative_cards}</div>'
            '</section>'
        )

    note = ""
    if max_rows > 0 and len(rows) > max_rows:
        note = f"<p>Showing {max_rows} of {len(rows)} rows. JSONL/CSV contain all rows.</p>"

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hard Negative Images</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; background: #f7f9fb; }}
.summary {{ margin-bottom: 20px; }}
.row {{ background: white; border: 1px solid #d8dee6; border-radius: 8px; padding: 16px; margin-bottom: 18px; }}
.meta {{ color: #53616f; font-size: 13px; margin-bottom: 6px; }}
p {{ margin: 0 0 12px; line-height: 1.45; }}
h3 {{ font-size: 14px; margin: 14px 0 8px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(138px, 1fr)); gap: 10px; }}
.thumb {{ margin: 0; border: 1px solid #d8dee6; border-radius: 8px; overflow: hidden; background: #fff; }}
.thumb.paired {{ outline: 3px solid #2563eb; }}
.thumb img {{ display: block; width: 100%; aspect-ratio: 2 / 3; object-fit: cover; background: #e5e9ef; }}
figcaption {{ padding: 8px; font-size: 12px; line-height: 1.35; }}
.path {{ margin-top: 4px; color: #6b7785; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<div class="summary">
<h1>Hard Negative Images</h1>
<p>Total rows: {len(rows)}</p>
{note}
</div>
{''.join(blocks)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def load_grid_font(size, bold=False):
    candidates = []
    if os.name == "nt":
        candidates.extend([
            Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        ])
    candidates.extend([
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ])
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def line_height(draw, font):
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    return bbox[3] - bbox[1] + 4


def wrap_text(draw, text, font, max_width, max_lines=0):
    words = str(text).replace("\n", " ").split()
    if not words:
        return [""]

    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_width(draw, candidate, font) <= max_width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = word
        else:
            lines.append(word)
            current = ""
    if current:
        lines.append(current)

    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and text_width(draw, f"{lines[-1]}...", font) > max_width:
            lines[-1] = lines[-1][:-1]
        lines[-1] = f"{lines[-1]}..."
    return lines


def grid_config(args, draw):
    thumb_w = int(args.grid_thumb_width)
    thumb_h = int(args.grid_thumb_height)
    card_pad = 8
    footer_h = 78
    card_w = thumb_w + card_pad * 2
    card_h = thumb_h + footer_h + card_pad * 2
    margin = 24
    gap = 12
    available_w = int(args.grid_width) - margin * 2
    cols = max(1, (available_w + gap) // (card_w + gap))
    return {
        "width": int(args.grid_width),
        "margin": margin,
        "gap": gap,
        "row_gap": 18,
        "row_pad": 16,
        "thumb_w": thumb_w,
        "thumb_h": thumb_h,
        "card_pad": card_pad,
        "card_w": card_w,
        "card_h": card_h,
        "cols": cols,
        "title_font": load_grid_font(26, bold=True),
        "meta_font": load_grid_font(16, bold=True),
        "body_font": load_grid_font(15),
        "section_font": load_grid_font(17, bold=True),
        "label_font": load_grid_font(14, bold=True),
        "small_font": load_grid_font(12),
    }


def visible_positive_images(row, args):
    positives = row["positive_images"]
    if args.grid_positive_limit > 0:
        return positives[:args.grid_positive_limit]
    return positives


def section_height(item_count, cfg):
    if item_count <= 0:
        return 0
    grid_rows = math.ceil(item_count / cfg["cols"])
    return 24 + grid_rows * cfg["card_h"] + max(0, grid_rows - 1) * cfg["gap"] + 8


def measure_grid_row(row, args, cfg, draw):
    inner_w = cfg["width"] - cfg["margin"] * 2 - cfg["row_pad"] * 2
    meta_h = line_height(draw, cfg["meta_font"])
    caption_lines = wrap_text(draw, row["caption"], cfg["body_font"], inner_w, max_lines=3)
    caption_h = len(caption_lines) * line_height(draw, cfg["body_font"])
    pos_h = section_height(len(visible_positive_images(row, args)), cfg)
    neg_h = section_height(len(row["hard_negative_images"]), cfg)
    return cfg["row_pad"] * 2 + meta_h + 6 + caption_h + 12 + pos_h + neg_h


def load_grid_thumbnail(image_path, size):
    canvas = Image.new("RGB", size, "#e5e9ef")
    try:
        image = read_image(image_path)
        image = ImageOps.contain(image, size)
        x = (size[0] - image.width) // 2
        y = (size[1] - image.height) // 2
        canvas.paste(image, (x, y))
    except Exception:
        draw = ImageDraw.Draw(canvas)
        font = load_grid_font(13, bold=True)
        draw.text((10, size[1] // 2 - 10), "missing image", fill="#9a1c1c", font=font)
    return canvas


def draw_grid_card(draw, page, item, label, x, y, cfg):
    card_w = cfg["card_w"]
    card_h = cfg["card_h"]
    outline = "#2563eb" if item.get("is_paired_positive") else "#d8dee6"
    width = 4 if item.get("is_paired_positive") else 1
    draw.rounded_rectangle(
        [x, y, x + card_w, y + card_h],
        radius=8,
        fill="#ffffff",
        outline=outline,
        width=width,
    )

    pad = cfg["card_pad"]
    thumb = load_grid_thumbnail(item["image_path"], (cfg["thumb_w"], cfg["thumb_h"]))
    page.paste(thumb, (x + pad, y + pad))

    text_y = y + pad + cfg["thumb_h"] + 8
    draw.text((x + pad, text_y), label, fill="#111827", font=cfg["label_font"])
    text_y += line_height(draw, cfg["label_font"])
    pid = item.get("pid")
    if pid is not None:
        draw.text((x + pad, text_y), f"pid: {pid}", fill="#374151", font=cfg["small_font"])
        text_y += line_height(draw, cfg["small_font"])
    draw.text((x + pad, text_y), f"sim: {item['similarity']:.4f}", fill="#374151", font=cfg["small_font"])
    text_y += line_height(draw, cfg["small_font"])

    filename = Path(item["image_path"]).name
    max_text_w = card_w - pad * 2
    for line in wrap_text(draw, filename, cfg["small_font"], max_text_w, max_lines=1):
        draw.text((x + pad, text_y), line, fill="#6b7785", font=cfg["small_font"])


def draw_grid_section(draw, page, title, items, x, y, cfg):
    if not items:
        return y
    draw.text((x, y), title, fill="#111827", font=cfg["section_font"])
    y += 24

    for index, item in enumerate(items):
        col = index % cfg["cols"]
        row = index // cfg["cols"]
        card_x = x + col * (cfg["card_w"] + cfg["gap"])
        card_y = y + row * (cfg["card_h"] + cfg["gap"])
        if title.startswith("Positive"):
            label = "positive"
        else:
            label = f"negative #{item['rank']}"
        draw_grid_card(draw, page, item, label, card_x, card_y, cfg)

    grid_rows = math.ceil(len(items) / cfg["cols"])
    return y + grid_rows * cfg["card_h"] + max(0, grid_rows - 1) * cfg["gap"] + 8


def draw_grid_row(draw, page, row, args, cfg, y, row_h):
    x = cfg["margin"]
    w = cfg["width"] - cfg["margin"] * 2
    draw.rounded_rectangle(
        [x, y, x + w, y + row_h],
        radius=8,
        fill="#ffffff",
        outline="#d8dee6",
        width=1,
    )

    inner_x = x + cfg["row_pad"]
    inner_y = y + cfg["row_pad"]
    inner_w = w - cfg["row_pad"] * 2
    meta = (
        f"split={row['split']}  pid={row['pid']}  "
        f"caption_index={row['caption_index']}  feature={row['feature']}"
    )
    draw.text((inner_x, inner_y), meta, fill="#53616f", font=cfg["meta_font"])
    inner_y += line_height(draw, cfg["meta_font"]) + 6

    for line in wrap_text(draw, row["caption"], cfg["body_font"], inner_w, max_lines=3):
        draw.text((inner_x, inner_y), line, fill="#1f2933", font=cfg["body_font"])
        inner_y += line_height(draw, cfg["body_font"])
    inner_y += 12

    positives = visible_positive_images(row, args)
    inner_y = draw_grid_section(draw, page, "Positive images", positives, inner_x, inner_y, cfg)
    inner_y = draw_grid_section(draw, page, "Hard negative images", row["hard_negative_images"], inner_x, inner_y, cfg)
    return inner_y


def grid_page_path(base_path, page_index, page_count):
    if page_count == 1:
        return base_path
    return base_path.with_name(f"{base_path.stem}_page{page_index + 1:03d}{base_path.suffix}")


def write_grid_images(rows, base_path, args):
    if not rows:
        return []

    rows_per_page = int(args.grid_rows_per_page)
    if rows_per_page <= 0:
        pages = [rows]
    else:
        pages = [rows[i:i + rows_per_page] for i in range(0, len(rows), rows_per_page)]

    dummy = Image.new("RGB", (int(args.grid_width), 100), "#ffffff")
    dummy_draw = ImageDraw.Draw(dummy)
    cfg = grid_config(args, dummy_draw)
    created = []

    page_iter = tqdm(pages, desc="Writing grid image pages", unit="page")
    for page_index, page_rows in enumerate(page_iter):
        page_iter.set_postfix(rows=len(page_rows))
        row_heights = [measure_grid_row(row, args, cfg, dummy_draw) for row in page_rows]
        header_h = 70
        page_h = (
            cfg["margin"] * 2
            + header_h
            + sum(row_heights)
            + max(0, len(page_rows) - 1) * cfg["row_gap"]
        )
        page = Image.new("RGB", (cfg["width"], page_h), "#f7f9fb")
        draw = ImageDraw.Draw(page)
        y = cfg["margin"]

        title = f"Hard Negative Images - page {page_index + 1}/{len(pages)}"
        draw.text((cfg["margin"], y), title, fill="#111827", font=cfg["title_font"])
        y += line_height(draw, cfg["title_font"]) + 6
        summary = f"rows on page: {len(page_rows)} / total rows: {len(rows)}"
        draw.text((cfg["margin"], y), summary, fill="#53616f", font=cfg["body_font"])
        y = cfg["margin"] + header_h

        row_iter = tqdm(
            zip(page_rows, row_heights),
            total=len(page_rows),
            desc=f"Drawing grid page {page_index + 1}/{len(pages)}",
            unit="row",
            leave=False,
        )
        for row, row_h in row_iter:
            draw_grid_row(draw, page, row, args, cfg, y, row_h)
            y += row_h + cfg["row_gap"]

        output_path = grid_page_path(base_path, page_index, len(pages))
        page.save(output_path)
        created.append(output_path)
    return created


def safe_filename_token(value):
    token = str(value).strip()
    safe = [char if char.isalnum() or char in ("-", "_", ".") else "_" for char in token]
    safe = "".join(safe).strip("_")
    return safe or "run"


def comparison_output_paths(args, split):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{args.dataset_name}_{split}_{args.feature}"
    baseline_name = safe_filename_token(args.baseline_name)
    best_name = safe_filename_token(args.best_name)
    prefix = f"{prefix}_{baseline_name}_vs_{best_name}_compare"
    return {
        "jsonl": output_dir / f"{prefix}.jsonl",
        "csv": output_dir / f"{prefix}.csv",
        "grid": output_dir / f"{prefix}_grid.png",
    }


def comparison_row_key(row):
    return (row.get("split"), int(row["caption_index"]))


def rank_value(row):
    rank = row.get("first_positive_rank")
    if rank is None:
        return 10 ** 9
    return int(rank)


def compact_comparison_run(row):
    return {
        "checkpoint_name": row.get("checkpoint_name", ""),
        "first_positive_rank": row.get("first_positive_rank"),
        "recall_hits": row.get("recall_hits", {}),
        "rank1_pid": row.get("rank1_pid"),
        "rank1_image_path": row.get("rank1_image_path"),
        "rank1_similarity": row.get("rank1_similarity"),
        "rank1_is_positive": row.get("rank1_is_positive"),
        "top_retrieved_images": row.get("top_retrieved_images", []),
    }


def build_comparison_rows(baseline_rows, best_rows, ranks, top_m):
    baseline_by_key = {comparison_row_key(row): row for row in baseline_rows}
    comparison_rows = []
    for best_row in best_rows:
        baseline_row = baseline_by_key.get(comparison_row_key(best_row))
        if baseline_row is None:
            continue

        baseline_rank = rank_value(baseline_row)
        best_rank = rank_value(best_row)
        baseline_r1 = baseline_rank <= 1
        best_r1 = best_rank <= 1
        if not best_r1 or baseline_r1:
            continue

        baseline_rank1 = baseline_row.get("rank1_similarity")
        best_rank1 = best_row.get("rank1_similarity")
        score_delta = None
        if baseline_rank1 is not None and best_rank1 is not None:
            score_delta = float(best_rank1) - float(baseline_rank1)

        comparison_rows.append({
            "dataset": best_row["dataset"],
            "split": best_row["split"],
            "feature": best_row["feature"],
            "ensemble_alpha": best_row.get("ensemble_alpha"),
            "pid": best_row["pid"],
            "caption_index": best_row["caption_index"],
            "caption_rank_within_pid": best_row["caption_rank_within_pid"],
            "caption": best_row["caption"],
            "paired_positive_image_path": best_row["paired_positive_image_path"],
            "rank_improvement": int(baseline_rank - best_rank),
            "rank1_score_delta": score_delta,
            "ranks": [int(rank) for rank in ranks],
            "baseline": compact_comparison_run(baseline_row),
            "best": compact_comparison_run(best_row),
        })

    comparison_rows.sort(
        key=lambda row: (
            row["rank_improvement"],
            row["rank1_score_delta"] if row["rank1_score_delta"] is not None else float("-inf"),
            -int(row["caption_index"]),
        ),
        reverse=True,
    )
    if top_m > 0:
        comparison_rows = comparison_rows[:int(top_m)]
    return comparison_rows


def comparison_transition_summary(baseline_rows, best_rows):
    baseline_by_key = {comparison_row_key(row): row for row in baseline_rows}
    best_by_key = {comparison_row_key(row): row for row in best_rows}
    matched_keys = sorted(set(baseline_by_key).intersection(best_by_key))
    summary = {
        "baseline_queries": len(baseline_rows),
        "best_queries": len(best_rows),
        "matched_queries": len(matched_keys),
        "baseline_only_keys": len(set(baseline_by_key) - set(best_by_key)),
        "best_only_keys": len(set(best_by_key) - set(baseline_by_key)),
        "both_hit_r1": 0,
        "baseline_only_hit_r1": 0,
        "best_only_hit_r1": 0,
        "both_miss_r1": 0,
    }
    for key in matched_keys:
        baseline_hit = rank_value(baseline_by_key[key]) <= 1
        best_hit = rank_value(best_by_key[key]) <= 1
        if baseline_hit and best_hit:
            summary["both_hit_r1"] += 1
        elif baseline_hit:
            summary["baseline_only_hit_r1"] += 1
        elif best_hit:
            summary["best_only_hit_r1"] += 1
        else:
            summary["both_miss_r1"] += 1

    matched = summary["matched_queries"]
    baseline_hits = summary["both_hit_r1"] + summary["baseline_only_hit_r1"]
    best_hits = summary["both_hit_r1"] + summary["best_only_hit_r1"]
    summary["baseline_r1_percent"] = float(baseline_hits * 100.0 / matched) if matched else 0.0
    summary["best_r1_percent"] = float(best_hits * 100.0 / matched) if matched else 0.0
    return summary


def joined_top_values(run, key, fmt=None):
    values = []
    for item in run.get("top_retrieved_images", []):
        value = item.get(key)
        if value is None:
            values.append("")
        elif fmt is not None:
            values.append(fmt(value))
        else:
            values.append(str(value))
    return ";".join(values)


def write_comparison_csv(rows, path, ranks):
    fieldnames = [
        "dataset",
        "split",
        "feature",
        "pid",
        "caption_index",
        "caption_rank_within_pid",
        "caption",
        "paired_positive_image_path",
        "rank_improvement",
        "rank1_score_delta",
        "baseline_first_positive_rank",
        "best_first_positive_rank",
        "baseline_rank1_pid",
        "best_rank1_pid",
        "baseline_rank1_image_path",
        "best_rank1_image_path",
    ]
    for rank in ranks:
        fieldnames.extend([f"baseline_R{rank}", f"best_R{rank}"])
    fieldnames.extend([
        "baseline_top_pids",
        "best_top_pids",
        "baseline_top_image_paths",
        "best_top_image_paths",
        "baseline_top_scores",
        "best_top_scores",
    ])

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            baseline = row["baseline"]
            best = row["best"]
            output = {
                "dataset": row["dataset"],
                "split": row["split"],
                "feature": row["feature"],
                "pid": row["pid"],
                "caption_index": row["caption_index"],
                "caption_rank_within_pid": row["caption_rank_within_pid"],
                "caption": row["caption"],
                "paired_positive_image_path": row["paired_positive_image_path"],
                "rank_improvement": row["rank_improvement"],
                "rank1_score_delta": f"{row['rank1_score_delta']:.6f}" if row["rank1_score_delta"] is not None else "",
                "baseline_first_positive_rank": baseline.get("first_positive_rank"),
                "best_first_positive_rank": best.get("first_positive_rank"),
                "baseline_rank1_pid": baseline.get("rank1_pid"),
                "best_rank1_pid": best.get("rank1_pid"),
                "baseline_rank1_image_path": baseline.get("rank1_image_path"),
                "best_rank1_image_path": best.get("rank1_image_path"),
                "baseline_top_pids": joined_top_values(baseline, "pid"),
                "best_top_pids": joined_top_values(best, "pid"),
                "baseline_top_image_paths": joined_top_values(baseline, "image_path"),
                "best_top_image_paths": joined_top_values(best, "image_path"),
                "baseline_top_scores": joined_top_values(baseline, "similarity", lambda value: f"{value:.6f}"),
                "best_top_scores": joined_top_values(best, "similarity", lambda value: f"{value:.6f}"),
            }
            for rank in ranks:
                key = f"R{rank}"
                output[f"baseline_R{rank}"] = baseline.get("recall_hits", {}).get(key, False)
                output[f"best_R{rank}"] = best.get("recall_hits", {}).get(key, False)
            writer.writerow(output)


def compare_panel_cols(panel_w, cfg):
    return max(1, int((panel_w + cfg["gap"]) // (cfg["card_w"] + cfg["gap"])))


def compare_panel_height(item_count, panel_w, cfg):
    if item_count <= 0:
        return 24
    cols = compare_panel_cols(panel_w, cfg)
    grid_rows = math.ceil(item_count / cols)
    return 28 + grid_rows * cfg["card_h"] + max(0, grid_rows - 1) * cfg["gap"]


def measure_comparison_row(row, args, cfg, draw):
    inner_w = cfg["width"] - cfg["margin"] * 2 - cfg["row_pad"] * 2
    panel_w = (inner_w - cfg["side_gap"]) // 2
    meta_h = line_height(draw, cfg["meta_font"])
    caption_lines = wrap_text(draw, row["caption"], cfg["body_font"], inner_w, max_lines=3)
    caption_h = len(caption_lines) * line_height(draw, cfg["body_font"])
    panel_h = max(
        compare_panel_height(len(row["baseline"].get("top_retrieved_images", [])), panel_w, cfg),
        compare_panel_height(len(row["best"].get("top_retrieved_images", [])), panel_w, cfg),
    )
    return cfg["row_pad"] * 2 + meta_h + 6 + caption_h + 14 + panel_h


def draw_compare_card(draw, page, item, x, y, cfg):
    card_w = cfg["card_w"]
    card_h = cfg["card_h"]
    if item.get("is_positive"):
        outline = "#16a34a"
        width = 4
    elif int(item.get("rank", 0)) == 1:
        outline = "#dc2626"
        width = 3
    else:
        outline = "#d8dee6"
        width = 1
    draw.rounded_rectangle([x, y, x + card_w, y + card_h], radius=8, fill="#ffffff", outline=outline, width=width)

    pad = cfg["card_pad"]
    thumb = load_grid_thumbnail(item["image_path"], (cfg["thumb_w"], cfg["thumb_h"]))
    page.paste(thumb, (x + pad, y + pad))

    text_y = y + pad + cfg["thumb_h"] + 8
    label = f"rank #{item['rank']}"
    draw.text((x + pad, text_y), label, fill="#111827", font=cfg["label_font"])
    text_y += line_height(draw, cfg["label_font"])
    draw.text((x + pad, text_y), f"pid: {item['pid']}", fill="#374151", font=cfg["small_font"])
    text_y += line_height(draw, cfg["small_font"])
    draw.text((x + pad, text_y), f"sim: {item['similarity']:.4f}", fill="#374151", font=cfg["small_font"])
    text_y += line_height(draw, cfg["small_font"])
    filename = Path(item["image_path"]).name
    for line in wrap_text(draw, filename, cfg["small_font"], card_w - pad * 2, max_lines=1):
        draw.text((x + pad, text_y), line, fill="#6b7785", font=cfg["small_font"])


def draw_compare_panel(draw, page, title, run, ranks, x, y, panel_w, cfg):
    first_rank = run.get("first_positive_rank")
    hit_bits = []
    for rank in ranks:
        key = f"R{rank}"
        hit_bits.append(f"{key}:{'hit' if run.get('recall_hits', {}).get(key) else 'miss'}")
    header = f"{title}  first_pos_rank={first_rank}  " + "  ".join(hit_bits)
    draw.text((x, y), header, fill="#111827", font=cfg["section_font"])
    y += 28

    items = run.get("top_retrieved_images", [])
    cols = compare_panel_cols(panel_w, cfg)
    for index, item in enumerate(items):
        col = index % cols
        row = index // cols
        card_x = x + col * (cfg["card_w"] + cfg["gap"])
        card_y = y + row * (cfg["card_h"] + cfg["gap"])
        draw_compare_card(draw, page, item, card_x, card_y, cfg)


def draw_comparison_row(draw, page, row, args, cfg, y, row_h, ranks):
    x = cfg["margin"]
    w = cfg["width"] - cfg["margin"] * 2
    draw.rounded_rectangle([x, y, x + w, y + row_h], radius=8, fill="#ffffff", outline="#d8dee6", width=1)

    inner_x = x + cfg["row_pad"]
    inner_y = y + cfg["row_pad"]
    inner_w = w - cfg["row_pad"] * 2
    meta = (
        f"split={row['split']}  pid={row['pid']}  caption_index={row['caption_index']}  "
        f"rank_improvement={row['rank_improvement']}"
    )
    draw.text((inner_x, inner_y), meta, fill="#53616f", font=cfg["meta_font"])
    inner_y += line_height(draw, cfg["meta_font"]) + 6
    for line in wrap_text(draw, row["caption"], cfg["body_font"], inner_w, max_lines=3):
        draw.text((inner_x, inner_y), line, fill="#1f2933", font=cfg["body_font"])
        inner_y += line_height(draw, cfg["body_font"])
    inner_y += 14

    panel_w = (inner_w - cfg["side_gap"]) // 2
    left_x = inner_x
    right_x = inner_x + panel_w + cfg["side_gap"]
    draw.line([(right_x - cfg["side_gap"] // 2, inner_y), (right_x - cfg["side_gap"] // 2, y + row_h - cfg["row_pad"])], fill="#d8dee6", width=1)
    draw_compare_panel(draw, page, args.baseline_name, row["baseline"], ranks, left_x, inner_y, panel_w, cfg)
    draw_compare_panel(draw, page, args.best_name, row["best"], ranks, right_x, inner_y, panel_w, cfg)


def write_comparison_grid_images(rows, base_path, args, ranks, baseline_summary, best_summary):
    if not rows:
        return []

    rows_per_page = int(args.compare_rows_per_page)
    if rows_per_page <= 0:
        pages = [rows]
    else:
        pages = [rows[i:i + rows_per_page] for i in range(0, len(rows), rows_per_page)]

    dummy = Image.new("RGB", (int(args.grid_width), 100), "#ffffff")
    dummy_draw = ImageDraw.Draw(dummy)
    cfg = grid_config(args, dummy_draw)
    cfg["side_gap"] = 28
    created = []

    page_iter = tqdm(pages, desc="Writing comparison grid pages", unit="page")
    for page_index, page_rows in enumerate(page_iter):
        row_heights = [measure_comparison_row(row, args, cfg, dummy_draw) for row in page_rows]
        header_h = 90
        page_h = cfg["margin"] * 2 + header_h + sum(row_heights) + max(0, len(page_rows) - 1) * cfg["row_gap"]
        page = Image.new("RGB", (cfg["width"], page_h), "#f7f9fb")
        draw = ImageDraw.Draw(page)
        y = cfg["margin"]

        title = f"Baseline vs Best R@K - page {page_index + 1}/{len(pages)}"
        draw.text((cfg["margin"], y), title, fill="#111827", font=cfg["title_font"])
        y += line_height(draw, cfg["title_font"]) + 6
        rank_text = "  ".join(
            f"R{rank}: {args.baseline_name}={baseline_summary[f'R{rank}']:.2f}% {args.best_name}={best_summary[f'R{rank}']:.2f}%"
            for rank in ranks
        )
        summary = f"selected rows: {len(rows)}  total queries: {baseline_summary['num_queries']}  {rank_text}"
        draw.text((cfg["margin"], y), summary, fill="#53616f", font=cfg["body_font"])
        y = cfg["margin"] + header_h

        row_iter = tqdm(
            zip(page_rows, row_heights),
            total=len(page_rows),
            desc=f"Drawing comparison page {page_index + 1}/{len(pages)}",
            unit="row",
            leave=False,
        )
        for row, row_h in row_iter:
            draw_comparison_row(draw, page, row, args, cfg, y, row_h, ranks)
            y += row_h + cfg["row_gap"]

        output_path = grid_page_path(base_path, page_index, len(pages))
        page.save(output_path)
        created.append(output_path)
    return created


def save_comparison_outputs(comparison_rows, baseline_rows, best_rows, args, split, ranks):
    paths = comparison_output_paths(args, split)
    baseline_summary = recall_summary(baseline_rows, ranks)
    best_summary = recall_summary(best_rows, ranks)
    transition = comparison_transition_summary(baseline_rows, best_rows)

    print(
        f"[{split}] comparison R@1 transitions: "
        f"matched={transition['matched_queries']} "
        f"{args.baseline_name}_R1={transition['baseline_r1_percent']:.2f}% "
        f"{args.best_name}_R1={transition['best_r1_percent']:.2f}% "
        f"best_only={transition['best_only_hit_r1']} "
        f"baseline_only={transition['baseline_only_hit_r1']} "
        f"both_hit={transition['both_hit_r1']} both_miss={transition['both_miss_r1']}"
    )

    write_jsonl(comparison_rows, paths["jsonl"])
    print(f"[{split}] wrote {paths['jsonl']}")
    write_comparison_csv(comparison_rows, paths["csv"], ranks)
    print(f"[{split}] wrote {paths['csv']}")

    if comparison_rows:
        grid_paths = write_comparison_grid_images(comparison_rows, paths["grid"], args, ranks, baseline_summary, best_summary)
        for grid_path in grid_paths:
            print(f"[{split}] wrote {grid_path}")
    else:
        print(f"[{split}] no queries where {args.best_name} improves {args.baseline_name} at R@1")


def save_outputs(rows, args, split, run_name=""):
    paths = output_paths(args, split, run_name=run_name)
    write_jsonl(rows, paths["jsonl"])
    print(f"[{split}] wrote {paths['jsonl']}")

    if not args.no_csv:
        write_csv(rows, paths["csv"])
        print(f"[{split}] wrote {paths['csv']}")

    if not args.no_html:
        write_html(rows, paths["html"], args.html_max_rows, args.html_positive_limit)
        print(f"[{split}] wrote {paths['html']}")

    if args.grid_image:
        grid_paths = write_grid_images(rows, paths["grid"], args)
        for grid_path in grid_paths:
            print(f"[{split}] wrote {grid_path}")


def load_model_for_run(args, num_classes, checkpoint_path, run_name, device):
    model_args = build_model_args(args)
    model = build_model(model_args, num_classes=num_classes)
    label = f"[{run_name}] " if run_name else ""
    if checkpoint_path:
        stats = load_checkpoint_for_inference(model, checkpoint_path)
        print(
            f"{label}Loaded checkpoint tensors: {stats['loaded']} "
            f"(skipped missing={stats['skipped_missing']}, shape={stats['skipped_shape']})"
        )
    else:
        print(f"{label}No checkpoint supplied; using pretrained backbone {args.pretrain_choice}.")

    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()
    return model


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.input_jsonl:
        rows = read_jsonl(args.input_jsonl)
        if not rows:
            raise RuntimeError(f"No rows found in JSONL file: {args.input_jsonl}")
        if not args.output_prefix:
            args.output_prefix = Path(args.input_jsonl).stem
        split = rows[0].get("split", "all")
        save_outputs(rows, args, split)
        return

    pair_mode = bool(args.baseline_checkpoint) or bool(args.best_checkpoint)
    if pair_mode and not (args.baseline_checkpoint and args.best_checkpoint):
        raise RuntimeError("Use both --baseline-checkpoint and --best-checkpoint for comparison mode.")
    if pair_mode and args.checkpoint:
        raise RuntimeError("Use either --checkpoint, or --baseline-checkpoint with --best-checkpoint, not both.")

    compare_ranks = clean_compare_ranks(args)

    device = torch.device(args.device)
    if args.feature in ("grab", "ensemble") and device.type != "cuda":
        raise RuntimeError("GRAB features use half tensors in this codebase. Use --device cuda or --feature global.")

    dataset = load_dataset_annotations(args.dataset_name, args.root_dir)
    num_classes = dataset["num_train_ids"]

    transform = build_eval_transform(tuple(args.img_size))

    if pair_mode:
        compare_enabled = not args.no_compare_plot and args.compare_split in args.splits
        if not args.no_compare_plot and args.compare_split not in args.splits:
            print(f"[{args.compare_split}] skipped comparison because split is not in --splits")

        run_specs = [
            (args.baseline_name, args.baseline_checkpoint),
            (args.best_name, args.best_checkpoint),
        ]
        rows_by_run = {}
        for run_name, checkpoint_path in run_specs:
            model = load_model_for_run(args, num_classes, checkpoint_path, run_name, device)
            try:
                for split in args.splits:
                    top_k = max(compare_ranks) if compare_enabled and split == args.compare_split else 0
                    recall_ks = compare_ranks if compare_enabled and split == args.compare_split else []
                    rows = mine_split(
                        dataset,
                        args.dataset_name,
                        split,
                        model,
                        transform,
                        args,
                        device,
                        run_name=run_name,
                        top_retrieval_k=top_k,
                        recall_ks=recall_ks,
                    )
                    rows_by_run[(run_name, split)] = rows
                    save_outputs(rows, args, split, run_name=safe_filename_token(run_name))
            finally:
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        if compare_enabled:
            baseline_rows = rows_by_run[(args.baseline_name, args.compare_split)]
            best_rows = rows_by_run[(args.best_name, args.compare_split)]
            comparison_rows = build_comparison_rows(baseline_rows, best_rows, compare_ranks, args.compare_top_m)
            save_comparison_outputs(comparison_rows, baseline_rows, best_rows, args, args.compare_split, compare_ranks)
        return

    model = load_model_for_run(args, num_classes, args.checkpoint, "", device)
    for split in args.splits:
        rows = mine_split(dataset, args.dataset_name, split, model, transform, args, device)
        save_outputs(rows, args, split)


if __name__ == "__main__":
    main()
