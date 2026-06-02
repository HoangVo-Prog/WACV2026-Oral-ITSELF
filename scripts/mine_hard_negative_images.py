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
from PIL import Image, ImageFile
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


def mine_split(dataset, dataset_name, split, model, transform, args, device):
    caption_rows, image_rows, pid_to_image_indices = collect_split_rows(dataset, dataset_name, split)
    selected_caption_rows = select_caption_rows(
        caption_rows,
        captions_per_id=args.captions_per_id,
        mode=args.caption_selection,
        seed=args.seed,
    )

    print(
        f"[{split}] images={len(image_rows)} captions={len(caption_rows)} "
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
            rows.append({
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
            })
    return rows


def output_paths(args, split):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix or f"{args.dataset_name}_{split}_{args.feature}"
    return {
        "jsonl": output_dir / f"{prefix}.jsonl",
        "csv": output_dir / f"{prefix}.csv",
        "html": output_dir / f"{prefix}.html",
    }


def write_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows, path):
    fieldnames = [
        "dataset",
        "split",
        "feature",
        "pid",
        "caption_index",
        "caption_rank_within_pid",
        "caption",
        "paired_positive_image_path",
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
                "feature": row["feature"],
                "pid": row["pid"],
                "caption_index": row["caption_index"],
                "caption_rank_within_pid": row["caption_rank_within_pid"],
                "caption": row["caption"],
                "paired_positive_image_path": row["paired_positive_image_path"],
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


def save_outputs(rows, args, split):
    paths = output_paths(args, split)
    write_jsonl(rows, paths["jsonl"])
    print(f"[{split}] wrote {paths['jsonl']}")

    if not args.no_csv:
        write_csv(rows, paths["csv"])
        print(f"[{split}] wrote {paths['csv']}")

    if not args.no_html:
        write_html(rows, paths["html"], args.html_max_rows, args.html_positive_limit)
        print(f"[{split}] wrote {paths['html']}")


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if args.feature in ("grab", "ensemble") and device.type != "cuda":
        raise RuntimeError("GRAB features use half tensors in this codebase. Use --device cuda or --feature global.")

    dataset = load_dataset_annotations(args.dataset_name, args.root_dir)
    num_classes = dataset["num_train_ids"]

    model_args = build_model_args(args)
    model = build_model(model_args, num_classes=num_classes)
    if args.checkpoint:
        stats = load_checkpoint_for_inference(model, args.checkpoint)
        print(
            f"Loaded checkpoint tensors: {stats['loaded']} "
            f"(skipped missing={stats['skipped_missing']}, shape={stats['skipped_shape']})"
        )
    else:
        print(f"No checkpoint supplied; using pretrained backbone {args.pretrain_choice}.")

    model.to(device)
    if device.type == "cpu":
        model.float()
    model.eval()

    transform = build_eval_transform(tuple(args.img_size))
    for split in args.splits:
        rows = mine_split(dataset, args.dataset_name, split, model, transform, args, device)
        save_outputs(rows, args, split)


if __name__ == "__main__":
    main()
