import argparse
import gc
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.bases import tokenize as dataset_tokenize
from datasets.build import build_transforms
from model.clip_model import build_CLIP_from_openai_pretrained
from utils.simple_tokenizer import SimpleTokenizer


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

TEXT_TOKENIZER = SimpleTokenizer()

def compute_rollout(attentions: torch.Tensor, 
            head_fusion = 'mean', 
            discard: bool = True,
            discard_ratios: list = [0.25, 1., 1., 1., 0.25, 0.25, 1., 1., 1., 1., 0.25, 0.25], 
            start_layer: int = 4, 
            skip_layer: list = [5,6,7,8,9,10]):
    """ 
    Compute rollout attention with optional discarding low attention scores.

    Args:
        attentions (torch.Tensor): [L, B, N, N] attention maps (averaged across heads).
        discard_ratio (float): ratio ratioof lowest attentions to discard (set to 0).
        start_layer (int): layer index to start rollout from. (0 --> 11)

    Returns:
        torch.Tensor: [B, N, N] rollout attention map.
    """
    if len(attentions.shape) == 5:
        L, B, _, N, _ = attentions.shape
    else:
        L, B, N, _ = attentions.shape
    device = attentions.device
    result = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
                
    for layer in range(start_layer, L):
        if layer in skip_layer:
            continue
        attn = attentions[layer]  
        # have H shape (L, B, H, N, N)
        if len(attentions.shape) == 5:
            with torch.no_grad():
                if head_fusion == "mean":
                    attn = attn.mean(axis=1) # [B, H, N, N] --> axis == 1
                elif head_fusion == "max":
                    attn = attn.max(axis=1)[0]
                elif head_fusion == "min":
                    attn = attn.min(axis=1)[0]
                else:
                    raise "Attention head fusion type Not supported"
        
        if discard:
            discard_ratio = discard_ratios[layer]
            flat = attn.view(B, -1)  # [B, N*N]
            num_to_discard = int(flat.size(-1) * discard_ratio)

            if num_to_discard > 0:
                _, indices = flat.topk(num_to_discard, dim=-1, largest=False)
                for b in range(B):
                    idx = indices[b]
                    idx = idx[idx != 0]
                    flat[b, idx] = 0
                attn = flat.view(B, N, N)

        I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
        attn = (attn + I) / 2.0
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn, result)

    return result  # [B, N, N]


def visualize_rollout_attention(image_paths, attention_map, save_dir):
    for b, img_path in enumerate(image_paths):
        attn_map = attention_map[b, 0]  # [N]
        filename = image_paths[b].split('/')[-1].split('.')[0]
        
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # remove CLS
        attn_map = attn_map[1:]
        attn_map = attn_map.reshape(24, 8).detach().cpu().numpy()

        # prepare figure with 2 subplots
        fig, ax = plt.subplots()

        keep_ratio = 0.4
        flat = attn_map.flatten()
        k = int(len(flat) * keep_ratio)
        thresh = np.partition(flat, -k)[-k]
        mask = (attn_map >= thresh).astype(np.float32)

        attn_map_selective = attn_map * mask
        if attn_map_selective.max() > 0:
            attn_map_selective = attn_map_selective / attn_map_selective.max()
        
        attn_uint8 = np.clip(attn_map_selective * 255.0, 0, 255).astype(np.uint8)
        attn_map_resized = np.array(Image.fromarray(attn_uint8).resize((128, 384), Image.NEAREST)) / 255.0
        img_np = np.array(img)
        img_resized = img.resize((128, 384), Image.BILINEAR)
        img_np = np.array(img_resized)

        ax.imshow(img)
        ax.imshow(attn_map_resized, cmap='hot', alpha=0.6)
        ax.axis("off")
        ax.set_title(f"Rollout {int(keep_ratio*100)}%")

        # save combined figure
        save_path = os.path.join(save_dir, f"{filename}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print("Saved:", save_path)


@dataclass
class QueryExample:
    query_index: int
    pid: int
    image_index: int
    caption_id_within_image: int
    image_path: str
    caption: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visual Grad-CAM comparison for all captions in a test split.")
    parser.add_argument("--dataset-name", choices=sorted(DATASETS), default="RSTPReid")
    parser.add_argument("--root-dir", default="data", help="Dataset root or parent folder containing the dataset.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to visualize.")
    parser.add_argument("--baseline-checkpoint", required=True, help="Path to baseline checkpoint.")
    parser.add_argument("--best-checkpoint", required=True, help="Path to best/ours checkpoint.")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--best-name", default="best")
    parser.add_argument("--output-dir", default="outputs/v_gradcam_compare")
    parser.add_argument("--pretrain-choice", default="ViT-B/16")
    parser.add_argument("--img-size", nargs=2, type=int, default=[384, 128], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--stride-size", type=int, default=16)
    parser.add_argument("--text-length", type=int, default=77)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-captions", type=int, default=0, help="Optional cap for debugging. Use 0 for all captions.")
    parser.add_argument("--start-index", type=int, default=0, help="Start query index after flattening captions.")
    parser.add_argument("--overlay-alpha", type=float, default=0.48)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--metadata-jsonl", default="metadata.jsonl")
    parser.add_argument("--grid", action="store_true", help="Also write paged grid images containing all comparisons.")
    parser.add_argument("--grid-rows-per-page", type=int, default=4, help="Number of comparison rows per grid page.")
    parser.add_argument("--grid-dir-name", default="grid_pages", help="Subdirectory name for paged grid images.")
    return parser.parse_args()


def resolve_path(path):
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def dataset_dir_from_root(dataset_name, root_dir):
    config = DATASETS[dataset_name]
    root = resolve_path(root_dir)
    dataset_dir_name = config["dataset_dir"]
    if root.name.lower() == dataset_dir_name.lower():
        return root
    return root / dataset_dir_name


def annotation_pid(dataset_name, split, anno):
    pid = int(anno["id"])
    if dataset_name == "CUHK-PEDES" and split == "train":
        pid -= 1
    return pid


def annotation_image_path(dataset_dir, image_dir, anno, path_key):
    rel_path = anno.get(path_key)
    if rel_path is None:
        raise KeyError(f"Annotation has no path key: {path_key}")
    candidate = Path(str(rel_path)).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    image_candidate = image_dir / candidate
    if image_candidate.exists():
        return str(image_candidate.resolve())
    dataset_candidate = dataset_dir / candidate
    if dataset_candidate.exists():
        return str(dataset_candidate.resolve())
    return str(image_candidate.resolve())


def load_query_examples(args):
    config = DATASETS[args.dataset_name]
    dataset_dir = dataset_dir_from_root(args.dataset_name, args.root_dir)
    image_dir = dataset_dir / config["image_dir"]
    annotation_path = dataset_dir / config["annotation_file"]

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8") as file:
        annotations = json.load(file)

    train_pids = {
        annotation_pid(args.dataset_name, "train", anno)
        for anno in annotations
        if anno.get("split") == "train"
    }
    split_annotations = []
    for anno in annotations:
        anno_split = anno.get("split")
        if args.split == "val":
            if anno_split not in ("train", "test"):
                split_annotations.append(anno)
        elif anno_split == args.split:
            split_annotations.append(anno)

    examples = []
    query_index = 0
    for image_index, anno in enumerate(split_annotations):
        pid = annotation_pid(args.dataset_name, args.split, anno)
        image_path = annotation_image_path(dataset_dir, image_dir, anno, config["path_key"])
        for caption_id, caption in enumerate(anno["captions"]):
            if query_index >= args.start_index:
                examples.append(QueryExample(
                    query_index=query_index,
                    pid=pid,
                    image_index=image_index,
                    caption_id_within_image=caption_id,
                    image_path=image_path,
                    caption=str(caption),
                ))
                if args.max_captions > 0 and len(examples) >= args.max_captions:
                    return examples, len(train_pids)
            query_index += 1

    return examples, len(train_pids)


def build_global_clip_model(args):
    model, _ = build_CLIP_from_openai_pretrained(
        args.pretrain_choice,
        tuple(args.img_size),
        args.stride_size,
    )
    return model


def torch_load_checkpoint(checkpoint_path):
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, Mapping):
        for key in ("state_dict", "model", "model_state_dict", "net", "network", "module"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return value
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping or contain a model state mapping.")
    return checkpoint


def strip_repeated_prefixes(key, prefixes):
    stripped = key
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                changed = True
    return stripped


CLIP_BACKBONE_EXACT_KEYS = {"positional_embedding", "text_projection"}
CLIP_BACKBONE_PREFIXES = (
    "visual.",
    "transformer.",
    "token_embedding.",
    "ln_final.",
)


def normalized_clip_key(key):
    normalized = strip_repeated_prefixes(str(key), ("module.", "model.", "net.", "network."))
    if normalized.startswith("base_model."):
        normalized = normalized[len("base_model."):]
    return normalized


def is_clip_backbone_checkpoint_key(key):
    normalized = normalized_clip_key(key)
    return normalized in CLIP_BACKBONE_EXACT_KEYS or normalized.startswith(CLIP_BACKBONE_PREFIXES)


def candidate_state_keys(key):
    raw = str(key)
    candidates = []

    def add(candidate):
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
    add(normalized_clip_key(raw))
    return candidates

def load_checkpoint_for_inference(model, checkpoint_path):
    checkpoint = torch_load_checkpoint(checkpoint_path)
    loaded_state = checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    update_state = {}
    skipped_missing = 0
    skipped_shape = 0
    skipped_non_tensor = 0
    ignored_non_clip = 0

    for raw_key, value in loaded_state.items():
        if not torch.is_tensor(value):
            skipped_non_tensor += 1
            continue

        if not is_clip_backbone_checkpoint_key(str(raw_key)):
            ignored_non_clip += 1
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
        "ignored_non_clip": ignored_non_clip,
    }


def load_model_for_checkpoint(args, checkpoint_path, device, label):
    checkpoint = resolve_path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint}")
    model = build_global_clip_model(args)
    stats = load_checkpoint_for_inference(model, checkpoint)
    print(
        f"[{label}] Loaded global CLIP tensors: {stats['loaded']} "
        f"(ignored_non_clip={stats['ignored_non_clip']}, missing={stats['skipped_missing']}, "
        f"shape={stats['skipped_shape']}, non_tensor={stats['skipped_non_tensor']})"
    )
    model.to(device)
    model.float()
    model.eval()
    return model


def read_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    return Image.open(image_path).convert("RGB")


def tokenize_caption(caption, text_length):
    return dataset_tokenize(caption, tokenizer=TEXT_TOKENIZER, text_length=text_length, truncate=True)


class VisualGradCam:
    def __init__(self, model, img_size):
        self.model = model
        self.img_size = img_size
        self.visual = model.visual
        # Patch tokens at the final block output no longer affect the CLS image feature.
        # Hook the pre-attention normalization instead, where CLS still attends to patches.
        self.target = self.visual.transformer.resblocks[-1].ln_1
        self.activations = None
        self.handle = self.target.register_forward_hook(self._forward_hook)

    def close(self):
        self.handle.remove()

    def _forward_hook(self, _module, _inputs, output):
        tensor = output[0] if isinstance(output, (list, tuple)) else output
        tensor.retain_grad()
        self.activations = tensor

    def __call__(self, image_tensor, text_tokens):
        self.model.zero_grad(set_to_none=True)
        self.activations = None

        image_batch = image_tensor.unsqueeze(0)
        text_batch = text_tokens.unsqueeze(0)

        with torch.no_grad():
            text_tokens_long = text_batch.long()
            text_all, _ = self.model.encode_text(text_tokens_long)
            text_feat = text_all[torch.arange(text_all.shape[0], device=text_all.device), text_tokens_long.argmax(dim=-1)].float()
            text_feat = F.normalize(text_feat, p=2, dim=1).detach()

        image_all, _ = self.model.encode_image(image_batch)
        image_feat = image_all[:, 0, :].float()
        image_feat = F.normalize(image_feat, p=2, dim=1)
        score = (image_feat * text_feat).sum()
        score.backward()

        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Could not capture visual transformer activations/gradients for Grad-CAM.")

        activations = self.activations.detach()[:, 0, :].float()
        gradients = self.activations.grad.detach()[:, 0, :].float()
        patch_activations = activations[1:]
        patch_gradients = gradients[1:]
        weights = patch_gradients.mean(dim=0)
        signed_cam = (patch_activations * weights).sum(dim=1)
        cam = torch.relu(signed_cam)

        if (not torch.isfinite(cam).all()) or float(cam.max().detach().cpu()) <= 1e-12:
            cam = signed_cam.abs()
        if (not torch.isfinite(cam).all()) or float(cam.max().detach().cpu()) <= 1e-12:
            cam = patch_gradients.norm(p=2, dim=1)
        num_y = int(getattr(self.visual, "num_y", self.img_size[0] // 16))
        num_x = int(getattr(self.visual, "num_x", self.img_size[1] // 16))
        if cam.numel() != num_y * num_x:
            raise RuntimeError(f"Grad-CAM patch count mismatch: {cam.numel()} vs {num_y}x{num_x}.")

        cam = cam.reshape(num_y, num_x)
        cam = cam - cam.min()
        max_value = cam.max()
        if max_value > 0:
            cam = cam / max_value

        height, width = self.img_size
        cam_np = cam.cpu().numpy()
        cam_uint8 = heatmap_to_uint8(cam_np)
        return np.array(Image.fromarray(cam_uint8).resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0


def safe_filename(text, max_len=80):
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")
    return cleaned[:max_len] or "item"


def heatmap_to_uint8(heatmap):
    heatmap = np.asarray(heatmap, dtype=np.float32)
    heatmap = heatmap - float(heatmap.min())
    max_value = float(heatmap.max())
    if max_value > 0:
        heatmap = heatmap / max_value
    return np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)


def overlay_heatmap(image, heatmap_uint8, alpha, img_size):
    height, width = img_size
    image_np = np.array(image.resize((width, height), Image.BILINEAR)).astype(np.float32)
    heatmap = heatmap_uint8.astype(np.float32) / 255.0
    color = plt.get_cmap("jet")(heatmap)[..., :3] * 255.0
    alpha_map = (alpha * heatmap)[..., None]
    overlay = (1.0 - alpha_map) * image_np + alpha_map * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_comparison_panels(example, baseline_heatmap, best_heatmap, args, image=None):
    height, width = tuple(args.img_size)
    image = image if image is not None else read_image(example.image_path)
    original = np.array(image.resize((width, height), Image.BILINEAR))
    baseline_overlay = overlay_heatmap(image, baseline_heatmap, args.overlay_alpha, (height, width))
    best_overlay = overlay_heatmap(image, best_heatmap, args.overlay_alpha, (height, width))
    return [
        ("image", original),
        (args.baseline_name, baseline_overlay),
        (args.best_name, best_overlay),
    ]


def save_comparison_figure(example, panels, args, comparison_dir):
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 6.2))
    for ax, (title, panel) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    caption = textwrap.fill(example.caption, width=92)
    fig.suptitle(
        f"q={example.query_index} pid={example.pid} image={Path(example.image_path).name}\n{caption}",
        fontsize=9,
        y=0.995,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])

    filename = f"q{example.query_index:06d}_pid{example.pid}_img{example.image_index:05d}_c{example.caption_id_within_image:02d}_{safe_filename(Path(example.image_path).stem)}.png"
    output_path = comparison_dir / filename
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def grid_page_path(grid_dir, page_index):
    return grid_dir / f"grid_page_{page_index:04d}.png"


def save_comparison_grid_page(entries, page_index, page_count, args, grid_dir):
    rows = len(entries)
    fig_width = 11.8
    fig_height = max(3.15 * rows, 3.8)
    fig, axes = plt.subplots(
        rows,
        4,
        figsize=(fig_width, fig_height),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.55, 1.0, 1.0, 1.0]},
    )

    headers = ["caption", "image", args.baseline_name, args.best_name]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header, fontsize=10, pad=8)

    for row_index, (example, panels) in enumerate(entries):
        text_ax = axes[row_index, 0]
        text_ax.axis("off")
        caption = textwrap.fill(example.caption, width=36)
        text_ax.text(
            0.0,
            0.5,
            f"q={example.query_index}  pid={example.pid}\n{Path(example.image_path).name}\n{caption}",
            ha="left",
            va="center",
            fontsize=7.5,
            transform=text_ax.transAxes,
        )
        for col_index, (_title, panel) in enumerate(panels, start=1):
            ax = axes[row_index, col_index]
            ax.imshow(panel)
            ax.axis("off")

    fig.suptitle(f"Grad-CAM comparison page {page_index}/{page_count}", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.975])
    output_path = grid_page_path(grid_dir, page_index)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_heatmaps_for_checkpoint(model, examples, args, device, desc):
    transform = build_transforms(img_size=tuple(args.img_size), is_train=False)
    cam = VisualGradCam(model, img_size=tuple(args.img_size))
    heatmaps = []
    try:
        for example in tqdm(examples, desc=desc, unit="caption"):
            image = read_image(example.image_path)
            image_tensor = transform(image).to(device)
            text_tokens = tokenize_caption(example.caption, args.text_length).to(device)
            heatmap = cam(image_tensor, text_tokens)
            heatmaps.append(heatmap_to_uint8(heatmap))
    finally:
        cam.close()
    return heatmaps


def write_metadata_row(file, example, figure_path):
    file.write(json.dumps({
        "query_index": example.query_index,
        "pid": example.pid,
        "image_index": example.image_index,
        "caption_id_within_image": example.caption_id_within_image,
        "image_path": example.image_path,
        "caption": example.caption,
        "figure_path": str(figure_path),
    }, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if args.max_captions < 0:
        raise ValueError("--max-captions must be >= 0.")
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if not (0.0 <= float(args.overlay_alpha) <= 1.0):
        raise ValueError("--overlay-alpha must be in [0, 1].")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")

    output_dir = resolve_path(args.output_dir)
    comparison_dir = output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / args.metadata_jsonl

    examples, num_classes = load_query_examples(args)
    if not examples:
        raise RuntimeError(f"No captions found for {args.dataset_name} split={args.split}.")

    print(
        f"[Dataset] {args.dataset_name} split={args.split} captions={len(examples)} "
        f"num_train_ids={num_classes}"
    )

    baseline_model = load_model_for_checkpoint(args, args.baseline_checkpoint, device, args.baseline_name)
    baseline_heatmaps = compute_heatmaps_for_checkpoint(
        baseline_model,
        examples,
        args,
        device,
        desc=f"Grad-CAM {args.baseline_name}",
    )
    del baseline_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_model = load_model_for_checkpoint(args, args.best_checkpoint, device, args.best_name)
    transform = build_transforms(img_size=tuple(args.img_size), is_train=False)
    best_cam = VisualGradCam(best_model, img_size=tuple(args.img_size))

    plt.ioff()
    grid_dir = output_dir / args.grid_dir_name
    grid_rows = int(args.grid_rows_per_page)
    grid_page_count = 0
    grid_entries = []
    if args.grid:
        if grid_rows <= 0:
            raise ValueError("--grid-rows-per-page must be > 0.")
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_page_count = (len(examples) + grid_rows - 1) // grid_rows

    written_grid_pages = []
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        try:
            page_index = 1
            for example, baseline_heatmap in tqdm(
                zip(examples, baseline_heatmaps),
                total=len(examples),
                desc=f"Rendering comparison {args.best_name}",
                unit="caption",
            ):
                image = read_image(example.image_path)
                image_tensor = transform(image).to(device)
                text_tokens = tokenize_caption(example.caption, args.text_length).to(device)
                best_heatmap = heatmap_to_uint8(best_cam(image_tensor, text_tokens))
                panels = make_comparison_panels(example, baseline_heatmap, best_heatmap, args, image=image)
                figure_path = save_comparison_figure(example, panels, args, comparison_dir)
                write_metadata_row(metadata_file, example, figure_path)

                if args.grid:
                    grid_entries.append((example, panels))
                    if len(grid_entries) == grid_rows:
                        written_grid_pages.append(
                            save_comparison_grid_page(grid_entries, page_index, grid_page_count, args, grid_dir)
                        )
                        page_index += 1
                        grid_entries = []

            if args.grid and grid_entries:
                written_grid_pages.append(
                    save_comparison_grid_page(grid_entries, page_index, grid_page_count, args, grid_dir)
                )
        finally:
            best_cam.close()
            del best_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print(f"[Done] wrote {len(examples)} comparison figures to {comparison_dir}")
    if args.grid:
        print(f"[Done] wrote {len(written_grid_pages)} grid pages to {grid_dir}")
    print(f"[Done] wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
