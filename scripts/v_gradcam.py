from __future__ import annotations

import argparse
import os
import os.path as op

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def compute_rollout(
    attentions: "torch.Tensor",
    head_fusion="mean",
    discard: bool = True,
    discard_ratios=None,
    start_layer: int = 4,
    skip_layer=None,
):
    """Compute rollout attention with optional low-attention pruning."""
    import torch

    if discard_ratios is None:
        discard_ratios = [0.25, 1.0, 1.0, 1.0, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 0.25, 0.25]
    if skip_layer is None:
        skip_layer = [5, 6, 7, 8, 9, 10]

    if len(attentions.shape) == 5:
        layers, batch_size, _, num_tokens, _ = attentions.shape
    else:
        layers, batch_size, num_tokens, _ = attentions.shape

    device = attentions.device
    result = torch.eye(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)

    for layer in range(start_layer, layers):
        if layer in skip_layer:
            continue

        attn = attentions[layer]
        if len(attentions.shape) == 5:
            with torch.no_grad():
                if head_fusion == "mean":
                    attn = attn.mean(axis=1)
                elif head_fusion == "max":
                    attn = attn.max(axis=1)[0]
                elif head_fusion == "min":
                    attn = attn.min(axis=1)[0]
                else:
                    raise ValueError("Attention head fusion type is not supported")

        if discard:
            discard_ratio = discard_ratios[layer]
            flat = attn.view(batch_size, -1)
            num_to_discard = int(flat.size(-1) * discard_ratio)
            if num_to_discard > 0:
                _, indices = flat.topk(num_to_discard, dim=-1, largest=False)
                for batch_idx in range(batch_size):
                    idx = indices[batch_idx]
                    idx = idx[idx != 0]
                    flat[batch_idx, idx] = 0
                attn = flat.view(batch_size, num_tokens, num_tokens)

        identity = torch.eye(num_tokens, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        attn = (attn + identity) / 2.0
        attn = attn / attn.sum(dim=-1, keepdim=True)
        result = torch.bmm(attn, result)

    return result


def visualize_rollout_attention(image_paths, attention_map, save_dir):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for rollout attention visualization.") from exc

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    for batch_idx, img_path in enumerate(image_paths):
        attn_map = attention_map[batch_idx, 0]
        filename = os.path.splitext(os.path.basename(img_path))[0]

        img = Image.open(img_path).convert("RGB")
        attn_map = attn_map[1:].reshape(24, 8).detach().cpu().numpy()

        keep_ratio = 0.4
        flat = attn_map.flatten()
        k = max(1, int(len(flat) * keep_ratio))
        thresh = np.partition(flat, -k)[-k]
        mask = (attn_map >= thresh).astype(np.float32)

        attn_map_selective = attn_map * mask
        if attn_map_selective.max() > 0:
            attn_map_selective = attn_map_selective / attn_map_selective.max()

        attn_map_resized = cv2.resize(attn_map_selective, (128, 384), interpolation=cv2.INTER_NEAREST)
        img_resized = img.resize((128, 384), Image.BILINEAR)

        fig, ax = plt.subplots()
        ax.imshow(np.array(img_resized))
        ax.imshow(attn_map_resized, cmap="hot", alpha=0.6)
        ax.axis("off")
        ax.set_title("Rollout {}%".format(int(keep_ratio * 100)))

        save_path = os.path.join(save_dir, "{}.png".format(filename))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        print("Saved:", save_path)


def parse_args():
    sub = "/home/s48gb/Desktop/GenAI4E/kltn/LPNC_WACV/PPL_logs1/ICFG-PEDES/ablation_study/xyy"
    parser = argparse.ArgumentParser(description="Generate text-conditioned Grad-CAM maps for ITSELF.")
    parser.add_argument("--config_file", default=f"{sub}/configs.yaml")
    parser.add_argument("--run_dir", default=None, help="Directory containing the checkpoint. Defaults to config output_dir.")
    parser.add_argument("--checkpoint", default="best.pth")
    parser.add_argument("--save_dir", default="visualize/grad_cam_ICFG_ours_test")
    parser.add_argument("--score_mode", default="auto", choices=["auto", "global", "grab", "global+grab"])
    return parser.parse_args()


def render_gradcam_overlay(img, attn_map, save_path):
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    attn_map = attn_map / (attn_map.max() + 1e-8)
    attn_map = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize((128, 384), resample=Image.BILINEAR)
    ) / 255.0

    fig, ax = plt.subplots()
    ax.imshow(img_np)
    ax.imshow(attn_map, cmap="jet", alpha=0.5)
    ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print("Saved:", save_path)


def main():
    cli_args = parse_args()

    import torch

    from datasets import build_dataloader
    from gradcam import GradCam
    from model import build_model
    from utils.checkpoint import Checkpointer
    from utils.iotools import load_train_configs
    from utils.logger import setup_logger

    args = load_train_configs(cli_args.config_file)
    args.training = False
    args.test_batch_size = 1
    args.output_dir = cli_args.run_dir or getattr(args, "output_dir", None) or op.dirname(cli_args.config_file)

    logger = setup_logger("PPL", save_dir=args.output_dir, if_train=args.training)
    logger.info(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)

    ckpt_path = op.join(args.output_dir, cli_args.checkpoint)
    if not op.exists(ckpt_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(ckpt_path))

    model = build_model(args, num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=ckpt_path)
    model = model.to(device).float().eval()

    cam = GradCam(
        model=model,
        target_layer=model.base_model.visual.transformer.resblocks[-1],
        score_mode=cli_args.score_mode,
    )

    img_dataset = test_img_loader.test_img_set
    dataset_length = len(img_dataset)
    print(dataset_length)

    try:
        for idx, (pid, img) in enumerate(test_img_loader):
            batch_size = img.shape[0]
            start = idx * batch_size
            end = min(start + batch_size, dataset_length)
            image_paths = img_dataset.img_paths[start:end]
            img = img.to(device)

            for tid, caption in test_txt_loader:
                if not torch.equal(pid.view(-1).cpu(), tid.view(-1).cpu()):
                    continue

                caption = caption.to(device)
                attn_map = cam(img, caption)
                filename = os.path.splitext(os.path.basename(image_paths[0]))[0] + ".png"
                save_path = os.path.join(cli_args.save_dir, filename)
                render_gradcam_overlay(img, attn_map, save_path)
                break
    finally:
        cam.remove_hooks()


if __name__ == "__main__":
    main()
