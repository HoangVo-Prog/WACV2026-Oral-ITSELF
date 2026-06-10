import copy
import os
from gradcam import GradCam
import cv2
import torchvision.transforms as T
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os.path as op
from datasets import build_dataloader
from processor.processor import do_inference1
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
import argparse
from utils.iotools import load_train_configs
import torch

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

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

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
        
        attn_map_resized = cv2.resize(attn_map_selective, (128, 384), interpolation=cv2.INTER_NEAREST)
        img_np = np.array(img)
        img_resized = img.resize((384, 128), Image.BILINEAR)   # W=384, H=128
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TranTextReID Text")
    # sub = '/home/s48gb/Desktop/GenAI4E/kltn/LPNC_WACV/PPL_logs1/RSTPReid/baseline_10e_{warmup5e}' # basseline RSTP
    # sub = '/home/s48gb/Desktop/GenAI4E/kltn/LPNC_WACV/PPL_logs1/RSTPReid/ablation_study/main/xyy' # RSTP ours
    # sub = '/home/s48gb/Desktop/GenAI4E/kltn/LPNC_WACV/PPL_logs1/CUHK-PEDES/ablation_study/xxx' # baseline CUHK
    # sub = '/home/s48gb/Desktop/GenAI4E/kltn/LPNC_WACV/PPL_logs1/CUHK-PEDES/ablation_study/xyy' # CUHK ours
    # sub = '/home/s48gb/Desktop/GenAI4E/kltn/LPNC_WACV/PPL_logs1/ICFG-PEDES/ablation_study/xxx' # baseline ICFG
    sub = '/home/s48gb/Desktop/GenAI4E/kltn/LPNC_WACV/PPL_logs1/ICFG-PEDES/ablation_study/xyy' # ICFG ours
    # print('Ours ICFG')
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    logger = setup_logger('PPL', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    args.output_dir = sub
    
    ### Source
    args.test_batch_size = 1
    test_img_loader, test_txt_loader,refer, num_classes = build_dataloader(args)  
    save_dir = "visualize/grad_cam_ICFG_ours_test"  
    
    asss = ['best.pth']
    for i in range(len(asss)):
        print(i)
        if os.path.exists(op.join(args.output_dir, asss[i])):
            model = build_model(args,num_classes)
            checkpointer = Checkpointer(model)
            checkpointer.load(f=op.join(args.output_dir, asss[i]))
            model = model.cuda()
            model = model.float()
            
            cam = GradCam(model=model.base_model.visual, target=model.base_model.visual.transformer.resblocks[-1])
            img_dataset = test_img_loader.test_img_set
            dataset_length = len(img_dataset)
            print(dataset_length)
            bz = 1
            for idx, (pid, img) in enumerate(test_img_loader):
                # get image_paths
                if ((idx+1)*bz) > dataset_length:
                    image_paths = img_dataset.img_paths[(idx*bz):]
                else:
                    image_paths = img_dataset.img_paths[(idx*bz):((idx+1)*bz)]   # lấy theo index
                img = img.cuda()
                
                for tid,caption in test_txt_loader:
                    if pid == tid:
                        caption = caption.cuda()
                        attn_map = cam(img, caption, model)
                        
                        img_np = img[0].detach().cpu().permute(1, 2, 0).numpy()
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
                        
                        attn_map = attn_map / (attn_map.max() + 1e-8)
                        attn_map = np.array(Image.fromarray((attn_map*255).astype(np.uint8)).resize((128, 384), resample=Image.BILINEAR)) / 255.0

                        fig, ax = plt.subplots()
                        ax.imshow(img_np)
                        ax.imshow(attn_map, cmap='jet', alpha=0.5)
                        ax.axis('off')
                    
                        os.makedirs(save_dir, exist_ok=True)
                        filename = os.path.basename(image_paths[0]).split('.')[0] + ".png"
                        save_path = os.path.join(save_dir, filename)
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
                        plt.close(fig)
                        print("Saved:", save_path)
                        break