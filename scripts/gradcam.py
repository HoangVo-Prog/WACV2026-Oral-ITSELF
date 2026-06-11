import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class GradCam:
    """Text-conditioned Grad-CAM for this repo's CLIP/ITSELF visual encoder.

    The jacobgil/pytorch-grad-cam implementation stores activations and gradients
    from a target layer, averages gradients spatially, and weights activations by
    those averages. This local version keeps that flow but adapts it to this repo:
    visual transformer blocks return ``[tokens, attention]`` and their tokens are
    ``[num_tokens, batch, channels]`` instead of the usual batch-first layout.
    """

    def __init__(
        self,
        model,
        target_layer=None,
        target=None,
        reshape_transform=None,
        score_mode: str = "auto",
        global_weight: float = 0.68,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_extra_tokens: int = 1,
    ):
        self.model = model.eval()
        self.target_layer = target_layer if target_layer is not None else target
        if self.target_layer is None:
            raise ValueError("GradCam requires a target_layer, for example model.base_model.visual.transformer.resblocks[-1].")

        self.reshape_transform_fn = reshape_transform
        self.score_mode = score_mode
        self.global_weight = float(global_weight)
        self.num_extra_tokens = int(num_extra_tokens)

        visual = self._find_visual_module(model)
        self.height = height if height is not None else getattr(visual, "num_y", None)
        self.width = width if width is not None else getattr(visual, "num_x", None)

        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    @staticmethod
    def _find_visual_module(model):
        base_model = getattr(model, "base_model", None)
        if base_model is not None and hasattr(base_model, "visual"):
            return base_model.visual
        if hasattr(model, "visual"):
            return model.visual
        return model

    @staticmethod
    def _first_tensor(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (list, tuple)):
            for item in output:
                tensor = GradCam._first_tensor(item)
                if tensor is not None:
                    return tensor
        return None

    def _register_hooks(self):
        self.handles.append(self.target_layer.register_forward_hook(self._save_activation))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _save_activation(self, module, inputs, output):
        activation = self._first_tensor(output)
        if activation is None:
            raise RuntimeError("The target layer did not return a tensor that Grad-CAM can use.")

        self.activations = self.reshape_transform(activation.detach())
        if activation.requires_grad:
            activation.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = self.reshape_transform(grad.detach())

    def _grid_size(self, patch_count: int) -> Tuple[int, int]:
        if self.height is not None and self.width is not None:
            height = int(self.height)
            width = int(self.width)
            if height * width == patch_count:
                return height, width

        side = int(math.sqrt(patch_count))
        if side * side == patch_count:
            return side, side

        raise ValueError(
            "Cannot infer a ViT token grid for {} patch tokens. Pass height and width to GradCam.".format(patch_count)
        )

    def reshape_transform(self, tensor):
        if self.reshape_transform_fn is not None:
            return self.reshape_transform_fn(tensor)

        if not torch.is_tensor(tensor):
            tensor = self._first_tensor(tensor)
        if tensor is None or tensor.ndim != 3:
            raise ValueError("Expected a 3D ViT token tensor from the target layer.")

        expected_tokens = None
        if self.height is not None and self.width is not None:
            expected_tokens = int(self.height) * int(self.width) + self.num_extra_tokens

        # This repo's transformer blocks emit [tokens, batch, channels]. The
        # common Grad-CAM ViT examples use [batch, tokens, channels].
        if expected_tokens is not None and tensor.shape[0] == expected_tokens:
            tensor = tensor.permute(1, 0, 2)
        elif expected_tokens is not None and tensor.shape[1] == expected_tokens:
            pass
        elif tensor.shape[0] > tensor.shape[1]:
            tensor = tensor.permute(1, 0, 2)

        start = self.num_extra_tokens
        patch_count = tensor.shape[1] - start
        if patch_count <= 0:
            raise ValueError("Target layer output has no patch tokens after removing special tokens.")

        height, width = self._grid_size(patch_count)
        tokens = tensor[:, start:, :]
        tokens = tokens.reshape(tokens.shape[0], height, width, tokens.shape[-1])
        return tokens.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _cosine_score(image_features, text_features):
        image_features = F.normalize(image_features.float(), p=2, dim=-1)
        text_features = F.normalize(text_features.float(), p=2, dim=-1)
        return (image_features * text_features).sum(dim=-1)

    @staticmethod
    def _has_grab(model):
        args = getattr(model, "args", None)
        only_global = bool(getattr(args, "only_global", False))
        return (not only_global) and hasattr(model, "encode_image_grab") and hasattr(model, "encode_text_grab")

    def _score_global(self, model, images, captions):
        image_features = model.encode_image(images)
        with torch.no_grad():
            text_features = model.encode_text(captions.long())
        return self._cosine_score(image_features, text_features)

    def _score_grab(self, model, images, captions):
        image_features = model.encode_image_grab(images)
        with torch.no_grad():
            text_features = model.encode_text_grab(captions.long())
        return self._cosine_score(image_features, text_features)

    def _score_global_grab(self, model, images, captions):
        if hasattr(model, "_compute_host_embeddings"):
            features = model._compute_host_embeddings(images, captions.long())
            global_score = self._cosine_score(features["i_feats"], features["t_feats"].detach())
            if "i_grab_f" not in features or "t_grab_f" not in features:
                return global_score
            grab_score = self._cosine_score(features["i_grab_f"], features["t_grab_f"].detach())
        else:
            global_score = self._score_global(model, images, captions)
            if not self._has_grab(model):
                return global_score
            grab_score = self._score_grab(model, images, captions)
        return self.global_weight * global_score + (1.0 - self.global_weight) * grab_score

    def _target_score(self, images, captions, model):
        if captions is None:
            raise ValueError("Text-conditioned Grad-CAM requires caption token ids.")

        mode = self.score_mode.lower()
        if mode == "auto":
            mode = "grab" if self._has_grab(model) else "global"

        if mode == "global":
            return self._score_global(model, images, captions)
        if mode == "grab":
            if not self._has_grab(model):
                return self._score_global(model, images, captions)
            return self._score_grab(model, images, captions)
        if mode in {"global+grab", "mixed", "combined"}:
            return self._score_global_grab(model, images, captions)

        raise ValueError("Unsupported score_mode: {}".format(self.score_mode))

    @staticmethod
    def _normalize_cam(cam):
        cam = cam - cam.flatten(1).min(dim=1)[0].view(-1, 1, 1)
        denom = cam.flatten(1).max(dim=1)[0].view(-1, 1, 1).clamp_min(1e-8)
        return cam / denom

    def __call__(self, images, captions=None, model=None):
        model = self.model if model is None else model.eval()
        self.activations = None
        self.gradients = None

        try:
            model.zero_grad(set_to_none=True)
        except TypeError:
            model.zero_grad()

        with torch.enable_grad():
            scores = self._target_score(images, captions, model)
            if scores.ndim == 0:
                target_score = scores
            else:
                target_score = scores.sum()
            target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM target layer was not reached. Make sure target_layer belongs to the same model used for the forward pass."
            )
        if self.gradients is None:
            raise RuntimeError("Grad-CAM gradients were not captured. Check that the target score depends on the target layer.")

        activations = self.activations.float()
        gradients = self.gradients.float()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)
        cam = self._normalize_cam(cam[:, 0])

        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        return cam_np[0] if cam_np.shape[0] == 1 else cam_np

    def __del__(self):
        self.remove_hooks()
