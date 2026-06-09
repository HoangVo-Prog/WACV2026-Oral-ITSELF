import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import symmetric_identity_proxy_loss
from .memory import PrototypeMemory


class IdentityProjector(nn.Module):
    def forward(self, features):
        return features.float()


class ResidualIdentityProjector(nn.Module):
    def __init__(self, dim, scale=0.1):
        super().__init__()
        self.scale = float(scale)
        self.residual = nn.Linear(dim, dim)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features):
        features = features.float()
        self.residual.float()
        return features + self.scale * self.residual(features)


def _linear_projector(feature_dim, prototype_dim, orthogonal=False, zero_bias=False):
    projector = nn.Linear(feature_dim, prototype_dim)
    if orthogonal:
        nn.init.orthogonal_(projector.weight)
        nn.init.zeros_(projector.bias)
    elif zero_bias:
        nn.init.zeros_(projector.bias)
    return projector


def _first_device(module, fallback):
    for param in module.parameters(recurse=True):
        return param.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return fallback


def _complete_orthonormal_rows(rows, out_dim, feature_dim):
    rows = F.normalize(rows.float(), p=2, dim=1) if rows.numel() > 0 else rows.float()
    completed = [row for row in rows]
    eye = torch.eye(feature_dim, dtype=torch.float32, device=rows.device)

    for candidate in eye:
        vector = candidate.clone()
        for row in completed:
            vector = vector - torch.dot(vector, row) * row
        norm = vector.norm()
        if norm > 1e-6:
            completed.append(vector / norm)
        if len(completed) == out_dim:
            break

    if len(completed) < out_dim:
        raise ValueError("could not build enough PCA projector components")
    return torch.stack(completed[:out_dim], dim=0)


def _pca_components(features, out_dim):
    if features.ndim != 2:
        raise ValueError("PCA projector initialization expects a 2D feature tensor")
    feature_dim = features.shape[1]
    if out_dim > feature_dim:
        raise ValueError(
            f"prototype_dim ({out_dim}) must be <= feature_dim ({feature_dim}) for PCA projector initialization"
        )

    features = features.detach().float().cpu()
    centered = features - features.mean(dim=0, keepdim=True)
    if centered.norm() <= 1e-12:
        return torch.eye(feature_dim, dtype=torch.float32)[:out_dim]

    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rows = vh[: min(out_dim, vh.shape[0])].contiguous()
    if rows.shape[0] < out_dim:
        rows = _complete_orthonormal_rows(rows, out_dim, feature_dim)
    return rows[:out_dim]


class PrototypeBranch(nn.Module):
    def __init__(self, args, num_classes, feature_dim):
        super().__init__()
        self.args = args
        self.feature_dim = feature_dim
        self.prototype_dim = getattr(args, "prototype_dim", 512)
        self.projector_mode = getattr(args, "prototype_projector", "default")
        self._pca_initialized = False
        prototype_feature = getattr(args, "prototype_feature", "auto")
        self.use_local = prototype_feature == "auto" and not args.only_global
        self.use_local = self.use_local or (prototype_feature == "local" and not args.only_global)

        self.image_projector, self.text_projector = self._build_projectors()
        self.memory = PrototypeMemory(
            num_classes=num_classes,
            prototypes_per_id=getattr(args, "prototype_per_id", 2),
            dim=self.prototype_dim,
            momentum=getattr(args, "prototype_momentum", 0.2),
        )

    def _require_matching_dims(self):
        if self.feature_dim != self.prototype_dim:
            raise ValueError(
                f"--prototype_projector {self.projector_mode} requires feature_dim == prototype_dim, "
                f"got feature_dim={self.feature_dim}, prototype_dim={self.prototype_dim}"
            )

    def _require_pca_dims(self):
        if self.prototype_dim > self.feature_dim:
            raise ValueError(
                f"--prototype_projector {self.projector_mode} requires prototype_dim <= feature_dim, "
                f"got prototype_dim={self.prototype_dim}, feature_dim={self.feature_dim}"
            )

    def _build_projectors(self):
        mode = self.projector_mode
        if mode == "default":
            return (
                nn.Sequential(nn.Linear(self.feature_dim, self.prototype_dim), nn.LayerNorm(self.prototype_dim)),
                nn.Sequential(nn.Linear(self.feature_dim, self.prototype_dim), nn.LayerNorm(self.prototype_dim)),
            )
        if mode == "identity":
            self._require_matching_dims()
            return IdentityProjector(), IdentityProjector()
        if mode == "residual_identity":
            self._require_matching_dims()
            scale = getattr(self.args, "prototype_residual_scale", 0.1)
            return ResidualIdentityProjector(self.feature_dim, scale), ResidualIdentityProjector(self.feature_dim, scale)
        if mode == "random_orthogonal":
            return (
                _linear_projector(self.feature_dim, self.prototype_dim, orthogonal=True),
                _linear_projector(self.feature_dim, self.prototype_dim, orthogonal=True),
            )
        if mode == "pca_init":
            self._require_pca_dims()
            return (
                _linear_projector(self.feature_dim, self.prototype_dim, zero_bias=True),
                _linear_projector(self.feature_dim, self.prototype_dim, zero_bias=True),
            )
        if mode == "shared":
            shared = _linear_projector(self.feature_dim, self.prototype_dim)
            return shared, shared
        if mode == "shared_pca_init":
            self._require_pca_dims()
            shared = _linear_projector(self.feature_dim, self.prototype_dim, zero_bias=True)
            return shared, shared
        raise ValueError(f"Unknown prototype projector mode: {mode}")

    def is_ready(self):
        return self.memory.is_ready()

    def needs_pca_init(self):
        return self.projector_mode in ("pca_init", "shared_pca_init") and not self._pca_initialized

    @torch.no_grad()
    def _copy_pca_components(self, projector, features):
        if not isinstance(projector, nn.Linear):
            raise ValueError(f"--prototype_projector {self.projector_mode} expects a linear projector for PCA init")
        projector.float()
        components = _pca_components(features, self.prototype_dim)
        projector.weight.data.copy_(components.to(device=projector.weight.device, dtype=projector.weight.dtype))
        if projector.bias is not None:
            projector.bias.data.zero_()

    @torch.no_grad()
    def initialize_projector_from_features(self, image_features, text_features):
        if not self.needs_pca_init():
            return

        if self.projector_mode == "shared_pca_init":
            features = torch.cat([image_features.detach().cpu(), text_features.detach().cpu()], dim=0)
            self._copy_pca_components(self.image_projector, features)
        else:
            self._copy_pca_components(self.image_projector, image_features)
            self._copy_pca_components(self.text_projector, text_features)
        self._pca_initialized = True

    def _project(self, image_features, text_features):
        self.image_projector.float()
        self.text_projector.float()
        device = _first_device(self, image_features.device)
        image_features = image_features.to(device=device, dtype=torch.float32, non_blocking=True)
        text_features = text_features.to(device=device, dtype=torch.float32, non_blocking=True)
        image_features = F.normalize(self.image_projector(image_features), p=2, dim=1)
        text_features = F.normalize(self.text_projector(text_features), p=2, dim=1)
        return image_features, text_features

    def _project_token_sequence(self, token_features, projector, name):
        if token_features is None:
            return None
        if token_features.ndim != 3:
            raise ValueError(f"{name} token evidence expects a 3D tensor, got shape={tuple(token_features.shape)}")
        if token_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"{name} token evidence feature_dim mismatch: tokens have dim={token_features.shape[-1]}, "
                f"prototype branch expects dim={self.feature_dim}"
            )
        projector.float()
        device = _first_device(self, token_features.device)
        token_features = token_features.to(device=device, dtype=torch.float32, non_blocking=True)
        bsz, ntokens, _ = token_features.shape
        projected = projector(token_features.reshape(bsz * ntokens, -1))
        projected = projected.reshape(bsz, ntokens, self.prototype_dim)
        return F.normalize(projected.float(), p=2, dim=-1)

    def _prototype_banks_for_loss(self):
        if getattr(self.args, "no_pbt", False):
            return self.memory.text_prototypes, self.memory.image_prototypes, "text_prototypes", "image_prototypes"
        return self.memory.text_to_image, self.memory.image_to_text, "text_to_image", "image_to_text"

    def _identity_slot_indices(self, pids, max_prototypes_per_id=None):
        pids = pids.detach().long().cpu().tolist()
        proto_pids = self.memory.proto_pids.detach().cpu().long()
        slots = []
        for pid in pids:
            idx = (proto_pids == int(pid)).nonzero(as_tuple=False).flatten().tolist()
            if max_prototypes_per_id is not None:
                idx = idx[: int(max_prototypes_per_id)]
            slots.append(idx)
        width = max((len(x) for x in slots), default=0)
        if width <= 0:
            raise RuntimeError("No identity-owned prototype slots found for the requested pids.")
        device = self.memory.proto_pids.device
        indices = torch.full((len(slots), width), -1, device=device, dtype=torch.long)
        mask = torch.zeros((len(slots), width), device=device, dtype=torch.bool)
        for row, idx in enumerate(slots):
            if not idx:
                continue
            values = torch.tensor(idx, device=device, dtype=torch.long)
            indices[row, : values.numel()] = values
            mask[row, : values.numel()] = True
        return indices, mask

    @staticmethod
    def _masked_token_softmax(scores, token_mask=None, slot_mask=None, prior=None, eps=1e-8):
        if token_mask is None:
            token_mask = torch.ones(scores.shape[0], scores.shape[-1], device=scores.device, dtype=torch.bool)
        else:
            token_mask = token_mask.to(device=scores.device, dtype=torch.bool)
        if slot_mask is None:
            slot_mask = torch.ones(scores.shape[0], scores.shape[1], device=scores.device, dtype=torch.bool)
        else:
            slot_mask = slot_mask.to(device=scores.device, dtype=torch.bool)

        valid = slot_mask.unsqueeze(-1) & token_mask.unsqueeze(1)
        if prior is not None:
            prior = prior.to(device=scores.device, dtype=scores.dtype).clamp_min(0.0)
            prior = prior * token_mask.to(dtype=scores.dtype)
            prior_sum = prior.sum(dim=-1, keepdim=True)
            uniform = token_mask.to(dtype=scores.dtype)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
            prior = torch.where(prior_sum > eps, prior / prior_sum.clamp_min(eps), uniform)
            scores = scores + prior.clamp_min(eps).log().unsqueeze(1)

        scores = scores.masked_fill(~valid, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(valid, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        return weights

    @torch.no_grad()
    def evidence(
        self,
        image_features,
        text_features,
        pids,
        image_token_features=None,
        text_token_features=None,
        image_attention_prior=None,
        text_attention_prior=None,
        image_token_mask=None,
        text_token_mask=None,
        max_prototypes_per_id=None,
    ):
        """Collect prototype-conditioned diagnostic evidence without changing training forward outputs."""
        image_features, text_features = self._project(image_features, text_features)
        zero = image_features.sum() * 0.0
        if not self.is_ready():
            raise RuntimeError("Prototype evidence requested, but prototype memory is not initialized.")

        image_token_features = self._project_token_sequence(image_token_features, self.image_projector, "visual")
        text_token_features = self._project_token_sequence(text_token_features, self.text_projector, "text")
        if image_token_features is None and text_token_features is None:
            raise RuntimeError("Prototype evidence requested, but no visual or text token features were provided.")

        slot_indices, slot_mask = self._identity_slot_indices(pids, max_prototypes_per_id=max_prototypes_per_id)
        safe_slot_indices = slot_indices.clamp_min(0)
        visual_bank, text_bank, visual_bank_key, text_bank_key = self._prototype_banks_for_loss()
        visual_bank = F.normalize(visual_bank.to(image_features.device).float(), p=2, dim=1)
        text_bank = F.normalize(text_bank.to(text_features.device).float(), p=2, dim=1)
        visual_slots = visual_bank[safe_slot_indices.to(visual_bank.device)]
        text_slots = text_bank[safe_slot_indices.to(text_bank.device)]

        visual_slot_logits = torch.bmm(visual_slots, image_features.unsqueeze(-1)).squeeze(-1)
        text_slot_logits = torch.bmm(text_slots, text_features.unsqueeze(-1)).squeeze(-1)
        slot_logits = 0.5 * (visual_slot_logits.to(text_features.device) + text_slot_logits)
        slot_logits = slot_logits.masked_fill(~slot_mask.to(slot_logits.device), float("-inf"))
        tau = max(float(getattr(self.args, "prototype_tau", 0.05)), 1e-6)
        contribution = torch.softmax(slot_logits / tau, dim=1)
        contribution = torch.where(slot_mask.to(contribution.device), contribution, torch.zeros_like(contribution))
        contribution = contribution / contribution.sum(dim=1, keepdim=True).clamp_min(1e-8)

        ret = {
            "prototype_indices": slot_indices.detach(),
            "prototype_slot_mask": slot_mask.detach(),
            "prototype_pids": self.memory.proto_pids[safe_slot_indices].detach(),
            "gate": None,
            "gate_available": False,
            "contribution": contribution.detach(),
            "slot_similarity": slot_logits.detach(),
            "visual_bank_key": visual_bank_key,
            "text_bank_key": text_bank_key,
            "diagnostic_zero": zero.detach(),
        }

        scale = 1.0 / tau
        if image_token_features is not None:
            visual_scores = torch.einsum("bkd,bmd->bkm", visual_slots.to(image_token_features.device), image_token_features)
            ret["visual_attention"] = self._masked_token_softmax(
                visual_scores * scale,
                token_mask=image_token_mask,
                slot_mask=slot_mask,
                prior=image_attention_prior,
            ).detach()
        if text_token_features is not None:
            text_scores = torch.einsum("bkd,bld->bkl", text_slots.to(text_token_features.device), text_token_features)
            ret["text_attention"] = self._masked_token_softmax(
                text_scores * scale,
                token_mask=text_token_mask,
                slot_mask=slot_mask,
                prior=text_attention_prior,
            ).detach()
        return ret


    @torch.no_grad()
    def initialize(self, image_features, text_features, pids):
        if self.needs_pca_init():
            self.initialize_projector_from_features(image_features, text_features)
        image_features, text_features = self._project(image_features, text_features)
        self.initialize_projected(image_features, text_features, pids)

    @torch.no_grad()
    def initialize_projected(self, image_features, text_features, pids):
        prototype_seed = getattr(self.args, "seed", None)
        if prototype_seed is not None:
            prototype_seed = int(prototype_seed) + 1000
        self.memory.initialize(
            F.normalize(image_features.float(), p=2, dim=1).detach(),
            F.normalize(text_features.float(), p=2, dim=1).detach(),
            pids.detach(),
            num_iters=getattr(self.args, "prototype_kmeans_iters", 20),
            seed=prototype_seed,
        )

    @torch.no_grad()
    def project_for_memory(self, image_features, text_features):
        return self._project(image_features, text_features)

    def forward(self, image_features, text_features, pids, use_loss_id=True):
        image_features, text_features = self._project(image_features, text_features)
        zero = image_features.sum() * 0.0
        if not self.is_ready():
            ret = {}
            if use_loss_id:
                ret["proto_id_loss"] = zero
            return ret

        pids = pids.long()
        ret = {}
        if use_loss_id:
            ret["proto_id_loss"] = symmetric_identity_proxy_loss(
                image_features,
                text_features,
                pids,
                self.memory,
                tau=getattr(self.args, "prototype_tau", 0.05),
                hard_k=getattr(self.args, "prototype_hard_k", 16),
                use_pbt=not getattr(self.args, "no_pbt", False),
            )

        self.memory.ema_update(image_features.detach(), text_features.detach(), pids.detach())
        return ret

    @torch.no_grad()
    def score(self, text_features, image_features):
        was_training = self.training
        self.eval()
        try:
            image_features, text_features = self._project(image_features, text_features)
            scores = self.memory.prototype_score_matrix(text_features, image_features)
        finally:
            self.train(was_training)
        return scores
