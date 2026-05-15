import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import symmetric_identity_proxy_loss
from .memory import PrototypeMemory


_CWTU_WEIGHT_MIN = 0.25
_CWTU_WEIGHT_MAX = 1.0
_CWTU_EPS = 1e-6


def _text_update_weight_stats(weights, fallback):
    weights = weights.detach().float()
    if weights.numel() == 0:
        zero = weights.new_tensor(0.0)
        return {
            "prototype_text_update_weight_mean": zero,
            "prototype_text_update_weight_p10": zero,
            "prototype_text_update_weight_p90": zero,
            "prototype_text_update_weight_uniform_fallback": weights.new_tensor(float(fallback)),
        }
    return {
        "prototype_text_update_weight_mean": weights.mean(),
        "prototype_text_update_weight_p10": torch.quantile(weights, 0.10),
        "prototype_text_update_weight_p90": torch.quantile(weights, 0.90),
        "prototype_text_update_weight_uniform_fallback": weights.new_tensor(float(fallback)),
    }


def _uniform_text_update_weights(batch_size, device, fallback):
    weights = torch.ones(batch_size, device=device, dtype=torch.float32)
    return weights, _text_update_weight_stats(weights, fallback=fallback)


@torch.no_grad()
def _confidence_text_update_weights(image_host_features, text_host_features, pids):
    batch_size = int(pids.shape[0]) if pids is not None else 0
    if image_host_features is not None:
        device = image_host_features.device
    elif text_host_features is not None:
        device = text_host_features.device
    elif pids is not None:
        device = pids.device
    else:
        device = torch.device("cpu")

    if (
        batch_size < 2
        or image_host_features is None
        or text_host_features is None
    ):
        return _uniform_text_update_weights(batch_size, device, fallback=True)

    pids = pids.long().to(device)
    if pids.unique(sorted=False).numel() < 2:
        return _uniform_text_update_weights(batch_size, device, fallback=True)

    image_host_features = F.normalize(image_host_features.detach().float().to(device), p=2, dim=1)
    text_host_features = F.normalize(text_host_features.detach().float().to(device), p=2, dim=1)
    sims = text_host_features @ image_host_features.t()

    pos_mask = pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not neg_mask.any():
        return _uniform_text_update_weights(batch_size, device, fallback=True)

    best_pos = sims.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    hardest_wrong = sims.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
    margins = best_pos - hardest_wrong
    valid = torch.isfinite(best_pos) & torch.isfinite(hardest_wrong) & torch.isfinite(margins)
    if not bool(valid.all().item()):
        return _uniform_text_update_weights(batch_size, device, fallback=True)

    margin_std = margins.float().std(unbiased=False)
    if not bool(torch.isfinite(margin_std).item()) or margin_std.item() <= _CWTU_EPS:
        return _uniform_text_update_weights(batch_size, device, fallback=True)

    z = (margins.float() - margins.float().mean()) / (margin_std + _CWTU_EPS)
    weights = torch.sigmoid(z).clamp(_CWTU_WEIGHT_MIN, _CWTU_WEIGHT_MAX)
    weights = weights / weights.mean().clamp_min(_CWTU_EPS)
    return weights.detach(), _text_update_weight_stats(weights, fallback=False)


class PrototypeBranch(nn.Module):
    def __init__(self, args, num_classes, feature_dim):
        super().__init__()
        self.args = args
        self.feature_dim = feature_dim
        self.prototype_dim = getattr(args, "prototype_dim", 512)
        prototype_feature = getattr(args, "prototype_feature", "auto")
        self.use_local = prototype_feature == "auto" and not args.only_global
        self.use_local = self.use_local or (prototype_feature == "local" and not args.only_global)

        self.image_projector = nn.Sequential(
            nn.Linear(feature_dim, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim),
        )
        self.text_projector = nn.Sequential(
            nn.Linear(feature_dim, self.prototype_dim),
            nn.LayerNorm(self.prototype_dim),
        )
        self.memory = PrototypeMemory(
            num_classes=num_classes,
            prototypes_per_id=getattr(args, "prototype_per_id", 2),
            dim=self.prototype_dim,
            momentum=getattr(args, "prototype_momentum", 0.2),
        )

    def is_ready(self):
        return self.memory.is_ready()

    def _project(self, image_features, text_features):
        self.image_projector.float()
        self.text_projector.float()
        device = next(self.image_projector.parameters()).device
        image_features = image_features.to(device=device, dtype=torch.float32, non_blocking=True)
        text_features = text_features.to(device=device, dtype=torch.float32, non_blocking=True)
        image_features = F.normalize(self.image_projector(image_features), p=2, dim=1)
        text_features = F.normalize(self.text_projector(text_features), p=2, dim=1)
        return image_features, text_features

    @torch.no_grad()
    def initialize(self, image_features, text_features, pids):
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

    def forward(
        self,
        image_features,
        text_features,
        pids,
        use_loss_id=True,
        epoch=None,
        host_image_features=None,
        host_text_features=None,
    ):
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
            proto_loss, proto_details = symmetric_identity_proxy_loss(
                image_features,
                text_features,
                pids,
                self.memory,
                tau=getattr(self.args, "prototype_tau", 0.05),
                hard_k=getattr(self.args, "prototype_hard_k", 16),
                hard_k_mode=getattr(self.args, "prototype_hard_k_mode", "fixed"),
                pressure_mode=getattr(self.args, "prototype_pressure_mode", "fixed"),
                host_image_features=host_image_features,
                host_text_features=host_text_features,
                epoch=epoch,
                warmup_epochs=getattr(self.args, "prototype_warmup_epochs", 0),
                return_details=True,
            )
            ret["proto_id_loss"] = proto_loss
            ret.update(proto_details)

        text_update_mode = str(getattr(self.args, "prototype_text_update_mode", "uniform")).lower()
        if text_update_mode == "confidence_weighted":
            text_update_weights, text_update_details = _confidence_text_update_weights(
                host_image_features,
                host_text_features,
                pids,
            )
        else:
            text_update_weights, text_update_details = _uniform_text_update_weights(
                int(pids.shape[0]),
                image_features.device,
                fallback=False,
            )
            text_update_weights = None

        self.memory.ema_update(
            image_features.detach(),
            text_features.detach(),
            pids.detach(),
            image_to_text_weights=text_update_weights,
        )
        if use_loss_id or text_update_mode == "confidence_weighted":
            ret.update(text_update_details)
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
