import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import prototype_pair_ranking_loss, symmetric_identity_proxy_loss
from .memory import PrototypeMemory


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
        self.memory.initialize(
            F.normalize(image_features.float(), p=2, dim=1).detach(),
            F.normalize(text_features.float(), p=2, dim=1).detach(),
            pids.detach(),
            num_iters=getattr(self.args, "prototype_kmeans_iters", 20),
        )

    @torch.no_grad()
    def project_for_memory(self, image_features, text_features):
        return self._project(image_features, text_features)

    def _projected_selection_scores(self, text_features, image_features):
        return F.normalize(text_features, p=2, dim=1) @ F.normalize(image_features, p=2, dim=1).t()

    def _rank_selection_scores(self, text_features, image_features, host_scores=None):
        source = getattr(self.args, "prototype_hard_negative_source", "host")
        if source == "projected" or host_scores is None:
            return self._projected_selection_scores(text_features, image_features)

        host_scores = host_scores.to(device=text_features.device, dtype=torch.float32)
        expected_shape = (text_features.shape[0], image_features.shape[0])
        if host_scores.shape != expected_shape:
            raise ValueError(
                "host_scores must have shape {}, got {}".format(expected_shape, tuple(host_scores.shape))
            )
        return host_scores

    def forward(self, image_features, text_features, pids, use_loss_id=True, use_loss_rank=True, host_scores=None):
        image_features, text_features = self._project(image_features, text_features)
        zero = image_features.sum() * 0.0
        if not self.is_ready():
            ret = {}
            if use_loss_id:
                ret["proto_id_loss"] = zero
            if use_loss_rank:
                ret["proto_rank_loss"] = zero
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
            )

        if use_loss_rank:
            proto_scores = self.memory.training_score_matrix(text_features, image_features)
            selection_scores = self._rank_selection_scores(text_features, image_features, host_scores=host_scores)
            ret["proto_rank_loss"] = prototype_pair_ranking_loss(
                proto_scores,
                pids,
                host_scores=selection_scores,
                margin=getattr(self.args, "prototype_margin", 0.2),
                hard_k=getattr(self.args, "prototype_hard_k", 16),
            )

        self.memory.ema_update(image_features.detach(), text_features.detach(), pids.detach())
        return ret

    @torch.no_grad()
    def score(self, text_features, image_features):
        was_training = self.training
        self.eval()
        try:
            image_features, text_features = self._project(image_features, text_features)
            score_mode = getattr(self.args, "prototype_inference_score", "training")
            if score_mode == "assigned":
                scores = self.memory.prototype_score_matrix(text_features, image_features)
            else:
                scores = self.memory.training_score_matrix(text_features, image_features)
        finally:
            self.train(was_training)
        return scores
