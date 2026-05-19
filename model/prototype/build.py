import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import symmetric_identity_proxy_loss
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
            group_mean_impl=getattr(args, "prototype_group_mean_impl", "deterministic"),
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
            kmeans_init=getattr(self.args, "prototype_kmeans_init", "deterministic"),
            seed=getattr(self.args, "prototype_seed", 1001),
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
                pressure_mode=getattr(self.args, "prototype_pressure_mode", "fixed"),
                host_image_features=host_image_features,
                host_text_features=host_text_features,
                epoch=epoch,
                warmup_epochs=getattr(self.args, "prototype_warmup_epochs", 0),
                return_details=True,
            )
            ret["proto_id_loss"] = proto_loss
            ret.update(proto_details)

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
