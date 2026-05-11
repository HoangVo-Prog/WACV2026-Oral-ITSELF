import torch
import torch.nn as nn
import torch.nn.functional as F

from .kmeans import identity_kmeans


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, prototypes_per_id, dim, momentum=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_id = prototypes_per_id
        self.dim = dim
        self.momentum = momentum
        total = num_classes * prototypes_per_id

        self.register_buffer("image_prototypes", torch.zeros(total, dim))
        self.register_buffer("text_prototypes", torch.zeros(total, dim))
        self.register_buffer("text_to_image", torch.zeros(total, dim))
        self.register_buffer("image_to_text", torch.zeros(total, dim))
        self.register_buffer("proto_pids", torch.arange(num_classes).repeat_interleave(prototypes_per_id).long())
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def total_prototypes(self):
        return self.num_classes * self.prototypes_per_id

    def is_ready(self):
        return bool(self.initialized.item())

    @torch.no_grad()
    def initialize(self, image_features, text_features, pids, num_iters=20):
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()

        image_bank = identity_kmeans(
            image_features, pids, self.num_classes, self.prototypes_per_id, num_iters=num_iters
        )
        text_bank = identity_kmeans(
            text_features, pids, self.num_classes, self.prototypes_per_id, num_iters=num_iters
        )

        self.image_prototypes.copy_(image_bank.to(self.image_prototypes.device))
        self.text_prototypes.copy_(text_bank.to(self.text_prototypes.device))
        self._rebuild_pbt(image_features, text_features, pids)
        self.initialized.fill_(True)

    @torch.no_grad()
    def ema_update(self, image_features, text_features, pids):
        if not self.is_ready():
            return

        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()

        image_assign = self.assign_identity(image_features, pids, self.image_prototypes)
        text_assign = self.assign_identity(text_features, pids, self.text_prototypes)
        self._ema_scatter(self.image_prototypes, image_assign, image_features)
        self._ema_scatter(self.text_prototypes, text_assign, text_features)
        self._ema_scatter(self.text_to_image, text_assign, image_features)
        self._ema_scatter(self.image_to_text, image_assign, text_features)

    @torch.no_grad()
    def _rebuild_pbt(self, image_features, text_features, pids):
        image_features = image_features.to(self.image_prototypes.device)
        text_features = text_features.to(self.text_prototypes.device)
        pids = pids.to(self.proto_pids.device)

        image_assign = self.assign_identity(image_features, pids, self.image_prototypes)
        text_assign = self.assign_identity(text_features, pids, self.text_prototypes)

        self.text_to_image.copy_(self.image_prototypes)
        self.image_to_text.copy_(self.text_prototypes)
        self._mean_scatter(self.text_to_image, text_assign, image_features)
        self._mean_scatter(self.image_to_text, image_assign, text_features)

    @torch.no_grad()
    def _mean_scatter(self, bank, assignments, features):
        valid, means = self._deterministic_group_means(assignments, features, bank)
        if valid.numel() > 0:
            bank[valid] = means

    @torch.no_grad()
    def _ema_scatter(self, bank, assignments, features):
        valid, means = self._deterministic_group_means(assignments, features, bank)
        if valid.numel() > 0:
            bank[valid] = F.normalize((1.0 - self.momentum) * bank[valid] + self.momentum * means, p=2, dim=1)

    @torch.no_grad()
    def _deterministic_group_means(self, assignments, features, bank):
        assignments = assignments.detach().to("cpu").long()
        features = features.detach().to("cpu", dtype=torch.float32)
        valid = torch.unique(assignments, sorted=True)
        means = []
        for idx in valid.tolist():
            means.append(features[assignments == idx].mean(dim=0))
        means = torch.stack(means, dim=0).to(bank.device, dtype=bank.dtype)
        valid = valid.to(bank.device)
        return valid, F.normalize(means, p=2, dim=1)

    def assign_identity(self, features, pids, bank):
        features = F.normalize(features.float(), p=2, dim=1)
        pids = pids.long().to(features.device)
        bank = bank.to(features.device)
        local_bank = bank.view(self.num_classes, self.prototypes_per_id, self.dim)[pids]
        sims = torch.bmm(local_bank, features.unsqueeze(-1)).squeeze(-1)
        local_idx = sims.argmax(dim=1)
        return pids * self.prototypes_per_id + local_idx

    def assign_global(self, features, bank, chunk_size=4096):
        features = F.normalize(features.float(), p=2, dim=1)
        bank = F.normalize(bank.to(features.device).float(), p=2, dim=1)
        assignments = []
        for start in range(0, features.shape[0], chunk_size):
            sims = features[start:start + chunk_size] @ bank.t()
            assignments.append(sims.argmax(dim=1))
        return torch.cat(assignments, dim=0)

    def prototype_score_matrix(self, text_features, image_features):
        if not self.is_ready():
            return text_features.new_zeros((text_features.shape[0], image_features.shape[0]))

        text_features = F.normalize(text_features.float(), p=2, dim=1)
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_idx = self.assign_global(text_features, self.text_prototypes)
        image_idx = self.assign_global(image_features, self.image_prototypes)

        visual_pbt = self.text_to_image.to(image_features.device).float()[text_idx]
        visual_cluster = self.image_prototypes.to(image_features.device).float()[image_idx]
        text_pbt = self.image_to_text.to(text_features.device).float()[image_idx]
        text_cluster = self.text_prototypes.to(text_features.device).float()[text_idx]

        visual_scores = F.normalize(visual_pbt, p=2, dim=1) @ F.normalize(visual_cluster, p=2, dim=1).t()
        text_scores = F.normalize(text_cluster, p=2, dim=1) @ F.normalize(text_pbt, p=2, dim=1).t()
        return visual_scores + text_scores

    def training_score_matrix(self, text_features, image_features):
        if not self.is_ready():
            return text_features.new_zeros((text_features.shape[0], image_features.shape[0]))

        text_features = F.normalize(text_features.float(), p=2, dim=1)
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        with torch.no_grad():
            text_idx = self.assign_global(text_features.detach(), self.text_prototypes)
            image_idx = self.assign_global(image_features.detach(), self.image_prototypes)

        visual_pbt = self.text_to_image.to(image_features.device).float()[text_idx]
        text_pbt = self.image_to_text.to(text_features.device).float()[image_idx]

        visual_scores = F.normalize(visual_pbt, p=2, dim=1) @ image_features.t()
        text_scores = text_features @ F.normalize(text_pbt, p=2, dim=1).t()
        return visual_scores + text_scores
