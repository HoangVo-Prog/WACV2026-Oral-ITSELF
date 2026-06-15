import torch
import torch.nn as nn
import torch.nn.functional as F

from .kmeans import global_kmeans, identity_kmeans


ASSIGNMENT_MODES = ("identity_hard", "global_hard", "global_soft")


class PrototypeMemory(nn.Module):
    def __init__(
        self,
        num_classes,
        prototypes_per_id,
        dim,
        momentum=0.2,
        assignment_mode="identity_hard",
        assignment_tau=0.05,
        identity_owned_init=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes_per_id = prototypes_per_id
        self.dim = dim
        self.momentum = momentum
        self.assignment_mode = self._validate_assignment_mode(assignment_mode)
        self.assignment_tau = float(assignment_tau)
        self.identity_owned_init = bool(identity_owned_init)
        total = num_classes * prototypes_per_id

        self.register_buffer("image_prototypes", torch.zeros(total, dim))
        self.register_buffer("text_prototypes", torch.zeros(total, dim))
        self.register_buffer("text_to_image", torch.zeros(total, dim))
        self.register_buffer("image_to_text", torch.zeros(total, dim))
        self.register_buffer("proto_pids", torch.arange(num_classes).repeat_interleave(prototypes_per_id).long())
        self.register_buffer("initialized", torch.tensor(False))

    @staticmethod
    def _validate_assignment_mode(mode):
        if mode not in ASSIGNMENT_MODES:
            raise ValueError(f"Unknown prototype assignment mode: {mode}")
        return mode

    @property
    def uses_soft_assignment(self):
        return self.assignment_mode == "global_soft"

    @property
    def total_prototypes(self):
        return self.num_classes * self.prototypes_per_id

    def is_ready(self):
        return bool(self.initialized.item())

    def _tau(self):
        return max(float(self.assignment_tau), 1e-6)

    @torch.no_grad()
    def initialize(self, image_features, text_features, pids, num_iters=20, seed=None, identity_owned=None):
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()
        identity_owned = self.identity_owned_init if identity_owned is None else bool(identity_owned)

        if identity_owned:
            image_bank = identity_kmeans(
                image_features,
                pids,
                self.num_classes,
                self.prototypes_per_id,
                num_iters=num_iters,
                seed=seed,
            )
            text_bank = identity_kmeans(
                text_features,
                pids,
                self.num_classes,
                self.prototypes_per_id,
                num_iters=num_iters,
                seed=None if seed is None else int(seed) + 1,
            )
        else:
            image_bank = global_kmeans(
                image_features,
                self.total_prototypes,
                num_iters=num_iters,
                seed=seed,
            )
            text_bank = global_kmeans(
                text_features,
                self.total_prototypes,
                num_iters=num_iters,
                seed=None if seed is None else int(seed) + 1,
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

        image_assign = self.assign_for_update(image_features, pids, self.image_prototypes)
        text_assign = self.assign_for_update(text_features, pids, self.text_prototypes)
        self._ema_scatter_assignment(self.image_prototypes, image_assign, image_features)
        self._ema_scatter_assignment(self.text_prototypes, text_assign, text_features)
        self._ema_scatter_assignment(self.text_to_image, text_assign, image_features)
        self._ema_scatter_assignment(self.image_to_text, image_assign, text_features)

    @torch.no_grad()
    def _rebuild_pbt(self, image_features, text_features, pids):
        image_features = image_features.to(self.image_prototypes.device)
        text_features = text_features.to(self.text_prototypes.device)
        pids = pids.to(self.proto_pids.device)

        self.text_to_image.copy_(self.image_prototypes)
        self.image_to_text.copy_(self.text_prototypes)

        if self.uses_soft_assignment:
            self._mean_soft_scatter(self.text_to_image, self.text_prototypes, image_features)
            self._mean_soft_scatter(self.image_to_text, self.image_prototypes, text_features)
            return

        image_assign = self.assign_hard_for_mode(image_features, pids, self.image_prototypes)
        text_assign = self.assign_hard_for_mode(text_features, pids, self.text_prototypes)
        self._mean_scatter(self.text_to_image, text_assign, image_features)
        self._mean_scatter(self.image_to_text, image_assign, text_features)

    @torch.no_grad()
    def _mean_scatter(self, bank, assignments, features):
        valid, means = self._group_means(assignments, features, bank)
        if valid.numel() > 0:
            bank[valid] = means

    @torch.no_grad()
    def _ema_scatter(self, bank, assignments, features):
        valid, means = self._group_means(assignments, features, bank)
        if valid.numel() > 0:
            bank[valid] = F.normalize((1.0 - self.momentum) * bank[valid] + self.momentum * means, p=2, dim=1)

    @torch.no_grad()
    def _mean_scatter_assignment(self, bank, assignments, features):
        if assignments.ndim == 2:
            valid, means = self._weighted_group_means(assignments, features, bank)
        else:
            valid, means = self._group_means(assignments, features, bank)
        if valid.numel() > 0:
            bank[valid] = means.to(device=bank.device, dtype=bank.dtype)

    @torch.no_grad()
    def _ema_scatter_assignment(self, bank, assignments, features):
        if assignments.ndim == 2:
            valid, means = self._weighted_group_means(assignments, features, bank)
        else:
            valid, means = self._group_means(assignments, features, bank)
        if valid.numel() > 0:
            means = means.to(device=bank.device, dtype=bank.dtype)
            bank[valid] = F.normalize((1.0 - self.momentum) * bank[valid] + self.momentum * means, p=2, dim=1)

    @torch.no_grad()
    def _group_means(self, assignments, features, bank):
        return self._scatter_group_means(assignments, features, bank)

    @torch.no_grad()
    def _scatter_group_means(self, assignments, features, bank):
        assignments = assignments.to(bank.device).long()
        features = features.to(bank.device, dtype=bank.dtype)
        sums = torch.zeros_like(bank)
        counts = torch.zeros(bank.shape[0], 1, device=bank.device, dtype=bank.dtype)
        sums.index_add_(0, assignments, features)
        counts.index_add_(
            0,
            assignments,
            torch.ones(assignments.shape[0], 1, device=bank.device, dtype=bank.dtype),
        )
        valid_mask = counts.squeeze(1) > 0
        valid = valid_mask.nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            return valid, bank.new_empty((0, bank.shape[1]))
        means = sums[valid] / counts[valid].clamp_min(1.0)
        return valid, F.normalize(means, p=2, dim=1)

    @torch.no_grad()
    def _weighted_group_means(self, weights, features, bank):
        weights = weights.to(bank.device).float()
        features = features.to(bank.device).float()
        sums = weights.t() @ features
        counts = weights.sum(dim=0, keepdim=True).t()
        valid_mask = counts.squeeze(1) > 0
        valid = valid_mask.nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            return valid, bank.new_empty((0, bank.shape[1]))
        means = sums[valid] / counts[valid].clamp_min(1e-12)
        return valid, F.normalize(means, p=2, dim=1).to(dtype=bank.dtype)

    @torch.no_grad()
    def _soft_group_means(self, assignment_bank, features, target_bank, chunk_size=1024):
        device = target_bank.device
        assignment_bank = F.normalize(assignment_bank.to(device).float(), p=2, dim=1)
        features = features.to(device).float()
        sums = torch.zeros(target_bank.shape, device=device, dtype=torch.float32)
        counts = torch.zeros(target_bank.shape[0], 1, device=device, dtype=torch.float32)
        for start in range(0, features.shape[0], max(int(chunk_size), 1)):
            chunk = features[start:start + chunk_size]
            weights = torch.softmax((chunk @ assignment_bank.t()) / self._tau(), dim=1)
            sums += weights.t() @ chunk
            counts += weights.sum(dim=0, keepdim=True).t()
        valid_mask = counts.squeeze(1) > 0
        valid = valid_mask.nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            return valid, target_bank.new_empty((0, target_bank.shape[1]))
        means = sums[valid] / counts[valid].clamp_min(1e-12)
        return valid, F.normalize(means, p=2, dim=1).to(dtype=target_bank.dtype)

    @torch.no_grad()
    def _mean_soft_scatter(self, target_bank, assignment_bank, features, chunk_size=1024):
        valid, means = self._soft_group_means(assignment_bank, features, target_bank, chunk_size=chunk_size)
        if valid.numel() > 0:
            target_bank[valid] = means.to(device=target_bank.device, dtype=target_bank.dtype)

    def assign_for_update(self, features, pids, bank):
        if self.assignment_mode == "global_soft":
            return self.assign_soft_global(features, bank)
        return self.assign_hard_for_mode(features, pids, bank)

    def assign_hard_for_mode(self, features, pids, bank):
        if self.assignment_mode == "identity_hard":
            return self.assign_identity(features, pids, bank)
        return self.assign_global(features, bank)

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

    def assign_soft_global(self, features, bank):
        features = F.normalize(features.float(), p=2, dim=1)
        bank = F.normalize(bank.to(features.device).float(), p=2, dim=1)
        return torch.softmax((features @ bank.t()) / self._tau(), dim=1)

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
