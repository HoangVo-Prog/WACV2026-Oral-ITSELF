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
    def initialize(self, image_features, text_features, pids, num_iters=20, seed=None):
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()

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

        self.image_prototypes.copy_(image_bank.to(self.image_prototypes.device))
        self.text_prototypes.copy_(text_bank.to(self.text_prototypes.device))
        self._rebuild_pbt(image_features, text_features, pids)
        self.initialized.fill_(True)

    @torch.no_grad()
    def refresh(self, image_features, text_features, pids, num_iters=20, seed=None, alpha=0.35, refresh_epoch=None, refresh_step=-1):
        if not self.is_ready():
            return {"prototype_refresh_done": 0.0}
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("prototype refresh alpha must be in [0, 1]")

        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()

        old_image = self.image_prototypes.detach().clone()
        old_text = self.text_prototypes.detach().clone()
        old_text_to_image = self.text_to_image.detach().clone()
        old_image_to_text = self.image_to_text.detach().clone()

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

        image_bank = self._align_identity_slots(old_image, image_bank)
        text_bank = self._align_identity_slots(old_text, text_bank)

        refreshed_image = self._blend_bank(old_image, image_bank, alpha)
        refreshed_text = self._blend_bank(old_text, text_bank, alpha)
        self.image_prototypes.copy_(refreshed_image)
        self.text_prototypes.copy_(refreshed_text)

        text_to_image, image_to_text = self._build_pbt_banks(
            image_features,
            text_features,
            pids,
            self.image_prototypes,
            self.text_prototypes,
        )
        refreshed_text_to_image = self._blend_bank(old_text_to_image, text_to_image, alpha)
        refreshed_image_to_text = self._blend_bank(old_image_to_text, image_to_text, alpha)
        self.text_to_image.copy_(refreshed_text_to_image)
        self.image_to_text.copy_(refreshed_image_to_text)

        return {
            "prototype_refresh_done": 1.0,
            "prototype_refresh_epoch": float(refresh_epoch) if refresh_epoch is not None else -1.0,
            "prototype_refresh_alpha": float(alpha),
            "prototype_refresh_step": float(refresh_step),
            "prototype_refresh_image_delta": self._bank_delta(old_image, self.image_prototypes),
            "prototype_refresh_text_delta": self._bank_delta(old_text, self.text_prototypes),
            "prototype_refresh_text_to_image_delta": self._bank_delta(old_text_to_image, self.text_to_image),
            "prototype_refresh_image_to_text_delta": self._bank_delta(old_image_to_text, self.image_to_text),
        }

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
        text_to_image, image_to_text = self._build_pbt_banks(
            image_features,
            text_features,
            pids,
            self.image_prototypes,
            self.text_prototypes,
        )
        self.text_to_image.copy_(text_to_image)
        self.image_to_text.copy_(image_to_text)

    @torch.no_grad()
    def _build_pbt_banks(self, image_features, text_features, pids, image_bank, text_bank):
        device = self.image_prototypes.device
        image_features = F.normalize(image_features.to(device).float(), p=2, dim=1)
        text_features = F.normalize(text_features.to(device).float(), p=2, dim=1)
        pids = pids.to(device)
        image_bank = F.normalize(image_bank.to(device).float(), p=2, dim=1)
        text_bank = F.normalize(text_bank.to(device).float(), p=2, dim=1)

        image_assign = self.assign_identity(image_features, pids, image_bank)
        text_assign = self.assign_identity(text_features, pids, text_bank)

        text_to_image = image_bank.clone()
        image_to_text = text_bank.clone()
        self._mean_scatter(text_to_image, text_assign, image_features)
        self._mean_scatter(image_to_text, image_assign, text_features)
        return text_to_image, image_to_text

    @torch.no_grad()
    def _align_identity_slots(self, old_bank, new_bank):
        device = self.image_prototypes.device
        old_bank = F.normalize(old_bank.to(device).float(), p=2, dim=1)
        new_bank = F.normalize(new_bank.to(device).float(), p=2, dim=1)
        if self.prototypes_per_id <= 1:
            return new_bank

        old_by_id = old_bank.view(self.num_classes, self.prototypes_per_id, self.dim)
        new_by_id = new_bank.view(self.num_classes, self.prototypes_per_id, self.dim)
        aligned = torch.empty_like(new_by_id)

        for pid in range(self.num_classes):
            sims = old_by_id[pid] @ new_by_id[pid].t()
            order = sims.reshape(-1).argsort(descending=True).tolist()
            used_old = set()
            used_new = set()
            for flat_idx in order:
                old_idx = flat_idx // self.prototypes_per_id
                new_idx = flat_idx % self.prototypes_per_id
                if old_idx in used_old or new_idx in used_new:
                    continue
                aligned[pid, old_idx] = new_by_id[pid, new_idx]
                used_old.add(old_idx)
                used_new.add(new_idx)
                if len(used_old) == self.prototypes_per_id:
                    break

        return aligned.reshape(self.total_prototypes, self.dim)

    @torch.no_grad()
    def _blend_bank(self, old_bank, new_bank, alpha):
        old_bank = old_bank.to(self.image_prototypes.device).float()
        new_bank = new_bank.to(self.image_prototypes.device).float()
        return F.normalize((1.0 - float(alpha)) * old_bank + float(alpha) * new_bank, p=2, dim=1)

    @torch.no_grad()
    def _bank_delta(self, old_bank, new_bank):
        old_bank = F.normalize(old_bank.to(new_bank.device).float(), p=2, dim=1)
        new_bank = F.normalize(new_bank.float(), p=2, dim=1)
        return (1.0 - (old_bank * new_bank).sum(dim=1)).mean().detach().cpu().item()

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
