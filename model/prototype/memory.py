import torch
import torch.nn as nn
import torch.nn.functional as F

from .kmeans import identity_kmeans


_SLOT_MODES = {"fixed", "adaptive_masked"}
_UTILITY_EPS = 1e-6


class PrototypeMemory(nn.Module):
    def __init__(self, num_classes, prototypes_per_id, dim, momentum=0.2,
                 group_mean_impl="deterministic", slot_mode="fixed",
                 prototype_max_per_id=None):
        super().__init__()
        if group_mean_impl not in ("deterministic", "scatter"):
            raise ValueError("group_mean_impl must be 'deterministic' or 'scatter'")
        self.num_classes = num_classes
        self.target_prototypes_per_id = max(int(prototypes_per_id), 1)
        self.dim = dim
        self.momentum = momentum
        self.group_mean_impl = group_mean_impl
        self.slot_mode = str(slot_mode).lower()
        if self.slot_mode not in _SLOT_MODES:
            raise ValueError(f"Unknown prototype slot mode: {slot_mode}")

        if self.slot_mode == "adaptive_masked":
            max_per_id = self.target_prototypes_per_id if prototype_max_per_id is None else int(prototype_max_per_id)
            if max_per_id <= self.target_prototypes_per_id:
                raise ValueError(
                    "--prototype_slot_mode adaptive_masked requires "
                    "--prototype_max_per_id > --prototype_per_id"
                )
            self.prototypes_per_id = max_per_id
        else:
            self.prototypes_per_id = self.target_prototypes_per_id
        self.prototype_max_per_id = self.prototypes_per_id
        total = num_classes * self.prototypes_per_id

        self.register_buffer("image_prototypes", torch.zeros(total, dim))
        self.register_buffer("text_prototypes", torch.zeros(total, dim))
        self.register_buffer("text_to_image", torch.zeros(total, dim))
        self.register_buffer("image_to_text", torch.zeros(total, dim))
        self.register_buffer("proto_pids", torch.arange(num_classes).repeat_interleave(self.prototypes_per_id).long())
        self.register_buffer("proto_active_mask", torch.ones(total, dtype=torch.bool))
        self.register_buffer("prototype_allocation_fallback", torch.tensor(False))
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def total_prototypes(self):
        return self.num_classes * self.prototypes_per_id

    @property
    def target_total_prototypes(self):
        return self.num_classes * self.target_prototypes_per_id

    def is_ready(self):
        return bool(self.initialized.item())

    def _fixed_active_mask(self, device=None):
        device = self.proto_active_mask.device if device is None else device
        mask = torch.zeros(
            self.num_classes,
            self.prototypes_per_id,
            device=device,
            dtype=torch.bool,
        )
        active_k = min(self.target_prototypes_per_id, self.prototypes_per_id)
        mask[:, :active_k] = True
        return mask.reshape(-1)

    def _all_active_mask(self, device=None):
        device = self.proto_active_mask.device if device is None else device
        return torch.ones(self.total_prototypes, device=device, dtype=torch.bool)

    def _set_active_mask(self, active_mask, fallback=False):
        active_mask = active_mask.to(self.proto_active_mask.device, dtype=torch.bool).view(-1)
        if active_mask.numel() != self.proto_active_mask.numel():
            raise ValueError("proto_active_mask has incompatible shape")
        self.proto_active_mask.copy_(active_mask)
        self.prototype_allocation_fallback.fill_(bool(fallback))

    def _place_fixed_bank_in_physical_rows(self, fixed_bank, candidate_bank):
        expanded = candidate_bank.clone()
        active_k = min(self.target_prototypes_per_id, self.prototypes_per_id)
        for pid in range(self.num_classes):
            dst_start = pid * self.prototypes_per_id
            src_start = pid * self.target_prototypes_per_id
            expanded[dst_start:dst_start + active_k] = fixed_bank[src_start:src_start + active_k]
        return expanded

    def _identity_slot_mass(self, features, slots):
        if features.numel() == 0:
            return slots.new_zeros(slots.shape[0])
        sims = F.normalize(features.float(), p=2, dim=1) @ F.normalize(slots.float(), p=2, dim=1).t()
        assignments = sims.argmax(dim=1)
        counts = torch.bincount(assignments, minlength=slots.shape[0]).float().to(slots.device)
        return counts / max(int(features.shape[0]), 1)

    def _slot_redundancy_to_kept(self, image_slots, text_slots, slot_idx, kept):
        if not kept:
            return image_slots.new_tensor(0.0)
        kept = torch.tensor(kept, device=image_slots.device, dtype=torch.long)
        image_slot = F.normalize(image_slots[slot_idx:slot_idx + 1].float(), p=2, dim=1)
        text_slot = F.normalize(text_slots[slot_idx:slot_idx + 1].float(), p=2, dim=1)
        kept_image = F.normalize(image_slots[kept].float(), p=2, dim=1)
        kept_text = F.normalize(text_slots[kept].float(), p=2, dim=1)
        image_redundancy = (image_slot @ kept_image.t()).max()
        text_redundancy = (text_slot @ kept_text.t()).max()
        return torch.maximum(image_redundancy, text_redundancy).clamp(-1.0, 1.0)

    @torch.no_grad()
    def _adaptive_active_mask(self, image_features, text_features, pids, image_bank, text_bank):
        if pids.numel() == 0 or self.target_total_prototypes < self.num_classes:
            return self._fixed_active_mask(device=image_bank.device), True

        k = self.prototypes_per_id
        target_total = min(self.target_total_prototypes, self.total_prototypes)
        active = torch.zeros(self.num_classes, k, device=image_bank.device, dtype=torch.bool)
        utilities = []

        image_bank_by_id = image_bank.view(self.num_classes, k, self.dim)
        text_bank_by_id = text_bank.view(self.num_classes, k, self.dim)
        image_features = image_features.to(image_bank.device)
        text_features = text_features.to(text_bank.device)
        pids = pids.long().to(image_bank.device)

        for pid in range(self.num_classes):
            pid_mask = pids.eq(pid)
            image_slots = image_bank_by_id[pid]
            text_slots = text_bank_by_id[pid]
            mass_img = self._identity_slot_mass(image_features[pid_mask], image_slots)
            mass_txt = self._identity_slot_mass(text_features[pid_mask], text_slots)
            mass = torch.maximum(mass_img, mass_txt)
            if not torch.isfinite(mass).all():
                return self._fixed_active_mask(device=image_bank.device), True

            first_slot = int(mass.argmax().item()) if mass.numel() > 0 else 0
            active[pid, first_slot] = True
            kept = [first_slot]

            for slot_idx in range(k):
                if slot_idx == first_slot:
                    continue
                redundancy = self._slot_redundancy_to_kept(image_slots, text_slots, slot_idx, kept)
                novelty = (1.0 - redundancy).clamp(0.0, 1.0)
                utility = (mass[slot_idx] * novelty).detach()
                if torch.isfinite(utility):
                    utilities.append((float(utility.item()), pid, slot_idx))

        needed = int(target_total - active.sum().item())
        if needed < 0:
            return self._fixed_active_mask(device=image_bank.device), True
        if needed == 0:
            return active.reshape(-1), False

        utilities = [item for item in utilities if item[0] > _UTILITY_EPS]
        if len(utilities) < needed:
            return self._fixed_active_mask(device=image_bank.device), True

        utilities.sort(key=lambda item: item[0], reverse=True)
        if utilities[0][0] <= _UTILITY_EPS:
            return self._fixed_active_mask(device=image_bank.device), True

        for _, pid, slot_idx in utilities[:needed]:
            active[pid, slot_idx] = True

        if int(active.sum().item()) != target_total:
            return self._fixed_active_mask(device=image_bank.device), True
        return active.reshape(-1), False

    @torch.no_grad()
    def initialize(self, image_features, text_features, pids, num_iters=20,
                   kmeans_init="deterministic", seed=None):
        image_features = F.normalize(image_features.float(), p=2, dim=1)
        text_features = F.normalize(text_features.float(), p=2, dim=1)
        pids = pids.long()
        text_seed = None if seed is None else int(seed) + self.num_classes

        image_bank = identity_kmeans(
            image_features,
            pids,
            self.num_classes,
            self.prototypes_per_id,
            num_iters=num_iters,
            init_method=kmeans_init,
            seed=seed,
        )
        text_bank = identity_kmeans(
            text_features,
            pids,
            self.num_classes,
            self.prototypes_per_id,
            num_iters=num_iters,
            init_method=kmeans_init,
            seed=text_seed,
        )

        if self.slot_mode == "adaptive_masked":
            active_mask, fallback = self._adaptive_active_mask(
                image_features,
                text_features,
                pids,
                image_bank,
                text_bank,
            )
            if fallback:
                fixed_image_bank = identity_kmeans(
                    image_features,
                    pids,
                    self.num_classes,
                    self.target_prototypes_per_id,
                    num_iters=num_iters,
                    init_method=kmeans_init,
                    seed=seed,
                )
                fixed_text_bank = identity_kmeans(
                    text_features,
                    pids,
                    self.num_classes,
                    self.target_prototypes_per_id,
                    num_iters=num_iters,
                    init_method=kmeans_init,
                    seed=text_seed,
                )
                image_bank = self._place_fixed_bank_in_physical_rows(fixed_image_bank, image_bank)
                text_bank = self._place_fixed_bank_in_physical_rows(fixed_text_bank, text_bank)
        else:
            active_mask = self._all_active_mask(device=image_bank.device)
            fallback = False

        self.image_prototypes.copy_(image_bank.to(self.image_prototypes.device))
        self.text_prototypes.copy_(text_bank.to(self.text_prototypes.device))
        self._set_active_mask(active_mask, fallback=fallback)
        self._rebuild_pbt(image_features, text_features, pids)
        self.initialized.fill_(True)
        return self.allocation_metrics()

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
        if self.group_mean_impl == "deterministic":
            return self._deterministic_group_means(assignments, features, bank)
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
    def _deterministic_group_means(self, assignments, features, bank):
        assignments = assignments.detach().to("cpu").long()
        features = features.detach().to("cpu", dtype=torch.float32)
        valid = torch.unique(assignments, sorted=True)
        if valid.numel() == 0:
            return valid.to(bank.device), bank.new_empty((0, bank.shape[1]))

        means = []
        for idx in valid.tolist():
            means.append(features[assignments == idx].mean(dim=0))
        means = torch.stack(means, dim=0).to(bank.device, dtype=bank.dtype)
        return valid.to(bank.device), F.normalize(means, p=2, dim=1)

    def _resolve_active_mask(self, bank, active_mask, device):
        if active_mask is None and bank.shape[0] == self.total_prototypes:
            active_mask = self.proto_active_mask
        if active_mask is None:
            return torch.ones(bank.shape[0], device=device, dtype=torch.bool)
        return active_mask.to(device=device, dtype=torch.bool).view(-1)

    def assign_identity(self, features, pids, bank, active_mask=None):
        features = F.normalize(features.float(), p=2, dim=1)
        pids = pids.long().to(features.device)
        bank = bank.to(features.device)
        active_mask = self._resolve_active_mask(bank, active_mask, features.device)
        local_bank = bank.view(self.num_classes, self.prototypes_per_id, self.dim)[pids]
        local_active = active_mask.view(self.num_classes, self.prototypes_per_id)[pids]
        empty_rows = ~local_active.any(dim=1)
        if empty_rows.any():
            local_active = local_active.clone()
            local_active[empty_rows] = True
        sims = torch.bmm(local_bank, features.unsqueeze(-1)).squeeze(-1)
        sims = sims.masked_fill(~local_active, float("-inf"))
        local_idx = sims.argmax(dim=1)
        return pids * self.prototypes_per_id + local_idx

    def assign_global(self, features, bank, chunk_size=4096, active_mask=None):
        features = F.normalize(features.float(), p=2, dim=1)
        bank = F.normalize(bank.to(features.device).float(), p=2, dim=1)
        active_mask = self._resolve_active_mask(bank, active_mask, features.device)
        assignments = []
        for start in range(0, features.shape[0], chunk_size):
            sims = features[start:start + chunk_size] @ bank.t()
            if active_mask.any():
                sims = sims.masked_fill(~active_mask.unsqueeze(0), float("-inf"))
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

    def active_slots_per_identity(self):
        return self.proto_active_mask.view(self.num_classes, self.prototypes_per_id).sum(dim=1)

    def allocation_metrics(self):
        active_counts = self.active_slots_per_identity().detach().float().cpu()
        active_rate = self.proto_active_mask.detach().float().mean().cpu()
        if active_counts.numel() == 0:
            zero = 0.0
            mean = p10 = p90 = min_value = max_value = zero
        else:
            mean = active_counts.mean().item()
            p10 = torch.quantile(active_counts, 0.10).item()
            p90 = torch.quantile(active_counts, 0.90).item()
            min_value = active_counts.min().item()
            max_value = active_counts.max().item()
        return {
            "prototype_max_per_id": float(self.prototype_max_per_id),
            "prototype_active_slots_mean": mean,
            "prototype_active_slots_p10": p10,
            "prototype_active_slots_p90": p90,
            "prototype_active_slots_min": min_value,
            "prototype_active_slots_max": max_value,
            "prototype_active_slot_rate": active_rate.item(),
            "prototype_allocation_fallback": float(self.prototype_allocation_fallback.item()),
        }

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        active_key = prefix + "proto_active_mask"
        fallback_key = prefix + "prototype_allocation_fallback"
        if active_key not in state_dict:
            state_dict[active_key] = self._fixed_active_mask(device=self.proto_active_mask.device)
        if fallback_key not in state_dict:
            state_dict[fallback_key] = torch.tensor(False, device=self.prototype_allocation_fallback.device)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
