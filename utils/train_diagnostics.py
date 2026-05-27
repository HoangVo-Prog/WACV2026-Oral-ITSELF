import math

import torch
import torch.nn.functional as F


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _to_float(value):
    if torch.is_tensor(value):
        value = value.detach()
        if value.numel() == 0:
            return None
        value = value.float().mean().item()
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _mean(values):
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return values.float().mean().item()


def _quantile(values, q):
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return torch.quantile(values.float(), q).item()


def _corrcoef(x, y):
    mask = torch.isfinite(x) & torch.isfinite(y)
    x = x[mask].float()
    y = y[mask].float()
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if denom.item() <= 0:
        return None
    return (x @ y / denom).item()


def _row_masked_mean(values, mask):
    counts = mask.sum(dim=1)
    sums = values.masked_fill(~mask, 0.0).sum(dim=1)
    out = values.new_full((values.shape[0],), float("nan"))
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid].float()
    return out


def _row_masked_max(values, mask):
    masked = values.masked_fill(~mask, float("-inf"))
    out = masked.max(dim=1).values
    out = out.masked_fill(~mask.any(dim=1), float("nan"))
    return out


def _row_topk_finite_mean(values, mask, k):
    k = min(max(int(k), 1), values.shape[1])
    top_values, top_idx = values.masked_fill(~mask, float("-inf")).topk(k=k, dim=1)
    finite = torch.isfinite(top_values)
    counts = finite.sum(dim=1)
    sums = top_values.masked_fill(~finite, 0.0).sum(dim=1)
    out = values.new_full((values.shape[0],), float("nan"))
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid].float()
    return out, top_idx, finite


def _batch_identity_centroids(image_features, text_features, pids):
    unique_pids, inverse = pids.unique(sorted=True, return_inverse=True)
    features = torch.cat([image_features, text_features], dim=0)
    inverse = inverse.repeat(2)

    centroids = image_features.new_zeros((unique_pids.numel(), image_features.shape[1]))
    centroids.index_add_(0, inverse, features)
    counts = torch.bincount(inverse, minlength=unique_pids.numel()).float().to(image_features.device)
    centroids = centroids / counts.clamp_min(1.0).unsqueeze(1)
    centroids = F.normalize(centroids, p=2, dim=1)
    return centroids, unique_pids


def _batch_host_metrics(image_features, text_features, pids, hard_k=16):
    image_features = F.normalize(image_features.float(), p=2, dim=1)
    text_features = F.normalize(text_features.float(), p=2, dim=1)
    pids = pids.long().to(image_features.device)

    sims = text_features @ image_features.t()
    image_sims = image_features @ image_features.t()
    text_sims = text_features @ text_features.t()
    pos_mask = pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    eye = torch.eye(pids.numel(), device=image_features.device, dtype=torch.bool)
    same_id_offdiag = pos_mask & ~eye

    pos_logits = sims.masked_fill(~pos_mask, float("-inf"))
    pos_for_min = sims.masked_fill(~pos_mask, float("inf"))
    neg_logits = sims.masked_fill(~neg_mask, float("-inf"))

    best_pos = pos_logits.max(dim=1).values
    worst_pos = pos_for_min.min(dim=1).values
    hard_neg, hard_neg_idx = neg_logits.max(dim=1)
    has_neg = neg_mask.any(dim=1)
    hard_neg = hard_neg.masked_fill(~has_neg, float("nan"))
    hard_neg_pids = pids[hard_neg_idx].detach().cpu()
    hard_neg_pids = hard_neg_pids.masked_fill(~has_neg.detach().cpu(), -1)

    host_margin = best_pos - hard_neg
    hard_pos_margin = worst_pos - hard_neg
    negative_intrusion = hard_neg > best_pos
    negative_intrusion = negative_intrusion.float().masked_fill(~has_neg, float("nan"))

    sorted_idx = sims.argsort(dim=1, descending=True)
    sorted_pos = pos_mask.gather(1, sorted_idx)
    first_pos_rank = sorted_pos.float().argmax(dim=1).float() + 1.0

    intra_i2i = _row_masked_mean(image_sims, same_id_offdiag)
    intra_t2t = _row_masked_mean(text_sims, same_id_offdiag)
    intra_xmod = _row_masked_mean(sims, pos_mask)

    inter_i2i_nearest = _row_masked_max(image_sims, neg_mask)
    inter_t2t_nearest = _row_masked_max(text_sims, neg_mask)
    inter_xmod_nearest = hard_neg

    topk_neg_attr, topk_neg_idx, topk_neg_valid = _row_topk_finite_mean(sims, neg_mask, hard_k)

    centroids, centroid_pids = _batch_identity_centroids(image_features, text_features, pids)
    if centroid_pids.numel() > 1:
        pid_to_centroid = pids.unsqueeze(1).eq(centroid_pids.unsqueeze(0)).float()
        sample_centroids = pid_to_centroid @ centroids
        sample_centroid_sims = sample_centroids @ sample_centroids.t()
        topk_neg_identity = sample_centroid_sims.gather(1, topk_neg_idx)
        topk_neg_identity = topk_neg_identity.masked_fill(~topk_neg_valid, float("nan"))
        topk_neg_identity = _row_masked_mean(topk_neg_identity, torch.isfinite(topk_neg_identity))

        centroid_sims = centroids @ centroids.t()
        centroid_eye = torch.eye(centroid_pids.numel(), device=centroids.device, dtype=torch.bool)
        nearest_centroid = _row_masked_max(centroid_sims, ~centroid_eye)
    else:
        topk_neg_identity = image_features.new_full((pids.numel(),), float("nan"))
        nearest_centroid = image_features.new_full((centroid_pids.numel(),), float("nan"))

    metrics = {
        "host_margin_mean": _mean(host_margin.detach().cpu()),
        "host_margin_p10": _quantile(host_margin.detach().cpu(), 0.10),
        "hard_pos_margin_mean": _mean(hard_pos_margin.detach().cpu()),
        "negative_intrusion_rate": _mean(negative_intrusion.detach().cpu()),
        "mean_first_positive_rank": _mean(first_pos_rank.detach().cpu()),
        "host_intra_i2i_sim_mean": _mean(image_sims[same_id_offdiag].detach().cpu()),
        "host_intra_t2t_sim_mean": _mean(text_sims[same_id_offdiag].detach().cpu()),
        "host_intra_xmod_sim_mean": _mean(sims[pos_mask].detach().cpu()),
        "host_paired_xmod_sim_mean": _mean(sims.diag().detach().cpu()),
        "host_inter_i2i_nearest_sim_mean": _mean(inter_i2i_nearest.detach().cpu()),
        "host_inter_t2t_nearest_sim_mean": _mean(inter_t2t_nearest.detach().cpu()),
        "host_inter_xmod_nearest_sim_mean": _mean(inter_xmod_nearest.detach().cpu()),
        "host_i2i_identity_margin_mean": _mean((intra_i2i - inter_i2i_nearest).detach().cpu()),
        "host_t2t_identity_margin_mean": _mean((intra_t2t - inter_t2t_nearest).detach().cpu()),
        "host_xmod_identity_margin_mean": _mean((intra_xmod - inter_xmod_nearest).detach().cpu()),
        "host_topk_neg_attr_sim_mean": _mean(topk_neg_attr.detach().cpu()),
        "host_topk_neg_identity_centroid_sim_mean": _mean(topk_neg_identity.detach().cpu()),
        "host_topk_neg_identity_centroid_distance_mean": _mean((1.0 - topk_neg_identity).detach().cpu()),
        "host_topk_attr_id_decoupling": _mean((topk_neg_attr - topk_neg_identity).detach().cpu()),
        "host_same_id_alignment_gap": _mean((intra_xmod - topk_neg_attr).detach().cpu()),
        "host_identity_centroid_nearest_sim_mean": _mean(nearest_centroid.detach().cpu()),
        "host_identity_centroid_margin_mean": _mean((1.0 - nearest_centroid).detach().cpu()),
    }
    return metrics, {
        "host_margin": host_margin.detach().cpu(),
        "hard_neg_pids": hard_neg_pids,
    }


def _prototype_margin(features, prototypes, proto_pids, pids, hard_k):
    features = F.normalize(features.float(), p=2, dim=1)
    prototypes = F.normalize(prototypes.to(features.device).float(), p=2, dim=1)
    proto_pids = proto_pids.to(features.device)
    pids = pids.long().to(features.device)

    logits = features @ prototypes.t()
    pos_mask = proto_pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    pos = logits.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
    neg = neg_logits.max(dim=1).values

    k = min(max(int(hard_k), 1), neg_logits.shape[1])
    topk_idx = neg_logits.topk(k=k, dim=1).indices
    topk_pids = proto_pids[topk_idx].detach().cpu()
    return (pos - neg).detach().cpu(), topk_pids


def _identity_proxy_banks(memory, use_pbt=True):
    if use_pbt:
        return memory.text_to_image, memory.image_to_text
    return memory.text_prototypes, memory.image_prototypes


def _assignment_metrics(memory, image_features, text_features, pids, indices, state):
    image_assign = memory.assign_identity(image_features, pids, memory.image_prototypes).detach().cpu()
    text_assign = memory.assign_identity(text_features, pids, memory.text_prototypes).detach().cpu()
    pids_cpu = pids.detach().cpu().long()
    k = int(memory.prototypes_per_id)
    present = pids_cpu.unique(sorted=True)

    dead_slots = 0
    total_slots = 0
    effective_slots = []

    for pid in present.tolist():
        pid_mask = pids_cpu.eq(pid)
        for assignments in (image_assign, text_assign):
            local = assignments[pid_mask] - pid * k
            counts = torch.bincount(local.clamp(0, k - 1), minlength=k).float()
            total = counts.sum()
            if total <= 0:
                continue
            probs = counts / total
            entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
            effective_slots.append(entropy.exp())
            dead_slots += counts.eq(0).sum().item()
            total_slots += k

    flip_rate = None
    if indices is not None:
        indices = indices.detach().cpu().long()
        flips = 0
        seen = 0
        for index, img_slot, txt_slot in zip(indices.tolist(), image_assign.tolist(), text_assign.tolist()):
            prev = state["assignments"].get(index)
            if prev is not None:
                seen += 2
                flips += int(prev[0] != img_slot)
                flips += int(prev[1] != txt_slot)
            state["assignments"][index] = (img_slot, txt_slot)
        if seen > 0:
            flip_rate = flips / seen

    return {
        "dead_slot_rate": (dead_slots / total_slots) if total_slots > 0 else None,
        "effective_slots_per_id": _mean(torch.stack(effective_slots).cpu()) if effective_slots else None,
        "assignment_flip_rate": flip_rate,
    }


def _slot_redundancy(memory):
    k = int(memory.prototypes_per_id)
    if k <= 1:
        return 0.0

    redundancies = []
    for bank in (memory.image_prototypes, memory.text_prototypes):
        bank = F.normalize(bank.float(), p=2, dim=1)
        bank = bank.view(memory.num_classes, k, memory.dim)
        sims = torch.bmm(bank, bank.transpose(1, 2))
        mask = ~torch.eye(k, device=sims.device, dtype=torch.bool).unsqueeze(0)
        mask = mask.expand(memory.num_classes, -1, -1)
        redundancies.append(sims[mask].mean().detach().cpu())
    return _mean(torch.stack(redundancies))


@torch.no_grad()
def compute_train_diagnostics(model, ret, args, state):
    diag = ret.get("_diag")
    if not diag:
        return {}

    pids = diag["pids"].detach()
    host_image = diag["host_image_feats"].detach()
    host_text = diag["host_text_feats"].detach()
    metrics, host_extra = _batch_host_metrics(
        host_image,
        host_text,
        pids,
        hard_k=getattr(args, "prototype_hard_k", 16),
    )

    model = _unwrap_model(model)
    branch = getattr(model, "prototype_branch", None)
    if branch is None or not branch.is_ready():
        return {key: value for key, value in metrics.items() if _to_float(value) is not None}

    proto_image = diag["proto_image_feats"].detach()
    proto_text = diag["proto_text_feats"].detach()
    proto_image, proto_text = branch.project_for_memory(proto_image, proto_text)
    memory = branch.memory
    hard_k = getattr(args, "prototype_hard_k", 16)
    image_prototypes, text_prototypes = _identity_proxy_banks(
        memory,
        use_pbt=not getattr(args, "no_pbt", False),
    )

    img_margin, img_hard_pids = _prototype_margin(
        proto_image,
        image_prototypes,
        memory.proto_pids,
        pids,
        hard_k,
    )
    txt_margin, txt_hard_pids = _prototype_margin(
        proto_text,
        text_prototypes,
        memory.proto_pids,
        pids,
        hard_k,
    )
    proto_margin = 0.5 * (img_margin + txt_margin)

    negative_proto = torch.cat([img_margin, txt_margin]).lt(0).float()
    host_hard = host_extra["hard_neg_pids"]
    overlap = []
    for row, host_pid in enumerate(host_hard.tolist()):
        hard_ids = set(img_hard_pids[row].tolist()) | set(txt_hard_pids[row].tolist())
        overlap.append(float(host_pid in hard_ids))

    metrics.update({
        "proto_margin_img_mean": _mean(img_margin),
        "proto_margin_txt_mean": _mean(txt_margin),
        "negative_proto_margin_rate": _mean(negative_proto),
        "hard_negative_overlap": _mean(torch.tensor(overlap)) if overlap else None,
        "proto_to_host_margin_corr": _corrcoef(proto_margin, host_extra["host_margin"]),
        "slot_redundancy": _slot_redundancy(memory),
    })
    metrics.update(_assignment_metrics(
        memory,
        proto_image,
        proto_text,
        pids,
        diag.get("indices"),
        state,
    ))
    return {key: value for key, value in metrics.items() if _to_float(value) is not None}
