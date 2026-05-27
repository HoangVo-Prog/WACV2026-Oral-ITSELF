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


def _batch_host_metrics(image_features, text_features, pids):
    image_features = F.normalize(image_features.float(), p=2, dim=1)
    text_features = F.normalize(text_features.float(), p=2, dim=1)
    pids = pids.long().to(image_features.device)

    sims = text_features @ image_features.t()
    pos_mask = pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask

    pos_logits = sims.masked_fill(~pos_mask, float("-inf"))
    pos_for_min = sims.masked_fill(~pos_mask, float("inf"))
    neg_logits = sims.masked_fill(~neg_mask, float("-inf"))

    best_pos = pos_logits.max(dim=1).values
    worst_pos = pos_for_min.min(dim=1).values
    hard_neg, hard_neg_idx = neg_logits.max(dim=1)
    hard_neg_pids = pids[hard_neg_idx].detach().cpu()

    host_margin = best_pos - hard_neg
    hard_pos_margin = worst_pos - hard_neg
    negative_intrusion = hard_neg > best_pos

    sorted_idx = sims.argsort(dim=1, descending=True)
    sorted_pos = pos_mask.gather(1, sorted_idx)
    first_pos_rank = sorted_pos.float().argmax(dim=1).float() + 1.0

    metrics = {
        "host_margin_mean": _mean(host_margin.detach().cpu()),
        "host_margin_p10": _quantile(host_margin.detach().cpu(), 0.10),
        "hard_pos_margin_mean": _mean(hard_pos_margin.detach().cpu()),
        "negative_intrusion_rate": _mean(negative_intrusion.float().detach().cpu()),
        "mean_first_positive_rank": _mean(first_pos_rank.detach().cpu()),
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
    metrics, host_extra = _batch_host_metrics(host_image, host_text, pids)

    model = _unwrap_model(model)
    branch = getattr(model, "prototype_branch", None)
    if branch is None or not branch.is_ready():
        return metrics

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
