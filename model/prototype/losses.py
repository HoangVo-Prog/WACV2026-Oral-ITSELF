import math

import torch
import torch.nn.functional as F


def _masked_logsumexp(values, mask, dim):
    return values.masked_fill(~mask, float("-inf")).logsumexp(dim=dim)


def _positive_int(value, default=1):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(value, 1)


def _schedule_progress(epoch, warmup_epochs, ramp_epochs=3):
    if epoch is None:
        return 1.0
    try:
        progress = (float(epoch) - float(warmup_epochs)) / float(ramp_epochs)
    except (TypeError, ValueError, ZeroDivisionError):
        return 1.0
    return max(0.0, min(1.0, progress))


def _ceil_quantile(values, q):
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    values = torch.sort(values.float()).values
    idx = int(math.ceil((values.numel() - 1) * q))
    return int(math.ceil(values[idx].item()))


def _gate_stats(gate):
    gate = gate.detach().float()
    if gate.numel() == 0:
        zero = gate.new_tensor(0.0)
        return {"mean": zero, "p10": zero, "p90": zero}
    return {
        "mean": gate.mean(),
        "p10": torch.quantile(gate, 0.10),
        "p90": torch.quantile(gate, 0.90),
    }


def _uniform_gate(batch_size, device):
    gate = torch.ones(batch_size, device=device, dtype=torch.float32)
    return gate, _gate_stats(gate)


def _host_margins_from_similarity(similarity, pids):
    pids = pids.long().to(similarity.device)
    pos_mask = pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask

    pos = similarity.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg = similarity.masked_fill(~neg_mask, float("-inf")).max(dim=1).values
    margin = pos - neg
    valid = torch.isfinite(pos) & torch.isfinite(neg)
    return margin, valid


@torch.no_grad()
def _host_aligned_gates(image_host_features, text_host_features, pids, eps=1e-6):
    batch_size = int(pids.shape[0]) if pids is not None else 0
    if image_host_features is not None:
        device = image_host_features.device
    elif text_host_features is not None:
        device = text_host_features.device
    elif pids is not None:
        device = pids.device
    else:
        device = torch.device("cpu")

    img_gate, img_stats = _uniform_gate(batch_size, device)
    txt_gate, txt_stats = _uniform_gate(batch_size, device)

    if (
        batch_size < 2
        or image_host_features is None
        or text_host_features is None
    ):
        return img_gate, txt_gate, {
            "prototype_host_gate_img_mean": img_stats["mean"],
            "prototype_host_gate_img_p10": img_stats["p10"],
            "prototype_host_gate_img_p90": img_stats["p90"],
            "prototype_host_gate_txt_mean": txt_stats["mean"],
            "prototype_host_gate_txt_p10": txt_stats["p10"],
            "prototype_host_gate_txt_p90": txt_stats["p90"],
        }

    image_host_features = F.normalize(image_host_features.detach().float().to(device), p=2, dim=1)
    text_host_features = F.normalize(text_host_features.detach().float().to(device), p=2, dim=1)
    pids = pids.long().to(device)

    if pids.unique(sorted=False).numel() < 2:
        return img_gate, txt_gate, {
            "prototype_host_gate_img_mean": img_stats["mean"],
            "prototype_host_gate_img_p10": img_stats["p10"],
            "prototype_host_gate_img_p90": img_stats["p90"],
            "prototype_host_gate_txt_mean": txt_stats["mean"],
            "prototype_host_gate_txt_p10": txt_stats["p10"],
            "prototype_host_gate_txt_p90": txt_stats["p90"],
        }

    sims = text_host_features @ image_host_features.t()
    txt_margin, txt_valid = _host_margins_from_similarity(sims, pids)
    img_margin, img_valid = _host_margins_from_similarity(sims.t(), pids)

    if bool(txt_valid.all().item()):
        txt_std = txt_margin.float().std(unbiased=False)
        if bool(torch.isfinite(txt_std).item()) and txt_std.item() > eps:
            txt_z = (txt_margin.float() - txt_margin.float().mean()) / (txt_std + eps)
            txt_gate = batch_size * torch.softmax(-txt_z, dim=0)
            txt_stats = _gate_stats(txt_gate)

    if bool(img_valid.all().item()):
        img_std = img_margin.float().std(unbiased=False)
        if bool(torch.isfinite(img_std).item()) and img_std.item() > eps:
            img_z = (img_margin.float() - img_margin.float().mean()) / (img_std + eps)
            img_gate = batch_size * torch.softmax(-img_z, dim=0)
            img_stats = _gate_stats(img_gate)

    return img_gate, txt_gate, {
        "prototype_host_gate_img_mean": img_stats["mean"],
        "prototype_host_gate_img_p10": img_stats["p10"],
        "prototype_host_gate_img_p90": img_stats["p90"],
        "prototype_host_gate_txt_mean": txt_stats["mean"],
        "prototype_host_gate_txt_p10": txt_stats["p10"],
        "prototype_host_gate_txt_p90": txt_stats["p90"],
    }


@torch.no_grad()
def adaptive_hard_negative_k(
    features,
    pids,
    prototypes,
    proto_pids,
    tau=0.05,
    anchor_hard_k=16,
    epoch=None,
    warmup_epochs=0,
):
    """Select one batch-level hard-negative budget for a prototype direction."""
    base_k = _positive_int(anchor_hard_k, default=16)
    if features.numel() == 0 or prototypes.numel() == 0:
        return base_k

    features = F.normalize(features.detach().float(), p=2, dim=1)
    prototypes = F.normalize(prototypes.detach().to(features.device).float(), p=2, dim=1)
    proto_pids = proto_pids.to(features.device)
    pids = pids.long().to(features.device)

    logits = features @ prototypes.t()
    pos_mask = proto_pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not neg_mask.any():
        return base_k

    pos = logits.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
    hardest_neg = neg_logits.max(dim=1).values
    neg_pool = neg_mask.sum(dim=1)
    valid = torch.isfinite(pos) & torch.isfinite(hardest_neg) & neg_pool.gt(0)
    if not valid.any():
        return base_k

    pool_size = int(neg_pool[valid].min().item())
    if pool_size <= 0:
        return base_k

    k_min = min(max(1, int(round(base_k * 0.25))), pool_size)
    k_max = min(pool_size, max(k_min, int(round(base_k * 2.0))))

    band = max(float(tau), 1e-6) * math.log(4.0)
    near_threshold = pos.unsqueeze(1) - band
    near_counts = (neg_mask & (logits >= near_threshold)).sum(dim=1).float()[valid]
    candidate = _ceil_quantile(near_counts, 0.75)
    if candidate is None:
        candidate = base_k
    candidate = max(k_min, min(k_max, candidate))

    margins = (pos - hardest_neg)[valid]
    intrusion = margins.lt(0).float().mean().item()
    intrusion_guard = max(0.0, min(1.0, 1.0 - intrusion / 0.35))
    progress = _schedule_progress(epoch, warmup_epochs)
    scheduled = k_min + progress * intrusion_guard * (candidate - k_min)
    return max(1, min(k_max, int(round(scheduled))))


def identity_proxy_contrastive(features, pids, prototypes, proto_pids, tau=0.05, hard_k=16):
    losses, details = identity_proxy_contrastive_per_sample(
        features, pids, prototypes, proto_pids, tau=tau, hard_k=hard_k
    )
    valid = details["valid_mask"]
    if not valid.any():
        return features.new_zeros(())
    return losses[valid].mean()


def identity_proxy_contrastive_per_sample(features, pids, prototypes, proto_pids, tau=0.05, hard_k=16):
    features = F.normalize(features.float(), p=2, dim=1)
    prototypes = F.normalize(prototypes.to(features.device).float(), p=2, dim=1)
    proto_pids = proto_pids.to(features.device)
    pids = pids.long().to(features.device)

    batch_size = features.shape[0]
    losses = features.new_zeros(batch_size)
    valid = torch.zeros(batch_size, dtype=torch.bool, device=features.device)
    if features.numel() == 0 or prototypes.numel() == 0:
        return losses, {"valid_mask": valid, "hard_k": features.new_tensor(float(_positive_int(hard_k, default=16)))}

    logits = features @ prototypes.t()
    pos_mask = proto_pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not neg_mask.any():
        return losses, {"valid_mask": valid, "hard_k": features.new_tensor(float(_positive_int(hard_k, default=16)))}

    pos_lse = _masked_logsumexp(logits / tau, pos_mask, dim=1)
    neg_fill = float("-inf")
    neg_logits = (logits / tau).masked_fill(~neg_mask, neg_fill)
    k = min(_positive_int(hard_k, default=16), neg_logits.shape[1])
    hard_values, hard_idx = neg_logits.topk(k=k, dim=1)
    hard_mask = torch.zeros_like(neg_mask)
    hard_mask.scatter_(1, hard_idx, torch.isfinite(hard_values))
    neg_lse = _masked_logsumexp(logits / tau, hard_mask, dim=1)

    valid = torch.isfinite(pos_lse) & torch.isfinite(neg_lse)
    if not valid.any():
        return losses, {"valid_mask": valid, "hard_k": features.new_tensor(float(k))}

    denom = torch.logaddexp(pos_lse[valid], neg_lse[valid])
    losses[valid] = -(pos_lse[valid] - denom)
    return losses, {"valid_mask": valid, "hard_k": features.new_tensor(float(k))}


def symmetric_identity_proxy_loss(
    image_features,
    text_features,
    pids,
    memory,
    tau=0.05,
    hard_k=16,
    hard_k_mode="fixed",
    pressure_mode="fixed",
    host_image_features=None,
    host_text_features=None,
    epoch=None,
    warmup_epochs=0,
    return_details=False,
):
    hard_k = _positive_int(hard_k, default=16)
    pressure_mode = str(pressure_mode).lower()

    if pressure_mode != "host_aligned" and str(hard_k_mode).lower() == "adaptive":
        image_hard_k = adaptive_hard_negative_k(
            image_features,
            pids,
            memory.text_to_image,
            memory.proto_pids,
            tau=tau,
            anchor_hard_k=hard_k,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
        )
        text_hard_k = adaptive_hard_negative_k(
            text_features,
            pids,
            memory.image_to_text,
            memory.proto_pids,
            tau=tau,
            anchor_hard_k=hard_k,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
        )
    else:
        image_hard_k = hard_k
        text_hard_k = hard_k

    image_losses, image_details = identity_proxy_contrastive_per_sample(
        image_features, pids, memory.text_to_image, memory.proto_pids, tau=tau, hard_k=image_hard_k
    )
    text_losses, text_details = identity_proxy_contrastive_per_sample(
        text_features, pids, memory.image_to_text, memory.proto_pids, tau=tau, hard_k=text_hard_k
    )

    image_valid = image_details["valid_mask"]
    text_valid = text_details["valid_mask"]

    if pressure_mode == "host_aligned":
        image_gate, text_gate, gate_details = _host_aligned_gates(
            host_image_features, host_text_features, pids
        )
        image_gate = image_gate.to(image_losses.device)
        text_gate = text_gate.to(text_losses.device)
        image_loss = (
            (image_gate[image_valid] * image_losses[image_valid]).mean()
            if image_valid.any()
            else image_losses.new_zeros(())
        )
        text_loss = (
            (text_gate[text_valid] * text_losses[text_valid]).mean()
            if text_valid.any()
            else text_losses.new_zeros(())
        )
    else:
        image_gate, text_gate, gate_details = _host_aligned_gates(
            None, None, pids
        )
        image_loss = (
            image_losses[image_valid].mean()
            if image_valid.any()
            else image_losses.new_zeros(())
        )
        text_loss = (
            text_losses[text_valid].mean()
            if text_valid.any()
            else text_losses.new_zeros(())
        )

    loss = 0.5 * (image_loss + text_loss)
    if not return_details:
        return loss
    details = {
        "prototype_k_img": image_features.new_tensor(float(image_hard_k)),
        "prototype_k_txt": text_features.new_tensor(float(text_hard_k)),
    }
    details.update(gate_details)
    return loss, details
