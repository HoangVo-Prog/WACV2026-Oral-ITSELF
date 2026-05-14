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
    features = F.normalize(features.float(), p=2, dim=1)
    prototypes = F.normalize(prototypes.to(features.device).float(), p=2, dim=1)
    proto_pids = proto_pids.to(features.device)
    pids = pids.long().to(features.device)

    logits = features @ prototypes.t()
    pos_mask = proto_pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not neg_mask.any():
        return features.new_zeros(())

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
        return features.new_zeros(())

    denom = torch.logaddexp(pos_lse[valid], neg_lse[valid])
    return -(pos_lse[valid] - denom).mean()


def symmetric_identity_proxy_loss(
    image_features,
    text_features,
    pids,
    memory,
    tau=0.05,
    hard_k=16,
    hard_k_mode="fixed",
    epoch=None,
    warmup_epochs=0,
    return_details=False,
):
    hard_k = _positive_int(hard_k, default=16)
    if str(hard_k_mode).lower() == "adaptive":
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

    image_loss = identity_proxy_contrastive(
        image_features, pids, memory.text_to_image, memory.proto_pids, tau=tau, hard_k=image_hard_k
    )
    text_loss = identity_proxy_contrastive(
        text_features, pids, memory.image_to_text, memory.proto_pids, tau=tau, hard_k=text_hard_k
    )
    loss = 0.5 * (image_loss + text_loss)
    if not return_details:
        return loss
    details = {
        "prototype_k_img": image_features.new_tensor(float(image_hard_k)),
        "prototype_k_txt": text_features.new_tensor(float(text_hard_k)),
    }
    return loss, details
