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
    value = torch.quantile(values.float(), q)
    if not torch.isfinite(value):
        return None
    return int(math.ceil(float(value.item())))


def _mean_or_none(values):
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return float(values.float().mean().item())


def _corrcoef_or_none(x, y):
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
    value = x @ y / denom
    if not torch.isfinite(value):
        return None
    return float(value.item())


def _host_direction_margin(query_features, gallery_features, pids):
    query_features = F.normalize(query_features.float(), p=2, dim=1)
    gallery_features = F.normalize(gallery_features.float(), p=2, dim=1)
    pids = pids.long().to(query_features.device)

    sims = query_features @ gallery_features.t()
    pos_mask = pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not neg_mask.any():
        return None, None

    pos = sims.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg_logits = sims.masked_fill(~neg_mask, float("-inf"))
    neg, neg_idx = neg_logits.max(dim=1)
    margin = pos - neg
    hard_pids = pids[neg_idx]
    valid = torch.isfinite(margin)
    if not valid.any():
        return None, None
    return margin.detach(), hard_pids.detach()


def _detail_tensor(reference, value):
    if value is None:
        value = float("nan")
    return reference.new_tensor(float(value))


@torch.no_grad()
def calibrated_hard_negative_k(
    features,
    pids,
    prototypes,
    proto_pids,
    tau=0.05,
    anchor_hard_k=16,
    epoch=None,
    warmup_epochs=0,
    host_query_features=None,
    host_gallery_features=None,
    state=None,
    state_key=None,
):
    base_k = _positive_int(anchor_hard_k, default=16)
    default_details = {
        "k_raw": float(base_k),
        "k_target": float(base_k),
        "k_min": float(max(1, int(round(base_k * 0.25)))),
        "k_max": float(base_k),
        "n_eff_mean": float("nan"),
        "n_eff_q75": float("nan"),
        "unsafe_rate": float("nan"),
        "easy_rate": float("nan"),
        "intrusion": float("nan"),
        "intrusion_guard": 1.0,
        "alignment_guard": 1.0,
        "corr": float("nan"),
        "overlap": float("nan"),
        "gap_q05": float("nan"),
        "gap_q50": float("nan"),
        "gap_q75": float("nan"),
        "fallback": 1.0,
    }

    if features.numel() == 0 or prototypes.numel() == 0:
        return base_k, default_details

    features = F.normalize(features.detach().float(), p=2, dim=1)
    prototypes = F.normalize(prototypes.detach().to(features.device).float(), p=2, dim=1)
    proto_pids = proto_pids.to(features.device)
    pids = pids.long().to(features.device)

    if pids.unique().numel() < 2:
        return base_k, default_details

    logits = features @ prototypes.t()
    pos_mask = proto_pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not neg_mask.any():
        return base_k, default_details

    pos = logits.masked_fill(~pos_mask, float("-inf")).max(dim=1).values
    neg_logits = logits.masked_fill(~neg_mask, float("-inf"))
    hardest_neg, hardest_idx = neg_logits.max(dim=1)
    neg_pool = neg_mask.sum(dim=1)
    valid = torch.isfinite(pos) & torch.isfinite(hardest_neg) & neg_pool.gt(0)
    if not valid.any():
        return base_k, default_details

    pool_size = int(neg_pool[valid].min().item())
    if pool_size <= 0:
        return base_k, default_details

    probe_k = min(pool_size, max(4 * base_k, base_k + 8))
    top_values, top_idx = neg_logits.topk(k=probe_k, dim=1)
    probe_valid = torch.isfinite(top_values) & valid.unsqueeze(1)
    gaps = pos.unsqueeze(1) - top_values
    finite_gaps = gaps[probe_valid & torch.isfinite(gaps)]
    if finite_gaps.numel() == 0:
        return base_k, default_details

    q05 = torch.quantile(finite_gaps.float(), 0.05)
    q50 = torch.quantile(finite_gaps.float(), 0.50)
    q75 = torch.quantile(finite_gaps.float(), 0.75)
    if not (torch.isfinite(q05) and torch.isfinite(q50) and torch.isfinite(q75)):
        return base_k, default_details

    low = max(0.0, float(q05.item()))
    high = float(q75.item())
    if high <= low:
        return base_k, default_details

    safe = probe_valid & gaps.gt(low) & gaps.le(high)
    if not safe.any():
        return base_k, default_details

    tau = max(float(tau), 1e-6)
    safe_gaps = gaps.masked_fill(~safe, 0.0)
    weights = torch.exp((-safe_gaps / tau).clamp(min=-80.0, max=80.0)).masked_fill(~safe, 0.0)
    sum_w = weights.sum(dim=1)
    sum_w2 = weights.pow(2).sum(dim=1)
    n_eff = sum_w.pow(2) / sum_w2.clamp_min(1e-12)
    n_eff = n_eff.masked_fill(sum_w.le(0), 0.0)
    valid_eff = n_eff[valid]

    k_raw = _ceil_quantile(valid_eff, 0.75)
    n_eff_q90 = _ceil_quantile(valid_eff, 0.90)
    if k_raw is None or n_eff_q90 is None:
        return base_k, default_details

    k_min = min(max(1, int(round(base_k * 0.25))), pool_size)
    k_max = min(pool_size, max(int(round(2.0 * base_k)), n_eff_q90, k_min))
    k_candidate = max(k_min, min(k_max, k_raw))

    margins = (pos - hardest_neg)[valid]
    intrusion = margins.lt(0).float().mean().item()
    intrusion_guard = max(0.0, min(1.0, 1.0 - intrusion / 0.30))
    progress = _schedule_progress(epoch, warmup_epochs)

    corr = None
    overlap = None
    alignment_guard = 1.0
    if host_query_features is not None and host_gallery_features is not None:
        host_margin, host_hard_pids = _host_direction_margin(
            host_query_features.detach().to(features.device),
            host_gallery_features.detach().to(features.device),
            pids,
        )
        if host_margin is not None and host_hard_pids is not None:
            corr = _corrcoef_or_none(margins.detach().cpu(), host_margin[valid].detach().cpu())
            corr_value = 0.0 if corr is None else max(0.0, min(1.0, corr))
            corr_guard = 0.5 + 0.5 * corr_value

            base_probe_k = min(pool_size, base_k)
            base_hard_pids = proto_pids[top_idx[:, :base_probe_k]].detach().cpu()
            host_pids_cpu = host_hard_pids.detach().cpu()
            overlap_values = []
            for row, host_pid in enumerate(host_pids_cpu.tolist()):
                if not bool(valid[row].item()):
                    continue
                overlap_values.append(float(host_pid in set(base_hard_pids[row].tolist())))
            if overlap_values:
                overlap = float(sum(overlap_values) / len(overlap_values))
            overlap_value = 0.0 if overlap is None else max(0.0, min(1.0, overlap))
            overlap_guard = 0.5 + 0.5 * overlap_value
            alignment_guard = 0.5 * corr_guard + 0.5 * overlap_guard

    k_target = k_min + progress * intrusion_guard * alignment_guard * (k_candidate - k_min)
    previous_k = None if state is None or state_key is None else state.get(state_key)
    if previous_k is None:
        smoothed = k_target
    else:
        smoothed = 0.7 * float(previous_k) + 0.3 * k_target
    k_final = max(1, min(k_max, int(round(smoothed))))
    if state is not None and state_key is not None:
        state[state_key] = float(k_final)

    total_probe = probe_valid.float().sum().clamp_min(1.0)
    details = {
        "k_raw": float(k_raw),
        "k_target": float(k_target),
        "k_min": float(k_min),
        "k_max": float(k_max),
        "n_eff_mean": _mean_or_none(valid_eff.detach().cpu()),
        "n_eff_q75": float(k_raw),
        "unsafe_rate": float(((probe_valid & gaps.le(low)).float().sum() / total_probe).item()),
        "easy_rate": float(((probe_valid & gaps.gt(high)).float().sum() / total_probe).item()),
        "intrusion": float(intrusion),
        "intrusion_guard": float(intrusion_guard),
        "alignment_guard": float(alignment_guard),
        "corr": corr,
        "overlap": overlap,
        "gap_q05": float(q05.item()),
        "gap_q50": float(q50.item()),
        "gap_q75": float(q75.item()),
        "fallback": 0.0,
    }
    return k_final, details


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


def _prefix_details(reference, prefix, details):
    return {
        f"prototype_{prefix}_{key}": _detail_tensor(reference, value)
        for key, value in details.items()
    }


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
    host_image_features=None,
    host_text_features=None,
    scheduler_state=None,
    return_details=False,
):
    hard_k = _positive_int(hard_k, default=16)
    image_hard_k = hard_k
    text_hard_k = hard_k
    image_details = None
    text_details = None

    if str(hard_k_mode).lower() == "calibrated":
        image_hard_k, image_details = calibrated_hard_negative_k(
            image_features,
            pids,
            memory.text_to_image,
            memory.proto_pids,
            tau=tau,
            anchor_hard_k=hard_k,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            host_query_features=host_image_features,
            host_gallery_features=host_text_features,
            state=scheduler_state,
            state_key="img",
        )
        text_hard_k, text_details = calibrated_hard_negative_k(
            text_features,
            pids,
            memory.image_to_text,
            memory.proto_pids,
            tau=tau,
            anchor_hard_k=hard_k,
            epoch=epoch,
            warmup_epochs=warmup_epochs,
            host_query_features=host_text_features,
            host_gallery_features=host_image_features,
            state=scheduler_state,
            state_key="txt",
        )

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
    if image_details is not None:
        details.update(_prefix_details(image_features, "img", image_details))
    if text_details is not None:
        details.update(_prefix_details(text_features, "txt", text_details))
    return loss, details
