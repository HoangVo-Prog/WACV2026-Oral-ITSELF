import torch
import torch.nn.functional as F


def _masked_logsumexp(values, mask, dim):
    return values.masked_fill(~mask, float("-inf")).logsumexp(dim=dim)


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
    k = min(hard_k, neg_logits.shape[1])
    hard_values, hard_idx = neg_logits.topk(k=k, dim=1)
    hard_mask = torch.zeros_like(neg_mask)
    hard_mask.scatter_(1, hard_idx, torch.isfinite(hard_values))
    neg_lse = _masked_logsumexp(logits / tau, hard_mask, dim=1)

    valid = torch.isfinite(pos_lse) & torch.isfinite(neg_lse)
    if not valid.any():
        return features.new_zeros(())

    denom = torch.logaddexp(pos_lse[valid], neg_lse[valid])
    return -(pos_lse[valid] - denom).mean()


def symmetric_identity_proxy_loss(image_features, text_features, pids, memory, tau=0.05, hard_k=16):
    image_loss = identity_proxy_contrastive(
        image_features, pids, memory.text_to_image, memory.proto_pids, tau=tau, hard_k=hard_k
    )
    text_loss = identity_proxy_contrastive(
        text_features, pids, memory.image_to_text, memory.proto_pids, tau=tau, hard_k=hard_k
    )
    return 0.5 * (image_loss + text_loss)


def _directional_rank_loss(proto_scores, pids, selection_scores, margin=0.2, hard_k=16):
    device = proto_scores.device
    pids = pids.long().to(device)
    pos_mask = pids.unsqueeze(0).eq(pids.unsqueeze(1))
    neg_mask = ~pos_mask
    if not pos_mask.any() or not neg_mask.any():
        return proto_scores.new_zeros(())

    pos_lse = _masked_logsumexp(proto_scores, pos_mask, dim=1)
    neg_fill = float("-inf")
    selector = selection_scores.detach().masked_fill(~neg_mask, neg_fill)
    k = min(hard_k, selector.shape[1])
    hard_values, hard_idx = selector.topk(k=k, dim=1)
    hard_mask = torch.zeros_like(neg_mask)
    hard_mask.scatter_(1, hard_idx, torch.isfinite(hard_values))
    neg_lse = _masked_logsumexp(proto_scores, hard_mask, dim=1)

    valid = torch.isfinite(pos_lse) & torch.isfinite(neg_lse)
    if not valid.any():
        return proto_scores.new_zeros(())

    return F.softplus(neg_lse[valid] - pos_lse[valid] + margin).mean()


def prototype_pair_ranking_loss(proto_scores, pids, host_scores=None, margin=0.2, hard_k=16):
    if host_scores is None:
        host_scores = proto_scores
    host_scores = host_scores.to(proto_scores.device).float()
    proto_scores = proto_scores.float()

    text_to_image = _directional_rank_loss(proto_scores, pids, host_scores, margin=margin, hard_k=hard_k)
    image_to_text = _directional_rank_loss(
        proto_scores.t(), pids, host_scores.t(), margin=margin, hard_k=hard_k
    )
    return 0.5 * (text_to_image + image_to_text)
