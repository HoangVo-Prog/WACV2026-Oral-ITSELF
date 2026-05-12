import torch
import torch.nn.functional as F


def _initial_centroids(features, num_clusters, generator=None):
    perm = torch.randperm(
        features.shape[0],
        device=features.device,
        generator=generator,
    )[:num_clusters]
    return features[perm].clone()


def _cluster_means(features, assignments, centroids):
    new_centroids = centroids.clone()
    for cluster_idx in range(centroids.shape[0]):
        cluster_features = features[assignments == cluster_idx]
        if cluster_features.numel() > 0:
            new_centroids[cluster_idx] = cluster_features.mean(dim=0)
    return F.normalize(new_centroids, p=2, dim=1)


@torch.no_grad()
def torch_kmeans(features, num_clusters, num_iters=20, chunk_size=4096, generator=None):
    """Spherical K-Means over L2-normalized features."""
    if features.ndim != 2:
        raise ValueError("features must be a 2D tensor")
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")

    features = F.normalize(features.float(), p=2, dim=1)
    num_samples = features.shape[0]
    if num_samples == 0:
        raise ValueError("cannot run k-means on an empty tensor")

    if num_samples < num_clusters:
        repeats = (num_clusters + num_samples - 1) // num_samples
        centroids = features.repeat(repeats, 1)[:num_clusters].clone()
        return F.normalize(centroids, p=2, dim=1)

    centroids = _initial_centroids(features, num_clusters, generator=generator)

    for _ in range(num_iters):
        assignments = []
        for start in range(0, num_samples, chunk_size):
            sims = features[start:start + chunk_size] @ centroids.t()
            assignments.append(sims.argmax(dim=1))
        assignments = torch.cat(assignments, dim=0)

        centroids = _cluster_means(features, assignments, centroids)

    return centroids


@torch.no_grad()
def identity_kmeans(features, pids, num_classes, prototypes_per_id, num_iters=20, generator=None):
    """Build fixed-count prototypes for each identity."""
    features = F.normalize(features.float(), p=2, dim=1)
    pids = pids.long()
    dim = features.shape[1]
    global_fallback = F.normalize(features.mean(dim=0, keepdim=True), p=2, dim=1)
    banks = []

    for pid in range(num_classes):
        identity_features = features[pids == pid]
        if identity_features.numel() == 0:
            centroids = global_fallback.repeat(prototypes_per_id, 1)
        elif identity_features.shape[0] <= prototypes_per_id:
            repeats = (prototypes_per_id + identity_features.shape[0] - 1) // identity_features.shape[0]
            centroids = identity_features.repeat(repeats, 1)[:prototypes_per_id]
        else:
            centroids = torch_kmeans(
                identity_features,
                prototypes_per_id,
                num_iters=num_iters,
                generator=generator,
            )
        banks.append(F.normalize(centroids.reshape(prototypes_per_id, dim), p=2, dim=1))

    return torch.cat(banks, dim=0)
