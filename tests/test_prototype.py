import os
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

model_pkg = types.ModuleType("model")
model_pkg.__path__ = [os.path.join(REPO_ROOT, "model")]
sys.modules.setdefault("model", model_pkg)

prototype_pkg = types.ModuleType("model.prototype")
prototype_pkg.__path__ = [os.path.join(REPO_ROOT, "model", "prototype")]
sys.modules.setdefault("model.prototype", prototype_pkg)

from model.prototype.kmeans import torch_kmeans
from model.prototype.build import PrototypeBranch
from model.prototype.losses import prototype_pair_ranking_loss
from model.prototype.memory import PrototypeMemory


def test_torch_kmeans_shape_and_normalization():
    torch.manual_seed(1)
    features = torch.randn(12, 5)
    centroids = torch_kmeans(features, num_clusters=3, num_iters=3)

    assert centroids.shape == (3, 5)
    assert torch.allclose(centroids.norm(dim=1), torch.ones(3), atol=1e-5)


def test_torch_kmeans_does_not_consume_rng_state():
    features = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    torch.manual_seed(7)
    rng_before = torch.get_rng_state()

    _ = torch_kmeans(features, num_clusters=3, num_iters=3)

    assert torch.equal(torch.get_rng_state(), rng_before)


def test_torch_kmeans_random_init_consumes_rng_state():
    features = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    torch.manual_seed(7)
    rng_before = torch.get_rng_state()

    _ = torch_kmeans(features, num_clusters=3, num_iters=3, init_method="random")

    assert not torch.equal(torch.get_rng_state(), rng_before)


def test_scatter_group_means_match_deterministic_group_means_on_cpu():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ]
        ),
        p=2,
        dim=1,
    )
    pids = torch.tensor([0, 0, 1, 1])
    deterministic = PrototypeMemory(num_classes=2, prototypes_per_id=2, dim=2)
    scatter = PrototypeMemory(num_classes=2, prototypes_per_id=2, dim=2, group_mean_impl="scatter")

    deterministic.initialize(features, features, pids, num_iters=2)
    scatter.initialize(features, features, pids, num_iters=2)

    assert torch.allclose(scatter.image_prototypes, deterministic.image_prototypes, atol=1e-6)
    assert torch.allclose(scatter.text_to_image, deterministic.text_to_image, atol=1e-6)


def test_identity_assignment_stays_inside_identity_slots():
    features = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ]
        ),
        p=2,
        dim=1,
    )
    pids = torch.tensor([0, 0, 1, 1])
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=2, dim=2)
    memory.initialize(features, features, pids, num_iters=2)

    assignments = memory.assign_identity(features, pids, memory.image_prototypes)
    assigned_pids = memory.proto_pids[assignments]

    assert torch.equal(assigned_pids, pids)


def test_pbt_empty_slots_fall_back_to_same_side_prototypes():
    image_features = F.normalize(torch.tensor([[1.0, 0.0]]), p=2, dim=1)
    text_features = F.normalize(torch.tensor([[0.0, 1.0]]), p=2, dim=1)
    pids = torch.tensor([0])
    memory = PrototypeMemory(num_classes=1, prototypes_per_id=2, dim=2)
    memory.initialize(image_features, text_features, pids, num_iters=2)

    assert torch.allclose(memory.text_to_image[1], memory.image_prototypes[1], atol=1e-6)
    assert torch.allclose(memory.image_to_text[1], memory.text_prototypes[1], atol=1e-6)


def test_prototype_score_prefers_positive_pairs():
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=1, dim=2)
    image_bank = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)
    text_bank = image_bank.clone()
    memory.image_prototypes.copy_(image_bank)
    memory.text_prototypes.copy_(text_bank)
    memory.text_to_image.copy_(image_bank)
    memory.image_to_text.copy_(text_bank)
    memory.initialized.fill_(True)

    scores = memory.prototype_score_matrix(text_bank, image_bank)
    pids = torch.tensor([0, 1])
    loss = prototype_pair_ranking_loss(scores, pids, hard_k=1)

    assert scores.diag().min() > scores[0, 1]
    assert scores.diag().min() > scores[1, 0]
    assert torch.isfinite(loss)


def test_training_score_keeps_rank_loss_differentiable():
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=1, dim=2)
    image_bank = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)
    text_bank = image_bank.clone()
    memory.image_prototypes.copy_(image_bank)
    memory.text_prototypes.copy_(text_bank)
    memory.text_to_image.copy_(image_bank)
    memory.image_to_text.copy_(text_bank)
    memory.initialized.fill_(True)

    image_features = image_bank.clone().requires_grad_(True)
    text_features = text_bank.clone().requires_grad_(True)
    scores = memory.training_score_matrix(text_features, image_features)
    loss = prototype_pair_ranking_loss(scores, torch.tensor([0, 1]), hard_k=1)
    grads = torch.autograd.grad(loss, [image_features, text_features], allow_unused=True)

    assert loss.requires_grad
    assert all(grad is not None for grad in grads)


def test_branch_forward_and_score_shapes():
    args = SimpleNamespace(
        only_global=True,
        prototype_feature="auto",
        prototype_dim=3,
        prototype_per_id=1,
        prototype_momentum=0.2,
        prototype_kmeans_iters=2,
        prototype_tau=0.05,
        prototype_hard_k=1,
        prototype_margin=0.2,
        use_loss_id=True,
        use_loss_rank=True,
    )
    branch = PrototypeBranch(args, num_classes=2, feature_dim=4)
    image_features = torch.randn(4, 4)
    text_features = torch.randn(4, 4)
    pids = torch.tensor([0, 0, 1, 1])

    cold_ret = branch(image_features, text_features, pids)
    assert cold_ret["proto_id_loss"].shape == ()
    assert cold_ret["proto_rank_loss"].shape == ()

    image_projected, text_projected = branch.project_for_memory(image_features, text_features)
    branch.initialize_projected(image_projected, text_projected, pids)
    warm_ret = branch(image_features, text_features, pids)
    id_only_ret = branch(image_features, text_features, pids, use_loss_id=True, use_loss_rank=False)
    rank_only_ret = branch(image_features, text_features, pids, use_loss_id=False, use_loss_rank=True)
    scores = branch.score(text_features, image_features)

    assert torch.isfinite(warm_ret["proto_id_loss"])
    assert torch.isfinite(warm_ret["proto_rank_loss"])
    assert warm_ret["proto_rank_loss"].requires_grad
    assert list(id_only_ret.keys()) == ["proto_id_loss"]
    assert list(rank_only_ret.keys()) == ["proto_rank_loss"]
    assert scores.shape == (4, 4)


def test_rank_hard_negative_source_can_use_host_or_projected_scores():
    args = SimpleNamespace(
        only_global=True,
        prototype_feature="auto",
        prototype_dim=3,
        prototype_per_id=1,
        prototype_momentum=0.2,
        prototype_hard_negative_source="host",
    )
    branch = PrototypeBranch(args, num_classes=2, feature_dim=4)
    image_features = F.normalize(torch.randn(4, 3), p=2, dim=1)
    text_features = F.normalize(torch.randn(4, 3), p=2, dim=1)
    host_scores = torch.randn(4, 4)

    selected = branch._rank_selection_scores(text_features, image_features, host_scores=host_scores)
    assert torch.allclose(selected, host_scores)

    args.prototype_hard_negative_source = "projected"
    projected = branch._rank_selection_scores(text_features, image_features, host_scores=host_scores)
    expected_projected = text_features @ image_features.t()
    assert torch.allclose(projected, expected_projected)


def test_branch_score_accepts_cpu_features_when_module_is_cuda():
    if not torch.cuda.is_available():
        return

    args = SimpleNamespace(
        only_global=True,
        prototype_feature="auto",
        prototype_dim=3,
        prototype_per_id=1,
        prototype_momentum=0.2,
        prototype_kmeans_iters=2,
        prototype_tau=0.05,
        prototype_hard_k=1,
        prototype_margin=0.2,
        use_loss_id=True,
        use_loss_rank=True,
    )
    branch = PrototypeBranch(args, num_classes=2, feature_dim=4).cuda()
    image_features = torch.randn(4, 4)
    text_features = torch.randn(4, 4)
    pids = torch.tensor([0, 0, 1, 1])

    image_projected, text_projected = branch.project_for_memory(image_features.cuda(), text_features.cuda())
    branch.initialize_projected(image_projected.cpu(), text_projected.cpu(), pids)
    scores = branch.score(text_features.cpu(), image_features.cpu())

    assert scores.is_cuda
    assert scores.shape == (4, 4)


if __name__ == "__main__":
    test_torch_kmeans_shape_and_normalization()
    test_torch_kmeans_does_not_consume_rng_state()
    test_torch_kmeans_random_init_consumes_rng_state()
    test_scatter_group_means_match_deterministic_group_means_on_cpu()
    test_identity_assignment_stays_inside_identity_slots()
    test_pbt_empty_slots_fall_back_to_same_side_prototypes()
    test_prototype_score_prefers_positive_pairs()
    test_training_score_keeps_rank_loss_differentiable()
    test_branch_forward_and_score_shapes()
    test_branch_score_accepts_cpu_features_when_module_is_cuda()
