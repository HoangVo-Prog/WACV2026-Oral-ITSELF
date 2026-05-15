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
from model.prototype.losses import (
    identity_proxy_contrastive,
    identity_proxy_contrastive_per_sample,
    symmetric_identity_proxy_loss,
)
from model.prototype.memory import PrototypeMemory


def test_torch_kmeans_shape_and_normalization():
    torch.manual_seed(1)
    features = torch.randn(12, 5)
    centroids = torch_kmeans(features, num_clusters=3, num_iters=3)

    assert centroids.shape == (3, 5)
    assert torch.allclose(centroids.norm(dim=1), torch.ones(3), atol=1e-5)


def test_torch_kmeans_respects_explicit_generator():
    features = torch.arange(60, dtype=torch.float32).reshape(12, 5)
    gen_a = torch.Generator()
    gen_b = torch.Generator()
    gen_a.manual_seed(7)
    gen_b.manual_seed(7)

    centroids_a = torch_kmeans(features, num_clusters=3, num_iters=3, generator=gen_a)
    centroids_b = torch_kmeans(features, num_clusters=3, num_iters=3, generator=gen_b)

    assert torch.allclose(centroids_a, centroids_b, atol=1e-6)


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
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=2, dim=2)
    assignments = torch.tensor([0, 0, 2, 2])
    valid, means = memory._scatter_group_means(assignments, features, memory.image_prototypes)

    expected = F.normalize(torch.stack([features[:2].mean(dim=0), features[2:].mean(dim=0)]), p=2, dim=1)
    assert torch.equal(valid, torch.tensor([0, 2]))
    assert torch.allclose(means, expected, atol=1e-6)


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

    assert scores.diag().min() > scores[0, 1]
    assert scores.diag().min() > scores[1, 0]


def test_per_sample_identity_proxy_matches_reduced_loss():
    features = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)
    prototypes = features.clone()
    pids = torch.tensor([0, 1])
    proto_pids = torch.tensor([0, 1])

    per_sample, details = identity_proxy_contrastive_per_sample(
        features, pids, prototypes, proto_pids, tau=0.1, hard_k=1
    )
    reduced = identity_proxy_contrastive(features, pids, prototypes, proto_pids, tau=0.1, hard_k=1)

    assert details["valid_mask"].all()
    assert torch.allclose(per_sample.mean(), reduced, atol=1e-6)


def test_host_aligned_pressure_uses_fixed_hard_k_and_logs_gates():
    torch.manual_seed(3)
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=1, dim=3)
    bank = F.normalize(torch.eye(3)[:2], p=2, dim=1)
    memory.image_prototypes.copy_(bank)
    memory.text_prototypes.copy_(bank)
    memory.text_to_image.copy_(bank)
    memory.image_to_text.copy_(bank)
    memory.initialized.fill_(True)

    image_features = torch.randn(4, 3, requires_grad=True)
    text_features = torch.randn(4, 3, requires_grad=True)
    host_image_features = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.6, 0.8, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    host_text_features = torch.tensor(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.7, 0.3, 0.0],
            [0.4, 0.6, 0.0],
        ]
    )
    pids = torch.tensor([0, 1, 0, 1])

    loss, details = symmetric_identity_proxy_loss(
        image_features,
        text_features,
        pids,
        memory,
        hard_k=1,
        hard_k_mode="adaptive",
        pressure_mode="host_aligned",
        host_image_features=host_image_features,
        host_text_features=host_text_features,
        return_details=True,
    )
    grads = torch.autograd.grad(loss, [image_features, text_features], allow_unused=True)

    assert torch.isfinite(loss)
    assert all(grad is not None for grad in grads)
    assert float(details["prototype_k_img"]) == 1.0
    assert float(details["prototype_k_txt"]) == 1.0
    assert torch.allclose(details["prototype_host_gate_img_mean"], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(details["prototype_host_gate_txt_mean"], torch.tensor(1.0), atol=1e-6)


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
        prototype_hard_k_mode="fixed",
        prototype_pressure_mode="fixed",
        prototype_warmup_epochs=0,
        use_loss_id=True,
    )
    branch = PrototypeBranch(args, num_classes=2, feature_dim=4)
    image_features = torch.randn(4, 4)
    text_features = torch.randn(4, 4)
    pids = torch.tensor([0, 0, 1, 1])

    cold_ret = branch(image_features, text_features, pids)
    assert cold_ret["proto_id_loss"].shape == ()

    image_projected, text_projected = branch.project_for_memory(image_features, text_features)
    branch.initialize_projected(image_projected, text_projected, pids)
    warm_ret = branch(image_features, text_features, pids)
    id_only_ret = branch(image_features, text_features, pids, use_loss_id=True)
    no_loss_ret = branch(image_features, text_features, pids, use_loss_id=False)
    scores = branch.score(text_features, image_features)

    assert torch.isfinite(warm_ret["proto_id_loss"])
    assert warm_ret["proto_id_loss"].requires_grad
    assert "prototype_host_gate_img_mean" in warm_ret
    assert "proto_id_loss" in id_only_ret
    assert no_loss_ret == {}
    assert scores.shape == (4, 4)


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
        prototype_hard_k_mode="fixed",
        prototype_pressure_mode="fixed",
        prototype_warmup_epochs=0,
        use_loss_id=True,
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
    test_torch_kmeans_respects_explicit_generator()
    test_scatter_group_means_match_deterministic_group_means_on_cpu()
    test_identity_assignment_stays_inside_identity_slots()
    test_pbt_empty_slots_fall_back_to_same_side_prototypes()
    test_prototype_score_prefers_positive_pairs()
    test_per_sample_identity_proxy_matches_reduced_loss()
    test_host_aligned_pressure_uses_fixed_hard_k_and_logs_gates()
    test_branch_forward_and_score_shapes()
    test_branch_score_accepts_cpu_features_when_module_is_cuda()
