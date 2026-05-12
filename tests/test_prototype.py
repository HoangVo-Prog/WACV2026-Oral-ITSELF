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

from model.prototype.build import PrototypeBranch
from model.prototype.kmeans import torch_kmeans
from model.prototype.losses import symmetric_identity_proxy_loss
from model.prototype.memory import PrototypeMemory


def _branch_args(**overrides):
    args = SimpleNamespace(
        only_global=True,
        prototype_feature="auto",
        prototype_dim=3,
        prototype_per_id=1,
        prototype_momentum=0.2,
        prototype_kmeans_iters=2,
        prototype_tau=0.05,
        prototype_hard_k=1,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_torch_kmeans_shape_and_normalization():
    torch.manual_seed(1)
    features = torch.randn(12, 5)
    centroids = torch_kmeans(features, num_clusters=3, num_iters=3)

    assert centroids.shape == (3, 5)
    assert torch.allclose(centroids.norm(dim=1), torch.ones(3), atol=1e-5)


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


def test_symmetric_identity_proxy_loss_is_finite():
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=1, dim=2)
    image_bank = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)
    text_bank = image_bank.clone()
    memory.image_prototypes.copy_(image_bank)
    memory.text_prototypes.copy_(text_bank)
    memory.text_to_image.copy_(image_bank)
    memory.image_to_text.copy_(text_bank)
    memory.initialized.fill_(True)

    loss = symmetric_identity_proxy_loss(image_bank, text_bank, torch.tensor([0, 1]), memory, hard_k=1)

    assert torch.isfinite(loss)


def test_branch_forward_returns_only_id_regularizer():
    branch = PrototypeBranch(_branch_args(), num_classes=2, feature_dim=4)
    image_features = torch.randn(4, 4)
    text_features = torch.randn(4, 4)
    pids = torch.tensor([0, 0, 1, 1])

    cold_ret = branch(image_features, text_features, pids)
    assert list(cold_ret.keys()) == ["proto_id_loss"]
    assert cold_ret["proto_id_loss"].shape == ()

    cold_disabled = branch(image_features, text_features, pids, use_loss_id=False)
    assert cold_disabled == {}

    image_projected, text_projected = branch.project_for_memory(image_features, text_features)
    branch.initialize_projected(image_projected, text_projected, pids)
    warm_ret = branch(image_features, text_features, pids)

    assert list(warm_ret.keys()) == ["proto_id_loss"]
    assert torch.isfinite(warm_ret["proto_id_loss"])
    assert warm_ret["proto_id_loss"].requires_grad
    assert not hasattr(branch, "score")


def test_memory_has_no_inference_scoring_api():
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=1, dim=2)

    assert not hasattr(memory, "prototype_score_matrix")
    assert not hasattr(memory, "training_score_matrix")
    assert not hasattr(memory, "assign_global")


if __name__ == "__main__":
    test_torch_kmeans_shape_and_normalization()
    test_identity_assignment_stays_inside_identity_slots()
    test_pbt_empty_slots_fall_back_to_same_side_prototypes()
    test_symmetric_identity_proxy_loss_is_finite()
    test_branch_forward_returns_only_id_regularizer()
    test_memory_has_no_inference_scoring_api()
