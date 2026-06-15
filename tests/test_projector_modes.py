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


def _args(mode, prototype_dim, residual_scale=0.1):
    return SimpleNamespace(
        only_global=True,
        prototype_feature="auto",
        prototype_projector=mode,
        prototype_residual_scale=residual_scale,
        prototype_dim=prototype_dim,
        prototype_per_id=1,
        prototype_momentum=0.2,
        prototype_kmeans_iters=2,
        prototype_tau=0.05,
        prototype_hard_k=1,
        seed=7,
    )


def test_identity_projector_has_no_params_and_preserves_shape():
    branch = PrototypeBranch(_args("identity", prototype_dim=3), num_classes=2, feature_dim=3)
    image_features = torch.randn(4, 3)
    text_features = torch.randn(4, 3)

    image_projected, text_projected = branch.project_for_memory(image_features, text_features)

    assert list(branch.image_projector.parameters()) == []
    assert list(branch.text_projector.parameters()) == []
    assert image_projected.shape == image_features.shape
    assert text_projected.shape == text_features.shape
    assert torch.allclose(image_projected, F.normalize(image_features.float(), p=2, dim=1), atol=1e-6)
    assert torch.allclose(text_projected, F.normalize(text_features.float(), p=2, dim=1), atol=1e-6)


def test_identity_projector_rejects_mismatched_dims():
    try:
        PrototypeBranch(_args("identity", prototype_dim=3), num_classes=2, feature_dim=4)
    except ValueError as exc:
        assert "requires feature_dim == prototype_dim" in str(exc)
        return
    raise AssertionError("identity projector should reject mismatched dimensions")


def test_residual_identity_starts_as_normalized_input():
    branch = PrototypeBranch(_args("residual_identity", prototype_dim=3), num_classes=2, feature_dim=3)
    image_features = torch.randn(4, 3)
    text_features = torch.randn(4, 3)

    image_projected, text_projected = branch.project_for_memory(image_features, text_features)

    assert torch.allclose(image_projected, F.normalize(image_features.float(), p=2, dim=1), atol=1e-6)
    assert torch.allclose(text_projected, F.normalize(text_features.float(), p=2, dim=1), atol=1e-6)


def test_random_orthogonal_projector_shape_and_rows():
    branch = PrototypeBranch(_args("random_orthogonal", prototype_dim=2), num_classes=2, feature_dim=4)
    image_features = torch.randn(5, 4)
    text_features = torch.randn(5, 4)

    image_projected, text_projected = branch.project_for_memory(image_features, text_features)
    image_gram = branch.image_projector.weight.detach() @ branch.image_projector.weight.detach().t()

    assert image_projected.shape == (5, 2)
    assert text_projected.shape == (5, 2)
    assert torch.allclose(image_gram, torch.eye(2), atol=1e-5)


def test_shared_projector_uses_one_module_once():
    branch = PrototypeBranch(_args("shared", prototype_dim=2), num_classes=2, feature_dim=4)
    projector_param_ids = [
        id(param)
        for name, param in branch.named_parameters()
        if "projector" in name
    ]

    assert branch.image_projector is branch.text_projector
    assert len(projector_param_ids) == len(set(projector_param_ids))


def test_pca_init_copies_weights_in_place_and_initializes_memory():
    torch.manual_seed(11)
    branch = PrototypeBranch(_args("pca_init", prototype_dim=2), num_classes=2, feature_dim=4)
    image_features = torch.randn(8, 4)
    text_features = torch.randn(8, 4)
    pids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    image_weight = branch.image_projector.weight
    image_weight_before = image_weight.detach().clone()

    branch.initialize_projector_from_features(image_features, text_features)
    image_projected, text_projected = branch.project_for_memory(image_features, text_features)
    branch.initialize_projected(image_projected, text_projected, pids)

    assert branch.image_projector.weight is image_weight
    assert not torch.allclose(branch.image_projector.weight.detach(), image_weight_before)
    assert branch.is_ready()
    assert branch.memory.image_prototypes.shape == (2, 2)
    assert branch.memory.text_prototypes.shape == (2, 2)


def test_shared_pca_init_uses_concatenated_shared_projector():
    torch.manual_seed(13)
    branch = PrototypeBranch(_args("shared_pca_init", prototype_dim=2), num_classes=2, feature_dim=4)
    image_features = torch.randn(8, 4)
    text_features = torch.randn(8, 4)

    branch.initialize_projector_from_features(image_features, text_features)
    image_projected, text_projected = branch.project_for_memory(image_features, text_features)

    assert branch.image_projector is branch.text_projector
    assert branch.needs_pca_init() is False
    assert image_projected.shape == (8, 2)
    assert text_projected.shape == (8, 2)


if __name__ == "__main__":
    test_identity_projector_has_no_params_and_preserves_shape()
    test_identity_projector_rejects_mismatched_dims()
    test_residual_identity_starts_as_normalized_input()
    test_random_orthogonal_projector_shape_and_rows()
    test_shared_projector_uses_one_module_once()
    test_pca_init_copies_weights_in_place_and_initializes_memory()
    test_shared_pca_init_uses_concatenated_shared_projector()
