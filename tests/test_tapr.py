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

from model.prototype.memory import PrototypeMemory
from model.prototype.build import PrototypeBranch
from utils.prototype_refresh import prototype_refresh_due, validate_prototype_refresh_args


def _args(start=-1, step=-1, alpha=0.35):
    return SimpleNamespace(
        prototype_refresh_start_epoch=start,
        prototype_refresh_step=step,
        prototype_refresh_alpha=alpha,
    )


def test_prototype_refresh_schedule_disabled_by_default():
    assert not prototype_refresh_due(_args(), 1)
    assert not prototype_refresh_due(_args(), 100)


def test_prototype_refresh_schedule_one_time():
    args = _args(start=3, step=-1)
    assert not prototype_refresh_due(args, 2)
    assert prototype_refresh_due(args, 3)
    assert not prototype_refresh_due(args, 4)


def test_prototype_refresh_schedule_periodic():
    args = _args(start=3, step=10)
    assert prototype_refresh_due(args, 3)
    assert not prototype_refresh_due(args, 12)
    assert prototype_refresh_due(args, 13)
    assert prototype_refresh_due(args, 23)


def test_prototype_refresh_schedule_rejects_zero_step():
    try:
        validate_prototype_refresh_args(_args(start=3, step=0))
    except ValueError:
        return
    raise AssertionError("zero prototype refresh step should raise ValueError")


def test_prototype_refresh_keeps_bank_shapes_and_normalization():
    torch.manual_seed(3)
    image_features = F.normalize(torch.randn(12, 4), p=2, dim=1)
    text_features = F.normalize(torch.randn(12, 4), p=2, dim=1)
    pids = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)
    memory = PrototypeMemory(num_classes=3, prototypes_per_id=2, dim=4)
    memory.initialize(image_features, text_features, pids, num_iters=2, seed=11)

    shapes = {
        name: getattr(memory, name).shape
        for name in ("image_prototypes", "text_prototypes", "text_to_image", "image_to_text")
    }
    metrics = memory.refresh(image_features, text_features, pids, num_iters=2, seed=17, alpha=0.35, refresh_epoch=3)

    assert metrics["prototype_refresh_done"] == 1.0
    for name, shape in shapes.items():
        bank = getattr(memory, name)
        assert bank.shape == shape
        assert torch.allclose(bank.norm(dim=1), torch.ones(bank.shape[0]), atol=1e-5)


def test_prototype_refresh_alpha_zero_leaves_banks_unchanged():
    torch.manual_seed(5)
    image_features = F.normalize(torch.randn(8, 3), p=2, dim=1)
    text_features = F.normalize(torch.randn(8, 3), p=2, dim=1)
    pids = torch.tensor([0] * 4 + [1] * 4)
    memory = PrototypeMemory(num_classes=2, prototypes_per_id=2, dim=3)
    memory.initialize(image_features, text_features, pids, num_iters=2, seed=13)
    before = {
        name: getattr(memory, name).detach().clone()
        for name in ("image_prototypes", "text_prototypes", "text_to_image", "image_to_text")
    }

    memory.refresh(image_features, text_features, pids, num_iters=2, seed=19, alpha=0.0, refresh_epoch=3)

    for name, bank_before in before.items():
        assert torch.allclose(getattr(memory, name), bank_before, atol=1e-6)


def test_prototype_refresh_slot_alignment_handles_swapped_candidates():
    memory = PrototypeMemory(num_classes=1, prototypes_per_id=2, dim=2)
    old_bank = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=1)
    swapped_bank = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), p=2, dim=1)

    aligned = memory._align_identity_slots(old_bank, swapped_bank)

    assert torch.allclose(aligned, old_bank, atol=1e-6)


def test_prototype_branch_refresh_projected_smoke():
    args = SimpleNamespace(
        only_global=True,
        prototype_feature="auto",
        prototype_dim=3,
        prototype_per_id=1,
        prototype_momentum=0.2,
        prototype_kmeans_iters=2,
        prototype_tau=0.05,
        prototype_hard_k=1,
        prototype_refresh_alpha=0.35,
        prototype_refresh_step=-1,
        seed=7,
    )
    branch = PrototypeBranch(args, num_classes=2, feature_dim=4)
    image_features = torch.randn(6, 4)
    text_features = torch.randn(6, 4)
    pids = torch.tensor([0, 0, 0, 1, 1, 1])

    image_projected, text_projected = branch.project_for_memory(image_features, text_features)
    branch.initialize_projected(image_projected, text_projected, pids)
    metrics = branch.refresh_projected(image_projected, text_projected, pids, epoch=3)

    assert metrics["prototype_refresh_done"] == 1.0
    assert torch.isfinite(branch.memory.text_to_image).all()


if __name__ == "__main__":
    test_prototype_refresh_schedule_disabled_by_default()
    test_prototype_refresh_schedule_one_time()
    test_prototype_refresh_schedule_periodic()
    test_prototype_refresh_schedule_rejects_zero_step()
    test_prototype_refresh_keeps_bank_shapes_and_normalization()
    test_prototype_refresh_alpha_zero_leaves_banks_unchanged()
    test_prototype_refresh_slot_alignment_handles_swapped_candidates()
    test_prototype_branch_refresh_projected_smoke()
