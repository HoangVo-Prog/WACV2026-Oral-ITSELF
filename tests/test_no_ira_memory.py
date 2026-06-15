import os
import sys
import types
import unittest
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
from utils.options import get_args


def _normalize(values):
    return F.normalize(torch.tensor(values, dtype=torch.float32), p=2, dim=1)


def _branch_args(**overrides):
    values = dict(
        only_global=True,
        prototype_feature="auto",
        prototype_dim=2,
        prototype_per_id=1,
        prototype_momentum=0.2,
        prototype_tau=0.05,
        no_pbt=False,
        no_ira=False,
        no_ira_mode="hard",
        no_iopm=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class NoIraMemoryTest(unittest.TestCase):
    def test_default_assignment_stays_inside_identity_slots(self):
        memory = PrototypeMemory(num_classes=2, prototypes_per_id=1, dim=2)
        bank = _normalize([[1.0, 0.0], [0.0, 1.0]])
        features = _normalize([[0.0, 1.0]])
        pids = torch.tensor([0])

        identity_assignment = memory.assign_for_update(features, pids, bank)
        global_assignment = memory.assign_global(features, bank)

        self.assertEqual(identity_assignment.item(), 0)
        self.assertEqual(global_assignment.item(), 1)

    def test_global_hard_assignment_updates_global_nearest_prototype(self):
        memory = PrototypeMemory(
            num_classes=2,
            prototypes_per_id=1,
            dim=2,
            momentum=1.0,
            assignment_mode="global_hard",
        )
        image_bank = _normalize([[1.0, 0.0], [0.1, 1.0]])
        text_bank = image_bank.clone()
        memory.image_prototypes.copy_(image_bank)
        memory.text_prototypes.copy_(text_bank)
        memory.text_to_image.copy_(image_bank)
        memory.image_to_text.copy_(text_bank)
        memory.initialized.fill_(True)

        features = _normalize([[0.0, 1.0]])
        pids = torch.tensor([0])
        memory.ema_update(features, features, pids)

        self.assertTrue(torch.allclose(memory.image_prototypes[0], image_bank[0], atol=1e-6))
        self.assertTrue(torch.allclose(memory.image_prototypes[1], features[0], atol=1e-6))

    def test_global_soft_assignment_weights_sum_and_update_all_prototypes(self):
        memory = PrototypeMemory(
            num_classes=2,
            prototypes_per_id=1,
            dim=2,
            momentum=1.0,
            assignment_mode="global_soft",
            assignment_tau=1.0,
        )
        bank = _normalize([[1.0, 0.0], [0.0, 1.0]])
        memory.image_prototypes.copy_(bank)
        memory.text_prototypes.copy_(bank)
        memory.text_to_image.copy_(bank)
        memory.image_to_text.copy_(bank)
        memory.initialized.fill_(True)

        features = _normalize([[1.0, 0.0], [0.0, 1.0]])
        weights = memory.assign_soft_global(features, memory.image_prototypes)
        before = memory.image_prototypes.clone()
        memory.ema_update(features, features, torch.tensor([0, 1]))

        self.assertTrue(torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6))
        self.assertTrue(torch.isfinite(memory.image_prototypes).all())
        self.assertTrue(torch.allclose(memory.image_prototypes.norm(dim=1), torch.ones(2), atol=1e-6))
        self.assertFalse(torch.allclose(memory.image_prototypes[0], before[0], atol=1e-6))
        self.assertFalse(torch.allclose(memory.image_prototypes[1], before[1], atol=1e-6))

    def test_no_iopm_initialization_uses_total_global_slots(self):
        features = _normalize([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        pids = torch.tensor([0, 0, 1])
        memory = PrototypeMemory(
            num_classes=2,
            prototypes_per_id=2,
            dim=2,
            identity_owned_init=False,
        )

        memory.initialize(features, features, pids, num_iters=2, seed=5)

        self.assertTrue(memory.is_ready())
        self.assertEqual(tuple(memory.image_prototypes.shape), (4, 2))
        self.assertEqual(tuple(memory.text_prototypes.shape), (4, 2))
        self.assertTrue(torch.allclose(memory.image_prototypes.norm(dim=1), torch.ones(4), atol=1e-6))

    def test_cli_flags_parse_independently(self):
        args = get_args(["--no_pbt", "--no_ira", "--no_ira_mode", "soft", "--no_iopm"])

        self.assertTrue(args.no_pbt)
        self.assertTrue(args.no_ira)
        self.assertEqual(args.no_ira_mode, "soft")
        self.assertTrue(args.no_iopm)

    def test_branch_maps_no_ira_flags_to_memory_modes(self):
        default_branch = PrototypeBranch(_branch_args(), num_classes=2, feature_dim=2)
        hard_branch = PrototypeBranch(_branch_args(no_ira=True, no_ira_mode="hard"), num_classes=2, feature_dim=2)
        soft_branch = PrototypeBranch(
            _branch_args(no_pbt=True, no_ira=True, no_ira_mode="soft", no_iopm=True),
            num_classes=2,
            feature_dim=2,
        )

        self.assertEqual(default_branch.assignment_mode, "identity_hard")
        self.assertTrue(default_branch.memory.identity_owned_init)
        self.assertEqual(hard_branch.assignment_mode, "global_hard")
        self.assertTrue(hard_branch.memory.identity_owned_init)
        self.assertEqual(soft_branch.assignment_mode, "global_soft")
        self.assertFalse(soft_branch.memory.identity_owned_init)
        self.assertTrue(soft_branch.args.no_pbt)


if __name__ == "__main__":
    unittest.main()
