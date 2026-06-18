import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from utils.options import get_args


def test_finetune_itself_preset_sets_itself_flags():
    args = get_args(["--finetune_itself"])

    assert args.finetune_itself is True
    assert args.return_all is True
    assert args.topk_type == "custom"
    assert args.modify_k is True
    assert args.only_global is False


def test_finetune_itself_rejects_only_global_conflict():
    with pytest.raises(SystemExit):
        get_args(["--finetune_itself", "--only_global"])


def test_eval_after_epoch_defaults_to_zero():
    args = get_args([])

    assert args.eval_after_epoch == 0


def test_eval_after_epoch_parses_custom_value():
    args = get_args(["--eval_after_epoch", "5"])

    assert args.eval_after_epoch == 5
