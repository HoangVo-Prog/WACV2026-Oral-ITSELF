import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, REPO_ROOT)

from processor.processor import _should_run_epoch_eval, _should_run_initial_eval


def test_initial_eval_runs_immediately_when_threshold_is_zero():
    assert _should_run_initial_eval(start_epoch=1, eval_after_epoch=0) is True


def test_initial_eval_is_skipped_before_threshold_for_fresh_run():
    assert _should_run_initial_eval(start_epoch=1, eval_after_epoch=5) is False


def test_initial_eval_runs_when_resuming_after_threshold():
    assert _should_run_initial_eval(start_epoch=6, eval_after_epoch=5) is True


def test_epoch_eval_starts_at_threshold_and_runs_afterward():
    assert _should_run_epoch_eval(epoch=4, eval_period=1, eval_after_epoch=5) is False
    assert _should_run_epoch_eval(epoch=5, eval_period=1, eval_after_epoch=5) is True
    assert _should_run_epoch_eval(epoch=6, eval_period=1, eval_after_epoch=5) is True


def test_epoch_eval_still_respects_eval_period():
    assert _should_run_epoch_eval(epoch=5, eval_period=2, eval_after_epoch=5) is False
    assert _should_run_epoch_eval(epoch=6, eval_period=2, eval_after_epoch=5) is True
