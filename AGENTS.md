# AGENTS.md

## Purpose

This is a fresh research record for this repository. Do not treat it as a
continuation of any older AGENTS.md content, older prototype proposal protocol,
or any single preselected method.

The only active goal is:

```text
push ITSELF + prototype and CLIP/global + prototype retrieval scores higher
```

Score means validation/test retrieval metrics, especially:

- `R1`
- `mAP`
- `mINP`
- `rSum`

Prototype diagnostics matter only when they help explain or predict better
retrieval. Better prototype diagnostics without better retrieval is not enough.

## Required Repo Survey

Before proposing or implementing any new direction, inspect the repository as a
whole:

- current branch code
- available git branches
- relevant `docs/` research notes
- `logs/` experiment outputs
- `run.sh`
- `utils/options.py`
- model, processor, diagnostics, metrics, and prototype modules

Do not rely on the current checkout alone. This is a multi-branch research
workspace, and completed methods may live on separate branches.

Use the branch history as implementation evidence and the docs/logs as research
evidence. If the current branch and docs disagree, inspect the method-owning
branch before making a conclusion.

## Branch Awareness

Known prototype-related branches include:

- `prototype-seed`: strong fixed prototype baselines, seed control, and logs.
- `prototype-hard-k-v2`: adaptive hard-negative scheduling work.
- `prototype-pressure`: host-aligned prototype pressure work.
- `prototype-translate`: translated-bank text update work.
- `prototype-slot`: adaptive slot/budget work.
- `prototype-refresh`: prototype refresh work.
- `prototype-infer`: prototype scoring and inference experiments.

Do not assume a branch is the best next step just because it exists. Rank
branches and methods by score potential, implementation risk, and evidence from
logs.

## Current Score Anchors

Use the latest logs as empirical anchors before suggesting changes.

Observed RSTP-ReID anchors from existing logs:

- `ITSELF + prototype`: best observed `R1 ~= 68.20` with local/GRAB feature
  path, `batch_size=256`, `seed=1`, `prototype_hard_k=8`, and
  `prototype_id_weight=8.0`.
- `CLIP/global + prototype`: best observed `R1 ~= 62.70` with
  `only_global=True`, `batch_size=256`, `seed=1`, `prototype_hard_k=16`, and
  `prototype_id_weight=8.0`.

When a new experiment is proposed, compare it against the exact matching
baseline settings whenever possible.

## What To Optimize

Prioritize changes that can plausibly improve retrieval for both:

- `ITSELF + prototype`
- `CLIP/global + prototype`

Prefer controlled experiments that change one mechanism at a time. Keep the
baseline, dataset, seed, schedule, feature source, and evaluation row fixed
unless the purpose of the experiment is specifically to test one of those
variables.

Useful diagnostic signals include:

- `dead_slot_rate`
- `active_dead_slot_rate`
- `effective_slots_per_id`
- `slot_redundancy`
- `assignment_flip_rate`
- `proto_margin_img_mean`
- `proto_margin_txt_mean`
- `proto_margin_gap`
- `negative_proto_margin_rate`
- `hard_negative_overlap`
- `proto_to_host_margin_corr`
- `loss_grad_norm/proto_id_loss`

Diagnostics should be interpreted through retrieval results, not in isolation.

## What Not To Do By Default

Do not automatically continue an old method plan from a previous AGENTS.md.

Do not force the next step to be any single named method unless the user asks
for that method explicitly or the repo/log survey shows it is the strongest
score-first candidate.

Do not make inference-time prototype score fusion the main direction unless the
user explicitly asks. Existing logs show nonzero prototype fusion can hurt
ranking, so treat it cautiously.

Do not make host architecture, GRAB/global fusion, or unrelated ITSELF changes
the center of the work unless the user explicitly asks. The default target is
the prototype branch and its effect on retrieval.

Do not tune many knobs at once. Avoid broad grids that make it impossible to
attribute gains.

## Expected Workflow

For any score-improvement task:

1. Read this file.
2. Inspect the current branch and relevant branches.
3. Read only the docs needed to understand the candidate mechanisms.
4. Extract the strongest baseline scores from logs.
5. Identify which method or code path is most likely to improve retrieval now.
6. State the evidence, expected files, and acceptance criteria.
7. Implement only after the intended direction is clear.
8. Verify with focused tests or static checks when possible.

When reporting back, always distinguish:

- what is true in the current checkout;
- what exists on another branch;
- what the logs actually show;
- what is an inference or recommendation.

## Acceptance Criteria

A change is worth keeping only if it improves or preserves retrieval while also
keeping prototype behavior sane.

Promising:

- `R1`, `mAP`, `mINP`, or `rSum` improves against the matching baseline.
- Retrieval is preserved while prototype health becomes clearly better.
- Gains transfer from one host setting to the other, or the difference is
  explainable from logs.
- Prototype gradients stay controlled relative to host losses.

Not enough:

- prototype margins improve but retrieval is flat or worse;
- slot diagnostics improve but retrieval is flat or worse;
- gains require dataset-specific magic values;
- improvements appear only in a prototype-fusion evaluation row that is not the
  main deployment score.

## Final Guidance

This file is a new high-level operating record. Its job is to make Codex read
the whole repository, understand all branches and experiments, and recommend
the most score-worthy next action for `ITSELF + prototype` and
`CLIP/global + prototype`.

Do not let older AGENTS.md content, old proposal-first protocols, or a single
previous method override this score-first repo-wide survey.

## Prototype Backbone Method Rebase Plan

### Summary

- Base every new method branch on `prototype-backbone`, not on `prototype-seed`.
- Keep each method isolated: one branch, one mechanism, one commit.
- Do not edit or depend on `tests/test_prototype.py`.
- Preserve `prototype-backbone` as the canonical reproducible prototype base.

### Shared Backbone Rules

- Preserve the `prototype-backbone` training protocol exactly:
  - no sampler/DataLoader generator/worker-seed changes;
  - no cuDNN/TF32/strict deterministic backend changes;
  - no whole-stack seed protocol from `prototype-seed`;
  - no rank loss;
  - no inference-time prototype score fusion;
  - no `+proto` evaluator rows.
- Preserve prototype reproducibility utilities:
  - keep prototype RNG isolation;
  - keep full train dataset identity init;
  - keep deterministic KMeans init default;
  - keep deterministic group mean default;
  - keep W&B helper utilities already ported.
- Every branch must keep default CLI behavior equivalent to `prototype-backbone`;
  a method only activates when its new option is enabled.

### Branch Plan

Create these branches from a clean `prototype-backbone`, never from each other:

- `prototype-hard-k-backbone`
- `prototype-hard-k-v2-backbone`
- `prototype-pressure-backbone`
- `prototype-projector-backbone`
- `prototype-refresh-backbone`
- `prototype-slot-backbone`
- `prototype-translate-backbone`

For each branch:

1. Switch back to `prototype-backbone`.
2. Create the method branch.
3. Port only method-specific code from the old method branch.
4. Run static checks.
5. Commit with `Port <method> onto prototype-backbone`.

### Method-Specific Scope

- `prototype-hard-k-backbone`: port only legacy
  `--prototype_hard_k_mode fixed|adaptive`; limit changes to prototype ID
  hard-negative selection and diagnostics.
- `prototype-hard-k-v2-backbone`: port DMB-calibrated scheduler
  `--prototype_hard_k_mode fixed|calibrated`; use
  `docs/Adaptive_Hard_Negative_Scheduler.md` as source of truth.
- `prototype-pressure-backbone`: port
  `--prototype_pressure_mode fixed|host_aligned`; change only per-sample
  weighting/gating of prototype ID loss using detached host margins.
- `prototype-projector-backbone`: port projector modes `default`, `identity`,
  `residual_identity`, `random_orthogonal`, `pca_init`, `shared`, and
  `shared_pca_init`; preserve prototype seed/RNG isolation.
- `prototype-refresh-backbone`: port scheduled anchored refresh with
  `--prototype_refresh_start_epoch`, `--prototype_refresh_step`, and
  `--prototype_refresh_alpha`; reuse the current full-dataset init/collection
  path and RNG isolation.
- `prototype-slot-backbone`: port only adaptive effective prototype budgeting
  via `--prototype_slot_mode fixed|adaptive_masked` and
  `--prototype_max_per_id`; use padded physical slots plus active masks.
- `prototype-translate-backbone`: port only translated-bank text update via
  `--prototype_text_update_mode uniform|confidence_weighted`; weighted update
  applies only to `image_to_text`.

### Verification

- Run `python -m py_compile` on touched Python modules in each branch.
- Smoke-check CLI parser defaults and enabled method options.
- Grep for forbidden regressions: `seed_worker`, custom DataLoader generator,
  strict deterministic flags, `use_loss_rank`, `proto_rank_loss`,
  `prototype_score_weights`, evaluator `branch.score`, and `+proto` rows.
- Confirm default mode stays equivalent to `prototype-backbone`.
- Confirm enabled mode returns finite losses and expected diagnostics.
- Confirm prototype init still covers the full train dataset identity set.
- Compare experiments against matching `prototype-backbone` baselines using
  `R1`, `mAP`, `mINP`, and `rSum`; diagnostics alone are not enough.

### Assumptions

- Old method branches are implementation evidence, not code to cherry-pick
  wholesale.
- `docs/*` are local research notes and may be ignored by Git, but they remain
  source material for implementation.
- Branches remain separate until single-method results justify a combined
  experiment.
