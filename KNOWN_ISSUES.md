# EXP-10 engineering and research ledger

This document is the server-side handoff for EXP-10.  It separates safeguards
that are implemented in the repository from open research risks.  A successful
launch only proves the first category; it must not be used to claim that the
second category is solved.

## Implemented safeguards

| Item | Status | Verification on the server |
|---|---|---|
| Optimizer-step accounting | Fixed | Every `state.pt` must contain `step_unit: optimizer_update`; `step_4000` means 4,000 optimizer updates, not 4,000 micro-batches. |
| LR schedule accounting | Fixed | Scheduler is not passed through `Accelerator.prepare`; it advances exactly once per optimizer update. |
| Stale inherited validation file | Fixed | `exp10_curated_*` resolves to `train.val_manifest == ""`; preflight rejects any non-empty inherited path. |
| Paired temporal augmentation | Fixed for one pass | With the same seed and manifest index, both arms use the same decode offset and synthetic transform. |
| Treatment smoke test | Fixed | `smoke` creates `exp10_curated_smoke_exp10_curated_sft_s0/step_2` and `exp10_curated_smoke_exp10_curated_mse_s0/step_2`. |
| MTP trivial baseline | Added | Treatment logs `mtp_persistence_mse`, `mtp_persistence_ratio`, and `mtp_gain_vs_persistence`; the ratio should be below 1 before calling the MTP non-trivial. |
| Answer supervision truncation | Fixed and logged | `answer_tokens` must be positive; inspect `answer_truncated_frac` and `question_truncated_frac` in `log.jsonl`. |
| Exact cross-source duplicates | Fixed conservatively | `qa_train.jsonl.report.json` records dropped exact `realpath` collisions. Per-source minimum counts still gate the build. |

## Required launch checks

Run these stages in order; do not bypass a failed gate by editing a minimum or
reusing a stale manifest.

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
git pull --ff-only origin main
bash scripts/direct/run_exp10_curated_4gpu.sh preflight
bash scripts/direct/run_exp10_curated_4gpu.sh audit
bash scripts/direct/run_exp10_curated_4gpu.sh prep
bash scripts/direct/run_exp10_curated_4gpu.sh smoke
```

Before `train`, verify the following:

- `source_audit.json` has all four selected sources and locally resolvable media;
- `qa_train_clean.jsonl` has the required total and all source provenance fields;
- `qa_train.jsonl.report.json` reports the exact-path duplicate count;
- both smoke directories contain `step_2/state.pt`;
- the smoke `state.pt` has `step_unit == "optimizer_update"`.

For a running MTP arm, investigate immediately if either condition persists
after warmup:

- `mtp_persistence_ratio >= 1`: the learned predictor is no better than copying
  the previous target frame;
- `target_std` trends toward zero or `adj_cos` trends toward one: the online
  target representation is becoming temporally flat.

## Open research risks — do not “fix” by assertion

| Risk | Why it remains open | Appropriate next experiment |
|---|---|---|
| Re-encoded contamination | Path/ID filtering cannot identify a renamed or re-encoded copy. | Use source provenance exclusion and, if needed, sampled video perceptual-hash review; report the limit explicitly. |
| Moving target space | The target is `LayerNorm(h).detach()` from the same online ViT, with no EMA teacher. | Compare the current K=1 MTP arm with a frozen/EMA target encoder, retaining the identical CE control and seeds. |
| One-step smoothing shortcut | K=1 emphasizes short-term persistence; it may help speed/direction while hurting action order. | Advance only if paired TempCompass results and the persistence ratio are positive; then test a time-conditioned random horizon, not a blind K=4 average. |
| Duplicate-frame temporal input | Each physical frame is copied to fill Qwen's two-frame patch, preserving 16 slots but weakening its native local two-frame signal. | Compare current input against 32 real adjacent frames packed into 16 temporal slots, with unchanged data and CE control. |
| Caption/benchmark mismatch | Generic caption QA mainly trains description, whereas TempCompass/MVBench require option discrimination and temporal reasoning. | Upweight provenance-traceable natural temporal QA and use balanced, non-benchmark temporal QA formats; do not add a benchmark source to training. |
| One-pass augmentation | Deterministic transforms deliberately repeat if the same row is visited in a later epoch. | For a planned multi-epoch experiment, add a shared epoch counter to the dataset before launch. |
| Custom absolute scores | The current evaluator uses answer likelihood over pooled visual tokens; it is valid for within-run paired deltas but not direct comparison with VLMEvalKit numbers. | Establish a separately validated standard evaluation adapter before making external score claims. |

## Decision rule for longer training

The 4,000-step EXP-10 arm is a mechanism test, not a reason to extend one
arm alone.  Continue to a fresh 8,000-update run only when both seeds show a
consistent positive paired delta on the preregistered benchmark readout and
the MTP persistence diagnostic is healthy.  Start fresh with a new cosine
schedule and output directory; do not resume a completed 4,000-update run to
change `max_steps`.
