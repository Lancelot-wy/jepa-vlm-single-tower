# EXP-11: minimal Orca-inspired overnight pilot

## Decision and scope

EXP-10 found no useful benchmark gain from the existing online-target adjacent-frame
MSE.  EXP-11 does not reproduce the full Orca recipe.  It tests the smallest causal
and optimization changes that can distinguish Orca-style state prediction from the
failed legacy MTP objective:

1. Keep the clean VQA/CE path unchanged.
2. Freeze the visual encoder in every arm, so the comparison controls for freezing.
3. Encode transition frames as independent one-frame visual entries.
4. Compare the same two-layer predictor with and without four learned query vectors.
5. Predict the four frozen target tokens one second later.
6. Run the untested 15% mask objective as a secondary, separate arm.

Event-conditioned prediction is deliberately deferred.  The current caption mixture
does not yet provide a uniformly audited previous/current/next event schema, so adding
it now would conflate a model test with noisy pseudo-event construction.

## Sampling and leakage audit

All arms use 16 sampled frames at 2 fps, normally spanning 7.5 seconds.  Clips shorter
than that span fall back to uniform sampling across the available segment.

Qwen3-VL's ViT constructs `cu_seqlens` per temporal unit.  Together with this project's
`duplicate_frames=true`, each sampled frame is already an independent visual-attention
segment even when all 16 frames are submitted in one batched call.  The Orca arm still
executes an explicit one-frame-entry path and logs `orca_frame_encoding_mse` against
the native path.  A value near zero confirms the no-cross-frame-attention invariant;
a material value is a stop-and-debug signal.

## Reused control and paired arms

| Arm | Status | CE view | Visual target | Auxiliary objective | Weight |
|---|---|---|---|---|---:|
| `exp11_frozen_sft_s0` | reuse today's completed run | native clean video | none | none | 0 |
| `exp11_mask15_s0` | train | native clean video | frozen native ViT | configured 15% tube-mask latent reconstruction | 0.1 |
| `exp11_orca_noquery_s0` | train | native clean video | frozen independent-frame ViT | current-token hidden states + matched 2-layer MLP, gap=2 | 0.1 |
| `exp11_orca_obs_s0` | train | native clean video | frozen independent-frame ViT | 4 learned queries + matched 2-layer MLP, gap=2 | 0.1 |

The earlier 4,000-step EXP-10 SFT cannot be used as this control because it trained
the ViT.  Today's 1,000-step Frozen SFT can be reused only if the launch gate confirms
that its scientific config matches `configs/exp11_frozen_sft_s0.yaml`, its checkpoint
is optimizer update 1,000, its embedded config matches `config.json`, it points to the
same clean manifest, that manifest is not newer than the checkpoint, and its declared
world size gives effective batch 128.  The current manifest SHA-256 is printed in the
gate log.  If any check fails, the job exits before training; do not call a mismatched
run the control.

All three new EXP-11 arms share seed 0, the exact same clean manifest, sampling
regime, effective batch, update count, and frozen ViT.  The two Orca arms differ in
only `orca_use_queries`, making `no-query` versus `query` the primary mechanism
comparison.

With 16 whole-frame slots, the existing deterministic rounding masks two frames for
`mask_ratio=0.15`, so the realized fraction is 12.5%.  This is the closest available
whole-frame ratio to the requested approximately 15%; `mask_fraction` is logged rather
than presenting it as exactly 15%.

## 24-Worker allocation

The company job requests 24 Workers, each with 4 L40S GPUs.  The completed Frozen SFT
uses no training allocation:

- 3 new arms in parallel (`mask15`, `Orca-no-query`, `Orca-query`);
- 8 Workers / 32 GPUs per arm;
- per-device batch 4, gradient accumulation 1;
- effective batch `4 * 32 * 1 = 128` per optimizer update;
- 1,000 optimizer updates, warmup 100;
- atomic checkpoints at steps 250, 500, 750, and 1,000;
- full MVBench and TempCompass evaluation for the reused control and all three new
  arms after training finishes;
- `restartPolicy: Never`; the entrypoint exits after evaluation so the allocation is released.

This is a mechanism pilot, not a final positive-claim run.  If it passes, promote the
winning objective to 4,000 steps and a second seed.  If it fails, do not scale the same
objective merely by adding steps or generic caption data.

## Launch

First locate today's completed Frozen-SFT output.  It must contain both
`config.json` and `step_1000/state.pt`.  From the shared server checkout:

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
git pull --ff-only origin main
CONTROL=/absolute/path/to/todays/exp11_frozen_sft_s0
bash scripts/cluster/submit_exp11_orca24.sh --control-dir "$CONTROL" --control-world-size 32 --dry-run
bash scripts/cluster/submit_exp11_orca24.sh --control-dir "$CONTROL" --control-world-size 32
```

`--control-world-size` is the DDP world used for today's control, not the new job's
96-GPU total.  For example, use `4` if the control used one 4-GPU Worker with gradient
accumulation 8.  The gate verifies that `batch_size * grad_accum * world_size = 128`
and checks retained launcher logs when available.

Record the printed run ID.  Inspect without mutating the job:

```bash
bash scripts/cluster/inspect_exp11_run.sh <run-id>
```

If the allocation is reclaimed, resubmit against the same shared run root.  The script
selects the newest complete optimizer-update checkpoint and refuses partial writes:

```bash
bash scripts/cluster/submit_exp11_orca24.sh --resume --run-id <run-id> \
  --control-dir "$CONTROL" --control-world-size 32
```

## Morning readout

Primary result:

```text
/data/vjuicefs_sz_ocr_wl/public_data/11193960/runs/exp11_orca/<run-id>/results/scorecard.json
```

The scorecard includes benchmark deltas against the frozen-SFT control and final
mechanism metrics.  Interpret the Orca arm in this order:

1. `orca_frame_encoding_mse` should be near numerical zero.  Otherwise the presumed
   single-frame equivalence is false and the run needs inspection.
2. `orca_persistence_ratio` must fall below 1.0 to beat copying the current frozen
   state; below 0.9 is a reasonable promotion threshold.
3. `orca_target_std` should remain stable and `orca_pred_std` must not collapse.
4. CE should not degrade materially against the frozen-SFT control.
5. Compare `exp11_orca_obs_s0` directly with `exp11_orca_noquery_s0`.  This is the
   clean test of whether learned queries add value beyond frozen future-state
   prediction plus the MLP.
6. Only then inspect MVBench/TempCompass deltas.  A pilot gain around or above 1 point
   in temporal categories is worth a full paired run; sub-0.3-point aggregate changes
   should be treated as noise until a second seed and paired item-level test exist.

The 15% mask arm is secondary.  Its purpose is to close the unfinished ablation, not
to displace the predictive-query direction unless it shows a clear, reproducible gain.
