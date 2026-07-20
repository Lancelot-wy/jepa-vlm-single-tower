# EXP-11: minimal Orca-inspired overnight pilot

## Decision and scope

EXP-10 found no useful benchmark gain from the existing online-target adjacent-frame
MSE.  EXP-11 does not reproduce the full Orca recipe.  It tests the smallest causal
and optimization changes that can distinguish Orca-style state prediction from the
failed legacy MTP objective:

1. Keep the clean VQA/CE path unchanged.
2. Freeze the visual encoder in every arm, so the comparison controls for freezing.
3. Encode transition frames as independent one-frame visual entries.
4. Append four learned query vectors to one current frame and use a two-layer MLP to
   predict the four frozen target tokens one second later.
5. Run the untested 15% mask objective as a secondary, separate arm.

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

## Paired arms

| Arm | CE view | Visual target | Auxiliary objective | Weight |
|---|---|---|---|---:|
| `exp11_frozen_sft_s0` | native clean video | none | none | 0 |
| `exp11_mask15_s0` | native clean video | frozen native ViT | configured 15% tube-mask latent reconstruction | 0.1 |
| `exp11_orca_obs_s0` | native clean video | frozen independent-frame ViT | 4 learned queries + 2-layer MLP, predict gap=2 frames | 0.1 |

The control is new rather than reusing EXP-10 because EXP-10 trained the ViT.  All
three EXP-11 arms share seed 0, the exact same clean manifest, sampling regime,
effective batch, update count, and frozen ViT.

With 16 whole-frame slots, the existing deterministic rounding masks two frames for
`mask_ratio=0.15`, so the realized fraction is 12.5%.  This is the closest available
whole-frame ratio to the requested approximately 15%; `mask_fraction` is logged rather
than presenting it as exactly 15%.

## 24-Worker allocation

The company job requests 24 Workers, each with 4 L40S GPUs:

- 3 arms in parallel;
- 8 Workers / 32 GPUs per arm;
- per-device batch 4, gradient accumulation 1;
- effective batch `4 * 32 * 1 = 128` per optimizer update;
- 1,000 optimizer updates, warmup 100;
- atomic checkpoints at steps 250, 500, 750, and 1,000;
- full MVBench and TempCompass evaluation after all arms finish;
- `restartPolicy: Never`; the entrypoint exits after evaluation so the allocation is released.

This is a mechanism pilot, not a final positive-claim run.  If it passes, promote the
winning objective to 4,000 steps and a second seed.  If it fails, do not scale the same
objective merely by adding steps or generic caption data.

## Launch

From the shared server checkout:

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
git pull --ff-only origin main
bash scripts/cluster/submit_exp11_orca24.sh --dry-run
bash scripts/cluster/submit_exp11_orca24.sh
```

Record the printed run ID.  Inspect without mutating the job:

```bash
bash scripts/cluster/inspect_exp11_run.sh <run-id>
```

If the allocation is reclaimed, resubmit against the same shared run root.  The script
selects the newest complete optimizer-update checkpoint and refuses partial writes:

```bash
bash scripts/cluster/submit_exp11_orca24.sh --resume --run-id <run-id>
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
5. Only then inspect MVBench/TempCompass deltas.  A pilot gain around or above 1 point
   in temporal categories is worth a full paired run; sub-0.3-point aggregate changes
   should be treated as noise until a second seed and paired item-level test exist.

The 15% mask arm is secondary.  Its purpose is to close the unfinished ablation, not
to displace the predictive-query direction unless it shows a clear, reproducible gain.
