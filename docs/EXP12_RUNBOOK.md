# EXP-12 server runbook

This is the authoritative handoff for `EXP-12 — Orca Single-Tower Visual Token Sweep`.
Run commands on the shared development machine unless a step explicitly says it runs
inside the scheduled Worker. GPU Pods must use a fixed commit and must not `git pull`.

## 1. Scientific contract

First batch:

| Arm | K | State mode | Event |
|---|---:|---|---|
| A0 `a0_ce_k4` | 4 | none | off |
| A1 `a1_query_k4` | 4 | query | off |
| A2 `a2_ce_k16` | 16 | none | off |
| A3 `a3_query_k16` | 16 | query | off |
| A4 `a4_ce_k64` | 64 | none | off |
| A5 `a5_query_k64` | 64 | query | off |

All arms use the same EXP-10 clean manifest, seed/data order, answer-only CE recipe,
32 raw frames at 4 fps, 16 real temporal units, full LLM update scope, 800 optimizer
updates and effective batch 32. Only K and Observation Query enablement may differ.

The company YAML requests 24 Workers × 4 L40S = 96 GPUs. It creates six independent
4-Worker/16-GPU DDP worlds and sets gradient accumulation to 2. It never creates one
96-GPU world. If the scheduler quota actually means 24 total GPUs rather than 24
Workers, do not use `job_exp12.yaml` unchanged.

## 2. Prepare the fixed checkout

Fresh shared checkout:

```bash
BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
cd "$BASE"
git clone --branch exp12-orca-token-sweep \
  https://github.com/Lancelot-wy/jepa-vlm-single-tower.git jepa-vlm-single-tower
cd jepa-vlm-single-tower
git status --short
git rev-parse HEAD
```

Existing clean shared checkout:

```bash
BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
cd "$BASE/jepa-vlm-single-tower"
git status --short
git fetch origin exp12-orca-token-sweep
git switch exp12-orca-token-sweep
git pull --ff-only origin exp12-orca-token-sweep
git status --short
git rev-parse HEAD
```

Stop if tracked files are modified. Do not reset or delete unknown files. The commit
printed here is the hash injected into the job YAML and checked by every Worker.

## 3. Cheap checks before requesting GPUs

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
source /data/vjuicefs_sz_ocr_wl/public_data/11193960/envs/jepa311/bin/activate
python -m compileall -q jepa_vlm scripts/exp12 tests
python -m pytest -q tests
bash -n scripts/exp12/*.sh scripts/cluster/job_exp12_entry.sh
git diff --check
```

Expected repository test count is recorded in `docs/EXP12_IMPLEMENTATION_REPORT.md`.
These CPU/tiny-model checks do not authorize formal training.

Inspect the exact submission without allocating resources:

```bash
PROJECT_ROOT=$PWD bash scripts/exp12/07_submit_a0_a5.sh --dry-run
```

The rendered command must contain a full `EXP12_GIT_COMMIT`, a new run ID, 24 Workers,
4 GPUs/Worker, four Workers/arm and `EXP12_GRAD_ACCUM=2`.

## 4. Submit the gated six-arm job

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
bash scripts/exp12/07_submit_a0_a5.sh | tee /tmp/exp12-submit.log
```

The job automatically runs, in order:

1. frozen-manifest build/validation and SHA256;
2. model/data/GPU/runtime preflight;
3. full unit suite;
4. K=4,16,64 smoke, each with CE+Query, two optimizer updates, 4-GPU DDP,
   checkpoint save, resume, TempCompass/MVBench mini-eval and mechanism assertions;
5. A0–A5 training only if every smoke passed;
6. checkpoint-400 fixed TempCompass subset evaluation;
7. checkpoint-800 full MVBench and TempCompass evaluation, one arm leader per machine;
8. paired result collection, K selection, clean process exit and automatic allocation release.

Any gate failure stops formal training. Do not bypass it by editing thresholds, K, batch,
cutoff, data or evaluator settings.

## 5. Monitor

Use the run ID printed by the submit script:

```bash
RUN_ID=<printed-run-id>
BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
cd "$BASE/jepa-vlm-single-tower"
bash scripts/exp12/08_status_a0_a5.sh "$RUN_ID"
find "$BASE/runs/exp12/$RUN_ID/logs" -maxdepth 2 -type f -print
tail -f "$BASE/runs/exp12/$RUN_ID/logs"/*/rank0.log
```

Per-arm training logs are under:

```text
$BASE/runs/exp12/$RUN_ID/results/exp12_orca_token_sweep/<arm>/trainer_log.jsonl
```

Check immediately:

- `model/visual_tokens_per_unit` is the arm's K;
- `model/deepstack_token_count == K`;
- `video/raw_frame_count=32`, `temporal_unit_count=16`, and duplicate ratio near zero;
- eligible, short, temporal-augmentation, retry and nearest-substitution fractions are present;
- Query arms emit centered margin, shuffle, retrieval, effective-rank and persistence metrics;
- `loss`, CE/state metrics and learning rates are finite;
- `max_memory_gb` and `samples_per_sec` are plausible for the actual world;
- checkpoint 400 contains `state.pt` and matching `checkpoint_meta.json`.

Do not call the mechanism successful unless validation centered margin is above 0.10
and persistence ratio is below 0.90. A decreasing state loss alone is insufficient.

## 6. Resume or patch on the server

If an allocation is reclaimed, and code/config did not change:

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
bash scripts/exp12/09_resume_failed.sh <existing-run-id>
```

The launcher finds the numerically latest checkpoint with a valid atomic metadata
sidecar. Completed arms exit without retraining; incomplete arms require explicit
resume and continue from saved model, optimizer, scheduler, running center, RNG and
data-batch position.

If real Qwen/L40S/DDP/platform behavior exposes a code bug, it is acceptable to patch
on the shared server development checkout:

```bash
git switch exp12-orca-token-sweep
# edit only the scoped fix
python -m pytest -q tests
git diff --check
git add <scoped-files>
git commit -m "fix: <exp12 server issue>"
git push origin exp12-orca-token-sweep
```

Then submit the new fixed hash. An incomplete run may resume only if the scientific
config and trainable state contract are unchanged. If any arm already completed under
the old commit, start a new run ID and rerun all six arms; the launcher intentionally
rejects a mixed-commit comparison.

For K=64 OOM, save the failure log and `nvidia-smi` first. Check target gradients,
visual activation retention, duplicate forwards and `use_cache=false`. Do not silently
change per-device batch, sequence cutoff, K, steps or effective batch.

## 7. Results and promotion

Final root:

```text
/data/vjuicefs_sz_ocr_wl/public_data/11193960/runs/exp12/<RUN_ID>/results/exp12_orca_token_sweep/
```

Expected top-level artifacts include `comparison.{md,json,csv}`, `selection.json`,
`selection.md`, the frozen checkpoint-400 subset/IDs/SHA, and six arm directories.
Each arm retains config, manifest SHA, commits, environment/parameter/resource audits,
trainer log, TensorBoard events, checkpoints, evaluator config, raw option scores,
per-question JSONL, category JSON and scorecard.

`13_select_best_k.py` never starts Event jobs. If no K passes all gates, the correct
result is `FAIL`; prioritize a registered no-query or beat-copy follow-up rather than
extending one favorable-looking arm.

Event templates are materialized only after a human accepts the A-batch result:

```bash
bash scripts/exp12/14_materialize_event_configs.sh 4   # or 16 / 64
```

This command only writes B0–B5 configs. It does not submit them, and an audited
`event_dataset_path` is still required for B3/B5.
