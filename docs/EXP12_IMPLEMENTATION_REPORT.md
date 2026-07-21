# EXP-12 implementation report

Date: 2026-07-21  
Repository: `Lancelot-wy/jepa-vlm-single-tower`  
Base commit: `2f74c55356da86731b8393b8d8a600e7a6b66bc0`  
Branch: `exp12-orca-token-sweep`  
Final delivery commit: use `git rev-parse HEAD` from the delivered branch; the exact
immutable hash is also reported in the handoff response and injected into every job.

## Outcome and evidence boundary

EXP-12 code, configs, tests, result tooling and 24-Worker launcher are implemented.
Local validation covers tensor logic, two-process Gloo center synchronization, a real
tiny Qwen3-VL forward/backward, and a two-stage tiny training checkpoint/resume over an
encoded video. The local host has no CUDA/NVIDIA GPU and does not contain the real 2B
weight path. Therefore:

- real Qwen3-VL-2B + 4×L40S smoke: **not run yet**;
- A0–A5 formal jobs: **not submitted**;
- six platform/logical job IDs: **not available**;
- frozen production manifest SHA256: **pending server preflight**;
- benchmark result: **none; no channel test or synthetic number is an EXP-12 result**.

The cluster entrypoint enforces all three K smoke gates before training. This unresolved
runtime gate is intentional and may not be replaced by the local tiny-model result.

## Actual training framework

The base repository does not contain a LLaMA-Factory Trainer integration. Its actual
chain is:

```text
vtraining -> torchrun -> Accelerate DDP -> custom jepa_vlm/train.py loop
          -> Hugging Face Qwen3VLForConditionalGeneration
```

EXP-12 extends this chain. It does not add a second trainer or change the existing
MVBench/TempCompass answer-likelihood scoring rule.

## Architecture delivered

- One physical `hf_model.model.visual`, including one merger. No teacher copy, EMA or
  trainable target tower exists.
- ViT and merger are frozen and run under inference mode; detached clones feed the
  trainable LLM. Frozen visual parameters in any optimizer group are a hard error.
- EXP-11's actual full-LLM policy is retained: decoder blocks train; `embed_tokens` and
  `lm_head` remain frozen as in the existing repository.
- Query/no-query state predictions share a two-layer transition MLP. Query mode adds
  learned index, row, column and horizon embeddings.
- Future targets are independently encoded by the same frozen physical visual module,
  normalized, detached, and never concatenated into the student LLM.
- CE and state use separate LLM forwards. CE labels cover assistant-answer tokens only.
- Event source/direction/condition/Event Query enter the LLM; true and same-video wrong
  targets never do. Event code is disabled in A0–A5.

## Real temporal units and K shapes

EXP-12 resolves:

```text
raw frames = 32
sample fps = 4
temporal patch size = 2
temporal units = 16
state source/target gap = 2 units = about 1 second
```

`duplicate_frames=false` removes the historical `repeat_interleave(2)` path. Tests use
32 numbered/colored frames and verify `(0,1),(2,3),...,(30,31)`, source `i`, target
`i+2`, and no `0,0,1,1` order. Runtime eligibility is based on actual decoded frame IDs,
not merely requested IDs; duplicate, wrong-effective-fps and short samples keep CE but
receive no state loss.

The common pooler maps `[B,T,H,W,D] -> [B,T,K,D]` and uses one aspect-aware row-major
grid for main merger and every DeepStack feature:

| K | Square 8×8 input grid | Output |
|---:|---|---|
| 4 | 2×2 | `[B,16,4,D]` |
| 16 | 4×4 | `[B,16,16,D]` |
| 64 | 8×8 | `[B,16,64,D]` |

Placeholder count, DeepStack injection, visual positions and MRoPE are generated from
the same pooled grid. A non-square-grid unit test verifies aspect orientation.

## Loss and parameter updates

The target center is fp32, non-gradient, all-reduced from global target sum/count,
momentum 0.99, warm-started and checkpointed. State loss is centered cosine with
global DDP dynamic-weight normalization:

```text
L = L_answer_CE + 0.05 * L_state
dynamic_weight = clamp(stopgrad(copy_distance) / 0.05, 0, 1)
```

`beat_copy_loss_weight=0` in A0–A5, but the configurable margin implementation exists.
Logs include raw/centered true and shuffled cosine, margin, norms/std/effective rank,
retrieval top1/top5, pred/copy distance, global persistence ratio and dynamic fraction.

Optimizer groups:

| Group | Contents | LR |
|---|---|---:|
| `base_model` | trainable LLM decoder | `1e-5` |
| `state_query_head` | Query position/horizon embeddings and transition head | `1e-4` |

Pure CE arms have no Query/head group. ViT, merger, token embedding, LM head and unused
legacy mask/reg/MTP modules are absent from the optimizer.

## First-batch configs

- `configs/orca_token_sweep/a0_ce_k4.yaml`
- `configs/orca_token_sweep/a1_query_k4.yaml`
- `configs/orca_token_sweep/a2_ce_k16.yaml`
- `configs/orca_token_sweep/a3_query_k16.yaml`
- `configs/orca_token_sweep/a4_ce_k64.yaml`
- `configs/orca_token_sweep/a5_query_k64.yaml`

Shared resolved training settings: batch/device 1; 800 optimizer updates; cosine warmup
40; save at 400/800; weight decay 0.1; grad clip 1.0; bf16; SDPA; no gradient
checkpointing; no random mask, legacy regression/MTP, dual-view, EMA, beat-copy or Event.

Default local allocation is 4 GPUs/arm, accumulation 8, effective batch 32. The company
job's actual requested allocation is 4 Workers×4 GPUs=16 GPUs/arm, accumulation 2,
effective batch 32; six arms consume 24 Workers×4 GPUs=96 GPUs.

## Checkpoint and evaluation chain

Atomic `checkpoint-N/state.pt` stores trainable tensors, optimizer, scheduler, optimizer
step, data batches seen, RNG, config and state-center auxiliary state. A separately
atomic `checkpoint_meta.json` records byte size and commit. Resume rejects legacy
micro-batch counters, scientific config drift, missing trainable tensors and mixed
completed-arm commits.

Checkpoint 400 uses one frozen stratified TempCompass subset with IDs and SHA256.
Checkpoint 800 evaluates full TempCompass and MVBench. Six arm leaders evaluate in
parallel on one GPU each; the final process collects per-question option scores,
category results, McNemar tests, paired bootstrap intervals, mechanism metrics and
pre-registered deltas. The selection script applies the stated margin/persistence/
MVBench/effective-rank/completeness gates and never auto-submits Event jobs.

Expected server result root:

```text
/data/vjuicefs_sz_ocr_wl/public_data/11193960/runs/exp12/<RUN_ID>/results/exp12_orca_token_sweep
```

## Validation completed locally

Commands:

```bash
python -m compileall -q jepa_vlm scripts/exp12 tests
bash -n scripts/exp12/*.sh scripts/cluster/job_exp12_entry.sh
ruff check jepa_vlm scripts/exp12 tests
git diff --check
pytest -q tests
```

Covered cases include the 14 required test files plus tiny Qwen and result-tool
integration: 32 real frames, 16 units, K=4/16/64, non-square pooling, three DeepStack
levels, MRoPE, frozen single visual tower, optimizer isolation, target/query isolation,
query/no-query parity, centered loss, shuffle baselines, persistence, two-rank Gloo
center equality, center checkpoint, actual tiny train save/resume, Event adjacency/
duration/split/hard-negative/isolation, config equality, job partitioning, materializer,
collector and K selector.

Final local validation result: `27 passed in 9.23s`. Compilation, Ruff, Bash syntax
checking and `git diff --check` also passed.

## Files changed

Core: `jepa_vlm/config.py`, data decoder/datasets plus new temporal/state/Event modules,
pooling/state-loss/state-query/model/train modules, and K-aware evaluator loading paths.

Experiment operations: `configs/orca_token_sweep/`, `configs/orca_event/`,
`scripts/exp12/`, `scripts/cluster/job_exp12_entry.sh`, `job_exp12.yaml`.

Verification/docs: `tests/`, `docs/EXP12_CODE_AUDIT.md`, this report/runbook, README,
ARCHITECTURE, REGISTRY and KNOWN_ISSUES.

## Known limitations and next action

1. Run all six real four-GPU smokes on the exact server Transformers/Qwen build.
2. Confirm production manifest SHA/path loading and record resource audit.
3. Inspect K=64 memory without changing batch/cutoff; patch implementation defects on
   this branch if server-only behavior appears, then recommit and rerun smoke.
4. Submit A0–A5 only after every gate passes; record the platform job ID plus six
   logical IDs and monitor through checkpoint-800 evaluation/automatic release.
5. Do not materialize or submit Event B arms until A selection passes and an audited
   timestamped event manifest is registered.
