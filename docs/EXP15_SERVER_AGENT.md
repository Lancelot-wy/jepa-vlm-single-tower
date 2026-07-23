# EXP-15 server Agent execution contract

This is the authoritative handoff for the next server-side coding Agent. It is
not a suggestion list: follow the stages in order, satisfy every hard gate, and
record evidence in `results/exp15/AGENT_STATUS.md`. The machine-readable mirror
is `contracts/exp15.yaml`.

## 1. Decision and expected outcome

EXP-15 replaces the current custom pooled training path with a native-Qwen
training path and runs six controlled arms:

| arm | seed | answer objective | temporal objectives |
|---|---:|---|---|
| `c0_native_ce_seed0` | 0 | native answer-only CE | none |
| `c1_native_ce_seed1` | 1 | native answer-only CE | none |
| `o0_native_ce_obs_seed0` | 0 | same CE batches | isolated-frame Observation |
| `o1_native_ce_obs_seed1` | 1 | same CE batches | isolated-frame Observation |
| `e0_native_ce_obs_event_seed0` | 0 | same CE batches | Observation + adjacent Vript Event |
| `e1_native_ce_obs_event_seed1` | 1 | same CE batches | Observation + adjacent Vript Event |

Use 24 Workers × 4 L40S. Split them into six independent worlds of 4 Workers /
16 GPUs. Each arm runs 4,000 optimizer updates and atomically saves at 500,
1,000, 2,000, and 4,000. A successful candidate may later resume to 8,000 and
10,844; that continuation is not part of the first launch.

The job must train, run native-generation evaluation, collect paired results,
and exit. It must not depend on a Blue Code terminal staying open after the
platform accepts the submission.

## 2. Why this correction is necessary

The existing evidence does not validate the intended ORCA mechanism:

- Raw Qwen under the native matched-32 runner reached 60.75% MVBench and 66.96%
  TempCompass. EXP-12 A4 reached 58.57% and 65.70% on that same native runner;
  the official-budget MVBench comparison was 60.68% raw versus 59.57% A4.
- The large K=4→64 gains were mostly restoration of visual information discarded
  by manual pooling. K=64 was the native 256-pixel merger grid; it was not an
  ORCA gain and does not justify a larger-K sweep.
- The current training collator manually creates a continuous visual block and
  custom position semantics, while the stronger evaluator uses native Qwen
  video/chat semantics. Framework choice is not the cause: native preprocessing
  and train/eval parity are. Do not migrate to LLaMA-Factory merely to change a
  label; keep Transformers + Accelerate and factor one native I/O implementation
  shared by training and evaluation.
- EXP-12 used 32 frames as 16 two-frame temporal units. Its one-second target
  therefore was not the requested independent single-frame source/target
  construction. EXP-14 then produced persistence≈1 and margin≈0 for every state
  arm.
- The old Event path omitted Query1 and assumed one video path plus timestamps.
  Processed Vript rows are scene clips, so an adjacent event pair normally has
  different source and target clip paths.
- The old “CE” manifest mixed 180k VQA rows with 300k generic captions and 80k
  grounding rows, and inherited `temporal_qa_ratio=0.3`. It was not a clean
  native VQA SFT control.
- EXP-11's batch-128 and batch-96 records are both seed 0. Their direction/order
  signal is weak evidence, not a replicated two-seed result and not a reason to
  discard the idea without the corrected implementation.
- EXP-14 seed-1 aggregates need raw provenance verification because some values
  are exactly equal to reused seed-0 arms. Do not base a claim on summaries alone.

## 3. Exact server start

If the repository is absent:

```bash
BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
cd "$BASE"
git clone --branch exp15-native-orca --single-branch \
  https://github.com/Lancelot-wy/jepa-vlm-single-tower.git
cd jepa-vlm-single-tower
```

If it already exists:

```bash
PROJECT_ROOT=/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
cd "$PROJECT_ROOT"
git status --short
git fetch origin
git switch exp15-native-orca 2>/dev/null || \
  git switch --track -c exp15-native-orca origin/exp15-native-orca
git pull --ff-only origin exp15-native-orca
```

Do not overwrite a dirty checkout. Preserve and identify existing changes before
switching. Then record the checkout and run the first gate:

```bash
git branch --show-current
git rev-parse HEAD
git log -12 --oneline
bash scripts/exp15/00_agent_preflight.sh
```

The preflight report is written under the shared `runs/exp15` tree. If a mount
is absent, report the exact path. If code or data validation fails, diagnose and
fix it; do not stop after pasting the first traceback.

## 4. Execution stages

### S0 — checkout and environment

Required evidence:

- branch and full commit;
- clean starting status;
- model, VQA source, Vript, InternVid, MVBench, and TempCompass paths readable;
- Python/Torch/Transformers/Accelerate/PyAV versions;
- GPU model/count on the development Worker;
- vtraining CLI path and shared-run-root write test.

Do not pull inside a GPU Pod. Every Pod compares `git rev-parse HEAD` with a full
commit supplied in the job YAML and aborts on a dirty checkout or mismatch.

### S1 — recover raw evidence before interpreting old results

Inspect the shared EXP-12/13/14 run roots. Preserve or copy into an EXP-15
provenance directory:

- resolved configs and job YAML;
- trainer logs and environment snapshots;
- per-item native predictions for raw Qwen and A4;
- EXP-14 per-item predictions and mechanism logs for every seed;
- model/checkpoint, manifest, evaluator, and git hashes.

Explicitly compare the supposedly independent EXP-14 seed rows. If only an
aggregate summary exists, mark the arm `UNVERIFIED_PROVENANCE`; do not manufacture
confidence from the summary. Native-evaluate A4 checkpoint 400 if it exists.
This is a cheap check for early benefit versus 800-step overtraining, but it
must not select the new EXP-15 checkpoint.

### S2 — one native Qwen path for training and evaluation

Factor common native video/chat processing from
`jepa_vlm/probes/native_qwen_mcq_eval.py` into a reusable module. Both training
and evaluation must use that implementation for:

- native processor/chat-template construction;
- video frame metadata/timestamps and dynamic `grid_thw`;
- placeholder expansion, attention mask, and MRoPE/position IDs;
- assistant answer extraction and generation formatting.

The CE forward is complete native video + question + answer. Labels are `-100`
for system text, user text, video placeholders, prompt separators, and padding;
only assistant answer tokens contribute CE. Do not insert state queries into
this forward. Do not mask the answer video. Set `temporal_qa_ratio=0`.

Required parity tests:

1. The shared train/eval builder produces identical prompt IDs, visual values,
   grids, masks, and positions for the same inference example.
2. With labels removed, the training wrapper's logits/generation match the
   native evaluator within dtype tolerance.
3. Altering prompt/video/padding labels does not alter answer-only CE; altering
   an answer label does.
4. Raw Qwen through the refactored evaluator remains in the existing native
   diagnostic band. A large score change blocks training.

Dynamic native visual tokens replace the artificial K=4/16/64 independent
variable. Do not manually pool the CE video to K64 and call it native.

### S3 — construct three source-specific data streams

#### Clean VQA/CE stream

Use audited LLaVA-Video-178K VQA rows only. Preserve original question and
answer; no online synthetic order/speed/pan replacement. Generic Vript and
InternVid captions and OpenVid grounding answers do not enter this control.

Each row must carry `source_dataset`, `source_split`, normalized parent
`video_id`, original `source_id`, absolute media path, and provenance. Remove
known MVBench source collections and exact normalized source-ID/path collisions
with both evaluation manifests. Group-split train/validation by
`source_dataset + parent_video_id`; no parent may cross splits. Report source
counts before/after every filter and a manifest SHA-256.

This is “known-source excluded + exact-ID/path checked,” not a claim of universal
semantic decontamination. Motion filtering and decontamination are separate;
never call frame difference or optical flow a contamination check.

#### Observation stream

Use validated processed InternVid and Vript clips for the primary pilot. They
provide breadth without turning generic captions into CE. Sample a source frame
and an actual adjacent future frame from the same clip. Decode timestamps and
reject/record too-short, duplicate, corrupt, or near-static pairs. Keep OpenVid
out of the primary six arms; it can be a later data ablation because its short
synthetic-grounding distribution is not needed to test the mechanism.

Write train/validation manifests grouped by parent video ID. Record effective
time gap, frame hashes, motion score, source histogram, decode failure rate,
static rejection rate, and overlap audit.

#### Vript adjacent-event stream

Read the processed `Vript_caption_4k/*.jsonl` rows. Group scene clips by parent
video identifier, parse and sort `Scene-NNN`, and pair only N with N−1 or N+1.
The schema must contain separate `source_video_path` and `target_video_path`.
Use the adjacent target scene caption as the event condition, balance previous
and next directions, and sample a real inner frame from the target clip.

Reject missing/ambiguous scene numbers, non-adjacent pairs, duplicate paths,
missing media, and groups with fewer than two valid scenes. Build a same-parent
wrong-event negative that is not the true adjacent target. Split by parent video
before expanding directions. Assert zero parent overlap and report all counts.

### S4 — implement the corrected ORCA objectives

The [ORCA paper](https://arxiv.org/abs/2606.30534) is a reference, not a reason
to copy every scale choice. The critical mechanisms for this test are isolated
single-frame states, learnable queries, the two-layer matching head, and
adjacent-event prediction.

#### Frozen state encoder

Use one physical Qwen ViT and one merger. Freeze both and assert they are absent
from optimizer groups. The LLM is trainable as in EXP-11. Targets run under
`torch.inference_mode()` and are detached.

Qwen's ViT has `temporal_patch_size=2`. For an Observation/Event state, encode
one physical frame independently and duplicate that same frame only inside that
isolated ViT call: `[f_t, f_t]`. This is padding of one timestamp, not the old
continuous `[f0,f1]` temporal unit. The target uses a separate call
`[f_target,f_target]`. Never place source and target timestamps in the same state
ViT call, and never expose target visual tokens to the student LLM. CE continues
to use a normal complete native video.

#### Observation

Student input: independently encoded current frame, fixed observation
instruction, then 256 learnable Query1 tokens. Query1 hidden states pass through
a shared two-layer `D -> 8D -> D` MLP and predict the separately encoded adjacent
future frame latent. This is next-state prediction, not interpolation between
past and future frames.

#### Event

Student order is exactly:

```text
source frame -> Query1 -> adjacent-event instruction -> Query2
```

Both query sets contain 256 learnable tokens. Query2 hidden states pass through
the transition MLP and predict a random frame from the specified adjacent
previous or next Vript scene. The target clip/frame is never in the student
sequence. This fixes the missing Query1 in the existing event prototype.

#### Loss and optimizer

For each visual pair:

```text
L_pair = 0.1 * MSE(pred, detached_target)
       + 0.9 * (1 - cosine(pred, detached_target))
```

The paper's task coefficients are Observation/Event/VQA = 0.1/0.5/0.4. For the
controlled arms, keep the CE gradient scale identical to CE-only and express
the auxiliary terms relative to VQA:

```text
CE:             L = L_ce
CE+Observation: L = L_ce + 0.25 * L_obs
CE+Obs+Event:   L = L_ce + 0.25 * L_obs + 1.25 * L_event
```

Record this exact formula in resolved configs and logs. Keep the same CE sample
sequence and CE exposure for matched seed arms. Treat the paper's 5:15:1
Observation/Event/VQA sampling ratio as a reference. Benchmark a 50-step
throughput sample; if the exact ratio misses the overnight budget, preregister
one reduced ratio in the status file and use it for both seeds. Never change it
silently mid-run.

Use one optimizer policy across all six arms. Start conservatively with LLM LR
`1e-5`, new-query/head LR `1.2e-4`, minimum LR `1e-6`, weight decay `1e-8`,
cosine decay, and 120 warmup steps. The paper's base LR `3.5e-5` is a reference,
not a mandatory copy; do not add an LR sweep to the six-arm causal comparison.

Do not add blanket copy loss to the primary arms. Static or near-static frames
can legitimately match. Log copy/persistence and negative margins. If a
faithful arm still copies, a later *dynamic-only* contrastive margin is a new
experiment, not an unregistered hot fix.

### S5 — required tests and smoke ladder

Create the exact artifacts listed in `contracts/exp15.yaml`, then run:

```bash
python scripts/exp15/validate_contract.py implementation
bash scripts/exp15/01_run_tests.sh
bash scripts/exp15/02_run_smokes.sh
```

Tests must cover at least:

- native train/eval tensor and logits parity;
- answer-only CE masking;
- one-frame isolation despite `temporal_patch_size=2`;
- source and target are different timestamps and separate ViT calls;
- frozen ViT/merger, target detach, and optimizer exclusion;
- Query1/Query2 order, shape, gradients, and target exclusion;
- `D -> 8D -> D` head and 0.1/0.9 pair loss;
- Event previous/next adjacency and same-video wrong-event negative;
- no parent-video split overlap and deterministic manifests;
- per-objective DDP reduction and equal CE exposure;
- atomic checkpoint saving plus full RNG/dataloader/optimizer/scheduler/center
  resume equivalence;
- six 16-GPU groups from 24 Workers, unique ports/output paths, peer failure
  propagation, fixed commit, and automatic exit.

Smoke order is mandatory:

1. CPU/tiny unit tests (logic only).
2. One GPU with real Qwen and a tiny real sample for CE, Observation, Event.
3. One 4-GPU Worker: all objectives plus save/resume/native eval.
4. Two Workers / 8 GPUs: one short DDP run for each objective family.
5. 24-Worker job dry-run with no placeholders.

Do not substitute a tiny random model for the real-Qwen gates. If native parity
fails, do not fall back to the old custom collator/evaluator.

### S6 — full job and resume behavior

The completed launcher interface must be:

```bash
# inspect only
bash scripts/exp15/03_submit.sh --dry-run

# submit a new fixed-commit run
bash scripts/exp15/03_submit.sh

# status
bash scripts/exp15/04_status.sh --run-id <run-id>

# resume failed/incomplete arms from their latest atomic checkpoint
bash scripts/exp15/05_resume.sh --run-id <run-id>
```

Before submission, save the rendered job YAML, fixed full commit, clean status,
resolved six configs, manifest hashes, source audits, package/CUDA environment,
resource audit, and smoke logs under the run root. The entrypoint derives six
groups from `TF_CONFIG`; it must never create one 96-GPU process group.

Each arm leader runs held-out VQA/mechanism checks at checkpoints. Do not use an
MVBench or TempCompass subset to choose a checkpoint. After step 4,000, each arm
leader runs full native greedy generation on both benchmarks, writes per-item
predictions, and computes paired comparisons. Followers should exit as soon as
their required train/eval work is complete. When all collectors finish, the
rank-0 process writes `completed` and exits so the platform releases all Pods.

On failure, preserve the failed attempt logs, fix code on this branch, commit
and push, then submit a new attempt pinned to the new full commit. Resume only
checkpoints whose config/model/manifest hashes match. Never mutate code or pull
inside a running Pod.

### S7 — mechanism and benchmark verdict

Observation passes only on held-out dynamic pairs when:

- persistence ratio < 0.90;
- centered true-vs-shuffle margin > 0.10;
- true targets beat shuffled-batch and shuffled-position targets;
- target/prediction variance and effective rank show no collapse.

Event passes only when true adjacent targets beat same-video wrong-event and
shuffled targets, both previous and next directions are represented, and the
target-leakage test passes. Event is an independent objective: do not cancel it
solely because Observation failed in EXP-14.

For benchmark claims, report every seed and the two-seed mean, matched-seed
deltas against native CE, McNemar tests, and paired bootstrap confidence
intervals. A benchmark gain without a mechanism gain is a data/format effect,
not proof of state prediction. A mechanism gain with flat benchmarks is still
a valid representation result but not a TempCompass/MVBench improvement.

The raw-Qwen native anchor remains the absolute sanity reference. A clean CE
arm that falls materially below it requires a data/optimization diagnosis; do
not hide the regression behind a custom pooled score.

## 5. Explicitly out of scope for the first launch

- K>64 or another visual-token sweep;
- mask15 (already run) or masked-video answer CE;
- generic caption scaling as the CE baseline;
- NExTQA dependency;
- OpenVid in the primary Observation mix;
- blanket copy penalty;
- trainable ViT in the six ORCA arms;
- benchmark-based checkpoint selection;
- changing more than one scientific factor between matched arms.

The old trainable-ViT/variance-regularized line remains a separate follow-up:
Round 2 showed reverse-probe +4.6pp and later runs had weak positive signals.
Run it only after EXP-15 or on newly available resources, never by silently
mixing it into the frozen-ViT ORCA comparison.

## 6. Completion evidence

The Agent may mark `COMPLETE` only after the repository contains:

- implementation and all required tests;
- exact commands and rendered 24-Worker YAML;
- clean data-audit reports and manifest hashes;
- fixed-commit smoke and resume logs;
- all four checkpoints for every arm (or a documented terminal arm failure);
- native per-item MVBench/TempCompass predictions;
- paired result tables and mechanism metrics;
- a scientific verdict that distinguishes preprocessing, CE, Observation, and
  Event effects;
- platform evidence that the training/evaluation Pods exited.

Commit the final report and result metadata to `exp15-native-orca`. Large model
checkpoints and raw videos stay on shared storage; record their paths and hashes
in the repository.
