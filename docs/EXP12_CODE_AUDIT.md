# EXP-12 code audit

Date: 2026-07-21  
Repository: `Lancelot-wy/jepa-vlm-single-tower`  
Audited base commit: `2f74c55356da86731b8393b8d8a600e7a6b66bc0`  
Branch: `exp12-orca-token-sweep`

## Scope and evidence boundary

This audit describes the code at the base commit before EXP-12 implementation. It is
based on the checked-out source, configs, cluster YAML, and launch/evaluation scripts.
The checkout does **not** contain LLaMA-Factory Trainer integration. The actual training
stack is a custom PyTorch loop in `jepa_vlm/train.py`, Hugging Face Transformers for
Qwen3-VL, Accelerate for DDP, and the company `vtraining` launcher. EXP-12 will extend
that existing path rather than create a second framework.

The local audit host has no `nvidia-smi` and no installed Transformers package, so
GPU/model-runtime claims require later cluster smoke evidence. Source-level facts and
CPU-testable tensor logic are distinguished from those pending runtime checks.

## 1. Required location audit

| # | Item | Current implementation at `2f74c55` | EXP-12 implication |
|---:|---|---|---|
| 1 | Qwen3-VL load entry | `jepa_vlm/modeling/model.py::build_model`; real model uses `Qwen3VLForConditionalGeneration.from_pretrained`, tiny tests use `tiny_qwen3vl_config` | Keep this loader and Transformers model class. |
| 2 | Trainer integration | No LLaMA-Factory Trainer. `jepa_vlm/train.py::main` is the optimizer-step loop; Accelerate wraps model/optimizer/dataloader | Extend this loop and its checkpoint contract. Do not add a parallel trainer. |
| 3 | Video decoder | `jepa_vlm/data/video_io.py::decode_frames`, PyAV | Add explicit decode/sample diagnostics without replacing PyAV. |
| 4 | Frame sampler | `video_io.py::_sample_indices`; `fps_or_uniform` uses target fps or `linspace` fallback | EXP-12 requests 32 raw frames at 4 fps. Short clips can repeat indices under the existing fallback and must become state-ineligible rather than silently treated as valid state pairs. |
| 5 | `temporal_patch_size=2` | Constant `TEMPORAL_PATCH_SIZE = 2` in `video_io.py`; tiny HF vision config also sets 2 | Parameterize and validate the public config while preserving the Qwen requirement. |
| 6 | Per-frame duplication | `video_io.py::patchify`, `x.repeat_interleave(2, dim=0)` when `duplicate_frames=true`; EXP-11 inherits `true` | This is exactly `f0,f0,f1,f1,...`. EXP-12 must set it off and group real adjacent frames `(f0,f1),(f2,f3),...`. |
| 7 | Tokens after merger | At 256 square input: patch grid 16x16, spatial merge 2 gives an 8x8 merger grid, hence 64 tokens per temporal unit before project pooling | K=64 is the unpooled 8x8 merger output for the current resolution. |
| 8 | Current 64 -> 4 pooling | `modeling/model.py::encode_video` reshapes merger output to `[B,T,8,8,D]`; `modeling/pooling.py::avg_pool_frames(..., 2)` produces 2x2=4 | Replace fixed side 2 with a spatial pooler driven by K and actual grid. |
| 9 | DeepStack production/injection | `self.visual(...)` returns `out.deepstack_features`; `encode_video` pools each with fixed side 2; `_deepstack_embeds` flattens; LLM receives `visual_pos_masks` and `deepstack_visual_embeds` in visual/text forwards | Main and all DeepStack levels must share one pool-grid mapping and token count. |
| 10 | `grid_thw` | Produced by `patchify` in patch units as `[grid_t, grid_h, grid_w]`; model derives merger H/W by `/2`; collators retain only the first batch grid and require equal shapes | EXP-12 needs both original patch grid and derived pooled grid metadata. The processor grid cannot be reused unchanged as an LLM K-token grid. |
| 11 | MRoPE / `position_ids` | `visual_only_position_ids` and `mixed_position_ids` hard-code a 2x2 frame grid and `POOL_SIDE=2` | Must generate row-major positions for the selected pooled HxW grid and advance offsets consistently. |
| 12 | Visual positions in LLM | Phase B collator inserts `num_frames * tokens_per_frame` video placeholders; `_forward_with_text` replaces positions where `input_ids == video_token_id` and asserts exactly `B*T*P` | Placeholder count must use temporal units times K, not raw frame count. |
| 13 | EXP-11 Query | `JepaQwen3VL.__init__` creates `orca_queries`; `_forward_orca_joint` runs separate CE and transition branches; `_orca_transition` appends four queries | It has no spatial row/column or horizon embedding and no fixed instruction. It is the migration source, not the final EXP-12 interface. |
| 14 | Transition head | `orca_head = MLPHead(D, hidden)` in `modeling/heads.py`; no-query and query share it | Retain a common two-layer head specification; create explicit state modules and optimizer grouping. |
| 15 | Current target layer | EXP-11 uses `target_norm(encode_frames_independently(...))`, i.e. normalized frozen Qwen visual merger output after fixed 2x2 pooling | EXP-12 target stays frozen single-tower merger output after the same K-pooling as source. |
| 16 | `requires_grad` today | `build_model`: `lm_head` and token embeddings always frozen; entire `visual` follows `train_vision`; EXP-11 sets `train_vision=false`; full LLM otherwise trainable; query/head trainable | Add hard single-tower/frozen-vision assertions and separate ViT/merger counts. EXP-11 actually uses full LLM, so EXP-12 must keep full LLM. |
| 17 | Optimizer groups | `train.py::make_optimizer`: names matching heads/mask/pool/LoRA/query go at `train.lr`; remaining trainable backbone at `lr_backbone`; frozen params are skipped | Add named state-query/head groups at `state_head_learning_rate`; fail if any visual parameter enters any group. |
| 18 | Accumulation/step count | Accelerate accumulation; `step` increments only when `sync_gradients` and optimizer step was not skipped; scheduler is deliberately not wrapped; checkpoints mark `step_unit=optimizer_update` | Preserve optimizer-update semantics. EXP-12 effective batch must be audited from world size, per-device batch, and accumulation. |
| 19 | Checkpoint/resume | Atomic `step_N/state.pt` stores trainable model tensors, optimizer, scheduler, step, config. Resume restores these. It does not save RNG/sampler state; no running center exists. The EXP-11 helper also emits an invalid historical override name `train.resume_from` although the config field is `train.resume` | Save/restore running center plus RNG and data progress required for reproducibility. EXP-12 launchers must use `train.resume`. |
| 20 | Evaluators | `jepa_vlm/probes/mcq_eval.py`; scripts invoke it for full MVBench and TempCompass and save per-sample JSON | Preserve scoring and answer parsing. Extend orchestration/output only; do not change evaluator rules. |
| 21 | Cluster submit/resource unit | `job_exp11_orca24.yaml` requests 24 **Workers**, each 4 L40S. `job_exp11_orca24_entry.sh` partitions ranks into independent DDP groups; `vtraining run -f` submits | Company resource unit is Worker, not GPU. A 24-Worker EXP-12 job means six independent arms x 4 Workers x 4 GPUs = 16 GPUs/arm and accumulation 2 for effective batch 32. |
| 22 | Manifest/source fields | EXP-10 frozen manifest is `qa_train_clean.jsonl`; emitted rows include `video`, `question`, `answer`, `source_dataset`, `source_category`, `source_id`, provenance, and optional temporal bounds | All A arms must use one immutable file and SHA256. Event records need a separate validated schema. |
| 23 | Data order/reproducibility | Manifest construction sorts deterministically. Per-item temporal augmentation is deterministic from `seed` and manifest index. DataLoader uses `shuffle=True`; global seed makes a fresh run reproducible, but exact sampler position is not checkpointed | Add a resumable deterministic sampler or equivalent saved data-order state for strict resume parity. |
| 24 | Existing no-query | Code and `configs/exp11_orca_noquery_s0.yaml` exist. The committed EXP-11 results do not include a completed no-query score | Reuse behavior only after parity tests. First A sweep requests CE/query; no-query remains implemented for follow-up/B configs. |
| 25 | Existing Event interface | No event dataset, sampler, query path, split guard, hard negative, or event metrics exist | Implement and test it, but keep disabled in A0-A5 and never auto-submit B configs. |

## 2. Current data flow

```text
manifest row
  -> PyAV decode of `train.num_frames` sampled images
  -> resize + center crop
  -> optional repeat_interleave(2) of every image
  -> Qwen temporal patchification (temporal patch size 2)
  -> `grid_thw`
  -> one physical `hf_model.model.visual`
  -> merger output and three DeepStack features
  -> fixed adaptive 2x2 pooling for all four feature streams
  -> T*4 video placeholders in a Qwen chat sequence
  -> trainable LLM
  -> answer-only CE

EXP-11 auxiliary branch:
same patch tensor
  -> rebatched as nominal one-frame entries through the same frozen visual module
  -> four current state tokens (+ optional four learned query vectors)
  -> same LLM
  -> two-layer MLP
  -> raw MSE against normalized future frozen tokens
```

The EXP-11 auxiliary target never uses a second physical ViT, but the duplicate-frame
preprocessor means each old "frame" temporal patch consists of two copies of one
sampled image. EXP-12 changes the semantic unit to two different adjacent real frames.

## 3. Current parameter update path

For the EXP-11 query config as resolved from YAML:

- `train.train_vision=false`: the full `model.visual` subtree, including its merger,
  is frozen.
- `train.train_llm=full`: the language-model decoder blocks are trainable.
- `language_model.embed_tokens` is explicitly frozen.
- `lm_head` is explicitly frozen; gradients still flow through it to decoder hidden
  states when computing answer CE.
- `orca_queries` and `orca_head` are trainable.
- Legacy `mask_embed` and `reg_head` are constructed and can remain trainable even
  when unused; DDP uses `find_unused_parameters=true`. EXP-12 should avoid putting
  unused legacy parameters in the optimizer for its modes.
- The optimizer assigns query/head/new parameters `train.lr` and decoder parameters
  `train.lr_backbone`.

## 4. Concrete hard-coded locations

1. `modeling/model.py`: global `POOL_SIDE=2`.
2. `visual_only_position_ids`: fixed four offsets.
3. `mixed_position_ids`: fixed 2x2 offsets and fixed increments.
4. `JepaQwen3VL.__init__`: asserts `tokens_per_frame == 4`.
5. `encode_video` and DeepStack loop: fixed `avg_pool_frames(..., 2)`.
6. `extract_features`: reshapes with four tokens.
7. `train.py::build_dataloaders`: placeholder count is sampled frame count times
   legacy `tokens_per_frame`.
8. `mcq_eval.py` and other probes: same legacy placeholder calculation and old
   duplicate-frame preprocessing.
9. `orca_transition`: comments, query construction, and positions assume four tokens.
10. `video_io.py`: temporal patch size and duplicate behavior are constants/boolean.

## 5. DeepStack and MRoPE mismatch risks

- Pooling only the main merger stream to K while leaving DeepStack at four tokens will
  make DeepStack injection indices disagree with `visual_pos_masks`.
- Changing placeholders without rebuilding MRoPE will assign invalid spatial geometry.
- Treating original `grid_thw=[16,16,16]` as the pooled LLM grid is incorrect: it is in
  pre-merge patch units, not K-token units.
- A non-square input cannot be represented by `sqrt(K) x sqrt(K)` without an explicit
  aspect-aware target grid rule.
- Query vectors need spatial positions aligned one-to-one with target row-major token
  positions; the EXP-11 sequential query positions do not express that alignment.
- The target may never be concatenated into the student sequence. This must be tested
  at the state-branch boundary, not inferred from attention masks.

## 6. Planned file changes

Core modules:

- `jepa_vlm/config.py`
- `jepa_vlm/data/video_io.py`
- `jepa_vlm/data/datasets.py`
- new `jepa_vlm/data/temporal_units.py`
- new `jepa_vlm/data/event_dataset.py`
- `jepa_vlm/modeling/pooling.py`
- new `jepa_vlm/modeling/state_prediction.py`
- new `jepa_vlm/modeling/state_loss.py`
- `jepa_vlm/modeling/model.py`
- `jepa_vlm/train.py`
- evaluator loading paths only where required for K-aware preprocessing; scoring stays
  unchanged.

Experiment and execution:

- `configs/orca_token_sweep/*.yaml`
- `configs/orca_event/*.yaml`
- `scripts/exp12/*`
- EXP-12 company job YAML and entrypoint
- tests listed by the EXP-12 specification
- `README.md`, `ARCHITECTURE.md`, `REGISTRY.md`, `KNOWN_ISSUES.md`
- `docs/EXP12_RUNBOOK.md` and `docs/EXP12_IMPLEMENTATION_REPORT.md`

## 7. Risks and stop conditions

1. Frozen Qwen visual forward behavior, merger naming, and DeepStack tensor layouts
   depend on the exact cluster Transformers build and require real-model smoke.
2. K=64 multiplies the CE visual sequence length by 16 relative to the old four-token
   model and can OOM. Batch/cutoff may not be silently changed.
3. Real adjacent-frame temporal patches change the visual semantics from EXP-11 and
   invalidate reuse of old checkpoints as matched controls.
4. A 24-Worker company job is 96 GPUs, not 24 GPUs. Incorrect partitioning could create
   one 96-GPU world or six 4-GPU groups with idle workers; both are forbidden.
5. Running-center DDP collectives must execute on every rank whenever state loss is
   enabled. Per-rank conditional skipping would deadlock.
6. Short/duplicated samples need a synchronized validity mask rather than skipping an
   entire forward on only one rank.
7. The current local checkout cannot execute required 4-GPU smoke. Formal submission
   must remain gated until cluster smoke artifacts exist.
8. Event captions in the ordinary EXP-10 manifest are not event annotations and must
   not be auto-promoted.

## 8. Compatibility policy

Retained:

- Existing CE, masked regression, MTP, dual-view, EXP-10/11 config fields and checkpoint
  loading.
- Existing MVBench/TempCompass answer-likelihood evaluator and parsing.
- Existing `torchrun -> Accelerate` and `vtraining` launch chain.
- Legacy `num_frames`, `tokens_per_frame`, and `duplicate_frames` fields for old configs.

Deprecated but not deleted:

- `orca_enabled`, `orca_use_queries`, `orca_query_tokens`, and `orca_target_gap` as
  EXP-11 aliases.
- `duplicate_frames=true` as the historical per-image temporal-unit mode.
- Legacy online MTP and masked objectives.

EXP-12 configs will use only the new explicit temporal-unit and state-predictor fields.
Invalid combinations will fail during config validation rather than silently falling
back to an old path.
