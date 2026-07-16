# EXP-10: source-audited unified video-instruction mixture

This replaces the LLaVA-only derivative as the clean mainline for the next
comparison. It is a paired four-arm experiment:

- CE control vs CE + JEPA/MSE treatment;
- seed 0 and seed 1 for each arm;
- same fixed video-instruction manifest, same synthetic temporal templates, same
  optimizer schedule and 4,000 steps in every arm.

The default mixture uses four processed local sources: native LLaVA-Video QA
(180k), Vript dense temporal caption conversations (150k), InternVid broad
caption conversations (150k), and OpenVid temporal-grounding conversations
(80k). The target is up to 560k
examples and a minimum of 460k after local-file validation. This gives each
4,000-step arm approximately one pass over the mixture instead of looping a
small manifest many times. WebVid is intentionally deferred: its watermarked,
short-caption breadth is lower value than these four sources.

LLaVA-Video keeps its ActivityNetQA and NExT-QA components for this
MVBench+TempCompass-only experiment. The launcher excludes PerceptionTest and
the other named MVBench source collections. This makes a later evaluation on
NExT-QA or ActivityNetQA invalid unless the training rows are restricted to
their official train split.

## What the gates prove

`audit` proves that selected metadata and local video files can be read. `prep`
emits only existing videos, records source IDs and provenance, and removes exact
video-path / file-name / source-ID collisions with MVBench and TempCompass.
The source whitelist excludes known benchmark-derived mixtures.

This is **not** a proof against renamed, re-encoded, or cropped copies of the
same video.  Report results as “known-source excluded and ID/path checked”, not
as universally decontaminated.

## Launch safeguards (implemented)

- `max_steps` is an **optimizer-update** count. With four GPUs, batch four and
  `GRAD_ACCUM=8`, each 4,000-step arm consumes 512,000 examples. Checkpoints
  carry `step_unit=optimizer_update`; legacy checkpoints from before this rule
  must not be resumed silently.
- EXP-10 deliberately sets `val_manifest: ""`. The old inherited LLaVA-only
  validation file was neither prepared nor source-matched and could fail at
  step 500. It is not a valid proxy for this mixture.
- The CE and MTP arms use deterministic per-manifest-index temporal transforms
  for a fixed seed. This makes the paired comparison reproducible; an eventual
  multi-epoch run needs an explicit shared epoch seed rather than OS-random
  transforms.
- Both the CE and MTP code paths receive a real two-update, four-GPU smoke
  test. The MTP arm logs `mtp_persistence_mse`, `mtp_persistence_ratio`, and
  `mtp_gain_vs_persistence`; a falling MTP loss alone is not mechanism evidence.
- The collator reserves answer-token budget and logs answer/question truncation
  rates. The manifest builder removes exact cross-source real-path duplicates
  while retaining multiple native QA turns within one source/video.

Read [KNOWN_ISSUES.md](KNOWN_ISSUES.md) before changing source caps, loss
weights, temporal sampling, or the evaluation protocol.

For the canonical vivolm queue submission (four Workers × four L40S GPUs; one
arm per Worker), read [VIVOLM_EXP10.md](VIVOLM_EXP10.md).  Do not submit this
experiment through the historical `job.yaml` or `scripts/cluster/submit_batch.sh`.

The processed paths, conversation contract, category semantics, and bounded
loading behavior are documented in [UNIFIED_VIDEO_DATA.md](UNIFIED_VIDEO_DATA.md).

## Direct commands

Run the stages inside `tmux` so an SSH/terminal disconnect does not terminate
the foreground job:

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
git pull --ff-only origin main
mkdir -p /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs
tmux new -s exp10
bash scripts/direct/run_exp10_curated_4gpu.sh audit |& tee /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp10.audit.log
bash scripts/direct/run_exp10_curated_4gpu.sh prep |& tee /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp10.prep.log
bash scripts/direct/run_exp10_curated_4gpu.sh smoke |& tee /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp10.smoke.log
bash scripts/direct/run_exp10_curated_4gpu.sh train |& tee /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp10.train.log
bash scripts/direct/run_exp10_curated_4gpu.sh eval |& tee /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp10.eval.log
```

## Data scale rationale

With per-GPU batch 4, four GPUs, and gradient accumulation 8, each 4,000-step
arm consumes 512,000 examples. A 460k–560k manifest yields about 1.1–0.9
passes per arm, which uses diversity without pretending that every available
million-scale sample can be consumed by this fine-tune. `train.min_flow` is
zero: the existing `framediff` number is useful for diagnostics, but its old
LLaVA-derived threshold is not a validated measure of temporal quality for this
mixture.
