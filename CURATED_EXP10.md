# EXP-10: source-audited caption mixture on one 4×L40S host

This replaces the LLaVA-only derivative as the clean mainline for the next
comparison. It is a paired four-arm experiment:

- CE control vs CE + JEPA/MSE treatment;
- seed 0 and seed 1 for each arm;
- same fixed caption-QA manifest, same synthetic temporal templates, same
  optimizer schedule and 4,000 steps in every arm.

The default mixture uses four processed local sources: native LLaVA-Video QA
(180k), Vript dense temporal captions (150k), InternVid broad video captions
(150k), and OpenVid high-quality captions (80k). The target is up to 560k
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
