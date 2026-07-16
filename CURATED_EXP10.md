# EXP-10: source-audited caption mixture on one 4×L40S host

This replaces the LLaVA-only derivative as the clean mainline for the next
comparison.  It is a paired four-arm experiment:

- CE control vs CE + JEPA/MSE treatment;
- seed 0 and seed 1 for each arm;
- same fixed caption-QA manifest, same synthetic temporal templates, same
  optimizer schedule and 4,000 steps in every arm.

The default source is Vript only.  It is intentionally high-quality and dense
enough to supply 220k–260k locally-resolved clips; do not add a large collection
merely for its nominal row count.  InternVid is registered but disabled until
the mounted metadata is proven to resolve to actual local video files.  WebVid
is a later breadth ablation, not the default, because of short/noisy captions
and watermarks.  OpenVid has no supplied local video root and is therefore not
eligible yet.

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

After the Vript-only run, add the secondary source only if its audit is clean:

```bash
TRAIN_SOURCES='vript internvid' FORCE_PREP=1 \
  bash scripts/direct/run_exp10_curated_4gpu.sh prep
```

That rebuild changes the training manifest, so it is a new experiment rather
than a resume of a Vript-only arm.

## Data scale rationale

With per-GPU batch 4, four GPUs, and gradient accumulation 8, each 4,000-step
arm consumes 512,000 examples.  A 220k–260k manifest yields roughly 2.3–2.0
passes per arm, versus about eight passes for a 60k manifest.  This is enough
data to test the method without pretending that millions of clips can be used
in a 4,000-step fine-tune.  `train.min_flow` is zero: the existing `framediff`
number is useful for diagnostics, but its old LLaVA-derived threshold is not a
validated measure of temporal quality for this mixture.
