# Direct EXP-09 LLaVA-only run (one 4×L40S host)

This is the direct-server replacement for Channel-bus. It does not submit a
platform job and does not require Blue Code. The run is deliberately named
`exp09_llavaonly_*`: without an official NExT-QA **train** split, it is a
LLaVA-only derivative rather than the original full EXP-09.

The launcher performs these hard gates before formal training:

1. Confirms the project, local Qwen checkpoint, LLaVA source, both offline
   benchmarks, write access, and exactly four visible GPUs.
2. Takes a deterministic uniform reservoir sample over the complete eligible
   LLaVA copy (not the first files in directory order), records provenance, and
   excludes configured MVBench/NExT-QA upstream path patterns. Inspect the
   emitted source inventory as a second audit: path patterns alone cannot prove
   split-level source disjointness.
3. Removes direct benchmark video-key overlaps, scores flow once per unique
   video, expands the score to all QA pairs, and checks dataset-size/decode
   gates.
4. Runs a full-batch two-step 4-GPU smoke test before the four formal arms.

On the server:

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
git pull --ff-only
bash scripts/direct/run_exp09_llavaonly_4gpu.sh preflight
```

Run the full pipeline in a durable detached process (safe if the SSH/terminal
window closes):

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
mkdir -p /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs
nohup bash scripts/direct/run_exp09_llavaonly_4gpu.sh all \
  > /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp09_llavaonly.log 2>&1 < /dev/null &
echo $! > /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp09_llavaonly.pid
tail -F /data/vjuicefs_sz_ocr_wl/public_data/11193960/logs/exp09_llavaonly.log
```

It uses all four local GPUs for each training arm, one arm after another. The
original plan's 8-GPU effective batch is preserved with `batch_size=4`,
`GRAD_ACCUM=8`, and world size 4. If an interrupted arm already has a saved
checkpoint, restart only training with:

```bash
RESUME=1 bash scripts/direct/run_exp09_llavaonly_4gpu.sh train
```

Results are written to
`/data/vjuicefs_sz_ocr_wl/public_data/11193960/results/exp09_llavaonly/`;
`scorecard.json` contains the MVBench and TempCompass aggregate scores. The
four evaluation arms run concurrently, one per GPU, only after all training
arms finish.
