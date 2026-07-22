# EXP-12 / EXP-13 / EXP-14 Results

Branch `exp12-orca-token-sweep`. All three experiments are complete.
Result files live under `$BASE/runs/{exp12,exp13,exp13-official,exp14}/...`.

- EXP-12 (Orca single-tower visual-token sweep): 6 arms × 800 steps, all Succeeded.
- EXP-13 (Qwen anchor eval): official-budget 4 protocols + anchor 16 protocols, all Succeeded.
- EXP-14 (K=64 state diagnostics): 8 arms × 800 steps + ckpt-400/800 eval, all Succeeded.

Two blocking bugs were found and fixed during evaluation (see "Fixes" below).

## EXP-12 — Orca token sweep (K=4/16/64 × CE/Query)

`runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep/`
(accuracy, correct/total)

| arm | mode | K | MVBench | TempCompass |
|-----|------|---|---------|-------------|
| a0_ce_k4     | CE    | 4  | 47.61% (1902/3995) | 55.32% (874/1580) |
| a1_query_k4  | Query | 4  | 47.18% (1885/3995) | 55.00% (869/1580) |
| a2_ce_k16    | CE    | 16 | 52.07% (2080/3995) | 57.41% (907/1580) |
| a3_query_k16 | Query | 16 | 51.81% (2070/3995) | 57.09% (902/1580) |
| a4_ce_k64    | CE    | 64 | 54.47% (2176/3995) | 60.19% (951/1580) |
| a5_query_k64 | Query | 64 | 54.39% (2173/3995) | 60.19% (951/1580) |

Headline: **visual-token count K is the dominant driver** (K4→K64 ≈ +6.9pp MVBench,
+4.9pp TempCompass). CE vs Query at fixed K is within noise (≤0.4pp).

## EXP-13 — Qwen anchor evaluation

### official-budget (4 protocols) → GREEN

`runs/exp13-official/official-budget-20260722-192843-a68f784/official_budget_comparison.{json,md}`

Raw/full diagnostic: **GREEN_WITHIN_2_5PP** — 60.68% vs public 61.7% (−1.02pp).
Official-budget reproduction (native-compatible HF runner, not the private harness).

| protocol | accuracy | correct/total | coverage |
|----------|---------:|--------------:|---------:|
| base × full_generation (2048f)  | 60.68% | 2424/3995 | 99.88% |
| ckpt_k64 × full_generation      | 59.57% | 2380/3995 | 99.88% |
| base × cap32                    | 60.63% | 2422/3995 | 99.88% |
| ckpt_k64 × cap32                | 59.07% | 2360/3995 | 99.88% |

Raw/full back in the plausible band ⇒ the A4−raw training delta is interpretable.

### anchor (16 protocols) → Complete: True

`runs/exp13/native-anchor-20260722-192909-a68f784/native_anchor_comparison.{json,md,csv}`

| protocol | MVBench | TempCompass |
|----------|--------:|------------:|
| custom_base_k4_full_option    | 45.91% | 52.91% |
| custom_base_k16_full_option   | 53.02% | 56.71% |
| custom_base_k64_full_option   | 55.92% | 59.62% |
| custom_ckpt_k64_full_option   | 54.47% | 60.19% |
| custom_base_k64_letter        | 55.04% | 51.01% |
| custom_ckpt_k64_letter        | 44.76% | 46.65% |
| native_base_matched32_gen     | 60.75% | 66.96% |
| native_ckpt_k64_matched32_gen | 58.57% | 65.70% |

Paired deltas (significant ones):

| comparison | Δ pp | McNemar p |
|------------|-----:|----------:|
| K64 − K4 (custom base) MVBench        | **+10.01** | 1.2e-37 |
| K64 − K4 (custom base) TempCompass    | **+6.71**  | 5.6e-09 |
| training effect k64 full_option MVBench  | −1.45 | 0.012 |
| training effect k64 letter MVBench       | **−10.29** | 2.2e-34 |
| native − custom (base k64) MVBench    | +5.71  | 6.1e-14 |
| native − custom (base k64) TempCompass | +15.95 | 3.6e-31 |
| native training effect k64 MVBench    | −2.18 | 7.9e-05 |

Takeaways: (1) K is the main driver. (2) **native ≫ custom** (up to +16pp) — the
single-tower custom protocol underperforms native Qwen by a wide margin. (3) K=64
training is roughly neutral-to-slightly-negative on full_option, and **strongly
negative under the letter answer format** (−10.3pp) — letter scoring is fragile.

## EXP-14 — K=64 state diagnostics (8 arms)

`runs/exp14/exp14-20260722-144952-1a5c061/results/exp14_state_diagnostics/comparison.{csv,json}`

Mechanism gate: predictive arm requires `centered_margin>0.10` and
`persistence_ratio<0.90`; benchmark protection: ≤1pp drop vs same-seed CE.

| arm | mode | MVBench | TempCompass | centered_margin | persistence | mech gate | prot gate | candidate |
|-----|------|--------:|------------:|----------------:|------------:|:---------:|:---------:|:---------:|
| b0_ce_seed1              | CE            | 54.47% | 60.19% | 0.0000 | 1.0000 | False | — | False |
| b1_query_seed1           | Query         | 54.39% | 60.19% | 0.0000 | 1.0000 | False | False | False |
| b2_noquery_seed0         | Query(no-q)   | 53.92% | 60.06% | 0.0038 | 0.9981 | False | True | False |
| b3_noquery_seed1         | Query(no-q)   | 53.97% | 60.06% | 0.0000 | 1.0000 | False | True | False |
| b4_query_beatcopy_seed0  | Query(beatcopy)| 54.09% | 60.13% | 0.0043 | 0.9981 | False | True | False |
| b5_query_beatcopy_seed1  | Query(beatcopy)| 54.24% | 60.19% | 0.0041 | 0.9981 | False | True | False |
| a4_ce_k64 (EXP-12 join)  | CE            | 54.47% | 60.19% | 0.0000 | 1.0000 | False | — | False |
| a5_query_k64 (EXP-12 join)| Query        | 54.39% | 60.19% | 0.0000 | 1.0000 | False | False | False |

**No candidate arm.** All arms sit at persistence_ratio ≈ 1.0 with centered_margin ≈ 0 —
the K=64 state is still a **copy/retrieval solution** (retrieval_top1 ≈ 55–56%,
top5 ≈ 81%), not a predictive one. The anti-copy (beat-copy) and no-query variants
nudge persistence marginally below 1.0 but nowhere near the <0.90 gate.

## Fixes applied during evaluation

Two blocking bugs were root-caused and fixed (commits on this branch):

1. **official full_generation OOM** (`18f18bb`): 2048-frame / 224k-video-token
   forward OOM'd a single 46 GB L40S on the first clip. Fixed by sharding the 2B
   model across all 4 local GPUs via HF `device_map=auto` (single process, 1 shard);
   cap32 stayed one-process-per-GPU. Validated 20/20 then full 3995-clip run.

2. **MVBench JSON write crash** (`a68f784`): every MVBench protocol ran inference
   to completion then crashed at `json.dump(ensure_ascii=False)` —
   `UnicodeEncodeError` (ascii container locale) on generated text containing
   non-ASCII chars (e.g. U+2019). Tempcompass was unaffected (ASCII-only outputs).
   Fixed by pinning `encoding="utf-8"` on eval-result writes, shard merge, and
   `load_items` reads (`native_qwen_mcq_eval.py`, `mcq_eval.py`,
   `merge_mcq_results.py`). Also changed the official shard-merge provenance check
   to use the `SHARDS` var instead of a hardcoded 4.

## Reproduce

See `docs/EXP13_EXP14_PARALLEL_RUNBOOK.md` and `docs/EXP12_NATIVE_QWEN_EVAL_RUNBOOK.md`.
Eval is MVBench + TempCompass only; no other protocols.
