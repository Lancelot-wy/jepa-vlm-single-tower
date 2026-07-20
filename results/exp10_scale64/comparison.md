# EXP-10 (scale64) — MVBench + TempCompass @ step_4000

64-GPU / 16-Worker 版本（`jexp10-scale64`）。4 臂并行训练完成后由 rank0 统一评测，
run 目录：`runs/exp10_curated/exp10-scale-20260717-140637-bbdc9c3`。

## 设计

验证「帧预测 MSE 辅助损失 vs 纯 CE」在源审计干净数据上的下游时序增益，双种子做稳健性。

- **基座**：Qwen3-VL-2B-Instruct，bf16 / sdpa，256px、4 tokens/frame（avg pool），16 帧 @ 2fps。
- **2×2 = 4 臂**（唯一变量 = 目标函数 × 种子）：
  - `sft`：纯 CE（`lambda_reg=0`，mask v1，无回归 / MTP）。
  - `mse`：CE + 0.2·帧预测 MSE（`mtp_k=1` 预测下一帧，回归头关，无 mask）。
  - 种子 `s0=0` / `s1=1`。
- **数据**：`jepa_data/exp10_curated/qa_train_clean.jsonl`（560k QA），源审计、剔除 benchmark 派生 QA 混料；4 臂共享同一冻结 manifest。
- **训练**：4000 optimizer steps、warmup 400、有效批量 128 恒定，lr 1e-4（新参）/ 1e-5（backbone），wd 0.05，grad_clip 1.0，梯度检查点，训 ViT + 全量 LLM，每 250 步存 ckpt。
- **评测**（step_4000，贪心 MCQ）：MVBench_v3_5_0（total=3995，skip=5）+ Tempcompass_v3_5_0（total=1580，skip=5960，跨臂一致，为合并文件里非 TempCompass 条目按 task 过滤）。

## 结果（step_4000）

| arm | MVBench acc | TempCompass acc |
|---|---|---|
| exp10_curated_sft_s0 | 49.36% (1972/3995) | 57.09% (902/1580) |
| exp10_curated_sft_s1 | 49.04% (1959/3995) | 57.47% (908/1580) |
| exp10_curated_mse_s0 | 49.41% (1974/3995) | 56.58% (894/1580) |
| exp10_curated_mse_s1 | 49.21% (1966/3995) | 56.77% (897/1580) |

### SFT vs MSE（双种子均值）

| 目标 | MVBench | TempCompass |
|---|---|---|
| SFT (CE) | 49.20% | **57.28%** |
| MSE (CE + 0.2·frame-MSE) | **49.31%** | 56.68% |
| Δ (MSE − SFT) | +0.11 pt | −0.60 pt |

## 判读

两臂基本打平：MVBench MSE 略高 +0.11pt，TempCompass SFT 略高 +0.60pt，均在种子间方差
（<0.4pt）量级内，**无统计显著差异**。在该数据规模下，0.2·帧预测 MSE 辅助损失未展现出
可辨识的下游时序增益。原始逐题计数见 `scorecard.json`。
