# EXP-11 run24 (batch96) — 4-arm, 24-GPU, MVBench + TempCompass @ step_4000

24-GPU / **4-arm** 并行（每臂 6 节点）版本（`jexp10-scale64-20260721214654481`）。
相比主目录的 3-arm run（batch128），本次**新增第 4 臂 `orca_noquery`**，且
有效 batch 为 **96**（6 节点 × 4 GPU × 4 × GA1）。4 臂训练完成后由 rank0 统一评测。
run 目录：`runs/exp11_orca/exp11-orca24-20260721-214654`。

## 设计

在主 3-arm run 基础上加入 `orca_noquery_s0`：与 `orca_obs_s0` 用相同的
frozen independent-frame states + 两层 predictor，但**去掉 learnable query tokens**，
predictor 直接读当前帧 token hidden states。对比两臂可隔离 **query interface 本身的价值**。

## 实验臂

| 臂 | reg_enabled | mtp_enabled | mask_fraction | orca_use_queries | 说明 |
|----|-------------|-------------|---------------|------------------|------|
| exp11_frozen_sft_s0 | false | false | — | — | 对照：纯 CE |
| exp11_mask15_s0 | true | false | 0.125 | — | mask15 回归臂 |
| exp11_orca_noquery_s0 | false | false | — | **false** | Orca 无 query |
| exp11_orca_obs_s0 | false | false | — | true | Orca 观测目标 |

公共设置：冻结 ViT（`train_vision: false`），基座 Qwen3-VL-2B-Instruct，
数据 `exp10_curated/qa_train_clean.jsonl`，4000 步、有效 batch 96、lr 1e-4、
num_frames 16、mask_variant v1、seed 0。

## 结果（step_4000）

| 臂 | MVBench | TempCompass |
|----|---------|-------------|
| exp11_frozen_sft_s0 | 47.71% | 55.44% |
| exp11_mask15_s0 | 47.51% | **55.82%** |
| exp11_orca_noquery_s0 | **47.81%** | 55.44% |
| exp11_orca_obs_s0 | 47.33% | 55.70% |

### TempCompass 分类

| 类别 | frozen_sft | mask15 | orca_noquery | orca_obs |
|------|-----------|--------|--------------|----------|
| action | 92.01% | **92.60%** | 92.31% | **92.60%** |
| attribute_change | 53.82% | 53.47% | 53.82% | 53.12% |
| direction | 40.30% | 41.49% | **41.79%** | 41.19% |
| order | 52.32% | 52.65% | 51.66% | **53.64%** |
| speed | **36.91%** | **36.91%** | 35.65% | 35.96% |

## 结论

- **MVBench 四臂打平**（47.33–47.81%，差异 <0.5%，噪声内）。
- **TempCompass** mask15 最高（55.82%）、orca_obs 次之（55.70%），frozen_sft 与
  orca_noquery 并列（55.44%），差异均 <0.4%。
- **orca_obs vs orca_noquery（隔离 query interface）**：TempCompass 上 orca_obs 仅高
  +0.26%，MVBench 上 orca_noquery 反而高 +0.48%，均在噪声内 —— **query token 接口
  无显著增益**。
- **跨 run 一致性**：上次 3-arm run（batch128）中 orca_obs 的 TempCompass 优势
  （56.33%，+0.95% over frozen）本次未复现（55.70%，+0.26%），进一步说明此类
  差异大概率是噪声。
- **总体**：冻结 ViT 设定下，Orca 目标 / mask 正则 / query interface 对下游
  MVBench/TempCompass 均无显著影响。
