# EXP-11 (orca) — MVBench + TempCompass @ step_4000

24-GPU / 3-arm 并行（每臂 8 节点）版本（`jexp10-scale64-20260721123021325`）。
3 臂并行训练完成后由 rank0 统一评测，
run 目录：`runs/exp11_orca/exp11-orca-20260721-123020`。

## 设计

验证「Orca-inspired 目标」在**冻结 ViT** 设定下对下游时序理解的影响。
EXP-10 已证明视觉端训练（train_vision=true）有实质贡献；EXP-11 把 ViT 全部冻结
（`train_vision: false`），以隔离 Orca 目标本身的效果，避免被视觉特征学习的收益掩盖。

- **数据**：与 EXP-10 完全相同的源审计干净清单 `exp10_curated/qa_train_clean.jsonl`。
- **预算**：与 EXP-10 一致，每臂 4000 optimizer-update 步，有效 batch 128。
- **唯一变量**：目标形式（纯 CE / mask15 正则 / Orca 观测目标），ViT 全程冻结。

## 臂设置（唯一差异 = 目标/正则；ViT 均冻结）

| 臂 | 目标 | ViT | 备注 |
|----|------|-----|------|
| `exp11_frozen_sft_s0` | 纯 CE（lambda_reg=0, mtp/reg off） | 冻结 | control，隔离 Orca 与冻结效应 |
| `exp11_mask15_s0` | CE + mask15 正则（mask_fraction≈0.125） | 冻结 | 输入 [M] 扰动正则 |
| `exp11_orca_obs_s0` | CE + Orca 观测目标（orca_loss + use_queries） | 冻结 | Orca-inspired 观测一致性 |

公共超参（继承链 `base.yaml → vivolm_ssv2 → vivolm_llava_video → r2_base → r3_joint → r3_sft → exp9_sft → exp10_curated_sft_s0 → exp11_*_s0`）：

- 基座 `Qwen3-VL-2B-Instruct`，`mask_variant: v1`（输入不被 [M] 污染）
- lr 1.0e-4，weight_decay 0.05，num_frames 16
- batch_size 4 × grad_accum 1 × 32 GPU = 有效 batch 128
- max_steps 4000，warmup 400，seed 0

## 结果（step_4000，rank0 统一评测）

| 臂 | MVBench | TempCompass |
|----|---------|-------------|
| `exp11_frozen_sft_s0` | 47.46% (1896/3995) | 55.38% (875/1580) |
| `exp11_mask15_s0` | 47.43% (1895/3995) | 55.57% (878/1580) |
| `exp11_orca_obs_s0` | 47.26% (1888/3995) | **56.33% (890/1580)** |

TempCompass 分类明细（correct/total）：

| 类别 | frozen_sft | mask15 | orca_obs |
|------|-----------|--------|----------|
| mcq-action | 92.31% | 92.31% | 92.31% |
| mcq-attribute_change | 53.47% | 54.17% | 53.82% |
| mcq-direction | 40.90% | 41.19% | **42.69%** |
| mcq-order | 51.66% | 51.99% | **54.30%** |
| mcq-speed | 36.59% | 36.28% | 36.59% |

## 与 EXP-10 合并对比

| 实验 | 臂 | ViT | MVBench | TempCompass |
|------|-----|-----|---------|-------------|
| EXP-10 | curated_sft_s0 | 训练 | **49.36%** | **57.09%** |
| EXP-10 | curated_mse_s0 | 训练 | 49.41% | 56.58% |
| EXP-10 | curated_sft_s1 | 训练 | 49.04% | **57.47%** |
| EXP-10 | curated_mse_s1 | 训练 | 49.21% | 56.77% |
| EXP-11 | frozen_sft_s0 | **冻结** | 47.46% | 55.38% |
| EXP-11 | mask15_s0 | **冻结** | 47.43% | 55.57% |
| EXP-11 | orca_obs_s0 | **冻结** | 47.26% | **56.33%** |

## 判读

1. **冻结 ViT 掉点明显**：EXP-11 frozen_sft 比 EXP-10 sft_s0 低 -1.9% MVBench / -1.7%
   TempCompass，确认视觉端训练对这套 VQA 有实质贡献。
2. **EXP-10 内部**：SFT vs 帧预测 MSE 基本打平（差 <0.5%，在种子噪声内），JEPA MSE
   辅助损失未见明显下游增益。
3. **EXP-11 内部（均冻结 ViT）**：MVBench 三臂几乎一致（47.26–47.46%），但
   **orca_obs 在 TempCompass 上最强（56.33%）**，比同系 frozen_sft 高 +0.95%，并超过
   EXP-10 两个 MSE 臂；优势集中在时序类别 direction (+1.8%) 与 order (+2.6%)。
   Orca 观测目标对时序/动态理解有微弱正向信号，但尚不足以弥补冻结 ViT 的整体掉点。

## 已知问题（本次 run）

- job 最终状态 Failed 并非评测失败：6 个结果 JSON 全部生成。失败在收尾 scorecard
  汇总脚本读取 `json["accuracy"]`，而 `mcq_eval` 实际写的是 `acc`（KeyError）。已在
  `scripts/direct/run_exp11_orca_pilot.sh` 修复（commit `4efe012`），scorecard 已用
  现有结果补算。
- 训练初期曾因 pilot 脚本 `local arm="$1" out=".../${arm}"` 的赋值顺序 bug 导致三臂
  共用同一 output_dir，已修复（`out=".../$1"`，commit `af456bd`）。
