# Round 2：归因实验 + 基线补齐（无 SSv2 方案）

第一轮结论（详见 results/ 分析）：非平凡门槛全线未过；三个嫌疑因素——数据偏静态、
无 EMA 下的部分塌缩、MTP 加速平滑。本轮目的：**把三个因素拆开归因，并补上缺失的
对照读数**。数据方案：

| 角色 | 数据 | 来源 |
|---|---|---|
| Phase A 预训练 | LLaVA-Video-178K（**运动过滤后**） | 已在集群盘 |
| 类别 probe（评估①替代） | **Diving48**（48 类、外观同质、纯时序判别） | 免注册直链下载 |
| 时序 probe（评估②） | LLaVA val 打乱/倒放 | 无需标签 |
| SFT 基线数据 | LLaVA-Video 原生 QA | `--qa` 提取 |

SSv2 降级为机制验证通过后的最终背书，不阻塞本轮。

## 0. 一次性数据准备

```bash
source scripts/cluster/env.cluster.sh
DATA=/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data

# (a) manifest + QA（--qa 新增；先 head -1 源 jsonl 确认 conversations 字段名）
python scripts/prepare_llava_video.py \
    --root /data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K \
    --subsets 0_30_s_academic_v0_1 --out-dir $DATA/llava_video \
    --max-videos 20000 --qa          # 建议 ≥2 万，2000 步 eff.batch 128 会过 ~13 epoch@2万

# (b) 运动分数（纯 numpy 帧差，无需 opencv）；结束时会打印分位数和建议阈值
python scripts/compute_flow.py --manifest $DATA/llava_video/train.jsonl \
    --out $DATA/llava_video/train_flow.jsonl --method framediff --workers 16
# 记下输出的 "suggested train.min_flow=X"（~p30），下面所有训练用它覆盖

# (c) Diving48（开发机经代理下载后传共享盘；约 6GB）
#   http://www.svcl.ucsd.edu/projects/resound/Diving48_rgb.tar.gz
#   http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_train.json
#   http://www.svcl.ucsd.edu/projects/resound/Diving48_V2_test.json
python scripts/prepare_diving48.py --anno-dir $DATA/diving48 \
    --video-dir $DATA/diving48/rgb --out-dir $DATA/diving48
```

## 1. 训练矩阵（每臂 ~2000 步，3 节点约 40 分钟/臂）

全部经 `train_multinode.sh`，`MIN_FLOW` 换成 0(b) 步输出的阈值：

```bash
MF='train.min_flow=<p30>'
CONFIG=configs/r2_frozen.yaml       EXTRA_OVERRIDES=$MF bash scripts/cluster/train_multinode.sh  # A 数据归因
CONFIG=configs/r2_v21.yaml          EXTRA_OVERRIDES=$MF bash scripts/cluster/train_multinode.sh  # B 主处理
CONFIG=configs/r2_varreg.yaml       EXTRA_OVERRIDES=$MF bash scripts/cluster/train_multinode.sh  # C 抗塌缩
CONFIG=configs/r2_residual.yaml     EXTRA_OVERRIDES=$MF bash scripts/cluster/train_multinode.sh  # D 残差 target
CONFIG=configs/r2_sft_baseline.yaml bash scripts/cluster/train_multinode.sh                      # E SFT 基线(纯CE)
```

| 臂 | 回答的问题 | 看什么 |
|---|---|---|
| A r2_frozen | 换数据后，钉死 target 空间还打不打得赢抄帧？ | ratio vs 第一轮 1.6 |
| B r2_v21 | 数据修正后塌缩还发不发生？ | target_std / adj_cos vs A |
| C r2_varreg | 方差正则能不能按住塌缩且不毁学习？ | target_std≈γ 且 ratio ≤ B |
| D r2_residual | 抄帧收益归零后能不能学出动力学？ | ratio < 1（此时 <1 = 真信号） |
| E r2_sft | 任何训练都会带来 probe 提升吗？（对照） | 只看 probe，不看 ratio |

本轮纪律：**全部关 MTP**（r2_base 默认）；不跑 mask 比例/patch 消融；不跑 Phase B。

## 2. Probe 矩阵（训练完成后，rank0 单卡）

对 **base（不训练）+ A–E 五个 ckpt** 共 6 个模型，各跑 3 个 probe。base 用
`--config configs/r2_v21.yaml`（不带 --ckpt）；其余用各自 output_dir 的
`config.json + step_2000`。以 base 为例：

```bash
FEATS=/data/vjuicefs_sz_ocr_wl/public_data/11193960/feats
# (1) Diving48 类别 probe（主判据）
python -m jepa_vlm.probes.extract_features --config configs/r2_v21.yaml \
    --manifest $DATA/diving48/train.jsonl --data-root $DATA/diving48/rgb \
    --out $FEATS/base_d48_tr.pt --max-clips 8000
python -m jepa_vlm.probes.extract_features --config configs/r2_v21.yaml \
    --manifest $DATA/diving48/val.jsonl --data-root $DATA/diving48/rgb \
    --out $FEATS/base_d48_va.pt
python -m jepa_vlm.probes.linear_probe --train $FEATS/base_d48_tr.pt --val $FEATS/base_d48_va.pt

# (2)(3) 时序 probe：打乱 / 倒放（LLaVA train 子集训 probe，val 评估）
for T in random_shuffle random_reverse; do
  python -m jepa_vlm.probes.extract_features --config configs/r2_v21.yaml \
      --manifest $DATA/llava_video/train.jsonl --out $FEATS/base_${T}_tr.pt \
      --temporal-transform $T --max-clips 4000
  python -m jepa_vlm.probes.extract_features --config configs/r2_v21.yaml \
      --manifest $DATA/llava_video/val.jsonl --out $FEATS/base_${T}_va.pt \
      --temporal-transform $T
  python -m jepa_vlm.probes.linear_probe --train $FEATS/base_${T}_tr.pt \
      --val $FEATS/base_${T}_va.pt --feature layer27_frames
done
# 训练过的模型同理，如：--config <outputs>/r2_v21/config.json --ckpt <outputs>/r2_v21/step_2000
# 汇总：python scripts/summarize_runs.py <outputs_root>（val 列含 nontrivial_ratio）
```

顺手可做：对第一轮的 v21 ckpt 也跑同一套 probe（回答"旧 run 到底学没学到东西"）。

## 3. 判读

核心对比是 **probe(某臂) − probe(base)** 与 **probe(某臂) − probe(SFT)**：

| 结果形态 | 推论 | 下一步 |
|---|---|---|
| A 的 ratio 明显降（→1.2 以下） | 数据是第一因 | 后续全部用过滤数据；考虑更高 min_flow |
| A 不动 | 任务形态问题（copy 路由/4-token 瓶颈） | 提高 tokens_per_frame 或双向注意力再试 |
| B/C/D 的 probe > base 且 > E | 回归目标带来真实表征增益 | 走正式重跑 + SSv2 背书 |
| B/C/D 的 probe ≈ E > base | 提升来自"见过视频"而非回归目标 | 方向存疑，重新设计目标 |
| C 的 target_std 稳住且 ratio 改善 | 塌缩可控，无 EMA 方案可保留 | varreg 并入主配置 |
| D 的 ratio < 1 | 学到了超越抄帧的动力学（最强信号） | 残差形态并入主配置 |

任何臂出现 `target_std → 0`（完全塌缩）立即停该臂，不浪费卡时。

---

# Round 3：主线终审（Phase B pilot，联合 vs 纯 CE）

Round-2 已证明表征增益真实存在（时序 probe +4~4.6pp，非"见过视频"效应，8000 步稳固）。
本轮直接回答总命题：**加了回归目标的联合训练，在时序任务上能否打赢同配置的纯 CE？**
probe 链（Diving48 / ckpt 扫描）降级为诊断工具：失败时归因用，成功时补机理证据用，
**不阻塞本轮**。

## 实验臂（唯一差异 = 回归目标包）

| 臂 | 配置 | loss | 输入 |
|---|---|---|---|
| r3_joint | `configs/r3_joint.yaml` | CE + 0.2·reg | v2.1 mask（50% 帧被 [M] 遮蔽） |
| r3_sft | `configs/r3_sft.yaml` | 纯 CE | 干净视频（v1 无 mask） |

同数据（qa_train_flow.jsonl，**已修复 Round-2 sft 未过滤的混杂**）、同时序 QA 混比
（temporal_qa_ratio=0.3，打乱/倒放判别在线生成）、同 4000 步、MTP off、ViT 放开。
save_every=1000 → 每档 ckpt 都可评，顺带回答"联合训练的最优停点"。

## 运行

```bash
# 0) 一次性：给 QA manifest 补 flow 字段（与 Phase A 同一个脚本、同一阈值）
python scripts/compute_flow.py --manifest $DATA/llava_video/qa_train.jsonl \
    --out $DATA/llava_video/qa_train_flow.jsonl --method framediff --workers 16

# 1) 两臂训练（MIN_FLOW 用 Phase A 定过的阈值，如 8.42）
CONFIG=configs/r3_joint.yaml EXTRA_OVERRIDES='train.min_flow=8.42' bash scripts/cluster/train_multinode.sh
CONFIG=configs/r3_sft.yaml   EXTRA_OVERRIDES='train.min_flow=8.42' bash scripts/cluster/train_multinode.sh

# 2) held-out 时序 QA 评测（单卡；对每档 ckpt 各跑一次，两臂同一 manifest 同一 seed）
python -m jepa_vlm.probes.temporal_qa_eval \
    --config <outputs>/r3_joint/config.json --ckpt <outputs>/r3_joint/step_4000 \
    --manifest $DATA/llava_video/val.jsonl --max-clips 500
```

## 读数与判定（三层，由近及远）

1. **训练稳定性**（跑的过程中就看）：joint 臂的 ce_loss 与 reg_loss 曲线。若 CE 明显高于
   sft 臂且不收敛 → 梯度冲突（方案第 6 节风险项），先调小 λ（0.1）或 mask_ratio 再重跑。
2. **held-out 时序 QA**（主判定，零下载）：`temporal_qa_eval` 的 overall 及 reverse/shuffle
   分项。N=500 时二项噪声 ~2.2pp。**判定：joint − sft ≥ +3pp 且各档 ckpt 方向一致 → 主线
   修改 work**；差距 <2pp 或反向 → 不下结论，进入诊断（Diving48 probe + ckpt 扫描归因）。
   注意：不要拿未训练 base 的绝对值来比——base 没见过 4-token pooled 布局，属于分布外。
3. **正式 benchmark**（结论对外前必补）：lmms-eval 跑 TempCompass / Vinoground / TOMATO
   （时序主指标）+ VideoMME / MVBench（不退化检验）。数据集需开发机代理下载后传集群，
   与 1)/2) 并行准备，不阻塞判定。

## 诊断工具（不通过时再启用）

- Diving48 类别 probe：区分"动作语义"与"低级运动方向"（下载好即可跑，run_probes.sh 追加）。
- ckpt 扫描：formal 跑的 2k/4k/6k/8k 档 + 本轮每千步档，画 probe/QA 随步数曲线找退化点。
- 已知注意事项：joint 臂的 CE 在被 mask 的视频上训练（这是"回归目标包"的一部分）；若怀疑
  mask 本身伤 QA，可加第三臂 r3_joint + mask_ratio=0.25 消融。

# EXP-11：Orca-inspired 目标 @ 冻结 ViT

24-GPU / 3-arm 并行（每臂 8 节点）。EXP-10 已证视觉端训练有贡献，EXP-11 冻结 ViT
（`train_vision: false`）以隔离 Orca 目标本身的效果。基座 Qwen3-VL-2B-Instruct，
数据 `exp10_curated/qa_train_clean.jsonl`（与 EXP-10 相同），4000 步、有效 batch 128、
lr 1e-4、num_frames 16、mask_variant v1、seed 0。

## 实验臂（唯一差异 = 目标/正则）

| 臂 | reg_enabled | mtp_enabled | mask_fraction | 说明 |
|----|-------------|-------------|---------------|------|
| exp11_frozen_sft_s0 | false | false | — | 对照：纯 CE |
| exp11_mask15_s0 | true | false | 0.125 | mask15 回归臂 |
| exp11_orca_obs_s0 | false | false | — | Orca 观测目标臂 |

## 结果（step_4000，rank0 统一评测）

| 臂 | MVBench | TempCompass |
|----|---------|-------------|
| exp11_frozen_sft_s0 | 47.46% | 55.38% |
| exp11_mask15_s0 | 47.43% | 55.57% |
| exp11_orca_obs_s0 | 47.26% | **56.33%** |

完整明细（含 TempCompass 分类、EXP-10 合并对比）见 `results/exp11_orca/comparison.md`。

## 判读

- 冻结 ViT 相比 EXP-10（训练 ViT）掉点 ~1.7-1.9%，视觉端训练对这套 VQA 有实质贡献。
- 三臂 MVBench 打平（差异 <0.2%，噪声内）。
- orca_obs 在 TempCompass 上最优（56.33%），比同系 frozen_sft 高 +0.95%，主要体现在
  direction（+1.79%）与 order（+2.64%）两类时序推理。Orca 观测目标对下游时序理解有微弱正向作用。
