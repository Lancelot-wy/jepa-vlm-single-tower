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

# Round 3（lite）：主线终审——只改 loss 的最小控制变量对照

回答总命题：**CE 训练加上 masked latent 回归目标，能否更会答时序问题？**

## 设计（一句话：以已训完的 r2_sft_baseline 为锚，只改 loss，其他一个字不动）

| 臂 | loss | 视频输入 | 是否要训 |
|---|---|---|---|
| r2_sft_baseline | 纯 CE | 干净（v1） | **复用，不重跑** |
| r3lite_joint | CE + 0.2·reg | v2.1 mask 50% | 训（~1h） |
| r3lite_joint_m25 | CE + 0.2·reg | v2.1 mask 25% | 训（~1h） |

三臂共享：未过滤 qa_train.jsonl（**不要重新生成 manifest，用 r2_sft 训练时的同一份文件**）、
2000 步、warmup 200、MTP off、ViT 放开、temporal_qa_ratio 0.3。启动时**不加** min_flow 覆盖。

说明：本轮放弃运动过滤——过滤是为 Phase A 纯回归引入的，CE 不需要；未过滤只会让
处理臂吃亏（静态片段稀释回归信号），属于**保守检验**：在不利数据上赢下来的结论更硬。

## 运行

```bash
# 1) 两个新臂（不加 min_flow）
CONFIG=configs/r3lite_joint.yaml     bash scripts/cluster/train_multinode.sh
CONFIG=configs/r3lite_joint_m25.yaml bash scripts/cluster/train_multinode.sh

# 2) 评测：三臂同 manifest 同 seed（r2_sft 现在就能先跑，不用等训练）
for E in r2_sft_baseline r3lite_joint r3lite_joint_m25; do
  python -m jepa_vlm.probes.temporal_qa_eval \
      --config <outputs>/$E/config.json --ckpt <outputs>/$E/step_2000 \
      --manifest $DATA/llava_video/val.jsonl --max-clips 500
done
```

## 判定（预注册）

1. **主命题**：joint − sft ≥ +3pp（overall，N=500 二项噪声 ~2.2pp）→ 回归目标 work；
2. **剂量**：m25 ≈ m50 → 比例不敏感；m25 > m50 → 50% 过高，后续降比例；
3. **joint ≤ sft 时不下"没用"结论**——补一对过滤数据的 joint/sft（configs/r3_joint.yaml +
   r3_sft.yaml 仍保留，即为此分支准备）分辨"方法不行"还是"静态数据拖累"。
4. 训练中盯 joint 臂 ce_loss vs sft 臂历史曲线，明显劈叉 = 梯度冲突，降 λ=0.1 重跑。

与 vlm-jepa 侧的关系：训练层面两边永远无法控制变量（数据/管线/架构均不同），
统一的是**评测尺子**（VLMEvalKit / TempCompass 线照旧并行）；不为跨仓库可比性
迁就本仓库的训练设定。
