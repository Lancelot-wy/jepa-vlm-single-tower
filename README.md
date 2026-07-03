# Masked Latent Regression on Qwen3-VL（视频时序表征实验）

在 Qwen3-VL-2B 的 LLM 主干上施加视觉 latent 回归目标（V1 / V2.1 / V2.2 + MTP），
验证模型能否学到帧间动力学信息。实现对应实验方案全部内容，所有变体与消融由 config 控制。

已在本地完成端到端冒烟验证（tiny 随机模型 + 合成视频，CPU/MPS）：训练全变体、
Phase B 联合训练、三项评估、断点恢复、梯度冻结、**DeepStack 防泄漏单测**均通过；
真实 `Qwen/Qwen3-VL-2B-Instruct` 的 config 已确认与代码兼容（meta device 实例化验证）。

---

## 1. Qwen2.5-VL → Qwen3-VL 的架构适配决策

原方案基于 Qwen2.5-VL-7B。Qwen3-VL 有三个结构性差异，对应如下适配（均已实现）：

| Qwen3-VL 特性 | 对本方案的影响 | 适配方式 |
|---|---|---|
| **temporal_patch_size=2**：视觉塔按 2 帧一组打包成一个时间组 | "帧级 mask"的最小时间单元变成 2 帧 | 默认 `duplicate_frames: true`：每个采样帧复制 2 份进 ViT，1 个 latent 槽位 == 1 个采样帧，mask 语义与原方案严格一致（代价：ViT 算力 ×2，相对 LLM 可忽略）。设 false 则 16 帧 → 8 槽 |
| **DeepStack**：ViT 第 [5,11,17] 层特征额外注入 LLM 前 3 层的视觉 token 位置 | **回归目标泄漏的新通道**：只在输入 embedding 层替换 [M] 不够，真实特征仍会从早层注入 | deepstack 特征做同样的 2×2 pooling 后注入，**mask 位置置零**。已有单测：扰动被 mask 帧像素，LLM 全部 hidden state 差异为 0，target 正常变化（`max |hidden diff| = 0.00e+00`） |
| **文本时间戳 + 逐帧 MRoPE**：视频按帧拆成独立 grid，时间信息靠时间戳文本 token | Phase A 无文本输入 | 每个 latent 帧构造 1×2×2 的 (t,h,w) MRoPE grid，帧间 offset 按原生约定推进；时序信息由 t 维位置编码承载，不引入时间戳 token |

其余与原方案一致的设计点：

- **Pooling 至 4 token/帧**：merger 输出（256px → 8×8=64 token/帧）经 2×2 无参平均池化 → 4 token。
  无参池化是有意的：pooling 参与 target 生成，可学习池化会给"把 target 变简单"开后门。
  `model.pooling: attn` 提供 4-query 注意力池化作对照。
- **h 双重身份**：pooled 特征既是 LLM 输入也是回归 target；target 过无仿射 LayerNorm 后 stop-grad，无 EMA、无独立 target encoder。
- **因果注意力语义**：LLM 是因果解码器，[M] 位置只能看到过去帧 —— V2.1 实际是"从历史重建/预测被 mask 帧"，
  非平凡性 copy 基线相应取**最近的过去未 mask 帧**。`abl_bidir.yaml` 提供双向视觉注意力消融
  （4D mask 直通，此时 MTP 必须关闭，代码会强制检查）。
- **MTP**：k=4 个独立 2 层 MLP，位置 (t,i) 的 hidden 回归 h_{t+j,i}（token 对齐），仅在非 mask 源位置计算。
- **新增参数量**：回归 head + 4 个 MTP head + [M] embedding ≈ 42M（2B 主干的 2%）。

### 塌缩风险（方案第 6 节的补充）

target 侧只有 stop-grad、无 EMA，而视觉编码器默认可训练 —— 编码器存在把特征
拉平从而让回归变容易的塌缩通道。两道保险（已实现）：

1. 训练日志每步输出 **target_std**（→0 = 塌缩）和 **adj_cos**（→1 = 相邻帧 target 无差异，回归退化为平凡）；
2. `abl_frozen_vit.yaml`：冻结 ViT 钉死 target 空间的安全变体，主跑异常时用它定位。

---

## 2. 代码结构

```
jepa_vlm/
  config.py                 # dataclass 配置 + YAML 继承(_base_) + 命令行 key=value 覆盖
  modeling/
    model.py                # JepaQwen3VL 包装器：编码→pool→mask→deepstack→MRoPE→LLM→loss
    pooling.py              # 2x2 avg / 4-query attention pooling
    masking.py              # tube(帧级) / patch(token级,负对照) mask 采样
    heads.py                # 回归 head + MTP heads
  data/
    video_io.py             # PyAV 解码 + fps/uniform 采样 + patchify(照搬 HF 官方布局)
    datasets.py             # manifest 数据集 / shuffle,reverse 变换 / Phase B QA + collator
  train.py                  # accelerate 训练循环(Phase A/B 通用)
  probes/
    extract_features.py     # 中层+末层 hidden state 提取(clip 级 & 逐帧拼接)
    linear_probe.py         # 线性分类 probe(评估 1、2)
    nontrivial_check.py     # copy 基线对比(评估 3, go/no-go 门槛)
scripts/
  prepare_ssv2.py           # SSv2 官方标注 → manifest
  prepare_diving48.py       # Diving48 V2 → manifest
  prepare_ego4d.py          # Ego4D/EPIC 长视频 → 窗口化 manifest(自监督, 无标签)
  compute_flow.py           # Farneback 光流均值 → manifest 的 flow 字段(静态片段过滤)
  make_synthetic.py         # 合成运动方块数据(冒烟/机制 sanity)
configs/                    # 主方案 + 全部消融, 见第 5 节
```

---

## 3. 数据集选型与格式

### 统一 manifest 格式（jsonl，一行一个 clip）

```json
{"video": "42.webm", "label": 87, "label_name": "pushing something from left to right",
 "start": null, "end": null, "flow": 3.21, "duration": 4.1}
```

只有 `video` 必填。`label` 供 linear probe；`flow` 供静态片段过滤（阈值在训练时用
`train.min_flow` 控制，一次计算全局复用）；`start/end` 支持长视频切段（Ego4D/EPIC）。
多数据集混合 = 直接 `cat` 多个 manifest（路径用绝对路径或共同 data_root）。

### 推荐数据集组合

| 数据集 | 角色 | 规模 | 许可/获取 | 适配脚本 |
|---|---|---|---|---|
| **SSv2** | Phase A 主训练 + 主 probe（评估 1） | 22 万 clip / 174 类 | Qualcomm 免费注册下载（webm） | `prepare_ssv2.py` |
| **Ego4D clips 子集** | Phase A 补充多样性（长时程第一视角） | 取 ~20 万窗口 | 需签 license（约 1-2 周审批） | `prepare_ego4d.py` |
| **EPIC-KITCHENS-100** | Ego4D 的替代（license 更快） | 9 万 action segment | 注册即下 | 同 `prepare_ego4d.py`（扫目录窗口化） |
| **Diving48** | 额外时序 probe（外观几乎无差异，时序表征的黄金检验） | 1.8 万 / 48 类 | 免费直接下载 | `prepare_diving48.py` |
| 合成运动方块 | 冒烟 + 机制 sanity | 任意 | 本地生成 | `make_synthetic.py` |

选型理由：SSv2 的类别定义本质依赖时序（"从左推到右" vs "从右推到左"），probe 提升
直接反映时序信息增益；Diving48 背景/外观完全同质，是排除外观捷径的最强检验；
Kinetics 外观偏置严重，**不**建议做主训练数据。时序 probe（评估 2）的打乱/倒放样本
由 `extract_features.py --temporal-transform random_shuffle|random_reverse` 在任意
manifest 上动态生成，不需要单独数据集。

Phase B 评估（TempCompass / Vinoground / TOMATO / VideoMME / MVBench）建议直接用
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) 跑，不在本仓库范围内。

---

## 4. 环境与运行

```bash
# python>=3.10, 先按集群 CUDA 装 torch>=2.4, 然后:
pip install -r requirements.txt          # transformers>=4.57 (本地验证于 5.12)

# ---- 数据准备 (SSv2 为例) ----
python scripts/prepare_ssv2.py --anno-dir /data/ssv2/anno --video-dir /data/ssv2/videos --out-dir data/ssv2
python scripts/compute_flow.py --manifest data/ssv2/train.jsonl --data-root /data/ssv2/videos \
    --out data/ssv2/train_flow.jsonl --workers 16

# ---- Phase A 主实验 (V2.1) ----
accelerate launch -m jepa_vlm.train --config configs/phase_a_v21.yaml \
    train.data_root=/data/ssv2/videos train.output_dir=runs/phase_a_v21
# 任意配置项可命令行覆盖: model.mask_ratio=0.75 train.batch_size=4 ...

# ---- 评估三件套 ----
# (1) 特征提取: 训练后模型 vs 基线(--ckpt 留空 = 未训练基座, 即方案要求的对照)
python -m jepa_vlm.probes.extract_features --config runs/phase_a_v21/config.json \
    --ckpt runs/phase_a_v21/step_6000 --manifest data/ssv2/train.jsonl --out feats/v21_tr.pt --max-clips 50000
python -m jepa_vlm.probes.extract_features --config runs/phase_a_v21/config.json \
    --ckpt runs/phase_a_v21/step_6000 --manifest data/ssv2/val.jsonl --out feats/v21_va.pt
python -m jepa_vlm.probes.extract_features --config runs/phase_a_v21/config.json \
    --manifest data/ssv2/train.jsonl --out feats/base_tr.pt --max-clips 50000   # 基线
# (2) SSv2 linear probe (评估 1) + 时序 probe (评估 2, --temporal-transform random_shuffle/reverse)
python -m jepa_vlm.probes.linear_probe --train feats/v21_tr.pt --val feats/v21_va.pt
# (3) 非平凡性检验 (评估 3, go/no-go)
python -m jepa_vlm.probes.nontrivial_check --config runs/phase_a_v21/config.json \
    --ckpt runs/phase_a_v21/step_6000 --manifest data/ssv2/val.jsonl

# ---- 本地冒烟 (无 GPU、无真权重) ----
python scripts/make_synthetic.py --out data/synthetic --num 32
python -m jepa_vlm.train --config configs/debug_tiny.yaml
```

---

## 5. 配置矩阵（对应方案第 5 节消融优先级）

| config | 内容 |
|---|---|
| `phase_a_v21.yaml` | **主实验**：mask 50% tube，loss 仅 [M] 位，MTP k=4 |
| `abl_patch_mask.yaml` | 消融 1：patch 级随机 mask（预期失效的负对照） |
| `abl_mask25/75.yaml` | 消融 2：mask 比例 |
| `phase_a_v22.yaml` / `phase_a_v1.yaml` | 消融 3：V2.2 全位置 loss / V1 无 mask 对照 |
| `abl_mtp_off.yaml` / `abl_mtp_k1.yaml` | 消融 4：MTP 开关 / k=1 |
| `phase_b.yaml` | 消融 5：λ（`train.lambda_reg=...` 扫 0.1–0.5） |
| `abl_frozen_vit.yaml` / `abl_lora.yaml` / `abl_bidir.yaml` | 附加：冻结 ViT / LoRA 低成本版 / 双向注意力 |

训练日志（`log.jsonl`）逐段输出：`reg_loss`、逐 k 的 `mtp_loss_k{j}`、`copy_mse`、
`target_std`、`adj_cos`；验证时额外输出 `val/nontrivial_ratio`（<0.8 视为非平凡，
详见 `nontrivial_check.py`）。**远步 MTP 塌缩监控**：若 `mtp_loss_k3/k4` 长期不低于
k1 的水平（无信息量），按方案把 `model.mtp_k` 砍到 2。

**决策点**（方案第 7 节）：SSv2 probe 相对基线 +2% 且 nontrivial 通过 → 进 Phase B；
否则停止并分析（先查 `adj_cos` 与 mask 比例/采样帧率）。

---

## 6. 上公司集群（多机多卡）要点

代码有意保持"裸 torch + accelerate DDP"，无 deepspeed/自研框架耦合：

1. **启动方式**：`accelerate launch` 换成公司 harness 的 `torchrun` 等价物即可，
   `train.py` 内部只依赖 `Accelerator()` 的默认 env 初始化（`RANK/WORLD_SIZE/LOCAL_RANK`）。
2. **2B 全参 bf16 + AdamW ≈ 2.1B×16 byte ≈ 34GB 优化器+梯度+权重**，80G 卡单卡即可放下；
   grad checkpointing 已默认开。若用 40G 卡或想加大 batch，接 FSDP/ZeRO-2 均可（模型是标准
   `nn.Module`，无特殊状态）。LoRA 版（`abl_lora.yaml`）单卡 24G 可跑。
3. **注意力后端**：集群上把 `model.attn_implementation` 改为 `flash_attention_2`
   （`abl_bidir.yaml` 例外——4D 自定义 mask 需要 sdpa/eager）。
4. **数据吞吐**：Phase A 序列极短（16 帧×4 token=64 视觉 token），瓶颈在视频解码。
   `num_workers` 建议 ≥8/卡；SSv2 的 webm(VP9) 解码偏慢，如吞吐不足优先转码 mp4/H.264
   或预抽帧。多机时确保视频在本地盘/高速共享存储。
5. **batch 与 lr**：base.yaml 假设有效 batch 128（8 卡×8×accum2）。扩到多机时按线性
   缩放 `lr/lr_backbone` 并等比缩短 `max_steps`（目标：SSv2+Ego4D 合计 ~40 万 clip 过 1–2 epoch）。
6. **断点恢复**：`train.resume=runs/xxx/step_N`（只存可训练参数 + 优化器，V2.1 全参约 8.5GB/档）。

## 7. 已知实现级注意事项

- `mask_tube_max_run=4`：mask 的连续段最长 4 帧，兼顾"连续或分散"两种形态；调大可加难。
- 帧采样：`fps_or_uniform` 在 clip 足够长时按 2fps 随机偏移采样，否则整段均匀采样
  （SSv2 平均 ~4s，多数走 uniform 分支；方案风险项"拉大帧间隔"对应调低 `sample_fps`）。
- `nontrivial_check` 用固定 seed 的 mask，跨 checkpoint 可比。
- Phase B 的时序必要型 QA（帧序打乱/正放倒放判别）由 `QAVideoDataset.temporal_qa_ratio`
  在线生成；SSv2 标签转 QA 需自备转换（把 `label_name` 填进问答模板写成 qa jsonl 即可）。
