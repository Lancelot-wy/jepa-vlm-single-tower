# 代码架构细节（供设计 review）

以主配置（Qwen3-VL-2B，T=16 帧，256px，batch B）为例，所有形状用具体数字标注。

## 1. 端到端数据流（Phase A）

```
视频文件 (webm/mp4)
  │  data/video_io.py: decode_frames()
  │  PyAV 解码; fps_or_uniform 采样: clip 够长按 2fps + 随机偏移(训练)/居中(评估), 否则整段 linspace
  ▼
uint8 帧 (16, H, W, 3)
  │  resize_center_crop(): 短边缩放到 256 → 中心裁剪 → float32 [0,1]
  │  (probe 时可先做 shuffle/reverse 时序变换)
  ▼
(16, 3, 256, 256)
  │  patchify():  归一化 (x-0.5)/0.5
  │  duplicate_frames: 每帧复制 2 份 → 32 帧 (temporal_patch_size=2, 1 latent 槽=1 采样帧)
  │  按 HF 官方布局重排 (照搬 video_processing_qwen3_vl.py, 保证与视觉塔 100% 一致)
  ▼
pixel_values (4096, 1536)  +  grid_thw = (16, 16, 16)        # 4096 = 16槽×16×16 patch, 1536 = 3·2·16·16
  │  collate: 全 batch 同形状直接 stack → (B, 4096, 1536)
  ▼
═══ modeling/model.py: JepaQwen3VL.forward() ═══
  │  encode_video():
  │    visual(flat, grid)   Qwen3VLVisionModel, batch 内多视频拼接、cu_seqlens 隔离
  │       ├─ pooler_output      (B·1024, 2048)   # 1024 = 16槽 × 8×8 merged tokens
  │       └─ deepstack_features 3 × (B·1024, 2048)   # ViT 第[5,11,17]层, 已投影到 LLM 维
  │    reshape → (B, 16, 8, 8, 2048)
  │    pooling.py: 2×2 adaptive avg pool (无参, 默认) 或 AttnPool(4 query, 可选)
  ▼
h (B, 16, 4, 2048)          ← 双重身份: LLM 输入 & 回归 target
  │
  ├─ target = LayerNorm(h.float(), 无仿射).detach()          # stop-grad, 无 EMA
  │
  ├─ masking.py: sample_token_mask()
  │    tube: 帧级, 连续段长度 U[1,4] 混合散点, 直到 ⌈0.5×16⌉=8 帧; 保证 ≥1 mask 帧 & ≥1 未 mask 帧
  │    patch: token 级随机 (负对照消融)
  │    → token_mask (B, 16, 4) bool
  │
  ├─ h_in = torch.where(token_mask, [M] 可学习向量, h)        # apply_mask()
  │
  └─ deepstack 同样 2×2 avg pool → (B,16,4,2048)×3, mask 位置置零   # _deepstack_embeds(), 防泄漏
  ▼
LLM 输入组装 (_forward_visual_only):
  inputs_embeds  (B, 64, 2048)                               # 64 = 16×4, 纯视觉序列, 无任何文本 token
  position_ids   (4, B, 64)   visual_only_position_ids():
      row0 text  = arange(64)                                # 仅用于因果 mask 对齐
      rows1-3    = 帧 f 的 4 个 token: t=2f, h∈{2f,2f+1}, w∈{2f,2f+1}   # 原生"每帧独立 1×2×2 grid"约定
  visual_pos_masks = 全 True (B, 64)
  deepstack_visual_embeds = 3 × (B·64, 2048)                 # 注入 decoder 第 0/1/2 层, 加法注入
  attention_mask: 默认 None(因果);  bidirectional_visual=true 时传 4D 全 0 float mask 直通
  ▼
Qwen3VLTextModel (28 层, 因果, interleaved-MRoPE)  use_cache=False
  ▼
hidden (B, 64, 2048)  →  reshape (B, 16, 4, 2048)            # 已过 final RMSNorm
  │
  ├─ reg_head:  MLP(2048→2048→GELU→2048)  → pred (fp32)
  │     V1:   loss = MSE(pred, target)  全位置 (无 mask)
  │     V2.1: loss = MSE(pred[token_mask], target[token_mask])   仅 [M] 位   ← 主方案
  │     V2.2: loss = MSE(pred, target)  全位置 (输入有 mask)
  │
  ├─ mtp_heads: 4 个独立 MLP, head j: 位置(t,i) 回归 target[t+j, i] (token 对齐)
  │     源位置: 非 mask 且 t ≤ 15-j;  目标帧可以是被 mask 帧(target 是真值, 与输入 mask 无关)
  │     mtp_loss = mean_j MSE_j;  逐 j 记录 mtp_loss_k{j} (监控远步塌缩)
  │
  └─ total loss = reg_loss + mtp_loss   (等权, fp32)
```

监控指标（每步随 loss 记录，`_monitors()`，全部在 normed target 上算）：
- `target_std`：per-dim std over (B,T,P) 再平均 → →0 = 表征塌缩（无 EMA 的主风险）
- `adj_cos`：相邻帧 target 余弦 → →1 = 相邻帧无差异，回归退化平凡
- `copy_mse`：每个被 mask 帧用**最近的过去未 mask 帧**（无过去则取最近未来帧）的 target 直接当预测的 MSE
  → 评估 3 的分母；`val/nontrivial_ratio = reg_loss/copy_mse`，< 0.8 判非平凡

## 2. Phase B 差异路径 (_forward_with_text)

```
QACollator 拼 chat 格式:  <|im_start|>user\n <vs>[VID×64]<ve> {question}<|im_end|>\n<|im_start|>assistant\n {answer}<|im_end|>
labels: 仅 answer 段有效, 其余 -100;  右 padding
  ▼
embed_tokens(input_ids) → 视觉位置 (input_ids==video_token_id) 替换为 h_in
position_ids: mixed_position_ids()  文本段三维同值递增, 视频帧段同 Phase A 约定, 逐样本扫描
CE = cross_entropy(lm_head(hidden)[:-1], labels[1:])  fp32
total = CE + λ·(reg + mtp)     λ = train.lambda_reg (默认 0.2)
```

时序必要型 QA 由 `QAVideoDataset.temporal_qa_ratio`(默认 0.3) 在线生成：50% 概率打乱/倒放帧序，
问 "帧序是否正确"，答 yes/no。

## 3. 训练循环 (train.py)

- accelerate DDP，模型原生 bf16（无 AMP），pixel_values 在 loop 内 cast。
- 参数两组：新参数（reg_head/mtp_heads/mask_embed/attn_pool/lora_）lr=1e-4；主干（ViT+LLM）lr=1e-5。
  AdamW betas=(0.9, 0.95), wd=0.05；warmup 500 步 + cosine；grad clip 1.0；grad ckpt 默认开。
- 冻结逻辑（build_model）：lm_head、embed_tokens 恒冻结；`train_vision`/`train_llm: full|lora|frozen`；
  LoRA 用 peft `inject_adapter_in_model`（r=64, α=128, 全 7 类 proj）。
- checkpoint 只存可训练参数 + optimizer + scheduler + step + config（V2.1 全参 ~8.5GB/档）；
  `train.resume=` 恢复。
- 每 eval_every 步跑 quick_eval（val manifest，纯视觉 forward）输出 val/* + nontrivial_ratio。

## 4. 评估三件套 (probes/)

1. `extract_features.py`：no-mask forward + output_hidden_states，取指定层（默认 mid=14, last=27）
   视觉位置 hidden：`clip`=全 token 平均 (N,2048) 供类别 probe；`frames`=逐帧平均后 concat (N,16×2048)
   供时序 probe。`--ckpt` 留空 = 未训练基座（方案要求的对照基线）。
   `--temporal-transform random_shuffle|random_reverse` 在线生成 50/50 正负样本，label 覆写 0/1。
2. `linear_probe.py`：特征标准化 → 单层线性 AdamW 30 epoch，报 best top-1。
3. `nontrivial_check.py`：固定 seed 的 mask（跨 ckpt 可比），输出 reg/copy 比值 + 塌缩指标 + PASS/FAIL。

## 5. 配置系统

dataclass（ModelConfig/TrainConfig）+ YAML `_base_` 继承 + 命令行 `key=value` 覆盖；
未知键报错。全部变体/消融只改 config，不碰代码。

## 6. 方案未规定、由实现拍板的设计点（review 重点）

| # | 决策 | 理由 | 改动成本 |
|---|---|---|---|
| 1 | `duplicate_frames=true`：帧复制×2 保证 1 槽=1 帧 | mask 语义与原方案严格一致 | 关掉即 16 帧→8 槽，ViT 算力减半 |
| 2 | 默认无参 avg pool，attn pool 仅作变体 | pooling 参与 target 生成，可学习池化给"把 target 变简单"开后门 | 配置开关 |
| 3 | target = pooled merger 输出（即 LLM 输入本身），LN 无仿射 | 方案原文"h 同时作输入与 target"；LN 防尺度漂移 | 若想回归 ViT 更深特征需加一路投影 |
| 4 | 默认因果注意力：V2.1 实为"从过去帧预测被 mask 帧" | 不动预训练注意力模式；MTP 与之自洽 | `abl_bidir.yaml`（须关 MTP，代码强制） |
| 5 | copy 基线 = 最近**过去**未 mask 帧 | 与因果语义对齐（模型看不到未来） | 双向消融时应改为双侧最近帧（当前实现仍取过去优先） |
| 6 | reg 与 MTP 等权相加（1:1），无额外权重超参 | 方案未规定；两者量纲相同（都是 normed MSE） | 加一个 config 键即可 |
| 7 | MTP 目标帧可以是被 mask 帧（target 用真值） | target 与输入 mask 无关，监督信号更多 | 一行过滤即可改 |
| 8 | deepstack mask 位置置零（加法注入 → 置零=不注入），而非学习向量 | 最保守的防泄漏；已单测 hidden diff=0 | 换 per-level [M] 向量很容易 |
| 9 | heads 接在 final RMSNorm 之后的 hidden 上 | 取标准输出；probe 也取各层输出 | 若要 pre-norm hidden 需 hook |
| 10 | 256px → 每帧 64 merged token → pool 到 4（16:1 压缩） | 方案定 4 token/帧；256px 是留给 pooling 的信息余量 | frame_size 可调（32 的倍数） |
| 11 | Phase A 纯视觉序列：无 BOS、无 vision_start/end、无时间戳文本 | 最干净的回归设置；时序由 MRoPE t 维承载 | 加 special token 需改 `_forward_visual_only` 拼接 |
| 12 | MRoPE 帧间 offset=+2（原生 `max(h,w)//merge` 约定） | 复刻 Qwen3-VL 自身对视频帧的处理 | — |
| 13 | 效率默认：eff batch 128（8 卡×8×accum2）、6000 步 ≈ SSv2+Ego4D 2 epoch、lr 1e-4/1e-5 | 方案"1-2 epoch 即可评估" | 集群规模定了后按线性缩放 |
| 14 | probe 时序特征 = 逐帧 mean 后 concat（16×2048 维） | mean-pool 全 clip 会抹掉太多顺序信息 | 也可改成取 [t] 末 token 等 |
| 15 | `mask_tube_max_run=4`：连续 mask 段最长 4 帧 | "连续或分散"的折中实现 | 配置键 |

## 7. EXP-12 单冻结视觉塔状态预测

前六节描述历史 mask/MTP 路径；EXP-12 保留这些入口，但 A0–A5 不调用它们。

```text
32 raw frames @ 4 fps
  -> real temporal patching, tp=2
  -> 16 units: (f0,f1), (f2,f3), ..., (f30,f31)
  -> one frozen Qwen ViT + one frozen merger
       ├─ native full-video output -> common spatial pool K -> CE video placeholders
       └─ each unit independently encoded -> common spatial pool K
             source unit i -> student LLM -> query/no-query hidden -> 2-layer MLP
             target unit i+2 -> inference_mode + detach -> centered-cosine target
```

`SpatialVisualTokenPooler` 的输入为 `[B,T,H,W,D]`，根据真实 merger grid 的长宽比选择
恰好含 K 个位置的 factor grid，并按 row-major 输出 `[B,T,K,D]`。主 merger 特征和每个
DeepStack level 使用同一 `pooled_grid`；视频 placeholder、MRoPE、attention mask 和
DeepStack 注入位置同步变为 `16×K`。

Observation Query 的第 `(row,col)` 个向量为：

```text
learned_query[index]
+ spatial_row_embedding[row]
+ spatial_col_embedding[col]
+ horizon_embedding[1.0 second]
```

student 序列只含当前 K 个 frozen visual tokens、一个固定 observation special token 和
K 个 Query。未来 target 不进入 student LLM。no-query 模式使用相同 source/target/horizon、
相同 LLM 和相同 transition MLP，只把当前视觉位置的 hidden 送入 head。

状态目标在 fp32 中计算。每卡先累积 target sum/count，再 all-reduce 得到全局 batch center；
running center 以 momentum 0.99 更新并独立写入 checkpoint。动态权重和 DDP loss 分母也按
全局有效 token 归一化：

```text
d_pred = 1 - cos(normalize(pred-center), normalize(target-center))
d_copy = 1 - cos(normalize(current-center), normalize(target-center))
dynamic_weight = clamp(stopgrad(d_copy) / 0.05, 0, 1)
L_state = global_weighted_mean(dynamic_weight * d_pred)
L_total = L_CE + 0.05 * L_state
```

`persistence_ratio` 使用全局 `mean(d_pred) / mean(d_copy)`。小于 0.90 才满足预注册的
“明显优于复制当前状态”门槛；只看 state loss 下降不能证明学到了动态。

Event 路径复用同一物理视觉模块、transition head 和 running center。source、direction、
condition 与 Event Query 进入 LLM；true target 和 same-video wrong-event negative 只过冻结
视觉模块。Event manifest 按 video_id hash 切分，并检查事件边界、媒体时长和 previous/next
邻接关系。A0–A5 的 `event_condition_enable=false`，因此 Event 不影响首批结果。
