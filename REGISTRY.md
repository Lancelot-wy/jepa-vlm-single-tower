# 实验总账（唯一编号，禁止别名）

所有实验用 `EXP-NN` 编号引用；新实验先在此登记再开跑。

| 编号 | 名称 | 内容 | 状态 | 判定 | 记录位置 |
|---|---|---|---|---|---|
| EXP-01 | R1 表征线 | 2B，5 臂 masked latent 回归，未过滤数据，6000 步 | 关闭 | 非平凡门槛未过；部分塌缩；MTP=塌缩加速器 | results/（round1 各目录） |
| EXP-02 | R2 归因 | 过滤数据 5 臂 + probe 矩阵 + SFT 对照 | 关闭 | 表征增益真实（reverse probe +4.6pp，非"见过视频"效应） | results/round2 |
| EXP-03 | R2 正式重跑 | v21/varreg 8000 步 + probe | 关闭 | 增益稳固；shuffle 长训回落 | results/round2（*_formal） |
| EXP-04 | R3 联合终审 | 2B，r3_joint vs r3_sft，qa_train_flow，4000 步 | 关闭 | overall +1.03pp（<+3pp 门槛）；乱序察觉 19% vs 5%（真实机理信号）；公开 benchmark 持平 | results/round3（results-cluster 分支） |
| EXP-05 | V4 流式线 | 8B-LoRA，11 臂（S1/S2/双CE消融/双种子），OVO+StreamingBench | **关闭（预注册负结果）** | S2 净负；S1 未确立；dvce25 显著更差（正则假说否定）；种子噪声 ±3.1pp | results/v4_streaming_eval + V4_VERDICT.md |
| EXP-06 | vlm-jepa 评测锚定 | VLMEvalKit base 锚定 + 5 模型 × MVBench/TempCompass | 开放 | — | vlm-jepa 仓库 VLMEVALKIT.md |
| EXP-07 | mtp1 补种子（8B） | V4 判读唯一保留项（四表全正、幅度不足） | 待定（低优先级，仅闲置算力） | — | V4_VERDICT.md 第 3 条 |
| EXP-08 | 无 mask 纯 MSE 收益（旧小数据） | r3_mse vs r3_sft，2×2 种子 | **撤销**（旧数据 2.5 万 QA×20 epoch 不健康，并入 EXP-09 直接在扩充数据上测；r3_* 新配置不提交） | — | — |
| EXP-09 | 扩充数据上的纯 MSE 收益（原方案） | LLaVA-178K + NExT-QA train + v2 增广 | **暂停 / 不得作为正式主线启动**：NExT-QA train 未证实，LLaVA 复合来源仅靠路径名与 basename 不能完成来源级去污染 | — | EXECUTE_NOW.md（历史记录） |
| EXP-10 | 来源审计的四源混合数据 | LLaVA-Video QA + Vript + InternVid + OpenVid；CE vs CE+MSE × 2 seeds，4,000 步，v2 增广 | **开放（当前唯一主线）**；本地视频可解析、来源白名单和评测 ID/路径去重均为开训硬门槛 | MVBench 49.36 / TempCompass 57.09（sft_s0） | CURATED_EXP10.md |
| EXP-11 | Orca-inspired 目标 @ 冻结 ViT | 同 EXP-10 数据；frozen_sft vs mask15 vs orca_obs，`train_vision=false`，4,000 步 | **已完成**；冻结 ViT 隔离 Orca 目标效果；三臂 MVBench 打平，orca_obs TempCompass 最优 | MVBench 47.46 / TempCompass 56.33（orca_obs） | results/exp11_orca/comparison.md |
| EXP-12 | Orca 单塔 visual-token sweep | 32 真实帧/16 units；K=4/16/64 × CE/Observation Query；冻结 ViT+merger；800 updates | **已完成**；K=64（当前 256px 原生 grid）最优，Observation Query 无增益（persistence>1，门控 FAIL）；不得在同分辨率直接扩 K>64 | MVBench 54.47 / TempCompass 60.19（a4_ce_k64，内部自研尺） | results/exp12_orca_token_sweep/README.md、docs/EXP12_RUNBOOK.md |
| EXP-13 | Qwen 原生协议锚定 | matched-32 的 raw/A4 多协议诊断；另加 2fps、2048 帧、224K 总/640-unit 预算的 raw/A4×full/cap32 MVBench 锚点 | **已配置，待服务器运行**；official-budget 仅称 reproduction，不冒充私有官方 harness | — | docs/EXP12_NATIVE_QWEN_EVAL_RUNBOOK.md、docs/EXP13_EXP14_PARALLEL_RUNBOOK.md |
| EXP-14 | K64 state 机制诊断 | 复用 A4/A5 seed0；补 CE/Query seed1、no-query×2、Query+anti-copy×2；800 updates、双种子 | **已配置，待服务器运行**；先要求 margin>0.10、persistence<0.90，再看同 seed benchmark protection；禁止自动进 Event | — | docs/EXP13_EXP14_PARALLEL_RUNBOOK.md |

## 现行标准（改动须先改此处并全员周知）

- **当前主线**：EXP-12 训练已完成；EXP-13 baseline 恢复与 EXP-14 K64 机制诊断并行。
  24 Worker 分配为 7（matched-32）+4（official-budget）+12（六组训练）+1 预留。
  EXP-10/11 的已有结果继续作为历史证据，不冒充 EXP-12/13/14 对照结果。
- **训练数据**：EXP-04/05 的 `qa_train_flow.jsonl`/`min_flow=8.42` 仅是历史 LLaVA
  分布的记录，**不得迁移**到新来源。EXP-10 使用来源审计后的
  `qa_train_clean.jsonl`，`min_flow=0`；`framediff` 只保留为诊断指标，不能冒充
  真实光流或时间语义质量标签。
- **CE 代码**：HEAD 与 EXP-04/05 训练代码逐字节一致（model.py 自 V4 判负后零改动）；
  EXP-02 时代旧 CE 公式经单元测试与现公式数值等价。
- **纯 CE 锚点**（新实验直接配对，不得重训）：2B 用 `r3_sft`；8B 用 `v4_ctrl_s0/s1`（双种子）。
  r2_sft_baseline（未过滤数据）已作废，不得再充当对照。
- **种子纪律**：EXP-05 实测对照臂换种子自涨 3.1pp——**单种子结论一律无效**，
  新臂必须 ≥2 种子，或同时与两个 ctrl 种子配对复核。
- **统计纪律**：所有 delta 必须配逐题配对检验（McNemar/配对 t），预注册判定线，
  ±1~2pp 级别的单次读数不构成 finding。

## 设计空间矩阵（CE × mask × visual loss 全组合；新提案先查此表防重复）

| # | CE | mask | visual loss | 实例（实验编号） | 结果 |
|---|:-:|:-:|---|---|---|
| C1 | ✗ | ✓ | masked 回归 | v21（EXP-01~03） | 表征增益真实（probe +4.6pp），loss 门槛未过 |
| C2 | ✗ | ✗ | 回归（全位置） | v1（EXP-01） | 平凡解对照 |
| C3 | ✗ | ✓ | 回归+MTP | EXP-01 默认 | MTP=塌缩加速器 |
| C4 | ✓ | ✗ | ✗ | r3_sft / v4_ctrl×2 | 对照锚（2B/8B） |
| C5 | ✓ | ✓ | masked 回归 | r3_joint（EXP-04） | +1.03pp<门槛；乱序察觉 19% vs 5% |
| C6 | ✓ | ✓双视图 | masked 回归 | v4_dv*（EXP-05） | 净负 |
| C7 | ✓ | ✗ | MTP 帧预测 | v4_mtp1/4（EXP-05） | **唯一方向全正**（+1.7pp 单种子不显著）→ EXP-07 |
| C8 | ✓ | ✓ | 回归+MTP | v4_dv25_mtp1 | 未确立 |
| C9 | ✓ | ✓ | ✗（双CE） | v4_dvce25 | 显著更差：mask 污染 CE 实锤 |
| C10 | ✓ | ✗ | next-hidden 自预测 | vlm-jepa λ 系列 | 零效应（剂量平坦） |
| C11 | ✓ | ✗ | frozen-merger future state + spatial Query | EXP-12 A1/A3/A5 | 零/负效应：所有 K 上 query≤CE，persistence>1（退化为复制），门控 FAIL |
| C12 | ✓ | ✗ | Event-conditioned adjacent-event state | EXP-12 B3/B5 | 仅代码/模板；A 批完成前禁止提交 |
| C13 | ✓ | ✗ | frozen-merger future state，无 query / Query+anti-copy | EXP-14 | 已注册双种子；待跑，Event 不自动衔接 |

读法：mask 是净负资产（C9）；干净输入 + visual loss 中 target 空间决定生死
（C10 输出空间自预测=零，C7 编码器空间帧预测=唯一正方向）；设计空间已基本覆盖，
唯一悬置格 = C7，即 EXP-07 的由来。

## 评测口径（哪把尺子测的哪些数，禁止混用）

| 结果 | 尺子 | 绝对值可否外报 |
|---|---|---|
| vlm-jepa 8 臂 MVBench（base 55.65 等） | vlm-jepa 自研 eval_mvbench_gen（32帧/裸模板/生成式） | 否，待 EXP-06 重锚 |
| EXP-04 benchmark（MVBench 49.59 / TempCompass 57.78） | 本仓库 mcq_eval（pooled 管线，似然） | **永不可**（见下） |
| EXP-12 benchmark（MVBench 54.47 / TempCompass 60.19） | 本仓库 mcq_eval（K64、连续视觉 block、完整选项似然） | 否，仅内部配对 |
| EXP-13 native anchor | Qwen 官方数学的无 torchvision 兼容预处理/MRoPE + 匹配 32 帧 + greedy 字母生成 | 仍是内部锚；可判断同协议 SFT delta，不冒充官方榜单 |
| EXP-13 official-budget anchor | 真实视频 2fps、最多 2048 帧、224K 总/640-unit token budget、技术报告 prompt + greedy 生成 | 仅为官方预算复现；先看 raw/full 是否落入 61.7 的合理诊断带 |
| EXP-04 held-out 时序 QA | 本仓库 temporal_qa_eval | 否（内部判定用） |
| EXP-05 OVO/StreamingBench | 本仓库 streaming_eval（官方式协议自实现） | 否（内部配对判定用） |

规则：① 已有判定全部基于同尺内逐题配对差，不因换尺子推翻；② **绝对值禁止跨尺子
比较、禁止对外报**——对外唯一口径是 VLMEvalKit（EXP-06 建立）；③ 本仓库 pooled
4-token 管线的模型结构上无法进 VLMEvalKit，其绝对值永远仅内部有效，这是架构决定
而非历史遗留；④ 若 EXP-06 发现某尺内 delta 在强协议下消失，记为"格式效应" finding，
不改写尺内结论，但降级其外部意义。
