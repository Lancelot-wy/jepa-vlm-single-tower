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
| EXP-10 | 来源审计的四源混合数据 | LLaVA-Video QA + Vript + InternVid + OpenVid；CE vs CE+MSE × 2 seeds，4,000 步，v2 增广 | **开放（当前唯一主线）**；本地视频可解析、来源白名单和评测 ID/路径去重均为开训硬门槛 | — | CURATED_EXP10.md |

## 现行标准（改动须先改此处并全员周知）

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

读法：mask 是净负资产（C9）；干净输入 + visual loss 中 target 空间决定生死
（C10 输出空间自预测=零，C7 编码器空间帧预测=唯一正方向）；设计空间已基本覆盖，
唯一悬置格 = C7，即 EXP-07 的由来。

## 评测口径（哪把尺子测的哪些数，禁止混用）

| 结果 | 尺子 | 绝对值可否外报 |
|---|---|---|
| vlm-jepa 8 臂 MVBench（base 55.65 等） | vlm-jepa 自研 eval_mvbench_gen（32帧/裸模板/生成式） | 否，待 EXP-06 重锚 |
| EXP-04 benchmark（MVBench 49.59 / TempCompass 57.78） | 本仓库 mcq_eval（pooled 管线，似然） | **永不可**（见下） |
| EXP-04 held-out 时序 QA | 本仓库 temporal_qa_eval | 否（内部判定用） |
| EXP-05 OVO/StreamingBench | 本仓库 streaming_eval（官方式协议自实现） | 否（内部配对判定用） |

规则：① 已有判定全部基于同尺内逐题配对差，不因换尺子推翻；② **绝对值禁止跨尺子
比较、禁止对外报**——对外唯一口径是 VLMEvalKit（EXP-06 建立）；③ 本仓库 pooled
4-token 管线的模型结构上无法进 VLMEvalKit，其绝对值永远仅内部有效，这是架构决定
而非历史遗留；④ 若 EXP-06 发现某尺内 delta 在强协议下消失，记为"格式效应" finding，
不改写尺内结论，但降级其外部意义。
