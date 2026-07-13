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
| EXP-07 | mtp1 补种子 | V4 判读唯一保留项（四表全正、幅度不足） | 待定（低优先级，仅闲置算力） | — | V4_VERDICT.md 第 3 条 |

## 现行标准（改动须先改此处并全员周知）

- **训练数据**：`qa_train_flow.jsonl`，`min_flow=8.42`（EXP-04/05 现行标准）。
- **CE 代码**：HEAD 与 EXP-04/05 训练代码逐字节一致（model.py 自 V4 判负后零改动）；
  EXP-02 时代旧 CE 公式经单元测试与现公式数值等价。
- **纯 CE 锚点**（新实验直接配对，不得重训）：2B 用 `r3_sft`；8B 用 `v4_ctrl_s0/s1`（双种子）。
  r2_sft_baseline（未过滤数据）已作废，不得再充当对照。
- **种子纪律**：EXP-05 实测对照臂换种子自涨 3.1pp——**单种子结论一律无效**，
  新臂必须 ≥2 种子，或同时与两个 ctrl 种子配对复核。
- **统计纪律**：所有 delta 必须配逐题配对检验（McNemar/配对 t），预注册判定线，
  ±1~2pp 级别的单次读数不构成 finding。
