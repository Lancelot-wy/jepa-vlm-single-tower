# V4 训练线判读（2026-07-13，按预注册标准执行）

数据：results-cluster 分支 `results/v4_streaming_eval/`（12 臂 × 2 模式 × OVO+SB，逐题 jsonl）。
分析：`scripts/summarize_streaming.py`（配对翻转 + 符号检验），并以 ctrl 两个种子分别与合并复核。

## 判定

- **S2（dual-view tail mask）：未通过。** OVO/recent 两个 seed 均净负（−9/−5），种子合并 Δ=−1.16pp。
- **S1（mtp1 帧间预测）：未确立。** 四表方向全正但幅度 ~+1.7pp，vs 任一 ctrl 种子均不显著。
- **dvce25（双 CE 无预测项）：显著更差**（OVO/recent p=0.003）→ mask 污染 CE 有害；
  "增益来自正则"假说否定（本来也无增益可解释）。
- **种子噪声警示：ctrl 换种子自涨 3.1pp（SB/recent，p=0.041）**——第一轮对 ctrl_s0 的
  数个"显著"臂在 ctrl_s1 面前全部失效。任何后续单种子结论都不可信。
- **管线验证成功**：base→SFT 显著 +4~6pp（两榜 p<0.02），训练/评测/回传全链路可复用。

## 结论与禁止事项

1. 本线按预注册**关闭并记为负结果**：预测目标（tail-mask 回归 / MTP-k1）在
   8B-LoRA / 4000 步 / LLaVA-Video 数据上未产生可检出的流式增益。
2. **不要重跑任何 v4_* 训练臂或其 streaming_eval**——结果已在案，重复无意义。
3. 唯一可选后续（低优先级，仅当有闲置节点）：mtp1 补 1 个种子（它是唯一四表全正的臂）。
4. 剩余算力全部转向 simplestream-plus 仓库（见其 EXECUTE_NOW.md，a1/b3/b4 优先）。
