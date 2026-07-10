# V4 两天批量验证（20 × 单节点 4×L40S）

2026-07-10。在 Round-3 基础上回答两个判决性问题；每臂一台单节点、全部并行、判死线前置。

1. **S2**：dual-view（clean CE + masked latent reg）能否超过纯 CE 对照？（Round-3 单视图 50% 只有 +1.03pp，其记录自身指认的下一步就是 dual-view / 低 mask ratio）
2. **S1**：帧间预测辅助 loss（clean 输入 + MTP-k 下一帧 latent 预测）能否超过纯 CE 对照？

裁判从自建 temporal QA 换成 **流式 benchmark 官方式协议**（OVO-Bench Backward EPM/ASI + StreamingBench，按题目时间戳截断，likelihood 打分）；附带 L1 门控实验（免训练腿，算力轴）。

## 0. 一次性准备（开发机，10 分钟）

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower && git pull
# 两个 benchmark 已下载 —— 把实际路径填进下面三个变量（run_streaming_eval.sh 用）:
#   OVO_ROOT      : 含 ovo_bench_new.json 和 chunked_videos/<id>.mp4 的目录
#   SB_CSV        : StreamingBench 任务 csv（可逗号并列多个）
#   SB_VIDEO_ROOT : 对应 csv 的视频目录（内含 sample_N/video.mp4）
```

数据/环境/模型均沿用 Round-3（jepa311 env、Qwen3-VL-2B、qa_train_flow.jsonl），零新依赖。

## 1. 提交训练臂（11 台，Day-1 上午一次性全部提交）

```bash
NODES=1 EXTRA_OVERRIDES='train.min_flow=8.42' scripts/cluster/submit_batch.sh \
  v4_ctrl_s0 v4_ctrl_s1 \
  v4_dv25_s0 v4_dv25_s1 v4_dv50 v4_dv25_lam05 v4_sv25 \
  v4_mtp1 v4_mtp4 v4_dv25_mtp1 v4_dvce25
```

| 臂 | 组 | 回答什么 |
|---|---|---|
| v4_ctrl_s0 / s1 | 对照 | 机制基准（一切 claim 只认与它的配对差） |
| v4_dv25_s0 / s1 | S2 主线 | dual-view 25% λ0.2，2 seeds |
| v4_dv50 | S2 消融 | 与 r3_joint 对齐 ratio，分离 dual-view 效应 |
| v4_dv25_lam05 | S2 消融 | λ 敏感性 |
| v4_sv25 | 消融 | 单视图 25%：r3 弱结果是 mask 太强还是 CE 被污染 |
| v4_mtp1 | S1 主线 | clean CE + 0.2·MTP(k=1)（帧间预测） |
| v4_mtp4 | S1 消融 | 多步预测 k=4 |
| v4_dv25_mtp1 | 组合 | S1+S2 叠加 |
| v4_dvce25 | 关键消融 | 双 CE 无预测项——若 ≈ dv25，则有效成分是正则不是预测 |

单节点 eff.batch = 4×4×4=64（r3 的一半）：**保持 lr 不变、把步数放到 max_steps=4000 不变**
（样本量减半但两天窗口优先可比性——所有臂同一预算，配对比较不受影响）。
dual-view 臂显存更高：OOM 时对该臂 `EXTRA_OVERRIDES='train.batch_size=2 train.grad_accum=8'`。

## 2. 训练期间同时提交（Day-1，不等训练）

- **E-base（1 台）**：base 模型（不训练）跑 OVO/SB 两模式 → recent-window 基线 + prefix 参照。
  ```bash
  # 在单节点 DEBUG pod 内（JOB_SLEEP=1 提交后 exec 进入）:
  source scripts/cluster/env.cluster.sh
  OVO_ROOT=... SB_CSV=... SB_VIDEO_ROOT=... ARMS="base" bash scripts/cluster/run_streaming_eval.sh
  ```
- **E-gate（1 台）**：L1 门控（VLM 零训练）：
  ```bash
  python -m jepa_vlm.probes.gating_eval --config configs/v4_ctrl_s0.yaml \
    --bench sb --data $SB_CSV --video-root $SB_VIDEO_ROOT \
    --train-videos 20 --max-items 200 --budgets 0.1,0.25,0.5 \
    --out $BASE/outputs/gating/sb_rtvu.jsonl
  ```
  判死线（预注册）：同预算下 surprise 不敌 framediff/periodic → 门控叙事关闭。
- **E-mcq（1 台，可选）**：沿用 run_mcq_eval.sh 给各 ckpt 补 MVBench/TempCompass（与 Round-3 可比）。

## 3. 训练完成后（Day-2）

```bash
# 每台训练节点跑完自动落 ckpt 到 outputs/<arm>/step_4000。评测（2-3 台并行消化）:
ARMS="v4_ctrl_s0 v4_ctrl_s1 v4_dv25_s0 v4_dv25_s1 v4_dv50 v4_dv25_lam05 v4_sv25 v4_mtp1 v4_mtp4 v4_dv25_mtp1 v4_dvce25" \
  OVO_ROOT=... SB_CSV=... SB_VIDEO_ROOT=... bash scripts/cluster/run_streaming_eval.sh
# 汇总（准确率表 + 相对 v4_ctrl_s0 的配对翻转 + 符号检验）:
python scripts/summarize_streaming.py $BASE/outputs/streaming_eval
```

## 4. 判读（预注册，不许软化）

- **S1/S2 任一成立**：该组主线臂相对 ctrl 在 OVO(EPM/ASI, recent) 上配对净增益为正、
  2 seeds 同向、符号检验 p<0.1（353 题级样本；p<0.05 需后续扩样确认）。
- **v4_dvce25 ≈ v4_dv25**：增益来自一致性正则而非预测——结论定性必须如实改写。
- **全部为负**：负结果成立——"预测目标塑形表征"在该规模/该数据上无效，写清归因，
  剩余算力转向 L1 门控轴或止损复盘。
- 每个数字可独立复算：全部逐题 jsonl 落在 `outputs/streaming_eval/`。

## 5. 节点账本（20 台）

| 用途 | 台数 |
|---|---|
| 训练臂 | 11 |
| E-base / E-gate / E-mcq | 3 |
| Day-2 评测消化 | 3 |
| 机动（OOM 重跑 / 补 seed / 补 step_2000 评测） | 3 |

## 6. 本轮代码增量（相对 Round-3）

- `config.py`：`model.dual_view (off|reg|ce)`、`model.reg_enabled`（含合法性校验）
- `modeling/model.py`：`_forward_dual_view`（clean-CE + masked-reg 双视图；ce 模式为双 CE 一致性对照）、`reg_enabled` 关断
- `probes/streaming_eval.py`：OVO/StreamingBench 流式截断 MCQ 评测（likelihood 打分、断点续跑）
- `probes/gating_eval.py`：L1 门控（tiny GRU surprise vs framediff/periodic/always，预算扫描）
- `scripts/summarize_streaming.py`：准确率表 + 配对翻转 + 符号检验
- `scripts/cluster/run_streaming_eval.sh`：单节点 4 GPU 并行评测 worker
- `configs/v4_*.yaml`：11 个实验臂
- `submit_batch.sh`：`NODES` 可由环境变量覆盖（本轮全部 NODES=1）
