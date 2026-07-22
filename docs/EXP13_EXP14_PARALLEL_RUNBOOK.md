# EXP-13 / EXP-14：24 Worker 并行执行手册

这轮同时解决两个问题：先把 raw Qwen baseline 恢复到可解释的评测口径，再判断
Observation Query 为什么没有超过纯 CE。三项作业合计使用 23 台 Worker，保留 1 台作为
排队/故障余量。这里的 Worker 是公司队列资源单位，每台固定 4×L40S。

## 1. 冻结实验矩阵与资源

| 作业 | Worker / GPU | 内容 | 是否训练 |
|---|---:|---|---|
| EXP-13 matched-32 | 7 / 28 | 现有 8 协议×2 task 的 raw/A4 锚定 | 否 |
| EXP-13 official-budget | 4 / 16 | raw/A4 × full/32-frame cap，MVBench 2fps | 否 |
| EXP-14 state diagnostics | 12 / 48 | 6 臂，每臂 2 Worker/8 GPU | 是 |
| 预留 | 1 / 4 | 避免 24/24 同时占满导致调度僵持 | — |

EXP-14 每个训练 world 的 per-device batch=1、gradient accumulation=4，因此有效 batch
固定为 `8×1×4=32`，与 EXP-12 一致。六臂都使用相同 K=64、32 张真实帧、4fps、16 个
temporal units、冻结 ViT/merger、相同数据 manifest、800 optimizer updates：

| 臂 | seed | state 模式 | anti-copy |
|---|---:|---|---:|
| b0_ce_seed1 | 1 | none（纯 CE） | 0 |
| b1_query_seed1 | 1 | query | 0 |
| b2_noquery_seed0 | 0 | no_query | 0 |
| b3_noquery_seed1 | 1 | no_query | 0 |
| b4_query_beatcopy_seed0 | 0 | query | 1.0 |
| b5_query_beatcopy_seed1 | 1 | query | 1.0 |

seed-0 的纯 CE/Query 不重训，复用 EXP-12 A4/A5。预测臂总 loss 为：

```text
L = L_CE + 0.05 × (L_centered_cosine + w_copy × L_beat_copy)
```

所以 anti-copy 的 `w_copy=1.0` 是在 state objective 内与 centered cosine 等权；若误设
0.05，总 loss 中只剩 0.0025 倍，几乎测不到作用。

## 2. 服务器拉取

只在开发机/共享代码目录拉取；GPU Pod 内入口脚本会强制检查固定 commit 和 clean
worktree，Pod 启动后禁止再 `git pull`。

```bash
export BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
export PROJECT_ROOT=$BASE/jepa-vlm-single-tower

cd "$PROJECT_ROOT"
git fetch origin
git switch exp12-orca-token-sweep
git pull --ff-only origin exp12-orca-token-sweep
git status --short
git log -1 --oneline
```

`git status --short` 必须为空。提交脚本会把当前完整 commit 写进 YAML 和结果目录，队列中
若代码被换掉会立即失败，不会静默混跑。

## 3. 先检查最终队列 YAML

```bash
cd "$PROJECT_ROOT"
bash scripts/exp14/03_submit.sh --dry-run
bash scripts/exp13/03_submit_official.sh --dry-run
bash scripts/exp12/20_submit_native_anchor.sh --dry-run
```

必须分别看到 12、4、7 Workers；三者合计 23 Workers / 92 GPUs。不要把 Worker 数误填成
GPU 数，也不要把 EXP-14 合成一个 48-GPU world。

## 4. 并行提交

```bash
cd "$PROJECT_ROOT"

# 12 Worker：六组 K64 机制训练；内置真实模型 DDP save/resume/eval smoke。
bash scripts/exp14/03_submit.sh

# 4 Worker：raw Qwen 和 A4 的官方预算/32帧预算评测。
bash scripts/exp13/03_submit_official.sh

# 7 Worker：若之前的 matched-32 EXP-13 尚未完整完成，再提交；已完成则不要重复占卡。
bash scripts/exp12/20_submit_native_anchor.sh
```

每条命令都会打印 `run=<RUN_ID>`；立即复制保存。三条命令彼此独立，一条排队/失败不会
污染另外两条的结果目录。

## 5. checkpoint、续训与自动释放

EXP-14 每臂在 optimizer step 400、800 原子保存 `state.pt`、optimizer/scheduler、RNG、
数据游标和 `checkpoint_meta.json`。训练若被抢占，用原 RUN_ID 重提：

```bash
bash scripts/exp14/05_resume.sh <EXP14_RUN_ID>
```

脚本只接受完整、字节数匹配且 `step_unit=optimizer_update` 的最新 checkpoint；会裁掉
checkpoint 之后的孤立日志，再从相同步数继续。若已有日志却没有有效 checkpoint，会拒绝
覆盖，必须先查明存储问题。

训练完成后，每个 arm leader 自动评测 checkpoint-400 的固定 TempCompass 子集，以及
checkpoint-800 的完整 MVBench/TempCompass。所有结果齐全后 rank0 汇总；入口命令退出，
`restartPolicy: Never` 使机器自动释放。官方预算和 matched-32 作业同样在汇总后退出。

## 6. 状态检查

```bash
bash scripts/exp14/04_status.sh <EXP14_RUN_ID>
bash scripts/exp13/04_status_official.sh <EXP13_OFFICIAL_RUN_ID>

# matched-32 任务
find "$BASE/runs/exp13/<EXP13_RUN_ID>/coord" -type f -maxdepth 2 -print | sort
tail -n 100 "$BASE/runs/exp13/<EXP13_RUN_ID>/logs"/*/rank0.log
```

EXP-14 最终产物：

```text
$BASE/runs/exp14/<RUN_ID>/results/exp14_state_diagnostics/comparison.{json,csv,md}
```

官方预算最终产物：

```text
$BASE/runs/exp13-official/<RUN_ID>/official_budget_comparison.{json,md}
```

若任一 rank 报错，先看同 attempt 下 `failed_rank_*` 和该 rank 日志。不要修改 batch、帧数、
loss 权重来绕过 OOM/加载错误；修代码后提交新 commit，再使用同 RUN_ID 的 resume（训练）
或同结果目录重提（评测，完整 shard 会被复用）。

## 7. official-budget 的边界

该路径从 `视频` 字段读取真实 mp4，按 2fps 均匀采样，最多 2048 帧；使用 224,000 总
video-token 上限、每 temporal unit 640 token 上限和技术报告的 MVBench prompt。四行是：

- raw Qwen，full budget；
- A4 checkpoint，full budget；
- raw Qwen，同一 2fps 规则但 cap 32 frames；
- A4 checkpoint，同一规则但 cap 32 frames。

它使用本仓库的 torchvision-free native-compatible preprocessing、HF greedy generation 和
FlashAttention-2，标签始终写作 **official-budget reproduction**，不能冒充 Qwen 私有官方
harness。raw/full 会与公开 61.7 形成诊断带：≤2.5pp 为 green，2.5–5pp 需继续核验，>5pp
或 coverage<99.5% 为 red。只有 raw/full 回到合理区间，才能解释 A4−raw 的训练净效应。

## 8. EXP-14 判读

先看机制，再看 benchmark：

1. predictive arm 要求 `centered_margin>0.10` 且 `persistence_ratio<0.90`；否则仍是复制解；
2. 同 seed 相对纯 CE 的 MVBench、TempCompass 都不得下降超过 1pp；
3. Query、no-query、anti-copy 都做 seed0/seed1 配对和逐题 McNemar/bootstrap；
4. 只有机制门和 benchmark protection 同时通过才记为 candidate；
5. 汇总器永远不会自动提交 Event 实验，Event 是否值得做由结果人工决定。

如果 no-query 与 Query 一样差，问题更可能在 state target/objective，而非 learnable query；
如果 anti-copy 显著把 persistence 拉到 1 以下但 benchmark 不涨，说明“学会不复制”本身
还不足以改善 VQA；只有 anti-copy 同时改善机制和两项 benchmark，才值得扩大步数/数据。
