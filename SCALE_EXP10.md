# EXP-10 128 卡快速执行：训练 → 评测 → 自动释放

这是 EXP-10 的高吞吐正式入口，适用于队列可以同时调度 **32 台 L40S
Worker** 的情况。它不是把四个配对实验混成一个世界组，而是把平台给出的
32 个 Worker 按固定顺序切成四个独立的 8 节点 DDP 组：

| 组 | arm | 资源 | 每次 optimizer update 的样本数 |
|---|---|---:|---:|
| 0 | `exp10_curated_sft_s0` | 8 Worker × 4 GPU = 32 GPU | 4 × 32 × 1 = 128 |
| 1 | `exp10_curated_mse_s0` | 8 Worker × 4 GPU = 32 GPU | 128 |
| 2 | `exp10_curated_sft_s1` | 8 Worker × 4 GPU = 32 GPU | 128 |
| 3 | `exp10_curated_mse_s1` | 8 Worker × 4 GPU = 32 GPU | 128 |

因此它保留了原方案的 per-GPU batch=4、有效 batch=128、学习率和 4,000 个
optimizer-update；只把每个 arm 的 DDP world 从 4 卡扩到 32 卡，并将累积从 8
降到 1。四臂仍完全并行。

## 生命周期与自动释放

`job_exp10_scale_entry.sh` 的固定顺序是：

1. 全局 rank 0 做 preflight、数据审计、冻结 manifest、去污染和两臂 4 卡 smoke；
2. 四个 32 卡组同时训练各自的 arm；
3. 全局 rank 0 等待四个 `step_4000/state.pt`，在同一任务内自动运行 MVBench 和
   TempCompass，并写出 `results/scorecard.json`；
4. 评测成功后写 `coord/<attempt>/completed`，所有 entrypoint 直接退出。

YAML 使用 `restartPolicy: Never`，没有 `sleep`、`nohup` 或保活循环。因此最后一步
退出即让 vivolm 回收全部 32 个 Worker（128 张 GPU）。任一 gate、训练或评测失败时
shell 非零退出并写 `failed_rank_*`，同样不会留下一个正常运行的空闲保活任务。

每个 arm 每 **250 个 optimizer update** 写一次完整 checkpoint（`step_250`、
`step_500` … `step_4000`）。checkpoint 先写入同目录临时文件、`fsync` 后原子替换为
`state.pt`；如果分配恰好在写入时被收回，恢复逻辑会跳过不完整文件，回退到最近一份
带 `step_unit=optimizer_update` 的 checkpoint。重提时使用同一 run-id：

```bash
bash scripts/cluster/submit_exp10_scale.sh --resume --run-id <run-id>
```

## 开发机的一次性命令

只在有共享 `/data` 和 `vtraining` 的开发机执行，**不要**进 GPU Pod 拉代码或运行
Blue Code。公开仓库走 HTTPS，不需要 SSH key。

```bash
set -euo pipefail
BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
REPO="$BASE/jepa-vlm-single-tower"
URL=https://github.com/Lancelot-wy/jepa-vlm-single-tower.git

if [[ -d "$REPO/.git" ]]; then
  cd "$REPO"
  git remote set-url origin "$URL"
  git fetch --prune origin main
  git switch main
  git pull --ff-only origin main
elif [[ ! -e "$REPO" ]]; then
  git clone --branch main --depth 1 "$URL" "$REPO"
  cd "$REPO"
else
  echo "ERROR: $REPO exists but is not a Git checkout" >&2
  exit 1
fi

git status -sb
git rev-parse --short HEAD
bash scripts/cluster/submit_exp10_scale.sh --dry-run
bash scripts/cluster/submit_exp10_scale.sh
```

提交输出中的 `run_id=exp10-scale-...` 是后续检查与恢复的唯一标识。只读查看进度：

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
bash scripts/cluster/inspect_exp10_run.sh <run-id>
```

如果被抢占且日志证明不是数据、NaN 或 NCCL 故障，且 checkpoint 中的 `step_unit` 为
`optimizer_update`，可用同一个 run-id 恢复：

```bash
bash scripts/cluster/submit_exp10_scale.sh --resume --run-id <run-id>
```

恢复也会重新执行 preflight、审计、manifest gate 和 smoke；不能跳过这些检查。

## 时间预期与首批日志判定

128 卡方案的目标是将原来每臂 15–28 小时缩短到约 4–8 小时量级，但实际时间受
32 GPU DDP 通信和 JuiceFS 视频解码吞吐影响，不能在未运行前保证。为避免 128 GPU
同时拉起 1,024 个 PyAV worker，作业默认 `train.num_workers=2`（全局 256 个 reader）。

训练日志每 20 update 输出 `sec_per_step`。在各 arm 的首个 `step=20` 后，用
`sec_per_step × (4000 - 20)` 重算 ETA：若任何 arm 明显超出 8 小时窗口，应先检查
GPU 利用率、数据加载等待和 NCCL，而不要擅自修改 batch、学习率或去污染 gate。

## 排队不满足 32 Worker 时

不要把 `job_exp10_scale.yaml` 的 `Worker.num` 随意改小：入口会故意拒绝不是
`4 × EXP10_NODES_PER_ARM` 的拓扑，避免统计条件漂移。若队列无法一次给出 32 个
Worker，使用保守回退入口，它仍会自动评测并在结束后释放 16 卡：

```bash
bash scripts/cluster/submit_exp10.sh --dry-run
bash scripts/cluster/submit_exp10.sh
```
