# EXP-10：在 vivolm 队列提交（唯一正式入口）

这是当前主线 EXP-10 的平台交接手册。它替代历史 `job.yaml`、
`scripts/cluster/submit_batch.sh` 和 `EXECUTE_NOW.md` 中的训练入口；那些文件是旧
Phase-A / EXP-09 记录，**不能**用于当前实验。

## 资源评估与作业形态

| 工作 | vivolm 资源 | 原因 |
|---|---:|---|
| 数据审计、manifest 构建、两臂 smoke | rank 0 的 **1 Worker × 4 L40S** | 这是共享数据的唯一写入者，避免四台机器并发修改 manifest。 |
| 四个训练臂 | **4 Workers × 每台 4 L40S（共 16 卡）** | 每台机器独立 4 卡 DDP 跑一个 arm；有效 batch、LR、步数均不变。 |
| 四臂最终 MVBench + TempCompass 评测 | rank 0 的同一 4 L40S | 四臂 checkpoint 到齐后，四张卡各评一个 arm。 |

所以当前最快且不改变实验设计的配置是四台 4×L40S Pod（总共 16 卡、480 CPU、3960Gi
内存请求）。作业里复用了已验证的 `VideoFoundationModel1b-wl01` business 和 L40S
镜像。它不会把四臂拼成 16 卡 DDP；每臂保持原来的 4 卡条件，因而配对结论仍可比。

平台挂载已固定为以下三项，缺任意一项会在 GPU 训练前失败：

- `sz_ocr`：共享环境、Qwen3-VL 权重、输出、评测文件，以及 Vript/InternVid 视频；
- `ai_ocr`：LLaVA-Video-178K 媒体与标注；
- `ai_gpt_vision_wl04`：Vript 和 OpenVid 元数据。

## 新开发机：无 SSH 密钥的拉取与提交

仓库已公开，使用 HTTPS 不需要 GitHub SSH key。以下命令只在**开发机**执行；GPU Pod
不需要也不应在运行时 `git pull`（它可能无外网）。

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
  echo "ERROR: $REPO exists but is not a Git checkout; choose another empty path." >&2
  exit 1
fi

git status -sb
git rev-parse --short HEAD
bash scripts/cluster/submit_exp10.sh --dry-run
bash scripts/cluster/submit_exp10.sh
```

最后一条提交 `job_exp10.yaml`。rank 0 按固定顺序执行
`preflight → audit → prep → smoke`，随后四个 Pod 各训练一个 arm，最后 rank 0 评测。
每个 submission 使用独立的 run ID 和输出根目录；任一数据源不可解析、去污染后样本
不足、CE/MTP smoke 未生成 checkpoint，都会直接失败，绝不会静默进入正式训练。

不要用 `nohup`、`tmux` 或 Blue Code 保活：`vtraining` 作业本身管理 Pod 生命周期，日志会
持久写入：

```bash
/data/vjuicefs_sz_ocr_wl/public_data/11193960/runs/exp10_curated/<run-id>/logs/<attempt-id>/
```

## 预占或中断后的恢复

默认提交是新跑（`RESUME=0`）。若平台中断，先查看平台日志和上述共享日志；仅在确认
`state.pt` 带有 `step_unit: optimizer_update`、manifest 未变化且失败不是数据/NaN 问题时，才可：

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
bash scripts/cluster/submit_exp10.sh --resume --run-id <the-existing-run-id>
```

恢复仍会跑 preflight、审计、manifest gate 和 smoke；它不会跳过关键检查。旧版
micro-batch 计数 checkpoint 会被启动器拒绝，不能混入本次 EXP-10。

## Cloud 执行边界

Cloud 只需要执行上一节的开发机命令，并把以下信息回传：平台 job ID、Git revision、
`source_audit.json` 摘要、`qa_train_clean.jsonl` 行数、两份 smoke `step_2/state.pt` 的
存在性、每臂最后的 `log.jsonl` 指标及
`runs/exp10_curated/<run-id>/results/scorecard.json`。Cloud 不得更改源清单、卡数、
`max_steps`、loss 权重或跳过 gate；遇到失败先保留日志并报告。

完整的 Cloud 执行、历史入口识别、只读状态检查与故障处理见
[CLOUD_EXP10_RUNBOOK.md](CLOUD_EXP10_RUNBOOK.md)。
