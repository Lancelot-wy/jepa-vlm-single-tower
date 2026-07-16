# Cloud 执行手册：EXP-10 并行训练与评测

这份手册让新开发机上的 Cloud/Claude 直接执行当前实验，不需要 Blue Code、SSH key、
手动进 GPU Pod，也不能猜测或复用旧项目命令。

## 当前唯一正式流程

| 阶段 | 谁执行 | 固化实现 | 产物 / 成功条件 |
|---|---|---|---|
| 代码同步与提交 | 开发机上的 Cloud | `scripts/cluster/submit_exp10.sh` | vivolm job ID；脚本打印 run-id。 |
| 公共数据 gate | rank 0（4 卡） | `run_exp10_curated_4gpu.sh preflight/prep/smoke` | `source_audit.json`、clean manifest、两个 smoke `step_2/state.pt`。 |
| 四臂训练 | 4 个 Pod，各 4 L40S | `job_exp10_entry.sh` + `ONLY_ARM` | 每臂各有 `step_4000/state.pt`。 |
| 评测 | rank 0 的 4 卡 | `run_exp10_curated_4gpu.sh eval` | 四臂 MVBench/TempCompass JSON 与 `scorecard.json`。 |

资源请求固定为 **4 Worker × 4 L40S = 16 GPU**，每个 Pod 120 CPU、990Gi 内存。
四个臂不是 16 卡 DDP：每台只跑一臂的 4 卡 DDP，所以有效 batch、学习率、4,000
optimizer-update 步数均与原始配对设计相同。rank 0 先写共享 manifest，其他三台等
`gates_ready`，因此不会发生并发数据构建。

处理后路径、conversation 字段及 caption/grounding 语义见
[UNIFIED_VIDEO_DATA.md](UNIFIED_VIDEO_DATA.md)。Cloud 不得把 registry 改回旧版原始
metadata 或启用 basename 全盘索引。

## Cloud 只需执行的命令

在开发机执行，不能在调度到的 GPU Pod 内 `git pull` 或 `pip install`：

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
  echo "ERROR: existing non-Git path: $REPO" >&2
  exit 1
fi

git status -sb                  # 必须干净；submit 脚本也会强制检查
git rev-parse --short HEAD
bash scripts/cluster/submit_exp10.sh --dry-run
bash scripts/cluster/submit_exp10.sh
```

提交脚本会打印类似 `run_id=exp10-...`。保存这个值，然后用平台界面查看 job ID/排队状态，
并用共享盘做只读检查：

```bash
bash scripts/cluster/inspect_exp10_run.sh <run-id>
```

评测已在同一个 job 内集成，**不要**再为本次 EXP-10 调用旧的
`submit_mcq_eval.sh`、`submit_qa_eval.sh` 或 `submit_probes.sh`。它们面向历史 r2/r3
实验，路径、arm 名称和协议均不匹配。

## 代码与产物位置

```text
/data/vjuicefs_sz_ocr_wl/public_data/11193960/
├── jepa-vlm-single-tower/                 # 当前 public Git checkout
├── envs/jepa311/                           # 共享 Python/CUDA 环境；Pod 内自动激活
├── models/Qwen3-VL-2B-Instruct/            # 本地权重；禁止在线下载
├── jepa_data/exp10_curated/                # 共享审计与 frozen manifest
│   ├── source_audit.json
│   ├── qa_train_clean.jsonl
│   └── qa_train.jsonl.report.json
└── runs/exp10_curated/<run-id>/
    ├── outputs/<arm>/step_4000/state.pt
    ├── results/<arm>_{mvbench,tempcompass}.json
    ├── results/scorecard.json
    ├── coord/<attempt-id>/{gates_ready,failed_rank_*,completed}
    └── logs/<attempt-id>/rank{0,1,2,3}.log
```

## 历史项目方法：仅供识别，不能拿来执行当前实验

| 历史入口 | 原本用途 | 对 EXP-10 的处理 |
|---|---|---|
| `job.yaml` + `job_entry.sh` | 多节点 Phase-A / LLaVA 单臂训练 | 禁用；其默认 `Worker.num=2` 和配置不属于 EXP-10。 |
| `scripts/cluster/submit_batch.sh` | 旧 `cl_*`、EXP-09 / V4 批量消融 | 禁用；它按旧 config 生成 job。 |
| `scripts/direct/run_exp09_llavaonly_4gpu.sh` | LLaVA-only EXP-09 | 禁用；EXP-09 已暂停。 |
| `submit_mcq_eval.sh` / `submit_qa_eval.sh` / `submit_probes.sh` | r2/r3 历史评测 | 禁用；EXP-10 评测已在新的平台 job 中自动完成。 |
| `scripts/direct/run_exp10_curated_4gpu.sh` | 当前数据 gate、单臂训练、评测核心 | 只能由 `job_exp10_entry.sh` 编排，或在单一 4 卡机器上按 `CURATED_EXP10.md` 直接运行。 |

## 常见问题与正确处置

| 现象 / 证据 | 原因 | Cloud 应做什么 |
|---|---|---|
| `expected exactly 4 visible GPUs` | Pod 规格不对或调度未给满卡 | 不改 batch；检查 `job_exp10.yaml` 为 `num: 4`、每 Worker `gpu: "4"`，重新提交。 |
| `platform reported NNODES != 4` | job YAML 被改成了不一致的 Worker 数 | 不要尝试多节点 torchrun；恢复正式 YAML，重新提交。 |
| 源审计 `ready=false` 或 `missing_video` 高 | 某挂载/metadata/视频解析路径不可用 | 报告 `source_audit.json` 和对应路径；不能降低 `min_samples` 或移除审计。 |
| clean manifest 小于 gate | 本地可解析样本不足或污染过滤过多 | 保存 `qa_train.jsonl.report.json`、污染检查输出；不要修改阈值后强开。 |
| 三台机器长期等 `gates_ready` | rank 0 在准备中，或 rank 0 已失败 | 先读 rank 0 log；`inspect_exp10_run.sh` 有 `failed_rank_0` 时停止并报告。 |
| 某 arm 有 partial checkpoint | 抢占、节点异常或训练故障 | 先检查 log；只有 `state.pt` 写明 `step_unit: optimizer_update` 且非 NaN/数据失败时，使用 `--resume --run-id <id>`。 |
| `legacy micro-batch checkpoint` | 旧版计步 checkpoint 不能和当前 4,000 update 协议混用 | 不恢复；新建 run-id 重新跑。 |
| `HF_HUB_OFFLINE` / 权重下载报错 | GPU Pod 默认离线 | 不在 Pod 下载或 pip install；确认共享模型和 `envs/jepa311` 存在。 |
| CUDA OOM / NCCL 异常 | Pod 未按 4 卡独立运行或环境冲突 | 保存完整 rank log；不要擅自改 batch、grad accumulation、LR 或卡数。 |
| `mtp_persistence_ratio >= 1`、`target_std→0`、`adj_cos→1` | MTP 机制可能退化 | 让训练和评测完成，回传日志指标；不能因为 loss 下降就宣称有效。 |
| 没有 `scorecard.json` | 四臂尚未齐、某臂失败，或评测报错 | 先查 `inspect_exp10_run.sh`、各 rank log；不要运行历史评测脚本补结果。 |

## Cloud 回传格式与边界

回传必须包含：Git revision、平台 job ID、run-id、排队/开始时间、审计摘要、clean manifest
行数、两份 smoke checkpoint、四臂的最近 checkpoint、四个评测 JSON 的 `total/skipped`，以及
`scorecard.json`。若失败，回传 first error 前后日志、`failed_rank_*` 与上述状态脚本输出。

Cloud 不得修改数据源白名单、去污染 gate、`max_steps`、loss 权重、每臂 GPU 数、学习率，
也不得从历史 EXP-09/NExT-QA 文档拼接命令。需要改实验设计时先停在报告阶段。
