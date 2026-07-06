# 在 vivolm 平台多机多卡跑 jepa-vlm-single-tower

本仓库刻意保持"裸 torch + accelerate DDP"（见 README §6），因此上公司集群
**只换启动器，不改模型代码**：`accelerate launch` → 平台 job + `torchrun`。
`jepa_vlm.train` 里的 `Accelerator()` 直接读 torchrun 注入的
`RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT`，天然兼容。

部署根（共享盘 juicefs，三节点可见）：
```
BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
  ├── envs/jepa311/                  共享 conda env（复用，见下）
  ├── models/Qwen3-VL-2B-Instruct/   模型权重（本地，免下载）
  ├── jepa-vlm-single-tower/         本仓库
  └── outputs/                       训练产物（自动创建）
```

## 1. 依赖（无需 pip / 无需构建镜像）

直接复用已建好的共享环境 `envs/jepa311`（sibling 项目 `vlm_jepa_lf_three_machine`
建的），已含本仓库 `requirements.txt` 全部依赖：

| 包 | jepa311 版本 | requirements 要求 |
|---|---|---|
| python | 3.11 | >=3.10 ✓ |
| torch | 2.5.1+cu124 | >=2.4 ✓ |
| transformers | 5.6.0 | >=4.57（含 Qwen3-VL）✓ |
| accelerate | 1.11.0 | >=1.0 ✓ |
| av (PyAV) | 16.0.0 | >=12 ✓ |
| peft | 0.18.1 | >=0.14（lora 用）✓ |
| pyyaml / numpy | 6.0.3 / 2.4.6 | ✓ |

`env.cluster.sh` 会把 `envs/jepa311/bin` 加进 PATH 并 `unset LD_LIBRARY_PATH`
（容器自带的指向旧 torch1.13，必须清掉）。

**唯二缺口**（都可选，不影响 Phase A 主训练）：
- `flash_attn` 未装 → 配置里保持 `attn_implementation: sdpa`（正确且够用）。
  若要 flash-attention-2，需在开发机联网装进 env 后再改配置。
- `opencv-python-headless` 未装 → 只影响 `scripts/compute_flow.py`（静态片段过滤）。
  不需要 flow 过滤就把配置 `train.min_flow: 0.0` 并用 `train.jsonl`（而非 `train_flow.jsonl`）。

## 2. 数据集路径（需你填 / 准备）

**SSv2 目前不在任何挂载盘上**（已全盘搜索）。本地已有的视频数据集是
`/data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K` 和 `Tarsier2-Recap-585K`。
SSv2 需 Qualcomm 注册下载（节点离线，只能在开发机经代理下），下好后：

```bash
source scripts/cluster/env.cluster.sh
OUT=/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data/ssv2
# 1) 标注 → manifest
python scripts/prepare_ssv2.py --anno-dir <SSV2>/anno --video-dir <SSV2>/videos --out-dir $OUT
# 2)（可选）光流字段，供静态片段过滤（需先装 opencv）
python scripts/compute_flow.py --manifest $OUT/train.jsonl --data-root <SSV2>/videos \
    --out $OUT/train_flow.jsonl --workers 16
```
然后把 `configs/vivolm_ssv2.yaml` 里三个 `TODO(fill)` 路径对齐到上面的 `$OUT`
（`train_manifest` / `val_manifest` / `data_root`）。manifest 只有 `video` 必填，
所以任意视频集都能做 Phase A 自监督训练（linear probe 评估才需要 `label`）。

## 3. 启动（平台 job，多机自动 rendezvous）

不靠手动 ssh/IP：提交 `job.yaml`，平台调度 N 个 Worker pod（每个 4 卡 L40s），给每个
pod 注入 `TF_CONFIG`，同时跑同一条命令；`env.cluster.sh` 从 `TF_CONFIG` 推出
`MASTER_ADDR/NNODES/NODE_RANK`，`torchrun` 在 worker[0]:29500 汇合。

```bash
# 开发机提交（默认是 DEBUG pod：建好环境后 sleep，可 exec 进去手动验证）
/data/vtraining_04/code/vtraining/cli/vtraining run -f job.yaml
```
把 `job.yaml` 的 `run.command` 从默认的 `JOB_SLEEP=1 ...` 换成注释里的
FULL 版本即开始正式训练。`Worker.num` 控制机器数。

**先验证再烧卡**（推荐）：用默认 DEBUG pod exec 进去手动跑，确认 rendezvous + 前向：
```bash
source scripts/cluster/env.cluster.sh
bash scripts/cluster/train_multinode.sh          # 用 configs/vivolm_ssv2.yaml
```

### 有效 batch 与缩放
`eff_batch = batch_size(4) × grad_accum(4) × (num×4 GPU)`。3 机 = 192（≥方案目标 128）。
改机器数时按线性缩放 `train.lr/lr_backbone` 并等比调 `train.max_steps`。启动时可临时覆盖：
```bash
GRAD_ACCUM=2 EXTRA_OVERRIDES='train.max_steps=8000 model.mask_ratio=0.75' \
  bash scripts/cluster/train_multinode.sh
```

## 4. 评估（训练后，rank0 单卡即可）
见 README §4 三件套（`extract_features` / `linear_probe` / `nontrivial_check`），
`--config` 指向 `outputs/jepa_phase_a_v21/config.json`，用同一个 env311 python 跑。

## 5. 新增/改动文件一览
- `configs/vivolm_ssv2.yaml`      —— 集群配置（本地模型路径、sdpa、SSv2 路径占位、L40s batch）
- `scripts/cluster/env.cluster.sh` —— env311 激活 + TF_CONFIG→torchrun rendezvous
- `scripts/cluster/train_multinode.sh` —— `torchrun -m jepa_vlm.train`
- `scripts/cluster/job_entry.sh`   —— 每 pod 入口（含 JOB_SLEEP 调试模式）
- `job.yaml`                       —— vivolm 平台作业
（模型/训练代码零改动。）
