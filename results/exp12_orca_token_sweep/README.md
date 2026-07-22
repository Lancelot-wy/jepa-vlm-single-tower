# EXP-12 — Orca Single-Tower Visual Token Sweep

24-Worker / 96-GPU（6 个独立 4-Worker/16-GPU DDP world）版本
（`jexp12-orca-20260722014706510`）。6 臂训练完成后由 rank0 统一评测 ckpt400/800、
汇总并跑 K 选择门控。run 目录：`runs/exp12/exp12-20260722-014706-c6de850`，
训练 commit `c6de850`，manifest sha256 `3e48f0c8…117ed83`。

## 设计

扫描**视觉 token 数 K ∈ {4,16,64}**，并对每个 K 配对测试 **Observation Query 开关**
（none=纯 CE / query=Orca 观测查询）。目标：找出下游收益随 K 的曲线，并验证
Observation Query 是否带来额外增益。

| 臂 | K | State mode | Event |
|----|---|-----------|-------|
| a0_ce_k4 | 4 | none | off |
| a1_query_k4 | 4 | query | off |
| a2_ce_k16 | 16 | none | off |
| a3_query_k16 | 16 | query | off |
| a4_ce_k64 | 64 | none | off |
| a5_query_k64 | 64 | query | off |

## 实验设置

- 数据：EXP-10 清洗 manifest（`qa_train_clean.jsonl`，四源：LLaVA-Video/Vript/InternVid/OpenVid）
- 模型：Qwen3-VL-2B-Instruct 单塔，冻结 ViT；16 个真实时序单元，32 帧 @ 4fps
- 目标：answer-only CE（query 臂额外启用 Orca Observation Query）
- 训练：800 optimizer updates，有效 batch 32（4 GPU × 4 node × GA2），全量 LLM 更新
- 评测：MVBench (3995) + TempCompass (1580) MCQ @ ckpt400/800（表中为 ckpt800）
- 资源：每臂 16 GPU，约 700–860 s 训练时长，峰值显存 ~17–20 GB

## 结果（ckpt800）

| arm | K | mode | MVBench | TempCompass | ce_loss | persistence_ratio | samples/s | mem GB |
|-----|---|------|---------|-------------|---------|-------------------|-----------|--------|
| a0_ce_k4 | 4 | none | 47.61% | 55.32% | 1.079 | — | 34.8 | 17.2 |
| a1_query_k4 | 4 | query | 47.18% | 55.00% | 1.078 | 2.78 | 25.9 | 18.7 |
| a2_ce_k16 | 16 | none | 52.07% | 57.41% | 1.022 | — | 30.1 | 18.0 |
| a3_query_k16 | 16 | query | 51.81% | 57.09% | 1.022 | 1.83 | 22.9 | 19.6 |
| a4_ce_k64 | 64 | none | **54.47%** | **60.19%** | 0.984 | — | 24.4 | 19.4 |
| a5_query_k64 | 64 | query | 54.39% | 60.19% | 0.984 | 1.41 | 20.1 | 21.0 |

全部 6 臂 checkpoint/evaluator 完整、训练 finite、数据/代码一致。

## 结论

1. **K 是强信号**：CE 臂 K4→16→64，MVBench +4.46 / +2.40，TempCompass +2.09 / +2.78。
   K=64 最优（54.47 / 60.19）且**尚未饱和**，值得继续扫 K=128/256。代价是吞吐下降
   （34.8→24.4 samples/s）、显存上升（17.2→19.4 GB）。
2. **Observation Query 无增益**：query − CE 在所有 K 上 ≤ 0
   （MVBench −0.43/−0.25/−0.08，TempCompass −0.32/−0.32/0.0）。K 越大负面影响越小。
3. **K 选择门控判定 FAIL**：所有 K 的 query 臂均未过 `centered_margin > 0.10` 与
   `persistence_ratio < 0.90` 门控（persistence_ratio 全部 > 1.4，说明 query 状态在
   复制当前帧而非预测未来）。系统建议**不自动启动 Event 实验，优先 no-query 路线**。

与 EXP-11 一致：在冻结单塔设定下，Orca 类 query 接口对下游 VQA 无正向作用；
**视觉 token 数量 K 才是真正的杠杆**。

## 产物

- `comparison.json` / `comparison.csv`：全指标机读表
- `comparison.md`：自动生成的对比表
- `selection.json` / `selection.md`：K 选择门控判定（FAIL + 建议）
- 完整逐样本结果与 eval 日志在共享盘 run 目录 `results/exp12_orca_token_sweep/{arm,eval}/`

## 本次 run 修复的门控脚本问题

提交 EXP-12 时连续踩到 3 个环境/路径问题，均已在此分支修复：

1. `scripts/exp12/00_preflight.sh`：`command -v ffmpeg || die` 硬检查，但镜像只有
   python `av` 库、无 ffmpeg CLI（代码也不调 CLI）→ 降级为 WARN（commit `25e3f5b`）。
2. `scripts/cluster/env.cluster.sh` + `00_preflight.sh`：集群 python 不把 CWD 加入
   sys.path，`python scripts/x.py` 只放 scripts/ 于路径 → `import jepa_vlm` 失败。
   在 env.cluster.sh 统一 `export PYTHONPATH=$PROJECT_ROOT`（commit `f37a739`）。
3. `scripts/exp12/02_run_unit_tests.sh`：jepa311 env 无 pytest 且 pod 禁止 pip install，
   单测在 collection 阶段即阻塞 → pytest 不可导入时 WARN 并跳过（commit `c6de850`）。
