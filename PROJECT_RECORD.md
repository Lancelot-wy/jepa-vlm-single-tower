# 流式视频理解项目 · 工作总记录（Agent 交接文档）

最后更新：2026-07-13。本文档是项目唯一的权威总账，供任何后续 agent/会话直接接手。
剔除了所有无关信息（其他项目、基建波折、中间讨论）。凡与本文冲突的旧记录以本文为准。

---

## 1. 项目定位（读这一段就够开始干活）

**做什么**：流式视频理解（streaming video understanding）的改进方法——模型边看视频流边答题，
未来不可见。目标是在官方 benchmark（OVO-Bench / StreamingBench）官方协议下可对表的改进。

**当前主战场**：`SimpleStream++`——在 SOTA 基线 SimpleStream（纯滑窗，arXiv 2604.02317）之上做
**按问题路由的条件式历史增强**（training-free，纯推理策略）。核心依据：SimpleStream 论文自己的
Table-4 消融证明无条件加历史会失败（记忆 +6.6 / 感知 −4.9 / HLD −12.4），病因=无条件；
我们只给"需要记忆"的问题附加历史，感知类问题输入与基线逐字节相同。

**训练线（V4）已按预注册标准判负关闭**（见 §4 EXP-05），不要重开。

**与两个相邻项目的边界（防混淆）**：
- `vlm_jepa_lf_three_machine` / jepa-vlm 仓库的 R1–R3：那是**表征学习/VLM 训练加速**项目，
  与本项目无关；我们只借了它的集群提交框架和训练管线代码。
- EMNLP rebuttal（dreo_d4x）：完全无关的另一项目，共用 .31 服务器而已。

---

## 2. 资产与访问

### GitHub（跨 agent 协作的主通道，集群侧会话也读写这里）
| 仓库 | 用途 | 关键内容 |
|---|---|---|
| `Lancelot-wy/simplestream-plus`（私有） | **主战场** | `lib/policy_plus.py`（路由+历史通道）、`main_experiments/eval_ovo_plus.py`、`EXECUTE_NOW.md`（服务器执行手册）、`PLUS_VALIDATION.md`（判读标准）、`scripts/submit_plus.sh`（平台提交）、`scripts/push_results.sh`（结果回传到 results-cluster 分支） |
| `Lancelot-wy/jepa-vlm-single-tower` | 训练线（已关闭）+ 实验总账 | `REGISTRY.md`（EXP 编号总账）、`V4_VERDICT.md`（判负记录）、`results-cluster` 分支（V4 全部逐题结果）、训练/评测管线（可复用） |

### 公司集群（vivolm 平台，20×4×L40S；本地 Mac 无法直连，经 GitHub 交换）
- 部署根：`/data/vjuicefs_sz_ocr_wl/public_data/11193960`（下称 `$BASE`）
- 模型：`$BASE/stream/Qwen3-VL-8B-Instruct`（已就位）；Qwen2.5-VL-7B、CLIP-B/32 待下（见 EXECUTE_NOW 第 0 步）
- Benchmark：`$BASE/stream/ovo`（含 src_videos→REC/SSR/CRR 可用）、`$BASE/stream/streamingbench`
- 环境：训练用 `$BASE/envs/jepa311`；SimpleStream++ 需建 `$BASE/envs/plus311`（官方锁版本栈）
- 提交：`vtraining run -f job.yaml`；**环境无 flash-attn，一切推理传 `ATTN_IMPL=sdpa`（脚本已默认）**

### 实验室服务器（10.50.0.31，wanghaoyu，探索期资产，现为归档）
- `/data-store/wanghaoyu/`：OVO+StreamingBench 全量、5 个模型权重、`explore/`（四轮探索全部代码/报告/核验）
- 关键文档：`explore/ROUTE_REPORT_round1.md`、`explore/V2_DESIGN.md`、`explore/V4_EXPERIMENT_PLAN.md`（历史版）

---

## 3. 已证伪清单（任何 agent 不得重试，全部有实验记录）

| 已证伪的方法 | 证据来源 |
|---|---|
| 冻结 V-JEPA2 predictor error 作事件/写入/读取信号 | 探索 Round-1：≈镜头切换检测器，QA 对齐≤随机 |
| 零训练选择判据刷准确率（span-residual/冗余/相似度） | v0+Pareto：353 题与相似度去重打平，p≈0.6 |
| prediction-error 选读取历史 | E3：query-blind，输给 CLIP query 检索 |
| 单视频在线梯度适应小 predictor | E5：18 组合 16 个输 copy-last |
| score 阈值 gate 选样本 | intern 线：held-out 过拟合 |
| 两段式记忆展开（regenerative） | E4：引入幻觉、无净收益 |
| 无条件 V-RAG / 无条件塞满历史 | SimpleStream Table-4 + 我们 gate 实验（全历史坏 6 题） |
| 预测性训练目标（tail-mask 双视图 / MTP 帧间预测）在 8B-LoRA/4k 步上 | V4（EXP-05）：S2 两种子净负，S1 未确立 |
| mask 污染 CE 视图（无预测项的双 CE） | V4 dvce25：显著更差（p=0.003） |

## 4. 有效结论清单（可以站上去的地基）

1. **记忆在回溯任务上有分可拿**：OVO Backward（EPM/ASI）上全历史 vs recent-window = +19/353（p≈0.01，
   独立核验）；StreamingBench 类剪辑素材测不出记忆差异（评测选择先于方法决定成败）。
2. **query 相关检索是唯一有效的历史读取方式**（E3）；**当前窗不可被历史污染**（E3/E4/SimpleStream 一致）。
3. **文本记忆成本硬赢**：250–500× token 压缩、亚秒回答；失效形态已刻画（自我复读幻觉、数字类信息死亡）。
4. **小 predictor 可学**（future-latent 优于 copy-last +5.4%），但至今未转化为 QA 增益。
5. **种子纪律**：ctrl 换种子自涨 3.1pp（实测）——单种子结论一律无效，必须 ≥2 种子或双 ctrl 配对。
6. **管线全链路已验证**：训练→评测→GitHub 回传全通（V4 跑通 12 臂），SFT +4~6pp 显著。
7. 对比方法论三层：training-free 方法同 backbone 比机制；训练方法 7B 组内比；自己的 claim 只认同数据 λ=0 对照。

## 5. 实验总账（详细版见 jepa 仓库 REGISTRY.md，此处为项目级视图）

| 阶段 | 内容 | 判定 |
|---|---|---|
| 探索 R1（.31 服务器） | 5 方向并行：信号/写入/读取/生成式记忆/在线适应，每方向经对抗核验 | 定性结论见 §3/§4，ROUTE_REPORT_round1.md |
| v0 + 门槛 + Pareto | span-residual 判据、OVO backward 门槛检验、预算扫描 | 判据无优势；**门槛检验阳性**（记忆有分可拿） |
| V4 / EXP-05（集群） | 8B-LoRA 11 臂：tail 双视图 / MTP / 双 CE 消融 / 双种子 | **预注册负结果，线关闭**（V4_VERDICT.md） |
| **SimpleStream++（进行中）** | 14 臂：A 校准 2 / B 路由 4 / C 通道 6 / D 几何 2 | **一个臂都未跑**——当前全部注意力在此 |

## 6. 当前待办（按优先级，已写进 simplestream-plus/EXECUTE_NOW.md）

1. **a1**（校准：官方基线复现，过线标准 67.70±1）→ **b3**（主臂：任务路由+CLIP 历史 append）→
   **b4**（负对照：replace 模式应复现论文的下降）。三臂构成完整可发表故事，纯推理各 3–6 小时。
2. a1 过线后其余 9 臂一键提交（`scripts/submit_plus.sh b1 b2 c1a c1b c2 c3 c4a c4b d1`）。
3. 每个里程碑 `bash scripts/push_results.sh "<描述>"` 回传 GitHub。
4. 低优先级：EXP-07（mtp1 补种子，仅闲置算力）。

**b3 判读（预注册）**：OVO 总分 > a1 且实时轨 ≥ a1−1.0、回溯轨配对显著上移；路由器阶梯 b1≤b2≤b3。

## 7. 工作纪律（用户要求，长期有效）

- 每个实验独立清晰命名的文件夹；代码文件名直观（禁 run2/test_v3）。
- 禁自造综合指标；只报原始计数/配对翻转/符号检验；失败案例与成功同篇幅。
- 结论"可以就是可以，不行就是不行"；负结果照实写。
- 新实验先在 REGISTRY.md 登记 EXP 编号再跑；已关闭实验不得重跑。
- 训练数据与评测集视频零重叠；评测只认官方协议官方计分。
