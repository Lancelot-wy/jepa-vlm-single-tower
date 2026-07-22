# EXP-12 原生 Qwen 锚点评测执行手册（EXP-13）

## 1. 为什么必须补这组评测

EXP-12 在同一套自研评测器下给出了稳定结果：K=4/16/64 的纯 CE 臂在
MVBench 上为 47.61/52.07/54.47，在 TempCompass 上为 55.32/57.41/60.19；
Observation Query 在三个 K 上均无增益。这个结论足以说明**在当前自研管线内部，增加
真实视觉信息比 Query 辅助 loss 更有效**，但还不能回答两个问题：

1. K=4 为什么明显低于公开的 Qwen3-VL 结果；
2. 训练后的权重相对原始 Qwen 权重究竟是提升还是退化。

原因是历史 `mcq_eval.py` 同时改变了预处理、视频模板和评分规则：视频强制
center-crop 到 256×256；全部视觉占位符放在一个连续 block 中，没有 Qwen3-VL 的逐时序
单元时间戳/分隔符；最后以完整选项文本的平均 token CE 选答案，而不是生成答案字母。
EXP-13 只做诊断评测，不重新训练，用严格的 raw-base 对照把这些因素拆开。

## 2. Qwen 原生视觉 token 不是 K=4

Qwen3-VL 原生 processor 使用动态分辨率，视觉 token 数随视频尺寸和长宽比变化，不是
固定 4。在本项目当前的 256×256 输入下：

```text
patch grid       = 256 / 16 = 16 × 16
spatial merger   = 2 × 2
native K / unit  = (16 / 2) × (16 / 2) = 64
32 raw frames / temporal_patch_size 2 = 16 temporal units
native video tokens = 16 × 64 = 1024
```

本项目的 K=4 是在冻结 Qwen ViT/merger 的 8×8 输出后额外做 2×2 自适应平均池化，
也就是每个时序单元把 64 个原生 token 压成 4 个。K=64 才是 256×256 设置下不再进行
额外空间压缩的原生上限。**保持 256×256 时不要扫 K=128/256**：现有 pooler 只能把
8×8 特征插值放大，不会创造新视觉信息；若要增加 token，必须先注册更高分辨率或保留
长宽比的实验，并重算显存、上下文和数据吞吐。

## 3. 冻结的评测矩阵

所有行使用同一份 MVBench/TempCompass JSONL、同一个题目顺序和逐题 `idx`。训练权重
固定为 EXP-12 的 `a4_ce_k64/checkpoint-800`。

| 协议 | 权重 | K/视觉输入 | 决策规则 | 用途 |
|---|---|---|---|---|
| custom_base_k4_full_option | raw Qwen | K=4 | 完整选项平均 token CE | 历史 K=4 raw-base 锚点 |
| custom_base_k16_full_option | raw Qwen | K=16 | 同上 | raw-base K 曲线 |
| custom_base_k64_full_option | raw Qwen | K=64 | 同上 | raw-base K 曲线 |
| custom_ckpt_k64_full_option | EXP-12 K64 | K=64 | 同上 | 与已发布 EXP-12 尺子一致 |
| custom_base_k64_letter | raw Qwen | K=64 | 仅候选 A/B/C… 的平均 token CE | 量化完整选项长度/措辞偏差 |
| custom_ckpt_k64_letter | EXP-12 K64 | K=64 | 同上 | 同尺训练效应 |
| native_base_matched32_generation | raw Qwen | 原生兼容 processor，匹配 32 帧 | greedy 生成答案字母 | 原生 Qwen 内部锚点 |
| native_ckpt_k64_matched32_generation | EXP-12 K64 | 同上 | 同上 | 原生协议下的 SFT 效应 |

原生行仍固定使用 benchmark 的 32 张预抽帧，目的是和历史行做可配对诊断；它恢复
Qwen 的 aspect-preserving smart resize、动态视觉 grid、每个 temporal unit 的时间戳与
`vision_start/end`、原生 MRoPE 和生成式回答。若样本带 duration，就由 duration 推导时间戳；
否则明确回退到 4 fps。该协议不声称复现 Qwen 公布的 61.7：官方表格使用更大的上下文/
帧预算，公开绝对分数不能和这里的 32 帧内部集直接混用。

公司 `jepa311` 环境没有 torchvision，而对应 Transformers 版本的 `AutoProcessor` 会在
初始化 Qwen 视频 processor 时失败。为避免运行时装包，native 行从本地
`video_preprocessor_config.json` 读取官方 pixel budget，并用 torch 实现同一套
`smart_resize`、0.5/0.5 normalization 和官方 patch layout；文本侧用模型原生 tokenizer/
chat template 构造完全相同的时间戳与逐 unit placeholder，位置编码仍由原生
`Qwen3VLForConditionalGeneration` 计算。结果中把实现明确标为
`torchvision_free_qwen3vl_compat_v1`，不伪装成未经验证的 AutoProcessor 字节级复现。

## 4. 服务器完整执行顺序

### 4.1 拉取并确认代码

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower
git fetch origin
git switch exp12-orca-token-sweep
git pull --ff-only origin exp12-orca-token-sweep
git status --short
git rev-parse HEAD
```

`git status --short` 必须没有已跟踪文件改动。不要把共享盘 checkpoint 提交到 GitHub。

### 4.2 设置一次路径

```bash
export BASE=/data/vjuicefs_sz_ocr_wl/public_data/11193960
export PROJECT_ROOT=$BASE/jepa-vlm-single-tower
export MODEL_ROOT=$BASE/models/Qwen3-VL-2B-Instruct
export EXP12_SOURCE_RESULTS=$BASE/runs/exp12/exp12-20260722-014706-c6de850/results/exp12_orca_token_sweep
export MVB=/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/MVBench/MVBench_v3_5_0.jsonl
export TC=/data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/Tempcompass/Tempcompass_v3_5_0.jsonl
export GPU_LIST=0,1,2,3
```

### 4.3 先跑 20 题 smoke

有公司队列权限时推荐直接使用 7 Worker 评测任务（3 个 Worker 并行 12 个 custom 条目，
4 个 Worker 分别负责四个 native 协议/数据集组合；每个 native Worker 内再按 4 GPU 分片）。
总计 7×4=28 张 L40S，只占当前 24 台可用机器中的 7 台，任务结束后入口命令退出并自动
释放 Worker：

```bash
# 先只打印并检查最终 YAML
bash scripts/exp12/20_submit_native_anchor.sh --dry-run --smoke

# 20 题端到端 smoke
bash scripts/exp12/20_submit_native_anchor.sh --smoke
```

下面是一台 4 卡开发机上的直接运行方式，适合没有队列权限时使用：

```bash
export EXP12_NATIVE_ANCHOR_ROOT=$BASE/runs/exp13/native-anchor-smoke
mkdir -p $EXP12_NATIVE_ANCHOR_ROOT
MAX_CLIPS=20 bash scripts/exp12/19_run_native_anchor.sh \
  2>&1 | tee $EXP12_NATIVE_ANCHOR_ROOT/launcher.log
```

通过标准：preflight PASS；custom 和 native 两阶段都 PASS；最终
`native_anchor_comparison.json` 的 `complete=true`，每行 `total>0`。如果 native 日志中
连续出现 tokenizer、`video_grid_thw`、overlay shape 或 CUDA 错误，停止正式评测并修代码，
不能退回历史 evaluator 冒充原生结果。

### 4.4 跑完整评测

公司队列推荐命令：

```bash
bash scripts/exp12/20_submit_native_anchor.sh --dry-run
bash scripts/exp12/20_submit_native_anchor.sh
```

提交脚本会打印 `run=<RUN_ID>`；结果位于 `$BASE/runs/exp13/<RUN_ID>/`。重用指定目录时
可加 `--run-id <RUN_ID>`，完整 JSON 已通过协议和 `MAX_CLIPS` 校验的条目会被复用。

单台 4 卡开发机的完整命令：

```bash
export EXP12_NATIVE_ANCHOR_ROOT=$BASE/runs/exp13/native-anchor-full
mkdir -p $EXP12_NATIVE_ANCHOR_ROOT
MAX_CLIPS=0 bash scripts/exp12/19_run_native_anchor.sh \
  2>&1 | tee $EXP12_NATIVE_ANCHOR_ROOT/launcher.log
```

若终端可能断开，使用以下完整后台命令；脚本可复用已完成 JSON，断线后重复执行不会
重跑有效结果：

```bash
mkdir -p $BASE/runs/exp13/native-anchor-full
nohup env \
  BASE="$BASE" PROJECT_ROOT="$PROJECT_ROOT" MODEL_ROOT="$MODEL_ROOT" \
  EXP12_SOURCE_RESULTS="$EXP12_SOURCE_RESULTS" MVB="$MVB" TC="$TC" \
  EXP12_NATIVE_ANCHOR_ROOT="$BASE/runs/exp13/native-anchor-full" \
  GPU_LIST=0,1,2,3 MAX_CLIPS=0 \
  bash scripts/exp12/19_run_native_anchor.sh \
  > $BASE/runs/exp13/native-anchor-full/launcher.log 2>&1 < /dev/null &
echo $!
```

查看状态：

```bash
tail -f $BASE/runs/exp13/native-anchor-full/launcher.log
find $BASE/runs/exp13/native-anchor-full -maxdepth 2 -type f | sort
```

分阶段执行顺序与一键脚本完全相同：

```bash
bash scripts/exp12/15_native_anchor_preflight.sh
bash scripts/exp12/16_eval_custom_anchor.sh
bash scripts/exp12/17_eval_native_anchor.sh
bash scripts/exp12/18_collect_native_anchor.sh
```

`16` 把不同 custom 条目分配到 4 张卡；`17` 将每个 native 条目按题号分成 4 个 shard，
每卡一个模型实例，完成后严格检查重复 `idx` 并合并。训练 `state.pt` 含 optimizer，脚本先
一次性导出只含 Qwen 权重的 `a4_ce_k64_native_overlay.pt`，避免四个 shard 重复加载完整
optimizer 状态。所有进程结束后命令退出；若由公司队列提交，Worker 随任务退出自动释放。

## 5. 结果产物与判读顺序

根目录包含 16 个主结果 JSON（8 协议×2 数据集）、逐 shard JSON/日志、overlay 审计、
`protocol_manifest.json`，以及：

```text
native_anchor_comparison.json
native_anchor_comparison.csv
native_anchor_comparison.md
```

每个主结果保存总分、skipped、逐子类别统计、逐题预测和原始生成；native 结果还保存每题
`video_grid_thw` 与实际视觉 token 数。汇总自动计算逐题配对 bootstrap 95% CI 和 McNemar
exact p-value。

必须按以下顺序解释：

1. **先看 raw base**。没有同协议 raw Qwen 分数，不允许说 SFT 让模型变好或变差。
2. `custom_base_k64 - custom_base_k4` 只回答视觉压缩的影响。
3. `custom_ckpt - custom_base` 回答训练在历史尺子上的净效应。
4. `native_ckpt - native_base` 才回答训练在原生 Qwen 协议上的净效应。
5. `native - custom` 同时包含 resize、模板/MRoPE 和生成规则的协议差，不能归因给单一模块。

诊断例子：

| 观察 | 最可能含义 | 下一步 |
|---|---|---|
| native base 明显高，native ckpt 下降 | 当前 SFT 伤害原生能力 | 降 LR/步数、加原生模板训练或保留 base replay |
| native base/ckpt 都高，custom 都低 | 主要是自研输入/评分协议损失 | 先修训练 collator 与 aspect-preserving preprocessing |
| native base 也低 | 数据版本、帧预算或 prompt 仍未对齐 | 核验 benchmark 版本和官方评测 harness |
| custom K64 高于 K4，但 native 差异小 | K 收益主要在补偿自研压缩 | 不再把 K 当作新模块贡献 |
| full-option 与 letter 差异大 | 历史评分受选项长度/文本先验影响 | 后续统一使用生成式或 letter-only 配对口径 |

## 6. 明确禁止的比较

- 不把本仓库 54.47 与 Qwen 公布的 61.7 直接相减并称作模型退化；
- 不把 K=4 称作“Qwen 默认视觉 token”；
- 不在 256×256 上直接跑 K=128/256；
- 不用不同 skipped 集合的 overall 分数做非配对结论；
- 不因原生评测分数更好就改写 EXP-12 内部配对结论。它只会改变结论的外部解释边界。

## 7. 协议依据

- [Qwen3-VL-2B-Instruct model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Transformers Qwen3-VL processor：逐时序单元时间戳与视觉占位符](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/processing_qwen3_vl.py)
- [Transformers Qwen3-VL video processor：动态 smart resize 与 patch layout](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_vl/video_processing_qwen3_vl.py)
