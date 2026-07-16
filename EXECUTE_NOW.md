# EXECUTE_NOW — EXP-09：扩充数据上的纯 MSE 收益（单阶段，集群侧执行手册）

> 自足手册，按顺序执行。任何一步与描述不符 → 停下报告，**不要改代码语义或超参**。
> 平台：vivolm（L40s，4 卡/节点），`scripts/cluster/submit_batch.sh` 提交，
> 每臂 NODES=2（有效 batch 128）。**EXP-08（旧小数据）已撤销，直接在扩充数据上测。**

## 回答的问题

**在 CE 训练之上加"帧间预测 MSE"（无任何 mask），benchmark 值多少分？**

```
MSE: pred = MLP(LLM 末层在第 t 帧位置的 hidden)           ← 干净输入
     target = LayerNorm(ViT 对第 t+1 帧池化特征).detach()   ← 编码器空间
     total = CE + λ·MSE, 默认 λ = 0.2（方案起点值; 本 loss 形态未扫过 λ）
（代码实现名 MTP head, mtp_k=1 = 预测下一帧; 配置 mtp_* 键即指它）
```

## 实验臂（4 主 + 1 可选，全部同数据同增广同超参）

| 臂 | loss | seed | 备注 |
|---|---|---|---|
| exp9_sft | 纯 CE | 0 | 对照 |
| exp9_sft_s1 | 纯 CE | 1 | 对照第二种子 |
| exp9_mse | CE + 0.2·MSE | 0 | 处理 |
| exp9_mse_s1 | CE + 0.2·MSE | 1 | 处理第二种子 |
| exp9_mse_lam05（可选） | CE + 0.5·MSE | 0 | λ 剂量探针，主判定不依赖 |

（双种子 = 噪声量尺：V4 实测对照臂换 seed 自涨 3.1pp，单种子结论无效。）

## 数据：扩容三件套 + v2 时序增广

- 178K 多子集（在 academic 之外加 2–3 个子集）+ **NExT-QA train**（原生 MCQ，
  对症 order + 答题格式）→ 合并去重 → **污染检查（硬门槛）** → flow 打分；
- 目标规模 **≥10 万 QA**（4000 步 ≈ 4–5 epoch，摆脱旧数据翻 20 遍的过拟合）；
- 训练时 `temporal_qa_templates=v2`（已在 config）：五个自标注模板
  order_yn / order_mcq / playback / speed(帧距×2) / pan(滑窗)，
  对症 EXP-04 拆出的 TempCompass 弱项（speed 42% / direction 44% / order 53%）。

## 时间预估（墙上时间）

| 步骤 | 耗时 |
|---|---|
| 0 数据盘点 | ~0.5h |
| 1 数据构建（大头是 flow 打分 10–15 万 clip） | 3–5h（CPU） |
| 2 训练 4 臂（+1 可选） | 8 节点并行 ~2h；4 节点两批 ~4–5h |
| 3 评测 4–5 臂 ×{MVBench, TempCompass} + 时序 QA | 2–3h（单卡 × 并行） |
| **合计** | **约 1 个工作日** |

## 0. 前置检查

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower && git pull
python3 - <<'EOF'
from jepa_vlm.config import load_config
for n in ("exp9_sft","exp9_sft_s1","exp9_mse","exp9_mse_s1","exp9_mse_lam05"):
    c = load_config(f"configs/{n}.yaml")
    print(f"{n:16s} mse={c.model.mtp_enabled} reg={c.model.reg_enabled} "
          f"lambda={c.train.lambda_reg} seed={c.train.seed} tpl={c.train.temporal_qa_templates}")
EOF
# 预期: sft 臂 mse=False lambda=0; mse 臂 mse=True reg=False lambda=0.2/0.5; tpl 全部 v2
# 评测数据定位（内部集合, Round-3 同一批）:
ls /data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/
#   记 MVBench / TempCompass 的 jsonl 路径 → $MVB / $TC
# 训练数据源:
ls /data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K/jsonl/
ls /data/vjuicefs_ai_gpt_vision_wl04/public_data/origin_data/open_source_data/NExTQA/meta
head -2 <NExTQA meta 的 train.csv>    # 核对表头: video,question,answer,a0..a4
```

## 1. 数据构建

```bash
export DATA=/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data
mkdir -p $DATA/exp09
# (a) 178K 多子集 QA
python scripts/prepare_llava_video.py \
    --root /data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K \
    --subsets <子集1> <子集2> <子集3> --out-dir $DATA/exp09/llava_multi --qa --max-videos 60000
# (b) NExT-QA train -> MCQ（脚本拒绝 val/test）
python scripts/prepare_nextqa.py --csv <meta>/train.csv \
    --video-root <NExTQA 视频目录> --vid-map <meta>/map_vid_vidorID.json \
    --out $DATA/exp09/nextqa_train.jsonl
# (c) 合并 + 视频级去重（178K 本身含 NextQA）
cat $DATA/exp09/llava_multi/qa_train.jsonl $DATA/exp09/nextqa_train.jsonl \
    > $DATA/exp09/qa_train_raw.jsonl
python3 - <<'EOF'
import json, os
seen, kept = set(), 0
out = open(os.environ["DATA"] + "/exp09/qa_train_dedup.jsonl", "w")
for line in open(os.environ["DATA"] + "/exp09/qa_train_raw.jsonl"):
    d = json.loads(line)
    key = (os.path.splitext(os.path.basename(d["video"]))[0].lower(), d["question"][:80])
    if key in seen: continue
    seen.add(key); out.write(line); kept += 1
print("dedup ->", kept)
EOF
# (d) 污染检查（硬门槛! 178K 含 PerceptionTest/Charades/ActivityNet = MVBench 源库）
python scripts/check_contamination.py --train $DATA/exp09/qa_train_dedup.jsonl \
    --bench $MVB $TC --clean-out $DATA/exp09/qa_train_clean.jsonl
# (e) flow 打分（训练用全局阈值 min_flow=8.42）
python scripts/compute_flow.py --manifest $DATA/exp09/qa_train_clean.jsonl \
    --out $DATA/exp09/qa_train_flow.jsonl --method framediff --workers 16
wc -l $DATA/exp09/qa_train_flow.jsonl   # 报告最终规模
```

## 2. 训练

```bash
NODES=2 EXTRA_OVERRIDES='train.min_flow=8.42' \
  scripts/cluster/submit_batch.sh exp9_sft exp9_sft_s1 exp9_mse exp9_mse_s1
# 有富余节点时追加可选探针: scripts/cluster/submit_batch.sh exp9_mse_lam05
```
监控 `outputs/<exp>/log.jsonl`：mse 臂有 `mtp_loss_k1`（~1.3 起步下降）、无 `reg_loss`；
NaN/发散 → kill 报告。

## 3. 评测（单卡 ~1h/run，可并行）

```bash
OUT=/data/vjuicefs_sz_ocr_wl/public_data/11193960/outputs
mkdir -p results/exp09
for E in exp9_sft exp9_sft_s1 exp9_mse exp9_mse_s1 exp9_mse_lam05; do
  [ -d $OUT/$E/step_4000 ] || continue
  python -m jepa_vlm.probes.mcq_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --data $MVB --task MVBench     --output results/exp09/${E}_mvbench.json
  python -m jepa_vlm.probes.mcq_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --data $TC  --task Tempcompass --output results/exp09/${E}_tempcompass.json
  python -m jepa_vlm.probes.temporal_qa_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --manifest /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data/llava_video/val.jsonl \
      --max-clips 500 | tee results/exp09/${E}_temporalqa.txt
done
```

## 4. 判读重点（预注册，发起方执行）

1. **主判定**：exp9_mse − exp9_sft 每种子配对 + 跨种子合并（McNemar，两 benchmark）；
2. **数据贡献**：exp9_sft vs r3_sft（同尺可比，回答"扩容+增广本身值多少"）；
3. **增广对症性**：TempCompass 的 direction/speed/order 子任务单独拆（mcq json
   的 sub_type 分组），看弱项是否被拉起；
4. λ 探针只在 1 出正向后解读。

## 回传与完成标准

```bash
git add results/exp09 && git commit -m "EXP-09 results" && bash scripts/push_results.sh
```
- [ ] 数据：最终 manifest 规模 + 污染检查数字已报告
- [ ] 4（+1）臂 ckpt 训完；results/exp09 每臂 2 个 mcq json（逐题 results 完整）+ 时序 QA
- [ ] 执行报告：各步耗时、解码失败率、skipped/unparsed、偏差

## 禁止事项

1. 不动任何 config 数值与超参；旧数据三臂（r3_mse*/r3_sft_s1）**不要提交**（已撤销）；
2. NExT-QA 只准用 train split；污染检查不过不得开训；
3. 评测只用本仓库 mcq_eval / temporal_qa_eval；不要换 VLMEvalKit/lmms-eval；
4. 报告只交数字与过程，不下结论；
5. 与 vlm-jepa 仓库互不引用数字、互不抢配额。
