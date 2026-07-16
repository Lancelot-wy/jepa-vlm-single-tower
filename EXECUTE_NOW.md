# EXECUTE_NOW — 两阶段执行手册（集群侧）

> 自足手册，按顺序执行。任何一步与描述不符 → 停下报告，**不要改代码语义或超参**。
> 平台：vivolm（L40s，4 卡/节点），训练经 `scripts/cluster/submit_batch.sh`
> （TF_CONFIG→torchrun 自动 rendezvous），每臂默认 NODES=2（有效 batch 128）。

## 总览与时间预估

| 阶段 | 内容 | 依赖 | 预计墙上时间 |
|---|---|---|---|
| **阶段 1 = EXP-08** | 无 mask 纯 MSE 收益：3 训练臂 + 8 评测 | 无，立即可跑 | **约半天**（6 节点并行训 ~2h + 评测 ~2h） |
| **阶段 2 = EXP-09** | 数据扩容（178K 多子集 + NExT-QA train）+ v2 时序增广：4 训练臂 + 评测 | 数据准备 | **约 1 个工作日**（数据 3–5h + 训 2–5h + 评 2–3h） |

两阶段可流水：阶段 1 训练排队期间做阶段 2 的数据准备。节点紧张时优先级：
EXP-08 三臂 > EXP-09 数据 > EXP-09 训练。

---

# 阶段 1（EXP-08）：无 mask 纯 MSE（帧预测）收益

回答：**CE 之上加"帧间预测 MSE"（无任何 mask），benchmark 值多少分？**

```
MSE: pred = MLP(LLM 末层在第 t 帧位置的 hidden)          ← 干净输入
     target = LayerNorm(ViT 对第 t+1 帧池化特征).detach()  ← 编码器空间
     total = CE + 0.2 * MSE
（代码实现名叫 MTP head, mtp_k=1 = 预测下一帧; 配置里的 mtp_* 键即指它）
```

| 臂 | loss | seed | 状态 |
|---|---|---|---|
| r3_sft | 纯 CE | 0 | **已训完**，勿重跑 |
| r3_sft_s1 | 纯 CE | 1 | 待训 |
| r3_mse | CE + 0.2·MSE | 0 | 待训 |
| r3_mse_s1 | CE + 0.2·MSE | 1 | 待训 |

（双种子 = 噪声量尺：V4 实测对照臂换 seed 自涨 3.1pp，单种子结论无效。）

## 1.0 前置检查

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower && git pull
python3 - <<'EOF'
from jepa_vlm.config import load_config
for n in ("r3_mse", "r3_mse_s1", "r3_sft_s1"):
    c = load_config(f"configs/{n}.yaml")
    print(n, "| mask:", c.model.mask_variant, "| reg:", c.model.reg_enabled,
          "| mse(mtp):", c.model.mtp_enabled, c.model.mtp_k, "| lambda:", c.train.lambda_reg,
          "| seed:", c.train.seed)
EOF
# 预期: mse 臂 reg=False mtp=True k=1 lambda=0.2; sft_s1 lambda=0.0 seed=1
# 评测数据定位（Round-3 同一批）:
ls /data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/
#   记 MVBench / TempCompass 的 jsonl 路径 → $MVB / $TC
ls outputs/r3_sft/step_4000/    # 确认对照 ckpt 在
```

## 1.1 训练（~2h/臂，3 臂并行）

```bash
NODES=2 EXTRA_OVERRIDES='train.min_flow=8.42' \
  scripts/cluster/submit_batch.sh r3_mse r3_mse_s1 r3_sft_s1
```
监控 `outputs/<exp>/log.jsonl`：mse 臂应有 `mtp_loss_k1`（~1.3 起步下降）、无 `reg_loss`
字段；NaN/发散 → kill 报告。

## 1.2 评测（单卡，~1h/run，可并行；每 run 约 MVBench 4k / TempCompass 1.6k 题 × 选项数次前向）

```bash
MVB=<MVBench jsonl>; TC=<TempCompass jsonl>
OUT=/data/vjuicefs_sz_ocr_wl/public_data/11193960/outputs
mkdir -p results/exp08
for E in r3_mse r3_mse_s1 r3_sft_s1; do
  python -m jepa_vlm.probes.mcq_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --data $MVB --task MVBench     --output results/exp08/${E}_mvbench.json
  python -m jepa_vlm.probes.mcq_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --data $TC  --task Tempcompass --output results/exp08/${E}_tempcompass.json
done
# 补 r3_sft(seed0) 逐题记录（配对检验需要）:
for T in "MVBench $MVB" "Tempcompass $TC"; do set -- $T
  python -m jepa_vlm.probes.mcq_eval --config $OUT/r3_sft/config.json --ckpt $OUT/r3_sft/step_4000 \
      --data $2 --task $1 --output results/exp08/r3_sft_$(echo $1 | tr 'A-Z' 'a-z').json
done
# 可选: 时序 QA
for E in r3_mse r3_mse_s1 r3_sft_s1; do
  python -m jepa_vlm.probes.temporal_qa_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --manifest /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data/llava_video/val.jsonl \
      --max-clips 500 | tee results/exp08/${E}_temporalqa.txt
done
```

---

# 阶段 2（EXP-09）：数据扩容 + v2 时序增广下的 {sft, mse} 复验

动机：EXP-04 的 TempCompass 子任务拆解显示模型精确败在时序四项
（speed 42%/direction 44%/order 53%/attr_change 53%，而 action 94%）；且训练数据
（单子集 ~2.5 万 QA 翻 20 epoch）过度重复、无 MCQ 格式。本阶段三个数据动作 +
v2 增广（order_yn/order_mcq/playback/speed/pan 五模板，标签全部自生成）。

| 臂 | loss | seed |
|---|---|---|
| exp9_sft / exp9_sft_s1 | 纯 CE | 0 / 1 |
| exp9_mse / exp9_mse_s1 | CE + 0.2·MSE | 0 / 1 |

四臂同数据同增广同超参；两两配对。**temporal_qa_templates=v2 已在 config 里**。

## 2.0 数据盘点（先报数，再动手）

```bash
DATA=/data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data
ls /data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K/jsonl/ 
ls /data/vjuicefs_ai_gpt_vision_wl04/public_data/origin_data/open_source_data/NExTQA/meta
head -2 <NExTQA meta 的 train.csv>     # 核对表头: video,question,answer,a0..a4
```

## 2.1 数据构建（CPU，~3–5h，绝大部分是 flow 打分）

```bash
mkdir -p $DATA/exp09
# (a) 178K 多子集 QA（在现有 academic 之外加 2-3 个子集; 子集名以 2.0 盘点为准）
python scripts/prepare_llava_video.py \
    --root /data/vjuicefs_ai_ocr_wl/public_data/video_data/LLaVA-Video-178K \
    --subsets <子集1> <子集2> <子集3> --out-dir $DATA/exp09/llava_multi --qa --max-videos 60000
# (b) NExT-QA train -> MCQ manifest（只允许 train.csv, 脚本会拒绝 val/test）
python scripts/prepare_nextqa.py --csv <meta>/train.csv \
    --video-root <NExTQA 视频目录> --vid-map <meta>/map_vid_vidorID.json \
    --out $DATA/exp09/nextqa_train.jsonl
# (c) 合并 + 视频级去重（178K 里本身含 NextQA, 重复计票要去掉）
cat $DATA/exp09/llava_multi/qa_train.jsonl $DATA/exp09/nextqa_train.jsonl \
    > $DATA/exp09/qa_train_raw.jsonl
python3 - <<'EOF'
import json, os
seen_pairs, out = set(), open(os.environ["DATA"] + "/exp09/qa_train_dedup.jsonl", "w")
import sys
for line in open(os.environ["DATA"] + "/exp09/qa_train_raw.jsonl"):
    d = json.loads(line)
    key = (os.path.splitext(os.path.basename(d["video"]))[0].lower(), d["question"][:80])
    if key in seen_pairs: continue
    seen_pairs.add(key); out.write(line)
print("dedup ->", len(seen_pairs))
EOF
# (d) 污染检查（必过门槛! 178K 含 PerceptionTest/Charades/ActivityNet = MVBench 源库）
python scripts/check_contamination.py --train $DATA/exp09/qa_train_dedup.jsonl \
    --bench $MVB $TC <NextQA评测jsonl(若有)> --clean-out $DATA/exp09/qa_train_clean.jsonl
# (e) flow 打分（训练时用 min_flow=8.42 同一全局阈值）
python scripts/compute_flow.py --manifest $DATA/exp09/qa_train_clean.jsonl \
    --out $DATA/exp09/qa_train_flow.jsonl --method framediff --workers 16
wc -l $DATA/exp09/qa_train_flow.jsonl    # 报告最终规模（目标 >= 10 万 QA, epoch 数降到 <5）
```

## 2.2 训练（4 臂，~2h/臂；8 节点并行 wall ~2h，4 节点两批 ~4–5h）

```bash
NODES=2 EXTRA_OVERRIDES='train.min_flow=8.42' \
  scripts/cluster/submit_batch.sh exp9_sft exp9_sft_s1 exp9_mse exp9_mse_s1
```

## 2.3 评测（与 1.2 同式，输出到 results/exp09/；4 臂 × {MVBench, TempCompass} + 时序 QA）

**判读重点（预注册）**：① exp9_mse − exp9_sft 配对（新数据下 MSE 效应，主判定）；
② exp9_sft vs r3_sft（数据扩容+增广本身值多少，同尺可比）；③ TempCompass 的
direction/speed/order 三个子任务单独拆（增广是否对症起效——mcq json 的 sub_type 字段直接分组）。

---

# 回传与完成标准

```bash
git add results/exp08 results/exp09 && git commit -m "EXP-08/09 results" && bash scripts/push_results.sh
```

- [ ] EXP-08：3 新臂 ckpt + results/exp08 8 个 mcq json（含逐题 results）
- [ ] EXP-09：qa_train_flow.jsonl 规模与污染检查数字已报告；4 臂 ckpt + results/exp09 8 个 mcq json
- [ ] 执行报告：各步实际耗时、解码失败率、skipped/unparsed 数、任何偏差

# 禁止事项

1. 不重跑 r3_sft(seed0) 训练；不动任何 config 数值与超参；
2. NExT-QA 只准用 train split；污染检查不过（重叠>0 且未剔除）不得开训 EXP-09；
3. 评测只用本仓库 mcq_eval / temporal_qa_eval（与历史同尺）；不要换 VLMEvalKit/lmms-eval；
4. 报告只交数字与过程，不下结论——配对检验与判定由发起方完成；
5. 与 vlm-jepa 仓库（mask 扫描 / EXP-06 锚定）互不引用数字、互不抢配额。
