# EXECUTE_NOW — EXP-08：无 mask 纯 MSE（帧预测）收益（集群侧执行手册）

> 自足手册，按顺序执行。任何一步与描述不符 → 停下报告，**不要改代码语义或超参**。

## 背景（30 秒）

回答一个问题：**在 CE 训练之上加"帧间预测 MSE"（无任何 mask），benchmark 上值多少分？**

MSE 定义（jepa_vlm/modeling/model.py 的 MTP 分支，k=1）：
```
pred   = MLP( LLM 最后一层在第 t 帧位置的 hidden )        ← 干净输入, 无 [M]
target = LayerNorm( ViT 对第 t+1 帧的池化特征 ).detach()   ← 编码器空间, stop-grad
total  = CE + 0.2 * MSE      (逐 token 对齐, t = 1..T-1)
```

对照结构（2×2，种子纪律：单种子结论无效，见 REGISTRY）：

| 臂 | loss | seed | 状态 |
|---|---|---|---|
| r3_sft | 纯 CE | 0 | **已训完**（outputs/r3_sft），勿重跑 |
| r3_sft_s1 | 纯 CE | 1 | 待训 |
| r3_mtp1 | CE + 0.2·MSE | 0 | 待训 |
| r3_mtp1_s1 | CE + 0.2·MSE | 1 | 待训 |

## 0. 前置检查

```bash
cd /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa-vlm-single-tower && git pull
# (a) 配置自检 —— 预期: mtp 臂 reg=False mtp_k=1 lambda=0.2 无mask(v1); sft 臂 lambda=0
python3 - <<'EOF'
from jepa_vlm.config import load_config
for n in ("r3_mtp1", "r3_mtp1_s1", "r3_sft_s1"):
    c = load_config(f"configs/{n}.yaml")
    print(n, "| mask:", c.model.mask_variant, "| reg:", c.model.reg_enabled,
          "| mtp:", c.model.mtp_enabled, c.model.mtp_k, "| lambda:", c.train.lambda_reg,
          "| seed:", c.train.seed, "| steps:", c.train.max_steps)
EOF
# (b) 评测数据定位（Round-3 用过的同一批, 内部评测集合）:
ls /data/vjuicefs_sz_ocr_wl/public_data/11189192/automatic-evaluation/eval_data/v3_5_data/
#    记下 MVBench 与 TempCompass 的 jsonl 路径 → 下面 $MVB / $TC
# (c) 确认 r3_sft(seed0) 产物在: ls outputs/r3_sft/step_4000/
```

## 1. 训练（3 台并行，每臂 ~2h，数据/步数/超参全部继承 Round-3）

```bash
NODES=2 EXTRA_OVERRIDES='train.min_flow=8.42' \
  scripts/cluster/submit_batch.sh r3_mtp1 r3_mtp1_s1 r3_sft_s1
```

监控 `outputs/<exp>/log.jsonl`：mtp 臂会多出 `mtp_loss_k1` 字段（预期从 ~1.3 起步下降）；
`ce_loss` 与 sft 臂同量级属正常；NaN/发散 → kill 报告。

## 2. 评测（单卡；3 个新 ckpt × 2 benchmark + 时序 QA）

```bash
MVB=<第0步的 MVBench jsonl>
TC=<第0步的 TempCompass jsonl>
OUT=/data/vjuicefs_sz_ocr_wl/public_data/11193960/outputs
mkdir -p results/exp08

for E in r3_mtp1 r3_mtp1_s1 r3_sft_s1; do
  python -m jepa_vlm.probes.mcq_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --data $MVB --task MVBench     --output results/exp08/${E}_mvbench.json
  python -m jepa_vlm.probes.mcq_eval --config $OUT/$E/config.json --ckpt $OUT/$E/step_4000 \
      --data $TC  --task Tempcompass --output results/exp08/${E}_tempcompass.json
done

# 补齐 r3_sft(seed0) 的逐题记录（Round-3 只存了汇总文本, 配对检验需要逐题 json）:
for T in "MVBench $MVB" "Tempcompass $TC"; do set -- $T
  python -m jepa_vlm.probes.mcq_eval --config $OUT/r3_sft/config.json --ckpt $OUT/r3_sft/step_4000 \
      --data $2 --task $1 --output results/exp08/r3_sft_$(echo $1 | tr 'A-Z' 'a-z').json
done

# 可选第三读数: held-out 时序 QA（与 Round-3 同 manifest 同 seed）
for E in r3_mtp1 r3_mtp1_s1 r3_sft_s1; do
  python -m jepa_vlm.probes.temporal_qa_eval --config $OUT/$E/config.json \
      --ckpt $OUT/$E/step_4000 \
      --manifest /data/vjuicefs_sz_ocr_wl/public_data/11193960/jepa_data/llava_video/val.jsonl \
      --max-clips 500 | tee results/exp08/${E}_temporalqa.txt
done
```

## 3. 回传

```bash
git add results/exp08 && git commit -m "EXP-08: no-mask CE+MSE(mtp1) 2x2 seeds, MVBench/TempCompass per-sample records"
bash scripts/push_results.sh   # 或直接 git push（轻量 json 走 results-cluster 分支的惯例亦可）
```

## 4. 完成标准

- [ ] 3 个新臂训到 step_4000
- [ ] results/exp08/ 含 8 个 mcq json（4 臂 × 2 benchmark，逐题 results 字段完整）
- [ ] （可选）3 份 temporal QA 输出
- [ ] 全部 push；报告实际耗时与异常（解码失败率、skipped 数）

## 5. 禁止事项

1. 不重跑 r3_sft(seed0)（评测补测可以，训练不行）；不动任何 config 数值；
2. 评测必须用本仓库 mcq_eval（likelihood 口径，与 Round-3 可比）；不要换 VLMEvalKit/lmms-eval；
3. 报告只交数字，不下结论——配对检验（每种子 mtp1−sft、跨种子合并）由发起方做；
4. 与 vlm-jepa 仓库的 mask 扫描是两条独立线，互不引用数字。
