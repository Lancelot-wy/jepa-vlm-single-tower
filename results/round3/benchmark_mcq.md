# Round-3 readout #3 — 公开 benchmark MCQ (MVBench / TempCompass)

联合(CE+reg)vs 纯 CE sft,step_4000,全量评测。原始数字见 `mcq_eval_results.txt`。

## 方法

自研单塔模型无 generate,故用 **answer-likelihood 打分**(与 held-out 时序 QA 一致):
逐选项做 no-mask forward,取 mean per-token CE(`JepaOutput.ce_per_sample`,已长度归一),
CE 最低的选项为预测,letter 与 `目标值` 匹配即正确。帧取 benchmark 预抽的 `meta.images_info`
jpg(均匀采 `num_frames`),不重解 mp4(MVBench `_processed.mp4` 已裁剪,而 Charades
start/end 指向原始未裁剪视频、会越界 EOF)。适配器 `jepa_vlm/probes/mcq_eval.py`。

## 结果

| Benchmark | N | joint | sft | Δ(joint−sft) |
|---|---|---|---|---|
| MVBench      | 3995 | 49.59% | 49.56% | **+0.03pp** |
| Tempcompass  | 1580 | 57.78% | 58.16% | **−0.38pp** |

两臂 overall 差异均 < 0.4pp,落在噪声内(二项 se: MVBench ~0.8pp,Tempcompass ~1.2pp)。

### MVBench 分项 (joint / sft, %)

时序相关类互有胜负,无一致方向:

| 子类 | joint | sft | Δ |
|---|---|---|---|
| Object Shuffle       | 23.0 | 27.5 | **−4.5** |
| Action Sequence      | 56.5 | 59.0 | −2.5 |
| Fine-grained Pose    | 41.5 | 44.0 | −2.5 |
| Moving Count         | 46.0 | 43.0 | +3.0 |
| Moving Direction     | 20.0 | 17.5 | +2.5 |
| State Change         | 48.5 | 46.5 | +2.0 |
| Counterfactual Inf.  | 39.0 | 36.9 | +2.1 |
| Action Antonym       | 71.0 | 69.5 | +1.5 |
| Scene Transition     | 87.0 | 86.0 | +1.0 |

其余类 |Δ| ≤ 1pp。高准确率类(Scene Transition 87%、Unexpected Action 73%)两臂几乎相同。

### TempCompass 分项 (joint / sft, %)

| 子类 | joint | sft | Δ |
|---|---|---|---|
| mcq-action           | 93.79 | 93.79 | 0.0 |
| mcq-order            | 52.98 | 53.97 | −0.99 |
| mcq-attribute_change | 53.47 | 53.82 | −0.35 |
| mcq-direction        | 44.48 | 43.88 | +0.60 |
| mcq-speed            | 43.85 | 41.32 | **+2.53** |

## 判定

- **在公开 benchmark 上两臂等价**:MVBench +0.03pp、Tempcompass −0.38pp,均在噪声内;
  分项混合、多数 |Δ| < 3pp。
- reg 目标在自建 held-out 时序 QA 上看到的「时序破坏敏感度 3–4×」提升(见 `comparison.md`),
  **未转化为 MVBench / TempCompass 的 MCQ 准确率优势**。
- 最能考时序顺序的类(Object Shuffle / Action Sequence)joint 反而略低,与 held-out 结论
  相反 —— 说明该增益脆弱、依赖任务格式(yes/no 破坏判定 vs 4/N 选一 MCQ),不泛化到标准 benchmark。

**结论:与 Round-3 主判据一致 —— reg 增益真实但小且脆弱,在公开 benchmark 上不可见。**
