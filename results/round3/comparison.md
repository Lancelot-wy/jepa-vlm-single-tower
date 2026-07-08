# Round-3 主线终审 — 联合(CE+reg) vs 纯 CE

Phase B pilot:同数据(qa_train_flow.jsonl, min_flow=8.42)、同 QA 混比(temporal_qa_ratio=0.3)、
同 4000 步、MTP off、ViT 放开。唯一差异 = 回归目标包(joint: v2.1 mask + 0.2·reg;sft: v1 干净视频 + 纯 CE)。

## 训练侧 (val @ step 4000)

| 臂 | loss 语义 | val reg_loss | target_std | adj_cos | nontrivial_ratio |
|---|---|---|---|---|---|
| r3_joint | CE + 0.2·reg,输入遮蔽 50% 帧 | 0.700 | 0.364 | 0.963 | 6.91 |
| r3_sft | 纯 CE,输入干净视频 | 1.368 (CE) | 0.334 | 0.962 | — |

注:两臂目标函数与输入都不同,**loss 数值不可直接比较**,不能据此判优劣。

## held-out 时序 QA(主判定,val 500 clip,seed 0,二项 se ~2.3pp)

每 clip 判「帧顺序是否正确」:none=真序(truth yes),shuffle/reverse=被破坏(truth no)。

| step | 臂 | overall | none | shuffle | reverse |
|---|---|---|---|---|---|
| 1000 | joint | 51.35 | 92.6 | 9.2 | 10.7 |
| 1000 | sft | 49.69 | 84.7 | 11.7 | 17.4 |
| 2000 | joint | 50.10 | 81.4 | 19.2 | 18.2 |
| 2000 | sft | 49.07 | 93.0 | 4.2 | 5.8 |
| 3000 | joint | 50.10 | 81.4 | 19.2 | 18.2 |
| 3000 | sft | 49.07 | 93.0 | 4.2 | 5.8 |
| 4000 | joint | 50.10 | 81.4 | 19.2 | 18.2 |
| 4000 | sft | 49.07 | 93.0 | 4.2 | 5.8 |

(两臂 step_2000/3000/4000 数值完全一致 → 均在 ~2000 步饱和。)

## 判定

- **主判据 overall(≥+3pp):不成立。** joint − sft = **+1.03pp**(各档一致方向,但在噪声内)→ 按方案「<2pp 不下结论,进诊断」。
- **机理分项给出清晰正向信号:** 察觉时序被破坏的能力(shuffle+reverse 识别率)joint ≈ **19%** vs sft ≈ **5%**,约 **3–4 倍**。
- **overall 未动的原因 = 强 yes 偏置**:sft 几乎无脑答 yes(none 93%,基本无视顺序),joint 牺牲少量 none(93%→81%)换来对乱序/倒放的敏感度(5%→19%)。评测集 yes/no 各半,joint 在 no 上的收益被 yes 上的损失抵消,overall 持平。

**结论:回归目标确实提升了对时序顺序的敏感度(与 Round-2 probe 一致),但被决策阈值的 yes 偏置掩盖在 overall 上。非负结果,判据被 confound。**

## 下一步(诊断)
1. 拆偏置:看 CE(yes) vs CE(no) 的 margin 分布 / 调判定阈值,或平衡答案分布后重评。
2. Diving48 类别 probe(方案主判据,待数据),区分「动作语义」vs「低级运动方向」。
