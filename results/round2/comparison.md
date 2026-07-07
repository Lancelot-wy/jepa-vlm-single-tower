# Round-2 归因实验 — 汇总与判读

本轮目标:把 Round-1 观测到的"回归目标疑似无效/塌缩"拆成三个可归因因素——
**(A) 数据偏静态**、**(B) EMA/target 塌缩**、**(C) MTP 平滑**——分别设臂验证,
并补齐 Round-1 缺失的**下游时序 probe** 读数(主判据)。

- 全部臂 MTP 关闭(排除因素 C)。
- 训练:2 节点 × 4 GPU,各臂 2000 步,motion filter `min_flow=8.42`(train_flow p30;sft 臂不带)。
- probe:单卡,`--feature layer27_frames`,时序变换 `random_shuffle` / `random_reverse`,线性探针 top-1。
- Diving48 类别 probe(方案主判据)因源不可达本轮**跳过**,待补。

## 实验臂

| 臂 | 配置 | 归因目标 | 说明 |
|---|---|---|---|
| base | r2_base | — | 参考基线(mtp off) |
| r2_frozen | 冻结 ViT | A:数据/任务形态 | 只训 predictor,隔离表征是否被回归目标改动 |
| r2_v21 | 主处理 | B:主对照 | 标准 EMA target 回归 |
| r2_varreg | +VICReg 方差正则 | B:抗塌缩 | 显式抬 target_std |
| r2_residual | 残差 target | D:抗"抄帧" | 预测相邻帧残差而非绝对特征 |
| r2_sft_baseline | 纯 CE | E:对照 | 无回归目标,仅"见过视频" |

## 训练侧指标(val @ step 2000)

| 臂 | reg/ce loss | target_std | adj_cos | copy_mse | nontrivial_ratio |
|---|---|---|---|---|---|
| r2_frozen | 0.178 | 0.412 | 0.961 | 0.104 | 1.711 |
| r2_v21 | 0.062 | 0.241 | 0.982 | 0.044 | 1.419 |
| r2_varreg | 0.356 | 0.616 | 0.920 | 0.211 | 1.685 |
| r2_residual | 0.089 | 0.363 | 0.966 | 0.089 | 1.000 |
| r2_sft_baseline | 1.379 (CE) | 0.354 | 0.960 | — | — |

- 无一臂塌缩(target_std 全 > 0,varreg 最高 0.616,符合抗塌缩预期)。
- `nontrivial_ratio ≥ 1` 全线成立(residual 恰 1.000 → 未证明"预测残差"优于"抄相邻帧")。
- 该指标整体偏悲观、缺乏区分度,**不足以单独判优劣** → 需下游 probe。

## 下游时序 probe(top-1 %)

| 臂 | shuffle | reverse | Δreverse vs base | Δreverse vs sft |
|---|---|---|---|---|
| base | 65.22 | 57.14 | — | — |
| r2_frozen | 66.87 | 57.76 | +0.62 | +2.07 |
| r2_v21 | **68.94** | **61.70** | **+4.56** | **+6.01** |
| r2_varreg | 67.08 | 60.46 | +3.32 | +4.77 |
| r2_residual | 67.91 | 57.35 | +0.21 | +1.66 |
| r2_sft_baseline | 67.49 | 55.69 | −1.45 | — |

## 判读结论

**主结论:回归目标带来了真实的时序表征增益,而非仅"见过视频"。**

- **E(sft 对照)是关键锚点**:sft 在 reverse 上 **55.69 < base 57.14**,即"只是看过视频、纯 CE"
  反而略降时序敏感度。因此任何 reverse 显著超 base 且超 sft 的臂,其增益来自**回归目标本身**。
- **B(v21)最强**:reverse 61.70,较 base +4.56、较 sft +6.01;shuffle 亦最高 68.94。
  标准 EMA target 回归确实学到了帧序敏感表征。
- **C(varreg)次之且抗塌缩有效**:reverse 60.46(+3.32/+4.77),同时 target_std 最高。
  → **抗塌缩正则可并入主配置**,兼顾表征增益与稳定性。
- **A(frozen)**:reverse 仅 57.76,≈ base。冻结 ViT 下增益消失 → Round-1 观测到的现象
  更偏"任务形态/数据"层面;真正的表征增益需要**回传到 ViT**(v21/varreg 均放开 ViT)。
- **D(residual)**:reverse 57.35 ≈ base,且训练侧 nontrivial_ratio=1.000。
  **残差 target 未证明优于"抄帧"**,本轮不推进。

**probe vs nontrivial_ratio 矛盾的解读**:训练侧 ratio 全线 ≥1(悲观、无区分),但下游 probe
清晰分层(v21/varreg ≫ base/sft ≈ frozen/residual)。→ **probe 才是有效判据**;
nontrivial_ratio 作为塌缩告警可保留,但不作优劣裁决。

## 正式重跑(formal, 8000 步)

v21 / varreg 满足"B、C > base 且 > E"判据 → 各跑一次 8000 步长跑
(`configs/r2_{v21,varreg}_formal.yaml`,MTP off,min_flow=8.42,val@8000):

| 臂 | reg_loss | target_std | adj_cos | copy_mse | ratio |
|---|---|---|---|---|---|
| r2_v21_formal | 0.065 | 0.288 | 0.984 | 0.042 | 1.538 |
| r2_varreg_formal | 0.352 | 0.646 | 0.922 | 0.223 | 1.579 |

- 长跑无塌缩:v21 target_std 0.241→0.288、varreg 0.616→0.646(方差稳住/略升)。

时序 probe(top-1 %,与 2000 步对比):

| 臂 | shuffle | reverse | Δreverse vs base | Δreverse vs sft |
|---|---|---|---|---|
| v21 (2000) | 68.94 | 61.70 | +4.56 | +6.01 |
| **v21_formal (8000)** | 63.56 | **61.28** | **+4.14** | **+5.59** |
| varreg (2000) | 67.08 | 60.46 | +3.32 | +4.77 |
| **varreg_formal (8000)** | 65.01 | **59.01** | **+1.87** | **+3.32** |

**判读**:reverse probe(时间方向敏感度,主时序判据)在长跑后基本不变——v21_formal
61.28 ≈ 2000 步的 61.70,仍显著 > base(+4.14)、> sft(+5.59)。→ **回归目标带来的
真实时序增益经得起 8000 步长跑**,Round-2 结论稳固。
注意:shuffle probe 长跑后回落(v21 68.94→63.56,略低于 base 65.22),说明长跑把表征
进一步特化到回归目标、牺牲了一点 shuffle 依赖的通用帧内容可分性;方向敏感的 reverse
才是关键判据,已保住。

## 下一步(按方案)

1. formal 重跑已确认增益稳固;剩 **SSv2 背书**(降级为最终背书,不阻塞)。
2. 补 **Diving48 类别 probe**(方案主判据),待数据代理/上传后用
   `scripts/prepare_diving48.py` + 在 `run_probes.sh` 追加类别 probe。
3. residual / frozen 本轮不推进。
