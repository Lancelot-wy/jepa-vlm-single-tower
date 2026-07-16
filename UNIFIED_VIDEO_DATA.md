# EXP-10 统一处理后视频数据

EXP-10 的 Vript、InternVid 和 OpenVid 已切换到处理后的 conversation 导出，而非旧版
原始 metadata。正式路径和任务类型如下：

| source | metadata glob | category | 训练语义 |
|---|---|---|---|
| Vript | `open_data/Vript_caption_4k/*.jsonl` | `caption` | 视频描述 QA |
| InternVid | `open_data/InternVid_caption_4k/InternVid_caption_converted.jsonl` | `caption` | 视频描述 QA |
| OpenVid | `syn_data/OpenVid_4k_syn_grd/*.json` | `grounding` | 文本 query → 时间区间 QA |

记录必须提供稳定 `id`、`category`、`conversations`，以及顶层或嵌套的绝对媒体路径
（`video`、`video_path`、`media.path` 等）。`<|video_pad|>` 会被移除，human/gpt 对会
落成项目统一 manifest：`video/question/answer/source_dataset/source_category/source_id`。
`image_tokens`、尺寸和时长可保留为上游信息，但当前训练不直接读取。

OpenVid 的 answer 是时间区间，不得在报告中称为 caption；manifest/report 会单列
`source_category=grounding`。

## 加载与验证

- metadata 按 glob、分片流式读；伪装为 `.json` 的 JSONL 分片也会流式读；
- 不再构建视频目录 basename 索引，所有三源 `index_fallback: false`；
- 保留真实路径验证和 benchmark 去污染；构建时每 10,000 条打印进度、有效样本和
  `missing_video`，完整顺序扫描是正常线性 I/O，不是无输出卡死；
- 本地桌面不挂载 `/data`，只能验证代码契约；服务器必须先过 `audit` 和 `prep`，路径
  不可读会在训练前失败。
