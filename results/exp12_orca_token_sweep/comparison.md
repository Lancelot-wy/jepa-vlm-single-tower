# EXP-12 Orca visual-token sweep

| arm | K | mode | MVBench | TempCompass | order | direction | speed | ce_loss | centered_margin | persistence_ratio | dynamic_fraction | retrieval_top1 | retrieval_top5 | samples_per_sec | max_memory_gb | wall_time_sec | checkpoint_complete | evaluator_complete | finite_training | data_consistent | code_consistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a0_ce_k4 | 4 | none | 47.6095 | 55.3165 |  |  |  | 1.0793 |  |  |  |  |  | 34.8029 | 17.1557 | 857.1921 | True | True | True | True | True |
| a1_query_k4 | 4 | query | 47.1840 | 55.0000 |  |  |  | 1.0781 | -0.0009 | 2.7765 | 0.2704 | 0.0193 | 0.0922 | 25.9025 | 18.7192 | 1117.2852 | True | True | True | True | True |
| a2_ce_k16 | 16 | none | 52.0651 | 57.4051 |  |  |  | 1.0219 |  |  |  |  |  | 32.8501 | 17.1557 | 882.8996 | True | True | True | True | True |
| a3_query_k16 | 16 | query | 51.8148 | 57.0886 |  |  |  | 1.0224 | -0.0087 | 1.8310 | 0.2708 | 0.0079 | 0.0358 | 22.9310 | 19.6130 | 1148.1895 | True | True | True | True | True |
| a4_ce_k64 | 64 | none | 54.4681 | 60.1899 |  |  |  | 0.9840 |  |  |  |  |  | 29.9601 | 20.8725 | 962.0894 | True | True | True | True | True |
| a5_query_k64 | 64 | query | 54.3930 | 60.1899 |  |  |  | 0.9844 | -0.0369 | 1.4316 | 0.2644 | 0.0055 | 0.0213 | 21.4955 | 24.8258 | 1288.5389 | True | True | True | True | True |

## Pre-registered deltas

```json
{
  "query_minus_ce_k4": {
    "MVBench": -0.4255319148936181,
    "TempCompass": -0.31645569620251734
  },
  "query_minus_ce_k16": {
    "MVBench": -0.250312891113893,
    "TempCompass": -0.31645569620252445
  },
  "query_minus_ce_k64": {
    "MVBench": -0.07509386733416079,
    "TempCompass": 0.0
  },
  "ce_k16_minus_k4": {
    "MVBench": 4.455569461827288,
    "TempCompass": 2.088607594936711
  },
  "ce_k64_minus_k16": {
    "MVBench": 2.4030037546933656,
    "TempCompass": 2.784810126582279
  }
}
```
