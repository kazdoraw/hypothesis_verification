# Stage 2A Decision Memo

> **Source of truth**: `outputs/runs/20260419_235319/`

**Run ID**: `20260419_235319`  
**Timestamp**: 2026-04-20 14:13 UTC  
**Verdict**: **NEUTRAL**

---

## 1. Retrieval Comparison

| strategy   |   contextual::gold_in_context_rate |   contextual::hit_rate |   contextual::mean_recall |   contextual::mrr |   contextual::n_samples |   llm_enriched::gold_in_context_rate |   llm_enriched::hit_rate |   llm_enriched::mean_recall |   llm_enriched::mrr |   llm_enriched::n_samples |   plain::gold_in_context_rate |   plain::hit_rate |   plain::mean_recall |   plain::mrr |   plain::n_samples |   contextual::gold_in_context_rate::delta |   contextual::hit_rate::delta |   contextual::mean_recall::delta |   contextual::mrr::delta |   contextual::n_samples::delta |   llm_enriched::gold_in_context_rate::delta |   llm_enriched::hit_rate::delta |   llm_enriched::mean_recall::delta |   llm_enriched::mrr::delta |   llm_enriched::n_samples::delta |
|:-----------|-----------------------------------:|-----------------------:|--------------------------:|------------------:|------------------------:|-------------------------------------:|-------------------------:|----------------------------:|--------------------:|--------------------------:|------------------------------:|------------------:|---------------------:|-------------:|-------------------:|------------------------------------------:|------------------------------:|---------------------------------:|-------------------------:|-------------------------------:|--------------------------------------------:|--------------------------------:|-----------------------------------:|---------------------------:|---------------------------------:|
| S2         |                             0.5455 |                 0.6446 |                    0.5944 |            0.5855 |                121.0000 |                               0.5455 |                   0.6446 |                      0.5944 |              0.5855 |                  121.0000 |                        0.5455 |            0.6446 |               0.5944 |       0.5855 |           121.0000 |                                    0.0000 |                        0.0000 |                           0.0000 |                   0.0000 |                         0.0000 |                                      0.0000 |                          0.0000 |                             0.0000 |                     0.0000 |                           0.0000 |
| S3         |                             0.8678 |                 0.9339 |                    0.9056 |            0.8242 |                121.0000 |                               0.8264 |                   0.9256 |                      0.8864 |              0.8288 |                  121.0000 |                        0.8264 |            0.9504 |               0.8967 |       0.8380 |           121.0000 |                                    0.0414 |                       -0.0165 |                           0.0089 |                  -0.0138 |                         0.0000 |                                      0.0000 |                         -0.0248 |                            -0.0103 |                    -0.0092 |                           0.0000 |
| S4         |                             0.8099 |                 0.8843 |                    0.8547 |            0.7691 |                121.0000 |                               0.8347 |                   0.9091 |                      0.8795 |              0.7674 |                  121.0000 |                        0.8182 |            0.9008 |               0.8671 |       0.7712 |           121.0000 |                                   -0.0083 |                       -0.0165 |                          -0.0124 |                  -0.0021 |                         0.0000 |                                      0.0165 |                          0.0083 |                             0.0124 |                    -0.0038 |                           0.0000 |

## 2. Quality Comparison

| strategy   |   contextual::answerability_correct |   contextual::avg_fmr |   contextual::doctor_match |   contextual::doctor_total |   contextual::fmr_annotated |   contextual::n |   contextual::total_claims |   contextual::unsupported_claims |   llm_enriched::answerability_correct |   llm_enriched::avg_fmr |   llm_enriched::doctor_match |   llm_enriched::doctor_total |   llm_enriched::fmr_annotated |   llm_enriched::n |   llm_enriched::total_claims |   llm_enriched::unsupported_claims |   plain::answerability_correct |   plain::avg_fmr |   plain::doctor_match |   plain::doctor_total |   plain::fmr_annotated |   plain::n |   plain::total_claims |   plain::unsupported_claims |   contextual::answerability_rate |   contextual::doctor_match_rate |   llm_enriched::answerability_rate |   llm_enriched::doctor_match_rate |   plain::answerability_rate |   plain::doctor_match_rate |   contextual::answerability_correct::delta |   contextual::avg_fmr::delta |   contextual::doctor_match::delta |   contextual::doctor_total::delta |   contextual::fmr_annotated::delta |   contextual::n::delta |   contextual::total_claims::delta |   contextual::unsupported_claims::delta |   contextual::answerability_rate::delta |   contextual::doctor_match_rate::delta |   llm_enriched::answerability_correct::delta |   llm_enriched::avg_fmr::delta |   llm_enriched::doctor_match::delta |   llm_enriched::doctor_total::delta |   llm_enriched::fmr_annotated::delta |   llm_enriched::n::delta |   llm_enriched::total_claims::delta |   llm_enriched::unsupported_claims::delta |   llm_enriched::answerability_rate::delta |   llm_enriched::doctor_match_rate::delta |
|:-----------|------------------------------------:|----------------------:|---------------------------:|---------------------------:|----------------------------:|----------------:|---------------------------:|---------------------------------:|--------------------------------------:|------------------------:|-----------------------------:|-----------------------------:|------------------------------:|------------------:|-----------------------------:|-----------------------------------:|-------------------------------:|-----------------:|----------------------:|----------------------:|-----------------------:|-----------:|----------------------:|----------------------------:|---------------------------------:|--------------------------------:|-----------------------------------:|----------------------------------:|----------------------------:|---------------------------:|-------------------------------------------:|-----------------------------:|----------------------------------:|----------------------------------:|-----------------------------------:|-----------------------:|----------------------------------:|----------------------------------------:|----------------------------------------:|---------------------------------------:|---------------------------------------------:|-------------------------------:|------------------------------------:|------------------------------------:|-------------------------------------:|-------------------------:|------------------------------------:|------------------------------------------:|------------------------------------------:|-----------------------------------------:|
| B0         |                            117.0000 |                0.3280 |                    14.0000 |                    23.0000 |                    121.0000 |        137.0000 |                   282.0000 |                           0.0000 |                              117.0000 |                  0.3280 |                      14.0000 |                      23.0000 |                      121.0000 |          137.0000 |                     282.0000 |                             0.0000 |                       117.0000 |           0.3280 |               14.0000 |               23.0000 |               121.0000 |   137.0000 |              282.0000 |                      0.0000 |                           0.8540 |                          0.6087 |                             0.8540 |                            0.6087 |                      0.8540 |                     0.6087 |                                     0.0000 |                       0.0000 |                            0.0000 |                            0.0000 |                             0.0000 |                 0.0000 |                            0.0000 |                                  0.0000 |                                  0.0000 |                                 0.0000 |                                       0.0000 |                         0.0000 |                              0.0000 |                              0.0000 |                               0.0000 |                   0.0000 |                              0.0000 |                                    0.0000 |                                    0.0000 |                                   0.0000 |
| S1         |                            134.0000 |                0.6711 |                    23.0000 |                    23.0000 |                    121.0000 |        137.0000 |                   317.0000 |                           0.0000 |                              134.0000 |                  0.6460 |                      23.0000 |                      23.0000 |                      121.0000 |          137.0000 |                     323.0000 |                             0.0000 |                       134.0000 |           0.6562 |               23.0000 |               23.0000 |               121.0000 |   137.0000 |              319.0000 |                      0.0000 |                           0.9781 |                          1.0000 |                             0.9781 |                            1.0000 |                      0.9781 |                     1.0000 |                                     0.0000 |                       0.0149 |                            0.0000 |                            0.0000 |                             0.0000 |                 0.0000 |                           -2.0000 |                                  0.0000 |                                  0.0000 |                                 0.0000 |                                       0.0000 |                        -0.0102 |                              0.0000 |                              0.0000 |                               0.0000 |                   0.0000 |                              4.0000 |                                    0.0000 |                                    0.0000 |                                   0.0000 |
| S2         |                            111.0000 |                0.4523 |                    18.0000 |                    23.0000 |                    121.0000 |        137.0000 |                   302.0000 |                           0.0000 |                              109.0000 |                  0.4738 |                      18.0000 |                      23.0000 |                      121.0000 |          137.0000 |                     295.0000 |                             0.0000 |                       109.0000 |           0.4525 |               18.0000 |               23.0000 |               121.0000 |   137.0000 |              306.0000 |                      0.0000 |                           0.8102 |                          0.7826 |                             0.7956 |                            0.7826 |                      0.7956 |                     0.7826 |                                     2.0000 |                      -0.0002 |                            0.0000 |                            0.0000 |                             0.0000 |                 0.0000 |                           -4.0000 |                                  0.0000 |                                  0.0146 |                                 0.0000 |                                       0.0000 |                         0.0213 |                              0.0000 |                              0.0000 |                               0.0000 |                   0.0000 |                            -11.0000 |                                    0.0000 |                                    0.0000 |                                   0.0000 |
| S3         |                            132.0000 |                0.6232 |                    22.0000 |                    23.0000 |                    121.0000 |        137.0000 |                   336.0000 |                           0.0000 |                              129.0000 |                  0.6333 |                      23.0000 |                      23.0000 |                      121.0000 |          137.0000 |                     332.0000 |                             0.0000 |                       134.0000 |           0.6190 |               22.0000 |               23.0000 |               121.0000 |   137.0000 |              327.0000 |                      0.0000 |                           0.9635 |                          0.9565 |                             0.9416 |                            1.0000 |                      0.9781 |                     0.9565 |                                    -2.0000 |                       0.0042 |                            0.0000 |                            0.0000 |                             0.0000 |                 0.0000 |                            9.0000 |                                  0.0000 |                                 -0.0146 |                                 0.0000 |                                      -5.0000 |                         0.0143 |                              1.0000 |                              0.0000 |                               0.0000 |                   0.0000 |                              5.0000 |                                    0.0000 |                                   -0.0365 |                                   0.0435 |
| S4         |                            128.0000 |                0.6182 |                    21.0000 |                    23.0000 |                    121.0000 |        137.0000 |                   314.0000 |                           0.0000 |                              128.0000 |                  0.6350 |                      22.0000 |                      23.0000 |                      121.0000 |          137.0000 |                     327.0000 |                             0.0000 |                       130.0000 |           0.6076 |               21.0000 |               23.0000 |               121.0000 |   137.0000 |              307.0000 |                      0.0000 |                           0.9343 |                          0.9130 |                             0.9343 |                            0.9565 |                      0.9489 |                     0.9130 |                                    -2.0000 |                       0.0106 |                            0.0000 |                            0.0000 |                             0.0000 |                 0.0000 |                            7.0000 |                                  0.0000 |                                 -0.0146 |                                 0.0000 |                                      -2.0000 |                         0.0274 |                              1.0000 |                              0.0000 |                               0.0000 |                   0.0000 |                             20.0000 |                                    0.0000 |                                   -0.0146 |                                   0.0435 |

## 3. Statistical Significance (per-sample, paired)

Paired bootstrap-CI и тесты по per-sample артефактам прогона (`report.json` → `rank_analysis`, `*.jsonl`, eval_set gold). FMR здесь **не** представлен: `save_judge_scores` в pipeline пока не вызывается, per-sample judge-оценки не сохранены (см. `d4/analysis/significance.py`).

★ рядом со строкой — CI парной разности не пересекает 0 (слабый прокси значимости, без поправок на множественные сравнения).

### mrr  (n=121)

| mode | mean | CI mean | Δ vs baseline | CI Δ | p | test | sig |
|---|---|---|---|---|---|---|---|
| `plain` (baseline) | 0.8380 | [+0.7777, +0.8880] | — | — | — | — | — |
| `contextual` | 0.8242 | [+0.7625, +0.8774] | -0.0138 | [-0.0416, +0.0124] | 0.303 | wilcoxon |  |
| `llm_enriched` | 0.8288 | [+0.7689, +0.8822] | -0.0092 | [-0.0479, +0.0293] | 0.557 | wilcoxon |  |

### hit_rate  (n=121)

| mode | mean | CI mean | Δ vs baseline | CI Δ | p | test | sig |
|---|---|---|---|---|---|---|---|
| `plain` (baseline) | 0.9504 | [+0.9091, +0.9835] | — | — | — | — | — |
| `contextual` | 0.9339 | [+0.8843, +0.9752] | -0.0165 | [-0.0496, +0.0165] | 0.625 | mcnemar |  |
| `llm_enriched` | 0.9256 | [+0.8760, +0.9669] | -0.0248 | [-0.0661, +0.0165] | 0.453 | mcnemar |  |

### answerability_correct  (n=137)

| mode | mean | CI mean | Δ vs baseline | CI Δ | p | test | sig |
|---|---|---|---|---|---|---|---|
| `plain` (baseline) | 0.9781 | [+0.9489, +1.0000] | — | — | — | — | — |
| `contextual` | 0.9635 | [+0.9270, +0.9927] | -0.0146 | [-0.0365, +0.0000] | 0.500 | mcnemar |  |
| `llm_enriched` | 0.9416 | [+0.8978, +0.9781] | -0.0365 | [-0.0730, -0.0073] | 0.062 | mcnemar | ★ |

### doctor_match  (n=23)

| mode | mean | CI mean | Δ vs baseline | CI Δ | p | test | sig |
|---|---|---|---|---|---|---|---|
| `plain` (baseline) | 0.9565 | [+0.8696, +1.0000] | — | — | — | — | — |
| `contextual` | 0.9565 | [+0.8696, +1.0000] | +0.0000 | [+0.0000, +0.0000] | 1.000 | mcnemar |  |
| `llm_enriched` | 1.0000 | [+1.0000, +1.0000] | +0.0435 | [+0.0000, +0.1304] | 1.000 | mcnemar |  |

## 4. Decision Gate

**Verdict**: `neutral`  
**Recommendation**: FMR немного подрос, но retrieval просел — сигнал противоречивый. Без bootstrap-CI / paired-теста выигрыш FMR может быть шумом. Прогоните hard и blind eval-сеты, чтобы исключить переобучение под dev-распределение.

### Signals

- contextual: MRR drop -0.014 ≤ порог
- llm_enriched: hit_rate drop -0.025 ≤ порог
- llm_enriched: FMR improvement +0.014

## 5. Figures

![retrieval_comparison](figures/retrieval_comparison.png)
![quality_comparison](figures/quality_comparison.png)
![mode_deltas](figures/mode_deltas.png)
![rank_shift](figures/rank_shift.png)

## 6. Artifact Index

| Категория | Файл |
|-----------|------|
| meta | `config_snapshot.yaml` |
| figure | `figures/mode_deltas.png` |
| figure | `figures/quality_comparison.png` |
| figure | `figures/rank_shift.png` |
| figure | `figures/retrieval_comparison.png` |
| result | `full_dev_contextual.jsonl` |
| report | `full_dev_contextual.report.json` |
| result | `full_dev_llm_enriched.jsonl` |
| report | `full_dev_llm_enriched.report.json` |
| result | `full_dev_plain.jsonl` |
| report | `full_dev_plain.report.json` |
| meta | `manifest.json` |
| report | `reports/stage2a_decision_memo.md` |
