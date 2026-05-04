# Stage 1 Screening Decision Memo

> **Source of truth**: `outputs/runs/20260419_235319/`

**Run ID**: `20260419_235319`  
**Timestamp**: 2026-05-04 16:05 UTC  
**Mode (representation)**: `plain`

**Eval samples**: `137`  ·  **Elapsed**: `728.2s`

---

## 1. Retrieval Comparison (mode = `plain`)

| strategy   |   plain::gold_in_context_rate |   plain::hit_rate |   plain::mean_recall |   plain::mrr |   plain::n_samples |
|:-----------|------------------------------:|------------------:|---------------------:|-------------:|-------------------:|
| S2         |                        0.5455 |            0.6446 |               0.5944 |       0.5855 |           121.0000 |
| S3         |                        0.8264 |            0.9504 |               0.8967 |       0.8380 |           121.0000 |
| S4         |                        0.8182 |            0.9008 |               0.8671 |       0.7712 |           121.0000 |

## 2. Quality Comparison (mode = `plain`)

| strategy   |   plain::answerability_correct |   plain::avg_fmr |   plain::doctor_match |   plain::doctor_total |   plain::fmr_annotated |   plain::n |   plain::total_claims |   plain::unsupported_claims |   plain::answerability_rate |   plain::doctor_match_rate |
|:-----------|-------------------------------:|-----------------:|----------------------:|----------------------:|-----------------------:|-----------:|----------------------:|----------------------------:|----------------------------:|---------------------------:|
| B0         |                       117.0000 |           0.3280 |               14.0000 |               23.0000 |               121.0000 |   137.0000 |              282.0000 |                      0.0000 |                      0.8540 |                     0.6087 |
| S1         |                       134.0000 |           0.6562 |               23.0000 |               23.0000 |               121.0000 |   137.0000 |              319.0000 |                      0.0000 |                      0.9781 |                     1.0000 |
| S2         |                       109.0000 |           0.4525 |               18.0000 |               23.0000 |               121.0000 |   137.0000 |              306.0000 |                      0.0000 |                      0.7956 |                     0.7826 |
| S3         |                       134.0000 |           0.6190 |               22.0000 |               23.0000 |               121.0000 |   137.0000 |              327.0000 |                      0.0000 |                      0.9781 |                     0.9565 |
| S4         |                       130.0000 |           0.6076 |               21.0000 |               23.0000 |               121.0000 |   137.0000 |              307.0000 |                      0.0000 |                      0.9489 |                     0.9130 |

## 3. Strategy Ranking

| strategy   | Answerability Rate   | Doctor Match Rate   | Fact Match Rate   | MRR         | Mean Recall@5   | hit@5       |
|:-----------|:---------------------|:--------------------|:------------------|:------------|:----------------|:------------|
| S2         | 0.7956 (#5)          | 0.7826 (#4)         | 0.4525 (#4)       | 0.5855 (#3) | 0.5944 (#3)     | 0.6446 (#3) |
| S3         | 0.9781 (#1)          | 0.9565 (#2)         | 0.6190 (#2)       | 0.8380 (#1) | 0.8967 (#1)     | 0.9504 (#1) |
| S4         | 0.9489 (#3)          | 0.9130 (#3)         | 0.6076 (#3)       | 0.7712 (#2) | 0.8671 (#2)     | 0.9008 (#2) |
| S1         | 0.9781 (#1)          | 1.0000 (#1)         | 0.6562 (#1)       | —           | —               | —           |
| B0         | 0.8540 (#4)          | 0.6087 (#5)         | 0.3280 (#5)       | —           | —               | —           |

_Каждая ячейка: значение метрики и её ранг (`#1` — лучшая стратегия по этой метрике; для всех текущих Stage 1-метрик «больше — лучше»)._

## 4. Stage 1 Verdict

Лидер retrieval: **S3** (3 первое место(а) среди 3 retrieval-метрик). Лидер по FMR среди retrieval: **S3 = 0.6190**. Upper-bound по FMR: **S1 = 0.6562** (не retrieval-стратегия — даёт LLM весь контекст). Слабейшая retrieval-стратегия по FMR: **S2 = 0.4525**.

- **Лидер retrieval (по числу первых мест)**: `S3`.
- **Лидер по FMR среди retrieval-стратегий**: `S3` = `0.6190`.
- **Upper-bound по FMR (overall)**: `S1` = `0.6562`. Если это `S1` — это не retrieval-стратегия, а полный контекст.
- **Слабейшая retrieval-стратегия по FMR**: `S2` = `0.4525`.

## 5. Limitations

- Метрики получены на режиме `plain` (Stage 1 фиксирует представление чанков).
  Эффект различных представлений измеряется в Stage 2A.
- Decision Gate (`evaluate_gate`) не применяется в Stage 1: он рассчитан на сравнение режимов представления (delta-колонок), а в single-mode прогоне их нет.
- Per-sample `FMR` не сохраняется (`save_judge_scores` не вызывается в pipeline) — paired significance по FMR недоступен.
- На текущей итерации не прогонялись `hard` / `blind` eval-сеты, поэтому stress-валидация ранжирования стратегий вне `dev_v2` остаётся открытым шагом.

## 6. Figures

![stage1_retrieval_strategies](figures/stage1_retrieval_strategies.png)
![stage1_quality_strategies](figures/stage1_quality_strategies.png)

## 7. Artifact Index

| Категория | Файл |
|-----------|------|
| meta | `config_snapshot.yaml` |
| figure | `figures/mode_deltas.png` |
| figure | `figures/quality_comparison.png` |
| figure | `figures/rank_shift.png` |
| figure | `figures/retrieval_comparison.png` |
| figure | `figures/stage1_quality_strategies.png` |
| figure | `figures/stage1_retrieval_strategies.png` |
| result | `full_dev_contextual.jsonl` |
| report | `full_dev_contextual.report.json` |
| result | `full_dev_llm_enriched.jsonl` |
| report | `full_dev_llm_enriched.report.json` |
| result | `full_dev_plain.jsonl` |
| report | `full_dev_plain.report.json` |
| meta | `manifest.json` |
| report | `reports/stage1_decision_memo.md` |
| report | `reports/stage2a_decision_memo.md` |
