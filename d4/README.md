# D4v2: Выбор минимально достаточной архитектуры FAQ-модуля AI DENTIST

**Проект:** AI DENTIST — NLP-система для стоматологических клиник
**Автор:** Дмитрий (DS / клиницист-ортопед)
**Дата начала:** 2026-03-21

---

## Исследовательский вопрос

**Какой способ предоставления знаний LLM для FAQ-модуля клиники является минимально достаточным по качеству, стоимости и сложности сопровождения: передача всей KB в prompt, lexical retrieval, vector retrieval или hybrid retrieval?**

## Гипотезы

**Основная:** lightweight hybrid retrieval (структурная фильтрация + lexical + vector) наиболее рационален, поскольку вопросы пациентов содержат одновременно точные сущности и вариативные формулировки.

**Нулевая:** hybrid не даёт practically significant улучшения относительно full-context prompting и lexical retrieval.

**Инженерная:** если retrieval необходим, его можно реализовать без тяжёлого RAG-контура.

---

## Дизайн: 4 сценария + 1 baseline

```
Query → [Context Strategy] → LLM (единый) → FAQAnswer
```

| ID | Стратегия | Контекст для LLM | Назначение |
|----|-----------|-------------------|------------|
| **S1** | Full Context | Вся KB целиком | Достаточен ли подход без retrieval? |
| **S2** | Lexical Retrieval | Секции по keyword/BM25/FTS | Достаточно ли lexical/exact retrieval? |
| **S3** | Vector Retrieval | Top-k по embedding similarity | Насколько FAQ опирается на semantic? |
| **S4** | Hybrid Retrieval | Lexical + Vector (RRF) | Даёт ли комбинация лучший баланс? |
| **B0** | Keyword + Template | — (без LLM) | Production baseline (вне inferential block) |

**Фиксировано (S1-S4):** единый LLM, system prompt, temperature, output schema (`FAQAnswer`), grounding policy.

**Предмет сравнения** — стратегии извлечения знаний, а не backend-компоненты (pgvector, BM25 engine, RAG framework).

---

## Метрики

### Качество (primary: Factual Accuracy, secondary: Unsupported Claim Rate)
- **Factual Accuracy** — корректность фактов
- **Entity Correctness** — врач, специализация, филиал, адрес, график, услуга
- **Answer Completeness** — полнота ответа (1-5)
- **Unsupported Claim Rate** — claims_not_in_kb / total_claims
- **Answerability Classification Accuracy**

### Retrieval (обязательно для S2-S4)
- Hit@k, Recall@k, MRR / nDCG, gold_chunk_in_context

### Системные
- Latency p50/p95, prompt/completion tokens, total cost, fail rate

### Эксплуатационные (экспертно)
- Сложность реализации, обновления KB, дебага, переносимости

---

## Оценка качества (3 слоя)

1. **Deterministic checks** (primary) — answerability, entity match, claim extraction
2. **Ручная экспертная валидация** — ≥ 30-50 примеров, стратифицировано
3. **LLM-as-Judge** (вторичный) — `gpt-5.4-mini`, корреляция с экспертом

---

## Eval Set

- **150 запросов** (120-180), комбинированный подход
- **≥ 30-40% ручные / real-like** формулировки
- **Классы:** врач, специализация, жалоба, услуга, адрес, график, седация, дети, оплата, документы, смешанные, unanswerable
- **Разметка:** sample_id, query, category, subtype, answerable, expected_answer, expected_doctor, expected_specialization, expected_branch, expected_service, difficulty, notes
- **Аннотатор:** стоматолог-ортопед (1 аннотатор → Limitations)

---

## Статистический анализ

- Bootstrap CI (95%, 10000 resamples)
- Paired comparisons: S1 vs S2, S1 vs S3, S1 vs S4, S2 vs S4, S3 vs S4
- Коррекция: Holm / Benjamini-Hochberg
- Friedman test (S1-S4 одновременно)
- Effect size: Cohen's d + практическая значимость
- B0 **не включается** в основной inferential block

---

## Scale Experiment (опциональная фаза)

- **Nested distractor expansion**: core KB сохраняется, добавляются distractor-блоки
- Размеры: 5K (core) → 15K → 30K → 50K токенов
- Стратегии: S1, S3, S4 × core eval set

---

## Структура проекта

```
d4/
├── README.md
├── requirements.txt
├── __init__.py
├── __main__.py
├── configs/
│   ├── experiment.yaml          # Параметры эксперимента
│   └── taxonomy.yaml            # Таксономия запросов
├── prompts/
│   ├── faq_system.md            # System prompt (единый)
│   ├── judge.md                 # Prompt для LLM-judge
│   └── query_gen.md             # Prompt для генерации eval set
├── data/
│   ├── kb/
│   │   ├── clinic_info.yaml     # Из ai-core profile
│   │   ├── doctors.yaml         # Из ai-core profile
│   │   └── chunks.json          # Логические retrieval units
│   ├── eval_set_raw.yaml
│   └── eval_set.yaml            # Финальный (150 запросов)
├── models.py                    # Pydantic модели
├── strategies/
│   ├── base.py                  # ABC: select_context() → RetrievalResult
│   ├── full_context.py          # S1
│   ├── lexical.py               # S2
│   ├── vector.py                # S3
│   ├── hybrid.py                # S4
│   └── keyword_template.py      # B0
├── pipeline/
│   ├── llm_runner.py            # Единый LLM вызов
│   ├── chunker.py               # KB → logical retrieval units
│   └── orchestrator.py          # Запуск всех стратегий
├── evaluation/
│   ├── deterministic.py         # Автоматические checks
│   ├── llm_judge.py             # LLM-as-Judge
│   ├── retrieval_metrics.py     # Hit@k, Recall@k, MRR
│   ├── metrics.py               # Агрегатные метрики
│   └── statistics.py            # Bootstrap CI, Holm, Friedman
├── data_gen/
│   ├── query_generator.py       # Генерация eval set
│   └── kb_scaler.py             # Масштабирование KB
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_run_strategies.ipynb
│   ├── 03_evaluation.ipynb
│   ├── 04_scale_analysis.ipynb  # Опциональная фаза
│   └── 05_results_report.ipynb
└── outputs/
    ├── raw_results.jsonl
    ├── judge_scores.json
    ├── manual_validation.csv
    ├── metrics_summary.csv
    ├── pairwise_tests.csv
    ├── error_table.csv
    ├── figures/
    ├── tables/
    └── reports/
```

## Запуск

```bash
pip install -r requirements.txt

# Последовательно запустить notebooks 01-05
jupyter notebook notebooks/
```
