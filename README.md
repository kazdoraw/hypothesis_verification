# Medical AI AGENT — Hypothesis Verification Experiments

> Прикладные эксперименты по выбору технических подходов для модульного медицинского AI-агента (стоматология). Три независимые гипотезы — D1, D2, D4 — каждая отвечает за свой контур обработки текстового обращения пациента.

**Languages / Языки:** [Русский](#русский) · [English](#english)

---

## Русский

### 1. О проекте

Этот репозиторий — набор исследовательских экспериментов, в которых проверяются ключевые архитектурные решения для медицинского AI-агента, обрабатывающего входящий текстовый поток в стоматологическую клинику (мессенджеры, виджет на сайте, социальные сети). Идея проекта — **модульная декомпозиция вместо монолитного LLM-вызова**: каждое направление обработки решает специализированный контур (классификатор, схема извлечения данных, retrieval-стек), и каждое такое решение принимается на основе явно сформулированной гипотезы и измерений.

Сквозной исследовательский вопрос:

> Можно ли заменить или ограничить дорогие LLM-вызовы лёгкими специализированными контурами без потери качества и клинической безопасности?

Внутри проекта это раскладывается в три проверяемые гипотезы.

### 2. Эксперименты

| ID | Тема | Что проверяли | Ключевой результат | Документация |
|----|------|---------------|--------------------|--------------|
| **D1** | Маршрутизация обращений | Может ли лёгкий ML-классификатор (TF-IDF + LR / BGE-M3 + SVC) безопасно заменить часть LLM-роутинга простых обращений (`faq` / `booking` / `anamnesis` / `unsupported`) при сохранении медицинской recall на симптомах. | На синтетическом корпусе ~3.3K сообщений `SimpleRouter` принимает ~51% обращений с `accepted_accuracy = 0.963` и `accepted_recall_anamnesis = 0.993` на основном тесте. Сложные/неоднозначные кейсы корректно уходят в `defer`. | [d1/EXPERIMENT_D1.md](d1/EXPERIMENT_D1.md), [d1/README.md](d1/README.md) |
| **D2** | Структурированный сбор анамнеза | Как степень структурированности JSON-схемы извлечения (фиксированная S1 / адаптивная S2 / свободная S3) влияет на точность последующей маршрутизации к специалисту. Стенд: «модель-врач ↔ модель-пациент ↔ независимая модель-судья». | На 10 клинических сценариях S1 даёт 6/10 верных маршрутизаций, S2 — 9/10, S3 — 10/10. Адаптивная схема S2 — наиболее рациональный кандидат для продакшена (баланс качества и стоимости токенов). | [d2/EXPERIMENT_D2.md](d2/EXPERIMENT_D2.md) |
| **D4** | Поиск ответа в базе знаний клиники (RAG для FAQ) | Какая стратегия поиска и какое представление карточек знаний дают лучший компромисс «качество ответа / сложность» для модуля FAQ. Сравнивались 7 retrieval-стратегий и 3 представления чанков. | На полном валидационном наборе из 137 вопросов простой векторный поиск (`S3` поверх `C0`) — 134/137 корректных ответов или отказов (97.8%), 22/23 правильных врачей (95.7%), 62% обязательных фактов в ответе. Усложнения (гибрид, реранкер, обогащение чанков) **не дали** устойчивого выигрыша. | [d4/EXPERIMENT_D4.md](d4/EXPERIMENT_D4.md) |

Каждая гипотеза проверена изолированно, но связана с одним продуктовым контуром: D1 решает «что делать с сообщением», D2 — «какие данные собрать перед маршрутизацией к специалисту», D4 — «как ответить на информационный запрос по конкретной клинике».

### 3. Структура репозитория

```text
study/
├── README.md                  ← этот файл
├── requirements.txt           ← общие зависимости проекта
│
├── d1/                        ← Эксперимент D1: маршрутизация обращений
│   ├── EXPERIMENT_D1.md       ← полное описание эксперимента
│   ├── README.md              ← операционное руководство
│   ├── D1_domain_router_v6.ipynb
│   ├── baselines/             ← модели и роутеры (B0..B2.5, SelectiveRouter,
│   │                            B4HybridRouter, SimpleRouter, ComplexityGate)
│   ├── scripts/               ← runners (train, eval, sweep, latency, plots)
│   ├── data/                  ← синтетические splits (train/val/test/hard/blind/...)
│   ├── results/               ← CSV/JSON метрики, PNG-фигуры
│   ├── prompts/, ontology/, tests/
│   └── requirements.txt
│
├── d2/                        ← Эксперимент D2: схема сбора анамнеза
│   ├── EXPERIMENT_D2.md
│   ├── run.py                 ← CLI-оркестратор
│   ├── schemas.py             ← S1 / S2 / S3 (независимая переменная)
│   ├── doctor.py, patient.py, judge.py, session.py
│   ├── cases.py               ← 10 клинических сценариев + reference_routing
│   ├── prompts/               ← doctor, patient, judge, routing_infer
│   └── results/               ← диалоги (case_*.json), оценки судьи, отчёты, фигуры
│
└── d4/                        ← Эксперимент D4: RAG для FAQ
    ├── EXPERIMENT_D4.md
    ├── notebooks/             ← d4_stage1_screening, d4_stage2a_representation
    ├── strategies/            ← lexical, vector, hybrid, hybrid_rerank, tiered, ...
    ├── pipeline/              ← chunker, enrichment, llm_runner, orchestrator
    ├── analysis/              ← loaders, plots, reporting, significance, ...
    ├── evaluation/            ← retrieval / deterministic / nli_checker / llm_judge
    ├── data_gen/              ← генерация и парсинг базы знаний
    ├── raw_data/kb/           ← база знаний клиники (прайс, врачи, рекомендации)
    ├── outputs/runs/{run_id}/ ← версионированные результаты прогонов
    ├── configs/, prompts/, tests/
    └── requirements.txt
```

### 4. Установка

Требования: Python 3.12+ (рекомендуется 3.13), macOS arm64 / Linux x86_64.

```bash
git clone https://github.com/kazdoraw/hypothesis_verification.git
cd hypothesis_verification

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# Для D1/D4 при необходимости можно установить локальные requirements:
# pip install -r d1/requirements.txt
# pip install -r d4/requirements.txt
```

Для экспериментов, которые делают вызовы внешних LLM (D2 целиком, D4 частично, D1 только при пересборке датасета), потребуется ключ OpenRouter:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

### 5. Воспроизведение

| Эксперимент | Точка входа | Что произойдёт |
|-------------|-------------|----------------|
| D1 | `jupyter notebook d1/D1_domain_router_v6.ipynb` или `python -m d1.scripts.run_baselines` | Тренировка 11 baseline-моделей, расчёт closed-set / selective / hybrid / SimpleRouter метрик, bootstrap CI, paired-тесты, генерация фигур. |
| D2 | `python -m d2.run --cases all` | 10 кейсов × 3 схемы извлечения = 30 прогонов «врач ↔ пациент». `--judge` — оценка судьёй. `--report-only` — пересборка отчёта без LLM-вызовов. |
| D4 | `jupyter notebook d4/notebooks/d4_stage1_screening.ipynb` или `d4_stage2a_representation.ipynb` | Можно перезапустить прогон LLM (`run_new`) или пересобрать аналитику и фигуры из ранее сохранённого `outputs/runs/{run_id}/` (`analyze_existing`). |

Тесты:

```bash
python -m pytest d1/tests -q
python -m pytest d4/tests -q
```

> Бинарные артефакты (обученные модели `*.joblib`/`*.bin`, выгрузки прогонов `*.jsonl`, кэши обогащения) не хранятся в репозитории. Они автоматически пересобираются при запуске. CSV-метрики, JSON-отчёты и PNG-фигуры — закоммичены, чтобы результаты можно было просмотреть без выполнения кода.
>
> Сырая база знаний клиники (`d4/raw_data/kb/`: прайс, врачи, контакты, рекомендации в `.docx`) **не публикуется** в репозитории. Файл `d4/data/chunks_frozen.json` (уже разобранные чанки) остаётся под git и достаточен для режима `analyze_existing` в notebook'ах D4. Чтобы перезапустить retrieval с нуля, нужно подложить собственную `d4/raw_data/kb/` с такой же структурой (см. `d4/configs/experiment.yaml`).

### 6. Стек технологий

- **Python:** 3.12+
- **Классические ML-модели:** scikit-learn (LogisticRegression, LinearSVC, GroupShuffleSplit), fastText (sub-word).
- **Энкодеры (dense retrieval / классификация):** `BAAI/bge-m3`, `intfloat/multilingual-e5-small` через `sentence-transformers`.
- **LLM-вызовы:** OpenRouter (Qwen3, Grok-4.1, GPT-5.x, Llama 3.3) — единый API для генерации, диалогов и судейства.
- **Статистика:** bootstrap 95% CI (BCa), paired McNemar / paired bootstrap.
- **Эксперименты и отчёты:** Jupyter, pandas, matplotlib, seaborn.
- **Качество кода:** pytest, конфигурируемые pipeline-флаги, версионированные `outputs/runs/{run_id}/` с `config_snapshot.yaml` и `manifest.json`.

### 7. Ограничения и оговорки

- Все три корпуса — синтетические или частично синтетические; результаты переносятся на реальный поток обращений только через отдельный пилот.
- D1 и D2 проверяют только первое сообщение / однократный диалог; multi-turn-сценарии вне scope.
- D4 проверен на базе знаний одной клиники (63 карточки, 137 вопросов); генерализация на другие клиники и медицинские домены требует отдельной валидации.
- Бенчмарки латентности — micro-benchmark на CPU; production-SLA нужно мерить отдельно с учётом cold start, очередей и кэширования.
- Эксперименты не доказывают коммерческой эффективности продукта — они обосновывают **архитектурный выбор** для модульного медицинского AI-агента.

### 8. Лицензия

Проект публикуется в исследовательских целях. Все клинические сценарии и сообщения являются синтетическими и не содержат персональных данных реальных пациентов.

---

## English

### 1. About the project

This repository is a set of research experiments verifying key architectural choices for a medical AI agent that processes incoming text traffic for a dental clinic (messengers, web widget, social networks). The guiding idea is **modular decomposition instead of a monolithic LLM call**: each processing direction is handled by a specialized contour (classifier, extraction schema, retrieval stack), and every such decision is grounded in an explicit, measured hypothesis.

Cross-cutting research question:

> Can we replace or constrain expensive LLM calls with lightweight specialized contours without sacrificing quality or clinical safety?

The project breaks this question into three testable hypotheses.

### 2. Experiments

| ID | Topic | What was tested | Headline result | Documentation |
|----|-------|------------------|------------------|---------------|
| **D1** | Intent / domain routing | Whether a lightweight ML classifier (TF-IDF + LR / BGE-M3 + SVC) can safely replace part of LLM-based routing for simple messages (`faq` / `booking` / `anamnesis` / `unsupported`) while preserving medical recall on symptoms. | On a synthetic corpus of ~3.3K messages, `SimpleRouter` accepts ~51% of incoming traffic with `accepted_accuracy = 0.963` and `accepted_recall_anamnesis = 0.993` on the main test set. Hard / ambiguous cases are correctly sent to `defer`. | [d1/EXPERIMENT_D1.md](d1/EXPERIMENT_D1.md), [d1/README.md](d1/README.md) |
| **D2** | Structured intake schema | How the rigidity of the JSON extraction schema (fixed S1 / adaptive S2 / free S3) affects downstream specialist routing. Test bench: «doctor model ↔ patient model ↔ independent judge model». | Across 10 clinical scenarios S1 yields 6/10 correct routings, S2 — 9/10, S3 — 10/10. The adaptive schema S2 is the most rational production candidate (best balance of quality and token cost). | [d2/EXPERIMENT_D2.md](d2/EXPERIMENT_D2.md) |
| **D4** | FAQ retrieval over the clinic's knowledge base | Which retrieval strategy and which chunk representation deliver the best «answer quality / system complexity» trade-off for the FAQ module. 7 retrieval strategies × 3 chunk representations were compared. | On a 137-question validation set, plain dense retrieval (`S3` over `C0`) wins: 134/137 correct answers or refusals (97.8%), 22/23 correct doctor mentions (95.7%), 62% of gold facts present in the answer. More complex variants (hybrid, reranker, chunk enrichment) did **not** show a stable gain. | [d4/EXPERIMENT_D4.md](d4/EXPERIMENT_D4.md) |

Each hypothesis is verified in isolation but ties into a single product flow: D1 decides «what to do with the message», D2 decides «which data to collect before specialist routing», D4 decides «how to answer informational queries about a specific clinic».

### 3. Repository layout

```text
study/
├── README.md                  ← this file
├── requirements.txt           ← shared project dependencies
│
├── d1/                        ← Experiment D1: intent routing
│   ├── EXPERIMENT_D1.md       ← full experiment write-up
│   ├── README.md              ← operational guide
│   ├── D1_domain_router_v6.ipynb
│   ├── baselines/             ← models and routers (B0..B2.5, SelectiveRouter,
│   │                            B4HybridRouter, SimpleRouter, ComplexityGate)
│   ├── scripts/               ← runners (train, eval, sweep, latency, plots)
│   ├── data/                  ← synthetic splits (train/val/test/hard/blind/...)
│   ├── results/               ← CSV/JSON metrics, PNG figures
│   ├── prompts/, ontology/, tests/
│   └── requirements.txt
│
├── d2/                        ← Experiment D2: intake schema
│   ├── EXPERIMENT_D2.md
│   ├── run.py                 ← CLI orchestrator
│   ├── schemas.py             ← S1 / S2 / S3 (the independent variable)
│   ├── doctor.py, patient.py, judge.py, session.py
│   ├── cases.py               ← 10 clinical scenarios + reference_routing
│   ├── prompts/               ← doctor, patient, judge, routing_infer
│   └── results/               ← dialogs (case_*.json), judge scores, reports, figures
│
└── d4/                        ← Experiment D4: FAQ RAG
    ├── EXPERIMENT_D4.md
    ├── notebooks/             ← d4_stage1_screening, d4_stage2a_representation
    ├── strategies/            ← lexical, vector, hybrid, hybrid_rerank, tiered, ...
    ├── pipeline/              ← chunker, enrichment, llm_runner, orchestrator
    ├── analysis/              ← loaders, plots, reporting, significance, ...
    ├── evaluation/            ← retrieval / deterministic / nli_checker / llm_judge
    ├── data_gen/              ← knowledge-base generation and parsing
    ├── raw_data/kb/           ← clinic knowledge base (price list, doctors, aftercare)
    ├── outputs/runs/{run_id}/ ← versioned per-run results
    ├── configs/, prompts/, tests/
    └── requirements.txt
```

### 4. Installation

Requirements: Python 3.12+ (3.13 recommended), macOS arm64 / Linux x86_64.

```bash
git clone https://github.com/kazdoraw/hypothesis_verification.git
cd hypothesis_verification

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# For D1 / D4 you can additionally install local requirements:
# pip install -r d1/requirements.txt
# pip install -r d4/requirements.txt
```

Experiments that call external LLMs (D2 fully, D4 partially, D1 only when regenerating the dataset) require an OpenRouter key:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

### 5. Reproduction

| Experiment | Entry point | What happens |
|------------|-------------|--------------|
| D1 | `jupyter notebook d1/D1_domain_router_v6.ipynb` or `python -m d1.scripts.run_baselines` | Trains 11 baseline models, computes closed-set / selective / hybrid / SimpleRouter metrics, bootstrap CIs, paired tests and renders all figures. |
| D2 | `python -m d2.run --cases all` | 10 cases × 3 schemas = 30 doctor-patient dialogs. `--judge` runs the judge model. `--report-only` rebuilds reports without any LLM calls. |
| D4 | `jupyter notebook d4/notebooks/d4_stage1_screening.ipynb` or `d4_stage2a_representation.ipynb` | Either rerun the LLM (`run_new`) or rebuild analytics and figures from a previously saved `outputs/runs/{run_id}/` (`analyze_existing`). |

Tests:

```bash
python -m pytest d1/tests -q
python -m pytest d4/tests -q
```

> Binary artifacts (trained models `*.joblib`/`*.bin`, per-run LLM dumps `*.jsonl`, enrichment caches) are **not** stored in the repository. They are regenerated on demand. CSV metrics, JSON reports and PNG figures **are** committed so that results can be inspected without rerunning anything.
>
> The raw clinic knowledge base (`d4/raw_data/kb/`: price list, doctors, contacts, aftercare `.docx`) is **not published** in the repository. The `d4/data/chunks_frozen.json` file (pre-built chunks) is committed and is sufficient for the `analyze_existing` mode in the D4 notebooks. To rerun retrieval from scratch, drop your own `d4/raw_data/kb/` with the same layout (see `d4/configs/experiment.yaml`).

### 6. Tech stack

- **Python:** 3.12+
- **Classical ML:** scikit-learn (LogisticRegression, LinearSVC, GroupShuffleSplit), fastText (sub-word).
- **Encoders (dense retrieval / classification):** `BAAI/bge-m3`, `intfloat/multilingual-e5-small` via `sentence-transformers`.
- **LLM calls:** OpenRouter (Qwen3, Grok-4.1, GPT-5.x, Llama 3.3) — single API for generation, dialog and judging.
- **Statistics:** bootstrap 95% CI (BCa), paired McNemar / paired bootstrap.
- **Experiment plumbing:** Jupyter, pandas, matplotlib, seaborn.
- **Quality:** pytest, configurable pipeline flags, versioned `outputs/runs/{run_id}/` with `config_snapshot.yaml` and `manifest.json`.

### 7. Caveats and limitations

- All three corpora are synthetic or partially synthetic; results transfer to real traffic only through a dedicated pilot.
- D1 and D2 only address the first message / single dialog; multi-turn scenarios are out of scope.
- D4 was validated on the knowledge base of a single clinic (63 cards, 137 questions); generalization to other clinics and medical domains requires separate validation.
- Latency benchmarks are CPU micro-benchmarks; production SLAs must be measured separately, accounting for cold start, queues and caching.
- The experiments do not prove commercial viability of any product — they justify **architectural choices** for a modular medical AI agent.

### 8. License

Published for research purposes. All clinical scenarios and messages are synthetic and contain no personal data of real patients.
