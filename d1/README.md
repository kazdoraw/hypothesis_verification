# D1: Эксперимент по выбору лёгкого классификатора intent для AI-ядра

> Экспериментальный модуль исследования в рамках выпускной квалификационной
> работы. Цель — эмпирически выбрать модель классификации намерения
> пользователя, пригодную к встраиванию в production-каскад AI-ядра проекта
> `med-agent/ai-core` для снижения количества обращений к LLM на стадии
> маршрутизации диалога.

---

## Содержание

1. [Контекст и цель исследования](#1-контекст-и-цель-исследования)
2. [Исследовательские вопросы и гипотезы](#2-исследовательские-вопросы-и-гипотезы)
3. [Методология](#3-методология)
4. [Корпус данных](#4-корпус-данных)
5. [Сравниваемые модели](#5-сравниваемые-модели)
6. [Метрики и режимы оценки](#6-метрики-и-режимы-оценки)
7. [Дополнительные метрики (калибровка, латентность, статистика)](#7-дополнительные-метрики-калибровка-латентность-статистика)
8. [Экспериментальный дизайн](#8-экспериментальный-дизайн)
9. [Структура директории `d1/`](#9-структура-директории-d1)
10. [Воспроизведение эксперимента](#10-воспроизведение-эксперимента)
11. [Артефакты результатов](#11-артефакты-результатов)
12. [Тестирование](#12-тестирование)
13. [Аугментация данных через LLM](#13-аугментация-данных-через-llm)
14. [Известные ограничения](#14-известные-ограничения)
15. [Ссылки на компоненты проекта](#15-ссылки-на-компоненты-проекта)

---

## 1. Контекст и цель исследования

### 1.1. Проектный контекст

Проект `med-agent/ai-core` реализует диалогового медицинского ассистента для
стоматологических клиник на основе LangGraph. Архитектура графа включает
следующие узлы:

```
START → ROUTER → [FAQ | BOOKING | ANAMNESIS] → FINALIZE → END
```

В текущей реализации ROUTER — это вызов большой языковой модели (LLM), которая
определяет следующий доменный workflow. Каждый такой вызов несёт стоимость
(API tokens) и задержку (network round-trip + decoding). При высоком объёме
диалогов это создаёт ощутимое финансовое и техническое давление на систему.

### 1.2. Цель эксперимента

Эмпирически выбрать модель закрытой классификации интента, пригодную к
встраиванию в production-каскад AI-ядра вместо LLM-роутера. Сравнение
ведётся в одинаковых условиях (closed-set, top-1 предсказание, без режима
отказа) между пятью baseline-моделями разных архитектур:

1. **Безопасность** (recall_urgent на urgent/emergency safety_set) —
   приоритет над общим accuracy.
2. **Качество** (macro_F1 / balanced_accuracy на in-distribution test,
   hard_test, blind_test, entity_held_out, extended_eval) — сравнение
   между моделями попарно через bootstrap.
3. **Латентность** (encode + predict, per-text median + p95) — production-aligned
   замер на CPU.
4. **Sample-efficiency** (learning curves на 10/25/50/75/100% train) —
   достаточно ли текущего объёма train для plateau качества.

### 1.3. Производственная значимость

Положительный результат эксперимента позволяет внедрить ML-роутер в
production-цепочку AI-ядра, сократив количество LLM-вызовов и, соответственно,
операционные расходы и задержки. Отрицательный результат столь же ценен:
он формализует нижнюю границу применимости лёгких классификаторов в
медицинских диалоговых системах с дисбалансным распределением намерений.

---

## 2. Исследовательские вопросы и гипотезы

### 2.1. Research Questions

**RQ1 — Domain Routing.** Возможно ли по одному пользовательскому сообщению
определить доменный workflow `{anamnesis, faq, booking, unsupported}` с
качеством, достаточным для replacing-ready production-ноды без LLM?

**RQ2 — Architecture trade-off.** Какая архитектура (rule-based, sparse,
dense) даёт лучший trade-off по trio (safety × quality × latency)?

**RQ3 — Intent Switch.** Сохраняется ли качество классификации при
наличии активного предыдущего домена (стресс-тест на `switch_test`,
text-only)?

**RQ4 — Robustness.** Как модель ведёт себя на коротких репликах, шуме,
смешанных запросах, unseen entities и out-of-distribution фразировках?

### 2.2. Гипотезы

**H1 (основная).** Существует хотя бы одна лёгкая closed-set модель из
семейств TF-IDF, fastText или dense-embedding, достигающая macro_F1 ≥ 0.85
на test и recall_urgent ≥ 0.90 на safety_set без режима отказа.

**H2 (sample-efficiency).** На 10–25% train модели достигают не менее 70%
от plateau-качества — это критично для применимости при ограниченной
разметке.

**H3 (intent switch).** Модели достигают route_accuracy ≥ 0.90 на
switch_test без специальной адаптации под контекст предыдущего домена.

---

## 3. Методология

### 3.1. Формулировка задачи

Дано: текстовое сообщение пользователя $x \in \mathcal{X}$, представленное
последовательностью токенов на русском языке.

Требуется: closed-set классификатор $f: \mathcal{X} \rightarrow \mathcal{Y}$,
где
$\mathcal{Y} = \{\text{anamnesis}, \text{faq}, \text{booking}, \text{unsupported}\}$.
Модель обязана вернуть ровно одну метку для каждого сообщения
(без режима отказа); сравнение архитектур ведётся «честно» по top-1
предсказанию.

### 3.2. Целевые классы

| Класс | Семантика |
|---|---|
| `anamnesis` | Жалобы, симптомы, описание состояния — требуется сбор анамнеза |
| `faq` | Вопросы об услугах, ценах, расписании, правилах клиники |
| `booking` | Намерение записаться на приём (явное или неявное) |
| `unsupported` | Off-topic, нерелевантные сообщения, спам |

### 3.3. Дополнительные поля разметки

Для каждого сообщения фиксируются вспомогательные routing-флаги:

- `subtype` — уточнение внутри домена (`symptom`, `complaint`,
  `new_appointment`, `clinic_info`, `service_info`, `pricing`, ...);
- `urgency ∈ {normal, urgent, emergency}` — клиническая срочность;
- `explicit_booking: bool` — явный запрос на запись;
- `is_offtopic: bool` — нерелевантность медицинской теме;
- `specialization_hint: Optional[str]` — упомянутая специализация;
- `feedback_flag: bool` — упоминание оценок или жалоб на клинику;
- `faq_category: str` — категория FAQ (`policy_only`, `subjective`, ...).

Все поля нормализованы по таксономии `d1/ontology/route_domain.yaml`
(SSoT — Single Source of Truth).

---

## 4. Корпус данных

### 4.1. Источники

Корпус формируется из трёх независимых источников:

1. **Seeds** (`d1/data/d1_v6_seeds.yaml`) — эталонные размеченные вручную
   сообщения, включающие 415 уникальных смысловых единиц (`seed_id`).
   Каждый seed содержит canonical-формулировку и метаданные (домен, subtype,
   срочность, теги).
2. **Hard cases** (`d1/data/d1_v6_hard_cases.yaml`) — тщательно сконструированные
   сложные сообщения, нацеленные на стресс-тестирование роутера (mixed-intent,
   short ambiguous, doctor-name confusion, post-treatment complications,
   pediatric trauma и др.).
3. **LLM-аугментация** (Grok-4.1-fast через OpenRouter) — синтетические
   вариации каждого seed (10 вариаций по умолчанию) и hard cases с
   многоступенчатой защитой от загрязнения (см. §13).

### 4.2. Структура выборок (splits)

Корпус разделяется через `GroupShuffleSplit` по `seed_id` так, чтобы все
вариации одного семантического зерна оказались в одном split (предотвращение
data leakage):

| Split | Назначение | Размер |
|---|---|---|
| `train` | Обучение моделей | 2606 |
| `val` | Контроль обучения, выбор гиперпараметров | 351 |
| `test` | Основная in-distribution оценка | 730 |
| `hard_test` | Пограничные сложные кейсы | 355 |
| `blind_test` | Hold-out, не виданный во время разработки | 40 |
| `entity_held_out` | OOD-проверка на новых услугах/врачах | 100 |
| `extended_eval` | Edge-кейсы (policy_only, subjective FAQ) | 101 |
| `safety_set` | Ургентные сообщения для recall_urgent | 87 |
| `switch_test` | Стресс-тест intent switch (text-only) | 50 |

Параметры split фиксированы константами в `d1/config.py`:
`SPLIT_RANDOM_STATE=42`, `TEST_RATIO=0.20`, `VAL_RATIO=0.125`.

### 4.3. Контроль контаминации

При генерации синтетических данных применяются следующие проверки:

1. **Точные дубликаты** между splits — запрещены строго.
2. **Семантическое перекрытие** через косинусную близость BGE-M3 эмбеддингов
   с порогом `LEAKAGE_COSINE_THRESHOLD=0.92` для базового аудита и более
   жёстким порогом `0.85` для контроля синтетической аугментации
   (см. §13.3).
3. **Intra-pool deduplication** внутри одного pool аугментации с порогом
   `0.90`.
4. **Entity hold-out** — фиксированный набор сущностей
   (`ENTITY_HELD_OUT` в `d1/config.py`) намеренно изолируется в OOD-выборке
   и удаляется из train/val.

### 4.4. Аудит данных

Все решения по разметке фиксируются в `d1/data/seed_audit.csv`. Аудит
включает 415 решений в категориях `keep / relabel / drop` с обязательными
комментариями. Решения по сложным кейсам после Phase 5 находятся в
`d1/data/phase5_audit.csv`.

---

## 5. Сравниваемые модели

Реестр базовых моделей определён в `d1/baselines/trained_bundle.py`
(`BASELINE_CONFIGS`). Каждая модель регистрируется с фиксированными
гиперпараметрами и единым контрактом (`fit`, `predict`, `predict_proba`).

В эксперимент входят **5 канонических baseline'ов** (по одному
представителю на архитектурное семейство):

### 5.1. B0 — rule-based baseline

| Модель | Описание |
|---|---|
| `B0_rules` | Pure rule-based classifier на основе ключевых слов и регулярных выражений. Нижняя планка качества, нужна как baseline для оценки прироста ML. |

### 5.2. B1 — sparse baselines

| Модель | Архитектура | Параметры |
|---|---|---|
| `B1.1_tfidf_lr` | TF-IDF char_wb(2,5) + Logistic Regression | `head_type=logistic` |
| `B1.3_fasttext` | fastText supervised classifier | sub-word embeddings + softmax |

### 5.3. B2 — dense baselines

| Модель | Encoder | Head | Профиль |
|---|---|---|---|
| `B2.1_bge-m3_svc` | BGE-M3 (568 M) | LinearSVC | quality benchmark |
| `B2.5_e5-small_svc` | multilingual-e5-small (118 M) | LinearSVC | lightweight dense |

Энкодер `multilingual-e5-small` примерно в **5 раз легче** BGE-M3 при
сопоставимом качестве на этом корпусе, что делает его кандидатом
для production с минимальным footprint.

### 5.4. Кэширование обученных моделей

Реализован строгий контракт инвалидации кэша через `CacheKey`
(`trained_bundle.py`), включающий пять компонентов:

1. `params_hash` — хеш гиперпараметров из `BASELINE_CONFIGS`;
2. `dataset_hash` — SHA-256 от **содержимого** train CSV;
3. `code_hash` — SHA-256 от исходников baseline-модулей;
4. `env_hash` — SHA-256 от версий sklearn / torch / sentence-transformers / python;
5. `schema_hash` — SHA-256 от `CSV_COLUMNS`.

При изменении любого компонента кэш автоматически инвалидируется и модель
переобучается. Это гарантирует воспроизводимость без ручной очистки артефактов.

---

## 6. Метрики и режимы оценки

Каждый baseline оценивается top-1 предсказанием на восьми eval-сетах.
Различия в характере eval-сета — отражены в трёх режимах отчётности:

### 6.1. Closed-set routing (RoutingReport)

Используется для `val`, `test`, `hard_test`, `blind_test`,
`entity_held_out`, `extended_eval`. Метрики:
`accuracy`, `macro_f1`, `balanced_accuracy`, `false_faq_for_anamnesis`,
confusion matrix, per-class P/R/F1. Колонка `recall_anamnesis` удалена
после cleanup-релиза 2026-05-11 как дубль `anamnesis_recall` из
per-class блока; для диагностики безопасности по классу смотрите
`anamnesis_recall` в per-class колонках или на `figures/per_class_f1_*.png`.

### 6.2. Safety report (SafetyReport)

Используется для `safety_set` (87 urgent/emergency cases).
Главная метрика — `recall_urgent` (доля корректно поднятых urgent-кейсов),
дополнительно — `false_negative_urgent` и распределение `misrouted_to`
(куда уходят пропущенные ургентные кейсы).

### 6.3. Switch stress (SwitchStressReport)

Используется для `switch_test` (50 кейсов). Модель видит ТОЛЬКО текст
второй реплики, а `active_domain` предыдущего диалога нужен лишь для
per-transition breakdown в отчёте (никак не подаётся на вход модели).
Это даёт верхнюю оценку устойчивости text-only классификатора к смене
интента.

> **Latency — single source of truth:**
> С cleanup-релиза 2026-05-11 `latency_ms` удалён из `baseline_results.csv`
> (inline replay-latency давал расхождение в 2–3× с правильным замером).
> Единственный источник latency — `latency_breakdown.csv`
> (`benchmark_latency.py`): **per-text median + p95 с разбивкой
> encode/predict**, n=100, repeats=5.

---

## 7. Дополнительные метрики (калибровка, латентность, статистика)

### 7.1. Closed-set метрики

| Метрика | Формула / Описание |
|---|---|
| `accuracy` | Доля корректных предсказаний |
| `macro_f1` | $\frac{1}{|\mathcal{Y}|} \sum_{y \in \mathcal{Y}} F_1(y)$ |
| `balanced_accuracy` | $\frac{1}{|\mathcal{Y}|} \sum_{y} \text{recall}(y)$ |
| `false_faq_for_anamnesis` | $P(\hat{y} = \text{faq} \mid y = \text{anamnesis})$ — доля жалоб, ушедших в FAQ. Самый опасный класс ошибок |
| `anamnesis_recall` (per-class) | $P(\hat{y} = \text{anamnesis} \mid y = \text{anamnesis})$ — диагностическая метрика безопасности по классу, читается из per-class блока |
| `confusion_matrix` | Матрица ошибок 4×4 |

### 7.2. Калибровка

| Метрика | Описание |
|---|---|
| `ECE` | Expected Calibration Error — средняя ошибка между предсказанной уверенностью и реальной точностью по бинам |
| `Brier OvR` | Brier score в one-vs-rest формулировке |
| Reliability diagrams | Графическая проверка калибровки по бинам |

Calibration считается только для B1.1 и B2.1 — для остальных моделей
proba либо нет (B0_rules, B1.3 fastText), либо она degenerate.

### 7.3. Латентность

| Метрика | Описание |
|---|---|
| `encode_ms` | Время векторизации текста (TF-IDF / fastText / encoder) |
| `predict_ms` | Время классификации (LR / SVC / centroid) |
| `total_ms` | Суммарное время на запрос |

Замеряется как медиана и 95-й перцентиль на $n=100$ запросах с $r=5$
повторами на CPU.

### 7.4. Статистическая значимость

| Тест | Назначение |
|---|---|
| Bootstrap 95% CI (BCa) | Family-bootstrap CI для `macro_f1` (на `test`, `hard_test`) и `recall_urgent` (на `safety_set`) |
| Paired bootstrap | Парное сравнение моделей с одинаковыми bootstrap-выборками + колонка `significant` (CI не пересекает 0) |

Покрытие: все 5 baseline × попарно (C(5,2) = 10 пар) × 1 метрика на каждом
eval-сете × {`test`, `hard_test`, `safety_set`}.

---

## 8. Экспериментальный дизайн

### 8.1. Кривые обучения (sample efficiency)

Для B1.1 и B2.1 (контрольные представители sparse / dense) обучение
проводится на 10%, 25%, 50%, 75% и 100% train с 5 разными random seeds
по `seed_id`-families (без leakage). Замер: `macro_f1` на `test` и
`hard_test`, `recall_urgent` на `safety_set`.

В контексте проекта это критически важно: модель, выходящая на плато
при ~25% train, потенциально применима без масштабной аугментации
через LLM.

Реализовано в `d1/scripts/learning_curves.py`.

### 8.2. Анализ ошибок (error taxonomy)

Все ошибки категоризируются по правилам через `d1/scripts/error_taxonomy.py`:

- `anamnesis_to_faq`, `anamnesis_to_booking`, `faq_to_anamnesis` —
  парные категории по sparse-prediction;
- `mixed_intent_error` — `subtype == mixed_intent`;
- `post_treatment_ambiguity` — `subtype ∈ {faq_borderline, procedure_info}`;
- `specialization_confusion` — `subtype ∈ {booking_with_spec, booking_with_doctor, doctor_info}`;
- `vague_short_error` — ≤ 3 токена в тексте.

Каждая строка ошибки может иметь несколько тегов (multi-label).
Дополнительно создаётся `error_taxonomy_audit_sample.csv` — шаблон
для ручной проверки rule-категорий.

### 8.3. Пайплайн вычислений

Полный pipeline эксперимента (`d1/scripts/notebook_runner.py`) — 7 шагов:

```
run_baselines         — обучение и оценка всех 5 baselines
benchmark_latency     — encode/predict breakdown на CPU (n=100, repeats=5)
analyze_confidence    — calibration (ECE, Brier, reliability) для B1.1/B2.1
run_statistical_tests — bootstrap BCa CI + paired bootstrap (5 моделей попарно)
error_taxonomy        — категоризация ошибок B1.1 ∪ B2.1
learning_curves       — sample efficiency (B1.1/B2.1 × 5 fractions × 5 seeds)
plot_results          — генерация всех figures
```

---

## 9. Структура директории `d1/`

```
d1/
├── README.md                    — этот файл (операционное руководство)
├── EXPERIMENT_D1.md             — текст раздела 2.3 ВКР
├── D1_domain_router_v6.ipynb    — основной экспериментальный notebook
├── config.py                    — общие константы и пути
├── pytest.ini                   — конфигурация pytest
│
├── baselines/                   — реализации моделей
│   ├── b0_rules.py              — rule-based baseline
│   ├── b1_tfidf.py              — TF-IDF + LR
│   ├── b1_fasttext.py           — fastText classifier
│   ├── b2_embedding.py          — encoder + LR/SVC/centroid
│   ├── trained_bundle.py        — реестр 5 моделей + строгий cache
│   ├── eval_metrics.py          — RoutingReport / SafetyReport / SwitchStressReport
│   ├── calibration.py           — ECE, Brier, reliability table
│   └── statistical_tests.py     — bootstrap CI, paired bootstrap
│
├── scripts/                     — orchestration runners (pipeline)
│   ├── run_baselines.py         — train + eval всех 5 baselines
│   ├── benchmark_latency.py     — encode/predict breakdown (per-text median + p95)
│   ├── analyze_confidence.py    — calibration (B1.1 / B2.1, без selective tooling)
│   ├── run_statistical_tests.py — bootstrap CI + paired tests (5 моделей)
│   ├── error_taxonomy.py        — таксономия ошибок B1.1/B2.1
│   ├── learning_curves.py       — sample efficiency
│   ├── plot_results.py          — генерация figures
│   ├── notebook_runner.py       — orchestrator (7 шагов)
│   ├── artifact_io.py           — read-only I/O для reporting-артефактов
│   ├── generate_dataset.py      — генерация train через Grok (через BGE-M3 dedup)
│   ├── generate_hard_cases.py   — генерация hard cases
│   ├── generate_phase5_dataset.py — аугментация (Phase 5)
│   ├── seed_audit.py            — аудит seeds
│   ├── leakage_audit.py         — проверка контаминации между splits
│   ├── split_dataset.py         — GroupShuffleSplit по seed_id
│   └── save_models.py           — wrapper для force re-train
│
├── data/                        — датасеты и артефакты разметки
│   ├── d1_v6_seeds.yaml         — эталонные seeds
│   ├── d1_v6_hard_cases.yaml    — hard cases (gold)
│   ├── d1_v6_blind_test.yaml    — blind test (manual)
│   ├── d1_v6_{train,val,test,hard_test,safety_set,blind_test,entity_held_out,extended_eval,switch_test,full,core}.csv
│   ├── seed_audit.csv           — аудит seeds
│   ├── phase5_*.{csv,yaml}      — артефакты Phase 5 аугментации
│   ├── entity_held_out.json     — manifest unseen entities
│   └── .archive_pre_phase5/     — архив pre-Phase 5 артефактов
│
├── prompts/                     — LLM-промпты для аугментации
│
├── ontology/                    — таксономия доменов (SSoT)
│   └── route_domain.yaml
│
├── results/                     — артефакты эксперимента
│   ├── baseline_results.{csv,json}        — closed-set метрики (5 × 6 eval-сетов), БЕЗ latency/recall_urgent (см. ниже)
│   ├── safety_results.{csv,json}          — единственный источник recall_urgent / FN / misrouted_to на safety_set (n=87)
│   ├── switch_results.{csv,json}          — switch_test (5 строк)
│   ├── latency_breakdown.csv              — единственный источник latency (per-text median + p95, n=100, repeats=5)
│   ├── calibration_metrics_*.json         — ECE, Brier (B1.1/B2.1 на test/hard_test)
│   ├── reliability_table_*.csv            — bins для reliability диаграмм (B1.1/B2.1 на test/hard_test)
│   ├── bootstrap_ci.csv                   — BCa 95% CI (5 моделей × 2 метрики на test/hard_test, recall_urgent только на safety_set)
│   ├── paired_tests.csv                   — попарные сравнения (10 пар) + значимость через CI
│   ├── error_taxonomy_*.csv               — таксономия ошибок (test, hard_test, blind_test)
│   ├── learning_curves.csv, learning_curves_summary.csv
│   ├── figures/                           — PNG графики (19 файлов после cleanup)
│   └── models/                            — кэшированные модели + bundle_metadata.json
│
└── tests/                       — unit-тесты pytest
```

---

## 10. Воспроизведение эксперимента

### 10.1. Требования

- Python ≥ 3.12 (рекомендуется 3.13);
- Активированное виртуальное окружение `study/.venv`;
- Установленные зависимости из `study/requirements.txt`.

### 10.2. Установка

```bash
cd /Users/kazdoraw/developer/med-agent/study
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 10.3. Запуск через notebook (рекомендуется)

```bash
cd /Users/kazdoraw/developer/med-agent/study
source .venv/bin/activate
jupyter notebook d1/D1_domain_router_v6.ipynb
```

Notebook структурирован по этапам closed-set эксперимента (см. ниже §10.4).

### 10.4. Запуск через CLI

Полный pipeline (7 шагов):

```bash
cd /Users/kazdoraw/developer/med-agent/study
source .venv/bin/activate
python -m d1.scripts.notebook_runner
```

Отдельный шаг:

```bash
python -m d1.scripts.run_baselines              # обучение 5 baselines
python -m d1.scripts.benchmark_latency          # encode/predict per-text
python -m d1.scripts.analyze_confidence         # calibration (ECE, Brier, reliability)
python -m d1.scripts.run_statistical_tests      # bootstrap + paired
python -m d1.scripts.error_taxonomy             # таксономия ошибок
python -m d1.scripts.learning_curves            # sample efficiency
python -m d1.scripts.plot_results               # все figures
```

### 10.5. Фактическое время прогона (M-series MPS, sentence-transformers на mps)

| Этап | Время |
|---|---|
| `run_baselines` (5 моделей, cache miss на dense) | ~2–3 мин |
| `benchmark_latency` ($n=100, r=5$) | ~1 мин |
| `analyze_confidence` (B1.1/B2.1 × 3 evals) | ~30 c |
| `run_statistical_tests` (45 CI + 90 paired bootstrap) | ~30 с |
| `error_taxonomy` | ~10 с |
| `learning_curves` (B1.1/B2.1 × 5 fractions × 5 seeds × 3 evals) | ~8–10 мин |
| `plot_results` | ~10 с |
| **Итого полный прогон** | **~13–15 мин** |

После cache hit повторный прогон pipeline укладывается в ~5 мин.

### 10.6. Кэширование моделей

После первого прогона обученные модели кэшируются в `d1/results/models/`.
Повторные вызовы `train_bundle()` загружают модели из кэша — без переобучения.
Кэш автоматически инвалидируется при изменении кода, данных или гиперпараметров
через `CacheKey` (см. §5.4).

---

## 11. Артефакты результатов

Все артефакты сохраняются в `d1/results/`:

### 11.1. Single source of truth (cleanup-релиз 2026-05-11)

| Метрика | Единственный источник | Не считается в |
|---|---|---|
| `accuracy` / `macro_f1` / `balanced_accuracy` | `baseline_results.csv` (6 standard eval-сетов) | — |
| `false_faq_for_anamnesis` | `baseline_results.csv` | `safety_set` (там все = anamnesis) |
| per-class P/R/F1 (включая `anamnesis_recall`) | `baseline_results.csv` (12 колонок) | — |
| `recall_urgent` / `FN_urgent` / `misrouted_to` | `safety_results.csv` (только `safety_set`, n=87) | `baseline_results.csv` (там было bogus: на test всего 17 urgent) |
| `latency_ms` (per-text median, p95) | `latency_breakdown.csv` (n=100, repeats=5) | `baseline_results.csv` (там был inline replay, расхождение в 2–3×) |
| `route_accuracy` switch | `switch_results.csv` | — |
| `ECE` / `Brier` | `calibration_metrics_*.json` (B1.1/B2.1, test + hard_test) | `val` (не в final reporting) |
| Bootstrap CI / paired tests | `bootstrap_ci.csv` / `paired_tests.csv` | `macro_f1` на test/hard_test, `recall_urgent` на safety_set |

Колонка `recall_anamnesis` (отдельная safety-колонка в baseline_results.csv)
**удалена** в cleanup-релизе: дублировала `anamnesis_recall` из per-class
блока (один и тот же recall, посчитанный двумя путями).

### 11.2. Артефакты эксперимента

| Файл / Директория | Содержимое |
|---|---|
| `baseline_results.{csv,json}` | Closed-set метрики 5 моделей × 6 standard eval-сетов. Без `latency_ms` / `recall_urgent` (см. 11.1) |
| `safety_results.{csv,json}` | Safety set: recall_urgent, FN, misrouted_to (n=87) |
| `switch_results.{csv,json}` | Switch test: per-transition route accuracy |
| `latency_breakdown.csv` | Per-text encode/predict median + p95 |
| `bootstrap_ci.csv` | BCa 95% CI: `macro_f1` на test/hard_test, `recall_urgent` на safety_set |
| `paired_tests.csv` | Paired bootstrap (10 пар) + колонка `significant` (CI не пересекает 0) |
| `calibration_metrics_*.json` | ECE, Brier OvR (B1.1 / B2.1 × test + hard_test) |
| `reliability_table_*.csv` | Bins для reliability diagrams (B1.1 / B2.1 × test + hard_test) |
| `error_taxonomy_*.csv` | Multi-label категоризация ошибок (B1.1 ∪ B2.1). Теги: `both_wrong`, `models_disagree_both_wrong`, `anamnesis_to_faq`, `anamnesis_to_booking`, `faq_to_anamnesis`, `vague_short_error`, `generic_error`, `specialization_confusion`, `post_treatment_ambiguity`, `mixed_intent_error` |
| `error_taxonomy_audit_sample.csv` | Шаблон для ручной проверки rule-категорий |
| `learning_curves.csv`, `learning_curves_summary.csv` | Sample efficiency (B1.1/B2.1). `macro_f1` на test/hard_test, `recall_urgent` на safety_set |
| `models/` | Кэшированные обученные модели (joblib + fastText) + `bundle_metadata.json` |
| `figures/` | PNG графики (19 после cleanup): routing_comparison, per_class_f1, confusion_matrices, cross_eval_f1, learning_curves, safety_comparison, safety_misroute_breakdown, latency_per_text, confidence_dist, reliability — для B1.1 и B2.1 на test + hard_test |

**Удалено в cleanup-релизе** (relic selective/hybrid эпохи): `threshold_table_*.csv`, `threshold_curve_*.png`, `disagreement_*.png`, `disagreement_by_class_*.png`, `confidence_scatter_*.png`. На `val` калибровка / графики не строятся.

---

## 12. Тестирование

```bash
cd /Users/kazdoraw/developer/med-agent/study
source .venv/bin/activate
python -m pytest d1/tests -q                          # все тесты
python -m pytest d1/tests/test_trained_bundle.py -v   # один файл
```

Покрытие тестами:

- модели: `test_baselines_smoke.py`, `test_calibration.py`;
- метрики: `test_statistical_tests.py`, `test_error_taxonomy.py`;
- инфраструктура: `test_trained_bundle.py`, `test_learning_curves.py`,
  `test_plot_results_extension.py`;
- интеграции: `test_switch_stress.py`.

---

## 13. Аугментация данных через LLM

### 13.1. Постановка задачи

Для усиления статистической мощности эксперимента и стресс-тестирования
роутера на реалистичных edge-cases применяется управляемая LLM-аугментация
с защитой от загрязнения существующих splits.

### 13.2. Модель и параметры генерации

| Параметр | Значение |
|---|---|
| LLM | `x-ai/grok-4.1-fast` (через OpenRouter) |
| Temperature (per-scenario) | clean=0.7, hard=0.9, short=1.1 |
| Top-p | 0.95 |
| Multi-pass diversity | 2 прохода с разными seeds (42, 43) |
| Max tokens per request | 1500 |
| Total tokens cap (hard limit) | 80 000 (~$0.5–1) |

### 13.3. Защита от контаминации

1. **BGE-M3 similarity** к hard_cases.yaml, seeds.yaml, train.csv —
   порог отклонения 0.85;
2. **Intra-pool deduplication** — порог 0.90;
3. **Negative markers** — regex-блокировка маркеров чужих доменов
   (для clean train сценариев);
4. **JSON schema validation** — строгая проверка формата каждого кандидата.

### 13.4. Сценарии генерации

8 сценариев в двух категориях:

**Train clean (4):** `simple_faq_clean`, `simple_booking_clean`,
`simple_symptom_clean`, `entity_held_out_simple`.

**Eval addendum (4):** `faq_vs_anamnesis_borderline`, `doctor_name_stress`,
`mixed_intent_new`, `short_ambiguous_new`.

### 13.5. Auto-audit

Каждый кандидат проходит автоматический аудит через ансамбль обученных
ML-моделей (B1.1 и B2.1). Решение принимается с учётом сценария:

- **Clean сценарии** (train): ML-disagreement ⇒ `reject` (отбрасываем
  возможные галлюцинации разметки);
- **Hard / short сценарии** (eval): ML-disagreement ⇒ `accept`,
  помечается флагом `hard_ml_disagree` (это ценные edge-cases для
  стресс-теста).

### 13.6. Запуск аугментации

> **⚠️ ВНИМАНИЕ:** аугментация требует API-ключа OpenRouter и расходует
> токены. По умолчанию **не** запускается из notebook (для безопасности).

```bash
cd /Users/kazdoraw/developer/med-agent/study
source .venv/bin/activate
export OPENROUTER_API_KEY=sk-or-v1-...
python -m d1.scripts.generate_phase5_dataset --concurrency 4 --passes 2
```

```bash
python -c "
import pandas as pd
train = pd.read_csv('d1/data/d1_v6_train.csv')
add = pd.read_csv('d1/data/phase5_train_addendum.csv')
pd.concat([train, add]).to_csv('d1/data/d1_v6_train.csv', index=False)
"
```
