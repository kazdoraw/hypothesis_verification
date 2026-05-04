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
6. [Архитектура каскада маршрутизации](#6-архитектура-каскада-маршрутизации)
7. [Метрики оценки](#7-метрики-оценки)
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

Проверить эмпирическую возможность замены LLM-роутера на каскад из лёгких
классификаторов на основе классических методов машинного обучения (TF-IDF +
linear models, fastText, embedding + classifier head). Каскад должен:

1. **Уверенно** классифицировать "простые и однозначные" сообщения локально,
   без обращения к LLM.
2. **Безопасно делегировать** в LLM "сложные и неоднозначные" случаи через
   контролируемую политику отказа (selective deferral).
3. Удерживать целевую безопасность (recall на симптомах) на уровне или выше
   текущего LLM-baseline.
4. Обеспечивать существенно меньшую латентность и стоимость.

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
определить доменный workflow:
`{anamnesis, faq, booking, unsupported}`?

**RQ2 — Selective Deferral.** При каких уровнях уверенности роутер должен
маршрутизировать самостоятельно, задавать уточнение или передавать сообщение
в LLM-fallback?

**RQ3 — Intent Switch.** Возможно ли быстро и корректно выявлять смену
намерения внутри уже активного домена (отдельная подзадача switch detection)?

**RQ4 — Robustness.** Как роутер ведёт себя на коротких репликах, шуме,
смешанных запросах, unseen entities и out-of-distribution фразировках?

### 2.2. Гипотезы

**H1 (основная).** Лёгкий гибридный роутер, обученный на production-aligned
корпусе размеченных сообщений, достигает качества маршрутизации,
сопоставимого с LLM-роутером, при существенно меньших latency и cost.

**H2 (safety / deferral).** Добавление калиброванной уверенности и явной
политики отказа снижает количество опасных misroute-ошибок по сравнению с
forced classification.

**H3 (intent switch).** Отдельный быстрый детектор смены намерения улучшает
корректность обработки активных диалогов по сравнению с использованием
только primary-роутера.

---

## 3. Методология

### 3.1. Формулировка задачи

Дано: текстовое сообщение пользователя $x \in \mathcal{X}$, представленное
последовательностью токенов на русском языке.

Требуется: классификатор $f: \mathcal{X} \rightarrow \mathcal{Y} \cup \{\bot\}$,
где
$\mathcal{Y} = \{\text{anamnesis}, \text{faq}, \text{booking}, \text{unsupported}\}$,
а $\bot$ — формальный класс отказа от классификации (`defer`), при котором
сообщение делегируется в LLM-fallback.

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

| Split | Назначение | Размер (примерно) |
|---|---|---|
| `train` | Обучение моделей | ~2.5 K |
| `val` | Подбор гиперпараметров и порогов | ~600 |
| `test` | Основная in-distribution оценка | ~1.2 K |
| `hard_test` | Пограничные сложные кейсы | ~350 |
| `blind_test` | Hold-out, не виданный во время разработки | ~50 |
| `entity_held_out` | OOD-проверка на новых услугах/врачах | ~150 |
| `extended_eval` | Edge-кейсы (policy_only, subjective FAQ) | ~150 |
| `safety_set` | Ургентные сообщения для контроля recall_anamnesis | ~100 |
| `switch_test` | Стресс-тест intent switch | ~50 |

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

### 5.1. Семейство B0 — правила

| Модель | Описание |
|---|---|
| `B0_rules` | Pure rule-based classifier на основе ключевых слов и регулярных выражений. Нижняя планка качества, требуется как baseline для оценки прироста ML. |

### 5.2. Семейство B1 — sparse классификаторы

| Модель | Архитектура | Параметры |
|---|---|---|
| `B1_tfidf_svc` | TF-IDF (1–3 gram) + LinearSVC | `head_type=svc` |
| `B1.1_tfidf_lr` | TF-IDF + Logistic Regression | `head_type=logistic` |
| `B1.2_tfidf_lr_tuned` | TF-IDF (2–6 gram, min_df=1) + LR (`C=0.5`) | tuned |
| `B1.3_fasttext` | fastText supervised classifier | sub-word embeddings + softmax |

### 5.3. Семейство B2 — dense классификаторы

| Модель | Encoder | Head | Параметры |
|---|---|---|---|
| `B2_bge-m3_linear` | BGE-M3 (568 M) | Logistic Regression | default C |
| `B2.1_bge-m3_svc` | BGE-M3 | LinearSVC | — |
| `B2.2_bge-m3_centroid` | BGE-M3 | nearest-centroid | — |
| `B2.3_bge-m3_linear_tuned` | BGE-M3 | LR (`C=0.3`) | tuned |
| `B2.4_e5-small_linear` | multilingual-e5-small (118 M) | LR | — |
| `B2.5_e5-small_svc` | multilingual-e5-small | LinearSVC | — |

Энкодер `multilingual-e5-small` примерно в **5 раз легче** BGE-M3 при
сопоставимой латентности на тех же данных, что делает его кандидатом
для production-каскада с минимальным footprint.

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

## 6. Архитектура каскада маршрутизации

Эксперимент сравнивает **четыре** политики маршрутизации поверх базовых
моделей.

### 6.1. Closed-set baseline

Каждая модель обязана вернуть одну из четырёх меток для каждого сообщения,
без режима отказа. Используется как ориентир качества каждой архитектуры
в изоляции.

### 6.2. Selective Router

`SelectiveRouter` вводит режим отказа: модель возвращает метку только при
выполнении двух условий:

1. **Достаточная уверенность** — `confidence ≥ τ` (порог $\tau$ подобран
   на validation);
2. **Согласие моделей** — sparse-классификатор (B1) и dense-классификатор
   (B2) совпадают в предсказанной метке.

При нарушении любого условия результат — `defer`, и сообщение делегируется
в LLM-fallback. Это формализует safety-первый подход: лучше вернуть
"не знаю", чем misroute.

### 6.3. Hybrid Router (B4)

`B4HybridRouter` добавляет rules-first слой перед selective-каскадом:

```
input → Rules (high-precision) → if matched: accept
                              ↓
                              else: → SelectiveRouter
```

Простые сильные rule-сигналы (явный booking-keyword, прямой price-question,
очевидные greetings) принимаются сразу — это даёт прирост `coverage` без
просадки `accepted_accuracy`.

### 6.4. SimpleRouter (production-кандидат)

`SimpleRouter` — наиболее консервативная политика, кандидат для production.
Каскад работает в три этапа:

```
input → ComplexityGate → if complex: defer (без обращения к ML)
                       ↓
                       else → B4HybridRouter
                                ↓
                           → TagPolicy (post-filter)
```

#### 6.4.1. ComplexityGate

Rule-based pre-filter, отсекающий заведомо сложные кейсы:

- multi-intent сообщения (несколько разных намерений в одной фразе);
- очень короткие неоднозначные фразы (≤ 3 слова без явного intent);
- запросы с конкретной фамилией врача (требуют доменных знаний).

Все эти кейсы уходят в `defer` без обращения к ML — снижая ложные
срабатывания на самой сложной части входного потока.

#### 6.4.2. TagPolicy

Пост-фильтр для слабых зон классификатора (`simple_faq`, `simple_booking`,
`simple_symptom`). Принимает решение модели только при условиях:

- `confidence ≥ min_confidence_per_tag`;
- предсказанная метка ∈ `allowed_labels` для тега.

Текущие пороги (зафиксированы в `simple_router.DEFAULT_TAG_POLICIES` после
Pareto-sweep на validation):

| Тег | min_confidence | Допустимые метки |
|---|---|---|
| `simple_faq` | 0.75 | `{faq}` |
| `simple_booking` | 0.75 | `{booking}` |
| `simple_symptom` | 0.70 | `{anamnesis}` |

### 6.5. Сравнение каскадов

Эксперимент дополнительно сравнивает два production-кандидата:

| Каскад | Sparse (B1) | Dense (B2) | Профиль |
|---|---|---|---|
| Производственный | `B1.1_tfidf_lr` | `B2.1_bge-m3_svc` | Высокая точность, тяжёлая dense-модель |
| Лёгкий | `B1.3_fasttext` | `B2.5_e5-small_svc` | В разы быстрее, меньше памяти |

---

## 7. Метрики оценки

### 7.1. Closed-set метрики

| Метрика | Формула / Описание |
|---|---|
| `accuracy` | Доля корректных предсказаний |
| `macro_f1` | $\frac{1}{|\mathcal{Y}|} \sum_{y \in \mathcal{Y}} F_1(y)$ |
| `balanced_accuracy` | $\frac{1}{|\mathcal{Y}|} \sum_{y} \text{recall}(y)$ |
| `recall_anamnesis` | Safety-критичный recall на симптомах |
| `confusion_matrix` | Матрица ошибок 4×4 |

### 7.2. Selective метрики

| Метрика | Описание |
|---|---|
| `coverage` | Доля сообщений, на которые роутер вернул метку (а не `defer`) |
| `accepted_accuracy` | Точность среди принятых сообщений |
| `accepted_recall_anamnesis` | Safety: recall на симптомах среди принятых |
| `defer_rate` | $1 - \text{coverage}$ |

**Важно:** `accepted_accuracy` нельзя напрямую сравнивать с closed-set
`accuracy` — корректное сравнение это всегда пара
$(\text{coverage}, \text{accepted\_accuracy})$.

### 7.3. Калибровка

| Метрика | Описание |
|---|---|
| `ECE` | Expected Calibration Error — средняя ошибка между предсказанной уверенностью и реальной точностью по бинам |
| `Brier OvR` | Brier score в one-vs-rest формулировке |
| Reliability diagrams | Графическая проверка калибровки по бинам |

### 7.4. Латентность

| Метрика | Описание |
|---|---|
| `encode_ms` | Время векторизации текста (TF-IDF / fastText / encoder) |
| `predict_ms` | Время классификации (LR / SVC / centroid) |
| `total_ms` | Суммарное время на запрос |

Замеряется как медиана и 95-й перцентиль на $n=100$ запросах с $r=5$
повторами на CPU.

### 7.5. Статистическая значимость

| Тест | Назначение |
|---|---|
| Bootstrap 95% CI | Доверительные интервалы для `accuracy` и `macro_f1` |
| McNemar paired test | Парное сравнение пар моделей на одних примерах |
| Paired bootstrap | Чувствительный аналог при близких метриках |

---

## 8. Экспериментальный дизайн

### 8.1. Подбор порогов tag-policy

Используется чистый Pareto-sweep:

1. Сетка порогов перебирается **только** на `val`;
2. Для каждой точки считается пара $(\text{coverage}, \text{accepted\_accuracy})$;
3. Извлекается Pareto-frontier;
4. Выбирается компромисс по метрике-агрегатору;
5. Подобранные пороги проверяются на `test`, `hard_test`, `blind_test`,
   `entity_held_out`, `extended_eval` — при отсутствии регрессии фиксируются
   в `simple_router.DEFAULT_TAG_POLICIES`.

Реализовано в `d1/scripts/tag_policy_sweep.py`.

### 8.2. Кривые обучения (sample efficiency)

Для каждой модели обучение проводится на 25%, 50%, 75% и 100% train с
последующим замером accuracy/macro_F1 на test. Это отвечает на вопрос:
**сколько данных** нужно конкретной архитектуре для выхода на плато?

В контексте проекта это критически важно: лёгкая модель, выходящая на
плато при ~1 K примеров, потенциально применима без необходимости
масштабной аугментации через LLM.

Реализовано в `d1/scripts/learning_curves.py`.

### 8.3. Анализ ошибок (error taxonomy)

Все ошибки категоризируются по типам через `d1/scripts/error_taxonomy.py`:

- `boundary_domain_miss` — ошибка на границе доменов (FAQ ↔ ANAMNESIS);
- `urgency_miss` — пропущенная ургентность;
- `doctor_name_confusion` — путаница из-за упомянутой фамилии врача;
- `mixed_intent` — смешанные намерения в одной фразе;
- `short_ambiguous` — слишком короткое неоднозначное сообщение;
- `entity_unseen` — неизвестная клинике сущность.

Категоризация позволяет приоритизировать улучшения: ошибки одной группы
обычно требуют общего решения (rule-fix, data augmentation, retraining).

### 8.4. Пайплайн вычислений

Полный pipeline эксперимента (`d1/scripts/notebook_runner.py`):

```
run_baselines        — обучение и оценка всех baselines
analyze_confidence   — калибровка + Pareto-таблицы
evaluate_selective   — selective router metrics
evaluate_hybrid      — hybrid (rules + selective ML)
threshold_sweep      — подбор thresholds для hybrid
evaluate_simple_router — production cascade
run_statistical_tests — bootstrap CI + paired tests
plot_results         — генерация всех figures
```

---

## 9. Структура директории `d1/`

```
d1/
├── README.md                    — этот файл (операционное руководство)
├── D1_РЕЗУЛЬТАТЫ_ВКРС.md        — итоговый раздел 2.3 для ВКРС
├── D1_domain_router_v6.ipynb    — основной экспериментальный notebook
├── config.py                    — общие константы и пути
├── pytest.ini                   — конфигурация pytest
│
├── baselines/                   — реализации моделей и роутеров
│   ├── b0_rules.py              — rule-based baseline
│   ├── b1_tfidf.py              — TF-IDF + LR/SVC
│   ├── b1_fasttext.py           — fastText classifier
│   ├── b2_embedding.py          — encoder + LR/SVC/centroid
│   ├── b4_hybrid.py             — hybrid router (rules + selective)
│   ├── complexity_gate.py       — pre-filter сложных кейсов
│   ├── simple_router.py         — SimpleRouter + TagPolicy
│   ├── selective_router.py      — selective router
│   ├── trained_bundle.py        — реестр моделей + кэш
│   ├── eval_metrics.py          — метрики (RoutingReport, SafetyReport, ...)
│   ├── calibration.py           — калибровка confidence
│   └── statistical_tests.py     — bootstrap CI, paired tests
│
├── scripts/                     — runners (orchestration)
│   ├── run_baselines.py         — train + eval всех baselines
│   ├── analyze_confidence.py    — калибровка + Pareto
│   ├── evaluate_selective.py    — оценка selective router
│   ├── evaluate_hybrid.py       — оценка hybrid router
│   ├── evaluate_simple_router.py — оценка SimpleRouter
│   ├── threshold_sweep.py       — sweep порогов hybrid
│   ├── tag_policy_sweep.py      — Pareto-sweep tag-policy
│   ├── benchmark_latency.py     — замер латентности
│   ├── learning_curves.py       — кривые обучения
│   ├── run_statistical_tests.py — bootstrap CI + paired
│   ├── error_taxonomy.py        — таксономия ошибок
│   ├── plot_results.py          — генерация figures
│   ├── notebook_runner.py       — orchestrator pipeline для notebook
│   ├── notebook_reports.py      — show_* функции для notebook
│   ├── notebook_widgets.py      — интерактивные виджеты
│   ├── interactive_inference.py — ручная проверка модели
│   ├── analyze_confidence.py    — анализ confidence распределений
│   ├── generate_dataset.py      — генерация train/val/test через Grok
│   ├── generate_hard_cases.py   — генерация hard cases (Phase 4)
│   ├── generate_phase5_dataset.py — аугментация (Phase 5)
│   ├── integrate_hard_cases_audit.py — merge audited hard cases
│   ├── seed_audit.py            — аудит seeds
│   ├── leakage_audit.py         — проверка контаминации между splits
│   ├── split_dataset.py         — GroupShuffleSplit по seed_id
│   ├── build_complexity_audit_sample.py — sample для аудита gate
│   ├── diagnose_tag_distribution.py — диагностика tag-распределения
│   ├── save_models.py           — wrapper для force re-train
│   ├── artifact_io.py           — общий I/O для артефактов
│   └── cross_tune_sanity.py     — bootstrap stability check
│
├── data/                        — датасеты и артефакты разметки
│   ├── d1_v6_seeds.yaml         — эталонные seeds (415 шт)
│   ├── d1_v6_hard_cases.yaml    — hard cases (gold)
│   ├── d1_v6_blind_test.yaml    — blind test (manual)
│   ├── d1_v6_train.csv          — train split
│   ├── d1_v6_val.csv            — validation split
│   ├── d1_v6_test.csv           — test split
│   ├── d1_v6_hard_test.csv      — hard test split
│   ├── d1_v6_blind_test.csv     — blind hold-out
│   ├── d1_v6_entity_held_out.csv — OOD: unseen entities
│   ├── d1_v6_extended_eval.csv  — edge-cases (policy_only, subjective)
│   ├── d1_v6_safety_set.csv     — ургентные сообщения
│   ├── d1_v6_switch_test.csv    — стресс-тест intent switch
│   ├── d1_v6_full.csv           — combined view (output split_dataset)
│   ├── d1_v6_core.csv           — core variants
│   ├── seed_audit.csv           — аудит 415 seeds
│   ├── phase5_*.{csv,yaml}      — артефакты Phase 5 аугментации
│   ├── entity_held_out.json     — manifest unseen entities
│   ├── complexity_audit_sample.csv — sample для аудита gate
│   └── .archive_pre_phase5/     — архив pre-Phase 5 артефактов
│
├── prompts/                     — LLM-промпты
│   ├── d1_variation.md          — промпт variation generation (Grok)
│   ├── d1_hard_case_candidates.md — промпт hard cases (Phase 4)
│   └── d1_phase5_candidates.md  — промпт Phase 5 augmentation
│
├── ontology/                    — таксономия доменов (SSoT)
│   └── route_domain.yaml
│
├── results/                     — посчитанные артефакты эксперимента
│   ├── *.csv, *.json            — таблицы метрик (baseline / selective / hybrid /
│   │                             simple_router / safety / calibration / bootstrap /
│   │                             paired tests / error_taxonomy / learning_curves /
│   │                             tag_policy_sweep / latency_breakdown /
│   │                             accepted_errors_*)
│   ├── figures/                 — все графики (42 PNG)
│   ├── models/                  — кэшированные обученные модели (joblib + bin)
│   │                             + bundle_metadata.json (cache key и hash модели)
│   └── cascade_lightweight/     — артефакты лёгкого каскада (B1.3 + B2.5)
│
└── tests/                       — unit-тесты
    └── test_*.py                — pytest fixtures + тесты
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

Notebook структурирован как 16 разделов (0–15) с управляющими флагами в
разделе 0:

| Флаг | Включает блок |
|---|---|
| `RUN_TRAIN_BASELINES` | Принудительное переобучение всех моделей |
| `RUN_BENCHMARK_LATENCY` | Замер латентности на CPU |
| `RUN_TAG_POLICY_SWEEP` | Pareto-sweep порогов tag-policy на val |
| `RUN_CASCADE_COMPARE` | Сравнение производственного и лёгкого каскадов |
| `RUN_LEARNING_CURVES` | Замер кривых обучения |
| `RUN_STATISTICAL_TESTS` | Bootstrap CI + paired tests |
| `RUN_ERROR_TAXONOMY` | Категоризация типов ошибок |

Альтернатива — флаг `FORCE_RERUN = True` в разделе 2 запускает полный
основной pipeline (`run_baselines → ... → plot_results`) одной командой.

### 10.4. Запуск через CLI

Полный pipeline:

```bash
cd /Users/kazdoraw/developer/med-agent/study
source .venv/bin/activate
python -c "from d1.scripts.notebook_runner import run_d1_pipeline; run_d1_pipeline()"
```

Отдельный шаг:

```bash
python -m d1.scripts.run_baselines              # обучение всех baselines
python -m d1.scripts.evaluate_simple_router     # оценка SimpleRouter
python -m d1.scripts.benchmark_latency          # латентность
python -m d1.scripts.learning_curves            # кривые обучения
python -m d1.scripts.tag_policy_sweep           # подбор порогов
python -m d1.scripts.run_statistical_tests      # стат. тесты
python -m d1.scripts.plot_results               # генерация figures
```

### 10.5. Ожидаемое время прогона (CPU)

| Этап | Время |
|---|---|
| `run_baselines` (11 моделей с dense embeddings) | 3–5 мин |
| `analyze_confidence` + `evaluate_selective/hybrid/simple_router` | ~1 мин |
| `benchmark_latency` ($n=100, r=5$) | ~30 с |
| Сравнение каскадов (lightweight) | ~30 с |
| `learning_curves` (4 ratios × 11 моделей) | 2–4 мин |
| `error_taxonomy` | ~10 с |
| **Итого полный прогон** | **~10–12 мин** |

### 10.6. Кэширование моделей

После первого прогона обученные модели кэшируются в `d1/results/models/`.
Повторные вызовы `train_bundle()` загружают модели из кэша — без переобучения.
Кэш автоматически инвалидируется при изменении кода, данных или гиперпараметров
через `CacheKey` (см. §5.4).

---

## 11. Артефакты результатов

Все артефакты сохраняются в `d1/results/`:

| Файл / Директория | Содержимое |
|---|---|
| `baseline_results.{csv,json}` | Closed-set метрики всех моделей по eval-сетам |
| `selective_results.{csv,json}` | Selective router metrics |
| `hybrid_results.{csv,json}` | Hybrid router metrics |
| `simple_router_results.{csv,json}` | SimpleRouter metrics |
| `simple_vs_hybrid.csv` | Side-by-side сравнение каскадов |
| `safety_results.{csv,json}` | Safety set метрики (recall_anamnesis) |
| `switch_results.{csv,json}` | Switch test метрики |
| `bootstrap_ci.csv` | 95% доверительные интервалы |
| `paired_tests.csv` | Парные статистические тесты |
| `error_taxonomy_*.csv` | Категоризация ошибок по eval-сетам |
| `complexity_breakdown_*.csv` | Распределение по сложности |
| `latency_breakdown.csv` | Профиль латентности |
| `learning_curves_summary.csv` | Сводка по кривым обучения |
| `pareto_candidates_*.csv` | Pareto-frontier по моделям |
| `threshold_table_*.csv` | Подобранные пороги |
| `reliability_table_*.csv` | Reliability diagrams data |
| `calibration_metrics_*.json` | ECE, Brier OvR |
| `models/` | Кэшированные обученные модели (joblib) |
| `figures/` | Все графики (PNG) |
| `reports/` | Текстовые отчёты по этапам |
| `cascade_lightweight/` | Артефакты лёгкого каскада (B1.3 + B2.5) |

---

## 12. Тестирование

```bash
cd /Users/kazdoraw/developer/med-agent/study
source .venv/bin/activate
python -m pytest d1/tests -q                 # все тесты
python -m pytest d1/tests -q -m "not slow"   # без медленных
python -m pytest d1/tests/test_simple_router.py -v  # один файл
```

Покрытие тестами:

- модели: `test_baselines_smoke.py`, `test_calibration.py`;
- роутеры: `test_b4_hybrid.py`, `test_selective_router.py`,
  `test_simple_router.py`, `test_complexity_gate.py`;
- метрики: `test_statistical_tests.py`, `test_error_taxonomy.py`;
- инфраструктура: `test_trained_bundle.py`, `test_threshold_sweep.py`,
  `test_learning_curves.py`, `test_plot_results_extension.py`;
- интеграции: `test_evaluate_simple_router.py`, `test_switch_stress.py`,
  `test_interactive_inference.py`, `test_build_complexity_audit_sample.py`,
  `test_notebook_executor_only.py`.

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

Артефакты:
- `d1/data/phase5_candidates.yaml` — все после anti-contamination;
- `d1/data/phase5_similarity.csv` — лог BGE-сравнений;
- `d1/data/phase5_audit.csv` — auto-audit с решениями;
- `d1/data/phase5_train_addendum.csv` — accepted train cases;
- `d1/data/phase5_hard_test_addendum.csv` — accepted eval cases.

После аудита аддендум вручную сливается с основными split-CSV:

```bash
python -c "
import pandas as pd
train = pd.read_csv('d1/data/d1_v6_train.csv')
add = pd.read_csv('d1/data/phase5_train_addendum.csv')
pd.concat([train, add]).to_csv('d1/data/d1_v6_train.csv', index=False)
"
```

---

## 14. Известные ограничения

1. **Single-turn routing.** Эксперимент работает только с первым сообщением
   диалога. Multi-turn контекст и история вынесены за scope (RQ3 — отдельный
   подэксперимент `switch_test`).
2. **Русский язык.** Корпус, encoder и rule-based слой настроены на русский.
   Применимость к другим языкам не проверялась.
3. **Стоматологический домен.** Корпус специфичен для стоматологии.
   Перенос на другие медицинские домены требует пере-разметки seeds.
4. **CPU-инференс.** Латентность замеряется на CPU; GPU-замеры могут
   изменить выводы по dense-моделям.
5. **Синтетические данные.** Большая часть train сгенерирована через LLM.
   Несмотря на anti-contamination и auto-audit, тут потенциально присутствует
   distribution shift относительно реального production-трафика.
6. **Закрытый таксон.** Используется фиксированная 4-классовая таксономия
   (`anamnesis / faq / booking / unsupported`); открытые классы и hierarchical
   классификация не рассматриваются.

---

## 15. Ссылки на компоненты проекта

- **AI-ядро (production):** `med-agent/ai-core` — целевой проект для
  внедрения результатов эксперимента (LangGraph chatbot, FastAPI :8080).
- **Backend:** `med-agent/provider` — Go backend, проксирует chat-запросы
  в ai-core.
- **D2 эксперимент:** `study/d2/` — параллельный эксперимент по другому
  компоненту маршрутизации.

### 15.1. Основные документы

- [D1_РЕЗУЛЬТАТЫ_ВКРС.md](D1_РЕЗУЛЬТАТЫ_ВКРС.md) — итоговый текст раздела 2.3 ВКРС
  с актуальными метриками, анализом ошибок и выводами для архитектуры;
- [D1_domain_router_v6.ipynb](D1_domain_router_v6.ipynb) — экспериментальная
  тетрадь с control panel и точечными прогонами;
- `study/.windsurf/plans/` — рабочие планы по фазам реализации.

### 15.2. Источники методологии

- `LangGraph` — orchestration framework (используется в production);
- `sentence-transformers` — энкодеры (BGE-M3, multilingual-e5);
- `scikit-learn` — classifier heads (LogisticRegression, LinearSVC,
  GroupShuffleSplit);
- `fastText` — sub-word embeddings + softmax classifier;
- `OpenRouter` — единый API к LLM-моделям (Grok, GPT, Claude);
- McNemar test, bootstrap resampling — статистическая проверка.

---

> **Контактное лицо по эксперименту:** автор ВКР.
> **Последнее обновление README:** 2026-05-04.
