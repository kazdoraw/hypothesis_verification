# План: Оценка маршрутизации через сравнение с эталоном

## Цель

Корректно оценить качество сбора анамнеза, сравнивая **маршрутизацию доктора** (вывод Qwen на основе собранных данных) с **эталонной маршрутизацией** (из полного описания кейса).

## Архитектура (пайплайн)

```
                   ДИАЛОГ (существующий)
                         │
              Grok ←→ Qwen (N ходов)
                         │
                    ┌─────┴─────┐
                    │ extracted  │  (JSON анкета)
                    └─────┬─────┘
                          │
               ┌──────────┴──────────┐
               │  Qwen: отдельный    │  ← routing_infer.md
               │  вызов (слепой)     │     (НЕ видит бэкграунд)
               └──────────┬──────────┘
                          │
                   doctor_routing:
                   - specialists
                   - service_type
                   - examination
                          │
    Case.reference_routing│
    (из cases.py)         │
            │             │
            └──────┬──────┘
                   │
            ┌──────┴──────┐
            │   Judge     │  ← judge.md
            │(gpt-5.4-mini)│     (видит оба routing + extracted + диалог)
            └──────┬──────┘
                   │
            routing_match + scores
```

## Анализ кодовой базы

### Файлы и их роли

| Файл | Ответственность | Затрагивается? |
|------|----------------|----------------|
| `models.py` | Data models (Case, SchemaRun, CaseResult) | ✅ +2 поля |
| `cases.py` | 10 кейсов с reference_fields | ✅ +reference_routing |
| `schemas.py` | Pydantic-схемы извлечения (S1/S2/S3, DoctorTurn) | ❌ не трогаем |
| `doctor.py` | Qwen: doctor_process() + парсинг | ✅ +infer_routing() |
| `patient.py` | Grok: patient_speak() | ❌ не трогаем |
| `session.py` | Оркестрация диалога, save_case_result() | ✅ +вызов infer_routing |
| `client.py` | OpenRouter HTTP клиент | ❌ не трогаем |
| `config.py` | Константы, пути, модели | ✅ +MAX_TOKENS_ROUTING |
| `judge.py` | Оценка: judge_single(), judge_all_cases() | ✅ обновить _build_judge_input |
| `report.py` | Markdown + CSV отчёты | ✅ +doctor routing в отчёты |
| `viz.py` | Matplotlib графики | ✅ минимально |
| `metrics.py` | Embedding метрики (не используются) | ❌ не трогаем |
| `run.py` | CLI: --cases, --judge, --report-only | ✅ +--routing-only флаг |

### Текущий data flow

```
run.py:main()
  → session.run_case(client, case)
      → session.run_case_schema(client, case, "S1") → SchemaRun
      → session.run_case_schema(client, case, "S2") → SchemaRun
      → session.run_case_schema(client, case, "S3") → SchemaRun
  → session.save_case_result(result) → case_XX.json
  → report.generate_report()
  → viz.generate_all_plots()

run.py:_run_judge()
  → judge.judge_all_cases()
      → judge.judge_single(client, case, run) × 30
  → judge.save_judge_scores()
  → report.generate_report()
```

### Структура сохранённых данных

**case_XX.json** (SchemaRun):
```json
{
  "schema_name": "S1",
  "dialog": [...],
  "extracted": {"symptoms": "...", "localization": "..."},
  "turns": 5,
  "tokens_doctor": 9712,
  "tokens_patient": 5621,
  "duration_s": 103.8
}
```

**Важно**: поля `routing` в JSON нет → нужен `default_factory=dict` для обратной совместимости.

### S3 extracted — особая структура

S3 использует русские ключи и строковое представление dict:
```json
{
  "complaint_type": "боль",
  "fields": "{'длительность_боли': '3 дня', 'локализация_боли': '...'}",
  "ночная_боль": "есть",
  "аллергия": "на пенициллин"
}
```
→ Промпт routing_infer.md должен принимать произвольный JSON без привязки к ключам.

### Текущие константы (config.py)

| Константа | Текущий файл | Значение |
|-----------|-------------|----------|
| DOCTOR_MODEL | config.py | qwen/qwen3-235b-a22b-2507 |
| MAX_TOKENS_DOCTOR | config.py | 300 |
| TEMPERATURE_DOCTOR | config.py | 0.4 |
| JUDGE_MODEL | ⚠️ judge.py:20 | openai/gpt-5.4-mini |
| MAX_TOKENS_JUDGE | ⚠️ judge.py:22 | 2000 |
| TEMPERATURE_JUDGE | ⚠️ judge.py:23 | 0.1 |
| JUDGE_PROMPT_PATH | ⚠️ judge.py:21 | prompts/judge.md |

**Проблема:** настройки judge разбросаны по `judge.py` вместо `config.py`. Нужно вынести в единый конфиг.

## Шаги реализации

### Шаг 1. `config.py` — новые константы + перенос из judge.py

Добавить новые:
```python
# --- Routing inference ---
MAX_TOKENS_ROUTING = 500
TEMPERATURE_ROUTING = 0.2
```

Перенести из `judge.py` (строки 20-23):
```python
# --- Judge ---
JUDGE_MODEL = "openai/gpt-5.4-mini"
MAX_TOKENS_JUDGE = 2000
TEMPERATURE_JUDGE = 0.1
```

Удалить из `judge.py` эти константы, заменить на импорт из config.

**Файл:** `d2/config.py`, `d2/judge.py`
**Риски:** нет, чистый перенос.

---

### Шаг 2. `models.py` — новые поля

**Файл:** `d2/models.py`

2a. В `Case` добавить:
```python
reference_routing: dict[str, Any] = Field(
    default_factory=dict,
    description="Эталонная маршрутизация: specialists, service_type, examination",
)
```

2b. В `SchemaRun` добавить:
```python
routing: dict[str, Any] = Field(
    default_factory=dict,
    description="Маршрутизация доктора (пост-диалоговый вывод)",
)
```

**Обратная совместимость:** `default_factory=dict` → существующие JSON без `routing` загрузятся корректно через `CaseResult.model_validate()`.

**Риски:** нет. Pydantic v2 `model_validate()` игнорирует отсутствующие поля с default.

---

### Шаг 3. `cases.py` — эталонная маршрутизация

**Файл:** `d2/cases.py`

Добавить `reference_routing` к каждому из 10 кейсов. Данные по стоматологическому протоколу:

| # | case_type | specialists | service_type | examination |
|---|-----------|------------|-------------|-------------|
| 1 | acute_pulpitis | эндодонтист | экстренное эндодонтическое лечение | прицельный рентген, ЭОД, термопроба |
| 2 | implantation | хирург-имплантолог, ортопед | дентальная имплантация + временное протезирование | КЛКТ, коагулограмма (МНО), консультация кардиолога |
| 3 | wisdom_tooth | хирург-стоматолог | экстренное удаление ретинированного зуба мудрости | ОПТГ, общий анализ крови |
| 4 | esthetics_composite | стоматолог-терапевт, ортопед | эстетическая реставрация, виниры/люминиры | прицельный рентген, фотопротокол, диагностические модели |
| 5 | child_trauma | детский стоматолог-хирург | шинирование, динамическое наблюдение | прицельный рентген, ЭОД (отложенная через 2-3 нед) |
| 6 | post_treatment_pain | эндодонтист | ревизия/перелечивание каналов | прицельный рентген, КЛКТ |
| 7 | periodontitis | пародонтолог | комплексное пародонтологическое лечение | ОПТГ, пародонтограмма, HbA1c |
| 8 | tmj_dysfunction | гнатолог (стоматолог-ортопед) | сплинт-терапия, коррекция окклюзии | МРТ ВНЧС, окклюзиография |
| 9 | missing_teeth_prosthetics | ортопед, хирург-имплантолог | несъёмное протезирование (мосты/импланты) | КЛКТ, ОПТГ, маркёры костного метаболизма |
| 10 | adversarial_complex | стоматолог-терапевт, пародонтолог | лечение кариеса + пародонтологическое лечение | прицельный рентген, ОПТГ, пародонтограмма |

**Риски:** эталон должен быть реалистичным. Неточные данные → некорректная оценка.

---

### Шаг 4. `prompts/routing_infer.md` — новый промпт

**Файл:** `d2/prompts/routing_infer.md` (НОВЫЙ)

Промпт для Qwen:
- **Вход:** JSON анкета (extracted данные из диалога)
- **Qwen НЕ видит** полный бэкграунд кейса (слепой вывод)
- **Выход:** JSON с тремя полями:
  - `specialists` — список специалистов
  - `service_type` — тип процедуры/услуги
  - `examination` — необходимые обследования
- Формат строгий: только JSON, без пояснений

**Ключевое:** промпт должен работать с любой структурой extracted (S1=4 поля, S2=до 11 полей, S3=произвольные ключи на русском).

---

### Шаг 5. `doctor.py` — функция `infer_routing()`

**Файл:** `d2/doctor.py`

Новая функция:
```python
def infer_routing(
    client: OpenRouterClient,
    extracted: dict[str, str],
) -> tuple[dict[str, Any], int]:
    """Определить маршрутизацию по собранным данным (слепой вывод).

    Возвращает (routing_dict, tokens_used).
    """
```

- Загружает `prompts/routing_infer.md`
- Формирует user message из extracted JSON
- Вызывает `client.chat()` с `DOCTOR_MODEL`, `TEMPERATURE_ROUTING`, `MAX_TOKENS_ROUTING`
- Парсит JSON → dict с ключами `specialists`, `service_type`, `examination`
- Используем тот же `_extract_json()` что уже есть в `doctor.py`

**Расположение:** в `doctor.py` рядом с `doctor_process()` — оба работают с Qwen.

**Почему не отдельный файл:** функция маленькая (~30 строк), использует тот же клиент и модель, логически — часть "доктора". Отдельный файл = лишняя абстракция.

---

### Шаг 6. `session.py` — интеграция вызова

**Файл:** `d2/session.py`

В `run_case_schema()` (строка 126-137), после цикла диалога и перед return:

```python
    duration = time.time() - t0
    turns = (len(dialog) + 1) // 2

    # Пост-диалоговый вывод маршрутизации
    routing, routing_tokens = infer_routing(client, accumulated)
    tokens_doctor += routing_tokens
    print(f"    🔀 Routing: {routing.get('specialists', [])}", flush=True)

    return SchemaRun(
        ...,
        routing=routing,
    )
```

**Точка вставки:** между `duration = time.time() - t0` и `return SchemaRun(...)`.

**Токены:** добавляются к `tokens_doctor` — это тот же доктор (Qwen).

**Импорт:** добавить `from d2.doctor import doctor_process, infer_routing`.

---

### Шаг 7. `run.py` — флаг `--routing-only`

**Файл:** `d2/run.py`

Новый флаг для дозапуска routing inference к существующим диалогам:
- Читает `case_XX.json` из `DIALOGS_DIR`
- Для каждого SchemaRun без `routing` — вызывает `infer_routing()`
- Перезаписывает JSON

Это позволяет НЕ перезапускать диалоги (30 сессий × API calls), а добавить routing к существующим результатам (10 кейсов × 3 схемы = 30 коротких вызовов).

---

### Шаг 8. `prompts/judge.md` — обновить промпт

**Файл:** `d2/prompts/judge.md`

Judge теперь получает:
1. **reference_routing** (из Case) — эталон
2. **doctor_routing** (из SchemaRun.routing) — вывод Qwen
3. **extracted** данные — для контекста
4. **Диалог** — для оценки качества

Judge НЕ определяет маршрутизацию сам — только сравнивает и оценивает.

Критерии:
- `routing_match` (bool) — совпадает ли маршрутизация доктора с эталоном
- `specialist_match` (bool) — правильный специалист
- `service_match` (bool) — правильная услуга
- `data_sufficiency` (0-10) — достаточно ли данных для маршрутизации
- `accuracy` (0-10) — точность извлечённых данных
- `dialogue_quality` (0-10) — качество ведения диалога

---

### Шаг 9. `judge.py` — обновить _build_judge_input()

**Файл:** `d2/judge.py`

`_build_judge_input(case, run)` → добавить секции:
```
## Эталонная маршрутизация
- Специалисты: ...
- Услуга: ...
- Обследования: ...

## Маршрутизация доктора (на основе собранных данных)
- Специалисты: ...
- Услуга: ...
- Обследования: ...
```

Также обновить `_parse_judge_response()` — добавить `specialist_match`, `service_match`.

---

### Шаг 10. `report.py` + `viz.py` — обновить отчёты

**report.py:**
- Сводная таблица: добавить Specialist Match %, Service Match %
- Детали кейсов: показывать reference vs doctor routing
- CSV: добавить колонки doctor_specialists, doctor_service, doctor_examination

**viz.py:**
- Heatmap: оставить data_sufficiency + routing_match
- Bar chart: добавить specialist_match и service_match как отдельные группы

---

## Порядок изменения файлов

```
1. config.py          ← +routing константы + перенос judge констант из judge.py
   judge.py           ← удалить локальные константы, импорт из config
2. models.py          ← +2 поля с defaults (обратная совместимость)
3. cases.py           ← +reference_routing × 10 кейсов
4. prompts/routing_infer.md  ← новый файл
5. doctor.py          ← +infer_routing()
6. session.py         ← +вызов infer_routing после диалога
7. run.py             ← +--routing-only флаг
8. prompts/judge.md   ← обновить промпт сравнения
9. judge.py           ← обновить _build_judge_input, парсинг
10. report.py         ← +doctor routing в отчёты
11. viz.py            ← минимальные обновления
```

## Гарантии

- **Обратная совместимость:** существующие JSON загружаются без ошибок (default_factory=dict)
- **Нет дубликатов:** routing inference — одна функция в doctor.py, одна точка вызова в session.py
- **Нет костылей:** чистое расширение моделей через Pydantic поля
- **Нет лишних схем:** routing — простой dict, не отдельная Pydantic модель
- **Переиспользование:** `_extract_json()` из doctor.py для парсинга routing ответа
- **Минимальность:** 1 новый файл (промпт), остальное — расширение существующих

## Затронутые файлы: 9 из 13

Не меняются: `schemas.py`, `patient.py`, `client.py`, `metrics.py`
