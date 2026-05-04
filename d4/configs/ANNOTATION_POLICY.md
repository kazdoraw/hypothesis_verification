# Annotation Policy — eval_set_dev_v2 / eval_set_test_v2

## Цель

Формализовать правила разметки `gold_facts`, `expected_doctor`, `gold_chunk_ids`,
чтобы eval contract был воспроизводимым и не подвержен post-hoc подгонке.

---

## Правила по subtype

### `pricing/specific_service`

- Обязателен минимум **1 `fact_type: price`** с числовым `canonical_value`.
- Category-slug (`primary appointment & consultations`, `implantation`, etc.)
  как единственный gold_fact **запрещён**.
- Если запрос о конкретной услуге → 1–4 цены из чанка (ключевые позиции).
- Если запрос общий ("осмотр платный?") → минимальная + типичная цена.

### `pricing/category_overview`

- Допустимы `fact_type: text` без числовых цен, если запрос о наличии категории.
- Если в ответе ожидаются конкретные цены → см. правило `specific_service`.

### `pricing/price_comparison`

- Минимум 2 `fact_type: price` для обоих сравниваемых объектов.

### `aftercare/*`

- Обязательны **2+ actionable facts**, отвечающих на вопрос пользователя.
- Одно общее слово ("температура", "отёк", "восстановление", "чувствительность")
  как единственный gold_fact **запрещено**.
- Факты должны отражать ядро рекомендации, а не побочный факт из того же чанка.

### `reasoning/multi_fact`

- Если gold_facts содержат уникальный `fact_type: fio` →
  `expected_doctor` **обязателен**.
- Если запрос допускает множественный ответ (сравнение, список) →
  `expected_doctor` = `null`, но gold_facts покрывают ключевую сущность.

### `out_of_scope/*`

- `answerable: false` обязателен.
- `gold_facts` могут быть пустыми или содержать reason-маркер.

### `missing_info`

- `answerable: false` обязателен.
- Gold строится вокруг отсутствия запрошенного атрибута,
  а не вокруг косвенных фактов из KB.

---

## Принцип "не подгонять"

> gold_facts описывают, что **должен содержать правильный ответ**,
> а не как **текущая лучшая модель формулирует ответ**.

Запрещено:
- Расширять gold после каждого miss, чтобы "и это тоже считалось".
- Исправлять только samples, просевшие в pilot, не применяя правило ко всему subtype.
- Подбирать canonical_value под формулировку конкретной стратегии.

Разрешено:
- Убирать явно дефектные аннотации (category-slug вместо цены, одно слово вместо рекомендации).
- Добавлять `expected_doctor`, если gold уже содержит уникальный FIO.
- Уточнять слишком слабые факты, если старый gold не отвечал на вопрос пользователя.

---

## Changelog

### 2026-04-08 — Системный аудит v1

**Триггер**: pilot_dev_30 показал, что FMR частично измеряет качество аннотации,
а не качество стратегии.

**Метод**: не точечные фиксы по pilot failures, а полный subtype-level аудит:
- Все `pricing/specific_service` проверены на наличие category-level gold.
- Все `aftercare/*` проверены на слабые однословные gold_facts.
- Все `reasoning/multi_fact` проверены на наличие expected_doctor при уникальном FIO.

**Правки (22 sample):**

| sample_id | Тип правки | Было | Стало |
|-----------|-----------|------|-------|
| q_0073 | pricing gold | `text: primary appointment & consultations` | `price: 500` + `text: детский` |
| q_0228 | pricing gold | `text: primary appointment & consultations` | `price: 500` + `text: детский стоматолог` |
| q_0074 | pricing gold | `text: primary appointment & consultations` | `price: 300` + `price: 390` |
| q_0215 | pricing gold | `text: adult surgery (removals)` | `price: 4900` + `price: 6900` |
| q_0219 | pricing gold | `text: adult surgery (removals)` | `price: 6000` + `price: 4900` + `price: 6900` |
| q_0221 | pricing gold | `text: implantation` | `price: 35000` + `price: 65000` + `price: 140000` + `price: 250000` |
| q_0227 | pricing gold | `text: primary appointment & consultations` | `price: 500` + `text: детский` |
| q_0234 | pricing gold | `text: primary appointment & consultations` | `price: 390` + `price: 300` |
| q_0237 | pricing gold | `text: primary appointment & consultations` | `price: 300` + `price: 390` |
| q_0238 | pricing gold | `text: primary appointment & consultations` | `price: 300` + `price: 390` |
| q_ac_02 | aftercare gold | `text: нимесил` | 3 запретительных факта |
| q_ac_05 | aftercare gold | `text: восстановление` | 3 post-anesthesia факта |
| q_ac_06 | aftercare gold | `text: чувствительность` | 3 care-инструкции |
| q_ac_15 | aftercare gold + chunks | `text: температура` | `text: до 37,5` + `text: первые сутки` + доп. chunk |
| q_ac_16 | aftercare gold | `text: отёк` | 3 конкретных факта |
| q_0057 | expected_doctor | `null` | `Заволжский Александр Михайлович` |
| q_0058 | expected_doctor + gold | `null` + label fix | `Молгачева Оксана Александровна` + стаж |
| q_0059 | expected_doctor | `null` | `Васильева Елена Владимировна` |
| q_0194 | expected_doctor + gold | `null` + label fix | `Молгачева Оксана Александровна` + стаж |
| q_0202 | expected_doctor | `null` | `Васильева Елена Владимировна` |
| q_0203 | expected_doctor | `null` | `Васильева Елена Владимировна` |
| q_0204 | expected_doctor | `null` | `Заволжский Александр Михайлович` |

**Верификация**: после правок 0 category-level gold в pricing, 0 reasoning без expected_doctor,
0 aftercare с одним слабым фактом. YAML загружается: 137 samples.

### 2026-04-08 — Применение policy к eval_set_test_v2

**Метод**: тот же subtype-level аудит, что и для dev.

**Результаты аудита test (51 sample):**
- pricing/specific_service: 0 нарушений (обе CONCRETE)
- aftercare: 0 WEAK
- reasoning: 2 MISSING → исправлены

**Правки (2 sample):**

| sample_id | Тип правки | Было | Стало |
|-----------|-----------|------|-------|
| q_0213 | expected_doctor | `null` | `Заволжский Александр Михайлович` |
| q_0214 | expected_doctor | `null` | `Заволжский Александр Михайлович` |

**Верификация**: 51 samples, 0 нарушений policy.
