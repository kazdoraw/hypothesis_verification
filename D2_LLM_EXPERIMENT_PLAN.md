# D2: LLM vs LLM — План эксперимента

## Цель

Определить оптимальную схему извлечения анамнеза для production через реальные LLM-диалоги.

**Doctor** (Qwen 3 235B) — ведёт приём, задаёт вопросы, извлекает данные.  
**Patient** (Grok) — жалуется в свободной форме, имитирует реального человека.

Паттерн взаимодействия: `dialog_generator/poc.py` — turn-by-turn, промпты в отдельных файлах.

## Что измеряем

1. **Какие поля реально извлекаются** из свободного диалога
2. **Оптимальная схема** — фиксированная vs адаптивная vs свободная
3. **Качество извлечения** — embedding similarity extracted vs reference
4. **Эффективность** — turns, tokens, cost

## Ключевой принцип: Extraction по ходу диалога

Данные извлекаются **на каждом ходу**, а НЕ после окончания диалога.
Qwen возвращает **structured output** на каждый ответ пациента:

```json
{
  "extracted": {"symptoms": "...", "localization": "..."},
  "next_question": "Как давно это беспокоит?",
  "is_complete": false,
  "reasoning": "Не хватает duration и chronic_or_allergies"
}
```

Это соответствует production подходу: агент собирает анамнез инкрементально.

---

## Файловая структура

```
study/d2/                         # ← ВСЁ внутри study/d2/
├── __init__.py
├── run.py                        # Entry point: python -m d2.run
├── config.py                     # Константы, модели, API
├── client.py                     # OpenRouter HTTP клиент (httpx)
├── models.py                     # Pydantic модели: Case, Message, CaseResult
├── schemas.py                    # 3 extraction schema + DoctorTurn (Pydantic)
├── patient.py                    # Grok patient agent
├── doctor.py                     # Qwen doctor: structured output (extract + ask)
├── session.py                    # Один диалог doctor↔patient
├── metrics.py                    # Embedding similarity
├── report.py                     # MD + CSV export
├── viz.py                        # Графики
├── prompts/                      # Промпты в .md файлах
│   ├── patient.md                # System prompt пациента
│   └── doctor.md                 # System prompt доктора (+ extraction инструкции)
└── results/                      # Результаты
    ├── dialogs/                  # JSON: один файл = один кейс
    │   ├── case_01.json          # dialog + incremental extractions
    │   ├── case_02.json
    │   └── ...
    ├── figures/                  # PNG графики
    └── reports/                  # MD отчёты + CSV
```

**12 .py файлов + 2 промпта**, каждый < 150 строк.

---

## Стек

| Компонент | Технология | Почему |
|-----------|-----------|--------|
| Doctor LLM | `qwen/qwen3-235b-a22b-2507` via OpenRouter | Production model |
| Patient LLM | **`x-ai/grok-4.1-fast`** via OpenRouter | Быстрый, разговорный стиль |
| LLM Client | `httpx` (как poc.py) | Прямой контроль, retry, token tracking |
| Schema / Models | **Pydantic v2** | Structured output валидация |
| Metrics | `sentence-transformers` | Semantic similarity |
| Viz | `matplotlib` | Уже установлен |
| Runner | `argparse` CLI | `python -m d2.run` |

**Зависимости**: httpx, pydantic, sentence-transformers, matplotlib, python-dotenv.

---

## 10 Patient Cases

Кейсы определяются **только через промпт** — никаких scripted ответов. Patient-LLM (Grok) получает system prompt с описанием ситуации и отвечает **свободно**, как реальный человек в мессенджере.

**Все 10 типов уникальные** — ни один не повторяется.

| # | Тип | System prompt пациента (суть) |
|---|-----|-------------------------------|
| 1 | Острая боль | Сильно болит зуб справа внизу 2 дня, от холодного хуже. Здоров. |
| 2 | Пульпит | Пульсирует зуб слева, ночью не спал. Диабет 2 типа. |
| 3 | Хроническая боль | Ноет под старой коронкой 3 месяца. Аллергия на лидокаин. |
| 4 | Эстетика (виниры) | Хочет виниры, зона улыбки. Долго копил. Здоров. |
| 5 | Отбеливание | Потемнели зубы, курит 10 лет. Стесняется улыбаться. |
| 6 | Ортодонтия (ребёнок) | У ребёнка 12 лет кривые передние. Мама пишет. |
| 7 | Выпала пломба | Выпала пломба, дырка. Диабет, метформин. |
| 8 | Кровоточивость дёсен | Дёсны кровят при чистке полгода. Курит. |
| 9 | Adversarial: размытый | Болит "где-то", путает стороны, не помнит когда. Неохотно отвечает. |
| 10 | Adversarial: отвлекающий | Болит зуб + спрашивает про цены, расписание, парковку. Гипертония. |

### Формат system prompt (пример Case 1)

Используем паттерн из `dialog_generator/prompts/patient.md`:

```markdown
Ты — пациент, пишешь в чат стоматологической клиники.

Твоя ситуация: два дня назад начал сильно болеть зуб справа внизу, 
жевательный. Особенно от холодной воды. Обезболивающее помогает на пару часов. 
Ты здоров, аллергий нет.

Правила:
- Пиши как обычный человек в мессенджере: 1-2 предложения
- НЕ используй медицинские термины
- НЕ добавляй информацию, которой нет в ситуации
- Если спрашивают то, чего не знаешь — скажи "не знаю" / "не помню"
- Можешь переспрашивать, сомневаться
```

**Grok сам генерирует** первое сообщение и все ответы. Нет заранее заготовленных фраз.

---

## 3 Schema Designs (Pydantic)

Каждая схема определяет **какие поля собирать по ходу диалога**. Qwen на каждом ходу извлекает данные инкрементально и решает, какой вопрос задать следующим.

### S1: Fixed 4 (production-aligned)

```python
class S1Extraction(BaseModel):
    """Фиксированные 4 поля — текущая production схема."""
    symptoms: str = Field(default="", description="Основная жалоба")
    localization: str = Field(default="", description="Где болит/беспокоит")
    duration: str = Field(default="", description="Как давно")
    chronic_or_allergies: str = Field(default="", description="Хронические заболевания, аллергии")
```

### S2: Adaptive (базовые + расширения по типу жалобы)

```python
class S2Extraction(BaseModel):
    """Базовые 4 + дополнительные поля по типу жалобы."""
    complaint_type: str = Field(default="", description="Определённый тип жалобы")
    symptoms: str = Field(default="")
    localization: str = Field(default="")
    duration: str = Field(default="")
    chronic_or_allergies: str = Field(default="")
    # Адаптивные (заполняются если релевантны)
    intensity: Optional[str] = None
    triggers: Optional[str] = None
    onset: Optional[str] = None
    previous_treatment: Optional[str] = None
    desired_outcome: Optional[str] = None
    medications: Optional[str] = None
```

### S3: Free extraction (LLM сам решает)

```python
class S3Extraction(BaseModel):
    """LLM сам определяет набор клинически важных полей."""
    complaint_type: str = Field(default="")
    fields: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(default=0.0)
    missing_info: list[str] = Field(default_factory=list)
```

### DoctorTurn: Structured output на каждом ходу

```python
class DoctorTurn(BaseModel):
    """Structured output доктора на каждый ход."""
    extracted: dict[str, str] = Field(description="Поля, извлечённые/обновлённые на этом ходу")
    next_question: str = Field(description="Следующий вопрос пациенту (текст для отображения)")
    is_complete: bool = Field(description="True если анамнез собран достаточно")
    reasoning: str = Field(description="Почему задаём этот вопрос / почему завершаем")
```

**Сравнение**: 3 отдельных прогона одного кейса — каждый со своей схемой → 3 диалога → сравнение.

---

## Архитектура

```
Case (prompt)
    │
    ├─── Schema S1 ──┐
    ├─── Schema S2 ──┤  3 отдельных диалога на кейс
    └─── Schema S3 ──┘
                     │
                     ▼
         ┌─────────────────────────┐
         │  session.py: turn loop  │
         │                         │
         │  Patient msg             │
         │    → Doctor structured   │
         │      output:             │
         │      {extracted,         │
         │       next_question,     │
         │       is_complete}       │
         │    → Patient responds    │
         │    → repeat...           │
         └────────────┬────────────┘
                      │
                 case_XX.json
                 (3 диалога + 3 extraction snapshots)
```

### Поток session.py

```python
def run_case_schema(client, case, schema_name) -> SchemaRun:
    """Один прогон кейса с конкретной схемой."""
    dialog = []
    accumulated = {}  # инкрементальный сбор данных
    
    # 1. Patient: первое сообщение
    initial = patient_speak(client, case, dialog)
    dialog.append(Message(role="patient", text=initial))
    
    # 2. Turn loop
    for turn in range(MAX_TURNS):
        # Doctor: structured output — extract + next question
        doctor_turn = doctor_process(client, dialog, accumulated, schema_name)
        accumulated.update(doctor_turn.extracted)
        
        dialog.append(Message(role="doctor", text=doctor_turn.next_question))
        
        if doctor_turn.is_complete:
            break
        
        # Patient: ответ
        response = patient_speak(client, case, dialog)
        dialog.append(Message(role="patient", text=response))
    
    return SchemaRun(schema=schema_name, dialog=dialog, extracted=accumulated)

def run_case(client, case) -> CaseResult:
    """Полный прогон кейса: 3 диалога (по одному на схему)."""
    runs = {}
    for schema_name in ["S1", "S2", "S3"]:
        runs[schema_name] = run_case_schema(client, case, schema_name)
    return CaseResult(case=case, runs=runs)
```

### Формат case_XX.json

```json
{
  "case_id": 1,
  "case_type": "acute_pain",
  "patient_prompt_summary": "Боль справа внизу 2 дня, от холодного",
  "runs": {
    "S1": {
      "dialog": [
        {"role": "patient", "text": "Здравствуйте, зуб разболелся сильно..."},
        {"role": "doctor", "text": "Где именно болит?"},
        {"role": "patient", "text": "Справа внизу, жевательный"},
        {"role": "doctor", "text": "Как давно беспокоит?"},
        ...
      ],
      "extracted": {
        "symptoms": "сильная боль в зубе",
        "localization": "нижний правый моляр",
        "duration": "2 дня",
        "chronic_or_allergies": "нет"
      },
      "turns": 4,
      "tokens": 1800
    },
    "S2": { ... },
    "S3": { ... }
  }
}
```

---

## Метрики

### 1. Completeness Score
Сколько non-null полей извлечено / сколько полей в схеме.

### 2. Extraction Quality
Embedding similarity извлечённых значений vs reference (из сценария).  
`paraphrase-multilingual-MiniLM-L12-v2`, threshold=0.70.

### 3. Efficiency
- Turns
- Tokens (doctor + patient)
- Cost estimate

### 4. Schema Comparison
- S1 vs S2 vs S3: completeness, quality, tokens
- S3: какие поля LLM извлекает без подсказки
- S2 adaptive: какие доп. поля реально полезны

---

## CLI

```bash
# Полный прогон
python -m d2.run --cases all

# Один кейс для отладки  
python -m d2.run --cases 1 --verbose

# Только extraction на существующих диалогах (re-extract)
python -m d2.run --extract-only

# Только отчёт + визуализация
python -m d2.run --report-only
```

---

## Порядок реализации

| # | Файл | Описание | ~строк | Статус |
|---|------|----------|--------|--------|
| 1 | `config.py` | Константы, модели | 36 | ✅ |
| 2 | `models.py` | Pydantic: Case, Message, SchemaRun, CaseResult | 61 | ✅ |
| 3 | `schemas.py` | S1, S2, S3 + DoctorTurn structured output | 91 | ✅ |
| 4 | `client.py` | OpenRouter client (httpx, retry, token tracking) | 97 | ✅ |
| 5 | `prompts/*.md` | patient.md, doctor.md | 2 файла | ✅ |
| 6 | `cases.py` | 10 уникальных клинических сценариев | 155 | ✅ |
| 7 | `patient.py` | Grok patient agent | 54 | ✅ |
| 8 | `doctor.py` | Qwen: structured output (extract + ask per turn) | 169 | ✅ |
| 9 | `session.py` | Turn loop + incremental extraction + save JSON | 109 | ✅ |
| 10 | `run.py` | CLI entry point + --report-only | 116 | ✅ |
| 11 | `metrics.py` | Embedding similarity + S3 best-match | 115 | ✅ |
| 12 | `viz.py` | 3 графика (completeness, similarity, efficiency) | 118 | ✅ |
| 13 | `report.py` | MD + CSV export | 130 | ✅ |

**Итого: ~1251 строк** Python + 2 промпта. MVP протестирован на Case 1.

---

## Чего НЕ будет

- ❌ Notebook (ipynb)
- ❌ Scripted patient answers
- ❌ Gold standard / подгонка данных
- ❌ SequenceMatcher / regex extraction
- ❌ Monte Carlo симуляции
- ❌ State Machine для doctor
- ❌ Лишние абстракции и overengineering

---

## Ожидаемые результаты

1. **Оптимальная схема** для production: S1, S2, или S3
2. **Какие поля LLM реально извлекает** без подсказки (S3)
3. **Cost per intake** в tokens/руб
4. **Слабые места** по типам жалоб
5. **Production рекомендация** — конкретная Pydantic схема + промпт для `ai-core`
