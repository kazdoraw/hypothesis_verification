# DS Experiments: AI Dentist

Экспериментальные notebooks для проверки гипотез проекта AI Dentist.
Текущий контур включает направления D1, D2 и D4.

## Гипотезы

### D1: Surface Classifier + Routing

**Цель:** Проверить, что поверхностный классификатор (rule-based + ML) может дополнить LLM для роутинга сообщений.

### D2: State Machine + JSON-Schema Intake

**Цель:** Проверить, что state machine с JSON Schema собирает анамнез полнее, чем свободный LLM-диалог.

### D4: FAQ Retrieval / Search Strategies

**Цель:** Сравнить стратегии поиска FAQ-ответов и выбрать устойчивый
пайплайн для интеграции в production-бота.


## Структура проекта

```
study/
├── README.md                           # Этот файл
├── requirements.txt                    # Python зависимости
│
├── D1_surface_classifier_routing.ipynb  # Эксперимент D1
├── D2_state_machine_intake.ipynb        # Эксперимент D2
├── D4_faq_search_comparison.ipynb       # Эксперимент D4
│
├── configs/
│   ├── together_config.yaml            # Конфигурация LLM (Together AI)
│   └── label_map.json                  # Маппинг меток классов
│
├── data/
│   ├── d1_messages.csv                 # Датасет для D1 (120 сообщений)
│   └── d2_cases.jsonl                  # Датасет для D2 (30 кейсов)
│
├── models/                             # Сохранённые ML модели
│
├── outputs/                            # Результаты экспериментов
│   ├── figures/                        # Графики (PNG)
│   ├── tables/                         # Метрики (CSV)
│   ├── diagrams/                       # Диаграммы (SVG/PNG)
│   └── reports/                        # Отчёты (MD)
│
├── d4/                                 # Материалы и стратегии D4
│   ├── d4.md
│   └── strategies/
│
└── utils/                              # Вспомогательные модули
    ├── __init__.py
    ├── llm.py                          # Together AI клиент
    ├── data.py                         # Генерация синтетических данных
    ├── metrics.py                      # Вычисление метрик
    ├── schemas.py                      # JSON Schema для анамнеза
    └── viz.py                          # Визуализации
```

## Установка

### 1. Создайте виртуальное окружение

```bash
cd study
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# или
.venv\Scripts\activate     # Windows
```

### 2. Установите зависимости

```bash
pip install -r requirements.txt
```

### 3. Настройте API ключ Together AI

Добавьте ключ в `configs/together_config.yaml`:

```yaml
provider: together
api_key_env: tgp_v1_YOUR_API_KEY_HERE  # Ваш ключ

models:
  router:
    name: meta-llama/Llama-3.3-70B-Instruct-Turbo
    temperature: 0.1
    max_tokens: 256
```

Или через переменную окружения:

```bash
export TOGETHER_API_KEY=tgp_v1_YOUR_API_KEY_HERE
```

## Запуск экспериментов

### Jupyter Lab

```bash
cd study
jupyter lab
```

Откройте нужный notebook:
- `D1_surface_classifier_routing.ipynb` — эксперимент D1
- `D2_state_machine_intake.ipynb` — эксперимент D2
- `D4_faq_search_comparison.ipynb` — эксперимент D4

### Порядок выполнения ячеек

1. **Setup** — импорты, seed, пути
2. **Data** — загрузка/генерация данных
3. **Baseline A** — LLM classification/dialog
4. **Proposed B** — ML classifier / State Machine
5. **Evaluation** — сравнение метрик
6. **Visualization** — графики и таблицы
7. **Export** — сохранение артефактов

## Notebooks

### D1: Surface Classifier + Routing

| Секция | Описание |
|--------|----------|
| Baseline A | LLM (Together AI) классифицирует intent напрямую |
| Proposed B | Rule-based правила + TF-IDF + ML модели |
| ML Models | LogisticRegression, LinearSVC, RandomForest |
| Metrics | Accuracy, F1 (macro/weighted/per-class), Confusion Matrix |
| Economics | LLM calls, tokens, cost estimation |

**Результаты сохраняются в:**
- `outputs/figures/d1_*.png` — графики
- `outputs/tables/d1_*.csv` — метрики
- `outputs/reports/D1_summary.md` — отчёт

### D2: State Machine + Intake

| Секция | Описание |
|--------|----------|
| JSON Schema | Строгая схема с условными полями по типу жалобы |
| State Machine | Отслеживает заполненные/пропущенные поля |
| Baseline A | LLM свободно собирает анамнез |
| Proposed B | State Machine направляет LLM |
| Metrics | Completion rate, turns, expert sufficient rate |

**Типы жалоб (complaint_type):**
- `acute_pain` — острая боль
- `chronic_pain` — хроническая боль
- `esthetics` — эстетика
- `ortho` — ортодонтия
- `therapy` — терапия

**Результаты сохраняются в:**
- `outputs/figures/d2_*.png` — графики
- `outputs/tables/d2_*.csv` — метрики
- `outputs/diagrams/d2_state_machine.*` — диаграмма
- `outputs/reports/D2_summary.md` — отчёт

### D4: FAQ Search / Retrieval

| Секция | Описание |
|--------|----------|
| Dataset prep | Подготовка пар вопрос-ответ и сценариев валидации |
| Strategy compare | Сравнение keyword, template и гибридных подходов |
| Evaluation | Precision@K, Recall@K, F1 и ошибки ранжирования |
| Reporting | Формирование отчётов в `outputs/reports` |

**Материалы D4:**
- `d4/d4.md` — детальное описание эксперимента
- `d4/strategies/` — реализации стратегий поиска
- `outputs/reports/` — отчёты и дорожные карты по улучшениям

## Utils модули

### llm.py

```python
from utils import TogetherLLM

llm = TogetherLLM()  # Автоматически загружает ключ из конфига

# D1: Классификация intent
result = llm.classify_intent("Хочу записаться на приём")
# {'intent': 'booking', 'confidence': 0.9, 'tokens_used': 196}

# D2: Сбор анамнеза
result = llm.collect_anamnesis_free(case, history)
# {'response': '...', 'extracted_fields': {...}, 'tokens_used': 512}
```

### data.py

```python
from utils import load_or_generate_d1, load_or_generate_d2

# Загрузить или сгенерировать D1 датасет
df = load_or_generate_d1("data/d1_messages.csv", n=120)

# Загрузить или сгенерировать D2 кейсы
cases = load_or_generate_d2("data/d2_cases.jsonl", n=30)
```

### metrics.py

```python
from utils import compute_classification_metrics, compute_intake_metrics

# D1 метрики
metrics = compute_classification_metrics(y_true, y_pred, labels)

# D2 метрики
metrics = compute_intake_metrics(results_a, results_b)
```

### viz.py

```python
from utils import plot_confusion_matrix, plot_f1_by_class

# Confusion matrix
plot_confusion_matrix(y_true, y_pred, labels, save_path="outputs/figures/cm.png")

# F1 по классам
plot_f1_by_class(metrics, save_path="outputs/figures/f1.png")
```

### schemas.py

```python
from utils.schemas import INTAKE_SCHEMA, validate_intake, get_required_fields

# Получить обязательные поля для типа жалобы
required = get_required_fields("acute_pain")
# ['complaint_type', 'complaint_text', 'pain_intensity', 'pain_duration', ...]

# Валидировать данные
is_valid, errors = validate_intake(data)
```

## Синтетические данные

Данные генерируются автоматически при первом запуске:

- **d1_messages.csv** — 120 сообщений (по 20 на класс)
- **d2_cases.jsonl** — 30 кейсов (по 6 на тип жалобы)

Все записи помечены `source="synthetic"`.

## LLM Provider

Используется **Together AI** с моделью `meta-llama/Llama-3.3-70B-Instruct-Turbo`.

Конфигурация в `configs/together_config.yaml`:

```yaml
models:
  router:           # Для D1 классификации
    name: meta-llama/Llama-3.3-70B-Instruct-Turbo
    temperature: 0.1
    max_tokens: 256
  
  dialog:           # Для D2 диалогов
    name: meta-llama/Llama-3.3-70B-Instruct-Turbo
    temperature: 0.3
    max_tokens: 512
  
  fast:             # Быстрая модель для тестов
    name: meta-llama/Llama-3.2-3B-Instruct-Turbo
    temperature: 0.1
    max_tokens: 128
```

## Ожидаемые результаты

### D1

| Метрика | Baseline A (LLM) | Proposed B (ML) | Цель |
|---------|------------------|-----------------|------|
| F1 (macro) | ~0.75-0.80 | >= 0.85 | >= 0.85 |
| LLM calls | 1 per msg | 0 | -100% |
| Tokens | ~200 per msg | 0 | -100% |

### D2

| Метрика | Baseline A (Free) | Proposed B (SM) | Цель |
|---------|-------------------|-----------------|------|
| Required completion | 60-70% | >= 90% | >= 90% |
| Avg turns | ~5 | ~6-7 | +1-2 |
| Expert sufficient | ~50% | >= 70% | >= 70% |

## Troubleshooting

### API key not found

```
Warning: No API key found. Falling back to simulator.
```

**Решение:** Добавьте ключ в `configs/together_config.yaml` или установите переменную окружения `TOGETHER_API_KEY`.

### Import errors

```
ModuleNotFoundError: No module named 'sklearn'
```

**Решение:** Установите зависимости:
```bash
pip install -r requirements.txt
```

### Graphviz not found

```
ExecutableNotFound: failed to execute 'dot'
```

**Решение:** Установите Graphviz:
```bash
# macOS
brew install graphviz

# Ubuntu
sudo apt install graphviz

# Windows
choco install graphviz
```

