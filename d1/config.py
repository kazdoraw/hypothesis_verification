"""Конфигурация эксперимента D1 v6: модели, пути, константы.

Паттерн аналогичен d2/config.py — централизованные константы.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent
ONTOLOGY_PATH = ROOT_DIR / "ontology" / "route_domain.yaml"
DATA_DIR = ROOT_DIR / "data"
PROMPTS_DIR = ROOT_DIR / "prompts"
SCRIPTS_DIR = ROOT_DIR / "scripts"
RESULTS_DIR = ROOT_DIR / "results"
REPORTS_DIR = RESULTS_DIR / "reports"

# ---------------------------------------------------------------------------
# LLM модели (через OpenRouter)
# ---------------------------------------------------------------------------
VARIATION_MODEL = "x-ai/grok-4.1-fast"

# ---------------------------------------------------------------------------
# Параметры генерации вариаций
# ---------------------------------------------------------------------------
VARIATIONS_PER_SEED = 10
TEMPERATURE_VARIATION = 0.9
MAX_TOKENS_VARIATION = 1500

# ---------------------------------------------------------------------------
# Embedding модели для baselines
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_PRIMARY = "ai-forever/ru-en-RoSBERTa"
EMBEDDING_MODEL_COMPARISON = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Параметры split
# ---------------------------------------------------------------------------
SPLIT_RANDOM_STATE = 42
TEST_RATIO = 0.20
VAL_RATIO = 0.125   # от оставшегося после test: 0.125 * 0.8 ≈ 0.10

# ---------------------------------------------------------------------------
# Файлы данных (имена выходных CSV)
# ---------------------------------------------------------------------------
SEEDS_FILE = DATA_DIR / "d1_v6_seeds.yaml"
HARD_CASES_FILE = DATA_DIR / "d1_v6_hard_cases.yaml"

DATASET_PREFIX = "d1_v6"
SPLIT_NAMES = ["full", "train", "val", "test", "hard_test", "switch_test"]

# ---------------------------------------------------------------------------
# CSV колонки
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "id", "text", "route_domain", "subtype",
    "explicit_booking", "urgency", "is_offtopic",
    "specialization_hint", "feedback_flag", "source", "seed_id",
]
