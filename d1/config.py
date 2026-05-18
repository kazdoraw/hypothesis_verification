"""Конфигурация эксперимента D1 v6: модели, пути, константы.

Паттерн аналогичен d2/config.py — централизованные константы.
"""

from __future__ import annotations

from pathlib import Path


def resolve_model_path(model_id: str) -> str:
    """Резолв HF model ID → локальный snapshot path (без HTTP).

    Использует huggingface_hub.snapshot_download(local_files_only=True).
    Если модель — уже локальная директория, возвращает as-is.

    Зачем: transformers >=4.46 _patch_mistral_regex вызывает
    huggingface_hub.model_info() (HTTP GET) даже с local_files_only=True,
    если path не является os.path.isdir(). snapshot_download возвращает
    реальный путь к кэшу, для которого isdir()=True.

    Raises:
        FileNotFoundError: модель не найдена в локальном кэше.
    """
    import os

    if os.path.isdir(model_id):
        return model_id

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(model_id, local_files_only=True)
    except (LocalEntryNotFoundError, FileNotFoundError) as exc:
        raise FileNotFoundError(
            f"Модель '{model_id}' не найдена в локальном кэше HF Hub. "
            f"Скачайте: python -c \"from huggingface_hub import snapshot_download; "
            f"snapshot_download('{model_id}')\"",
        ) from exc

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
MIN_VARIATIONS_WARN = 5  # предупреждение если LLM вернул меньше
TEMPERATURE_VARIATION = 0.9
MAX_TOKENS_VARIATION = 1500
COSINE_DEDUP_THRESHOLD = 0.85  # внутри домена: similarity > порога → дубль

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
SEED_AUDIT_FILE = DATA_DIR / "seed_audit.csv"

DATASET_PREFIX = "d1_v6"
SPLIT_NAMES = [
    "full", "train", "val", "test",
    "hard_test", "safety_set", "blind_test", "switch_test",
]

# ---------------------------------------------------------------------------
# Entity-held-out: unseen сущности для test (план §4.1)
# ---------------------------------------------------------------------------
ENTITY_HELD_OUT = {
    "doctor_names": ["Демин", "Конновск"],
    "services": ["элайнер", "синус-лифтинг", "вкладк"],
    "specializations": ["гнатолог", "пародонтолог"],
}

# ---------------------------------------------------------------------------
# Extended eval: policy/subjective FAQ кейсы (отделяются от core splits)
# ---------------------------------------------------------------------------
EXTENDED_FAQ_CATEGORIES = {"policy_only", "subjective"}

# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------
LEAKAGE_COSINE_THRESHOLD = 0.92  # макс similarity train↔test

# ---------------------------------------------------------------------------
# CSV колонки
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "id", "text", "route_domain", "subtype",
    "explicit_booking", "urgency", "is_offtopic",
    "specialization_hint", "feedback_flag", "faq_category",
    "style", "source", "seed_id",
]

# switch_test.csv расширяет базовый контракт контекстной колонкой
# `active_domain` (домен предыдущей реплики ассистента). Используется ТОЛЬКО
# при загрузке switch_test и для построения per-transition breakdown в
# SwitchStressReport. Остальные eval-сеты работают с CSV_COLUMNS без изменений.
SWITCH_CSV_COLUMNS = CSV_COLUMNS + ["active_domain"]
