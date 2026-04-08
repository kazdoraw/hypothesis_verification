"""Конфигурация эксперимента D2: константы, модели, пути."""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Пути ---
ROOT_DIR = Path(__file__).parent
PROMPTS_DIR = ROOT_DIR / "prompts"
RESULTS_DIR = ROOT_DIR / "results"
DIALOGS_DIR = RESULTS_DIR / "dialogs"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# --- API ---
load_dotenv(ROOT_DIR.parent / ".env")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# --- Модели ---
DOCTOR_MODEL = "qwen/qwen3-235b-a22b-2507"
PATIENT_MODEL = "x-ai/grok-4.1-fast"

# --- Параметры диалога ---
MAX_TURNS = 8          # максимум пар вопрос-ответ
MAX_TOKENS_DOCTOR = 300
MAX_TOKENS_PATIENT = 80
MAX_TOKENS_EXTRACTION = 1000
TEMPERATURE_DOCTOR = 0.4
TEMPERATURE_PATIENT = 0.9

# --- Routing inference ---
MAX_TOKENS_ROUTING = 1000
TEMPERATURE_ROUTING = 0.1

# --- Judge ---
JUDGE_MODEL = "openai/gpt-5.4-mini"
MAX_TOKENS_JUDGE = 2000
TEMPERATURE_JUDGE = 0.1
ROUTING_MATCH_THRESHOLD = 5  # specialist/service/exam score >= этого = routing_match

# --- Метрики ---
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SEMANTIC_THRESHOLD = 0.70
