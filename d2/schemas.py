"""Схемы извлечения анамнеза + structured output доктора (Pydantic v2).

S1 — фиксированные 4 поля (production-aligned).
S2 — адаптивная: базовые 4 + доп. поля по типу жалобы.
S3 — свободная: LLM сам определяет набор полей.
DoctorTurn — structured output на каждый ход диалога.
"""

from typing import Optional

from pydantic import BaseModel, Field


# --- DoctorTurn: structured output на каждый ход ---

class DoctorTurn(BaseModel):
    """Structured output доктора на каждый ход диалога.

    Qwen возвращает JSON с этой структурой после каждого ответа пациента.
    """

    extracted: dict[str, str] = Field(
        default_factory=dict,
        description="Поля, извлечённые или обновлённые из последнего сообщения пациента",
    )
    next_question: str = Field(
        description="Следующий вопрос пациенту (обычный текст, не JSON)",
    )
    is_complete: bool = Field(
        default=False,
        description="True если анамнез собран достаточно полно",
    )
    reasoning: str = Field(
        default="",
        description="Почему задаём этот вопрос / почему завершаем сбор",
    )


# --- S1: Фиксированная схема (4 поля) ---

class S1Extraction(BaseModel):
    """Фиксированные 4 поля — текущая production схема."""

    symptoms: str = Field(default="", description="Основная жалоба пациента")
    localization: str = Field(default="", description="Где именно болит или беспокоит")
    duration: str = Field(default="", description="Как давно беспокоит")
    chronic_or_allergies: str = Field(default="", description="Хронические заболевания и аллергии")


# --- S2: Адаптивная схема (базовые + расширения) ---

class S2Extraction(BaseModel):
    """Базовые 4 поля + дополнительные по типу жалобы."""

    complaint_type: str = Field(default="", description="Определённый тип жалобы")
    symptoms: str = Field(default="", description="Основная жалоба")
    localization: str = Field(default="", description="Локализация")
    duration: str = Field(default="", description="Длительность")
    chronic_or_allergies: str = Field(default="", description="Хронические заболевания и аллергии")
    # Адаптивные поля (заполняются если релевантны)
    intensity: Optional[str] = Field(None, description="Интенсивность боли (1-10)")
    triggers: Optional[str] = Field(None, description="Что усиливает симптомы")
    onset: Optional[str] = Field(None, description="Что спровоцировало")
    previous_treatment: Optional[str] = Field(None, description="Предыдущее лечение")
    desired_outcome: Optional[str] = Field(None, description="Желаемый результат")
    medications: Optional[str] = Field(None, description="Текущие лекарства")


# --- S3: Свободная схема (LLM сам решает) ---

class S3Extraction(BaseModel):
    """LLM сам определяет набор клинически важных полей."""

    complaint_type: str = Field(default="", description="Тип жалобы")
    fields: dict[str, str] = Field(
        default_factory=dict,
        description="Все клинически важные данные из диалога (ключ: название, значение: данные)",
    )
    confidence: float = Field(default=0.0, description="Уверенность в полноте 0.0-1.0")
    missing_info: list[str] = Field(
        default_factory=list,
        description="Что не удалось узнать из диалога",
    )


# --- Реестр схем ---

SCHEMAS: dict[str, type[BaseModel]] = {
    "S1": S1Extraction,
    "S2": S2Extraction,
    "S3": S3Extraction,
}
