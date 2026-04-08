"""Pydantic модели данных для эксперимента D2.

Ключевые модели:
- Case: сценарий пациента
- Message: одно сообщение в диалоге
- SchemaRun: результат одного прогона (1 схема × 1 кейс)
- CaseResult: все 3 прогона одного кейса
"""

from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    """Сценарий пациента для одного кейса."""

    case_id: int
    case_type: str = Field(description="Тип жалобы (acute_pain, pulpitis, ...)")
    summary: str = Field(description="Краткое описание ситуации для логов")
    situation: str = Field(description="Полное описание ситуации пациента для промпта")
    reference_fields: dict[str, str] = Field(
        default_factory=dict,
        description="Ожидаемые значения полей из сценария (для метрик)",
    )
    reference_routing: dict[str, Any] = Field(
        default_factory=dict,
        description="Эталонная маршрутизация: specialists, service_type, examination",
    )


class Message(BaseModel):
    """Одно сообщение в диалоге."""

    role: str = Field(description="patient | doctor")
    text: str


class SchemaRun(BaseModel):
    """Результат одного прогона: один кейс × одна схема.

    Содержит полный диалог и инкрементально собранные данные.
    """

    schema_name: str
    dialog: list[Message] = Field(default_factory=list)
    extracted: dict[str, str] = Field(
        default_factory=dict,
        description="Финальный набор извлечённых полей",
    )
    routing: dict[str, Any] = Field(
        default_factory=dict,
        description="Маршрутизация доктора (пост-диалоговый вывод)",
    )
    turns: int = 0
    tokens_doctor: int = 0
    tokens_patient: int = 0
    duration_s: float = 0.0


class CaseResult(BaseModel):
    """Результат одного кейса: 3 прогона (S1, S2, S3)."""

    case_id: int
    case_type: str
    patient_prompt_summary: str
    runs: dict[str, SchemaRun] = Field(
        default_factory=dict,
        description="Прогоны по схемам: {'S1': SchemaRun, 'S2': ..., 'S3': ...}",
    )
