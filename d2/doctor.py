"""Doctor agent — Qwen собирает анамнез через structured output.

На каждый ход возвращает JSON: {extracted, next_question, is_complete, reasoning}.
Данные извлекаются инкрементально по ходу диалога.
"""

import json

from pydantic import ValidationError

from d2.client import OpenRouterClient
from d2.config import (
    DOCTOR_MODEL,
    MAX_TOKENS_DOCTOR,
    MAX_TOKENS_ROUTING,
    PROMPTS_DIR,
    TEMPERATURE_DOCTOR,
    TEMPERATURE_ROUTING,
)
from d2.models import Message
from d2.schemas import SCHEMAS, DoctorTurn


def _load_doctor_template() -> str:
    """Загрузить шаблон system prompt доктора."""
    return (PROMPTS_DIR / "doctor.md").read_text(encoding="utf-8")


def _load_routing_prompt() -> str:
    """Загрузить промпт для пост-диалогового вывода маршрутизации."""
    return (PROMPTS_DIR / "routing_infer.md").read_text(encoding="utf-8")


def build_schema_fields(schema_name: str) -> str:
    """Построить описание полей схемы для вставки в промпт.

    Генерирует список полей из Pydantic модели для {schema_fields} placeholder.
    """
    schema_cls = SCHEMAS[schema_name]
    schema_json = schema_cls.model_json_schema()
    props = schema_json.get("properties", {})
    required = set(schema_json.get("required", []))

    lines = [f"Схема '{schema_name}'. Собери следующие поля:"]
    for field_name, field_info in props.items():
        desc = field_info.get("description", field_name)
        marker = "(обязательное)" if field_name in required else "(если релевантно)"
        lines.append(f"- {field_name}: {desc} {marker}")

    return "\n".join(lines)


def doctor_process(
    client: OpenRouterClient,
    dialog: list[Message],
    accumulated: dict[str, str],
    schema_name: str,
    model: str = DOCTOR_MODEL,
) -> tuple[DoctorTurn, int]:
    """Обработать последнее сообщение пациента, вернуть structured output.

    Возвращает (DoctorTurn, кол-во токенов).
    При ошибке парсинга — 1 retry, потом fallback.
    """
    # Собираем system prompt с конкретной схемой
    template = _load_doctor_template()
    schema_fields = build_schema_fields(schema_name)
    system_prompt = template.format(schema_fields=schema_fields)

    # Добавляем контекст уже собранных данных
    context = ""
    if accumulated:
        filled = ", ".join(f"{k}: {v}" for k, v in accumulated.items() if v)
        context = f"\n\n[Уже собранные данные: {filled}]"

    messages: list[dict] = [{"role": "system", "content": system_prompt + context}]

    # Конвертируем dialog: doctor → assistant (JSON), patient → user
    for msg in dialog:
        if msg.role == "doctor":
            # Доктор "говорит" пациенту текст вопроса, но для LLM был JSON
            messages.append({"role": "assistant", "content": msg.text})
        else:
            messages.append({"role": "user", "content": msg.text})

    # Вызов LLM
    text, usage = client.chat(
        messages=messages,
        model=model,
        temperature=TEMPERATURE_DOCTOR,
        max_tokens=MAX_TOKENS_DOCTOR,
    )
    tokens_used = usage.get("total_tokens", 0)

    # Парсинг structured output
    result = _parse_doctor_turn(text)
    if result is not None:
        return result, tokens_used

    # Retry: просим исправить формат
    messages.append({"role": "assistant", "content": text})
    messages.append({
        "role": "user",
        "content": "Невалидный JSON. Верни ТОЛЬКО JSON: {\"extracted\": {}, \"next_question\": \"...\", \"is_complete\": false, \"reasoning\": \"...\"}",
    })

    text2, usage2 = client.chat(
        messages=messages,
        model=model,
        temperature=0.1,
        max_tokens=MAX_TOKENS_DOCTOR,
    )
    tokens_used += usage2.get("total_tokens", 0)

    result = _parse_doctor_turn(text2)
    if result is not None:
        return result, tokens_used

    # Fallback — не удалось распарсить
    return DoctorTurn(
        extracted={},
        next_question=text[:200],
        is_complete=False,
        reasoning="JSON parse failed, используем raw text как вопрос",
    ), tokens_used


def infer_routing(
    client: OpenRouterClient,
    extracted: dict[str, str],
    model: str = DOCTOR_MODEL,
) -> tuple[dict, int]:
    """Определить маршрутизацию по собранным данным (слепой вывод).

    Qwen получает только extracted данные, без полного бэкграунда кейса.
    Возвращает (routing_dict, tokens_used).
    """
    system_prompt = _load_routing_prompt()
    user_content = json.dumps(extracted, ensure_ascii=False, indent=2)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Данные анкеты пациента:\n\n{user_content}"},
    ]

    text, usage = client.chat(
        messages=messages,
        model=model,
        temperature=TEMPERATURE_ROUTING,
        max_tokens=MAX_TOKENS_ROUTING,
    )
    tokens_used = usage.get("total_tokens", 0)

    routing = _parse_routing_response(text)
    return routing, tokens_used


def _parse_routing_response(text: str) -> dict:
    """Парсим JSON ответ маршрутизации."""
    cleaned = _extract_json(text)
    if cleaned is None:
        return {}

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    # Нормализуем specialists в list
    specialists = data.get("specialists", [])
    if isinstance(specialists, str):
        specialists = [specialists]

    return {
        "specialists": specialists,
        "service_type": str(data.get("service_type", "")),
        "examination": str(data.get("examination", "")),
    }


def _parse_doctor_turn(text: str) -> DoctorTurn | None:
    """Парсим JSON ответ доктора в DoctorTurn."""
    cleaned = _extract_json(text)
    if cleaned is None:
        return None

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    # Приводим все значения extracted к str (LLM может вернуть int/float)
    if "extracted" in data and isinstance(data["extracted"], dict):
        data["extracted"] = {k: str(v) for k, v in data["extracted"].items()}

    try:
        turn = DoctorTurn.model_validate(data)
        turn.next_question = _sanitize_question(turn.next_question)
        return turn
    except ValidationError:
        # Пробуем собрать вручную если структура частичная
        if "next_question" in data:
            return DoctorTurn(
                extracted={k: str(v) for k, v in data.get("extracted", {}).items()},
                next_question=_sanitize_question(str(data["next_question"])),
                is_complete=bool(data.get("is_complete", False)),
                reasoning=str(data.get("reasoning", "")),
            )
        return None


_JSON_MARKERS = ("{", "```", '"extracted"', '"next_question"')


def _sanitize_question(text: str) -> str:
    """Убрать JSON/markdown мусор из next_question.

    Если текст содержит JSON — вернуть fallback вопрос.
    """
    stripped = text.strip()
    if not stripped:
        return "Расскажите подробнее о вашей проблеме."

    # Если похоже на JSON — это утечка structured output
    if stripped.startswith("{") or stripped.startswith("```json") or stripped.startswith('"extracted"'):
        return "Есть ли ещё что-то важное, что вы хотели бы сообщить?"

    # Убираем возможные обрезки JSON в конце
    for marker in _JSON_MARKERS:
        idx = stripped.find(marker)
        if idx > 20:  # есть нормальный текст перед JSON
            stripped = stripped[:idx].rstrip(" ,;:")
            break

    return stripped


def _extract_json(text: str) -> str | None:
    """Извлечь JSON из текста, убирая markdown обёртки."""
    cleaned = text.strip()

    # Убираем ```json ... ```
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    # Если всё ещё не начинается с { — ищем JSON в тексте
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            cleaned = cleaned[start:end]
        else:
            return None

    return cleaned
