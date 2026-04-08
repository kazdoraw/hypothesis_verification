"""Patient agent — Grok имитирует пациента по сценарию.

Получает system prompt с описанием ситуации,
отвечает свободно без скриптов.
"""

from d2.config import PATIENT_MODEL, MAX_TOKENS_PATIENT, TEMPERATURE_PATIENT, PROMPTS_DIR
from d2.client import OpenRouterClient
from d2.models import Case, Message


def _load_patient_prompt(case: Case) -> str:
    """Загрузить шаблон промпта и подставить ситуацию пациента."""
    template = (PROMPTS_DIR / "patient.md").read_text(encoding="utf-8")
    return template.format(situation=case.situation)


def patient_speak(
    client: OpenRouterClient,
    case: Case,
    dialog: list[Message],
    model: str = PATIENT_MODEL,
) -> tuple[str, int]:
    """Генерирует ответ пациента.

    Возвращает (текст ответа, кол-во использованных токенов).
    """
    system_prompt = _load_patient_prompt(case)

    # Собираем историю для LLM
    messages: list[dict] = [{"role": "system", "content": system_prompt}]

    if not dialog:
        # Первое сообщение — пациент начинает разговор
        messages.append({
            "role": "user",
            "content": "Напиши первое сообщение в чат клиники. ТОЛЬКО основная жалоба, 1 короткое предложение. Не упоминай аллергии, лекарства, хронические болезни, беременность.",
        })
    else:
        # Конвертируем dialog в формат OpenAI
        for msg in dialog:
            role = "assistant" if msg.role == "patient" else "user"
            messages.append({"role": role, "content": msg.text})

    text, usage = client.chat(
        messages=messages,
        model=model,
        temperature=TEMPERATURE_PATIENT,
        max_tokens=MAX_TOKENS_PATIENT,
    )

    tokens_used = usage.get("total_tokens", 0)
    return text.strip(), tokens_used
