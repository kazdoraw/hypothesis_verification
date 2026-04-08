"""
Proof of Concept: Multi-Agent Dialog Generator (v2).

Генерирует реалистичные диалоги пациент↔администратор
через OpenRouter API с двумя LLM-агентами.
Сценарии и персоны генерируются LLM динамически.

Использование:
    python -m dialog_generator.poc --n 2 --output data/d3_poc.json
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

# =============================================================================
# Constants
# =============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Модели для агентов (через OpenRouter)
DEFAULT_PATIENT_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_ADMIN_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_LABELER_MODEL = "x-ai/grok-4.1-fast"

MAX_TURNS = 20
MIN_TURNS = 4
WRAP_UP_THRESHOLD = 12

PROMPTS_DIR = Path(__file__).parent / "prompts"

# Параметры для генерации сценариев (только enum-значения)
L1_TYPES = ["anamnesis", "booking", "faq", "feedback"]
SPECIALTIES = [
    "терапия", "хирургия", "ортодонтия", "ортопедия",
    "имплантология", "пародонтология", "детская стоматология",
]
TONES = ["worried", "impatient", "skeptical", "friendly", "confused", "angry"]
COMPLEXITIES = ["simple", "medium", "hard"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Message:
    turn: int
    role: str
    text: str
    label: dict = field(default_factory=dict)


@dataclass
class Dialog:
    dialog_id: str
    scenario: dict
    patient_persona: dict
    messages: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# =============================================================================
# Scenario & Persona Generator (LLM-driven)
# =============================================================================

SCENARIO_GEN_PROMPT = """Сгенерируй реалистичный сценарий для диалога в чате стоматологической клиники.

Параметры:
- Тип обращения (L1): {l1_type}
- Специализация: {specialty}
- Сложность: {complexity}
- Тон пациента: {tone}

Верни ТОЛЬКО валидный JSON (без markdown) со следующей структурой:
{{
  "patient": {{
    "name": "русское имя",
    "age": число от 18 до 75,
    "gender": "male" или "female",
    "tone": "{tone}",
    "dental_iq": "low" или "medium" или "high",
    "behavior": "краткое описание поведения в 5-10 слов",
    "situation": "описание ситуации пациента в 2-3 предложения. Конкретные симптомы, жалобы, обстоятельства. Без медицинских терминов — как пациент бы описал."
  }},
  "scenario": {{
    "type": "{l1_type}",
    "specialty": "{specialty}",
    "complexity": "{complexity}",
    "summary": "краткое описание сценария в 1 предложение"
  }}
}}

Требования к ситуации пациента:
- Реалистичная бытовая жалоба для стоматологии
- Конкретные детали (какой зуб, когда началось, что беспокоит)
- Соответствует специализации: {specialty}
- Учитывай тон: {tone} (это влияет на то, как пациент воспринимает ситуацию)
- Ситуация должна быть УНИКАЛЬНОЙ, не шаблонной
"""


def generate_scenario(
    client: "OpenRouterClient",
    model: str,
    l1_type: str,
    specialty: str,
    complexity: str,
    tone: str,
) -> tuple[dict, dict]:
    """Генерирует сценарий и персону через LLM. Возвращает (persona_dict, scenario_dict)."""
    prompt = SCENARIO_GEN_PROMPT.format(
        l1_type=l1_type,
        specialty=specialty,
        complexity=complexity,
        tone=tone,
    )

    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.9,
        max_tokens=600,
    )

    data = _parse_json_response(response)
    if not data:
        return _fallback_persona(tone), _fallback_scenario(l1_type, specialty, complexity)

    persona = data.get("patient", {})
    scenario = data.get("scenario", {})
    return persona, scenario


def _parse_json_response(text: str) -> Optional[dict]:
    """Извлекает JSON из ответа LLM."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Пробуем найти JSON в тексте
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
    return None


def _fallback_persona(tone: str) -> dict:
    return {
        "name": "Пациент", "age": 35, "gender": "female",
        "tone": tone, "dental_iq": "low",
        "behavior": "обычный пациент",
        "situation": "Беспокоит зуб, хочет записаться на приём.",
    }


def _fallback_scenario(l1_type: str, specialty: str, complexity: str) -> dict:
    return {
        "type": l1_type, "specialty": specialty,
        "complexity": complexity, "summary": "Стандартное обращение пациента.",
    }


# =============================================================================
# LLM Client
# =============================================================================

class OpenRouterClient:
    """Минимальный клиент OpenRouter API."""

    def __init__(self, api_key: str, timeout_s: int = 60):
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.total_tokens = 0
        self.total_calls = 0

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> str:
        """Один вызов Chat Completion с retry."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(max_retries + 1):
            try:
                response = httpx.post(
                    OPENROUTER_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                data = response.json()

                if "choices" not in data or not data["choices"]:
                    error_msg = data.get("error", {}).get("message", str(data))
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(f"  [RETRY {attempt+1}] API вернул пустой ответ: {error_msg[:80]}. Жду {wait}s...")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"API error после {max_retries+1} попыток: {error_msg[:200]}")

                self.total_calls += 1
                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)
                return data["choices"][0]["message"]["content"]

            except httpx.HTTPStatusError as e:
                if attempt < max_retries and e.response.status_code in (429, 502, 503):
                    wait = 2 ** attempt
                    print(f"  [RETRY {attempt+1}] HTTP {e.response.status_code}. Жду {wait}s...")
                    time.sleep(wait)
                    continue
                raise


# =============================================================================
# Prompt Loader
# =============================================================================

def load_prompt(name: str) -> str:
    """Загрузить prompt-шаблон из файла."""
    path = PROMPTS_DIR / f"{name}.md"
    return path.read_text(encoding="utf-8")


def build_patient_prompt(persona: dict) -> str:
    """Собрать system prompt для пациента."""
    template = load_prompt("patient")
    return template.format(
        name=persona.get("name", "Пациент"),
        age=persona.get("age", 35),
        gender=persona.get("gender", "female"),
        tone=persona.get("tone", "neutral"),
        dental_iq=persona.get("dental_iq", "low"),
        behavior=persona.get("behavior", ""),
        situation=persona.get("situation", ""),
    )


def build_admin_prompt(scenario: dict) -> str:
    """Собрать system prompt для администратора."""
    template = load_prompt("admin")
    specialty = scenario.get("specialty", "")
    summary = scenario.get("summary", "")
    context = f"Специализация обращения: {specialty}. {summary}"
    return template.format(scenario_context=context)


# =============================================================================
# Orchestrator
# =============================================================================

def run_dialog(
    client: OpenRouterClient,
    persona: dict,
    scenario: dict,
    patient_model: str = DEFAULT_PATIENT_MODEL,
    admin_model: str = DEFAULT_ADMIN_MODEL,
    max_turns: int = MAX_TURNS,
) -> list[Message]:
    """Провести диалог между агентами turn-by-turn."""

    patient_system = build_patient_prompt(persona)
    admin_system = build_admin_prompt(scenario)

    patient_opener_prompt = (
        "Начни разговор с клиникой. Напиши первое сообщение — "
        "поприветствуй и опиши свою ситуацию коротко, как в мессенджере. "
        "1-2 предложения максимум."
    )

    patient_history: list[dict] = [
        {"role": "system", "content": patient_system},
        {"role": "user", "content": patient_opener_prompt},
    ]

    admin_history: list[dict] = [
        {"role": "system", "content": admin_system},
    ]

    messages: list[Message] = []
    turn = 0
    last_patient_topics: list[str] = []

    while turn < max_turns:
        # --- Patient turn ---
        turn += 1

        # Wrap-up injection: если приближаемся к лимиту — попросить завершить
        if turn >= WRAP_UP_THRESHOLD:
            patient_history.append({
                "role": "user",
                "content": "(Подведи итог и попрощайся — тебе пора.)",
            })

        patient_text = client.chat(
            messages=patient_history,
            model=patient_model,
            temperature=0.9,
            max_tokens=150,
        )
        patient_text = patient_text.strip()

        # Убираем wrap-up injection из истории (он был одноразовый)
        if turn >= WRAP_UP_THRESHOLD and patient_history[-1]["role"] == "user":
            patient_history.pop()

        messages.append(Message(turn=turn, role="patient", text=patient_text))
        patient_history.append({"role": "assistant", "content": patient_text})
        admin_history.append({"role": "user", "content": patient_text})

        # Проверяем завершение
        if turn >= MIN_TURNS and _is_farewell(patient_text):
            break

        # Анти-луп: проверяем повторяемость тем пациента
        if _is_looping(patient_text, last_patient_topics):
            # Сначала админ закрывает диалог
            admin_history.append({
                "role": "user",
                "content": "(Пациент повторяется. Подведи итог, подтверди запись и попрощайся.)",
            })
            admin_close = client.chat(
                messages=admin_history,
                model=admin_model,
                temperature=0.4,
                max_tokens=120,
            ).strip()
            admin_history.pop()
            messages.append(Message(turn=turn + 1, role="admin", text=admin_close))

            # Затем пациент прощается
            patient_history.append({"role": "user", "content": admin_close})
            patient_history.append({
                "role": "user",
                "content": "(Поблагодари и попрощайся коротко.)",
            })
            closing = client.chat(
                messages=patient_history,
                model=patient_model,
                temperature=0.7,
                max_tokens=60,
            ).strip()
            patient_history.pop()
            messages.append(Message(turn=turn + 2, role="patient", text=closing))
            break

        last_patient_topics.append(patient_text[:60].lower())
        if len(last_patient_topics) > 3:
            last_patient_topics.pop(0)

        # --- Admin turn ---
        turn += 1

        admin_text = client.chat(
            messages=admin_history,
            model=admin_model,
            temperature=0.5,
            max_tokens=250,
        )
        admin_text = admin_text.strip()
        messages.append(Message(turn=turn, role="admin", text=admin_text))
        admin_history.append({"role": "assistant", "content": admin_text})
        patient_history.append({"role": "user", "content": admin_text})

        if turn >= MIN_TURNS and _is_farewell(admin_text):
            break

    return messages


FAREWELL_MARKERS = [
    "до свидания", "всего доброго", "всего хорошего",
    "до встречи", "хорошего дня", "удачи", "будем ждать",
    "пока!", "до скорого", "ждём вас", "спасибо, пока",
    "спасибо за информацию", "спасибо большое",
]


def _is_farewell(text: str) -> bool:
    """Проверяем, содержит ли сообщение прощание."""
    text_lower = text.lower()
    return any(marker in text_lower for marker in FAREWELL_MARKERS)


STOP_WORDS = {
    "я", "и", "в", "на", "не", "что", "это", "а", "но", "с", "к", "у",
    "по", "от", "за", "из", "как", "то", "ещё", "еще", "мне", "мой",
    "да", "нет", "вы", "вас", "вам", "мы", "он", "она", "его", "её",
    "бы", "же", "ли", "уже", "так", "для", "при", "до", "или", "ок",
}


def _is_looping(current: str, history: list[str], threshold: float = 0.35) -> bool:
    """Детектируем зацикливание: если >35% значимых слов повторяются."""
    if len(history) < 2:
        return False
    current_words = set(current.lower().split()) - STOP_WORDS
    if len(current_words) < 3:
        return False
    prev_words = set()
    for h in history[-2:]:
        prev_words.update(set(h.split()) - STOP_WORDS)
    overlap = len(current_words & prev_words) / len(current_words)
    return overlap > threshold


# =============================================================================
# Labeler
# =============================================================================

def label_dialog(
    client: OpenRouterClient,
    messages: list[Message],
    labeler_model: str = DEFAULT_LABELER_MODEL,
) -> list[dict]:
    """Разметить диалог через LLM."""

    labeler_system = load_prompt("labeler")

    dialog_text = "\n".join(
        f"[Turn {m.turn}] [{m.role}]: {m.text}" for m in messages
    )

    labeler_prompt = (
        f"Размети каждое сообщение в следующем диалоге.\n\n"
        f"Диалог:\n{dialog_text}\n\n"
        f"Верни ТОЛЬКО валидный JSON массив. Без markdown, без ```json."
    )

    response = client.chat(
        messages=[
            {"role": "system", "content": labeler_system},
            {"role": "user", "content": labeler_prompt},
        ],
        model=labeler_model,
        temperature=0.1,
        max_tokens=2000,
    )

    return _parse_labels(response, len(messages))


def _parse_labels(response: str, expected_count: int) -> list[dict]:
    """Парсим JSON из ответа LLM."""
    # Убираем возможные markdown-обёртки
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        labels = json.loads(text)
        if isinstance(labels, list):
            return labels
    except json.JSONDecodeError:
        pass

    # Fallback: пустые лейблы
    print(f"  [WARN] Не удалось распарсить labels, возвращаю пустые")
    return [{"turn": i + 1, "parse_error": True} for i in range(expected_count)]


# =============================================================================
# Pipeline
# =============================================================================

def generate_dialogs(
    client: OpenRouterClient,
    n_dialogs: int = 2,
    patient_model: str = DEFAULT_PATIENT_MODEL,
    admin_model: str = DEFAULT_ADMIN_MODEL,
    labeler_model: str = DEFAULT_LABELER_MODEL,
) -> dict:
    """Генерация N диалогов с LLM-генерируемыми сценариями."""

    dialogs = []

    for i in range(n_dialogs):
        # Случайные параметры для генерации сценария
        l1_type = random.choice(L1_TYPES)
        specialty = random.choice(SPECIALTIES)
        complexity = random.choice(COMPLEXITIES)
        tone = random.choice(TONES)

        print(f"\n{'='*60}")
        print(f"Dialog {i+1}/{n_dialogs}")
        print(f"  Параметры: {l1_type} / {specialty} / {tone} / {complexity}")

        # LLM генерирует сценарий и персону
        print(f"  Генерирую сценарий...")
        persona, scenario = generate_scenario(
            client, patient_model, l1_type, specialty, complexity, tone,
        )
        print(f"  Персона: {persona.get('name', '?')} ({persona.get('tone', '?')})")
        print(f"  Ситуация: {persona.get('situation', '?')[:80]}...")
        print(f"{'='*60}")

        # Генерация диалога
        t0 = time.time()
        messages = run_dialog(
            client=client,
            persona=persona,
            scenario=scenario,
            patient_model=patient_model,
            admin_model=admin_model,
        )
        gen_time = time.time() - t0

        print(f"\n  Сгенерировано {len(messages)} сообщений за {gen_time:.1f}s")
        for m in messages:
            role_icon = "P" if m.role == "patient" else "A"
            print(f"  [{role_icon}:{m.turn}] {m.text[:80]}{'...' if len(m.text) > 80 else ''}")

        # Разметка
        print(f"\n  Размечаю диалог...")
        labels = label_dialog(client, messages, labeler_model)

        for msg, label in zip(messages, labels):
            msg.label = label

        dialog = Dialog(
            dialog_id=f"d3_{i+1:04d}",
            scenario=scenario,
            patient_persona=persona,
            messages=[
                {
                    "turn": m.turn,
                    "role": m.role,
                    "text": m.text,
                    "label": m.label,
                }
                for m in messages
            ],
            summary={
                "total_turns": len(messages),
                "generation_time_s": round(gen_time, 1),
            },
        )
        dialogs.append(dialog)

    return {
        "version": "0.2-poc",
        "generated_at": datetime.now().isoformat(),
        "total_dialogs": len(dialogs),
        "models": {
            "patient": patient_model,
            "admin": admin_model,
            "labeler": labeler_model,
        },
        "stats": {
            "total_api_calls": client.total_calls,
            "total_tokens": client.total_tokens,
        },
        "dialogs": [asdict(d) for d in dialogs],
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PoC: Multi-Agent Dialog Generator")
    parser.add_argument("--n", type=int, default=2, help="Количество диалогов")
    parser.add_argument("--output", type=str, default="data/d3_poc.json", help="Путь к выходному файлу")
    parser.add_argument("--patient-model", type=str, default=DEFAULT_PATIENT_MODEL)
    parser.add_argument("--admin-model", type=str, default=DEFAULT_ADMIN_MODEL)
    parser.add_argument("--labeler-model", type=str, default=DEFAULT_LABELER_MODEL)
    args = parser.parse_args()

    # Загружаем .env из корня study/
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: Установите OPENROUTER_API_KEY")
        print(f"  Создайте файл {env_path} с содержимым:")
        print("  OPENROUTER_API_KEY=sk-or-...")
        sys.exit(1)

    client = OpenRouterClient(api_key=api_key)

    print(f"Генерация {args.n} диалогов...")
    print(f"  Patient model: {args.patient_model}")
    print(f"  Admin model: {args.admin_model}")

    result = generate_dialogs(
        client=client,
        n_dialogs=args.n,
        patient_model=args.patient_model,
        admin_model=args.admin_model,
        labeler_model=args.labeler_model,
    )

    # Сохраняем
    output_path = Path(__file__).parent.parent / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print(f"Готово!")
    print(f"  Диалогов: {result['total_dialogs']}")
    print(f"  API calls: {result['stats']['total_api_calls']}")
    print(f"  Tokens: {result['stats']['total_tokens']}")
    print(f"  Файл: {output_path}")


if __name__ == "__main__":
    main()
