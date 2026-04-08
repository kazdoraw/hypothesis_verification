"""Единый LLM runner для стратегий S1-S4.

Фиксирует: модель, system prompt, temperature, output schema.
Единственная переменная — контекст, сформированный стратегией.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from d4.models import FAQAnswer

# Путь к system prompt
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "faq_system.md"


def _load_system_prompt() -> str:
    """Загрузка system prompt из файла."""
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_user_message(query: str, context: str) -> str:
    """Формирование user message: контекст + запрос.

    При пустом контексте явно указываем это — иначе LLM
    может галлюцинировать из training data.
    """
    if not context or not context.strip():
        ctx_block = (
            "## Контекст из базы знаний клиники\n\n"
            "КОНТЕКСТ ПУСТ. Информация из базы знаний не найдена.\n"
            "Ты ОБЯЗАН вернуть answerable: false."
        )
    else:
        ctx_block = f"## Контекст из базы знаний клиники\n\n{context}"
    return f"{ctx_block}\n\n---\n\n## Запрос пациента\n\n{query}"


def _parse_faq_answer(raw_content: str) -> FAQAnswer:
    """Парсинг JSON-ответа LLM → FAQAnswer.

    Один retry при ошибке формата — explicit error.
    """
    # Убираем markdown code fences если есть
    content = raw_content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Убираем первую и последнюю строку (```json и ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
        return FAQAnswer(**data)
    except (json.JSONDecodeError, ValueError) as e:
        return FAQAnswer(
            answer=raw_content,
            answerable=False,
            confidence=0.0,
            source_ids=[],
        )


class LLMRunner:
    """Единый LLM-вызов для всех стратегий S1-S4.

    Фиксированные параметры загружаются из experiment.yaml.
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_sec: float = 90.0,
    ) -> None:
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_sec, connect=10.0),
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._system_prompt = _load_system_prompt()

    def run(
        self,
        query: str,
        context: str,
    ) -> dict[str, Any]:
        """Вызов LLM и возврат структурированного результата.

        Args:
            query: запрос пациента
            context: текстовый контекст, сформированный стратегией

        Returns:
            dict с ключами: answer (FAQAnswer), latency_ms, tokens_prompt,
            tokens_completion, error
        """
        user_message = _build_user_message(query, context)

        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
            )
            latency_ms = (time.perf_counter() - start) * 1000

            raw_content = response.choices[0].message.content or ""
            answer = _parse_faq_answer(raw_content)

            usage = response.usage
            tokens_prompt = usage.prompt_tokens if usage else 0
            tokens_completion = usage.completion_tokens if usage else 0

            return {
                "answer": answer,
                "latency_ms": latency_ms,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "error": None,
            }

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return {
                "answer": FAQAnswer(answer="", answerable=False, confidence=0.0),
                "latency_ms": latency_ms,
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "error": str(e),
            }

