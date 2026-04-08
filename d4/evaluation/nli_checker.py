"""LLM-based NLI (Natural Language Inference) для проверки claims.

Заменяет строковый _claim_in_kb() на семантическую проверку:
LLM определяет, поддерживается ли факт текстом KB.

Три verdict: supported / not_supported / neutral.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal

import httpx
from openai import OpenAI

logger = logging.getLogger(__name__)

NLIVerdict = Literal["supported", "not_supported", "neutral"]

# System prompt для NLI проверки — компактный, чтобы снизить latency
_NLI_SYSTEM_PROMPT = """Ты — точный NLI-классификатор для проверки фактов.

Задача: определи, следует ли CLAIM из текста CONTEXT.

Ответ — JSON:
{"verdict": "supported"|"not_supported"|"neutral", "reason": "краткое объяснение"}

Правила:
- supported: CLAIM точно соответствует фактам из CONTEXT
- not_supported: CLAIM противоречит CONTEXT или содержит данные, которых нет в CONTEXT
- neutral: CLAIM не может быть проверен по CONTEXT (нет релевантной информации)
"""


class NLIClaimChecker:
    """Семантическая проверка claims через LLM NLI.

    Attributes:
        client: OpenAI клиент (совместимый с OpenRouter)
        model: модель для NLI проверки (рекомендуется fast-модель)
        max_workers: параллельных потоков для batch проверки
    """

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_sec: float = 30.0,
        max_workers: int = 8,
    ):
        self._client = OpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_sec, connect=10.0),
        )
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_workers = max_workers

    def check_claim(self, claim: str, kb_text: str) -> NLIVerdict:
        """Проверка одного claim через LLM NLI.

        Args:
            claim: проверяемый факт (телефон, ФИО, число и т.д.)
            kb_text: текст KB для сравнения

        Returns:
            NLIVerdict: supported / not_supported / neutral
        """
        if not claim.strip():
            return "supported"

        user_msg = f"CONTEXT:\n{kb_text[:3000]}\n\nCLAIM:\n{claim}"
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _NLI_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            data = json.loads(raw)
            verdict = data.get("verdict", "neutral")
            if verdict in ("supported", "not_supported", "neutral"):
                return verdict
            return "neutral"
        except Exception as exc:
            logger.warning("NLI ошибка для claim '%s': %s", claim[:50], exc)
            return "neutral"

    def check_claims_batch(
        self,
        claims: list[str],
        kb_text: str,
    ) -> list[NLIVerdict]:
        """Параллельная проверка нескольких claims.

        Args:
            claims: список проверяемых фактов
            kb_text: текст KB

        Returns:
            список NLIVerdict (по одному на каждый claim)
        """
        if not claims:
            return []

        verdicts: list[NLIVerdict] = ["neutral"] * len(claims)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(self.check_claim, claim, kb_text): idx
                for idx, claim in enumerate(claims)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    verdicts[idx] = future.result()
                except Exception as exc:
                    logger.warning("NLI batch ошибка [%d]: %s", idx, exc)

        return verdicts

    def count_unsupported(self, claims: list[str], kb_text: str) -> int:
        """Подсчёт количества неподтверждённых claims.

        Метод для прямой интеграции с deterministic.py.

        Args:
            claims: список проверяемых фактов
            kb_text: полный текст KB

        Returns:
            количество claims с verdict == "not_supported"
        """
        if not claims:
            return 0
        verdicts = self.check_claims_batch(claims, kb_text)
        return sum(1 for v in verdicts if v == "not_supported")
