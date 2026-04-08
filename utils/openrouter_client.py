"""Общий OpenRouter HTTP клиент для экспериментов D1/D2/D4.

Адаптирован из d2/client.py с декаплингом:
- API key из env/параметра (не из d2.config)
- httpx + retry (429/502/503) + exponential backoff
- Подсчёт токенов (in/out/calls)
- Удаление <think> тегов (для Qwen3)
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

# Загружаем .env из корня study/
load_dotenv(Path(__file__).parent.parent / ".env")

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Удаление <think>...</think> тегов из ответа (Qwen3 reasoning)."""
    return _THINK_RE.sub("", text).strip()


class OpenRouterClient:
    """Минимальный клиент OpenRouter API с retry и подсчётом токенов.

    Использование:
        client = OpenRouterClient()
        text, usage = client.chat(
            messages=[{"role": "user", "content": "Привет"}],
            model="x-ai/grok-4.1-fast",
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = _OPENROUTER_BASE_URL,
        timeout_s: int = 180,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY не установлен. "
                "Укажите через параметр api_key или env var."
            )
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_calls = 0

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        max_retries: int = 2,
    ) -> tuple[str, dict]:
        """Один вызов Chat Completion.

        Returns:
            (text, usage) — text ответа и dict с prompt_tokens/completion_tokens
        """
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
                    self.base_url,
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
                        print(f"  [RETRY {attempt + 1}] Пустой ответ: {error_msg[:80]}. Жду {wait}s...")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(
                        f"API error после {max_retries + 1} попыток: {error_msg[:200]}"
                    )

                usage = data.get("usage", {})
                self.total_tokens_in += usage.get("prompt_tokens", 0)
                self.total_tokens_out += usage.get("completion_tokens", 0)
                self.total_calls += 1

                text = data["choices"][0]["message"].get("content") or ""
                text = _strip_think_tags(text)

                if not text:
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(f"  [RETRY {attempt + 1}] Пустой content. Жду {wait}s...", flush=True)
                        time.sleep(wait)
                        continue
                    raise RuntimeError("API вернул пустой content")

                return text, usage

            except httpx.HTTPStatusError as e:
                if attempt < max_retries and e.response.status_code in (429, 502, 503):
                    wait = 2 ** attempt
                    print(
                        f"  [RETRY {attempt + 1}] HTTP {e.response.status_code}. Жду {wait}s...",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
                if attempt < max_retries:
                    wait = 5 * (attempt + 1)
                    print(
                        f"  [RETRY {attempt + 1}] {type(e).__name__}. Жду {wait}s...",
                        flush=True,
                    )
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("Не удалось получить ответ от API")

    def get_stats(self) -> dict:
        """Статистика использования API."""
        return {
            "total_calls": self.total_calls,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "total_tokens": self.total_tokens_in + self.total_tokens_out,
        }

    def reset_stats(self) -> None:
        """Сброс счётчиков."""
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self.total_calls = 0
