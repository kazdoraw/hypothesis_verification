"""OpenRouter HTTP клиент с retry и token tracking.

Паттерн из dialog_generator/poc.py — httpx, прямой контроль.
"""

import re
import time

import httpx

from d2.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL


class OpenRouterClient:
    """Минимальный клиент OpenRouter API с retry и подсчётом токенов."""

    def __init__(self, api_key: str = OPENROUTER_API_KEY, timeout_s: int = 180):
        self.api_key = api_key
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

        Возвращает (text, usage) где usage = {"prompt_tokens": N, "completion_tokens": N}.
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
                    OPENROUTER_BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                data = response.json()

                # Проверяем наличие ответа
                if "choices" not in data or not data["choices"]:
                    error_msg = data.get("error", {}).get("message", str(data))
                    if attempt < max_retries:
                        wait = 2 ** attempt
                        print(f"  [RETRY {attempt + 1}] Пустой ответ: {error_msg[:80]}. Жду {wait}s...")
                        time.sleep(wait)
                        continue
                    raise RuntimeError(f"API error после {max_retries + 1} попыток: {error_msg[:200]}")

                # Подсчёт токенов
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
                    print(f"  [RETRY {attempt + 1}] HTTP {e.response.status_code}. Жду {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise
            except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as e:
                if attempt < max_retries:
                    wait = 5 * (attempt + 1)
                    print(f"  [RETRY {attempt + 1}] {type(e).__name__}. Жду {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError("Не удалось получить ответ от API")


# --- Утилиты ---

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Убираем <think>...</think> теги из ответа Qwen3."""
    return _THINK_RE.sub("", text).strip()
