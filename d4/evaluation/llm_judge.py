"""Слой 3: LLM-as-Judge (вспомогательный, НЕ единственный источник).

Judge: openai/gpt-5.4-mini via OpenRouter, temperature=0.0.
Оценки: factual_accuracy (1-5), completeness (1-5), hallucination (0/1).
Корреляция judge vs expert проверяется на подвыборке из Слоя 2.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

from d4.models import EvalSample, JudgeScore, StrategyID, StrategyResult

# Путь к judge prompt
JUDGE_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "judge.md"


def _load_judge_prompt() -> str:
    """Загрузка judge prompt из файла."""
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


def _build_judge_message(
    query: str,
    answer: str,
    expected_answer: str,
    kb_excerpt: str,
) -> str:
    """Формирование сообщения для judge.

    Порядок секций отражает приоритет: KB → ответ → эталон.
    KB идёт первым как первичный источник истины.
    """
    return (
        f"## Запрос пациента\n{query}\n\n"
        f"## База знаний (первичный источник)\n{kb_excerpt}\n\n"
        f"## Ответ системы\n{answer}\n\n"
        f"## Эталонный ответ (вторичная подсказка)\n{expected_answer}"
    )


def _parse_judge_response(raw: str) -> dict[str, Any]:
    """Парсинг JSON-ответа judge."""
    content = raw.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        return {
            "factual_accuracy": 1,
            "completeness": 1,
            "hallucination": True,
            "reasoning": f"JSON parse error: {raw[:200]}",
        }


class LLMJudge:
    """LLM-as-Judge для оценки качества ответов."""

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-5.4-mini",
        api_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout_sec: float = 90.0,
        max_workers: int = 8,
    ) -> None:
        self.client = OpenAI(
            base_url=api_url,
            api_key=api_key,
            timeout=httpx.Timeout(timeout_sec, connect=10.0),
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self._system_prompt = _load_judge_prompt()

    def judge_single(
        self,
        result: StrategyResult,
        sample: EvalSample,
        kb_excerpt: str = "",
    ) -> JudgeScore:
        """Оценка одного ответа.

        Args:
            result: результат стратегии
            sample: эталонный запрос
            kb_excerpt: релевантный фрагмент KB

        Returns:
            JudgeScore
        """
        user_message = _build_judge_message(
            query=sample.query,
            answer=result.answer.answer,
            expected_answer=sample.expected_answer,
            kb_excerpt=kb_excerpt or result.retrieval.context_text,
        )

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

            raw = response.choices[0].message.content or ""
            data = _parse_judge_response(raw)

            return JudgeScore(
                sample_id=result.sample_id,
                strategy_id=result.strategy_id,
                factual_accuracy=max(1, min(5, data.get("factual_accuracy", 1))),
                completeness=max(1, min(5, data.get("completeness", 1))),
                hallucination=bool(data.get("hallucination", True)),
                reasoning=data.get("reasoning", ""),
            )

        except Exception as e:
            return JudgeScore(
                sample_id=result.sample_id,
                strategy_id=result.strategy_id,
                factual_accuracy=1,
                completeness=1,
                hallucination=True,
                reasoning=f"Judge error: {e}",
            )

    def judge_batch(
        self,
        results: list[StrategyResult],
        samples: list[EvalSample],
        kb_excerpt: str = "",
    ) -> list[JudgeScore]:
        """Параллельная batch оценка.

        Args:
            results: все StrategyResult
            samples: eval set
            kb_excerpt: полный текст KB (для контекста judge)

        Returns:
            список JudgeScore
        """
        sample_map = {s.sample_id: s for s in samples}

        # Фильтруем результаты с известными samples
        tasks = [
            (result, sample_map[result.sample_id])
            for result in results
            if result.sample_id in sample_map
        ]

        total = len(tasks)
        if not tasks:
            return []

        scores: list[JudgeScore] = []
        done = 0
        start_time = time.perf_counter()

        print(f"  judge: {total} оценок, {self.max_workers} потоков")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.judge_single, result, sample, kb_excerpt): result
                for result, sample in tasks
            }

            for future in as_completed(futures):
                try:
                    score = future.result()
                    scores.append(score)
                except Exception as exc:
                    result = futures[future]
                    print(f"  judge ошибка {result.sample_id}:{result.strategy_id}: {exc}")

                done += 1
                if done % 25 == 0:
                    elapsed = time.perf_counter() - start_time
                    print(f"  judge: {done}/{total} ({elapsed:.1f}s)")

        elapsed = time.perf_counter() - start_time
        print(f"  judge готово: {len(scores)}/{total} за {elapsed:.1f}s")
        return scores


def save_judge_scores(scores: list[JudgeScore], output_path: str | Path) -> None:
    """Сохранение judge scores в JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [s.model_dump() for s in scores]
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_judge_scores(path: str | Path) -> list[JudgeScore]:
    """Загрузка judge scores из JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [JudgeScore(**item) for item in data]
