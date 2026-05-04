"""Orchestrator: запуск всех стратегий на eval set.

Порядок обработки одного запроса (по плану §2.2):
1. Загрузка sample
2. Выбор стратегии
3. Формирование контекста
4. Вызов LLM или template responder
5. Сохранение сырого ответа
6. Автоматическая нормализация ответа
7. Вычисление inline-метрик (latency, tokens, context_length)
8. Запись результата

Checkpoint каждые N запросов.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from d4.models import (
    EvalSample,
    FAQAnswer,
    KBChunk,
    StrategyID,
    StrategyResult,
)
from d4.pipeline.llm_runner import LLMRunner
from d4.strategies.base import BaseContextStrategy

# ---------------------------------------------------------------------------
# Нормализация ответа
# ---------------------------------------------------------------------------


def _normalize_answer(answer: FAQAnswer) -> FAQAnswer:
    """Нормализация ответа: trim, стандартизация пустых полей."""
    return FAQAnswer(
        answer=answer.answer.strip(),
        answerable=answer.answerable,
        doctor=answer.doctor.strip() if answer.doctor else None,
        specialization=answer.specialization.strip() if answer.specialization else None,
        branch=answer.branch.strip() if answer.branch else None,
        service=answer.service.strip() if answer.service else None,
        suggest_booking=answer.suggest_booking,
        confidence=answer.confidence,
        source_ids=answer.source_ids,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Запуск стратегий на eval set с параллельным выполнением и checkpoint."""

    def __init__(
        self,
        strategies: list[BaseContextStrategy],
        llm_runner: LLMRunner,
        chunks: list[KBChunk],
        checkpoint_every: int = 25,
        max_workers: int = 8,
        output_path: str | Path = "outputs/raw_results.jsonl",
    ) -> None:
        self.strategies = {s.strategy_id: s for s in strategies}
        self.llm_runner = llm_runner
        self.chunks = chunks

        # Инжекция LLMRunner в стратегии, которым он нужен (S5 TieredStrategy)
        for strategy in strategies:
            if hasattr(strategy, "set_llm_runner"):
                strategy.set_llm_runner(llm_runner)
        self.checkpoint_every = checkpoint_every
        self.max_workers = max_workers
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def run_single(
        self,
        sample: EvalSample,
        strategy: BaseContextStrategy,
    ) -> StrategyResult:
        """Обработка одного запроса одной стратегией.

        Для стратегий с answer_directly (S5, B0): НЕ вызываем select_context()
        отдельно — стратегия сама решает, какой retrieval использовать.
        DirectAnswerResult содержит полный accounting (tokens, retrieval, route).
        """
        strategy_id = StrategyID(strategy.strategy_id)

        route_taken = ""

        conf_debug = None

        if hasattr(strategy, "answer_directly"):
            result = strategy.answer_directly(sample.query, self.chunks)
            answer = result.answer
            retrieval = result.retrieval
            latency_ms = result.latency_ms
            tokens_prompt = result.tokens_prompt
            tokens_completion = result.tokens_completion
            route_taken = result.route_taken
            error = result.error
            conf_debug = result.confidence_debug or None
        else:
            retrieval = strategy.select_context(sample.query, self.chunks)
            llm_result = self.llm_runner.run(sample.query, retrieval.context_text)
            answer = llm_result["answer"]
            latency_ms = llm_result["latency_ms"]
            tokens_prompt = llm_result["tokens_prompt"]
            tokens_completion = llm_result["tokens_completion"]
            error = llm_result["error"]

        answer = _normalize_answer(answer)

        return StrategyResult(
            sample_id=sample.sample_id,
            strategy_id=strategy_id,
            retrieval=retrieval,
            answer=answer,
            latency_ms=latency_ms,
            tokens_prompt=tokens_prompt,
            tokens_completion=tokens_completion,
            context_length=retrieval.context_token_count,
            route_taken=route_taken,
            error=error,
            confidence_debug=conf_debug,
        )

    def run_all(
        self,
        samples: list[EvalSample],
        strategy_ids: Optional[list[str]] = None,
        resume_from: Optional[set[str]] = None,
        existing_results: Optional[list[StrategyResult]] = None,
    ) -> list[StrategyResult]:
        """Параллельный запуск всех стратегий на всех запросах.

        Args:
            samples: eval set
            strategy_ids: какие стратегии запускать (None = все)
            resume_from: set(sample_id:strategy_id) уже обработанных — пропустить
            existing_results: ранее загруженные результаты для сохранения в checkpoint

        Returns:
            список НОВЫХ StrategyResult (без existing)
        """
        strategies_to_run = (
            [self.strategies[sid] for sid in strategy_ids if sid in self.strategies]
            if strategy_ids
            else list(self.strategies.values())
        )

        resume_from = resume_from or set()
        base_results: list[StrategyResult] = list(existing_results or [])

        # Собираем задачи, пропуская уже обработанные
        tasks: list[tuple[EvalSample, BaseContextStrategy]] = []
        skipped = 0
        for sample in samples:
            for strategy in strategies_to_run:
                key = f"{sample.sample_id}:{strategy.strategy_id}"
                if key in resume_from:
                    skipped += 1
                    continue
                tasks.append((sample, strategy))

        total = len(tasks)
        if skipped:
            print(f"  пропущено (resume): {skipped}, осталось: {total}")

        if not tasks:
            print("Все задачи уже выполнены.")
            return []

        results: list[StrategyResult] = []
        done = 0
        start_time = time.perf_counter()

        print(f"  запуск: {total} задач, {self.max_workers} потоков", flush=True)

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self.run_single, sample, strategy): (sample, strategy)
                    for sample, strategy in tasks
                }

                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as exc:
                        sample, strategy = futures[future]
                        print(
                            f"  ✗ {sample.sample_id}:{strategy.strategy_id} — {exc}",
                            flush=True,
                        )
                        continue

                    with self._lock:
                        results.append(result)
                        done += 1

                        elapsed = time.perf_counter() - start_time
                        rate = done / elapsed if elapsed > 0 else 0
                        eta_sec = (total - done) / rate if rate > 0 else 0
                        err_mark = "✗" if result.error else "✓"
                        route_mark = f" [{result.route_taken}]" if result.route_taken else ""
                        print(
                            f"  {err_mark} {done}/{total} "
                            f"{result.strategy_id.value}:{result.sample_id}"
                            f" {result.latency_ms:.0f}ms"
                            f"{route_mark}"
                            f" (ETA {eta_sec:.0f}s)",
                            flush=True,
                        )

                        if done % self.checkpoint_every == 0:
                            self._save_checkpoint(base_results + results)
                            print(
                                f"  💾 checkpoint: {done}/{total} "
                                f"({rate:.1f} req/s)",
                                flush=True,
                            )

        except KeyboardInterrupt:
            print(f"\n  Прерывание! Сохраняю checkpoint ({len(results)} новых)...", flush=True)
        finally:
            if results:
                with self._lock:
                    self._save_checkpoint(base_results + results)
            elapsed = time.perf_counter() - start_time
            print(f"\nГотово: {done}/{total} за {elapsed:.1f}s", flush=True)

        return results

    def _save_checkpoint(self, results: list[StrategyResult]) -> None:
        """Атомарное сохранение результатов в JSONL (tempfile + rename)."""
        parent_dir = self.output_path.parent
        fd, tmp_path = tempfile.mkstemp(
            dir=parent_dir, suffix=".tmp", prefix=".checkpoint_",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for r in results:
                    f.write(r.model_dump_json() + "\n")
            os.replace(tmp_path, self.output_path)
        except BaseException:
            # Удаляем временный файл при ошибке
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def load_results(results_path: str | Path) -> list[StrategyResult]:
        """Загрузка результатов из JSONL."""
        results: list[StrategyResult] = []
        path = Path(results_path)
        if not path.exists():
            return results
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(StrategyResult.model_validate_json(line))
        return results

    @staticmethod
    def get_completed_keys(results: list[StrategyResult]) -> set[str]:
        """Получить set ключей (sample_id:strategy_id) для resume."""
        return {f"{r.sample_id}:{r.strategy_id.value}" for r in results}
