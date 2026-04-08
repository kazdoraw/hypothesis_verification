"""Mini LLM Run: тестовый прогон S1-S4 + B0 на 10 samples.

Цель: проверить работу промпта, генерации и evaluation
ПЕРЕД полноценным прогоном (~$25). Стоимость: ~$1.

Запуск:
  cd study && source .env
  /opt/anaconda3/envs/ml-python312/bin/python d4/tests/mini_llm_run.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Настройка sys.path
_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

import yaml

from d4.config import load_config
from d4.data_gen.query_generator import load_eval_set
from d4.evaluation.deterministic import evaluate_batch
from d4.evaluation.gold_map import build_gold_map
from d4.evaluation.retrieval_metrics import compute_retrieval_score
from d4.models import EvalSample, StrategyResult
from d4.pipeline.chunker import load_chunks
from d4.pipeline.factory import (
    build_llm_runner,
    build_strategies,
    precompute_embeddings,
)
from d4.pipeline.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

_D4_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _D4_ROOT / "configs" / "experiment.yaml"
_CHUNKS_PATH = _D4_ROOT / "data" / "kb" / "chunks.json"
_DOCTORS_PATH = _D4_ROOT / "data" / "kb" / "doctors.yaml"
_MINI_EVAL_PATH = Path(__file__).resolve().parent / "mini_eval_set.yaml"
_OUTPUT_PATH = _D4_ROOT / "tests" / "mini_run_results.jsonl"


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ОШИБКА: OPENROUTER_API_KEY не установлен")
        sys.exit(1)

    # Загрузка конфигурации
    config = load_config(_CONFIG_PATH)
    print(f"LLM: {config.llm.model}")
    print(f"Judge: {config.judge.model}")

    # Загрузка данных
    chunks = load_chunks(_CHUNKS_PATH)
    mini_samples = load_eval_set(_MINI_EVAL_PATH)
    with open(_DOCTORS_PATH, encoding="utf-8") as f:
        doctors = yaml.safe_load(f).get("doctors", [])
    gold_map = build_gold_map(mini_samples, doctors)
    print(f"Chunks: {len(chunks)}, Doctors: {len(doctors)}, Samples: {len(mini_samples)}")

    # Инициализация стратегий (S1-S4 + B0)
    strategies = build_strategies(config, include_baseline=True)
    print(f"Стратегии: {[s.strategy_id for s in strategies]}")

    # Предвычисление embeddings (S3, S4)
    print("Предвычисление embeddings...")
    precompute_embeddings(strategies, chunks)

    # LLM Runner
    llm_runner = build_llm_runner(config, api_key)

    # Orchestrator (последовательный для mini run)
    orchestrator = Orchestrator(
        strategies=strategies,
        llm_runner=llm_runner,
        chunks=chunks,
        checkpoint_every=50,
        max_workers=4,
        output_path=_OUTPUT_PATH,
    )

    # Запуск: все стратегии на всех mini samples
    # 10 samples × 5 стратегий = 50 задач (из них 10 B0 без LLM = 40 LLM вызовов)
    print(f"\n{'='*60}")
    print(f"Mini LLM Run: {len(mini_samples)} samples × {len(strategies)} strategies")
    print(f"Ожидание: ~40 LLM вызовов")
    print(f"{'='*60}\n")

    start = time.perf_counter()
    results = orchestrator.run_all(mini_samples)
    elapsed = time.perf_counter() - start
    print(f"\nВсего результатов: {len(results)}, время: {elapsed:.1f}s")

    # ---------------------------------------------------------------------------
    # Анализ результатов
    # ---------------------------------------------------------------------------

    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ MINI RUN")
    print(f"{'='*60}\n")

    # Группировка по стратегии
    by_strategy: dict[str, list[StrategyResult]] = {}
    for r in results:
        sid = r.strategy_id.value
        by_strategy.setdefault(sid, []).append(r)

    # 1. Общая статистика по стратегиям
    print("--- Общая статистика ---\n")
    for sid in ["B0", "S1", "S2", "S3", "S4"]:
        strat_results = by_strategy.get(sid, [])
        if not strat_results:
            continue
        errors = [r for r in strat_results if r.error]
        answerable_true = sum(1 for r in strat_results if r.answer.answerable)
        avg_confidence = (
            sum(r.answer.confidence for r in strat_results) / len(strat_results)
        )
        avg_latency = (
            sum(r.latency_ms for r in strat_results if r.latency_ms) / len(strat_results)
        )
        print(
            f"  {sid}: {len(strat_results)} results, "
            f"{len(errors)} errors, "
            f"answerable={answerable_true}/{len(strat_results)}, "
            f"avg_confidence={avg_confidence:.2f}, "
            f"avg_latency={avg_latency:.0f}ms"
        )

    # 2. Детальный вывод по каждому sample
    print("\n--- Детальные ответы (S1 vs B0) ---\n")
    sample_map = {s.sample_id: s for s in mini_samples}
    for r in sorted(results, key=lambda x: (x.sample_id, x.strategy_id.value)):
        if r.strategy_id.value not in ("S1", "B0"):
            continue
        sample = sample_map.get(r.sample_id)
        if not sample:
            continue
        answer_preview = r.answer.answer[:100].replace("\n", " ")
        marker = "✓" if r.answer.answerable == sample.answerable else "✗"
        print(
            f"  [{r.strategy_id.value}] {sample.sample_id} "
            f"({sample.category}) "
            f"ans={r.answer.answerable} (gold={sample.answerable}) {marker}"
        )
        print(f"       Q: {sample.query}")
        print(f"       A: {answer_preview}...")
        if r.answer.doctor:
            print(f"       Doctor: {r.answer.doctor}")
        print()

    # 3. Retrieval метрики (S2, S3, S4)
    print("--- Retrieval метрики ---\n")
    for sid in ["S2", "S3", "S4"]:
        strat_results = by_strategy.get(sid, [])
        if not strat_results:
            continue
        hits = 0
        total = 0
        for r in strat_results:
            golds = gold_map.get(r.sample_id, [])
            if not golds:
                continue
            total += 1
            score = compute_retrieval_score(r, golds)
            if score.hit_at_k:
                hits += 1
        rate = hits / total if total else 0
        print(f"  {sid} hit@5: {rate:.0%} ({hits}/{total})")

    # 4. Deterministic evaluation
    print("\n--- Deterministic Evaluation ---\n")
    det_scores = evaluate_batch(results, mini_samples, chunks)
    for sid in ["B0", "S1", "S2", "S3", "S4"]:
        strat_scores = [s for s in det_scores if s.strategy_id.value == sid]
        if not strat_scores:
            continue
        n = len(strat_scores)
        ans_correct = sum(1 for s in strat_scores if s.answerability_correct)
        doc_correct = sum(1 for s in strat_scores if s.doctor_match)
        spec_correct = sum(1 for s in strat_scores if s.specialization_match)
        unsupported = sum(s.unsupported_claims for s in strat_scores)
        total_claims = sum(s.total_claims for s in strat_scores)
        print(
            f"  {sid}: answerability={ans_correct}/{n} ({ans_correct/n:.0%}), "
            f"doctor_match={doc_correct}/{n}, "
            f"spec_match={spec_correct}/{n}, "
            f"unsupported_claims={unsupported}/{total_claims}"
        )

    # 5. Out-of-scope детализация
    print("\n--- Out-of-scope (ожидание: answerable=false) ---\n")
    oos_samples = [s for s in mini_samples if s.category == "out_of_scope"]
    for sample in oos_samples:
        print(f"  {sample.sample_id}: '{sample.query}'")
        for sid in ["S1", "S2", "S3", "S4", "B0"]:
            strat_results = by_strategy.get(sid, [])
            r = next((r for r in strat_results if r.sample_id == sample.sample_id), None)
            if r:
                marker = "✓" if not r.answer.answerable else "✗"
                ans_preview = r.answer.answer[:60].replace("\n", " ")
                print(f"    {sid}: answerable={r.answer.answerable} {marker} | {ans_preview}")
        print()

    print(f"\nРезультаты сохранены: {_OUTPUT_PATH}")
    print(f"Всего LLM вызовов: ~{sum(1 for r in results if r.tokens_prompt > 0)}")


if __name__ == "__main__":
    main()
