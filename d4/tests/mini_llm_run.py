"""Mini LLM Run: тестовый прогон на mini/hard/pilot eval set.

По умолчанию — core strategies (S1, S2, S3, S4) + B0 (descriptive).
С флагом --experimental добавляются S4r, S5 (research branches).

Запуск:
  cd study && source .env
  .venv/bin/python d4/tests/mini_llm_run.py                # core only
  .venv/bin/python d4/tests/mini_llm_run.py --experimental  # + S4r, S5
  .venv/bin/python d4/tests/mini_llm_run.py --pilot          # pilot_dev_30
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
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
from d4.evaluation.retrieval_metrics import (
    compute_batch_retrieval,
    aggregate_retrieval_metrics,
    compute_retrieval_score,
)
from d4.models import EvalSample, RetrievalScore, StrategyResult
from d4.pipeline.chunker import load_chunks
from d4.pipeline.factory import (
    build_llm_runner,
    build_strategies,
    enrich_chunks_from_config,
    precompute_embeddings,
)
from d4.pipeline.orchestrator import Orchestrator
from d4.analysis.artifacts import D4_ROOT as _D4_ROOT, make_run_paths
from d4.analysis.run_manifest import build_manifest, save_manifest

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

_CONFIG_PATH = _D4_ROOT / "configs" / "experiment.yaml"
_GEN_CONFIG_PATH = _D4_ROOT / "configs" / "generation_config.yaml"
_CHUNKS_PATH = _D4_ROOT / "data" / "chunks_frozen.json"
_DOCTORS_PATH = _D4_ROOT / "data" / "kb" / "doctors.yaml"
_MINI_EVAL_PATH = Path(__file__).resolve().parent / "mini_eval_set.yaml"
_HARD_EVAL_PATH = Path(__file__).resolve().parent / "mini_eval_set_hard.yaml"
_BLIND_EVAL_PATH = Path(__file__).resolve().parent / "blind_hard_holdout.yaml"
_DEV_V2_PATH = _D4_ROOT / "data" / "eval_set_dev_v2.yaml"
_PILOT_EVAL_PATH = _D4_ROOT / "outputs" / "pilot_dev_30.yaml"
_OUTPUT_PATH = _D4_ROOT / "outputs" / "mini_run_results.jsonl"
_OUTPUT_PATH_HARD = _D4_ROOT / "outputs" / "mini_run_results_hard.jsonl"
_OUTPUT_PATH_BLIND = _D4_ROOT / "outputs" / "mini_run_results_blind.jsonl"

_PILOT_QUOTAS: dict[str, int] = {
    "clinic_info": 7,
    "doctor_info": 5,
    "pricing": 5,
    "reasoning": 5,
    "out_of_scope": 4,
    "aftercare": 4,
}
_PILOT_SEED = 42


def _build_pilot_subset(
    dev_path: Path,
    quotas: dict[str, int],
    seed: int,
) -> list:
    """Стратифицированная выборка из dev set с группировкой по seed_family_id.

    Гарантирует:
    - покрытие всех категорий с заданными квотами
    - целостность seed-семей (не разбивает вариации)
    - воспроизводимость через фиксированный seed
    """
    from d4.data_gen.query_generator import load_eval_set

    all_samples = load_eval_set(dev_path)
    rng = random.Random(seed)

    by_category: dict[str, list] = defaultdict(list)
    for s in all_samples:
        by_category[s.category].append(s)

    selected: list = []
    for cat, quota in quotas.items():
        pool = by_category.get(cat, [])
        if not pool:
            print(f"  ⚠ категория '{cat}' пуста в dev set")
            continue

        families: dict[str, list] = defaultdict(list)
        for s in pool:
            fid = s.seed_family_id or s.sample_id
            families[fid].append(s)

        family_keys = list(families.keys())
        rng.shuffle(family_keys)

        picked: list = []
        for fk in family_keys:
            if len(picked) >= quota:
                break
            members = families[fk]
            rng.shuffle(members)
            picked.append(members[0])

        if len(picked) < quota and len(pool) > len(picked):
            remaining = [s for s in pool if s not in picked]
            rng.shuffle(remaining)
            for s in remaining:
                if len(picked) >= quota:
                    break
                picked.append(s)

        selected.extend(picked)
        print(f"  {cat}: {len(picked)}/{quota} (pool={len(pool)}, families={len(families)})")

    print(f"  Итого: {len(selected)} samples")
    return selected


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Mini LLM Run")
    parser.add_argument(
        "--hard", action="store_true",
        help="Запуск на hard eval set вместо smoke",
    )
    parser.add_argument(
        "--both", action="store_true",
        help="Запуск на обоих наборах (smoke + hard) последовательно",
    )
    parser.add_argument(
        "--blind", action="store_true",
        help="Запуск на blind holdout set (independent sanity-check)",
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Pilot dev run: 30 stratified samples из eval_set_dev_v2",
    )
    parser.add_argument(
        "--full-dev", action="store_true",
        help="Full dev run: весь eval_set_dev_v2 (137 samples)",
    )
    parser.add_argument(
        "--experimental", action="store_true",
        help="Включить research branches (S4r, S5) в прогон",
    )
    parser.add_argument(
        "--representation",
        choices=["plain", "contextual", "llm_enriched"],
        default=None,
        help="Stage 2: chunk representation mode (C0/C1/C2). Если не указан — берётся из experiment.yaml",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Внешний run_id (YYYYMMDD_HHMMSS). Если не указан — генерируется автоматически.",
    )
    args = parser.parse_args()
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ОШИБКА: OPENROUTER_API_KEY не установлен")
        sys.exit(1)

    # Определяем набор(ы) для запуска
    runs: list[tuple[str, Path, Path]] = []
    if args.full_dev:
        runs.append(("full_dev", _DEV_V2_PATH, Path("/dev/null")))
    elif args.pilot:
        print(f"\n[0/6] Генерация pilot_dev_30 subset (seed={_PILOT_SEED})...", flush=True)
        pilot_samples = _build_pilot_subset(_DEV_V2_PATH, _PILOT_QUOTAS, _PILOT_SEED)
        from d4.data_gen.query_generator import save_eval_set
        save_eval_set(pilot_samples, _PILOT_EVAL_PATH)
        print(f"  Сохранён: {_PILOT_EVAL_PATH}\n", flush=True)
        runs.append(("pilot_dev_30", _PILOT_EVAL_PATH, Path("/dev/null")))
    elif args.both:
        runs.append(("smoke", _MINI_EVAL_PATH, _OUTPUT_PATH))
        runs.append(("hard", _HARD_EVAL_PATH, _OUTPUT_PATH_HARD))
    elif args.blind:
        runs.append(("blind", _BLIND_EVAL_PATH, _OUTPUT_PATH_BLIND))
    elif args.hard:
        runs.append(("hard", _HARD_EVAL_PATH, _OUTPUT_PATH_HARD))
    else:
        runs.append(("smoke", _MINI_EVAL_PATH, _OUTPUT_PATH))

    # Загрузка конфигурации
    config = load_config(_CONFIG_PATH)
    print(f"[1/6] Config: LLM={config.llm.model}, Judge={config.judge.model}", flush=True)

    # Загрузка KB (общая для всех runs)
    chunks = load_chunks(_CHUNKS_PATH)
    with open(_DOCTORS_PATH, encoding="utf-8") as f:
        doctors = yaml.safe_load(f).get("doctors", [])
    print(f"[2/6] Data: {len(chunks)} chunks, {len(doctors)} doctors", flush=True)

    # Stage 2: chunk representation enrichment (C0/C1/C2)
    if args.representation:
        config.representation.mode = args.representation
    rep_mode = config.representation.mode

    # Versioned output: каждый прогон → outputs/runs/{run_id}/
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_paths = make_run_paths(run_id)
    print(f"[RUN] {run_id}  mode={rep_mode}  dir={run_paths.run_dir}", flush=True)

    if args.pilot:
        pilot_out = run_paths.results_jsonl(rep_mode, run_name="pilot_dev_30")
        runs = [("pilot_dev_30", _PILOT_EVAL_PATH, pilot_out)]
    else:
        runs = [(name, ep, run_paths.results_jsonl(rep_mode, run_name=name)) for name, ep, _ in runs]

    if rep_mode != "plain":
        print(f"[2.5/6] Enrichment: mode={rep_mode}...", flush=True)
        chunks = enrich_chunks_from_config(config, chunks, api_key=api_key)
        print(f"[2.5/6] Enrichment OK: {len(chunks)} chunks enriched", flush=True)

    # Инициализация стратегий (core + optional experimental)
    strategies = build_strategies(
        config,
        include_baseline=True,
        include_experimental=args.experimental,
    )
    print(f"[3/6] Strategies: {[s.strategy_id for s in strategies]}", flush=True)

    # Предвычисление embeddings (S3, S4, S4r, S5)
    print("[4/6] Embeddings...", flush=True)
    precompute_embeddings(strategies, chunks)
    print("[4/6] Embeddings OK", flush=True)

    # LLM Runner
    llm_runner = build_llm_runner(config, api_key)
    print("[5/6] LLM Runner OK", flush=True)

    for run_name, eval_path, output_path in runs:
        mini_samples = load_eval_set(eval_path)
        gold_map = build_gold_map(mini_samples)

        is_pilot = run_name.startswith("pilot")
        orchestrator = Orchestrator(
            strategies=strategies,
            llm_runner=llm_runner,
            chunks=chunks,
            checkpoint_every=25 if is_pilot else 10,
            max_workers=3 if is_pilot else 2,
            output_path=output_path,
        )

        n_strats = len(strategies)
        n_samples = len(mini_samples)
        print(f"\n{'='*60}", flush=True)
        print(f"[6/6] Mini LLM Run [{run_name}]: {n_samples} samples × {n_strats} strategies = {n_samples * n_strats} задач", flush=True)
        print(f"{'='*60}\n", flush=True)

        start = time.perf_counter()
        results = orchestrator.run_all(mini_samples)
        elapsed = time.perf_counter() - start
        print(f"\nВсего результатов: {len(results)}, время: {elapsed:.1f}s")

        _print_report(run_name, rep_mode, run_id, results, mini_samples, gold_map, chunks, output_path, elapsed)

    # Manifest: сохраняем полный fingerprint прогона
    cmd = " ".join(sys.argv)
    actual_run_name = runs[0][0] if runs else "unknown"
    actual_eval_path = str(runs[0][1]) if runs else ""
    manifest = build_manifest(
        run_paths,
        modes=[rep_mode],
        run_name=actual_run_name,
        eval_set_path=actual_eval_path,
        command=cmd,
    )
    save_manifest(manifest, run_paths)
    print(f"\n📋 Manifest: {run_paths.manifest_json()}")


def _save_report_json(
    run_name: str,
    rep_mode: str,
    run_id: str,
    results: list[StrategyResult],
    samples: list[EvalSample],
    gold_map: dict,
    chunks: list,
    output_path: Path,
    elapsed_s: float,
) -> Path:
    """Сохраняет структурированный JSON-отчёт со всеми агрегированными метриками."""
    sample_map = {s.sample_id: s for s in samples}
    successful = [r for r in results if not r.error]
    all_sids = ["B0", "S1", "S2", "S3", "S4", "S4r", "S5"]

    by_strategy_ok: dict[str, list[StrategyResult]] = defaultdict(list)
    for r in successful:
        by_strategy_ok[r.strategy_id.value].append(r)

    # Quality metrics
    quality: dict[str, dict] = {}
    for sid in all_sids:
        strat = by_strategy_ok.get(sid, [])
        if not strat:
            continue
        quality[sid] = {
            "n": len(strat),
            "answerable_true": sum(1 for r in strat if r.answer.answerable),
            "avg_confidence": round(sum(r.answer.confidence for r in strat) / len(strat), 3),
            "avg_latency_ms": round(sum(r.latency_ms for r in strat if r.latency_ms) / len(strat), 0),
        }

    # Retrieval metrics
    ret_scores = compute_batch_retrieval(successful, gold_map)
    ret_agg = aggregate_retrieval_metrics(ret_scores)
    retrieval: dict[str, dict] = {}
    for sid, row in ret_agg.items():
        retrieval[sid] = {k: round(v, 4) for k, v in row.items()}

    # Deterministic evaluation
    det_scores = evaluate_batch(successful, samples, chunks)
    deterministic: dict[str, dict] = {}
    for sid in all_sids:
        strat_scores = [s for s in det_scores if s.strategy_id.value == sid]
        if not strat_scores:
            continue
        n = len(strat_scores)
        fmr_vals = [s.fact_match_rate for s in strat_scores if s.fact_match_rate is not None]
        doctor_bearing = [
            s for s in strat_scores
            if sample_map.get(s.sample_id) and sample_map[s.sample_id].expected_doctor
        ]
        deterministic[sid] = {
            "n": n,
            "answerability_correct": sum(1 for s in strat_scores if s.answerability_correct),
            "doctor_match": sum(1 for s in doctor_bearing if s.doctor_match),
            "doctor_total": len(doctor_bearing),
            "avg_fmr": round(sum(fmr_vals) / len(fmr_vals), 4) if fmr_vals else None,
            "fmr_annotated": len(fmr_vals),
            "unsupported_claims": sum(s.unsupported_claims for s in strat_scores),
            "total_claims": sum(s.total_claims for s in strat_scores),
        }

    # Rank analysis per sample
    rank_analysis: dict[str, dict[str, float]] = {}
    for s in ret_scores:
        rank_analysis.setdefault(s.sample_id, {})[s.strategy_id.value] = round(s.reciprocal_rank, 4)

    report = {
        "meta": {
            "run_id": run_id,
            "run_name": run_name,
            "representation_mode": rep_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_samples": len(samples),
            "n_results": len(results),
            "n_errors": len(results) - len(successful),
            "elapsed_s": round(elapsed_s, 1),
            "results_jsonl": str(output_path),
        },
        "quality": quality,
        "retrieval": retrieval,
        "deterministic": deterministic,
        "rank_analysis": rank_analysis,
    }

    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n📊 Report JSON: {report_path}")
    return report_path


def _print_rank_shift(
    scores: list[RetrievalScore],
    samples: list[EvalSample],
) -> None:
    """Per-sample rank analysis: показывает позицию gold chunk для каждого sample×strategy."""
    sample_map = {s.sample_id: s for s in samples}
    by_sample: dict[str, dict[str, float]] = defaultdict(dict)
    for s in scores:
        by_sample[s.sample_id][s.strategy_id.value] = s.reciprocal_rank

    sids = sorted({s.strategy_id.value for s in scores})
    if not sids:
        return

    print("\n--- Rank Analysis (reciprocal rank per sample, 0 = miss) ---\n")
    header = f"  {'sample_id':20s}  " + "  ".join(f"{sid:>6s}" for sid in sids)
    print(header)
    print(f"  {'-'*20}  " + "  ".join("------" for _ in sids))

    for sid_data in sorted(by_sample.items()):
        sample_id, rr_map = sid_data
        sample = sample_map.get(sample_id)
        cat = sample.category[:8] if sample else "?"
        cells = []
        for sid in sids:
            rr = rr_map.get(sid, 0.0)
            cells.append(f"{rr:6.3f}" if rr > 0 else "  miss")
        print(f"  {sample_id:20s}  " + "  ".join(cells) + f"  [{cat}]")

    for sid in sids:
        rr_vals = [v.get(sid, 0.0) for v in by_sample.values()]
        avg = sum(rr_vals) / len(rr_vals) if rr_vals else 0
        n_miss = sum(1 for v in rr_vals if v == 0)
        print(f"\n  {sid}: avg_RR={avg:.3f}, misses={n_miss}/{len(rr_vals)}")
    print()


def _print_report(
    run_name: str,
    rep_mode: str,
    run_id: str,
    results: list[StrategyResult],
    samples: list[EvalSample],
    gold_map: dict,
    chunks: list,
    output_path: Path,
    elapsed_s: float = 0.0,
) -> None:
    """Печать отчёта по результатам прогона + сохранение JSON."""

    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ [{run_name.upper()}]")
    print(f"{'='*60}\n")

    by_strategy: dict[str, list[StrategyResult]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy_id.value, []).append(r)

    all_sids = ["B0", "S1", "S2", "S3", "S4", "S4r", "S5"]

    # 0. Infra failures
    all_errors = [r for r in results if r.error]
    if all_errors:
        print("--- Infra Failures ---\n")
        for r in all_errors:
            print(f"  ✗ {r.strategy_id.value}:{r.sample_id} — {r.error}")
        print(f"\n  Итого: {len(all_errors)}/{len(results)} ({len(all_errors)/len(results):.0%})\n")
    else:
        print("--- Infra Failures: 0 ---\n")

    successful = [r for r in results if not r.error]
    by_strategy_ok: dict[str, list[StrategyResult]] = {}
    for r in successful:
        by_strategy_ok.setdefault(r.strategy_id.value, []).append(r)

    # 1. Quality metrics
    print("--- Quality Metrics (successful runs only) ---\n")
    for sid in all_sids:
        strat_results = by_strategy_ok.get(sid, [])
        strat_all = by_strategy.get(sid, [])
        if not strat_all:
            continue
        n_errors = len(strat_all) - len(strat_results)
        answerable_true = sum(1 for r in strat_results if r.answer.answerable)
        avg_confidence = (
            sum(r.answer.confidence for r in strat_results) / len(strat_results)
            if strat_results else 0.0
        )
        avg_latency = (
            sum(r.latency_ms for r in strat_results if r.latency_ms) / len(strat_results)
            if strat_results else 0.0
        )
        route_info = ""
        if sid == "S5":
            direct = sum(1 for r in strat_results if r.route_taken == "direct")
            fallback = sum(1 for r in strat_results if r.route_taken == "fallback")
            route_info = f", route: {direct} direct / {fallback} fallback"
        error_info = f" ({n_errors} infra errors excluded)" if n_errors else ""
        print(
            f"  {sid}: {len(strat_results)}/{len(strat_all)} successful, "
            f"predicted_answerable={answerable_true}/{len(strat_results)}, "
            f"avg_confidence={avg_confidence:.2f}, "
            f"avg_latency={avg_latency:.0f}ms{route_info}{error_info}"
        )

    # 2. Детальные ответы (S1 vs B0)
    print("\n--- Детальные ответы (S1 vs B0) ---\n")
    sample_map = {s.sample_id: s for s in samples}
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

    # 3. Retrieval метрики (hit@k, MRR, rank-shift)
    print("--- Retrieval метрики ---\n")
    ret_scores = compute_batch_retrieval(successful, gold_map)
    ret_agg = aggregate_retrieval_metrics(ret_scores)
    for sid in ["S2", "S3", "S4", "S4r", "S5"]:
        row = ret_agg.get(sid)
        if not row:
            continue
        n = int(row["n_samples"])
        print(
            f"  {sid}: hit@5={row['hit_rate']:.0%}, "
            f"MRR={row['mrr']:.3f}, "
            f"recall@5={row['mean_recall']:.2f}, "
            f"in_ctx={row['gold_in_context_rate']:.0%} "
            f"(n={n})"
        )

    # 3a. Per-sample rank-shift (gold rank для каждого sample)
    _print_rank_shift(ret_scores, samples)

    # 4. Deterministic evaluation
    print("\n--- Deterministic Evaluation (successful runs only) ---\n")
    det_scores = evaluate_batch(successful, samples, chunks)
    for sid in all_sids:
        strat_scores = [s for s in det_scores if s.strategy_id.value == sid]
        if not strat_scores:
            continue
        n = len(strat_scores)
        ans_correct = sum(1 for s in strat_scores if s.answerability_correct)

        doctor_bearing = [
            s for s in strat_scores
            if sample_map.get(s.sample_id) and sample_map[s.sample_id].expected_doctor
        ]
        doc_correct = sum(1 for s in doctor_bearing if s.doctor_match)
        n_doc = len(doctor_bearing)

        fmr_values = [s.fact_match_rate for s in strat_scores if s.fact_match_rate is not None]
        avg_fmr = sum(fmr_values) / len(fmr_values) if fmr_values else float("nan")
        fmr_info = f"{avg_fmr:.2f}" if fmr_values else "N/A (no gold_facts)"
        unsupported = sum(s.unsupported_claims for s in strat_scores)
        total_claims = sum(s.total_claims for s in strat_scores)
        doc_info = f"{doc_correct}/{n_doc}" if n_doc else "N/A"
        print(
            f"  {sid}: answerability={ans_correct}/{n} ({ans_correct/n:.0%}), "
            f"doctor_match={doc_info}, "
            f"fact_match_rate={fmr_info} ({len(fmr_values)}/{n} annotated), "
            f"unsupported_claims={unsupported}/{total_claims}"
        )

    # 5. Out-of-scope
    print("\n--- Out-of-scope (ожидание: answerable=false) ---\n")
    oos_samples = [s for s in samples if s.category == "out_of_scope"]
    for sample in oos_samples:
        print(f"  {sample.sample_id}: '{sample.query}'")
        for sid in all_sids:
            strat_results = by_strategy.get(sid, [])
            r = next((r for r in strat_results if r.sample_id == sample.sample_id), None)
            if r:
                marker = "✓" if not r.answer.answerable else "✗"
                ans_preview = r.answer.answer[:60].replace("\n", " ")
                print(f"    {sid}: answerable={r.answer.answerable} {marker} | {ans_preview}")
        print()

    # 6. S5 confidence breakdown
    s5_results = by_strategy.get("S5", [])
    if s5_results:
        print("--- S5 Confidence Breakdown ---\n")
        for r in sorted(s5_results, key=lambda x: x.sample_id):
            cd = r.confidence_debug or {}
            conf = cd.get("confidence", "?")
            mt = cd.get("match_type", "?")
            em = cd.get("entity_match", "?")
            top_cid = cd.get("top_chunk_id", "?")
            comps = cd.get("components", {})
            fb_reason = cd.get("fallback_reason", "")
            sample = sample_map.get(r.sample_id)
            cat = f"{sample.category}/{sample.subtype}" if sample else "?"
            reason_str = f"  reason={fb_reason}" if fb_reason else ""
            print(
                f"  {r.sample_id:15s}  route={r.route_taken:8s}  "
                f"conf={conf}  match={mt}  entity={em}  "
                f"chunk={top_cid}{reason_str}"
            )
            if comps:
                print(f"  {'':15s}  components: {comps}  category: {cat}")
        print()

    # JSON report
    _save_report_json(
        run_name, rep_mode, run_id, results, samples, gold_map, chunks, output_path, elapsed_s,
    )

    print(f"Результаты сохранены: {output_path}")
    print(f"Всего LLM вызовов: ~{sum(1 for r in results if r.tokens_prompt > 0)}")


if __name__ == "__main__":
    main()
