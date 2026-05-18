"""Runner: обучение и оценка всех D1 v6 baselines.

Запуск:
    cd study && python -m d1.scripts.run_baselines [--eval-sets test hard_test safety_set]

По §9 ТЗ:
  B0 — rules-only (no training)
  B1 — TF-IDF + LinearSVC
  B2 — strong embedding (bge-m3) + linear head

Safety_set оценивается через SafetyReport (recall_urgent, FN),
остальные eval sets — через RoutingReport (accuracy, macro_f1, etc.).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.baselines.eval_metrics import (
    SWITCH_INTERPRETATION_WARNING,
    RoutingReport,
    SafetyReport,
    SwitchStressReport,
    compute_all_metrics,
    compute_safety_report,
    compute_switch_stress_report,
    print_report,
    print_safety_report,
    print_switch_stress_report,
)
from d1.baselines.trained_bundle import BASELINE_CONFIGS, train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR, SWITCH_CSV_COLUMNS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(name: str) -> pd.DataFrame:
    """Загрузка split CSV.

    Для `switch_test` дополнительно валидируется наличие колонки
    `active_domain` (из SWITCH_CSV_COLUMNS) — требуется для per-transition
    breakdown в SwitchStressReport. Остальные eval-сеты используют базовые
    CSV_COLUMNS без изменений.
    """
    path = DATA_DIR / f"{DATASET_PREFIX}_{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split не найден: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")

    if name in _SWITCH_EVAL_SETS:
        missing = [c for c in SWITCH_CSV_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{name}.csv: отсутствуют колонки {missing} "
                f"(ожидается SWITCH_CSV_COLUMNS=...)",
            )

    logger.info("Loaded %s: %d rows", name, len(df))
    return df


def extract_texts_labels(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Извлечение текстов, меток и urgency из DataFrame."""
    texts = df["text"].tolist()
    labels = df["route_domain"].tolist()
    urgency = df["urgency"].tolist() if "urgency" in df.columns else []
    return texts, labels, urgency


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def _measure_latency(predict_fn, texts: list[str], n_warmup: int = 1) -> float:
    """Среднее время предсказания на запрос (ms)."""
    # Warmup
    for _ in range(n_warmup):
        predict_fn(texts[:min(5, len(texts))])

    t0 = time.perf_counter()
    predict_fn(texts)
    total_ms = (time.perf_counter() - t0) * 1000
    return total_ms / len(texts)


_SAFETY_EVAL_SETS = {"safety_set"}
_SWITCH_EVAL_SETS = {"switch_test"}


def run_all_baselines(
    test_set_name: str = "test",
    eval_sets: list[str] | None = None,
) -> tuple[
    dict[str, list[RoutingReport]],
    dict[str, list[SafetyReport]],
    dict[str, list[SwitchStressReport]],
]:
    """Обучение и оценка B0/B1/B2 на всех eval sets.

    Три режима оценки (по имени eval_set):
      - `safety_set` → SafetyReport (§10.2, urgency recall, misrouted)
      - `switch_test` → SwitchStressReport (text-only, per-transition)
      - остальные → RoutingReport (полный §10)

    Returns:
        (routing_reports, safety_reports, switch_reports)
    """
    if eval_sets is None:
        eval_sets = [test_set_name]

    # SSoT обучения — все baseline'ы через TrainedBundle (Task 0 из roadmap).
    # use_cache=True: если ключ совпадает (params + dataset content + code + env
    # + schema), модели загружаются мгновенно; иначе пере-обучаются.
    enabled_names = [n for n, cfg in BASELINE_CONFIGS.items() if cfg["enabled"]]
    logger.info("=== train_bundle: %d enabled baselines ===", len(enabled_names))
    bundle = train_bundle(names=enabled_names, use_cache=True)

    # Порядок важен для консистентности с историческими отчётами.
    # Контракт: все enabled=True baseline'ы из BASELINE_CONFIGS должны
    # присутствовать в этом списке (проверяется в run-time ниже).
    _BASELINE_ORDER = [
        "B0_rules",
        "B1.1_tfidf_lr",
        "B1.3_fasttext",
        "B2.1_bge-m3_svc",
        "B2.5_e5-small_svc",
    ]
    # Защита от drift: если enabled baseline добавлен в конфиг, но забыт
    # в _BASELINE_ORDER — падаем громко, а не тихо дообучаем.
    missing_in_order = [n for n in enabled_names if n not in _BASELINE_ORDER]
    if missing_in_order:
        raise RuntimeError(
            f"Enabled baseline'ы отсутствуют в _BASELINE_ORDER: {missing_in_order}. "
            f"Добавьте их в порядок либо отключите через enabled=False.",
        )
    baselines = [(name, bundle.get(name)) for name in _BASELINE_ORDER
                 if name in bundle.models]

    # --- Evaluate ---
    all_routing: dict[str, list[RoutingReport]] = {}
    all_safety: dict[str, list[SafetyReport]] = {}
    all_switch: dict[str, list[SwitchStressReport]] = {}

    for eval_name in eval_sets:
        logger.info("\n--- Evaluating on: %s ---", eval_name)
        try:
            eval_df = load_split(eval_name)
        except FileNotFoundError:
            logger.warning("Skip eval set %s: not found", eval_name)
            continue

        eval_texts, eval_labels, eval_urgency = extract_texts_labels(eval_df)
        is_safety = eval_name in _SAFETY_EVAL_SETS
        is_switch = eval_name in _SWITCH_EVAL_SETS

        if is_switch:
            # Text-only stress test. active_domain используется ТОЛЬКО для
            # per_transition breakdown, НЕ подаётся в модель.
            active_domain = eval_df["active_domain"].tolist()
            switch_reports: list[SwitchStressReport] = []
            for name, model in baselines:
                latency = _measure_latency(model.predict, eval_texts)
                preds = model.predict(eval_texts)
                sr = compute_switch_stress_report(
                    y_true=eval_labels,
                    y_pred=preds,
                    active_domain=active_domain,
                    baseline_name=f"{name} @ {eval_name}",
                    latency_ms=latency,
                )
                switch_reports.append(sr)
                print_switch_stress_report(sr)
            all_switch[eval_name] = switch_reports
        elif is_safety:
            safety_reports: list[SafetyReport] = []
            for name, model in baselines:
                latency = _measure_latency(model.predict, eval_texts)
                preds = model.predict(eval_texts)
                sr = compute_safety_report(
                    y_true=eval_labels,
                    y_pred=preds,
                    urgency=eval_urgency,
                    baseline_name=f"{name} @ {eval_name}",
                    latency_ms=latency,
                )
                safety_reports.append(sr)
                print_safety_report(sr)
            all_safety[eval_name] = safety_reports
        else:  # standard routing (test/val/hard_test/blind_test/entity_held_out/extended_eval)
            routing_reports: list[RoutingReport] = []
            for name, model in baselines:
                latency = _measure_latency(model.predict, eval_texts)
                preds = model.predict(eval_texts)
                report = compute_all_metrics(
                    y_true=eval_labels,
                    y_pred=preds,
                    baseline_name=f"{name} @ {eval_name}",
                    urgency=eval_urgency or None,
                    latency_ms=latency,
                )
                routing_reports.append(report)
                print_report(report)
            all_routing[eval_name] = routing_reports

    return all_routing, all_safety, all_switch


def save_results(
    routing_reports: dict[str, list[RoutingReport]],
    safety_reports: dict[str, list[SafetyReport]],
    switch_reports: dict[str, list[SwitchStressReport]] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Сохранение результатов в JSON и CSV."""
    out = output_dir or RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    # --- Routing CSV ---
    rows = []
    for eval_name, reps in routing_reports.items():
        for r in reps:
            row = r.summary_dict()
            row["eval_set"] = eval_name
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = out / "baseline_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)

    # --- Routing JSON (с confusion matrices) ---
    json_data: dict[str, Any] = {}
    for eval_name, reps in routing_reports.items():
        json_data[eval_name] = []
        for r in reps:
            entry = r.summary_dict()
            entry["confusion"] = r.confusion
            entry["per_class"] = r.per_class
            json_data[eval_name].append(entry)

    json_path = out / "baseline_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", json_path)

    # --- Safety JSON ---
    if safety_reports:
        safety_data: dict[str, Any] = {}
        safety_rows = []
        for eval_name, reps in safety_reports.items():
            safety_data[eval_name] = [sr.summary_dict() for sr in reps]
            for sr in reps:
                row = sr.summary_dict()
                row["eval_set"] = eval_name
                safety_rows.append(row)

        safety_json_path = out / "safety_results.json"
        with open(safety_json_path, "w", encoding="utf-8") as f:
            json.dump(safety_data, f, ensure_ascii=False, indent=2)
        logger.info("Saved: %s", safety_json_path)

        safety_csv_path = out / "safety_results.csv"
        pd.DataFrame(safety_rows).to_csv(safety_csv_path, index=False)
        logger.info("Saved: %s", safety_csv_path)

    # --- Summary tables ---
    if not df.empty:
        print(f"\n{'='*80}")
        print("  ROUTING SUMMARY")
        print(f"{'='*80}")
        summary_cols = [
            "eval_set", "baseline", "accuracy", "macro_f1",
            "balanced_accuracy", "false_faq_for_anamnesis",
        ]
        existing_cols = [c for c in summary_cols if c in df.columns]
        print(df[existing_cols].to_string(index=False))
        print("  Latency — d1/results/latency_breakdown.csv (n=100, repeats=5)")
        print("  recall_urgent / FN / misrouted — d1/results/safety_results.csv (safety_set n=87)")

    if safety_reports:
        print(f"\n{'='*80}")
        print("  SAFETY SUMMARY")
        print(f"{'='*80}")
        for eval_name, reps in safety_reports.items():
            for sr in reps:
                print(
                    f"  {sr.baseline_name:40s}  "
                    f"recall_urg={sr.recall_urgent:.4f}  "
                    f"FN={sr.false_negative_urgent}  "
                    f"misrouted={sr.misrouted_to or '-'}"
                )

    # --- Switch stress (отдельные файлы) ---
    if switch_reports:
        _save_switch_results(switch_reports, out)
    print()


# ---------------------------------------------------------------------------
# Switch results persistence (отдельные файлы — не мешают plot_results.py)
# ---------------------------------------------------------------------------

def _save_switch_results(
    switch_reports: dict[str, list[SwitchStressReport]],
    out: Path,
) -> None:
    """Сохранение switch stress results в switch_results.{json,csv}.

    Отдельный файл во избежание поломки plot_results.py (он читает
    baseline_results.csv и не ожидает transition-level метрик). JSON содержит
    полный per_transition breakdown + interpretation_warning; CSV — только
    плоские summary для сводных таблиц.
    """
    # JSON: полная структура с metadata-варнингом
    json_data: dict[str, Any] = {
        "interpretation_warning": SWITCH_INTERPRETATION_WARNING,
        "results": {},
    }
    csv_rows: list[dict[str, Any]] = []
    for eval_name, reps in switch_reports.items():
        json_data["results"][eval_name] = [sr.summary_dict() for sr in reps]
        for sr in reps:
            row = {
                "eval_set": eval_name,
                "baseline": sr.baseline_name,
                "n_samples": sr.n_samples,
                "route_accuracy": round(sr.route_accuracy, 4),
            }
            # latency_ms намеренно не пишем: источник истины
            # latency_breakdown.csv (n=100, repeats=5). Inline-замер во
            # время switch_test (n=38) шумный и вводит в заблуждение.
            for key in sorted(sr.per_transition):
                row[f"acc__{key}"] = round(
                    sr.per_transition[key]["accuracy"], 4,
                )
            csv_rows.append(row)

    json_path = out / "switch_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", json_path)

    csv_path = out / "switch_results.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info("Saved: %s", csv_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Дефолт CLI: все 8 eval-сетов (Task 2 roadmap).
# blind_test/entity_held_out/extended_eval — standard routing,
# switch_test — SwitchStressReport, safety_set — SafetyReport.
_DEFAULT_EVAL_SETS = [
    "test", "val", "hard_test", "safety_set",
    "blind_test", "entity_held_out", "extended_eval", "switch_test",
]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="D1 v6 baseline benchmarks")
    parser.add_argument(
        "--eval-sets", nargs="+",
        default=_DEFAULT_EVAL_SETS,
        help="Eval sets для оценки",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    routing, safety, switch = run_all_baselines(eval_sets=args.eval_sets)
    save_results(routing, safety, switch)


if __name__ == "__main__":
    main()
