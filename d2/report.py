"""Генерация отчётов: MD summary + CSV таблица.

Читает результаты из results/dialogs/ и scores.json,
сохраняет отчёт в results/reports/.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from d2.cases import CASES
from d2.config import DIALOGS_DIR, REPORTS_DIR, DOCTOR_MODEL, PATIENT_MODEL
from d2.models import CaseResult

ROUTING_AXES = ["specialist_score", "service_score", "examination_score"]
QUALITY_AXES = ["data_sufficiency", "accuracy", "dialogue_quality"]
SCORE_AXES = ROUTING_AXES + QUALITY_AXES

cases_map: dict[int, object] = {c.case_id: c for c in CASES}


def _load_all_results() -> list[CaseResult]:
    """Загрузить все case_XX.json."""
    results = []
    for path in sorted(DIALOGS_DIR.glob("case_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append(CaseResult.model_validate(data))
    return results


def _load_scores() -> dict | None:
    """Загрузить оценки качества."""
    path = REPORTS_DIR / "d2_judge_scores.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


def _run_stats(result: CaseResult, schema_name: str) -> dict:
    """Базовая статистика прогона (turns, tokens, fields)."""
    run = result.runs.get(schema_name)
    if not run:
        return {"turns": 0, "tokens": 0, "fields": 0}
    return {
        "turns": run.turns,
        "tokens": run.tokens_doctor + run.tokens_patient,
        "fields": len([v for v in run.extracted.values() if v]),
    }


def generate_csv(
    results: list[CaseResult],
    save_path: Path,
    scores: dict | None = None,
) -> None:
    """CSV: одна строка = один кейс × одна схема."""
    rows = []
    for result in results:
        for schema_name in ["S1", "S2", "S3"]:
            stats = _run_stats(result, schema_name)
            row = {
                "case_id": result.case_id,
                "case_type": result.case_type,
                "schema": schema_name,
                "turns": stats["turns"],
                "tokens": stats["tokens"],
                "fields_count": stats["fields"],
            }
            # Routing доктора из результатов диалога
            run = result.runs.get(schema_name)
            if run and run.routing:
                row["doctor_specialists"] = "; ".join(run.routing.get("specialists", []))
                row["doctor_service"] = run.routing.get("service_type", "")
                row["doctor_examination"] = run.routing.get("examination", "")

            if scores and result.case_id in scores:
                js = scores[result.case_id].get(schema_name, {})
                for axis in SCORE_AXES:
                    row[axis] = js.get(axis, "")
                row["routing_match"] = js.get("routing_match", "")
            rows.append(row)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_markdown(
    results: list[CaseResult],
    save_path: Path,
    scores: dict | None = None,
) -> None:
    """MD отчёт с таблицей сравнения схем."""
    has_scores = scores is not None

    lines = [
        "# D2: LLM vs LLM — Отчёт",
        "",
        f"**Дата**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Doctor**: {DOCTOR_MODEL}",
        f"**Patient**: {PATIENT_MODEL}",
        f"**Кейсов**: {len(results)}",
        "",
        "---",
        "",
        "## Сводная таблица по схемам",
        "",
    ]

    if has_scores:
        lines.extend([
            "| Schema | Spec | Svc | Exam | Suf | Acc | Dlg | Route | Turns | Tokens |",
            "|--------|------|-----|------|-----|-----|-----|-------|-------|--------|",
        ])
        for sn in ["S1", "S2", "S3"]:
            vals = {a: [] for a in SCORE_AXES}
            route_match, total_count = 0, 0
            turns_l, tokens_l = [], []
            for result in results:
                if result.case_id in scores and sn in scores[result.case_id]:
                    js = scores[result.case_id][sn]
                    for a in SCORE_AXES:
                        vals[a].append(js.get(a, 0))
                    if js.get("routing_match"):
                        route_match += 1
                    total_count += 1
                stats = _run_stats(result, sn)
                turns_l.append(stats["turns"])
                tokens_l.append(stats["tokens"])
            avgs = {a: sum(vals[a]) / len(vals[a]) if vals[a] else 0 for a in SCORE_AXES}
            avg_turns = sum(turns_l) / len(turns_l) if turns_l else 0
            avg_tokens = sum(tokens_l) / len(tokens_l) if tokens_l else 0
            rt_pct = f"{route_match}/{total_count}" if total_count else "—"
            lines.append(
                f"| {sn} | {avgs['specialist_score']:.1f} | {avgs['service_score']:.1f} | "
                f"{avgs['examination_score']:.1f} | {avgs['data_sufficiency']:.1f} | "
                f"{avgs['accuracy']:.1f} | {avgs['dialogue_quality']:.1f} | "
                f"{rt_pct} | {avg_turns:.1f} | {avg_tokens:.0f} |"
            )
    else:
        lines.extend([
            "| Schema | Turns (avg) | Tokens (avg) | Fields (avg) |",
            "|--------|-------------|-------------|-------------|",
        ])
        for sn in ["S1", "S2", "S3"]:
            turns_l, tokens_l, fields_l = [], [], []
            for result in results:
                stats = _run_stats(result, sn)
                turns_l.append(stats["turns"])
                tokens_l.append(stats["tokens"])
                fields_l.append(stats["fields"])
            n = len(turns_l) or 1
            lines.append(
                f"| {sn} | {sum(turns_l)/n:.1f} | "
                f"{sum(tokens_l)/n:.0f} | {sum(fields_l)/n:.1f} |"
            )

    # Детали по кейсам
    lines.extend(["", "---", "", "## Детали по кейсам", ""])

    for result in results:
        lines.append(f"### Case {result.case_id}: {result.case_type}")
        lines.append("")

        # Эталонная маршрутизация из cases.py
        case_obj = cases_map.get(result.case_id)
        if case_obj and case_obj.reference_routing:
            ref = case_obj.reference_routing
            lines.append(f"**Эталон:** {', '.join(ref.get('specialists', []))} — {ref.get('service_type', '')}")
            lines.append("")

        if has_scores and result.case_id in scores:
            best_sn = max(
                ["S1", "S2", "S3"],
                key=lambda s: scores[result.case_id].get(s, {}).get("data_sufficiency", 0),
            )
            ref_score = scores[result.case_id].get(best_sn, {})

            lines.append("| Schema | Spec | Svc | Exam | Suf | Acc | Dlg | Route | Turns | Fields |")
            lines.append("|--------|------|-----|------|-----|-----|-----|-------|-------|--------|")
            for sn in ["S1", "S2", "S3"]:
                js = scores[result.case_id].get(sn, {})
                stats = _run_stats(result, sn)
                rm = "✓" if js.get("routing_match") else "✗"
                lines.append(
                    f"| {sn} | {js.get('specialist_score', 0)} | {js.get('service_score', 0)} | "
                    f"{js.get('examination_score', 0)} | {js.get('data_sufficiency', 0)} | "
                    f"{js.get('accuracy', 0)} | {js.get('dialogue_quality', 0)} | "
                    f"{rm} | {stats['turns']} | {stats['fields']} |"
                )

            # Doctor routing для лучшей схемы
            best_run = result.runs.get(best_sn)
            if best_run and best_run.routing:
                doc_r = best_run.routing
                lines.append("")
                lines.append(f"**Доктор ({best_sn}):** {', '.join(doc_r.get('specialists', []))} — {doc_r.get('service_type', '')}")

            if ref_score.get("missing_for_routing"):
                lines.append(f"**Не хватает ({best_sn}):** {', '.join(ref_score['missing_for_routing'])}")
            if ref_score.get("errors"):
                lines.append(f"**Ошибки ({best_sn}):** {', '.join(ref_score['errors'])}")
            if ref_score.get("reasoning"):
                lines.append(f"**Вывод ({best_sn}):** {ref_score['reasoning']}")
        else:
            lines.append("| Schema | Turns | Tokens | Fields |")
            lines.append("|--------|-------|--------|--------|")
            for sn in ["S1", "S2", "S3"]:
                stats = _run_stats(result, sn)
                lines.append(
                    f"| {sn} | {stats['turns']} | {stats['tokens']} | {stats['fields']} |"
                )

        lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")


def generate_report() -> list[Path]:
    """Генерировать все отчёты."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_all_results()
    scores = _load_scores()

    if not results:
        print("  Нет результатов для отчёта.")
        return []

    paths = []

    p = REPORTS_DIR / "d2_summary.md"
    generate_markdown(results, p, scores)
    paths.append(p)
    print(f"  → {p}")

    p = REPORTS_DIR / "d2_per_case.csv"
    generate_csv(results, p, scores)
    paths.append(p)
    print(f"  → {p}")

    return paths
