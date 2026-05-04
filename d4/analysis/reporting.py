"""Генерация markdown decision memo и artifact index.

Формирует итоговый отчёт Stage 2A с таблицами, verdict и ссылками
на графики. Сохраняется в run_paths.reports_dir.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from d4.analysis.artifacts import RunPaths, list_run_artifacts


def _df_to_markdown(df: pd.DataFrame, float_fmt: str = ".4f") -> str:
    """DataFrame → markdown table."""
    return df.to_markdown(floatfmt=float_fmt) if not df.empty else "_Нет данных_"


def _relative(path: Path, base: Path) -> str:
    """Относительный путь от base."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _render_significance_section(
    sig_df: pd.DataFrame,
    baseline: str = "plain",
) -> list[str]:
    """Markdown-секция paired significance: CI среднего, Δ vs baseline, p-value.

    Формат — одна таблица на метрику + сноска об ограничениях.
    ``★`` ставится, если CI парной разности не пересекает 0 (слабый прокси
    значимости, без поправок на множественные сравнения).

    Args:
        sig_df: output ``compute_significance_table()``.
        baseline: имя режима, с которым сравниваются остальные (отдельная
            строка без Δ/p).
    """
    if sig_df is None or sig_df.empty:
        return []

    lines: list[str] = []
    lines.append("## 3. Statistical Significance (per-sample, paired)")
    lines.append("")
    lines.append(
        "Paired bootstrap-CI и тесты по per-sample артефактам прогона "
        "(`report.json` → `rank_analysis`, `*.jsonl`, eval_set gold). "
        "FMR здесь **не** представлен: `save_judge_scores` в pipeline пока не вызывается, "
        "per-sample judge-оценки не сохранены (см. `d4/analysis/significance.py`)."
    )
    lines.append("")
    lines.append(
        "★ рядом со строкой — CI парной разности не пересекает 0 "
        "(слабый прокси значимости, без поправок на множественные сравнения)."
    )
    lines.append("")

    for metric in sig_df["metric"].unique():
        sub = sig_df[sig_df["metric"] == metric].copy()
        if sub.empty:
            continue
        n = int(sub["n"].iloc[0])
        lines.append(f"### {metric}  (n={n})")
        lines.append("")
        lines.append("| mode | mean | CI mean | Δ vs baseline | CI Δ | p | test | sig |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            is_base = r["mode"] == baseline
            if is_base:
                mean_ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
                lines.append(
                    f"| `{r['mode']}` (baseline) | {r['mean']:.4f} | {mean_ci} "
                    f"| — | — | — | — | — |"
                )
                continue
            mean_ci = f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            diff_ci = f"[{r['diff_ci_lo']:+.4f}, {r['diff_ci_hi']:+.4f}]"
            p_str = "—" if pd.isna(r["p_value"]) else f"{r['p_value']:.3f}"
            sig_flag = "★" if bool(r.get("significant", False)) else ""
            lines.append(
                f"| `{r['mode']}` | {r['mean']:.4f} | {mean_ci} "
                f"| {r['diff_vs_baseline']:+.4f} | {diff_ci} "
                f"| {p_str} | {r['test']} | {sig_flag} |"
            )
        lines.append("")

    return lines


def generate_stage2_memo(
    retrieval_table: pd.DataFrame,
    det_table: pd.DataFrame,
    gate_result: dict,
    run_paths: RunPaths,
    figures: list[Path] | None = None,
    run_meta: dict[str, dict] | None = None,
    significance_df: pd.DataFrame | None = None,
    significance_baseline: str = "plain",
    memo_title: str = "Stage 2A Decision Memo",
    memo_filename: str | None = None,
) -> Path:
    """Генерирует decision memo (markdown).

    Секции:
    0. Source of truth notice
    1. Retrieval comparison (markdown table)
    2. Quality comparison (markdown table)
    3. Statistical significance (опционально, если передан ``significance_df``)
    4. Gate verdict
    5. Figures (relative links)
    6. Artifact index

    Args:
        run_meta: {mode: meta_dict} из report.json["meta"] для каждого mode.
                  Используется для n_errors, n_samples, elapsed_s.
        significance_df: output ``compute_significance_table()``. Если передан —
            добавляется секция "3. Statistical Significance" с bootstrap-CI
            и paired-тестами. Если ``None`` — секция пропускается, нумерация
            начинается сразу с Decision Gate (это корректно: memo остаётся
            валидным без significance).
        significance_baseline: baseline-mode для раздела significance.
        memo_title: заголовок H1 в memo. По умолчанию "Stage 2A Decision Memo"
            (обратная совместимость). Для Stage 1 notebook'а передавать
            "Stage 1 Screening Decision Memo".
        memo_filename: имя файла в reports_dir. По умолчанию None →
            "stage2a_decision_memo.md" (через ``run_paths.memo_md()``).
            Для Stage 1 передавать "stage1_decision_memo.md".
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = gate_result.get("verdict", "unknown")
    signals = gate_result.get("signals", [])
    recommendation = gate_result.get("recommendation", "")

    lines = [
        f"# {memo_title}",
        f"",
        f"> **Source of truth**: `outputs/runs/{run_paths.run_id}/`",
        f"",
        f"**Run ID**: `{run_paths.run_id}`  ",
        f"**Timestamp**: {ts}  ",
        f"**Verdict**: **{verdict.upper()}**",
        f"",
    ]

    # n_errors warning
    if run_meta:
        error_modes = {m: meta.get("n_errors", 0) for m, meta in run_meta.items()
                       if meta.get("n_errors", 0) > 0}
        if error_modes:
            lines.append(f"> **⚠ Ошибки в прогоне**:")
            for m, n in error_modes.items():
                n_samples = run_meta[m].get("n_samples", "?")
                n_results = run_meta[m].get("n_results", "?")
                n_ok = n_results - n if isinstance(n_results, int) and isinstance(n, int) else "?"
                lines.append(f">   - `{m}`: {n} error(s) из {n_samples} samples ({n_ok} successful из {n_results} tasks)")
            lines.append(f">")
            lines.append(f"> Результаты остаются информативными, но полная чистота прогона не достигнута.")
            lines.append("")

    # Нумерация секций подстраивается под наличие significance-блока:
    # без него Gate=3, Figures=4, Artifact=5; с ним Gate=4, Figures=5, Artifact=6.
    has_sig = significance_df is not None and not significance_df.empty
    gate_idx = 4 if has_sig else 3
    figures_idx = gate_idx + 1
    artifacts_idx = figures_idx + 1

    lines.extend([
        f"---",
        f"",
        f"## 1. Retrieval Comparison",
        f"",
        _df_to_markdown(retrieval_table),
        f"",
        f"## 2. Quality Comparison",
        f"",
        _df_to_markdown(det_table),
        f"",
    ])

    if has_sig:
        lines.extend(_render_significance_section(significance_df, significance_baseline))

    lines.extend([
        f"## {gate_idx}. Decision Gate",
        f"",
        f"**Verdict**: `{verdict}`  ",
        f"**Recommendation**: {recommendation}",
        f"",
        f"### Signals",
        f"",
    ])
    if signals:
        for s in signals:
            lines.append(f"- {s}")
    else:
        lines.append("_Нет явных сигналов_")

    if figures:
        lines.extend([
            f"",
            f"## {figures_idx}. Figures",
            f"",
        ])
        for fig_path in figures:
            rel = _relative(fig_path, run_paths.run_dir)
            lines.append(f"![{fig_path.stem}]({rel})")

    # Artifact index
    artifacts = list_run_artifacts(run_paths)
    if artifacts:
        lines.extend([
            f"",
            f"## {artifacts_idx}. Artifact Index",
            f"",
            f"| Категория | Файл |",
            f"|-----------|------|",
        ])
        for cat, p in artifacts:
            rel = _relative(p, run_paths.run_dir)
            lines.append(f"| {cat} | `{rel}` |")

    lines.append("")
    if memo_filename is None:
        memo_path = run_paths.memo_md()
    else:
        memo_path = run_paths.report_md(memo_filename)
    memo_path.write_text("\n".join(lines), encoding="utf-8")
    return memo_path


# ---------------------------------------------------------------------------
# Stage 1 helpers: ранжирование СТРАТЕГИЙ внутри одного режима
# ---------------------------------------------------------------------------

# Метрики Stage 1 ranking: ключ из таблицы → (читаемое имя, направление).
# direction: "max" — больше лучше; "min" — меньше лучше.
_STAGE1_METRICS: list[tuple[str, str, str, str]] = [
    ("hit_rate",          "hit@5",                "retrieval", "max"),
    ("mrr",               "MRR",                  "retrieval", "max"),
    ("mean_recall",       "Mean Recall@5",        "retrieval", "max"),
    ("avg_fmr",           "Fact Match Rate",      "quality",   "max"),
    ("doctor_match_rate", "Doctor Match Rate",    "quality",   "max"),
    ("answerability_rate","Answerability Rate",   "quality",   "max"),
]


def rank_strategies(
    retrieval_table: pd.DataFrame,
    det_table: pd.DataFrame,
    mode: str = "plain",
) -> pd.DataFrame:
    """Stage 1: построить ранжирование стратегий по ключевым метрикам.

    Возвращает long-DataFrame с колонками
    ``[metric, section, strategy, value, rank]``, где ``rank=1`` — лучшая
    стратегия по данной метрике (учитывая направление: для всех текущих
    Stage 1-метрик «больше — лучше»).

    Args:
        retrieval_table: output ``build_retrieval_table()``.
        det_table: output ``build_deterministic_table()``.
        mode: режим представления, для которого строим ранжирование.
            Stage 1 использует ``plain``.
    """
    rows: list[dict] = []
    for metric_key, metric_label, section, direction in _STAGE1_METRICS:
        table = retrieval_table if section == "retrieval" else det_table
        col = f"{mode}::{metric_key}"
        if col not in table.columns:
            continue
        sub = table[[col]].dropna().copy()
        if sub.empty:
            continue
        ascending = (direction == "min")
        sub["rank"] = sub[col].rank(ascending=ascending, method="min").astype(int)
        for strategy, row in sub.iterrows():
            rows.append({
                "metric":   metric_label,
                "metric_key": metric_key,
                "section":  section,
                "strategy": strategy,
                "value":    float(row[col]),
                "rank":     int(row["rank"]),
            })

    if not rows:
        return pd.DataFrame(
            columns=["metric", "metric_key", "section", "strategy", "value", "rank"]
        )
    return pd.DataFrame(rows).sort_values(
        ["metric", "rank"], kind="stable"
    ).reset_index(drop=True)


def _stage1_verdict(
    ranking_df: pd.DataFrame,
    retrieval_strategies: tuple[str, ...] = ("S2", "S3", "S4"),
) -> dict:
    """Stage 1 verdict: текстовый вывод о лидере и аутсайдерах.

    Логика:
    * **Лидер retrieval**: стратегия с наибольшим числом первых мест
      по retrieval-метрикам среди ``retrieval_strategies``.
    * **Лидер по FMR (среди retrieval)**: победитель по FMR среди
      ``retrieval_strategies``.
    * **Upper-bound reference**: победитель по FMR overall (как правило `S1`,
      полный контекст; не retrieval-стратегия).
    * **Самая слабая retrieval-стратегия**: последняя по FMR среди
      ``retrieval_strategies``.

    Args:
        ranking_df: output ``rank_strategies()``.
        retrieval_strategies: подмножество, которое считается «настоящими»
            retrieval-стратегиями (`S1` исключён как полный контекст,
            `B0` исключён как rule-based descriptive baseline).
    """
    if ranking_df.empty:
        return {
            "verdict": "no_data",
            "summary": "Нет данных для ранжирования",
            "retrieval_leader": None,
            "fmr_leader_retrieval": None,
            "fmr_upper_bound": None,
            "weakest_retrieval": None,
        }

    retr_subset = ranking_df[
        (ranking_df["section"] == "retrieval")
        & (ranking_df["strategy"].isin(retrieval_strategies))
    ]
    first_places = retr_subset[retr_subset["rank"] == 1].groupby("strategy").size()
    retrieval_leader = (
        first_places.sort_values(ascending=False).index[0]
        if not first_places.empty else None
    )

    fmr_rows = ranking_df[ranking_df["metric_key"] == "avg_fmr"]
    fmr_upper_bound = None
    fmr_leader_retrieval = None
    weakest_retrieval = None
    if not fmr_rows.empty:
        fmr_top = fmr_rows.sort_values("rank").iloc[0]
        fmr_upper_bound = (str(fmr_top["strategy"]), float(fmr_top["value"]))

        fmr_retr = fmr_rows[fmr_rows["strategy"].isin(retrieval_strategies)]
        if not fmr_retr.empty:
            fmr_retr_sorted = fmr_retr.sort_values("rank")
            top = fmr_retr_sorted.iloc[0]
            fmr_leader_retrieval = (str(top["strategy"]), float(top["value"]))
            bottom = fmr_retr_sorted.iloc[-1]
            weakest_retrieval = (str(bottom["strategy"]), float(bottom["value"]))

    summary_parts: list[str] = []
    if retrieval_leader:
        summary_parts.append(
            f"Лидер retrieval: **{retrieval_leader}** "
            f"({int(first_places[retrieval_leader])} первое место(а) среди "
            f"{int(first_places.sum())} retrieval-метрик)."
        )
    if fmr_leader_retrieval:
        summary_parts.append(
            f"Лидер по FMR среди retrieval: "
            f"**{fmr_leader_retrieval[0]} = {fmr_leader_retrieval[1]:.4f}**."
        )
    if fmr_upper_bound and fmr_upper_bound[0] not in retrieval_strategies:
        summary_parts.append(
            f"Upper-bound по FMR: **{fmr_upper_bound[0]} = {fmr_upper_bound[1]:.4f}** "
            f"(не retrieval-стратегия — даёт LLM весь контекст)."
        )
    if weakest_retrieval:
        summary_parts.append(
            f"Слабейшая retrieval-стратегия по FMR: "
            f"**{weakest_retrieval[0]} = {weakest_retrieval[1]:.4f}**."
        )

    return {
        "verdict": "ranked",
        "summary": " ".join(summary_parts) if summary_parts else "Ранжирование пустое.",
        "retrieval_leader": retrieval_leader,
        "fmr_leader_retrieval": fmr_leader_retrieval,
        "fmr_upper_bound": fmr_upper_bound,
        "weakest_retrieval": weakest_retrieval,
    }


def generate_stage1_memo(
    retrieval_table: pd.DataFrame,
    det_table: pd.DataFrame,
    run_paths: RunPaths,
    figures: list[Path] | None = None,
    run_meta: dict[str, dict] | None = None,
    mode: str = "plain",
    memo_filename: str = "stage1_decision_memo.md",
    memo_title: str = "Stage 1 Screening Decision Memo",
) -> Path:
    """Генерирует Stage 1 decision memo (markdown).

    В отличие от ``generate_stage2_memo``, фокус Stage 1 — сравнение
    **стратегий** (B0/S1/S2/S3/S4) внутри одного режима представления.
    Структура memo:

    1. Source of truth notice + meta
    2. Retrieval comparison (single-mode, без deltas)
    3. Quality comparison (single-mode, без deltas)
    4. Strategy ranking (по hit_rate / MRR / FMR / doctor_match)
    5. Stage 1 Verdict (текстовый: лидер, upper-bound, аутсайдеры)
    6. Limitations (Stage 1-специфичные ограничения)
    7. Figures
    8. Artifact Index

    Args:
        retrieval_table: output ``build_retrieval_table()``.
        det_table: output ``build_deterministic_table()``.
        run_paths: пути прогона.
        figures: список Path-ей к Stage 1-графикам.
        run_meta: ``{mode: meta_dict}`` для проверки n_errors.
        mode: режим представления, в котором сравниваются стратегии.
        memo_filename: имя файла в reports_dir.
        memo_title: H1-заголовок memo.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ranking_df = rank_strategies(retrieval_table, det_table, mode=mode)
    verdict = _stage1_verdict(ranking_df)

    lines: list[str] = [
        f"# {memo_title}",
        f"",
        f"> **Source of truth**: `outputs/runs/{run_paths.run_id}/`",
        f"",
        f"**Run ID**: `{run_paths.run_id}`  ",
        f"**Timestamp**: {ts}  ",
        f"**Mode (representation)**: `{mode}`",
        f"",
    ]

    if run_meta:
        meta = run_meta.get(mode, {})
        n_samples = meta.get("n_samples", "?")
        n_errors = meta.get("n_errors", 0)
        elapsed = meta.get("elapsed_s")
        lines.append(
            f"**Eval samples**: `{n_samples}`"
            + (f"  ·  **Errors**: `{n_errors}`" if n_errors else "")
            + (f"  ·  **Elapsed**: `{elapsed:.1f}s`"
               if isinstance(elapsed, (int, float)) else "")
        )
        lines.append("")

    lines.extend([
        f"---",
        f"",
        f"## 1. Retrieval Comparison (mode = `{mode}`)",
        f"",
        _df_to_markdown(retrieval_table),
        f"",
        f"## 2. Quality Comparison (mode = `{mode}`)",
        f"",
        _df_to_markdown(det_table),
        f"",
        f"## 3. Strategy Ranking",
        f"",
    ])

    if ranking_df.empty:
        lines.append("_Нет данных для ранжирования_")
    else:
        # Pivot: строки = стратегии, колонки = метрики (`value (rank)`).
        pivot_value = ranking_df.pivot(
            index="strategy", columns="metric", values="value",
        )
        pivot_rank = ranking_df.pivot(
            index="strategy", columns="metric", values="rank",
        )
        merged = pivot_value.copy()
        for col in merged.columns:
            merged[col] = [
                f"{v:.4f} (#{int(r)})" if pd.notna(v) else "—"
                for v, r in zip(pivot_value[col], pivot_rank[col])
            ]
        # Упорядочим стратегии: сначала retrieval (S2/S3/S4), затем S1, затем B0.
        order = [s for s in ["S2", "S3", "S4", "S4r", "S5", "S1", "B0"]
                 if s in merged.index]
        merged = merged.reindex(order)
        lines.append(merged.to_markdown())
        lines.append("")
        lines.append(
            "_Каждая ячейка: значение метрики и её ранг "
            "(`#1` — лучшая стратегия по этой метрике; для всех текущих "
            "Stage 1-метрик «больше — лучше»)._"
        )

    lines.extend([
        f"",
        f"## 4. Stage 1 Verdict",
        f"",
        verdict["summary"],
        f"",
    ])

    rl = verdict.get("retrieval_leader")
    fmr_r = verdict.get("fmr_leader_retrieval")
    fmr_u = verdict.get("fmr_upper_bound")
    weak = verdict.get("weakest_retrieval")
    structured: list[str] = []
    if rl:
        structured.append(f"- **Лидер retrieval (по числу первых мест)**: `{rl}`.")
    if fmr_r:
        structured.append(
            f"- **Лидер по FMR среди retrieval-стратегий**: "
            f"`{fmr_r[0]}` = `{fmr_r[1]:.4f}`."
        )
    if fmr_u:
        structured.append(
            f"- **Upper-bound по FMR (overall)**: `{fmr_u[0]}` = `{fmr_u[1]:.4f}`. "
            "Если это `S1` — это не retrieval-стратегия, а полный контекст."
        )
    if weak:
        structured.append(
            f"- **Слабейшая retrieval-стратегия по FMR**: "
            f"`{weak[0]}` = `{weak[1]:.4f}`."
        )
    if structured:
        lines.extend(structured)
        lines.append("")

    lines.extend([
        f"## 5. Limitations",
        f"",
        f"- Метрики получены на режиме `{mode}` (Stage 1 фиксирует представление чанков).",
        f"  Эффект различных представлений измеряется в Stage 2A.",
        f"- Decision Gate (`evaluate_gate`) не применяется в Stage 1: он "
        f"рассчитан на сравнение режимов представления (delta-колонок), "
        f"а в single-mode прогоне их нет.",
        f"- Per-sample `FMR` не сохраняется (`save_judge_scores` не вызывается "
        f"в pipeline) — paired significance по FMR недоступен.",
        f"- На текущей итерации не прогонялись `hard` / `blind` eval-сеты, "
        f"поэтому stress-валидация ранжирования стратегий вне `dev_v2` "
        f"остаётся открытым шагом.",
        f"",
    ])

    if figures:
        lines.extend([
            f"## 6. Figures",
            f"",
        ])
        for fig_path in figures:
            rel = _relative(fig_path, run_paths.run_dir)
            lines.append(f"![{fig_path.stem}]({rel})")
        lines.append("")

    artifacts = list_run_artifacts(run_paths)
    if artifacts:
        lines.extend([
            f"## 7. Artifact Index",
            f"",
            f"| Категория | Файл |",
            f"|-----------|------|",
        ])
        for cat, p in artifacts:
            rel = _relative(p, run_paths.run_dir)
            lines.append(f"| {cat} | `{rel}` |")
        lines.append("")

    memo_path = run_paths.report_md(memo_filename)
    memo_path.write_text("\n".join(lines), encoding="utf-8")
    return memo_path
