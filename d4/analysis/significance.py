"""Статистическая значимость per-sample различий между representation modes.

Без LLM-вызовов: использует уже сохранённые артефакты прогона
(``*.report.json`` и ``*.jsonl``) и эталон из ``eval_set_*.yaml``.

Чего не считаем здесь:
    * per-sample FMR — judge-оценки (см. ``d4/evaluation/llm_judge.py``)
      сейчас не сохраняются на сэмпл, только агрегаты в ``avg_fmr``.
      Чтобы добавить FMR в этот модуль, нужно дописать вызов
      ``save_judge_scores`` в runner и сделать перепрогон.

Что считаем:
    * MRR / hit_rate — из ``report.json["rank_analysis"]`` (per-sample RR).
    * answerability_correct — из jsonl (``answer.answerable``) vs
      ``answerable`` в eval_set.
    * doctor_match — из jsonl (``answer.doctor``) vs ``expected_doctor``
      в eval_set, нормализация — lower+strip.

Все статистики работают paired по ``sample_id``: только сэмплы, у которых
есть значения во **всех** сравниваемых modes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from d4.analysis.artifacts import make_run_paths
from d4.analysis.loaders import load_report

_DEFAULT_BOOTSTRAP_ITER = 2000
_DEFAULT_ALPHA = 0.05
_DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Bootstrap / paired tests (низкоуровневые)
# ---------------------------------------------------------------------------

def bootstrap_ci_mean(
    values: Iterable[float],
    n_iter: int = _DEFAULT_BOOTSTRAP_ITER,
    alpha: float = _DEFAULT_ALPHA,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI для среднего.

    Returns:
        (mean, ci_lo, ci_hi). Если values пустой — все три np.nan.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_iter, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def bootstrap_ci_paired_diff(
    a: Iterable[float],
    b: Iterable[float],
    n_iter: int = _DEFAULT_BOOTSTRAP_ITER,
    alpha: float = _DEFAULT_ALPHA,
    seed: int = _DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI для среднего парных разностей (a - b).

    Длины должны совпадать; NaN-пары отбрасываются.
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    mask = ~(np.isnan(a_arr) | np.isnan(b_arr))
    diff = a_arr[mask] - b_arr[mask]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_iter, diff.size))
    means = diff[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(diff.mean()), lo, hi


def wilcoxon_paired(a: Iterable[float], b: Iterable[float]) -> tuple[float, float]:
    """Двусторонний Wilcoxon signed-rank test для парных непрерывных метрик.

    Returns:
        (statistic, p_value). Если все разности нулевые — (nan, 1.0).
    """
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    mask = ~(np.isnan(a_arr) | np.isnan(b_arr))
    diff = a_arr[mask] - b_arr[mask]
    if diff.size == 0 or np.allclose(diff, 0):
        return float("nan"), 1.0
    res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def mcnemar_paired(a: Iterable[bool], b: Iterable[bool]) -> tuple[int, int, float]:
    """McNemar test (exact binomial) для парных бинарных предсказаний.

    Args:
        a: TRUE если модель A была корректна для сэмпла.
        b: TRUE если модель B была корректна для того же сэмпла.

    Returns:
        (b_only, c_only, p_value), где
            b_only = #{a=True, b=False}  — выигрыш A,
            c_only = #{a=False, b=True}  — выигрыш B,
        p_value — двусторонний exact binomial test на симметрию.
    """
    a_arr = np.asarray(list(a), dtype=bool)
    b_arr = np.asarray(list(b), dtype=bool)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"shape mismatch: {a_arr.shape} vs {b_arr.shape}")
    b_only = int(np.sum(a_arr & ~b_arr))
    c_only = int(np.sum(~a_arr & b_arr))
    n = b_only + c_only
    if n == 0:
        return b_only, c_only, 1.0
    res = stats.binomtest(b_only, n=n, p=0.5, alternative="two-sided")
    return b_only, c_only, float(res.pvalue)


# ---------------------------------------------------------------------------
# Извлечение per-sample значений из артефактов
# ---------------------------------------------------------------------------

def extract_per_sample_rr(
    run_id: str,
    mode: str,
    focus_strategy: str,
    run_name: str,
) -> pd.Series:
    """RR для каждого sample_id для одного (mode, strategy) из rank_analysis.

    Returns:
        Series indexed by sample_id с float значениями (0.0 = miss).
        Пустая Series если нет данных.
    """
    report = load_report(run_id, mode, run_name=run_name)
    rank = report.get("rank_analysis", {})
    if not rank:
        return pd.Series(dtype=float, name=f"{mode}_rr")
    rows = {sid: strat_rr.get(focus_strategy) for sid, strat_rr in rank.items()}
    s = pd.Series(rows, dtype=float, name=f"{mode}_rr")
    return s.dropna().sort_index()


def extract_per_sample_predictions(
    run_id: str,
    mode: str,
    focus_strategy: str,
    run_name: str,
) -> pd.DataFrame:
    """Достаёт answerable / doctor предсказания из jsonl для одного mode.

    Returns:
        DataFrame indexed by sample_id, columns:
            answerable_pred (bool), doctor_pred (str|None).
        Только записи с strategy_id == focus_strategy.
    """
    rp = make_run_paths(run_id)
    jsonl_path = rp.run_dir / f"{run_name}_{mode}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl не найден: {jsonl_path}")
    rows: list[dict] = []
    with jsonl_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("strategy_id") != focus_strategy:
                continue
            ans = rec.get("answer") or {}
            doc = ans.get("doctor")
            rows.append({
                "sample_id": rec.get("sample_id"),
                "answerable_pred": bool(ans.get("answerable", False)),
                "doctor_pred": (doc or "").strip().lower() or None,
            })
    if not rows:
        return pd.DataFrame(columns=["answerable_pred", "doctor_pred"])
    return pd.DataFrame(rows).set_index("sample_id").sort_index()


def load_eval_gold(eval_yaml_path: Path | str) -> pd.DataFrame:
    """Достаёт answerable / expected_doctor из eval_set yaml.

    Returns:
        DataFrame indexed by sample_id, columns:
            answerable_gold (bool), doctor_gold (str|None).
    """
    path = Path(eval_yaml_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Ожидался список в {path}, получено {type(data).__name__}")
    rows = []
    for item in data:
        doc = item.get("expected_doctor")
        rows.append({
            "sample_id": item.get("sample_id"),
            "answerable_gold": bool(item.get("answerable", False)),
            "doctor_gold": (doc or "").strip().lower() or None,
        })
    return pd.DataFrame(rows).set_index("sample_id").sort_index()


# ---------------------------------------------------------------------------
# Высокоуровневая сводная таблица значимости
# ---------------------------------------------------------------------------

@dataclass
class SignificanceRow:
    metric: str
    mode: str
    mean: float
    ci_lo: float
    ci_hi: float
    diff_vs_baseline: float
    diff_ci_lo: float
    diff_ci_hi: float
    p_value: float
    test: str
    n: int


def _row_to_dict(r: SignificanceRow) -> dict:
    return {
        "metric": r.metric,
        "mode": r.mode,
        "mean": round(r.mean, 4) if not np.isnan(r.mean) else float("nan"),
        "ci_lo": round(r.ci_lo, 4) if not np.isnan(r.ci_lo) else float("nan"),
        "ci_hi": round(r.ci_hi, 4) if not np.isnan(r.ci_hi) else float("nan"),
        "diff_vs_baseline": round(r.diff_vs_baseline, 4) if not np.isnan(r.diff_vs_baseline) else float("nan"),
        "diff_ci_lo": round(r.diff_ci_lo, 4) if not np.isnan(r.diff_ci_lo) else float("nan"),
        "diff_ci_hi": round(r.diff_ci_hi, 4) if not np.isnan(r.diff_ci_hi) else float("nan"),
        "p_value": round(r.p_value, 4) if not np.isnan(r.p_value) else float("nan"),
        "test": r.test,
        "n": r.n,
    }


def _significant(diff_lo: float, diff_hi: float) -> bool:
    """CI среднего разностей не пересекает 0."""
    if np.isnan(diff_lo) or np.isnan(diff_hi):
        return False
    return diff_lo > 0 or diff_hi < 0


def compute_significance_table(
    run_id: str,
    modes: list[str],
    eval_yaml_path: Path | str,
    focus_strategy: str = "S3",
    baseline: str = "plain",
    run_name: str = "pilot_dev_30",
    n_iter: int = _DEFAULT_BOOTSTRAP_ITER,
    seed: int = _DEFAULT_SEED,
) -> pd.DataFrame:
    """Сводка bootstrap-CI и paired-тестов по доступным per-sample метрикам.

    Метрики:
        * mrr        — Wilcoxon на reciprocal rank, bootstrap CI парных разностей.
        * hit_rate   — McNemar на (RR > 0).
        * answerable — McNemar на (pred == gold).
        * doctor_match — McNemar на (pred == gold & gold is not None).

    Returns:
        DataFrame со строками для каждой пары (metric, mode), включая baseline
        (для baseline diff_vs_baseline = 0, p_value = nan).
    """
    if baseline not in modes:
        raise ValueError(f"baseline={baseline!r} отсутствует в modes={modes!r}")

    rr: dict[str, pd.Series] = {
        m: extract_per_sample_rr(run_id, m, focus_strategy, run_name) for m in modes
    }
    preds: dict[str, pd.DataFrame] = {
        m: extract_per_sample_predictions(run_id, m, focus_strategy, run_name) for m in modes
    }
    gold = load_eval_gold(eval_yaml_path)

    rows: list[dict] = []

    common_rr = sorted(set.intersection(*(set(s.index) for s in rr.values())))
    rr_aligned = {m: rr[m].reindex(common_rr) for m in modes}
    base_rr = rr_aligned[baseline].to_numpy()
    base_hit = (base_rr > 0).astype(bool)

    for m in modes:
        vals = rr_aligned[m].to_numpy()
        mean, lo, hi = bootstrap_ci_mean(vals, n_iter=n_iter, seed=seed)
        if m == baseline:
            rows.append(_row_to_dict(SignificanceRow(
                "mrr", m, mean, lo, hi, 0.0, 0.0, 0.0, float("nan"),
                "bootstrap", len(common_rr),
            )))
        else:
            d, dlo, dhi = bootstrap_ci_paired_diff(vals, base_rr, n_iter=n_iter, seed=seed)
            _, p = wilcoxon_paired(vals, base_rr)
            rows.append(_row_to_dict(SignificanceRow(
                "mrr", m, mean, lo, hi, d, dlo, dhi, p, "wilcoxon", len(common_rr),
            )))

        hits = (vals > 0).astype(bool)
        hit_mean, hlo, hhi = bootstrap_ci_mean(hits.astype(float), n_iter=n_iter, seed=seed)
        if m == baseline:
            rows.append(_row_to_dict(SignificanceRow(
                "hit_rate", m, hit_mean, hlo, hhi, 0.0, 0.0, 0.0, float("nan"),
                "bootstrap", len(common_rr),
            )))
        else:
            d, dlo, dhi = bootstrap_ci_paired_diff(
                hits.astype(float), base_hit.astype(float), n_iter=n_iter, seed=seed,
            )
            _, _, p = mcnemar_paired(hits, base_hit)
            rows.append(_row_to_dict(SignificanceRow(
                "hit_rate", m, hit_mean, hlo, hhi, d, dlo, dhi, p, "mcnemar", len(common_rr),
            )))

    common_pred = sorted(
        set(gold.index).intersection(*(set(preds[m].index) for m in modes))
    )

    base_ans_correct = (
        preds[baseline].loc[common_pred, "answerable_pred"].to_numpy()
        == gold.loc[common_pred, "answerable_gold"].to_numpy()
    ).astype(bool)
    for m in modes:
        ans_correct = (
            preds[m].loc[common_pred, "answerable_pred"].to_numpy()
            == gold.loc[common_pred, "answerable_gold"].to_numpy()
        ).astype(bool)
        mean, lo, hi = bootstrap_ci_mean(ans_correct.astype(float), n_iter=n_iter, seed=seed)
        if m == baseline:
            rows.append(_row_to_dict(SignificanceRow(
                "answerability_correct", m, mean, lo, hi, 0.0, 0.0, 0.0, float("nan"),
                "bootstrap", len(common_pred),
            )))
        else:
            d, dlo, dhi = bootstrap_ci_paired_diff(
                ans_correct.astype(float), base_ans_correct.astype(float),
                n_iter=n_iter, seed=seed,
            )
            _, _, p = mcnemar_paired(ans_correct, base_ans_correct)
            rows.append(_row_to_dict(SignificanceRow(
                "answerability_correct", m, mean, lo, hi, d, dlo, dhi, p, "mcnemar",
                len(common_pred),
            )))

    doc_mask = gold.loc[common_pred, "doctor_gold"].notna().to_numpy()
    doc_ids = [sid for sid, keep in zip(common_pred, doc_mask) if keep]
    if doc_ids:
        gold_doc = gold.loc[doc_ids, "doctor_gold"].to_numpy()
        base_doc = (
            preds[baseline].loc[doc_ids, "doctor_pred"].to_numpy() == gold_doc
        ).astype(bool)
        for m in modes:
            mode_doc = (
                preds[m].loc[doc_ids, "doctor_pred"].to_numpy() == gold_doc
            ).astype(bool)
            mean, lo, hi = bootstrap_ci_mean(mode_doc.astype(float), n_iter=n_iter, seed=seed)
            if m == baseline:
                rows.append(_row_to_dict(SignificanceRow(
                    "doctor_match", m, mean, lo, hi, 0.0, 0.0, 0.0, float("nan"),
                    "bootstrap", len(doc_ids),
                )))
            else:
                d, dlo, dhi = bootstrap_ci_paired_diff(
                    mode_doc.astype(float), base_doc.astype(float),
                    n_iter=n_iter, seed=seed,
                )
                _, _, p = mcnemar_paired(mode_doc, base_doc)
                rows.append(_row_to_dict(SignificanceRow(
                    "doctor_match", m, mean, lo, hi, d, dlo, dhi, p, "mcnemar",
                    len(doc_ids),
                )))

    df = pd.DataFrame(rows)
    df["significant"] = df.apply(
        lambda r: _significant(r["diff_ci_lo"], r["diff_ci_hi"]) if r["mode"] != baseline else False,
        axis=1,
    )
    return df


def render_significance_summary(df: pd.DataFrame, baseline: str = "plain") -> str:
    """Человекочитаемая сводка: по одной строке на (metric, mode != baseline)."""
    lines: list[str] = []
    for metric in df["metric"].unique():
        sub = df[df["metric"] == metric]
        base_row = sub[sub["mode"] == baseline]
        base_mean = float(base_row["mean"].iloc[0]) if not base_row.empty else float("nan")
        n = int(sub["n"].iloc[0])
        lines.append(f"\n[{metric}]  n={n}  baseline({baseline})={base_mean:.4f}")
        for _, r in sub[sub["mode"] != baseline].iterrows():
            sig = "★" if r["significant"] else " "
            lines.append(
                f"  {sig} {r['mode']:14s} "
                f"mean={r['mean']:.4f} CI[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]  "
                f"Δ={r['diff_vs_baseline']:+.4f} CI[{r['diff_ci_lo']:+.4f},{r['diff_ci_hi']:+.4f}]  "
                f"p={r['p_value']:.3f} ({r['test']})"
            )
    return "\n".join(lines)
