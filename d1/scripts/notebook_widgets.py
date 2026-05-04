"""Widget UI для notebook — изолирован от source cells (Task 1 плана).

`ipywidgets` импортируется внутри функций (не на module-level), чтобы импорт
этого модуля не падал в headless-окружениях без ipywidgets.

API:
- `show_manual_query_sandbox()` — интерактивная форма ручного inference;
- `show_complexity_summary(eval_set)` — агрегация complexity tags из
  сохранённого decision-trace CSV.
"""

from __future__ import annotations

import pandas as pd
from IPython.display import Markdown, display

from d1.config import RESULTS_DIR
from d1.scripts.interactive_inference import complexity_summary, infer_text


# ---------------------------------------------------------------------------
# Manual sandbox — widgets + fallback
# ---------------------------------------------------------------------------

def show_manual_query_sandbox(default_text: str = "болит зуб сколько стоит") -> None:
    """Интерактивная форма для ручного inference без LLM-вызовов.

    При отсутствии ipywidgets — fallback-сообщение с инструкцией прямого вызова
    `infer_text(...)`.
    """
    try:
        import ipywidgets as widgets
    except ImportError:
        display(Markdown(
            "`ipywidgets` не установлен в текущем kernel. "
            "Используйте прямой вызов: "
            "`from d1.scripts.interactive_inference import infer_text; "
            "infer_text('болит зуб сколько стоит', mode='all')`."
        ))
        return

    text_box = widgets.Textarea(
        value=default_text,
        placeholder="Введите запрос пациента",
        description="Запрос",
        layout=widgets.Layout(width="100%", height="90px"),
    )
    mode_box = widgets.Dropdown(
        options=["all", "closed_set", "selective", "hybrid"],
        value="all",
        description="Mode",
    )
    gold_box = widgets.Dropdown(
        options=["", "anamnesis", "faq", "booking", "unsupported"],
        value="",
        description="Gold",
    )
    run_button = widgets.Button(description="Run", button_style="primary")
    output = widgets.Output()

    def _on_run(_: object) -> None:
        with output:
            output.clear_output()
            result = infer_text(
                text_box.value.strip(),
                mode=mode_box.value,
                gold_label=gold_box.value or None,
            )
            _display_manual_result(result)

    run_button.on_click(_on_run)
    display(widgets.VBox([
        text_box,
        widgets.HBox([mode_box, gold_box, run_button]),
        output,
    ]))


# ---------------------------------------------------------------------------
# Complexity summary из сохранённого trace CSV (diagnostic)
# ---------------------------------------------------------------------------

def show_complexity_summary(eval_set: str = "hard_test") -> None:
    """Агрегированные complexity tags по сохранённому decision-trace CSV.

    Читает `hybrid_decisions_<eval_set>.csv` (Task 5 roadmap) через существующий
    `complexity_summary()`. При отсутствии файла — graceful skip.
    """
    trace_path = RESULTS_DIR / f"hybrid_decisions_{eval_set}.csv"
    if not trace_path.exists():
        display(Markdown(
            f"`{trace_path.name}` не найден. Запустите `evaluate_hybrid` "
            "для генерации decision-trace."
        ))
        return
    display(Markdown(f"### Complexity summary — {eval_set}"))
    display(complexity_summary(trace_path))


# ---------------------------------------------------------------------------
# Private: форматирование результата ручного inference
# ---------------------------------------------------------------------------

def _display_manual_result(result: object) -> None:
    """Вывести ManualInferenceResult в notebook-friendly форме."""
    # Локальный импорт типа — чтобы избежать circular при загрузке.
    from d1.scripts.interactive_inference import ManualInferenceResult

    assert isinstance(result, ManualInferenceResult)

    display(Markdown(f"### Query\n`{result.text}`"))

    if result.closed_set:
        closed_rows = []
        for baseline, pred in result.closed_set.items():
            top2 = ", ".join(
                f"{x['label']}={x['probability']:.4f}"
                for x in pred.get("top2", [])
            )
            closed_rows.append({
                "baseline": baseline,
                "label": pred["label"],
                "confidence": pred["confidence"],
                "top2": top2,
            })
        display(Markdown("#### Closed-set predictions"))
        display(pd.DataFrame(closed_rows))

    policy_rows = []
    for name, decision in [
        ("SelectiveRouter", result.selective),
        ("B4 Hybrid", result.hybrid),
    ]:
        if decision is None:
            continue
        policy_rows.append({"router": name, **decision})
    if policy_rows:
        display(Markdown("#### Selective / Hybrid decisions"))
        display(pd.DataFrame(policy_rows))

    if result.rule_trace is not None:
        display(Markdown("#### Rule trace"))
        display(pd.DataFrame([result.rule_trace]))

    if result.correctness is not None:
        display(Markdown("#### Correctness"))
        display(
            pd.DataFrame([result.correctness["closed_set"]])
            .T.rename(columns={0: "correct"})
        )
        display(pd.DataFrame([{
            "gold_label": result.correctness["gold_label"],
            "selective_correct_if_accepted": result.correctness["selective"],
            "hybrid_correct_if_accepted": result.correctness["hybrid"],
        }]))


__all__ = [
    "show_complexity_summary",
    "show_manual_query_sandbox",
]
