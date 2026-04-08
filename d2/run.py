"""CLI entry point для эксперимента D2.

Использование:
    python -m d2.run --cases all           # Все 10 кейсов
    python -m d2.run --cases 1 --verbose   # Один кейс с выводом диалога
    python -m d2.run --cases 1,3,5         # Выбранные кейсы
    python -m d2.run --report-only         # Только отчёт + графики
    python -m d2.run --routing-only        # Дозапуск routing inference к существующим диалогам
"""

import argparse
import sys
import time

from d2.cases import CASES
from d2.client import OpenRouterClient
from d2.config import DOCTOR_MODEL, PATIENT_MODEL
from d2.report import generate_report
from d2.session import run_case, save_case_result


def parse_args() -> argparse.Namespace:
    """Парсинг аргументов CLI."""
    parser = argparse.ArgumentParser(description="D2: LLM vs LLM — сбор анамнеза")
    parser.add_argument(
        "--cases", type=str, default="all",
        help="Какие кейсы запустить: 'all', номер (1), или список (1,3,5)",
    )
    parser.add_argument("--verbose", action="store_true", help="Подробный вывод диалогов")
    parser.add_argument("--report-only", action="store_true", help="Только отчёт + графики")
    parser.add_argument("--resume", action="store_true", help="Пропустить уже выполненные кейсы")
    parser.add_argument("--judge", action="store_true", help="Оценить качество извлечения")
    parser.add_argument("--routing-only", action="store_true", help="Дозапуск routing inference к существующим диалогам")
    return parser.parse_args()


def _try_generate_plots() -> None:
    """Попытка генерации графиков (требует matplotlib)."""
    try:
        from d2.viz import generate_all_plots
        print("\n\U0001f4ca Графики:", flush=True)
        generate_all_plots()
    except ImportError:
        print("\n\u26a0\ufe0f  matplotlib не установлен — графики пропущены. pip install matplotlib", flush=True)


def select_cases(cases_arg: str) -> list:
    """Выбрать кейсы по аргументу CLI."""
    if cases_arg == "all":
        return CASES

    try:
        ids = [int(x.strip()) for x in cases_arg.split(",")]
    except ValueError:
        print(f"Ошибка: неверный формат --cases '{cases_arg}'. Используйте: all, 1, или 1,3,5")
        sys.exit(1)

    selected = [c for c in CASES if c.case_id in ids]
    if not selected:
        print(f"Ошибка: кейсы {ids} не найдены. Доступные: {[c.case_id for c in CASES]}")
        sys.exit(1)

    return selected


def _run_judge() -> None:
    """Оценка качества извлечения."""
    from d2.judge import judge_all_cases, save_judge_scores
    from d2.config import JUDGE_MODEL
    print(f"D2: Оценка качества (модель: {JUDGE_MODEL})")
    print(f"  Оцениваем все кейсы \u00d7 3 схемы...\n")
    scores = judge_all_cases()
    path = save_judge_scores(scores)
    print(f"\n  \u2192 Оценки сохранены: {path}")
    # Перегенерируем отчёты с новыми оценками
    _try_generate_plots()
    print("\n📄 Отчёты:", flush=True)
    generate_report()
    print("\nГотово!")


def _run_routing_only() -> None:
    """Дозапуск routing inference к существующим диалогам."""
    import json as _json
    from d2.config import DIALOGS_DIR
    from d2.doctor import infer_routing
    from d2.models import CaseResult

    client = OpenRouterClient()
    paths = sorted(DIALOGS_DIR.glob("case_*.json"))
    if not paths:
        print("Нет результатов диалогов. Сначала запустите: python -m d2.run --cases all")
        return

    print(f"D2: Routing inference для {len(paths)} кейсов")
    total_calls = 0

    for path in paths:
        data = _json.loads(path.read_text(encoding="utf-8"))
        result = CaseResult.model_validate(data)
        print(f"\n  Case {result.case_id}: {result.case_type}")

        updated = False
        for schema_name, run in result.runs.items():
            if run.routing:
                print(f"    {schema_name}: уже есть → {run.routing.get('specialists', [])}")
                continue
            routing, tokens = infer_routing(client, run.extracted)
            run.routing = routing
            updated = True
            total_calls += 1
            print(f"    {schema_name}: 🔀 {routing.get('specialists', [])}")

        if updated:
            path.write_text(
                _json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"    → Сохранено: {path}")

    print(f"\nГотово! Routing inference: {total_calls} вызовов")
    print(f"  API calls: {client.total_calls}")
    print(f"  Tokens: {client.total_tokens_in + client.total_tokens_out:,}")


def _run_report_only() -> None:
    """Генерация отчётов и графиков без запуска диалогов."""
    print("D2: Генерация отчётов")
    _try_generate_plots()
    print("\n📄 Отчёты:", flush=True)
    generate_report()
    print("\nГотово!")


def main() -> None:
    """Основной цикл эксперимента."""
    args = parse_args()

    if args.judge:
        _run_judge()
        return

    if args.routing_only:
        _run_routing_only()
        return

    if args.report_only:
        _run_report_only()
        return

    cases = select_cases(args.cases)

    print(f"D2: LLM vs LLM — Эксперимент")
    print(f"  Doctor: {DOCTOR_MODEL}")
    print(f"  Patient: {PATIENT_MODEL}")
    print(f"  Кейсов: {len(cases)}, схем: 3 (S1, S2, S3)")
    print(f"  Итого прогонов: {len(cases) * 3}")
    print()

    client = OpenRouterClient()
    t0 = time.time()

    from d2.config import DIALOGS_DIR

    for case in cases:
        existing = DIALOGS_DIR / f"case_{case.case_id:02d}.json"
        if args.resume and existing.exists():
            print(f"\u23e9 Case {case.case_id}: уже есть, пропускаю", flush=True)
            continue

        print(f"{'='*60}", flush=True)
        print(f"Case {case.case_id}: {case.case_type} — {case.summary}", flush=True)
        print(f"{'='*60}", flush=True)

        result = run_case(client, case, verbose=True)
        path = save_case_result(result)

        # Краткая сводка по кейсу
        for schema_name, run in result.runs.items():
            fields = len([v for v in run.extracted.values() if v])
            print(f"  {schema_name}: {run.turns} turns, {fields} fields, "
                  f"{run.tokens_doctor + run.tokens_patient} tokens, {run.duration_s}s")

        print(f"  → Сохранено: {path}")
        print()

    total_time = time.time() - t0
    print(f"{'='*60}")
    print(f"Готово! {len(cases)} кейсов × 3 схемы = {len(cases) * 3} прогонов")
    print(f"  Время: {total_time:.0f}s")
    print(f"  API calls: {client.total_calls}")
    print(f"  Tokens: {client.total_tokens_in + client.total_tokens_out:,}")

    # Автоматическая генерация отчётов после прогона
    _try_generate_plots()
    print("\n📄 Отчёты:", flush=True)
    generate_report()


if __name__ == "__main__":
    main()
