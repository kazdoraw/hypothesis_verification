"""Оценка качества извлечения анамнеза.

Для каждого кейса × схемы вызывает модель-оценщик, который сравнивает:
- reference_routing (из Case) vs doctor_routing (из SchemaRun)
- 6 числовых осей (0-10): specialist_score, service_score, examination_score,
  data_sufficiency, accuracy, dialogue_quality
- routing_match (bool): вычисляется программно (все routing scores >= ROUTING_MATCH_THRESHOLD)
"""

import json
import time
from pathlib import Path

from d2.client import OpenRouterClient
from d2.config import (
    DIALOGS_DIR,
    JUDGE_MODEL,
    MAX_TOKENS_JUDGE,
    REPORTS_DIR,
    ROUTING_MATCH_THRESHOLD,
    TEMPERATURE_JUDGE,
)
from d2.models import Case, CaseResult, SchemaRun
from d2.cases import CASES

JUDGE_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge.md"


def _load_judge_prompt() -> str:
    """Загрузить промпт судьи."""
    return JUDGE_PROMPT_PATH.read_text(encoding="utf-8")


def _build_judge_input(case: Case, run: SchemaRun) -> str:
    """Собрать входные данные для судьи."""
    dialog_text = "\n".join(
        f"{'Ассистент' if m.role == 'doctor' else 'Пациент'}: {m.text}"
        for m in run.dialog
    )
    extracted_text = "\n".join(
        f"- {k}: {v}" for k, v in run.extracted.items() if v
    )

    # Эталонная маршрутизация
    ref = case.reference_routing
    ref_text = (
        f"- Специалисты: {', '.join(ref.get('specialists', []))}\n"
        f"- Услуга: {ref.get('service_type', '')}\n"
        f"- Обследования: {ref.get('examination', '')}"
    )

    # Маршрутизация доктора
    doc = run.routing
    if doc:
        doc_text = (
            f"- Специалисты: {', '.join(doc.get('specialists', []))}\n"
            f"- Услуга: {doc.get('service_type', '')}\n"
            f"- Обследования: {doc.get('examination', '')}"
        )
    else:
        doc_text = "(не определена)"

    return (
        f"## Ситуация пациента (ground truth)\n\n{case.situation}\n\n"
        f"## Эталонная маршрутизация\n\n{ref_text}\n\n"
        f"## Маршрутизация доктора (на основе собранных данных)\n\n{doc_text}\n\n"
        f"## Диалог ({run.turns} ходов)\n\n{dialog_text}\n\n"
        f"## Извлечённые данные ({run.schema_name})\n\n{extracted_text}"
    )


ROUTING_AXES = ["specialist_score", "service_score", "examination_score"]
QUALITY_AXES = ["data_sufficiency", "accuracy", "dialogue_quality"]
SCORE_AXES = ROUTING_AXES + QUALITY_AXES


def judge_single(
    client: OpenRouterClient,
    case: Case,
    run: SchemaRun,
) -> dict:
    """Оценить один прогон (кейс × схема).

    Возвращает dict с 6 числовыми осями (0-10),
    routing_match (bool), missing_for_routing, errors, reasoning.
    """
    system_prompt = _load_judge_prompt()
    user_input = _build_judge_input(case, run)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    text, _usage = client.chat(
        messages=messages,
        model=JUDGE_MODEL,
        temperature=TEMPERATURE_JUDGE,
        max_tokens=MAX_TOKENS_JUDGE,
    )

    result = _parse_judge_response(text)
    if result.get("errors") == ["JSON parse error"]:
        # Retry: просим исправить формат
        messages.append({"role": "assistant", "content": text})
        messages.append({
            "role": "user",
            "content": (
                "Невалидный JSON. Верни ТОЛЬКО валидный JSON объект с полями: "
                "specialist_score, service_score, examination_score, "
                "data_sufficiency, accuracy, dialogue_quality, "
                "missing_for_routing, errors, reasoning."
            ),
        })
        text2, _ = client.chat(
            messages=messages,
            model=JUDGE_MODEL,
            temperature=0.1,
            max_tokens=MAX_TOKENS_JUDGE,
        )
        result = _parse_judge_response(text2)

    return result


def _extract_json(text: str) -> str:
    """Извлечь JSON из текста с fallback стратегиями."""
    import re

    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip()

    # Ищем самый внешний JSON объект
    start = cleaned.find("{")
    if start == -1:
        return cleaned
    depth, end = 0, start
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    result = cleaned[start:end]
    if depth > 0:
        result += "}" * depth
    return result


def _parse_judge_response(text: str) -> dict:
    """Парсинг JSON ответа с валидацией новой структуры."""
    json_str = _extract_json(text)
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return _empty_judge_result("JSON parse error")

    # Валидация числовых полей (6 осей)
    for key in SCORE_AXES:
        if key not in data:
            return _empty_judge_result(f"Missing key: {key}")
        try:
            data[key] = max(0, min(10, int(data[key])))
        except (ValueError, TypeError):
            data[key] = 0

    # routing_match вычисляется программно: все routing scores >= порога
    data["routing_match"] = all(
        data[k] >= ROUTING_MATCH_THRESHOLD for k in ROUTING_AXES
    )

    # Списки и строки
    data.setdefault("missing_for_routing", [])
    data.setdefault("errors", [])
    data.setdefault("reasoning", "")
    return data


def _empty_judge_result(reason: str) -> dict:
    """Пустой результат при ошибке парсинга."""
    result: dict = {k: 0 for k in SCORE_AXES}
    result.update({
        "routing_match": False,
        "missing_for_routing": [],
        "errors": [reason],
        "reasoning": f"Ошибка оценки: {reason}",
    })
    return result


def judge_all_cases(client: OpenRouterClient | None = None) -> dict:
    """Оценить все кейсы из results/dialogs/.

    Возвращает {case_id: {schema: judge_result}}.
    """
    if client is None:
        client = OpenRouterClient()

    cases_map = {c.case_id: c for c in CASES}
    all_scores: dict[int, dict[str, dict]] = {}

    for path in sorted(DIALOGS_DIR.glob("case_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        result = CaseResult.model_validate(data)
        case = cases_map.get(result.case_id)
        if not case:
            continue

        print(f"  Judging case {result.case_id}: {result.case_type}...", flush=True)
        case_scores: dict[str, dict] = {}

        for schema_name, run in result.runs.items():
            try:
                score = judge_single(client, case, run)
                rm = "✓" if score["routing_match"] else "✗"
                print(
                    f"    {schema_name}: route={rm} "
                    f"spec={score['specialist_score']} "
                    f"svc={score['service_score']} "
                    f"exam={score['examination_score']} "
                    f"suf={score['data_sufficiency']} "
                    f"acc={score['accuracy']} "
                    f"dlg={score['dialogue_quality']}",
                    flush=True,
                )
            except Exception as e:
                print(f"    {schema_name}: ERROR — {e}", flush=True)
                score = _empty_judge_result(str(e))
            case_scores[schema_name] = score
            time.sleep(0.5)

        all_scores[result.case_id] = case_scores

    return all_scores


def save_judge_scores(scores: dict, path: Path | None = None) -> Path:
    """Сохранить оценки судьи в JSON."""
    if path is None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORTS_DIR / "d2_judge_scores.json"
    path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_judge_scores(path: Path | None = None) -> dict | None:
    """Загрузить ранее сохранённые оценки судьи."""
    if path is None:
        path = REPORTS_DIR / "d2_judge_scores.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # JSON ключи — строки, конвертируем обратно в int
    return {int(k): v for k, v in data.items()}
