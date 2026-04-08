"""Оркестрация одного диалога doctor↔patient с инкрементальным extraction.

Один прогон = один кейс × одна схема.
Полный кейс = 3 прогона (S1, S2, S3).
"""

import json
import time
from pathlib import Path

from d2.client import OpenRouterClient
from d2.config import DIALOGS_DIR, MAX_TURNS
from d2.doctor import doctor_process, infer_routing
from d2.models import Case, CaseResult, Message, SchemaRun
from d2.patient import patient_speak
from d2.schemas import SCHEMAS


def _merge_field(existing: str, new: str) -> str:
    """Дополнить поле новой информацией, не дублируя.

    Если existing пустой — просто возвращаем new.
    Если значения — dict-строки (S3 fields) — мержим как dict.
    Если new — уточнение/повтор existing — заменяем (новое точнее).
    Если принципиально новая информация — дополняем через '; '.
    """
    if not existing:
        return new.strip()

    # Если оба значения — dict-строки (S3 fields) — мержим как dict
    merged_dict = _try_merge_dicts(existing, new)
    if merged_dict is not None:
        return merged_dict

    # Если новое значение уже содержится в старом — не дублируем
    new_lower = new.strip().lower()
    existing_lower = existing.lower()
    if new_lower in existing_lower:
        return existing

    # Если старое содержится в новом — новое полнее, заменяем
    if existing_lower in new_lower:
        return new.strip()

    # Новая информация — дополняем
    return f"{existing}; {new.strip()}"


def _try_merge_dicts(a: str, b: str) -> str | None:
    """Попытка мержа двух dict-строк (S3 fields). Возвращает None если не dict."""
    import ast
    try:
        dict_a = ast.literal_eval(a) if a.strip().startswith("{") else None
        dict_b = ast.literal_eval(b) if b.strip().startswith("{") else None
    except (ValueError, SyntaxError):
        return None

    if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
        return None

    # Мержим: новые ключи добавляем, существующие дополняем
    merged = dict(dict_a)
    for k, v in dict_b.items():
        if k in merged:
            old = str(merged[k])
            new_v = str(v)
            if new_v.lower() not in old.lower():
                merged[k] = f"{old}; {new_v}" if old else new_v
        else:
            merged[k] = str(v)
    return str(merged)


def run_case_schema(
    client: OpenRouterClient,
    case: Case,
    schema_name: str,
    verbose: bool = False,
) -> SchemaRun:
    """Один прогон: кейс × схема → диалог + извлечённые данные."""
    t0 = time.time()
    dialog: list[Message] = []
    accumulated: dict[str, str] = {}
    tokens_doctor = 0
    tokens_patient = 0

    # 1. Patient: первое сообщение (free-form жалоба)
    text, tok = patient_speak(client, case, dialog)
    tokens_patient += tok
    dialog.append(Message(role="patient", text=text))
    print(f"    👤 {text}", flush=True)

    # 2. Turn loop: doctor extract+ask → patient respond
    for turn in range(MAX_TURNS):
        # Doctor: structured output
        doctor_turn, tok = doctor_process(client, dialog, accumulated, schema_name)
        tokens_doctor += tok

        # Обновляем накопленные данные: дополняем, не перезаписываем
        for k, v in doctor_turn.extracted.items():
            if not v or not v.strip():
                continue
            accumulated[k] = _merge_field(accumulated.get(k, ""), v)

        # Добавляем вопрос доктора в диалог (текст для пациента)
        dialog.append(Message(role="doctor", text=doctor_turn.next_question))
        print(f"    🤖 {doctor_turn.next_question}", flush=True)
        if verbose:
            new_str = ", ".join(f"{k}={v}" for k, v in doctor_turn.extracted.items()) if doctor_turn.extracted else "-"
            print(f"       [new: {new_str}]", flush=True)
            print(f"       [total ({len(accumulated)} fields):]", flush=True)
            for ak, av in accumulated.items():
                print(f"         {ak}: {av}", flush=True)

        # Проверяем завершение
        if doctor_turn.is_complete:
            print(f"    ✅ Сбор завершён ({turn + 1} ходов)", flush=True)
            break

        # Patient: ответ
        text, tok = patient_speak(client, case, dialog)
        tokens_patient += tok
        dialog.append(Message(role="patient", text=text))
        print(f"    👤 {text}", flush=True)

    duration = time.time() - t0
    turns = (len(dialog) + 1) // 2  # пары сообщений

    # Пост-диалоговый вывод маршрутизации
    routing, routing_tokens = infer_routing(client, accumulated)
    tokens_doctor += routing_tokens
    print(f"    🔀 Routing: {routing.get('specialists', [])}", flush=True)

    return SchemaRun(
        schema_name=schema_name,
        dialog=dialog,
        extracted=accumulated,
        routing=routing,
        turns=turns,
        tokens_doctor=tokens_doctor,
        tokens_patient=tokens_patient,
        duration_s=round(duration, 1),
    )


def run_case(
    client: OpenRouterClient,
    case: Case,
    verbose: bool = False,
) -> CaseResult:
    """Полный прогон кейса: 3 диалога (S1, S2, S3)."""
    runs: dict[str, SchemaRun] = {}

    for schema_name in SCHEMAS:
        print(f"\n  --- Schema {schema_name} ---", flush=True)
        runs[schema_name] = run_case_schema(client, case, schema_name, verbose)

    return CaseResult(
        case_id=case.case_id,
        case_type=case.case_type,
        patient_prompt_summary=case.summary,
        runs=runs,
    )


def save_case_result(result: CaseResult) -> Path:
    """Сохранить результат кейса в JSON."""
    DIALOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = DIALOGS_DIR / f"case_{result.case_id:02d}.json"
    data = result.model_dump()
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
