"""Генерация seed_audit.csv — результат аудита всех 415 seeds.

Статусы: keep / relabel / drop
FAQ-категории: supported_faq / subjective / policy_only / n/a

Запуск: cd study && python -m d1.scripts.seed_audit
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import yaml

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.config import DATA_DIR, SEEDS_FILE

_OUTPUT = DATA_DIR / "seed_audit.csv"

# Конкретные решения по seeds, выявленные при аудите.
# Формат: seed_id → (status, new_domain, new_subtype, faq_category, notes)
# Если не указан — keep / без изменений.

_RELABELS: dict[str, dict[str, str]] = {
    # seed_349 и seed_413 уже исправлены в YAML (Фаза 0.3),
    # здесь фиксируем факт для трассируемости.
    "seed_349": {
        "status": "relabel",
        "new_domain": "faq",
        "new_subtype": "clinic_info",
        "notes": "greeting + вопрос 'это стоматология?' → faq (greeting_with_intent_rule). Исправлено в YAML.",
    },
    "seed_413": {
        "status": "relabel",
        "new_domain": "anamnesis",
        "new_subtype": "complaint",
        "notes": "post-treatment complaint → anamnesis (mixed_intent_rule). Исправлено в YAML.",
    },
    # Новые relabel
    "seed_165": {
        "status": "relabel",
        "new_domain": "faq",
        "new_subtype": "doctor_info",
        "notes": "'Сколько детских стоматологов?' — вопрос о врачах, не о клинике. clinic_info → doctor_info.",
    },
    "seed_242": {
        "status": "relabel",
        "new_domain": "faq",
        "new_subtype": "clinic_info",
        "notes": "'Можно ли записаться онлайн?' — вопрос о наличии онлайн-записи (FAQ), не сама запись.",
    },
    "seed_249": {
        "status": "relabel",
        "new_domain": "faq",
        "new_subtype": "clinic_info",
        "notes": "'Записываете первичных?' — вопрос о политике клиники (FAQ), не запись.",
    },
}

# FAQ-категории для спорных faq-seeds
_FAQ_CATEGORIES: dict[str, str] = {
    # policy_only: зависит от бизнес-политики
    "seed_134": "policy_only",   # Принимаете по ДМС?
    "seed_158": "policy_only",   # Есть налоговый вычет за лечение?
    "seed_159": "policy_only",   # Даёте справку для налоговой?
    "seed_160": "policy_only",   # Есть программа лояльности?
    "seed_161": "policy_only",   # Акции есть?
    "seed_242": "policy_only",   # Можно ли записаться онлайн? (после relabel → faq)
    "seed_249": "policy_only",   # Записываете первичных? (после relabel → faq)
    # subjective: субъективные вопросы без объективного ответа
    "seed_167": "subjective",    # Кто ваш лучший ортодонт?
    "seed_175": "subjective",    # Кто самый опытный врач?
    "seed_188": "subjective",    # У кого больше всего отзывов?
    "seed_192": "subjective",    # Хочу к женщине-стоматологу
}

# Дополнительные заметки (для keep seeds)
_EXTRA_NOTES: dict[str, str] = {
    "seed_135": "Пограничный price/clinic_info (форма оплаты). Оставлено как price.",
    "seed_146": "Пограничный clinic_info/procedure_info (наличие седации). Оставлено как clinic_info.",
    "seed_147": "Пограничный clinic_info/procedure_info (наличие закиси). Оставлено как clinic_info.",
    "seed_163": "'Куда жаловаться?' — информационный запрос, не feedback. Оставлено как clinic_info.",
    "seed_305": "'Больше не хочу к вам' — cancel + feedback tone. Рекомендация: добавить feedback_flag.",
    "seed_310": "'Пойду в другую клинику' — cancel + feedback tone. Рекомендация: добавить feedback_flag.",
    "seed_311": "'Отменить и записать заново' — mixed cancel + booking. Первое действие — cancel.",
    "seed_350": "Пограничный greeting/faq ('Вы работаете?' = 'вы онлайн?' vs 'вы открыты?'). Оставлено как greeting_only.",
    "seed_409": "'Спасибо доктору, зуб больше не болит' — feedback + resolved symptom. Оставлено как feedback.",
}


def load_seeds() -> list[dict[str, Any]]:
    """Загрузка seeds из YAML."""
    with open(SEEDS_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("seeds", [])


def build_audit_row(seed: dict[str, Any]) -> dict[str, str]:
    """Формирование строки аудита для одного seed."""
    sid = seed["id"]
    relabel = _RELABELS.get(sid)

    if relabel:
        return {
            "seed_id": sid,
            "text": seed["text"],
            "route_domain": seed["route_domain"],
            "subtype": seed["subtype"],
            "status": relabel["status"],
            "new_domain": relabel.get("new_domain", ""),
            "new_subtype": relabel.get("new_subtype", ""),
            "faq_category": _FAQ_CATEGORIES.get(sid, "n/a"),
            "notes": relabel.get("notes", ""),
        }

    notes = _EXTRA_NOTES.get(sid, "")
    return {
        "seed_id": sid,
        "text": seed["text"],
        "route_domain": seed["route_domain"],
        "subtype": seed["subtype"],
        "status": "keep",
        "new_domain": "",
        "new_subtype": "",
        "faq_category": _FAQ_CATEGORIES.get(sid, "n/a"),
        "notes": notes,
    }


def main() -> None:
    """Генерация seed_audit.csv."""
    seeds = load_seeds()
    rows = [build_audit_row(s) for s in seeds]

    fieldnames = [
        "seed_id", "text", "route_domain", "subtype",
        "status", "new_domain", "new_subtype", "faq_category", "notes",
    ]

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Статистика
    total = len(rows)
    keeps = sum(1 for r in rows if r["status"] == "keep")
    relabels = sum(1 for r in rows if r["status"] == "relabel")
    drops = sum(1 for r in rows if r["status"] == "drop")
    policy = sum(1 for r in rows if r["faq_category"] == "policy_only")
    subjective = sum(1 for r in rows if r["faq_category"] == "subjective")
    with_notes = sum(1 for r in rows if r["notes"])

    print(f"Seed audit: {_OUTPUT}")
    print(f"  Всего: {total}")
    print(f"  keep: {keeps}, relabel: {relabels}, drop: {drops}")
    print(f"  faq: policy_only={policy}, subjective={subjective}")
    print(f"  С заметками: {with_notes}")


if __name__ == "__main__":
    main()
