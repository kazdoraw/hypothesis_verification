"""B0: Rules-only baseline (keyword/regex).

Нижняя граница для D1 v6 (§9.1 ТЗ).
Правила взяты из production router.py domain_markers + routing_rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class RulePrediction:
    """Результат rule-based классификации."""

    route_domain: str
    confidence: float
    matched_rule: str


# ---------------------------------------------------------------------------
# Паттерны по доменам (из production + ontology)
# ---------------------------------------------------------------------------

_ANAMNESIS_PATTERNS = [
    # Боль/симптомы
    r"бол[иеюья]т|боль\b|ноет|ноющ|пульсир|стрел[яь]|давит|жжёт|жжен",
    r"опух|отёк|отек|воспал|красн|кровоточ|кровь|гной",
    r"чувствительн|реагир.*на (холод|горяч|сладк)",
    r"шат[ае]|подвижн|выпа[лд]|откол|трещин|слом[аи]|разруш",
    r"коронк[аиу].*шат|пломб[аеу].*(выпал|слет)|брекет.*отклеи",
    # Интерес к лечению
    r"хоч[уе].*(?:отбелив|винир|имплант|брекет|чистк|протез|удален|лечен|седаци)",
    r"интересу[ею]т?.*(?:имплант|отбелив|брекет|протезир|пластик)",
    r"нужн[аыо]?\s*(?:брекет|чистк|имплант|протез|пломб|удален|коронк|винир)",
    # Пост-лечение
    r"после\s+(?:удален|лечен|имплант|операц|пломб|чистк)",
    # Ургентность
    r"срочн|экстренн|невыносим|не\s*могу\s*терпеть|очень\s*сильн.*бол",
    r"флюс|абсцесс|периостит",
]

_FAQ_PATTERNS = [
    r"скольк[оиу]\s*сто[иі]т|цен[аыу]|прайс|стоимост|рассрочк",
    r"где\s*(?:вы\s*)?наход|адрес|как\s*(?:доехать|добраться|проехать)",
    r"работа[ею]т[ие]?\s*(?:в\s*)?(?:суббот|воскресень|выходн|праздник)",
    r"график|режим\s*работ|часы\s*работ|расписание\s*клиник",
    r"парковк|оплат[аеу]|карт[аыу]|налич",
    r"(?:есть|работает)\s*(?:ли\s*)?(?:у\s*вас\s*)?(?:детск|ортодонт|хирург|пародонтолог|гнатолог|терапевт)",
    r"(?:кто|какой)\s*(?:ваш|у\s*вас)?\s*(?:лучш|опытн|хорош)",
    r"стаж|опыт\s*работ|квалификац|сертификат",
    r"больно\s*ли|это\s*больно|длительност|сколько\s*длится|противопоказан",
    r"как\s*проходит|подготовк[аеу]|что\s*(?:нужно|надо)\s*(?:перед|после)",
    r"через\s*сколько\s*(?:можно|нельзя)",
    r"(?:какие|что\s*за)\s*(?:услуги|процедуры|виды)",
    r"лицензи[яию]|сертификат",
]

_BOOKING_PATTERNS = [
    r"запиш[иу]те|запис[аь]ться|хоч[уе]\s*запис",
    r"(?:когда|есть)\s*(?:ли\s*)?(?:свободн|окошк|время|запись)",
    r"на\s*(?:приём|прием|консультаци)",
    r"перенес[тиу]|перезапиш|другое\s*время|другой\s*день",
    r"отмен[иу]те?\s*(?:запись|приём|прием)|не\s*приду|отказываюсь\s*от\s*запис",
    r"(?:на|ко?)\s*(?:завтра|послезавтра|понедельник|вторник|сред[уе]|четверг|пятниц|суббот|воскресень)",
]

_UNSUPPORTED_PATTERNS = [
    r"^(?:привет|здравствуй|добр(?:ый|ое|ого)|алло|хай|хеллоу)[!.\s]*$",
    r"^(?:пока|до свидания|всего доброго|спасибо.*до встречи|удачи)[!.\s]*$",
    r"^(?:ок|ладно|понятно|хорошо|ага|угу|да|нет|ну|\.{2,}|хм+|ааа+)[!.\s]*$",
    r"(?:погод|анекдот|президент|пицц|такси|кино|футбол|политик)",
    r"(?:переведи|напиши\s*стих|сочини|нарисуй|расскажи\s*(?:сказку|историю))",
]

_COMPILED: dict[str, list[re.Pattern]] = {}


def _get_compiled() -> dict[str, list[re.Pattern]]:
    """Lazy compile всех паттернов."""
    if not _COMPILED:
        for domain, patterns in [
            ("anamnesis", _ANAMNESIS_PATTERNS),
            ("faq", _FAQ_PATTERNS),
            ("booking", _BOOKING_PATTERNS),
            ("unsupported", _UNSUPPORTED_PATTERNS),
        ]:
            _COMPILED[domain] = [
                re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns
            ]
    return _COMPILED


# Приоритет доменов (из routing_rules.domain_priority)
_DOMAIN_PRIORITY = {"anamnesis": 0, "faq": 1, "booking": 2, "unsupported": 3}


class B0RulesClassifier:
    """Rule-based domain router.

    Алгоритм:
    1. Проверяем все паттерны, собираем matched домены
    2. При нескольких совпадениях — берём по приоритету (anamnesis > faq > booking)
    3. Без совпадений → unsupported
    """

    def predict_one(self, text: str) -> RulePrediction:
        """Классификация одного текста."""
        compiled = _get_compiled()
        matches: dict[str, int] = {}

        for domain, patterns in compiled.items():
            count = sum(1 for p in patterns if p.search(text))
            if count > 0:
                matches[domain] = count

        if not matches:
            return RulePrediction("unsupported", 0.3, "no_match")

        # Если unsupported matched И другие тоже — приоритет не-unsupported
        non_unsupported = {d: c for d, c in matches.items() if d != "unsupported"}
        if non_unsupported:
            matches = non_unsupported

        best_domain = min(matches, key=lambda d: _DOMAIN_PRIORITY.get(d, 99))
        hit_count = matches[best_domain]
        confidence = min(0.5 + 0.1 * hit_count, 0.95)

        return RulePrediction(best_domain, confidence, f"{hit_count}_patterns")

    def predict(self, texts: list[str]) -> list[str]:
        """Batch prediction → list[route_domain]."""
        return [self.predict_one(t).route_domain for t in texts]

    def predict_with_confidence(self, texts: list[str]) -> list[RulePrediction]:
        """Batch prediction с confidence."""
        return [self.predict_one(t) for t in texts]
