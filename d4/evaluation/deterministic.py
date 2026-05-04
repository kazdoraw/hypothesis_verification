"""Слой 1: Deterministic checks (автоматические, primary).

Проверки без LLM:
- Answerability classification: model.answerable == gold.answerable
- Entity correctness: exact match врач/специализация/филиал/услуга
- Unsupported claim rate: извлечь claims → проверить vs KB
"""

from __future__ import annotations

import re
from typing import Optional

from typing import Callable

from d4.models import (
    DeterministicScore,
    EvalSample,
    FAQAnswer,
    GoldFact,
    KBChunk,
    StrategyID,
    StrategyResult,
)

# NLI checker подключается опционально через evaluate_batch(nli_checker=...)
from d4.evaluation.nli_checker import NLIClaimChecker

# Prefix-stem для ФИО (совместимо с gold_map.py)
_SURNAME_STEM_LEN = 4


# ---------------------------------------------------------------------------
# Slot-aware нормализаторы (по fact_type из GoldFact)
# ---------------------------------------------------------------------------

def normalize_phone(text: str) -> str:
    """Телефон → только цифры, ведущая 8 → 7 (РФ). '+7(8422)58-58-58' → '78422585858'."""
    digits = re.sub(r"\D", "", text)
    if len(digits) == 11 and digits.startswith("8"):
        digits = "7" + digits[1:]
    return digits


def normalize_price(text: str) -> str:
    """Цена → только цифры. 'от 35 000 ₽' → '35000'."""
    cleaned = re.sub(r"[^\d]", "", text)
    return cleaned


def normalize_fio(text: str) -> str:
    """ФИО → lowercase фамилия (первое слово ≥3 символов). 'Батылин Д.В.' → 'батылин'."""
    words = re.findall(r"[а-яёa-z]{3,}", text.lower())
    return words[0] if words else text.lower().strip()


def normalize_address(text: str) -> str:
    """Адрес → lowercase, убираем 'ул.', 'д.', 'г.', лишние пробелы. 'ул. Рябикова, д. 56' → 'рябикова 56'."""
    t = text.lower()
    t = re.sub(r"\b(г|ул|д|пр|пер|корп|стр)\.\s*", "", t)
    t = re.sub(r"[,.]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_SCHEDULE_SYNONYMS: dict[str, list[str]] = {
    "будни": ["пн", "пт"],
    "будням": ["пн", "пт"],
    "будних": ["пн", "пт"],
    "выходные": ["сб", "вс"],
    "выходных": ["сб", "вс"],
    "выходным": ["сб", "вс"],
}


def normalize_schedule(text: str) -> str:
    """Расписание → набор ключевых токенов (дни, часы, слова-маркеры).

    Synonym expansion: 'будни' → 'пн пт', 'выходные' → 'сб вс'.
    'ежедневно с 08:00 до 20:00' → 'ежедневно 8 20'
    'Пн-Пт 9:00-20:00'           → 'пн пт 9 20'
    'по будням с 9 до 20'         → 'пн пт 9 20'
    """
    t = text.lower()
    hours = [h.lstrip("0") or "0" for h in re.findall(r"\b(\d{1,2})(?::00)?\b", t)]
    day_markers = re.findall(
        r"\b(пн|вт|ср|чт|пт|сб|вс|ежедневно|будни|будням|будних|выходные|выходных|выходным)\b",
        t,
    )
    expanded: list[str] = []
    for m in day_markers:
        if m in _SCHEDULE_SYNONYMS:
            expanded.extend(_SCHEDULE_SYNONYMS[m])
        else:
            expanded.append(m)
    tokens = list(dict.fromkeys(expanded)) + hours
    return " ".join(tokens)


def normalize_text(text: str) -> str:
    """Текст → lowercase, без пунктуации, без лишних пробелов."""
    t = text.lower()
    t = re.sub(r"[^\w\sа-яёa-z0-9]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_NORMALIZERS: dict[str, Callable[[str], str]] = {
    "phone": normalize_phone,
    "price": normalize_price,
    "fio": normalize_fio,
    "address": normalize_address,
    "schedule": normalize_schedule,
    "text": normalize_text,
}


_SYMMETRIC_TYPES = frozenset({"phone", "price", "schedule"})

_TOKEN_SET_TYPES = frozenset({"schedule"})


def _compute_fact_match_rate(answer: str, gold_facts: list[GoldFact]) -> float | None:
    """Slot-aware fact_match_rate: проверяет каждый GoldFact в ответе.

    Для phone/price/schedule — symmetric normalizer (один и тот же для canonical и ответа).
    Для schedule — token set containment (canonical tokens ⊆ answer tokens),
    потому что LLM свободно парафразирует маркеры дней.
    Для остальных — canonical через специфичный нормализатор,
    ответ через normalize_text (сохраняет весь текст для substring-поиска).

    Returns None если gold_facts пусты — метрика не определена для сэмпла.
    """
    if not gold_facts:
        return None
    matched = 0
    for fact in gold_facts:
        normalizer = _NORMALIZERS.get(fact.fact_type, normalize_text)
        canonical = normalizer(fact.canonical_value)
        if fact.fact_type in _SYMMETRIC_TYPES:
            answer_n = normalizer(answer)
        else:
            answer_n = normalize_text(answer)

        if not canonical:
            continue

        if fact.fact_type in _TOKEN_SET_TYPES:
            canonical_tokens = set(canonical.split())
            answer_tokens = set(answer_n.split())
            if canonical_tokens and canonical_tokens.issubset(answer_tokens):
                matched += 1
        elif canonical in answer_n:
            matched += 1
    return matched / len(gold_facts)


def _normalize(text: Optional[str]) -> str:
    """Нормализация строки для сравнения: lowercase, trim, убрать лишние пробелы."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_claims(answer_text: str) -> list[str]:
    """Извлечение проверяемых claims из ответа.

    Claims — конкретные факты: телефоны, адреса, имена, числа, URL.
    """
    claims: list[str] = []

    # Телефоны
    phones = re.findall(r"\+?\d[\d\s\-\(\)]{6,}", answer_text)
    claims.extend(phones)

    # Email
    emails = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", answer_text)
    claims.extend(emails)

    # URL
    urls = re.findall(r"https?://\S+", answer_text)
    claims.extend(urls)

    # ФИО (три слова с заглавной подряд)
    fio_pattern = re.findall(r"[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+", answer_text)
    claims.extend(fio_pattern)

    # Цены: "от 35 000 ₽", "590 ₽", "от 2 000 ₽"
    prices = re.findall(r"(?:от\s+)?[\d][\d\s]*\d\s*₽", answer_text)
    claims.extend(prices)

    # Числа с контекстом (стаж, часы, месяцы) — БЕЗ ₽, чтобы не дублировать
    numbers = re.findall(r"\d+\s*(?:лет|год|час|мес|руб|%)", answer_text, re.IGNORECASE)
    claims.extend(numbers)

    return claims


def _claim_in_kb(claim: str, kb_text: str) -> bool:
    """Проверка: есть ли claim в тексте KB.

    Поддерживает нормализацию: точное вхождение, телефоны,
    числа с единицами, ФИО (prefix-stem), URL.
    """
    claim_n = _normalize(claim)
    kb_n = _normalize(kb_text)

    if not claim_n:
        return True

    # 1. Точное вхождение (основной путь)
    if claim_n in kb_n:
        return True

    # 2. Телефоны — сравнение только цифр
    claim_digits = re.sub(r"\D", "", claim)
    if len(claim_digits) >= 7:
        kb_digits = re.sub(r"\D", "", kb_text)
        if claim_digits in kb_digits:
            return True

    # 3. Числа с единицами: "7 лет" → проверить "7" в KB
    num_match = re.match(r"(\d+)\s*(?:лет|год|мес|час)", claim_n)
    if num_match:
        num = num_match.group(1)
        if num in kb_n:
            return True

    # 4. ФИО — prefix-stem (4 символа, как gold_map.py)
    fio_words = re.findall(r"[а-яё]{4,}", claim_n)
    if len(fio_words) >= 1:
        stem = fio_words[0][:_SURNAME_STEM_LEN]
        if stem in kb_n:
            return True

    # 5. URL — убрать протокол и trailing slash
    if claim_n.startswith("http"):
        url_clean = re.sub(r"^https?://", "", claim_n).rstrip("/")
        kb_clean = re.sub(r"https?://", "", kb_n)
        if url_clean in kb_clean:
            return True

    # 6. Цены — нормализация пробелов: "35 000" → "35000"
    price_match = re.search(r"[\d\s]+₽", claim)
    if price_match:
        claim_digits = re.sub(r"\s", "", claim)
        kb_digits_normalized = re.sub(r"(\d)\s+(\d)", r"\1\2", kb_text)
        price_num = claim_digits.replace("от", "").replace("₽", "").strip()
        if price_num and price_num in kb_digits_normalized:
            return True

    return False


def evaluate_single(
    result: StrategyResult,
    sample: EvalSample,
    kb_text: str,
    nli_checker: NLIClaimChecker | None = None,
) -> DeterministicScore:
    """Deterministic evaluation для одного ответа.

    Args:
        result: результат стратегии
        sample: эталонный запрос с разметкой
        kb_text: полный текст KB (для проверки claims)
        nli_checker: LLM NLI checker (если None — fallback на строковый _claim_in_kb)

    Returns:
        DeterministicScore
    """
    answer = result.answer

    # Answerability
    answerability_correct = answer.answerable == sample.answerable

    # Entity matching (exact match после нормализации)
    # Для сэмплов без expected_doctor — не штрафуем
    doctor_match = (
        _normalize(answer.doctor) == _normalize(sample.expected_doctor)
        if sample.expected_doctor
        else True
    )

    specialization_match = (
        _normalize(answer.specialization) == _normalize(sample.expected_specialization)
        if sample.expected_specialization
        else True
    )

    branch_match = (
        _normalize(answer.branch) == _normalize(sample.expected_branch)
        if sample.expected_branch
        else True
    )

    service_match = (
        _normalize(answer.service) == _normalize(sample.expected_service)
        if sample.expected_service
        else True
    )

    # Unsupported claims: NLI (семантический) или fallback (строковый)
    claims = _extract_claims(answer.answer)
    if nli_checker is not None:
        unsupported = nli_checker.count_unsupported(claims, kb_text)
    else:
        unsupported = sum(1 for c in claims if not _claim_in_kb(c, kb_text))

    # Slot-aware fact_match_rate (PRIMARY metric)
    fact_match_rate = _compute_fact_match_rate(answer.answer, sample.gold_facts)

    return DeterministicScore(
        sample_id=result.sample_id,
        strategy_id=result.strategy_id,
        answerability_correct=answerability_correct,
        doctor_match=doctor_match,
        specialization_match=specialization_match,
        branch_match=branch_match,
        service_match=service_match,
        unsupported_claims=unsupported,
        total_claims=len(claims),
        fact_match_rate=fact_match_rate,
    )


def evaluate_batch(
    results: list[StrategyResult],
    samples: list[EvalSample],
    chunks: list[KBChunk],
    nli_checker: NLIClaimChecker | None = None,
) -> list[DeterministicScore]:
    """Batch evaluation для всех результатов.

    Args:
        results: все StrategyResult
        samples: eval set (для gold labels)
        chunks: KB chunks (для проверки claims)
        nli_checker: LLM NLI checker (если None — fallback на строковый _claim_in_kb)

    Returns:
        список DeterministicScore
    """
    kb_text = "\n".join(f"{c.title}\n{c.content}" for c in chunks)
    sample_map = {s.sample_id: s for s in samples}

    scores: list[DeterministicScore] = []
    for result in results:
        if result.error is not None:
            continue
        sample = sample_map.get(result.sample_id)
        if sample is None:
            continue
        score = evaluate_single(result, sample, kb_text, nli_checker)
        scores.append(score)

    return scores
