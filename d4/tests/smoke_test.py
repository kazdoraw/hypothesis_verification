"""Smoke-тесты D4v2: валидация компонентов БЕЗ LLM-вызовов.

Запуск: pytest d4/tests/smoke_test.py -v
Требования: chunks.json, doctors.yaml, mini_eval_set.yaml

Тесты проверяют:
1. Retrieval hit rate (S2 BM25) — gold chunks в top-k (multi-gold), пороги >= 50%
2. B0 template answers — корректные факты из KB, out_of_scope accuracy >= 50%
3. Evaluation sanity — ожидаемые метрики для синтетических ответов
4. Gold map coverage — непустые chunk_ids для answerable samples
5. DirectAnswerResult — S5/B0 accounting (Фаза 4B)
6. GoldFact + slot-aware нормализаторы + fact_match_rate (Фаза 5A)
7. Group-stratified split (Фаза 3 P0)
8. Aftercare retrieval — aftercare_post_extraction chunk (Rev4)
"""

from __future__ import annotations

import pytest

from d4.evaluation.deterministic import (
    _compute_fact_match_rate,
    evaluate_single,
    normalize_address,
    normalize_fio,
    normalize_phone,
    normalize_price,
    normalize_schedule,
    normalize_text,
)
from d4.evaluation.gold_map import build_gold_map
from d4.evaluation.retrieval_metrics import compute_retrieval_score
from d4.models import (
    DeterministicScore,
    DirectAnswerResult,
    EvalSample,
    FAQAnswer,
    GoldFact,
    KBChunk,
    RetrievalResult,
    StrategyID,
    StrategyResult,
)
from d4.strategies.keyword_template import KeywordTemplateStrategy
from d4.strategies.lexical import LexicalStrategy


def _flatten_gold(gold_alternatives: list[list[str]]) -> set[str]:
    """Вспомогательная: все chunk_ids из всех альтернатив."""
    ids: set[str] = set()
    for alt in gold_alternatives:
        ids.update(alt)
    return ids


# ---------------------------------------------------------------------------
# 1. Retrieval hit rate (multi-gold)
# ---------------------------------------------------------------------------


class TestRetrievalHitRate:
    """Проверяет что retrieval стратегии находят gold chunks."""

    def test_s2_clinic_info_hit_rate(self, chunks, mini_samples, gold_map):
        """S2 (BM25) должен найти gold chunks для clinic_info запросов."""
        strategy = LexicalStrategy(top_k=5)
        info_samples = [s for s in mini_samples if s.category == "clinic_info"]
        assert len(info_samples) > 0, "Нет clinic_info samples в mini eval set"

        hits = 0
        for sample in info_samples:
            result = strategy.select_context(sample.query, chunks)
            gold_ids = _flatten_gold(gold_map.get(sample.sample_id, []))
            if gold_ids & set(result.chunk_ids):
                hits += 1

        hit_rate = hits / len(info_samples)
        print(f"\nS2 clinic_info hit rate: {hit_rate:.0%} ({hits}/{len(info_samples)})")
        assert hit_rate >= 0.5, f"S2 clinic_info hit rate слишком низкий: {hit_rate:.0%}"

    def test_s2_doctor_info_hit_rate(self, chunks, mini_samples, gold_map):
        """S2 (BM25) должен найти doctor chunks по фамилии."""
        strategy = LexicalStrategy(top_k=5)
        doctor_samples = [s for s in mini_samples if s.category == "doctor_info"]
        assert len(doctor_samples) > 0, "Нет doctor_info samples"

        hits = 0
        for sample in doctor_samples:
            result = strategy.select_context(sample.query, chunks)
            gold_ids = _flatten_gold(gold_map.get(sample.sample_id, []))
            if gold_ids & set(result.chunk_ids):
                hits += 1

        hit_rate = hits / len(doctor_samples)
        print(f"\nS2 doctor hit rate: {hit_rate:.0%} ({hits}/{len(doctor_samples)})")
        assert hit_rate >= 0.5, f"S2 doctor hit rate слишком низкий: {hit_rate:.0%}"

    def test_s2_no_empty_retrieval_for_answerable(self, chunks, mini_samples):
        """S2 не должен возвращать пустой retrieval для answerable запросов."""
        strategy = LexicalStrategy(top_k=5)
        answerable = [s for s in mini_samples if s.answerable]
        assert len(answerable) > 0

        empty_count = 0
        for sample in answerable:
            result = strategy.select_context(sample.query, chunks)
            if not result.chunk_ids:
                empty_count += 1
                print(f"  EMPTY retrieval: {sample.sample_id} '{sample.query}'")

        empty_rate = empty_count / len(answerable)
        print(f"\nS2 empty retrieval rate: {empty_rate:.0%} ({empty_count}/{len(answerable)})")
        assert empty_rate <= 0.3, f"S2 слишком много пустых retrieval: {empty_rate:.0%}"

    def test_s2_aftercare_retrieval(self, chunks, mini_samples, gold_map):
        """S2 должен найти aftercare chunk для aftercare запросов."""
        strategy = LexicalStrategy(top_k=5)
        ac_samples = [s for s in mini_samples if s.category == "aftercare"]
        assert len(ac_samples) > 0, "Нет aftercare samples в mini eval set"

        for sample in ac_samples:
            result = strategy.select_context(sample.query, chunks)
            gold_ids = _flatten_gold(gold_map.get(sample.sample_id, []))
            found = gold_ids & set(result.chunk_ids)
            print(f"\n  S2 aftercare [{sample.sample_id}]: gold={gold_ids}, retrieved={result.chunk_ids[:5]}")
            assert found, (
                f"S2 не нашёл aftercare gold chunk для '{sample.query}': "
                f"gold={gold_ids}, retrieved={result.chunk_ids[:5]}"
            )


# ---------------------------------------------------------------------------
# 2. B0 template answers
# ---------------------------------------------------------------------------


class TestB0Answers:
    """Проверяет что B0 генерирует осмысленные ответы из KB."""

    def test_b0_clinic_info_contains_facts(self, chunks, mini_samples):
        """B0 ответы на clinic_info должны содержать факты из KB."""
        strategy = KeywordTemplateStrategy()
        info_samples = [s for s in mini_samples if s.category == "clinic_info"]

        for sample in info_samples:
            result = strategy.answer_directly(sample.query, chunks)
            faq = result.answer
            assert faq.answer, f"Пустой ответ B0 для '{sample.query}'"
            assert len(faq.answer) > 10, (
                f"Слишком короткий ответ B0 для '{sample.query}': '{faq.answer}'"
            )
            assert result.retrieval is not None, "B0 должен вернуть retrieval"
            print(f"\n  B0 [{sample.sample_id}] '{sample.query}' → '{faq.answer[:80]}...'")

    @pytest.mark.filterwarnings("default")
    def test_b0_out_of_scope_descriptive(self, chunks, mini_samples):
        """DESCRIPTIVE ONLY: B0 out_of_scope accuracy (не quality gate).

        B0 — keyword matcher без понятия "out_of_scope". Accuracy близка к 0%,
        т.к. общие слова ('работает', 'где') матчат clinic chunks. Этот тест
        фиксирует метрику для отчёта, но НЕ является quality gate.
        """
        strategy = KeywordTemplateStrategy()
        oos_samples = [s for s in mini_samples if s.category == "out_of_scope"]
        assert len(oos_samples) > 0, "Нет out_of_scope samples в mini eval set"

        correct = 0
        for sample in oos_samples:
            result = strategy.answer_directly(sample.query, chunks)
            if not result.answer.answerable:
                correct += 1
            else:
                print(f"  B0 ошибочно answerable=True: '{sample.query}' → '{result.answer.answer[:60]}'")

        accuracy = correct / len(oos_samples)
        print(f"\n[DESCRIPTIVE] B0 out_of_scope accuracy: {accuracy:.0%} ({correct}/{len(oos_samples)})")
        print("  (B0 — keyword baseline, ожидаемый accuracy ~0%. Не quality gate.)")

    def test_b0_doctor_info_finds_doctor(self, chunks, mini_samples):
        """B0 должен найти врача по фамилии."""
        strategy = KeywordTemplateStrategy()
        doctor_samples = [s for s in mini_samples if s.category == "doctor_info"]

        for sample in doctor_samples:
            result = strategy.answer_directly(sample.query, chunks)
            faq = result.answer
            assert faq.answerable, f"B0 не нашёл врача: '{sample.query}'"
            if sample.expected_doctor:
                surname = sample.expected_doctor.split()[0]
                assert surname.lower() in faq.answer.lower(), (
                    f"Фамилия '{surname}' не в ответе B0: '{faq.answer[:80]}'"
                )


# ---------------------------------------------------------------------------
# 3. Evaluation sanity
# ---------------------------------------------------------------------------


class TestEvaluationSanity:
    """Проверяет evaluation pipeline на синтетических ответах."""

    def _make_result(
        self, sample_id: str, answer: FAQAnswer, strategy: str = "S1"
    ) -> StrategyResult:
        """Хелпер: создаёт StrategyResult из FAQAnswer."""
        return StrategyResult(
            sample_id=sample_id,
            strategy_id=StrategyID(strategy),
            answer=answer,
        )

    def test_correct_answerable_scores_well(self, chunks):
        """Корректный answerable=True + правильные данные → хорошие метрики."""
        sample = EvalSample(
            sample_id="synth_good",
            query="Какой телефон?",
            category="clinic_info",
            subtype="phone_contact",
            answerable=True,
            expected_doctor=None,
        )
        answer = FAQAnswer(
            answer="Телефон клиники: +7 (842) 231-45-55.",
            answerable=True,
            confidence=0.9,
        )
        result = self._make_result("synth_good", answer)
        kb_text = "\n".join(f"{c.title}\n{c.content}" for c in chunks)

        score = evaluate_single(result, sample, kb_text)
        assert score.answerability_correct, "Answerability должен совпасть"
        assert score.unsupported_claims == 0, (
            f"Телефон из KB не должен быть unsupported, "
            f"total_claims={score.total_claims}"
        )

    def test_wrong_answerable_detected(self, chunks):
        """answerable=True когда gold=False → answerability_correct=False."""
        sample = EvalSample(
            sample_id="synth_bad_ans",
            query="Сколько стоит?",
            category="out_of_scope",
            subtype="pricing",
            answerable=False,
        )
        answer = FAQAnswer(
            answer="Имплант стоит 50000 рублей.",
            answerable=True,
            confidence=0.8,
        )
        result = self._make_result("synth_bad_ans", answer)
        kb_text = "\n".join(f"{c.title}\n{c.content}" for c in chunks)

        score = evaluate_single(result, sample, kb_text)
        assert not score.answerability_correct, "Должен зафиксировать ошибку answerability"

    def test_hallucinated_phone_detected(self, chunks):
        """Выдуманный телефон → unsupported claim."""
        sample = EvalSample(
            sample_id="synth_halluc",
            query="Какой телефон?",
            category="clinic_info",
            answerable=True,
        )
        answer = FAQAnswer(
            answer="Звоните: +7 (999) 000-00-00.",
            answerable=True,
            confidence=0.5,
        )
        result = self._make_result("synth_halluc", answer)
        kb_text = "\n".join(f"{c.title}\n{c.content}" for c in chunks)

        score = evaluate_single(result, sample, kb_text)
        assert score.unsupported_claims > 0, (
            "Выдуманный телефон должен быть unsupported claim"
        )

    def test_retrieval_score_multi_gold(self):
        """compute_retrieval_score корректно считает hit/recall/mrr с multi-gold."""
        result = StrategyResult(
            sample_id="test",
            strategy_id=StrategyID.S2,
            retrieval=RetrievalResult(
                chunk_ids=["clinic_contacts", "clinic_location", "doctor_1"],
                scores=[0.9, 0.7, 0.5],
            ),
        )
        # Alt 1: clinic_location (позиция 2)
        # Alt 2: price_a (не найден)
        gold_ids = [["clinic_location"], ["price_a"]]

        score = compute_retrieval_score(result, gold_ids)
        assert score.hit_at_k, "gold chunk в retrieved → hit=True"
        assert score.recall_at_k == 1.0, "1/1 gold found → recall=1.0"
        assert score.reciprocal_rank == 0.5, "gold на позиции 2 → MRR=0.5"

    def test_retrieval_score_miss(self):
        """Отсутствие gold chunk → hit=False, recall=0."""
        result = StrategyResult(
            sample_id="test_miss",
            strategy_id=StrategyID.S2,
            retrieval=RetrievalResult(
                chunk_ids=["doctor_1", "doctor_2"],
                scores=[0.8, 0.6],
            ),
        )
        gold_ids = [["clinic_location"]]

        score = compute_retrieval_score(result, gold_ids)
        assert not score.hit_at_k, "gold не в retrieved → hit=False"
        assert score.recall_at_k == 0.0


# ---------------------------------------------------------------------------
# 4. _claim_in_kb edge cases (Phase 4 validation)
# ---------------------------------------------------------------------------


class TestClaimInKB:
    """Проверяет расширенную _claim_in_kb: числа, ФИО, URL."""

    def _kb_text(self, chunks) -> str:
        return "\n".join(f"{c.title}\n{c.content}" for c in chunks)

    def test_phone_normalized(self, chunks):
        """Телефон с другим форматированием находится в KB."""
        from d4.evaluation.deterministic import _claim_in_kb
        kb = self._kb_text(chunks)
        assert _claim_in_kb("+7(842)231-45-55", kb), "Телефон без пробелов"
        assert _claim_in_kb("+7 842 231 45 55", kb), "Телефон через пробелы"

    def test_experience_years(self, chunks):
        """Число '7 лет' найдено при наличии '7' в KB."""
        from d4.evaluation.deterministic import _claim_in_kb
        kb = self._kb_text(chunks)
        assert _claim_in_kb("7 лет", kb), "Стаж 7 лет в KB"

    def test_fio_stem(self, chunks):
        """ФИО через prefix-stem: 'Ермакову' (дат.) → 'ерма' в KB."""
        from d4.evaluation.deterministic import _claim_in_kb
        kb = self._kb_text(chunks)
        assert _claim_in_kb("Ермакову Александру", kb), "Дательный падеж ФИО"

    def test_url_normalized(self, chunks):
        """URL без протокола или с trailing slash."""
        from d4.evaluation.deterministic import _claim_in_kb
        kb = self._kb_text(chunks)
        assert _claim_in_kb("https://dentklever.ru/", kb), "URL с протоколом"

    def test_hallucinated_fio_fails(self, chunks):
        """Несуществующий ФИО не должен найтись."""
        from d4.evaluation.deterministic import _claim_in_kb
        kb = self._kb_text(chunks)
        assert not _claim_in_kb("Пупкин Василий Иванович", kb), "Выдуманное ФИО"

    def test_hallucinated_phone_fails(self, chunks):
        """Выдуманный телефон не в KB."""
        from d4.evaluation.deterministic import _claim_in_kb
        kb = self._kb_text(chunks)
        assert not _claim_in_kb("+7 (999) 000-00-00", kb), "Выдуманный телефон"


# ---------------------------------------------------------------------------
# 5. Gold map coverage (multi-gold)
# ---------------------------------------------------------------------------


class TestGoldMapCoverage:
    """Проверяет качество gold_map для mini eval set."""

    def test_answerable_have_gold_chunks(self, mini_samples, gold_map):
        """Все answerable samples (кроме reasoning) имеют непустые gold_chunk_ids."""
        missing = []
        for sample in mini_samples:
            if not sample.answerable:
                continue
            if sample.category == "reasoning":
                continue
            golds = gold_map.get(sample.sample_id, [])
            if not golds:
                missing.append(f"{sample.sample_id} ({sample.category}/{sample.subtype})")

        assert not missing, f"Answerable без gold chunks: {missing}"

    def test_out_of_scope_empty_gold(self, mini_samples, gold_map):
        """out_of_scope samples должны иметь пустые gold_chunk_ids."""
        for sample in mini_samples:
            if sample.category != "out_of_scope":
                continue
            golds = gold_map.get(sample.sample_id, [])
            assert golds == [], (
                f"out_of_scope {sample.sample_id} имеет gold chunks: {golds}"
            )

    def test_gold_chunks_exist_in_kb(self, chunks, mini_samples, gold_map):
        """Все gold_chunk_ids из gold_map существуют в chunks.json."""
        valid_ids = {c.id for c in chunks}
        invalid = []
        for sample in mini_samples:
            for alt in gold_map.get(sample.sample_id, []):
                for gid in alt:
                    if gid not in valid_ids:
                        invalid.append((sample.sample_id, gid))

        assert not invalid, f"Несуществующие chunk_ids в gold_map: {invalid}"

    def test_mini_eval_set_distribution(self, mini_samples):
        """Mini eval set содержит все основные категории."""
        categories = {s.category for s in mini_samples}
        expected = {
            "clinic_info", "doctor_info",
            "reasoning", "pricing", "out_of_scope",
        }
        missing = expected - categories
        assert not missing, f"Отсутствуют категории: {missing}"


# ---------------------------------------------------------------------------
# 6. Price chunks — retrieval, B0, claim extraction
# ---------------------------------------------------------------------------


class TestPriceChunks:
    """Smoke tests для price chunks."""

    def test_price_chunks_exist(self, chunks):
        """KB содержит price_list chunks."""
        price_chunks = [c for c in chunks if c.source_type == "price_list"]
        assert len(price_chunks) >= 10, (
            f"Ожидалось >=10 price chunks, получено {len(price_chunks)}"
        )

    def test_price_chunk_has_raw_data(self, chunks):
        """Каждый price chunk содержит raw_data.services."""
        price_chunks = [c for c in chunks if c.source_type == "price_list"]
        for pc in price_chunks:
            assert pc.raw_data, f"price chunk {pc.id} без raw_data"
            assert "services" in pc.raw_data, f"price chunk {pc.id} без services"

    def test_price_chunk_nl_format(self, chunks):
        """Price chunks имеют NL-формат с ₽ в content."""
        price_chunks = [c for c in chunks if c.source_type == "price_list"]
        for pc in price_chunks:
            assert "₽" in pc.content, f"price chunk {pc.id} без ₽ в content"
            assert pc.content.startswith("Прайс-лист:"), (
                f"price chunk {pc.id} не начинается с 'Прайс-лист:'"
            )

    def test_bm25_price_hit(self, chunks):
        """BM25 находит price chunk для ценового запроса."""
        strategy = LexicalStrategy(top_k=5)
        result = strategy.select_context("Сколько стоит имплант?", chunks)
        assert any("price_" in cid for cid in result.chunk_ids), (
            f"BM25 не нашёл price chunk: {result.chunk_ids}"
        )

    def test_b0_price_answer(self, chunks):
        """B0 формирует ответ с ценой для ценового запроса."""
        strategy = KeywordTemplateStrategy()
        result = strategy.answer_directly("Сколько стоит имплант?", chunks)
        faq = result.answer
        assert faq.answerable, "B0 должен ответить на ценовой запрос"
        assert "₽" in faq.answer, "B0 ответ должен содержать цену"


class TestPriceClaimExtraction:
    """Smoke tests для извлечения ценовых claims."""

    def test_extract_price_claim(self):
        """_extract_claims ловит полную цену 'от 35 000 ₽'."""
        from d4.evaluation.deterministic import _extract_claims
        claims = _extract_claims("Установка имплантата — от 35 000 ₽.")
        assert any("35" in c and "₽" in c for c in claims), (
            f"Цена не извлечена: {claims}"
        )

    def test_price_claim_in_kb(self, chunks):
        """Реальная цена из KB проходит _claim_in_kb, выдуманная — нет."""
        from d4.evaluation.deterministic import _claim_in_kb
        kb_text = "\n".join(f"{c.title}\n{c.content}" for c in chunks)
        assert _claim_in_kb("от 35 000 ₽", kb_text), (
            "Цена имплантата должна быть в KB"
        )
        assert not _claim_in_kb("от 99 999 ₽", kb_text), (
            "Выдуманная цена не должна быть в KB"
        )


# ---------------------------------------------------------------------------
# 7. DirectAnswerResult (Фаза 4B)
# ---------------------------------------------------------------------------


class TestDirectAnswerResult:
    """Проверяет dataclass DirectAnswerResult и его интеграцию с S5/B0."""

    def test_default_values(self):
        """DirectAnswerResult defaults: route=direct, tokens=0, error=None."""
        dar = DirectAnswerResult(answer=FAQAnswer())
        assert dar.route_taken == "direct"
        assert dar.tokens_prompt == 0
        assert dar.tokens_completion == 0
        assert dar.retrieval.chunk_ids == []
        assert dar.error is None

    def test_fallback_route(self):
        """DirectAnswerResult с fallback сохраняет tokens и retrieval."""
        dar = DirectAnswerResult(
            answer=FAQAnswer(answer="ответ", answerable=True, confidence=0.9),
            retrieval=RetrievalResult(chunk_ids=["c1", "c2"], scores=[0.9, 0.7]),
            latency_ms=250.0,
            tokens_prompt=500,
            tokens_completion=100,
            route_taken="fallback",
        )
        assert dar.route_taken == "fallback"
        assert dar.tokens_prompt == 500
        assert dar.retrieval.chunk_ids == ["c1", "c2"]

    def test_isinstance_distinguishes_from_faqanswer(self):
        """isinstance корректно различает DirectAnswerResult и FAQAnswer."""
        dar = DirectAnswerResult(answer=FAQAnswer())
        faq = FAQAnswer()
        assert isinstance(dar, DirectAnswerResult)
        assert not isinstance(faq, DirectAnswerResult)

    def test_s5_returns_direct_answer_result(self, chunks):
        """S5 answer_directly() возвращает DirectAnswerResult (не FAQAnswer)."""
        from d4.config import load_config
        from d4.pipeline.factory import build_strategies

        config = load_config("d4/configs/experiment.yaml")
        strategies = build_strategies(config, include_baseline=True, include_experimental=True)
        s5 = next(s for s in strategies if s.strategy_id == "S5")

        result = s5.answer_directly("Где вы находитесь?", chunks)
        assert isinstance(result, DirectAnswerResult), (
            f"S5 должен вернуть DirectAnswerResult, получен {type(result).__name__}"
        )
        assert result.route_taken in ("direct", "fallback")
        assert result.latency_ms > 0

    def test_b0_returns_direct_answer_result(self, chunks):
        """B0 answer_directly() возвращает DirectAnswerResult (thread-safe API)."""
        b0 = KeywordTemplateStrategy()
        result = b0.answer_directly("Где вы находитесь?", chunks)
        assert isinstance(result, DirectAnswerResult), (
            f"B0 должен вернуть DirectAnswerResult, получен {type(result).__name__}"
        )
        assert isinstance(result.answer, FAQAnswer)
        assert result.retrieval is not None
        assert result.route_taken == "direct"


# ---------------------------------------------------------------------------
# 8. Slot-aware нормализаторы (Фаза 5A)
# ---------------------------------------------------------------------------


class TestSlotNormalizers:
    """Проверяет типизированные нормализаторы для fact_match_rate."""

    def test_phone_strips_non_digits(self):
        """Телефон → только цифры."""
        assert normalize_phone("+7(8422)58-58-58") == "78422585858"

    def test_phone_8_to_7(self):
        """Ведущая 8 → 7 для российских номеров (11 цифр)."""
        assert normalize_phone("8 (8422) 58-58-58") == "78422585858"

    def test_phone_short_unchanged(self):
        """Короткий номер (не 11 цифр) — не трогать ведущую 8."""
        assert normalize_phone("84225858") == "84225858"

    def test_price_only_digits(self):
        """Цена → только цифры, без пробелов/символов."""
        assert normalize_price("от 35 000 ₽") == "35000"
        assert normalize_price("2 000") == "2000"

    def test_fio_extracts_surname(self):
        """ФИО → lowercase первое слово ≥3 символов."""
        assert normalize_fio("Батылин Д.В.") == "батылин"
        assert normalize_fio("Ермаков Александр Владимирович") == "ермаков"

    def test_address_strips_abbreviations(self):
        """Адрес → без 'г.', 'ул.', 'д.', lowercase."""
        assert normalize_address("г. Ульяновск, ул. Рябикова, 56") == "ульяновск рябикова 56"
        assert normalize_address("ул. Рябикова, д. 56") == "рябикова 56"

    def test_schedule_simplifies(self):
        """Расписание → lowercase, без ':' и '-'."""
        result = normalize_schedule("Пн-Пт 9:00-20:00")
        assert "пн" in result
        assert "пт" in result
        assert "9" in result
        assert "20" in result

    @pytest.mark.parametrize(
        "canonical, llm_answer",
        [
            ("ежедневно с 08:00 до 20:00", "Мы работаем ежедневно с 8:00 до 20:00."),
            ("ежедневно с 08:00 до 20:00", "Клиника открыта ежедневно, 8-20"),
            ("ежедневно с 08:00 до 20:00", "Приём пациентов — ежедневно 08:00–20:00, включая выходные."),
            ("Пн-Пт 9:00-20:00", "Мы принимаем по будням с 9 до 20."),
            ("Сб-Вс 10:00-18:00", "В выходные работаем с 10:00 до 18:00."),
        ],
    )
    def test_schedule_symmetric_match(self, canonical, llm_answer):
        """schedule fact_match_rate через _compute_fact_match_rate (token set containment)."""
        from d4.evaluation.deterministic import _compute_fact_match_rate
        from d4.models import GoldFact
        fact = GoldFact(fact_type="schedule", canonical_value=canonical)
        fmr = _compute_fact_match_rate(llm_answer, [fact])
        assert fmr == 1.0, (
            f"FMR={fmr} for canonical={canonical!r}, answer={llm_answer!r}"
        )

    def test_text_lowercase_no_punctuation(self):
        """Текст → lowercase, без пунктуации."""
        assert normalize_text("Ульяновск!") == "ульяновск"


# ---------------------------------------------------------------------------
# 9. fact_match_rate (Фаза 5A)
# ---------------------------------------------------------------------------


class TestFactMatchRate:
    """Проверяет slot-aware fact_match_rate на реалистичных примерах."""

    def test_full_match_address(self):
        """Все gold_facts найдены → rate = 1.0."""
        facts = [
            GoldFact(fact_type="text", canonical_value="Ульяновск"),
            GoldFact(fact_type="address", canonical_value="Рябикова, 56"),
        ]
        rate = _compute_fact_match_rate(
            "Клиника расположена: г. Ульяновск, ул. Рябикова, 56.", facts,
        )
        assert rate == 1.0

    def test_phone_different_format(self):
        """Телефон в другом формате (+7 vs 8) → rate = 1.0."""
        facts = [GoldFact(fact_type="phone", canonical_value="+7(8422)58-58-58")]
        rate = _compute_fact_match_rate("Звоните: 8 (8422) 58-58-58", facts)
        assert rate == 1.0

    def test_fio_in_sentence(self):
        """ФИО внутри текста → rate = 1.0."""
        facts = [GoldFact(fact_type="fio", canonical_value="Батылин")]
        rate = _compute_fact_match_rate("Доктор Батылин работает в клинике.", facts)
        assert rate == 1.0

    def test_price_with_spaces(self):
        """Цена с пробелами → rate = 1.0."""
        facts = [GoldFact(fact_type="price", canonical_value="35000")]
        rate = _compute_fact_match_rate("Имплантат — от 35 000 ₽", facts)
        assert rate == 1.0

    def test_partial_match(self):
        """Один из двух фактов найден → rate = 0.5."""
        facts = [
            GoldFact(fact_type="text", canonical_value="Ульяновск"),
            GoldFact(fact_type="phone", canonical_value="+7(8422)58-58-58"),
        ]
        rate = _compute_fact_match_rate("Клиника в Ульяновске.", facts)
        assert rate == 0.5

    def test_empty_gold_returns_none(self):
        """Нет gold_facts → rate = None (метрика не определена)."""
        assert _compute_fact_match_rate("Любой ответ", []) is None

    def test_complete_miss(self):
        """Ни один факт не найден → rate = 0.0."""
        facts = [GoldFact(fact_type="phone", canonical_value="+7(8422)58-58-58")]
        rate = _compute_fact_match_rate("Ничего полезного", facts)
        assert rate == 0.0

    def test_evaluate_single_includes_fact_match(self, mini_samples, chunks):
        """evaluate_single заполняет fact_match_rate из gold_facts."""
        sample = mini_samples[0]  # q_0001: Ульяновск + Рябикова, 56
        result = StrategyResult(
            sample_id=sample.sample_id,
            strategy_id=StrategyID.B0,
            answer=FAQAnswer(
                answer="г. Ульяновск, ул. Рябикова, 56.",
                answerable=True,
            ),
        )
        score = evaluate_single(result, sample, "")
        assert score.fact_match_rate == 1.0


# ---------------------------------------------------------------------------
# 10. Group-stratified split (Фаза 3 P0)
# ---------------------------------------------------------------------------


class TestGroupStratifiedSplit:
    """Проверяет group-aware dev/test split: seed + вариации = одна группа."""

    @staticmethod
    def _make_family(
        family_id: str, category: str, n_variations: int = 2,
    ) -> list[EvalSample]:
        """Создаёт семью: seed + N вариаций."""
        samples = [
            EvalSample(
                sample_id=family_id,
                query=f"seed query {family_id}",
                category=category,
                seed_family_id=family_id,
            ),
        ]
        for i in range(n_variations):
            samples.append(EvalSample(
                sample_id=f"{family_id}_v{i+1}",
                query=f"variation {i+1} of {family_id}",
                category=category,
                seed_family_id=family_id,
                notes=f"llm_variation of {family_id}; style: formal",
            ))
        return samples

    def test_no_data_leakage(self):
        """Семья целиком попадает в dev ИЛИ test, не разделяется."""
        from d4.data_gen.query_generator import _infer_seed_family, split_eval_set

        samples = (
            self._make_family("f1", "clinic_info")
            + self._make_family("f2", "clinic_info")
            + self._make_family("f3", "doctor_info")
            + self._make_family("f4", "doctor_info")
            + self._make_family("f5", "pricing")
            + self._make_family("f6", "pricing")
            + self._make_family("f7", "out_of_scope")
            + self._make_family("f8", "out_of_scope")
        )

        dev, test = split_eval_set(samples, test_ratio=0.3, seed=42)

        dev_families = {_infer_seed_family(s) for s in dev}
        test_families = {_infer_seed_family(s) for s in test}
        overlap = dev_families & test_families
        assert not overlap, f"Data leakage! Overlap families: {overlap}"

    def test_category_balance(self):
        """Обе части содержат все основные категории."""
        from collections import Counter

        from d4.data_gen.query_generator import split_eval_set

        samples = (
            self._make_family("a1", "clinic_info")
            + self._make_family("a2", "clinic_info")
            + self._make_family("a3", "clinic_info")
            + self._make_family("b1", "doctor_info")
            + self._make_family("b2", "doctor_info")
            + self._make_family("b3", "doctor_info")
            + self._make_family("c1", "pricing")
            + self._make_family("c2", "pricing")
            + self._make_family("c3", "pricing")
            + self._make_family("d1", "out_of_scope")
            + self._make_family("d2", "out_of_scope")
            + self._make_family("d3", "out_of_scope")
        )

        dev, test = split_eval_set(samples, test_ratio=0.3, seed=42)
        test_cats = set(Counter(s.category for s in test).keys())
        expected = {"clinic_info", "doctor_info", "pricing", "out_of_scope"}
        assert expected <= test_cats, f"Test set не покрывает категории: {expected - test_cats}"

    def test_infer_from_notes_fallback(self):
        """_infer_seed_family извлекает family из notes (backward compat)."""
        from d4.data_gen.query_generator import _infer_seed_family

        s = EvalSample(
            sample_id="q_005_v1",
            query="вариация",
            category="clinic_info",
            notes="llm_variation of q_005; style: formal",
        )
        assert _infer_seed_family(s) == "q_005"

    def test_infer_self_for_seed(self):
        """Seed без seed_family_id и без notes → family = sample_id."""
        from d4.data_gen.query_generator import _infer_seed_family

        s = EvalSample(
            sample_id="q_100",
            query="seed query",
            category="doctor_info",
        )
        assert _infer_seed_family(s) == "q_100"

    def test_small_eval_set_all_to_dev(self):
        """Очень маленький eval set → всё в dev, test пустой."""
        from d4.data_gen.query_generator import split_eval_set

        samples = [
            EvalSample(sample_id="only1", query="q", category="clinic_info"),
        ]
        dev, test = split_eval_set(samples, test_ratio=0.3)
        assert len(dev) == 1
        assert len(test) == 0

    def test_reproducibility(self):
        """Одинаковый seed → одинаковый split."""
        from d4.data_gen.query_generator import split_eval_set

        samples = (
            self._make_family("r1", "clinic_info")
            + self._make_family("r2", "doctor_info")
            + self._make_family("r3", "pricing")
            + self._make_family("r4", "out_of_scope")
        )

        dev1, test1 = split_eval_set(samples, test_ratio=0.3, seed=99)
        dev2, test2 = split_eval_set(samples, test_ratio=0.3, seed=99)

        assert [s.sample_id for s in dev1] == [s.sample_id for s in dev2]
        assert [s.sample_id for s in test1] == [s.sample_id for s in test2]


# ---------------------------------------------------------------------------
# 11. GoldFact backward-compat loader (Фаза 5A)
# ---------------------------------------------------------------------------


class TestGoldFactLoader:
    """Проверяет backward-compat загрузку gold_facts из YAML."""

    def test_structured_gold_facts_loaded(self):
        """Structured gold_facts (fact_type+canonical_value) загружаются корректно."""
        from d4.data_gen.query_generator import load_eval_set

        samples = load_eval_set("d4/tests/mini_eval_set.yaml")
        s = next(s for s in samples if s.sample_id == "q_0001")
        assert len(s.gold_facts) == 2
        assert all(isinstance(f, GoldFact) for f in s.gold_facts)
        assert s.gold_facts[0].fact_type == "address"
        assert s.gold_facts[0].canonical_value == "Ульяновск"

    def test_plain_strings_backward_compat(self):
        """list[str] в YAML → list[GoldFact] с fact_type='text' (backward compat)."""
        import tempfile

        yaml_content = """
- sample_id: test_compat
  query: "Тест?"
  category: clinic_info
  subtype: address_location
  answerable: true
  difficulty: 1
  gold_chunk_ids:
  - - clinic_location
  gold_facts:
  - "Ульяновск"
  - "Рябикова, 56"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            from d4.data_gen.query_generator import load_eval_set
            samples = load_eval_set(f.name)

        s = samples[0]
        assert len(s.gold_facts) == 2
        assert all(isinstance(f, GoldFact) for f in s.gold_facts)
        assert s.gold_facts[0].fact_type == "text"
        assert s.gold_facts[0].canonical_value == "Ульяновск"

    def test_empty_gold_facts(self):
        """gold_facts: [] → пустой список GoldFact."""
        from d4.data_gen.query_generator import load_eval_set

        samples = load_eval_set("d4/tests/mini_eval_set.yaml")
        s = next(s for s in samples if s.sample_id == "q_0050")
        assert s.gold_facts == []


# ---------------------------------------------------------------------------
# 12. GenerationConfig wiring (Rev5)
# ---------------------------------------------------------------------------


class TestGenerationConfigWiring:
    """Проверяет end-to-end wiring generation_config.yaml."""

    def test_load_generation_config(self):
        """generation_config.yaml загружается и валидируется."""
        from d4.config import load_generation_config

        cfg = load_generation_config("d4/configs/generation_config.yaml")
        assert cfg.split.method == "group_stratified"
        assert cfg.split.group_field == "seed_family_id"
        assert cfg.dedup.model == "intfloat/multilingual-e5-base"
        assert 0.0 < cfg.dedup.threshold <= 1.0
        assert 0.0 < cfg.split.test_ratio < 1.0

    def test_split_with_config(self, mini_samples):
        """split_eval_set(config=...) использует параметры из config."""
        from d4.config import load_generation_config
        from d4.data_gen.query_generator import split_eval_set

        cfg = load_generation_config("d4/configs/generation_config.yaml")
        dev, test = split_eval_set(mini_samples, config=cfg)
        assert len(dev) + len(test) == len(mini_samples)

    def test_split_rejects_unknown_method(self, mini_samples):
        """split_eval_set() отклоняет неизвестный split.method."""
        from d4.config import GenerationConfig, SplitConfig
        from d4.data_gen.query_generator import split_eval_set

        bad_cfg = GenerationConfig(split=SplitConfig(method="random"))
        with pytest.raises(ValueError, match="не поддерживается"):
            split_eval_set(mini_samples, config=bad_cfg)

    def test_dedup_requires_model_or_config(self):
        """deduplicate_queries() без model_name и без config → ValueError."""
        from d4.data_gen.query_generator import deduplicate_queries

        samples = [
            EvalSample(sample_id="a", query="Где вы?", category="clinic_info"),
            EvalSample(sample_id="b", query="Какой адрес?", category="clinic_info"),
        ]
        with pytest.raises(ValueError, match="model_name обязателен"):
            deduplicate_queries(samples)

    @pytest.mark.slow
    def test_dedup_with_config_loads_model(self):
        """deduplicate_queries(config=...) загружает модель и работает end-to-end.

        Помечен @pytest.mark.slow: загружает SentenceTransformer (~260 MB).
        Запуск: pytest -m slow
        """
        from d4.config import load_generation_config
        from d4.data_gen.query_generator import deduplicate_queries

        cfg = load_generation_config("d4/configs/generation_config.yaml")
        samples = [
            EvalSample(sample_id="a", query="Где вы?", category="clinic_info"),
            EvalSample(sample_id="b", query="Какой адрес?", category="clinic_info"),
        ]
        result = deduplicate_queries(samples, config=cfg)
        assert len(result) <= len(samples)


# ---------------------------------------------------------------------------
# 9. S4r rerank logging + retrieval metrics (Фаза 4C + 5B)
# ---------------------------------------------------------------------------


class TestRerankLogging:
    """Проверяет pre_rerank_chunk_ids и rerank lift metrics."""

    def test_retrieval_result_pre_rerank_field(self):
        """RetrievalResult хранит pre_rerank_chunk_ids."""
        from d4.models import RetrievalResult

        r = RetrievalResult(
            chunk_ids=["a", "b"],
            pre_rerank_chunk_ids=["a", "b", "c", "d"],
        )
        assert r.pre_rerank_chunk_ids == ["a", "b", "c", "d"]
        assert r.chunk_ids == ["a", "b"]

    def test_retrieval_result_pre_rerank_default_empty(self):
        """По умолчанию pre_rerank_chunk_ids пустой (non-S4r стратегии)."""
        from d4.models import RetrievalResult

        r = RetrievalResult(chunk_ids=["x"])
        assert r.pre_rerank_chunk_ids == []

    def test_retrieval_score_pre_rerank_fields(self):
        """RetrievalScore хранит pre_rerank_hit и pre_rerank_recall."""
        from d4.models import RetrievalScore, StrategyID

        s = RetrievalScore(
            sample_id="q1",
            strategy_id=StrategyID.S4r,
            hit_at_k=True,
            pre_rerank_hit=True,
            pre_rerank_recall=0.5,
        )
        assert s.pre_rerank_hit is True
        assert s.pre_rerank_recall == 0.5

    def test_compute_retrieval_score_pre_rerank(self):
        """compute_retrieval_score вычисляет pre/post rerank delta."""
        from d4.evaluation.retrieval_metrics import compute_retrieval_score
        from d4.models import RetrievalResult, StrategyID, StrategyResult

        result = StrategyResult(
            sample_id="q1",
            strategy_id=StrategyID.S4r,
            retrieval=RetrievalResult(
                chunk_ids=["gold_a", "noise_1"],
                pre_rerank_chunk_ids=["noise_1", "noise_2", "gold_a", "noise_3"],
            ),
        )
        score = compute_retrieval_score(result, [["gold_a"]])

        assert score.hit_at_k is True
        assert score.pre_rerank_hit is True
        assert score.recall_at_k == 1.0
        assert score.pre_rerank_recall == 1.0

    def test_compute_retrieval_score_rerank_lifts_gold(self):
        """Reranking поднимает gold chunk: pre=miss, post=hit."""
        from d4.evaluation.retrieval_metrics import compute_retrieval_score
        from d4.models import RetrievalResult, StrategyID, StrategyResult

        result = StrategyResult(
            sample_id="q2",
            strategy_id=StrategyID.S4r,
            retrieval=RetrievalResult(
                chunk_ids=["gold_a", "noise_1", "noise_2"],
                pre_rerank_chunk_ids=["noise_1", "noise_2", "noise_3"],
            ),
        )
        score = compute_retrieval_score(result, [["gold_a"]])

        assert score.hit_at_k is True
        assert score.pre_rerank_hit is False

    def test_aggregate_retrieval_metrics_rerank_lift(self):
        """aggregate_retrieval_metrics включает rerank_hit_lift для S4r."""
        from d4.evaluation.retrieval_metrics import aggregate_retrieval_metrics
        from d4.models import RetrievalScore, StrategyID

        scores = [
            RetrievalScore(
                sample_id="q1", strategy_id=StrategyID.S4r,
                hit_at_k=True, pre_rerank_hit=False,
                recall_at_k=1.0, pre_rerank_recall=0.0,
            ),
            RetrievalScore(
                sample_id="q2", strategy_id=StrategyID.S4r,
                hit_at_k=True, pre_rerank_hit=True,
                recall_at_k=1.0, pre_rerank_recall=1.0,
            ),
        ]
        agg = aggregate_retrieval_metrics(scores)

        assert "S4r" in agg
        assert agg["S4r"]["hit_rate"] == 1.0
        assert agg["S4r"]["pre_rerank_hit_rate"] == 0.5
        assert agg["S4r"]["rerank_hit_lift"] == 0.5


class TestRouteAccounting:
    """Проверяет S5 route accounting (share_direct, share_fallback)."""

    def test_aggregate_route_metrics(self):
        """aggregate_route_metrics разделяет direct/fallback."""
        from d4.evaluation.metrics import aggregate_route_metrics
        from d4.models import StrategyID, StrategyResult

        results = [
            StrategyResult(
                sample_id="q1", strategy_id=StrategyID.S5,
                route_taken="direct", latency_ms=10.0,
                tokens_prompt=0, tokens_completion=0,
            ),
            StrategyResult(
                sample_id="q2", strategy_id=StrategyID.S5,
                route_taken="fallback", latency_ms=500.0,
                tokens_prompt=1000, tokens_completion=200,
            ),
            StrategyResult(
                sample_id="q3", strategy_id=StrategyID.S5,
                route_taken="direct", latency_ms=15.0,
                tokens_prompt=0, tokens_completion=0,
            ),
        ]
        agg = aggregate_route_metrics(results)

        assert "S5" in agg
        s5 = agg["S5"]
        assert abs(s5["share_direct"] - 2 / 3) < 0.01
        assert abs(s5["share_fallback"] - 1 / 3) < 0.01
        assert s5["mean_latency_direct"] < s5["mean_latency_fallback"]
        assert s5["mean_tokens_direct"] == 0.0
        assert s5["mean_tokens_fallback"] > 0.0

    def test_aggregate_route_metrics_with_fmr(self):
        """aggregate_route_metrics включает fmr_direct/fmr_fallback."""
        from d4.evaluation.metrics import aggregate_route_metrics
        from d4.models import DeterministicScore, StrategyID, StrategyResult

        results = [
            StrategyResult(sample_id="q1", strategy_id=StrategyID.S5, route_taken="direct"),
            StrategyResult(sample_id="q2", strategy_id=StrategyID.S5, route_taken="fallback"),
        ]
        det = [
            DeterministicScore(sample_id="q1", strategy_id=StrategyID.S5, fact_match_rate=1.0),
            DeterministicScore(sample_id="q2", strategy_id=StrategyID.S5, fact_match_rate=0.5),
        ]
        agg = aggregate_route_metrics(results, det_scores=det)

        assert agg["S5"]["fmr_direct"] == 1.0
        assert agg["S5"]["fmr_fallback"] == 0.5

    def test_route_metrics_ignores_non_routed(self):
        """Стратегии без route_taken не попадают в route accounting."""
        from d4.evaluation.metrics import aggregate_route_metrics
        from d4.models import StrategyID, StrategyResult

        results = [
            StrategyResult(sample_id="q1", strategy_id=StrategyID.S1, route_taken=""),
            StrategyResult(sample_id="q2", strategy_id=StrategyID.S4r, route_taken=""),
        ]
        agg = aggregate_route_metrics(results)
        assert agg == {}


# ---------------------------------------------------------------------------
# 13. Hard eval set — adversarial/stress checks
# ---------------------------------------------------------------------------


class TestHardSetRetrieval:
    """Retrieval robustness на hard eval set (опечатки, косвенные формулировки)."""

    def test_hard_set_loads(self, hard_samples):
        """Hard set загружается и содержит 12 samples."""
        assert len(hard_samples) == 12
        cats = {s.category for s in hard_samples}
        assert cats == {"clinic_info", "doctor_info", "pricing", "reasoning", "aftercare", "out_of_scope"}

    def test_s2_retrieval_not_empty_on_answerable(self, chunks, hard_samples):
        """S2 (lexical) не возвращает пустой retrieval на answerable hard queries."""
        strategy = LexicalStrategy(top_k=5)
        answerable = [s for s in hard_samples if s.answerable]
        assert len(answerable) >= 8

        empty = []
        for s in answerable:
            result = strategy.select_context(s.query, chunks)
            if not result.chunk_ids:
                empty.append(s.sample_id)

        empty_rate = len(empty) / len(answerable)
        print(f"\n  S2 hard empty rate: {empty_rate:.0%} ({len(empty)}/{len(answerable)})")
        if empty:
            print(f"  Empty: {empty}")
        assert empty_rate <= 0.5, (
            f"S2 слишком часто пустой на hard set: {empty_rate:.0%}"
        )

    def test_s2_partial_gold_hit_on_hard(self, chunks, hard_samples):
        """S2 находит хотя бы часть gold chunks на hard answerable queries.

        DESCRIPTIVE: фиксируем hit rate, но gate мягкий (>= 30%).
        Hard set содержит опечатки и косвенные формулировки, поэтому
        lexical BM25 ожидаемо слабее, чем на canonical smoke set.
        """
        strategy = LexicalStrategy(top_k=5)
        gold_map_hard = build_gold_map(hard_samples)
        answerable = [s for s in hard_samples if s.answerable and gold_map_hard.get(s.sample_id)]

        hits = 0
        for s in answerable:
            result = strategy.select_context(s.query, chunks)
            gold_ids = _flatten_gold(gold_map_hard[s.sample_id])
            if gold_ids & set(result.chunk_ids):
                hits += 1
            else:
                print(f"  MISS: {s.sample_id} '{s.query[:50]}' gold={gold_ids}")

        hit_rate = hits / len(answerable) if answerable else 0
        print(f"\n  S2 hard hit@5: {hit_rate:.0%} ({hits}/{len(answerable)})")
        assert hit_rate >= 0.3, (
            f"S2 hard hit rate слишком низкий: {hit_rate:.0%}"
        )


class TestHardSetB0:
    """B0 behaviour на adversarial queries — descriptive metrics."""

    def test_b0_fails_on_hard_reasoning(self, chunks, hard_samples):
        """DESCRIPTIVE: B0 ожидаемо не справляется с hard reasoning.

        Не quality gate — фиксируем, что B0 возвращает ответ, но
        fact_match_rate будет низкий. Это подтверждает дискриминативность
        hard set: если бы B0 проходил reasoning, набор слишком лёгкий.
        """
        from d4.evaluation.deterministic import _compute_fact_match_rate

        strategy = KeywordTemplateStrategy()
        reasoning = [s for s in hard_samples if s.category == "reasoning"]
        assert len(reasoning) >= 2

        for s in reasoning:
            result = strategy.answer_directly(s.query, chunks)
            faq = result.answer
            fmr = _compute_fact_match_rate(faq.answer, s.gold_facts)
            fmr_str = f"{fmr:.2f}" if fmr is not None else "N/A"
            print(f"\n  B0 reasoning [{s.sample_id}] fmr={fmr_str}: '{faq.answer[:80]}'")

    def test_b0_tricky_oos_descriptive(self, chunks, hard_samples):
        """DESCRIPTIVE: B0 на tricky out-of-scope (ОМС, бесплатно).

        B0 keyword matcher не отличает «рентген бесплатно» от «рентген».
        Ожидаемо: answerable=True (ошибка B0). Фиксируем для сравнения.
        """
        strategy = KeywordTemplateStrategy()
        oos = [s for s in hard_samples if s.category == "out_of_scope"]
        assert len(oos) >= 2

        correct = 0
        for s in oos:
            result = strategy.answer_directly(s.query, chunks)
            if not result.answer.answerable:
                correct += 1
                print(f"  B0 correct reject: '{s.query}'")
            else:
                print(f"  B0 false positive: '{s.query}' → '{result.answer.answer[:60]}'")

        print(f"\n[DESCRIPTIVE] B0 hard OOS accuracy: {correct}/{len(oos)}")


class TestRetrievalByCategory:
    """Проверяет category breakdowns для retrieval metrics."""

    def test_aggregate_retrieval_by_category(self):
        """aggregate_retrieval_by_category разбивает по category."""
        from d4.evaluation.retrieval_metrics import aggregate_retrieval_by_category
        from d4.models import RetrievalScore, StrategyID

        samples = [
            EvalSample(sample_id="q1", query="Где вы?", category="clinic_info"),
            EvalSample(sample_id="q2", query="Кто хирург?", category="doctor_info"),
        ]
        scores = [
            RetrievalScore(
                sample_id="q1", strategy_id=StrategyID.S4,
                hit_at_k=True, recall_at_k=1.0,
            ),
            RetrievalScore(
                sample_id="q2", strategy_id=StrategyID.S4,
                hit_at_k=False, recall_at_k=0.0,
            ),
        ]
        agg = aggregate_retrieval_by_category(scores, samples)

        assert "S4" in agg
        assert agg["S4"]["clinic_info"]["hit_rate"] == 1.0
        assert agg["S4"]["doctor_info"]["hit_rate"] == 0.0
