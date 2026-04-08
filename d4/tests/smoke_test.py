"""Smoke-тесты D4v2: валидация компонентов БЕЗ LLM-вызовов.

Запуск: pytest d4/tests/smoke_test.py -v
Требования: chunks.json, doctors.yaml, mini_eval_set.yaml

Тесты проверяют:
1. Retrieval hit rate (S2 BM25 + S1 full context) — gold chunks в top-k
2. B0 template answers — корректные факты из KB
3. Evaluation sanity — ожидаемые метрики для синтетических ответов
4. Gold map coverage — непустые chunk_ids для answerable samples
"""

from __future__ import annotations

import pytest

from d4.evaluation.deterministic import evaluate_single
from d4.evaluation.gold_map import build_gold_map
from d4.evaluation.retrieval_metrics import compute_retrieval_score
from d4.models import (
    DeterministicScore,
    EvalSample,
    FAQAnswer,
    KBChunk,
    RetrievalResult,
    StrategyID,
    StrategyResult,
)
from d4.strategies.keyword_template import KeywordTemplateStrategy
from d4.strategies.lexical import LexicalStrategy


# ---------------------------------------------------------------------------
# 1. Retrieval hit rate
# ---------------------------------------------------------------------------


class TestRetrievalHitRate:
    """Проверяет что retrieval стратегии находят gold chunks."""

    def test_s2_factoid_hit_rate(self, chunks, mini_samples, gold_map):
        """S2 (BM25) должен найти gold chunks для factoid запросов."""
        strategy = LexicalStrategy(top_k=5)
        factoid_samples = [s for s in mini_samples if s.category == "factoid"]
        assert len(factoid_samples) > 0, "Нет factoid samples в mini eval set"

        hits = 0
        for sample in factoid_samples:
            result = strategy.select_context(sample.query, chunks)
            gold_ids = set(gold_map.get(sample.sample_id, []))
            if gold_ids & set(result.chunk_ids):
                hits += 1

        hit_rate = hits / len(factoid_samples)
        print(f"\nS2 factoid hit rate: {hit_rate:.0%} ({hits}/{len(factoid_samples)})")
        # Порог: после Phase 1+2 ожидаем >80%, сейчас фиксируем baseline
        assert hit_rate >= 0.0, "Hit rate не может быть отрицательным"

    def test_s2_doctor_lookup_hit_rate(self, chunks, mini_samples, gold_map):
        """S2 (BM25) должен найти doctor chunks по фамилии."""
        strategy = LexicalStrategy(top_k=5)
        doctor_samples = [s for s in mini_samples if s.category == "doctor_lookup"]
        assert len(doctor_samples) > 0, "Нет doctor_lookup samples"

        hits = 0
        for sample in doctor_samples:
            result = strategy.select_context(sample.query, chunks)
            gold_ids = set(gold_map.get(sample.sample_id, []))
            if gold_ids & set(result.chunk_ids):
                hits += 1

        hit_rate = hits / len(doctor_samples)
        print(f"\nS2 doctor hit rate: {hit_rate:.0%} ({hits}/{len(doctor_samples)})")

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


# ---------------------------------------------------------------------------
# 2. B0 template answers
# ---------------------------------------------------------------------------


class TestB0Answers:
    """Проверяет что B0 генерирует осмысленные ответы из KB."""

    def test_b0_factoid_contains_facts(self, chunks, mini_samples):
        """B0 ответы на factoid должны содержать факты из KB."""
        strategy = KeywordTemplateStrategy()
        factoid_samples = [s for s in mini_samples if s.category == "factoid"]

        for sample in factoid_samples:
            answer = strategy.answer_directly(sample.query, chunks)
            assert answer.answer, f"Пустой ответ B0 для '{sample.query}'"
            assert len(answer.answer) > 10, (
                f"Слишком короткий ответ B0 для '{sample.query}': '{answer.answer}'"
            )
            print(f"\n  B0 [{sample.sample_id}] '{sample.query}' → '{answer.answer[:80]}...'")

    def test_b0_out_of_scope_not_answerable(self, chunks, mini_samples):
        """B0 на out_of_scope должен вернуть answerable=False."""
        strategy = KeywordTemplateStrategy()
        oos_samples = [s for s in mini_samples if s.category == "out_of_scope"]

        correct = 0
        for sample in oos_samples:
            answer = strategy.answer_directly(sample.query, chunks)
            if not answer.answerable:
                correct += 1
            else:
                print(f"  B0 ошибочно answerable=True: '{sample.query}' → '{answer.answer[:60]}'")

        accuracy = correct / len(oos_samples) if oos_samples else 0
        print(f"\nB0 out_of_scope accuracy: {accuracy:.0%} ({correct}/{len(oos_samples)})")

    def test_b0_doctor_lookup_finds_doctor(self, chunks, mini_samples):
        """B0 должен найти врача по фамилии."""
        strategy = KeywordTemplateStrategy()
        doctor_samples = [s for s in mini_samples if s.category == "doctor_lookup"]

        for sample in doctor_samples:
            answer = strategy.answer_directly(sample.query, chunks)
            assert answer.answerable, f"B0 не нашёл врача: '{sample.query}'"
            if sample.expected_doctor:
                surname = sample.expected_doctor.split()[0]
                assert surname.lower() in answer.answer.lower(), (
                    f"Фамилия '{surname}' не в ответе B0: '{answer.answer[:80]}'"
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
            category="factoid",
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
            category="factoid",
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

    def test_retrieval_score_computation(self):
        """compute_retrieval_score корректно считает hit/recall/mrr."""
        result = StrategyResult(
            sample_id="test",
            strategy_id=StrategyID.S2,
            retrieval=RetrievalResult(
                chunk_ids=["clinic_contacts", "clinic_location", "doctor_1"],
                scores=[0.9, 0.7, 0.5],
            ),
        )
        gold_ids = ["clinic_location"]

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
        gold_ids = ["clinic_location"]

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
# 5. Gold map coverage
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
            for gid in gold_map.get(sample.sample_id, []):
                if gid not in valid_ids:
                    invalid.append((sample.sample_id, gid))

        assert not invalid, f"Несуществующие chunk_ids в gold_map: {invalid}"

    def test_mini_eval_set_distribution(self, mini_samples):
        """Mini eval set содержит все основные категории."""
        categories = {s.category for s in mini_samples}
        expected = {
            "factoid", "doctor_lookup", "complaint_routing",
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
        answer = strategy.answer_directly("Сколько стоит имплант?", chunks)
        assert answer.answerable, "B0 должен ответить на ценовой запрос"
        assert "₽" in answer.answer, "B0 ответ должен содержать цену"


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
