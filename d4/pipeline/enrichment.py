"""Stage 2A: обогащение текста чанков для retrieval-side эксперимента.

Три режима (C0/C1/C2):
- plain: без обогащения (baseline, Stage 1)
- contextual: детерминированный context_prefix (C1)
- llm_enriched: LLM-сгенерированный enrichment_prefix (C2)

Разделение ответственности:
- chunker.py → парсинг и чанкование KB
- enrichment.py → обогащение текста для embedding
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from d4.models import KBChunk

if TYPE_CHECKING:
    from d4.pipeline.llm_runner import LLMRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# C1: детерминированный контекстный заголовок
# ---------------------------------------------------------------------------

_SOURCE_TYPE_RU: dict[str, str] = {
    "clinic_info": "информация о клинике",
    "doctors": "информация о врачах",
    "price_list": "прайс-лист",
    "aftercare": "рекомендации после лечения",
}

_CLINIC_META = {
    "clinic_name": "Семейная клиника «Клевер»",
    "city": "Ульяновск",
}


def build_context_prefix(chunk: KBChunk) -> str:
    """C1: фиксированный шаблон — одинаковый для всех чанков, без LLM.

    Формат строго детерминирован: clinic_name + city + source_type + title.
    """
    doc_type = _SOURCE_TYPE_RU.get(chunk.source_type, chunk.source_type)
    return (
        f"Источник: {_CLINIC_META['clinic_name']}, {_CLINIC_META['city']}.\n"
        f"Тип документа: {doc_type}.\n"
        f"Раздел: {chunk.title}."
    )


# ---------------------------------------------------------------------------
# C2: LLM-сгенерированный контекстный префикс (class-aware, strict)
# ---------------------------------------------------------------------------

# Один шаблон на doc-class. Ограничиваем формат:
# - 1 строка «Контекст» (что за объект)
# - 1-2 intent hints (для какого поискового запроса этот чанк полезен)
# Без «вопросов пациентов», без переписывания фрагмента.

_C2_CLASS_PROMPTS: dict[str, str] = {
    "doctors": """\
Дан фрагмент карточки врача стоматологической клиники.

Фрагмент:
---
{chunk_text}
---

Ответь строго в формате (2 строки, ничего больше):
Контекст: <1 предложение — ФИО, специальность, ключевая экспертиза>
Интент: <1-2 типичных поисковых запроса пациента, через запятую>""",

    "price_list": """\
Дан фрагмент прайс-листа стоматологической клиники.

Фрагмент:
---
{chunk_text}
---

Ответь строго в формате (2 строки, ничего больше):
Контекст: <1 предложение — какие услуги/процедуры описаны>
Интент: <1-2 типичных поисковых запроса пациента, через запятую>""",

    "aftercare": """\
Дан фрагмент рекомендаций после лечения стоматологической клиники.

Фрагмент:
---
{chunk_text}
---

Ответь строго в формате (2 строки, ничего больше):
Контекст: <1 предложение — после какой процедуры эти рекомендации>
Интент: <1-2 типичных поисковых запроса пациента, через запятую>""",

    "clinic_info": """\
Дан фрагмент общей информации стоматологической клиники.

Фрагмент:
---
{chunk_text}
---

Ответь строго в формате (2 строки, ничего больше):
Контекст: <1 предложение — какой аспект работы клиники описан>
Интент: <1-2 типичных поисковых запроса пациента, через запятую>""",
}

_C2_FALLBACK_PROMPT = """\
Дан фрагмент из базы знаний стоматологической клиники.

Фрагмент:
---
{chunk_text}
---

Ответь строго в формате (2 строки, ничего больше):
Контекст: <1 предложение — о чём этот фрагмент>
Интент: <1-2 типичных поисковых запроса пациента, через запятую>"""


def build_enrichment_prefix(
    chunk: KBChunk,
    llm_runner: LLMRunner,
    temperature: float = 0.0,
    max_tokens: int = 64,
) -> str:
    """C2: class-aware LLM-generated prefix. Raw chunk text НЕ переписывается.

    Выбирает prompt-шаблон по source_type чанка, что даёт более
    таргетированный enrichment для каждого doc-class.
    max_tokens=64 ограничивает output строго до 2 строк.
    """
    tpl = _C2_CLASS_PROMPTS.get(chunk.source_type, _C2_FALLBACK_PROMPT)
    prompt = tpl.format(chunk_text=f"{chunk.title}\n{chunk.content}")

    response = llm_runner.client.chat.completions.create(
        model=llm_runner.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Batch-обогащение
# ---------------------------------------------------------------------------


def enrich_chunks(
    chunks: list[KBChunk],
    mode: Literal["plain", "contextual", "llm_enriched"] = "plain",
    llm_runner: LLMRunner | None = None,
    c2_temperature: float = 0.0,
    c2_max_tokens: int = 64,
) -> list[KBChunk]:
    """Обогащает чанки по выбранному режиму. Возвращает НОВЫЕ объекты.

    Args:
        chunks: исходные чанки (не мутируются)
        mode: plain (C0), contextual (C1), llm_enriched (C2)
        llm_runner: необходим только для mode=llm_enriched
        c2_temperature: temperature для C2 LLM-вызова
        c2_max_tokens: max_tokens для C2 LLM-вызова

    Returns:
        список новых KBChunk с заполненными prefix-полями
    """
    if mode == "plain":
        return chunks

    enriched: list[KBChunk] = []
    total = len(chunks)

    for i, chunk in enumerate(chunks):
        new_chunk = chunk.model_copy()

        if mode == "contextual":
            new_chunk.context_prefix = build_context_prefix(chunk)

        elif mode == "llm_enriched":
            if llm_runner is None:
                raise ValueError("llm_runner обязателен для mode='llm_enriched'")
            new_chunk.context_prefix = build_context_prefix(chunk)
            new_chunk.enrichment_prefix = build_enrichment_prefix(
                chunk, llm_runner, temperature=c2_temperature, max_tokens=c2_max_tokens,
            )
            logger.info("C2 enrichment: %d/%d — %s", i + 1, total, chunk.id)

        enriched.append(new_chunk)

    logger.info("Enrichment завершён: mode=%s, chunks=%d", mode, total)
    return enriched


# ---------------------------------------------------------------------------
# Кэширование enrichment (фиксированный артефакт)
# ---------------------------------------------------------------------------

# Prompt version — инкрементируется при любом изменении _C2_CLASS_PROMPTS
# или _C2_FALLBACK_PROMPT, чтобы гарантировать cache invalidation.
_PROMPT_VERSION = "v2-classaware-strict64"


def _prompts_hash() -> str:
    """SHA256 от всех prompt-шаблонов — автоматический fingerprint."""
    blob = json.dumps(
        {**_C2_CLASS_PROMPTS, "__fallback__": _C2_FALLBACK_PROMPT},
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def _content_hash(chunks: list[KBChunk]) -> str:
    """SHA256 от id+title+content всех чанков (отсортированных по id)."""
    parts = []
    for c in sorted(chunks, key=lambda x: x.id):
        parts.append(f"{c.id}|{c.title}|{c.content}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def _cache_key(
    mode: str,
    chunks: list[KBChunk],
    c2_max_tokens: int,
    c2_temperature: float,
    llm_model: str,
) -> str:
    """Полный fingerprint: mode + content + params + prompt + model."""
    components = "|".join([
        mode,
        _content_hash(chunks),
        str(c2_max_tokens),
        f"{c2_temperature:.2f}",
        _PROMPT_VERSION,
        _prompts_hash(),
        llm_model,
    ])
    return hashlib.sha256(components.encode()).hexdigest()[:16]


def _cache_path(
    cache_dir: Path,
    mode: str,
    chunks: list[KBChunk],
    c2_max_tokens: int = 64,
    c2_temperature: float = 0.0,
    llm_model: str = "",
) -> Path:
    key = _cache_key(mode, chunks, c2_max_tokens, c2_temperature, llm_model)
    return cache_dir / f"enrichment_{mode}_{key}.json"


def save_enrichment_cache(
    chunks: list[KBChunk],
    mode: str,
    cache_dir: Path,
    c2_max_tokens: int = 64,
    c2_temperature: float = 0.0,
    llm_model: str = "",
) -> Path:
    """Сохраняет context_prefix / enrichment_prefix в JSON-артефакт.

    Файл содержит metadata-header + records, позволяя валидировать
    при загрузке, что cache создан с теми же параметрами.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, mode, chunks, c2_max_tokens, c2_temperature, llm_model)

    records = []
    for c in chunks:
        rec: dict = {"id": c.id}
        if c.context_prefix:
            rec["context_prefix"] = c.context_prefix
        if c.enrichment_prefix:
            rec["enrichment_prefix"] = c.enrichment_prefix
        records.append(rec)

    artifact = {
        "meta": {
            "mode": mode,
            "prompt_version": _PROMPT_VERSION,
            "prompts_hash": _prompts_hash(),
            "c2_max_tokens": c2_max_tokens,
            "c2_temperature": c2_temperature,
            "llm_model": llm_model,
            "n_chunks": len(chunks),
            "content_hash": _content_hash(chunks),
        },
        "records": records,
    }
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Enrichment cache saved: %s (%d chunks)", path, len(records))
    return path


def load_enrichment_cache(
    chunks: list[KBChunk],
    mode: str,
    cache_dir: Path,
    c2_max_tokens: int = 64,
    c2_temperature: float = 0.0,
    llm_model: str = "",
) -> list[KBChunk] | None:
    """Загружает enrichment из кэша. Возвращает None если кэш не найден
    или fingerprint не совпадает."""
    path = _cache_path(cache_dir, mode, chunks, c2_max_tokens, c2_temperature, llm_model)
    if not path.exists():
        return None

    raw = json.loads(path.read_text(encoding="utf-8"))

    # Backward compat: старый формат без meta
    if "meta" in raw and "records" in raw:
        meta = raw["meta"]
        if meta.get("content_hash") != _content_hash(chunks):
            logger.warning("Cache content_hash mismatch — invalidated")
            return None
        records = raw["records"]
    else:
        records = raw if isinstance(raw, list) else raw.get("records", [])

    prefix_map = {r["id"]: r for r in records}

    enriched: list[KBChunk] = []
    for chunk in chunks:
        cached = prefix_map.get(chunk.id)
        if cached is None:
            logger.warning("Cache miss for chunk %s — cache invalidated", chunk.id)
            return None
        new_chunk = chunk.model_copy()
        new_chunk.context_prefix = cached.get("context_prefix", "")
        new_chunk.enrichment_prefix = cached.get("enrichment_prefix", "")
        enriched.append(new_chunk)

    logger.info("Enrichment cache loaded: %s (%d chunks)", path, len(enriched))
    return enriched
