"""Логический chunking KB → retrieval units.

Принцип: 1 врач = 1 unit, 1 тематическая секция clinic_info = 1 unit.
Не token-based дробление, а семантически целостные блоки.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer

from d4.models import KBChunk

# Токенизатор Qwen3.5 для точного подсчёта токенов (загружается однократно)
_TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")

# Секции clinic_info, которые становятся отдельными retrieval units
CLINIC_SECTION_MAP: dict[str, str] = {
    "summary": "Общая информация о клинике",
    "location": "Адрес и местоположение",
    "contacts": "Контакты клиники",
    "working_hours": "Режим работы",
    "patient_comfort_and_safety": "Комфорт и безопасность (седация)",
    "children_info": "Приём детей",
    "financing": "Оплата и рассрочка",
    "documents": "Документы для пациентов",
    "licenses": "Лицензии",
    "requisites": "Реквизиты",
    "supervisory_bodies": "Надзорные органы",
    "website": "Сайт клиники",
    "faq": "Часто задаваемые вопросы",
}


def count_tokens(text: str) -> int:
    """Подсчёт токенов в тексте (Qwen3.5 tokenizer)."""
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def _serialize_value(value: Any, indent: int = 0) -> str:
    """Рекурсивная сериализация значения в читаемый текст."""
    prefix = "  " * indent
    if isinstance(value, dict):
        lines = []
        for k, v in value.items():
            serialized = _serialize_value(v, indent + 1)
            if "\n" in serialized:
                lines.append(f"{prefix}{k}:\n{serialized}")
            else:
                lines.append(f"{prefix}{k}: {serialized}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = [f"{prefix}- {_serialize_value(item, 0)}" for item in value]
        return "\n".join(lines)
    return str(value)


# ---------------------------------------------------------------------------
# Natural Language сериализация секций clinic_info
# Каждая lambda принимает dict секции и возвращает NL-текст.
# ---------------------------------------------------------------------------

def _join_items(items: list[dict], key: str = "name") -> str:
    """Список dict-элементов → перечисление через '; '."""
    return "; ".join(item.get(key, "") for item in items if item.get(key))


def _clinic_section_to_text(section_key: str, data: Any) -> str:
    """Конвертация секции clinic_info в natural language текст."""
    if isinstance(data, dict):
        d = data
    else:
        return str(data)

    if section_key == "meta":
        name = d.get("name", "")
        brand = d.get("brand_name", "")
        return f"Название клиники: {name} ({brand})." if brand else f"Название клиники: {name}."

    if section_key == "summary":
        desc = d.get("description", "")
        msgs = d.get("key_messages", [])
        text = desc.strip()
        if msgs:
            text += f" Ключевые направления: {', '.join(msgs)}."
        return text

    if section_key == "location":
        return (
            f"Клиника расположена по адресу: Россия, "
            f"г. {d.get('city', '')}, {d.get('address', '')}."
        )

    if section_key == "contacts":
        phones = d.get("phones", [])
        emails = d.get("emails", [])
        parts = []
        if phones:
            parts.append(f"Телефоны клиники: {', '.join(phones)}")
        if emails:
            parts.append(f"Email: {', '.join(emails)}")
        return ". ".join(parts) + "." if parts else ""

    if section_key == "working_hours":
        return f"Режим работы: {d.get('schedule_text', '')}."

    if section_key == "patient_comfort_and_safety":
        sedation = d.get("sedation", {})
        nitrous = sedation.get("nitrous_oxide", {})
        parts = []
        if sedation.get("dental_sleep"):
            parts.append("лечение во сне (dental sleep)")
        if nitrous.get("available"):
            mixture = nitrous.get("mixture", "закись азота")
            purpose = nitrous.get("purpose", "")
            parts.append(f"{mixture} — {purpose}")
        return f"Седация: {'; '.join(parts)}." if parts else ""

    if section_key == "children_info":
        accepted = "Да" if d.get("accepted") else "Нет"
        min_age = d.get("min_age", "")
        note = d.get("note", "")
        text = f"Приём детей: {accepted.lower()}, {min_age}."
        if note:
            text += f" {note}."
        return text

    if section_key == "financing":
        inst = d.get("installments", d)
        ftype = inst.get("type", "рассрочка")
        months = inst.get("months", [])
        if months:
            return f"Доступна {ftype} на {', '.join(str(m) for m in months)} месяцев."
        return f"Доступна {ftype}."

    if section_key == "documents":
        url = d.get("page_url", "")
        items = d.get("items", [])
        text = f"Документы для пациентов: {url}."
        if items:
            names = [it.get("name", "") for it in items if it.get("name")]
            text += f" Доступные документы: {'; '.join(names)}."
        return text

    if section_key == "licenses":
        url = d.get("page_url", "")
        items = d.get("items", [])
        text = f"Лицензии клиники: {url}."
        if items:
            text += f" {_join_items(items)}."
        return text

    if section_key == "requisites":
        url = d.get("page_url", "")
        note = d.get("note", "")
        return f"Реквизиты клиники: {url}. {note}".strip()

    if section_key == "supervisory_bodies":
        url = d.get("page_url", "")
        items = d.get("items", [])
        text = f"Надзорные органы: {url}."
        if items:
            text += f" {_join_items(items)}."
        return text

    if section_key == "website":
        base = d.get("base_url", "")
        pages = d.get("pages", {})
        text = f"Сайт клиники: {base}."
        if pages:
            page_list = [f"{k}: {v}" for k, v in pages.items()]
            text += f" Страницы: {'; '.join(page_list)}."
        return text

    if section_key == "faq":
        url = d.get("page_url", "")
        return f"Часто задаваемые вопросы: {url}."

    # Fallback для неизвестных секций
    return _serialize_value(data)


# Маппинг длинных специализаций → короткая роль (для BM25 stem-совместимости)
_SPEC_TO_ROLE: dict[str, str] = {
    "Хирургическая стоматология": "хирург",
    "Терапевтическая стоматология": "терапевт",
    "Ортопедическая стоматология": "ортопед",
    "Ортодонтия": "ортодонт",
    "Детская стоматология": "детский стоматолог",
    "Пародонтология": "пародонтолог",
    "Гнатология": "гнатолог",
}


def _doctor_to_text(doctor: dict) -> str:
    """Конвертация врача в natural language текст."""
    name = doctor.get("full_name", "")
    spec = doctor.get("specialization", "")
    exp = doctor.get("experience_years", 0)
    cases = ", ".join(doctor.get("clinical_cases", []))
    complaints = ", ".join(doctor.get("patient_complaints", []))
    diag = ", ".join(doctor.get("diagnostics_methods", []))

    role = _SPEC_TO_ROLE.get(spec, "")
    if role:
        parts = [f"{name} — врач-{role} ({spec}), стаж {exp} лет."]
    else:
        parts = [f"{name} — {spec}, стаж {exp} лет."]
    if cases:
        parts.append(f"Клинические случаи: {cases}.")
    if complaints:
        parts.append(f"Жалобы пациентов: {complaints}.")
    if diag:
        parts.append(f"Методы диагностики: {diag}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Chunking: KB → retrieval units (NL-формат + raw_data)
# ---------------------------------------------------------------------------


def chunk_clinic_info(clinic_data: dict) -> list[KBChunk]:
    """Разбивает clinic_info.yaml на логические секции.

    Каждая секция становится retrieval unit с NL-текстом в content
    и оригинальным dict в raw_data (для B0 rule-based стратегии).
    """
    clinic = clinic_data.get("clinic", clinic_data)
    chunks: list[KBChunk] = []

    # Мета-информация (id + name + brand)
    meta_fields = {k: clinic[k] for k in ("id", "name", "brand_name") if k in clinic}
    if meta_fields:
        content = _clinic_section_to_text("meta", meta_fields)
        chunks.append(KBChunk(
            id="clinic_meta",
            title="Название клиники",
            content=content,
            source="clinic_info.yaml",
            source_type="clinic_info",
            entity_type="faq_section",
            entity_id="meta",
            token_count=count_tokens(content),
            raw_data=meta_fields,
        ))

    # Тематические секции → NL text
    for section_key, section_title in CLINIC_SECTION_MAP.items():
        if section_key not in clinic:
            continue
        section_data = clinic[section_key]
        content = _clinic_section_to_text(section_key, section_data)
        if not content.strip():
            continue
        # raw_data: оригинальный dict (или оборачиваем скаляр)
        raw = section_data if isinstance(section_data, dict) else {"value": section_data}
        chunks.append(KBChunk(
            id=f"clinic_{section_key}",
            title=section_title,
            content=content,
            source="clinic_info.yaml",
            source_type="clinic_info",
            entity_type="faq_section",
            entity_id=section_key,
            token_count=count_tokens(content),
            raw_data=raw,
        ))

    return chunks


def chunk_doctors(doctors_data: dict) -> list[KBChunk]:
    """Разбивает doctors.yaml: 1 врач = 1 retrieval unit.

    content — NL-текст для embedding/BM25, raw_data — dict для B0.
    """
    doctors_list = doctors_data.get("doctors", [])
    chunks: list[KBChunk] = []

    for doctor in doctors_list:
        doc_id = doctor.get("id", "unknown")
        full_name = doctor.get("full_name", "Неизвестный врач")
        specialization = doctor.get("specialization", "")

        content = _doctor_to_text(doctor)
        chunks.append(KBChunk(
            id=f"doctor_{doc_id}",
            title=f"{full_name} — {specialization}",
            content=content,
            source="doctors.yaml",
            source_type="doctors",
            entity_type="doctor",
            entity_id=str(doc_id),
            token_count=count_tokens(content),
            raw_data=doctor,
        ))

    return chunks


# ---------------------------------------------------------------------------
# Chunking: прайс-лист → retrieval units (NL-формат + raw_data)
# Универсальный pipeline: группирует CSV по Category, не зависит от отрасли.
# ---------------------------------------------------------------------------


def _category_to_slug(category: str) -> str:
    """Категория CSV → slug для chunk_id. Не зависит от языка."""
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def _price_category_to_text(
    category: str,
    services: list[dict],
    label_ru: str | None = None,
) -> str:
    """Конвертация категории с услугами в NL-текст для retrieval.

    Формат: русскоязычный заголовок + услуги через newline.
    Если label_ru указан — используется как заголовок (лучше для BM25).
    """
    display = label_ru if label_ru else category
    items = [f"- {s['Service']} — {s['Price']}" for s in services]
    return f"Прайс-лист: {display}.\n" + "\n".join(items)


def chunk_prices(
    price_path: Path,
    max_chunk_tokens: int = 600,
    category_labels: dict[str, str] | None = None,
) -> list[KBChunk]:
    """Универсальный chunker для прайс-листа (CSV).

    Группирует по Category, генерирует NL-текст.
    Если категория > max_chunk_tokens — разбивает на sub-chunks.

    Args:
        price_path: путь к CSV (колонки: Category, Service, Price)
        max_chunk_tokens: макс. токенов в одном chunk (адаптивный split)
        category_labels: {EN_category: RU_label} — русские названия категорий.
            Если не указан — используется оригинальное название из CSV.
    """
    import pandas as pd

    df = pd.read_csv(price_path, encoding="utf-8", skipinitialspace=True)
    df = df.dropna(subset=["Category", "Service", "Price"])

    chunks: list[KBChunk] = []

    labels = category_labels or {}

    for category, group in df.groupby("Category", sort=False):
        services = group.to_dict("records")
        slug = _category_to_slug(str(category))
        label_ru = labels.get(str(category))
        display = label_ru if label_ru else str(category)
        content = _price_category_to_text(str(category), services, label_ru)
        tokens = count_tokens(content)

        if tokens <= max_chunk_tokens:
            chunks.append(KBChunk(
                id=f"price_{slug}",
                title=f"Цены: {display}",
                content=content,
                source=price_path.name,
                source_type="price_list",
                entity_type="price_category",
                entity_id=slug,
                token_count=tokens,
                raw_data={"category": str(category), "services": services},
            ))
        else:
            # Адаптивный split: делим услуги на sub-chunks
            part_size = len(services) // ((tokens // max_chunk_tokens) + 1) + 1
            for i in range(0, len(services), part_size):
                part = services[i:i + part_size]
                part_content = _price_category_to_text(str(category), part, label_ru)
                part_tokens = count_tokens(part_content)
                part_idx = i // part_size + 1
                chunks.append(KBChunk(
                    id=f"price_{slug}_p{part_idx}",
                    title=f"Цены: {display} (часть {part_idx})",
                    content=part_content,
                    source=price_path.name,
                    source_type="price_list",
                    entity_type="price_category",
                    entity_id=f"{slug}_p{part_idx}",
                    token_count=part_tokens,
                    raw_data={"category": str(category), "services": part},
                ))

    return chunks


def load_and_chunk_kb(kb_dir: str | Path) -> list[KBChunk]:
    """Загрузка KB и разбиение на логические retrieval units.

    Args:
        kb_dir: путь к папке с clinic_info.yaml, doctors.yaml и опционально CSV прайса

    Returns:
        список KBChunk (clinic секции + прайс + врачи)
    """
    kb_path = Path(kb_dir)

    # clinic_info.yaml
    clinic_path = kb_path / "clinic_info.yaml"
    with open(clinic_path, encoding="utf-8") as f:
        clinic_data = yaml.safe_load(f)
    clinic_chunks = chunk_clinic_info(clinic_data)

    # doctors.yaml
    doctors_path = kb_path / "doctors.yaml"
    with open(doctors_path, encoding="utf-8") as f:
        doctors_data = yaml.safe_load(f)
    doctor_chunks = chunk_doctors(doctors_data)

    # Прайс-лист CSV (опционально — если файл найден в kb_dir)
    # Русские названия категорий для BM25-совместимости.
    # Для другой отрасли — обновить этот маппинг или передать через параметр.
    _PRICE_CATEGORY_LABELS_RU: dict[str, str] = {
        "Sedation under nitrous oxide (children)": "Седация закисью азота (дети)",
        "Sedation under nitrous oxide (adults)": "Седация закисью азота (взрослые)",
        "Treatment under microscope": "Лечение под микроскопом",
        "Anesthesia": "Анестезия, обезболивание",
        "Certificates": "Сертификаты",
        "Orthodontics": "Ортодонтия, брекеты, выравнивание зубов",
        "ENT": "ЛОР, оториноларингология",
        "Sedation/Removal in Sleep (children)": "Седация и удаление во сне (дети)",
        "Maxillofacial surgery (adults)": "Челюстно-лицевая хирургия (взрослые)",
        "Primary appointment & consultations": "Первичный приём и консультации",
        "Diagnostics 2D/3D & Viziograf": "Диагностика, рентген, КТ, снимки зубов",
        "Children therapy": "Детская стоматология, лечение зубов у детей",
        "Adult therapy": "Терапия, лечение зубов, гигиена полости рта",
        "Adult surgery (removals)": "Хирургия, удаление зубов",
        "Implantation": "Имплантация, установка имплантов",
        "Prosthetics": "Протезирование, коронки, виниры, протезы",
        "Parodontology": "Пародонтология, лечение дёсен",
    }
    price_chunks: list[KBChunk] = []
    price_files = list(kb_path.glob("*prices*.csv")) + list(kb_path.glob("*price*.csv"))
    if price_files:
        price_chunks = chunk_prices(
            price_files[0],
            category_labels=_PRICE_CATEGORY_LABELS_RU,
        )

    all_chunks = clinic_chunks + price_chunks + doctor_chunks
    return all_chunks


def save_chunks(chunks: list[KBChunk], output_path: str | Path) -> None:
    """Сохранение chunks в JSON файл."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [chunk.model_dump() for chunk in chunks]
    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_chunks(chunks_path: str | Path) -> list[KBChunk]:
    """Загрузка chunks из JSON файла."""
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)
    return [KBChunk(**item) for item in data]


def print_chunk_summary(chunks: list[KBChunk]) -> None:
    """Вывод сводки по chunks."""
    total_tokens = sum(c.token_count for c in chunks)
    clinic_chunks = [c for c in chunks if c.source_type == "clinic_info"]
    doctor_chunks = [c for c in chunks if c.source_type == "doctors"]
    price_chunks = [c for c in chunks if c.source_type == "price_list"]

    print(f"Всего chunks: {len(chunks)}")
    print(f"  clinic_info: {len(clinic_chunks)} секций")
    if price_chunks:
        print(f"  price_list: {len(price_chunks)} категорий")
    print(f"  doctors: {len(doctor_chunks)} врачей")
    print(f"  общий размер: {total_tokens} токенов")
