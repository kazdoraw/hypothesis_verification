"""Масштабирование KB через nested distractor expansion.

Добавляет синтетические (нерелевантные) чанки к базовой KB,
сохраняя оригинальные факты. Для scale experiment (notebook 04).

Дистракторы 3-х уровней сложности (устранение Bias 6):
- L1 (умеренный): другие стоматологические клиники
- L2 (сложный): похожие стоматологические услуги с другими ценами
- L3 (шумовой): близкие мед. направления (ЛОР, ЧЛХ)

Все дистракторы имеют корректные source_type/entity_type/raw_data,
идентичные по структуре реальным чанкам.
"""

from __future__ import annotations

import random
from typing import Any

from d4.models import KBChunk
from d4.pipeline.chunker import count_tokens


# ---------------------------------------------------------------------------
# L1: Другие стоматологические клиники (аналогичная структура clinic_info)
# ---------------------------------------------------------------------------

_L1_CLINICS: list[dict[str, Any]] = [
    {
        "name": "СтомаПлюс",
        "city": "Ульяновск",
        "address": "ул. Гончарова, 12",
        "phone": "+7 (8422) 33-44-55",
        "schedule": "Пн-Сб 8:00-20:00, Вс — выходной",
        "sedation": "Лечение под закисью азота. Общий наркоз не предоставляется.",
        "children": "Приём детей с 5 лет.",
    },
    {
        "name": "Дентал Эксперт",
        "city": "Самара",
        "address": "пр-т Ленина, 88",
        "phone": "+7 (846) 200-30-40",
        "schedule": "Пн-Пт 9:00-21:00, Сб 10:00-16:00",
        "sedation": "Седация для детей и взрослых. Dental sleep.",
        "children": "Приём детей с 3 лет.",
    },
    {
        "name": "Клиника 32",
        "city": "Казань",
        "address": "ул. Баумана, 45",
        "phone": "+7 (843) 555-66-77",
        "schedule": "Ежедневно 8:00-22:00",
        "sedation": "Закись азота, седация в/в.",
        "children": "Детский стоматолог с 2 лет.",
    },
    {
        "name": "Здоровая улыбка",
        "city": "Нижний Новгород",
        "address": "ул. Минина, 3А",
        "phone": "+7 (831) 444-55-66",
        "schedule": "Пн-Пт 8:30-20:00",
        "sedation": "Не предоставляется.",
        "children": "Приём детей с 6 лет.",
    },
]

# Стоматологические специализации для генерации врачей-дистракторов
_DENTAL_SPECIALIZATIONS: list[tuple[str, str]] = [
    ("Хирургическая стоматология", "хирург"),
    ("Терапевтическая стоматология", "терапевт"),
    ("Ортопедическая стоматология", "ортопед"),
    ("Ортодонтия", "ортодонт"),
    ("Детская стоматология", "детский стоматолог"),
    ("Пародонтология", "пародонтолог"),
]

# Пул ФИО для стоматологов-дистракторов
_DENTAL_NAMES: list[str] = [
    "Козлов Артём Петрович",
    "Семёнова Дарья Андреевна",
    "Никитин Олег Сергеевич",
    "Фролова Екатерина Витальевна",
    "Жуков Максим Дмитриевич",
    "Тихонова Алина Юрьевна",
    "Макаров Денис Игоревич",
    "Савельева Ольга Вячеславовна",
    "Власов Кирилл Алексеевич",
    "Данилова Анна Романовна",
    "Рыбаков Андрей Николаевич",
    "Беляева Ирина Степановна",
    "Морозов Пётр Евгеньевич",
    "Комарова Наталья Сергеевна",
    "Поляков Виталий Иванович",
    "Крылова Елена Александровна",
]


# ---------------------------------------------------------------------------
# L2: Стоматологические услуги с другими ценами
# ---------------------------------------------------------------------------

_L2_PRICE_CATEGORIES: list[dict[str, Any]] = [
    {
        "category": "Имплантация",
        "label_ru": "Имплантация, установка имплантов",
        "services": [
            {"Service": "Имплантат Straumann (Швейцария)", "Price": "от 55 000 ₽"},
            {"Service": "Имплантат Nobel Biocare (Швеция)", "Price": "от 60 000 ₽"},
            {"Service": "Имплантат Osstem (Корея)", "Price": "от 28 000 ₽"},
            {"Service": "Синус-лифтинг открытый", "Price": "от 45 000 ₽"},
            {"Service": "Костная пластика", "Price": "от 35 000 ₽"},
        ],
    },
    {
        "category": "Протезирование",
        "label_ru": "Протезирование, коронки, виниры, протезы",
        "services": [
            {"Service": "Коронка металлокерамика", "Price": "от 10 000 ₽"},
            {"Service": "Коронка E-max", "Price": "от 22 000 ₽"},
            {"Service": "Виниры керамические (за ед.)", "Price": "от 25 000 ₽"},
            {"Service": "Съёмный протез полный", "Price": "от 30 000 ₽"},
            {"Service": "Мостовидный протез (3 ед.)", "Price": "от 35 000 ₽"},
        ],
    },
    {
        "category": "Терапия",
        "label_ru": "Терапия, лечение зубов, гигиена полости рта",
        "services": [
            {"Service": "Лечение кариеса (1 поверхность)", "Price": "от 3 500 ₽"},
            {"Service": "Лечение пульпита (1 канал)", "Price": "от 6 000 ₽"},
            {"Service": "Проф. гигиена (ультразвук + Air Flow)", "Price": "от 5 500 ₽"},
            {"Service": "Реставрация зуба", "Price": "от 5 000 ₽"},
            {"Service": "Герметизация фиссур (1 зуб)", "Price": "от 2 500 ₽"},
        ],
    },
    {
        "category": "Хирургия",
        "label_ru": "Хирургия, удаление зубов",
        "services": [
            {"Service": "Удаление простое", "Price": "от 2 500 ₽"},
            {"Service": "Удаление сложное", "Price": "от 5 000 ₽"},
            {"Service": "Удаление зуба мудрости", "Price": "от 8 000 ₽"},
            {"Service": "Пластика десны", "Price": "от 7 000 ₽"},
        ],
    },
    {
        "category": "Ортодонтия",
        "label_ru": "Ортодонтия, брекеты, выравнивание зубов",
        "services": [
            {"Service": "Брекеты металлические (1 челюсть)", "Price": "от 30 000 ₽"},
            {"Service": "Брекеты керамические (1 челюсть)", "Price": "от 45 000 ₽"},
            {"Service": "Элайнеры (полный курс)", "Price": "от 120 000 ₽"},
            {"Service": "Ретейнер несъёмный", "Price": "от 5 000 ₽"},
        ],
    },
]


# ---------------------------------------------------------------------------
# L3: Близкие мед. направления (ЛОР, ЧЛХ)
# ---------------------------------------------------------------------------

_L3_TEMPLATES: list[dict[str, str]] = [
    {
        "title": "ЛОР-врач {name}",
        "content": (
            "{name} — врач-оториноларинголог, стаж {exp} лет. "
            "Специализация: лечение заболеваний уха, горла и носа. "
            "Клинические случаи: тонзиллит, синусит, отит, аденоиды. "
            "Методы: эндоскопия, аудиометрия, промывание пазух."
        ),
        "source_type": "doctors",
        "entity_type": "doctor",
    },
    {
        "title": "Челюстно-лицевой хирург {name}",
        "content": (
            "{name} — челюстно-лицевой хирург, стаж {exp} лет. "
            "Специализация: реконструктивная хирургия лица и челюсти. "
            "Клинические случаи: переломы челюсти, дисфункция ВНЧС, "
            "опухоли челюстно-лицевой области. Оперирует под общим наркозом."
        ),
        "source_type": "doctors",
        "entity_type": "doctor",
    },
    {
        "title": "Рекомендации после ЛОР-процедур",
        "content": (
            "После промывания пазух: не сморкаться 2 часа, "
            "избегать переохлаждения 24 часа, капли по назначению. "
            "После тонзиллэктомии: щадящая диета 7-10 дней, "
            "полоскание Мирамистином, контроль через 3 дня."
        ),
        "source_type": "aftercare",
        "entity_type": "aftercare_recommendation",
    },
]


# ---------------------------------------------------------------------------
# Генераторы дистракторов
# ---------------------------------------------------------------------------

def _gen_l1_clinic_chunks(
    clinic: dict[str, Any],
    rng: random.Random,
    start_idx: int,
) -> list[KBChunk]:
    """L1: Чанки клиники-дистрактора (clinic_info + врачи)."""
    chunks: list[KBChunk] = []
    name = clinic["name"]

    sections = [
        (
            f"Клиника «{name}» расположена по адресу: {clinic['city']}, "
            f"{clinic['address']}. Телефон: {clinic['phone']}.",
            "location",
        ),
        (f"Режим работы «{name}»: {clinic['schedule']}.", "working_hours"),
        (f"Седация в «{name}»: {clinic['sedation']}", "patient_comfort_and_safety"),
        (f"{clinic['children']}", "children_info"),
    ]

    for text, section_key in sections:
        cid = f"distractor_{start_idx + len(chunks):04d}"
        chunks.append(KBChunk(
            id=cid,
            title=f"{name}: {section_key}",
            content=text,
            source="synthetic",
            source_type="clinic_info",
            entity_type="faq_section",
            entity_id=section_key,
            token_count=count_tokens(text),
            raw_data={"clinic": name, "section": section_key},
        ))

    # 2-4 врача для этой клиники
    n_docs = rng.randint(2, 4)
    for _ in range(n_docs):
        doc_name = rng.choice(_DENTAL_NAMES)
        spec, role = rng.choice(_DENTAL_SPECIALIZATIONS)
        exp = rng.randint(3, 25)
        content = (
            f"{doc_name} — врач-{role} ({spec}), стаж {exp} лет. "
            f"Клиника «{name}»."
        )
        cid = f"distractor_{start_idx + len(chunks):04d}"
        chunks.append(KBChunk(
            id=cid,
            title=f"{doc_name} — {role}",
            content=content,
            source="synthetic",
            source_type="doctors",
            entity_type="doctor",
            entity_id=doc_name.split()[0].lower(),
            token_count=count_tokens(content),
            raw_data={
                "full_name": doc_name,
                "specialization": spec,
                "experience_years": exp,
                "clinic": name,
            },
        ))

    return chunks


def _gen_l2_price_chunks(
    rng: random.Random,
    start_idx: int,
    n_categories: int = 3,
) -> list[KBChunk]:
    """L2: Прайс-дистракторы (те же услуги, другие цены)."""
    chunks: list[KBChunk] = []
    categories = rng.sample(_L2_PRICE_CATEGORIES, min(n_categories, len(_L2_PRICE_CATEGORIES)))

    for cat in categories:
        items = [f"- {s['Service']} — {s['Price']}" for s in cat["services"]]
        content = f"Прайс-лист: {cat['label_ru']}.\n" + "\n".join(items)
        slug = cat["category"].lower().replace(" ", "_")

        cid = f"distractor_{start_idx + len(chunks):04d}"
        chunks.append(KBChunk(
            id=cid,
            title=f"Цены: {cat['label_ru']}",
            content=content,
            source="synthetic",
            source_type="price_list",
            entity_type="price_category",
            entity_id=slug,
            token_count=count_tokens(content),
            raw_data={"category": cat["category"], "services": cat["services"]},
        ))

    return chunks


def _gen_l3_noise_chunks(
    rng: random.Random,
    start_idx: int,
    n_chunks: int = 3,
) -> list[KBChunk]:
    """L3: Шумовые дистракторы (ЛОР, ЧЛХ)."""
    chunks: list[KBChunk] = []

    for _ in range(n_chunks):
        template = rng.choice(_L3_TEMPLATES)
        name = rng.choice(_DENTAL_NAMES)
        exp = rng.randint(5, 30)

        title = template["title"].format(name=name, exp=exp)
        content = template["content"].format(name=name, exp=exp)

        cid = f"distractor_{start_idx + len(chunks):04d}"
        chunks.append(KBChunk(
            id=cid,
            title=title,
            content=content,
            source="synthetic",
            source_type=template["source_type"],
            entity_type=template["entity_type"],
            entity_id=f"l3_{len(chunks)}",
            token_count=count_tokens(content),
            raw_data={"level": "L3", "template": template["title"]},
        ))

    return chunks


# ---------------------------------------------------------------------------
# Публичный API
# ---------------------------------------------------------------------------

def generate_distractors(
    target_tokens: int,
    base_tokens: int,
    seed: int = 42,
) -> list[KBChunk]:
    """Генерация domain-specific дистракторов до целевого размера KB.

    Пропорции уровней: L1 ~50%, L2 ~30%, L3 ~20%.

    Args:
        target_tokens: целевой размер KB в токенах
        base_tokens: размер базовой KB в токенах
        seed: random seed для воспроизводимости

    Returns:
        список синтетических KBChunk
    """
    rng = random.Random(seed)
    distractors: list[KBChunk] = []
    current_tokens = base_tokens
    clinic_idx = 0

    # Ротация уровней: L1 → L2 → L3 → L1 → ...
    level_cycle = 0
    while current_tokens < target_tokens:
        idx = len(distractors)
        level = level_cycle % 3

        if level == 0:
            # L1: клиника-дистрактор (с ротацией по списку)
            clinic = _L1_CLINICS[clinic_idx % len(_L1_CLINICS)]
            new_chunks = _gen_l1_clinic_chunks(clinic, rng, idx)
            clinic_idx += 1
        elif level == 1:
            # L2: прайс-дистракторы
            new_chunks = _gen_l2_price_chunks(rng, idx, n_categories=2)
        else:
            # L3: шумовые (ЛОР, ЧЛХ)
            new_chunks = _gen_l3_noise_chunks(rng, idx, n_chunks=2)

        distractors.extend(new_chunks)
        current_tokens += sum(c.token_count for c in new_chunks)
        level_cycle += 1

    return distractors


def build_scaled_kbs(
    base_chunks: list[KBChunk],
    sizes_tokens: list[int],
    seed: int = 42,
) -> dict[int, list[KBChunk]]:
    """Генерация KB разных размеров для scale experiment.

    Args:
        base_chunks: базовые chunks из KB
        sizes_tokens: целевые размеры (напр. [15000, 30000, 50000, 100000])
        seed: random seed

    Returns:
        словарь {target_tokens: list[KBChunk]}
    """
    base_tokens = sum(c.token_count for c in base_chunks)
    scale_kbs: dict[int, list[KBChunk]] = {}

    for target in sizes_tokens:
        if target <= base_tokens:
            scale_kbs[target] = list(base_chunks)
        else:
            distractors = generate_distractors(target, base_tokens, seed=seed)
            scale_kbs[target] = list(base_chunks) + distractors

        actual = sum(c.token_count for c in scale_kbs[target])
        n_distractors = len(scale_kbs[target]) - len(base_chunks)
        print(
            f"KB {target:>7,d} tok: {len(scale_kbs[target]):>4d} chunks "
            f"(+{n_distractors} distractors), actual {actual:,} tok"
        )

    return scale_kbs
