"""Масштабирование KB через nested distractor expansion.

Добавляет синтетические (нерелевантные) чанки к базовой KB,
сохраняя оригинальные факты. Для scale experiment (notebook 04).
"""

from __future__ import annotations

import random
from typing import Any

from d4.models import KBChunk
from d4.pipeline.chunker import count_tokens

# ---------------------------------------------------------------------------
# Шаблоны distractor-чанков (синтетические врачи и секции)
# ---------------------------------------------------------------------------

DISTRACTOR_TEMPLATES: list[dict[str, str]] = [
    {
        "title": "Врач-офтальмолог {name}",
        "content": (
            "{name} — офтальмолог, стаж {exp} лет. "
            "Клинические случаи: глаукома, катаракта, близорукость, дальнозоркость. "
            "Жалобы пациентов: плохо вижу, болят глаза, мушки перед глазами."
        ),
        "source_type": "doctors",
        "entity_type": "doctor",
    },
    {
        "title": "Врач-кардиолог {name}",
        "content": (
            "{name} — кардиолог, стаж {exp} лет. "
            "Клинические случаи: аритмия, гипертония, ишемическая болезнь. "
            "Жалобы пациентов: болит сердце, давление высокое, одышка."
        ),
        "source_type": "doctors",
        "entity_type": "doctor",
    },
    {
        "title": "Услуги косметологии",
        "content": (
            "Косметологические процедуры: ботокс, филлеры, "
            "пилинг, мезотерапия, лазерная эпиляция. Стоимость: по запросу."
        ),
        "source_type": "clinic_info",
        "entity_type": "faq_section",
    },
]

# Пул имён для distractor-врачей
FAKE_NAMES: list[str] = [
    "Петров Иван Сергеевич",
    "Козлова Мария Андреевна",
    "Новиков Алексей Петрович",
    "Федорова Анна Дмитриевна",
    "Волков Сергей Николаевич",
    "Орлова Елена Викторовна",
    "Зайцев Дмитрий Александрович",
    "Павлова Ольга Игоревна",
    "Соколов Андрей Владимирович",
    "Лебедева Татьяна Михайловна",
    "Григорьев Максим Олегович",
    "Кузьмина Наталья Сергеевна",
    "Романов Виктор Павлович",
    "Егорова Светлана Юрьевна",
    "Титов Николай Иванович",
    "Белова Ирина Александровна",
]


def generate_distractors(
    target_tokens: int,
    base_tokens: int,
    seed: int = 42,
) -> list[KBChunk]:
    """Генерация distractor-чанков до целевого размера KB.

    Args:
        target_tokens: целевой размер KB в токенах
        base_tokens: размер базовой KB в токенах
        seed: random seed для воспроизводимости

    Returns:
        список синтетических KBChunk
    """
    random.seed(seed)

    distractors: list[KBChunk] = []
    current_tokens = base_tokens
    idx = 0

    while current_tokens < target_tokens:
        template = random.choice(DISTRACTOR_TEMPLATES)
        name = FAKE_NAMES[idx % len(FAKE_NAMES)]
        exp = random.randint(1, 30)

        title = template["title"].format(name=name, exp=exp)
        content = template["content"].format(name=name, exp=exp)
        tokens = count_tokens(content)

        chunk = KBChunk(
            id=f"distractor_{idx:04d}",
            title=title,
            content=content,
            source="synthetic",
            source_type=template["source_type"],
            entity_type=template["entity_type"],
            entity_id=f"distractor_{idx}",
            token_count=tokens,
        )
        distractors.append(chunk)
        current_tokens += tokens
        idx += 1

    return distractors


def build_scaled_kbs(
    base_chunks: list[KBChunk],
    sizes_tokens: list[int],
    seed: int = 42,
) -> dict[int, list[KBChunk]]:
    """Генерация KB разных размеров для scale experiment.

    Args:
        base_chunks: базовые chunks из KB
        sizes_tokens: целевые размеры (напр. [5000, 15000, 30000, 50000])
        seed: random seed

    Returns:
        словарь {target_tokens: list[KBChunk]}
    """
    base_tokens = sum(c.token_count for c in base_chunks)
    scale_kbs: dict[int, list[KBChunk]] = {}

    for target in sizes_tokens:
        if target <= base_tokens:
            # Базовая KB уже больше целевого размера
            scale_kbs[target] = list(base_chunks)
        else:
            distractors = generate_distractors(target, base_tokens, seed=seed)
            scale_kbs[target] = list(base_chunks) + distractors

        actual = sum(c.token_count for c in scale_kbs[target])
        print(
            f"KB {target:>6d} tokens: {len(scale_kbs[target]):>4d} chunks, "
            f"actual {actual} tokens"
        )

    return scale_kbs
