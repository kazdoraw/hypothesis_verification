"""Парсинг aftercare-рекомендаций из .doc/.docx → YAML.

Переиспользуемый скрипт для конвертации стоматологических памяток
из Word-документов в структурированный YAML для KB.

Зависимости:
    pip install python-docx>=1.1 pyyaml

Использование:
    python -m d4.data_gen.parse_aftercare
    python -m d4.data_gen.parse_aftercare --input-dir d4/raw_data/kb/recomendations
    python -m d4.data_gen.parse_aftercare --output d4/raw_data/kb/aftercare_recommendations.yaml

Поддерживаемые форматы:
    .docx — напрямую через python-docx
    .doc  — конвертация через textutil (macOS) или libreoffice (Linux/Windows)
"""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from docx import Document

# ---------------------------------------------------------------------------
# Конфигурация: маппинг файлов → procedure_type / title
# Расширяется при добавлении новых документов в клинику.
# ---------------------------------------------------------------------------

# Файлы, содержащие несколько секций (разделяются по заголовкам внутри).
# Ключ = имя файла (без расширения, для матчинга .doc/.docx).
# Значение = список (regex_заголовка, procedure_type, display_title).
MULTI_SECTION_FILES: dict[str, list[tuple[str, str, str]]] = {
    "Рекомендации_после_хир_вмешательства": [
        (
            r"Рекомендации после удаления",
            "post_extraction",
            "Рекомендации после удаления зуба",
        ),
        (
            r"Рекомендации при сложном удалении",
            "post_implantation",
            "Рекомендации после имплантации и костной пластики",
        ),
        (
            r"Рекомендации после синус-лифтинга",
            "post_sinus_lift",
            "Рекомендации после синус-лифтинга",
        ),
    ],
}

# Файлы с одной секцией.
# Ключ = имя файла (без расширения), значение = (procedure_type, display_title).
SINGLE_SECTION_FILES: dict[str, tuple[str, str]] = {
    "Памятка пациенту после лечения в условиях общего наркоза": (
        "post_anesthesia",
        "Рекомендации после лечения под общим наркозом",
    ),
    "ПАМЯТКА_ортодонтическое лечение": (
        "post_orthodontics",
        "Памятка пациенту при ортодонтическом лечении (брекеты)",
    ),
    "Рекомендации несъемные протезы": (
        "post_prosthetics_fixed",
        "Рекомендации по уходу за несъёмными протезами (коронки, мосты)",
    ),
    "Рекомендации съемные протезы": (
        "post_prosthetics_removable",
        "Рекомендации по пользованию съёмными протезами",
    ),
    "Рекомендации гигиена": (
        "post_hygiene",
        "Рекомендации после профессиональной гигиены полости рта",
    ),
    "Рекомендации_гигиена": (
        "post_hygiene",
        "Рекомендации после профессиональной гигиены полости рта",
    ),
}

# Регулярные выражения для удаления шаблонных артефактов бумажных бланков.
_JUNK_PATTERNS: list[str] = [
    r"Дата\s*«[_\s]+»[_\s]*2\d{2}__?г\.?",
    r"Подпись\s+(пациента|врача|родителя|заказчика|законного).*?\(подпись,?\s*ФИО\)?",
    r"Второй\s+экземпляр\s+получил.*",
    r"Памятку\s+получил\(а\).*",
    r"Мною\s+заданы\s+все\s+интересующие.*?осложнений\.?",
    r"Я\s+обязуюсь\s+соблюдать.*?осложнений\.?",
    r"Повторное\s+проведение.*?рекомендовано:.*",
    r"_{4,}",
    r"^\d+\.\s*Другое:\s*$",
]

# Паттерн для дублирующего заголовка в начале content
_TITLE_PREFIX_RE = re.compile(
    r"^(ПАМЯТКА|Рекомендации|по\s+уходу|по\s+пользованию|полости\s+рта)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Модели данных
# ---------------------------------------------------------------------------


@dataclass
class AftercarRecommendation:
    """Одна aftercare-рекомендация для YAML."""

    procedure_type: str
    title: str
    source_file: str
    content: str


# ---------------------------------------------------------------------------
# Утилиты: конвертация и парсинг
# ---------------------------------------------------------------------------


def _convert_doc_to_docx(doc_path: Path, output_dir: Path | None = None) -> Path:
    """Конвертация .doc → .docx через системный инструмент.

    macOS: textutil (встроен).
    Linux/Windows: libreoffice --headless.

    Returns:
        путь к сконвертированному .docx файлу
    """
    if output_dir is None:
        output_dir = doc_path.parent

    docx_path = output_dir / (doc_path.stem + ".converted.docx")

    system = platform.system()

    if system == "Darwin" and shutil.which("textutil"):
        subprocess.run(
            ["textutil", "-convert", "docx", str(doc_path), "-output", str(docx_path)],
            check=True,
            capture_output=True,
        )
    elif shutil.which("libreoffice"):
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    "libreoffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    tmp,
                    str(doc_path),
                ],
                check=True,
                capture_output=True,
            )
            converted = Path(tmp) / (doc_path.stem + ".docx")
            shutil.copy2(converted, docx_path)
    else:
        raise RuntimeError(
            f"Нет инструмента для конвертации .doc: "
            f"textutil (macOS) или libreoffice не найдены. "
            f"Сконвертируйте {doc_path.name} в .docx вручную."
        )

    return docx_path


def _read_docx_paragraphs(path: Path) -> list[str]:
    """Извлечение непустых параграфов из .docx."""
    doc = Document(str(path))
    return [p.text.strip() for p in doc.paragraphs if p.text.strip()]


def _read_document(path: Path) -> list[str]:
    """Чтение .doc или .docx → список параграфов.

    .doc файлы конвертируются во временный .docx.
    """
    if path.suffix.lower() == ".docx":
        return _read_docx_paragraphs(path)

    if path.suffix.lower() == ".doc":
        docx_path = _convert_doc_to_docx(path)
        try:
            return _read_docx_paragraphs(docx_path)
        finally:
            docx_path.unlink(missing_ok=True)

    raise ValueError(f"Неподдерживаемый формат: {path.suffix}")


# ---------------------------------------------------------------------------
# Очистка текста
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Удаление шаблонных артефактов бумажных бланков."""
    for pat in _JUNK_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.DOTALL)

    # Tab → пробелы (PyYAML не поддерживает \t в literal block style)
    text = text.replace("\t", " ")
    # Буллеты: ○ /● /■  → •
    text = re.sub(r"[○●■]\s*", "• ", text)
    # Нумерованные списки с лишними пробелами: "2.  " → "2. "
    text = re.sub(r"(\d+\.)\s{2,}", r"\1 ", text)
    # Схлопываем 3+ пустых строк → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Убираем ведущую точку (артефакт разбиения)
    text = re.sub(r"^\.\s*\n", "", text)
    return text.strip()


def _strip_title_prefix(text: str) -> str:
    """Убирает дублирующий заголовок документа из начала content."""
    lines = text.split("\n")
    while lines and _TITLE_PREFIX_RE.match(lines[0]):
        lines.pop(0)
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Разбиение мульти-секционных документов
# ---------------------------------------------------------------------------


def _split_by_headers(
    full_text: str,
    headers: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Разбиение текста на секции по regex-заголовкам.

    Returns:
        список (procedure_type, display_title, section_content)
    """
    # Находим позиции заголовков
    positions: list[tuple[int, str, str, str]] = []
    for pattern, proc_type, title in headers:
        m = re.search(pattern, full_text, re.IGNORECASE)
        if m:
            positions.append((m.start(), pattern, proc_type, title))

    if not positions:
        return []

    positions.sort(key=lambda x: x[0])

    sections: list[tuple[str, str, str]] = []
    for i, (pos, pattern, proc_type, title) in enumerate(positions):
        # Конец = начало следующей секции или конец текста
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        # Убираем сам заголовок из начала body
        header_match = re.search(pattern, full_text[pos:end], re.IGNORECASE)
        body_start = pos + header_match.end() if header_match else pos
        body = full_text[body_start:end].strip()
        sections.append((proc_type, title, body))

    return sections


# ---------------------------------------------------------------------------
# Основной парсер
# ---------------------------------------------------------------------------


def parse_aftercare_directory(
    input_dir: Path,
    skip_duplicates: bool = True,
) -> list[AftercarRecommendation]:
    """Парсинг всех .doc/.docx из директории → список рекомендаций.

    Args:
        input_dir: папка с Word-документами
        skip_duplicates: пропускать файлы с 'copy' в имени

    Returns:
        список AftercarRecommendation
    """
    results: list[AftercarRecommendation] = []
    seen_types: set[str] = set()

    files = sorted(input_dir.iterdir())
    word_files = [
        f
        for f in files
        if f.suffix.lower() in (".doc", ".docx")
        and (not skip_duplicates or "copy" not in f.name.lower())
        and not f.name.startswith(".")
        and "converted" not in f.name
    ]

    print(f"Найдено {len(word_files)} документов в {input_dir}")

    for path in word_files:
        stem = path.stem
        print(f"\n  Парсинг: {path.name}")

        paragraphs = _read_document(path)
        full_text = "\n".join(paragraphs)

        # Мульти-секционный файл?
        multi_key = next(
            (k for k in MULTI_SECTION_FILES if k in stem),
            None,
        )

        if multi_key:
            headers = MULTI_SECTION_FILES[multi_key]
            sections = _split_by_headers(full_text, headers)
            for proc_type, title, body in sections:
                cleaned = clean_text(body)
                if proc_type not in seen_types and cleaned:
                    results.append(AftercarRecommendation(
                        procedure_type=proc_type,
                        title=title,
                        source_file=path.name,
                        content=cleaned,
                    ))
                    seen_types.add(proc_type)
                    print(f"    + {proc_type}: {len(cleaned):,} chars")
        else:
            # Однофайловый документ
            single_key = next(
                (k for k in SINGLE_SECTION_FILES if k in stem),
                None,
            )
            if single_key is None:
                print(f"    ⚠ Файл не найден в конфигурации, пропускаю")
                continue

            proc_type, title = SINGLE_SECTION_FILES[single_key]
            if proc_type in seen_types:
                print(f"    ⚠ {proc_type} уже обработан, пропускаю дубликат")
                continue

            cleaned = clean_text(full_text)
            cleaned = _strip_title_prefix(cleaned)

            results.append(AftercarRecommendation(
                procedure_type=proc_type,
                title=title,
                source_file=path.name,
                content=cleaned,
            ))
            seen_types.add(proc_type)
            print(f"    + {proc_type}: {len(cleaned):,} chars")

    return results


# ---------------------------------------------------------------------------
# YAML-экспорт с human-readable literal block
# ---------------------------------------------------------------------------


class _LiteralStr(str):
    """Маркер для YAML literal block style (|)."""


def _literal_representer(dumper: yaml.Dumper, data: str) -> Any:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(_LiteralStr, _literal_representer)


def save_recommendations_yaml(
    recommendations: list[AftercarRecommendation],
    output_path: Path,
) -> None:
    """Сохранение рекомендаций в YAML с literal block style для content."""
    records = []
    for rec in recommendations:
        records.append({
            "procedure_type": rec.procedure_type,
            "title": rec.title,
            "source_file": rec.source_file,
            "content": _LiteralStr(rec.content),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {"recommendations": records},
            f,
            default_flow_style=False,
            allow_unicode=True,
            width=120,
            sort_keys=False,
        )

    print(f"\nСохранено в {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Парсинг aftercare .doc/.docx → YAML для KB",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("d4/raw_data/kb/recomendations"),
        help="Папка с Word-документами (default: d4/raw_data/kb/recomendations)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("d4/raw_data/kb/aftercare_recommendations.yaml"),
        help="Путь к выходному YAML (default: d4/raw_data/kb/aftercare_recommendations.yaml)",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Ошибка: директория не найдена: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    recommendations = parse_aftercare_directory(args.input_dir)

    if not recommendations:
        print("Ошибка: ни одной рекомендации не найдено", file=sys.stderr)
        sys.exit(1)

    save_recommendations_yaml(recommendations, args.output)

    # Итоговая сводка
    print(f"\nИтого: {len(recommendations)} рекомендаций")
    total_chars = 0
    for rec in recommendations:
        total_chars += len(rec.content)
        print(f"  {rec.procedure_type:35s} | {len(rec.content):>6,} chars | {rec.title}")
    print(f"  {'':35s} | {total_chars:>6,} chars total")


if __name__ == "__main__":
    main()
