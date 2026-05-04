"""Guard-тесты: D1 notebook должен быть executor-only (Task 1+2).

Архитектурный invariant: `d1/D1_domain_router_v6.ipynb` играет роль исполнителя
и визуализатора готовых артефактов. Вся логика (обучение, чтение артефактов,
plotting, artifact writes, widget UI) живёт в модулях `d1/scripts/*` и
`d1/baselines/*`.

Проверка построена через `nbformat` + `ast`. Magic-lines (`%...`, `!...`)
удаляются из source перед парсингом — это штатный паттерн Jupyter, не код.

Тесты намеренно жёсткие. До реализации Task 1 refactor они падают на
текущем notebook — это expected TDD-red-phase.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterator

import nbformat
import pytest

NOTEBOOK_PATH = Path(__file__).resolve().parent.parent / "D1_domain_router_v6.ipynb"

# ---------------------------------------------------------------------------
# Forbidden patterns (фиксированные черные списки)
# ---------------------------------------------------------------------------

# Методы чтения артефактов (attr name): pd.read_csv, json.load, Path.read_text, ...
_FORBIDDEN_READ_ATTRS: frozenset[str] = frozenset({
    "read_csv", "read_json", "read_excel", "read_parquet",
    "load", "loads",
    "read_text", "read_bytes",
})

# Методы записи артефактов (attr name).
_FORBIDDEN_WRITE_ATTRS: frozenset[str] = frozenset({
    "to_csv", "to_json", "to_excel", "to_parquet",
    "write_text", "write_bytes",
    "savefig",
})

# Запрещённые namespace для Attribute access (plt.<anything>, sns.<anything>, ...).
_FORBIDDEN_NAMESPACES: frozenset[str] = frozenset({
    "plt", "sns", "widgets", "ipywidgets",
})

# Whitelist top-level module prefixes в from-import'ах.
# Всё, что не попадает — потенциальный violation (проверка мягкая, см. тест).
_ALLOWED_IMPORT_PREFIXES: tuple[str, ...] = (
    "d1.",  # все внутренние модули
    "IPython",  # display
    "os", "sys", "pathlib", "warnings", "logging",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def notebook() -> nbformat.NotebookNode:
    """Загрузить notebook через nbformat."""
    assert NOTEBOOK_PATH.exists(), f"Notebook не найден: {NOTEBOOK_PATH}"
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


@pytest.fixture(scope="module")
def code_cells(notebook: nbformat.NotebookNode) -> list[tuple[int, str]]:
    """Список (index, source) для всех code-cells."""
    return [
        (i, "".join(c["source"]))
        for i, c in enumerate(notebook["cells"])
        if c["cell_type"] == "code"
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_magics(source: str) -> str:
    """Удалить строки-magics (`%...`, `!...`) перед ast.parse.

    Ipykernel поддерживает их как не-Python синтаксис. Для AST-проверок они
    несущественны.
    """
    lines = source.splitlines()
    kept = [ln for ln in lines if not ln.lstrip().startswith(("%", "!"))]
    return "\n".join(kept)


def _parse_cell(source: str) -> ast.Module:
    """Безопасный парсинг с stripped magics.

    При SyntaxError возвращает пустой Module — такая cell пропускается
    для соответствующего теста (отдельный тест проверит parseability).
    """
    return ast.parse(_strip_magics(source))


def _walk_calls(tree: ast.AST) -> Iterator[ast.Call]:
    """Все ast.Call в дереве."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _call_attr_name(call: ast.Call) -> str | None:
    """Имя атрибута для `x.method(...)` вызовов; None для других форм."""
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _attribute_base_name(node: ast.Attribute) -> str | None:
    """Рекурсивно достать корневое имя: plt.subplots → 'plt'."""
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_notebook_all_cells_parse_as_python(code_cells: list[tuple[int, str]]) -> None:
    """Каждая code-cell парсится как валидный Python после strip magic-lines."""
    errors = []
    for idx, source in code_cells:
        try:
            ast.parse(_strip_magics(source))
        except SyntaxError as exc:
            errors.append(f"Cell {idx}: {exc}")
    assert not errors, "Cells с SyntaxError:\n" + "\n".join(errors)


def test_no_function_or_class_defs(code_cells: list[tuple[int, str]]) -> None:
    """В code-cells не должно быть `def`/`class`/`async def` на любом уровне.

    Вся логика должна жить в импортированных модулях. Определения функций в
    ноутбуке — признак того, что reporting-код не вынесен.
    """
    offenders: list[str] = []
    for idx, source in code_cells:
        tree = _parse_cell(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                offenders.append(f"Cell {idx}: {type(node).__name__} `{node.name}`")
    assert not offenders, (
        "Notebook содержит определения функций/классов (вынесите в модуль):\n"
        + "\n".join(offenders)
    )


def test_no_artifact_reads(code_cells: list[tuple[int, str]]) -> None:
    """Нет `pd.read_csv`, `json.load`, `Path(...).read_text`, и т.д.

    Чтение артефактов должно идти через `d1.scripts.artifact_io`.
    """
    offenders: list[str] = []
    for idx, source in code_cells:
        tree = _parse_cell(source)
        for call in _walk_calls(tree):
            attr = _call_attr_name(call)
            if attr in _FORBIDDEN_READ_ATTRS:
                offenders.append(f"Cell {idx}: вызов .{attr}(...)")
    assert not offenders, (
        "Notebook содержит прямое чтение артефактов (используйте artifact_io):\n"
        + "\n".join(offenders)
    )


def test_no_plotting_calls(code_cells: list[tuple[int, str]]) -> None:
    """Нет Attribute access на `plt.*`/`sns.*` в cells.

    Plotting инкапсулируется в `d1.scripts.plot_results.plot_*` функциях.
    """
    offenders: list[str] = []
    for idx, source in code_cells:
        tree = _parse_cell(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                base = _attribute_base_name(node)
                if base in {"plt", "sns"}:
                    offenders.append(f"Cell {idx}: {base}.{node.attr}")
    assert not offenders, (
        "Notebook напрямую использует plotting библиотеки (вынесите в plot_results):\n"
        + "\n".join(offenders)
    )


def test_no_artifact_writes(code_cells: list[tuple[int, str]]) -> None:
    """Нет `.to_csv`, `.to_json`, `.savefig`, `Path.write_*` в cells.

    Запись артефактов — только из orchestration-скриптов.
    """
    offenders: list[str] = []
    for idx, source in code_cells:
        tree = _parse_cell(source)
        for call in _walk_calls(tree):
            attr = _call_attr_name(call)
            if attr in _FORBIDDEN_WRITE_ATTRS:
                offenders.append(f"Cell {idx}: .{attr}(...)")
    assert not offenders, (
        "Notebook пишет артефакты напрямую (перенесите в scripts):\n"
        + "\n".join(offenders)
    )


def test_no_widgets_top_level(code_cells: list[tuple[int, str]]) -> None:
    """Нет access на `widgets.*`/`ipywidgets.*` в source cells.

    Widget UI инкапсулируется в `d1.scripts.notebook_widgets`.
    """
    offenders: list[str] = []
    for idx, source in code_cells:
        tree = _parse_cell(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                base = _attribute_base_name(node)
                if base in {"widgets", "ipywidgets"}:
                    offenders.append(f"Cell {idx}: {base}.{node.attr}")
    assert not offenders, (
        "Notebook напрямую использует ipywidgets (вынесите в notebook_widgets):\n"
        + "\n".join(offenders)
    )


def test_imports_are_whitelisted(code_cells: list[tuple[int, str]]) -> None:
    """Все from-imports ведут на whitelisted префиксы.

    Защищает от случайного импорта тяжёлых библиотек (sklearn, torch, ...) в
    notebook: вся heavy-lifting — через `d1.*` модули.
    """
    offenders: list[str] = []
    for idx, source in code_cells:
        tree = _parse_cell(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if not module.startswith(_ALLOWED_IMPORT_PREFIXES):
                    names = ", ".join(a.name for a in node.names)
                    offenders.append(
                        f"Cell {idx}: from {module} import {names}"
                    )
    assert not offenders, (
        "Notebook импортирует не-whitelisted модули:\n" + "\n".join(offenders)
    )
