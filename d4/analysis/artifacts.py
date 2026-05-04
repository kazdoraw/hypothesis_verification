"""Versioned path management для экспериментальных прогонов.

Единый источник правды для всех путей артефактов.
Каждый прогон (run) получает уникальный run_id (YYYYMMDD_HHMMSS)
и сохраняет артефакты в outputs/runs/{run_id}/.

Используется всеми analysis-модулями и CLI runner (mini_llm_run.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Корневые пути
# ---------------------------------------------------------------------------

D4_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = D4_ROOT / "outputs"
RUNS_DIR = OUTPUTS / "runs"
CHUNKS_FROZEN = D4_ROOT / "data" / "chunks_frozen.json"


# ---------------------------------------------------------------------------
# RunPaths — все пути одного прогона
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """Пути артефактов одного экспериментального прогона.

    Все поддиректории создаются при вызове make_run_paths().
    Frozen dataclass — пути неизменяемы после создания.
    """

    run_id: str
    run_dir: Path
    figures_dir: Path = field(init=False, repr=False)
    tables_dir: Path = field(init=False, repr=False)
    reports_dir: Path = field(init=False, repr=False)
    logs_dir: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "figures_dir", self.run_dir / "figures")
        object.__setattr__(self, "tables_dir", self.run_dir / "tables")
        object.__setattr__(self, "reports_dir", self.run_dir / "reports")
        object.__setattr__(self, "logs_dir", self.run_dir / "logs")

    # --- Файловые пути (per run_name × mode) ---

    def results_jsonl(self, mode: str, run_name: str = "pilot_dev_30") -> Path:
        """Путь к raw-результатам: {run_dir}/{run_name}_{mode}.jsonl"""
        return self.run_dir / f"{run_name}_{mode}.jsonl"

    def report_json(self, mode: str, run_name: str = "pilot_dev_30") -> Path:
        """Путь к JSON-отчёту: {run_dir}/{run_name}_{mode}.report.json"""
        return self.run_dir / f"{run_name}_{mode}.report.json"

    # --- Файловые пути (per-name) ---

    def figure(self, name: str) -> Path:
        return self.figures_dir / name

    def table(self, name: str) -> Path:
        return self.tables_dir / name

    def report_md(self, name: str) -> Path:
        return self.reports_dir / name

    def memo_md(self) -> Path:
        return self.reports_dir / "stage2a_decision_memo.md"

    def manifest_json(self) -> Path:
        return self.run_dir / "manifest.json"

    def config_snapshot(self) -> Path:
        return self.run_dir / "config_snapshot.yaml"


# ---------------------------------------------------------------------------
# Фабрики и helpers
# ---------------------------------------------------------------------------

def make_run_paths(run_id: str) -> RunPaths:
    """Создаёт все поддиректории и возвращает RunPaths."""
    run_dir = RUNS_DIR / run_id
    paths = RunPaths(run_id=run_id, run_dir=run_dir)
    for d in (paths.run_dir, paths.figures_dir, paths.tables_dir,
              paths.reports_dir, paths.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    return paths


def latest_run_id() -> str | None:
    """Последний run_id по имени (YYYYMMDD_HHMMSS → лексикографический порядок)."""
    runs = list_runs()
    return runs[-1] if runs else None


def list_runs() -> list[str]:
    """Все run_ids, отсортированные по дате (лексикографически)."""
    if not RUNS_DIR.exists():
        return []
    return sorted(
        d.name for d in RUNS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def list_run_artifacts(run_paths: RunPaths) -> list[tuple[str, Path]]:
    """Все файлы в run_dir с категорией.

    Returns:
        [(category, path), ...] где category ∈ {result, report, figure, table, log, meta}
    """
    _CATEGORY_MAP = {
        run_paths.figures_dir: "figure",
        run_paths.tables_dir: "table",
        run_paths.reports_dir: "report",
        run_paths.logs_dir: "log",
    }
    artifacts: list[tuple[str, Path]] = []
    if not run_paths.run_dir.exists():
        return artifacts

    for f in sorted(run_paths.run_dir.rglob("*")):
        if not f.is_file():
            continue
        category = "meta"
        for dir_path, cat in _CATEGORY_MAP.items():
            if f.is_relative_to(dir_path):
                category = cat
                break
        else:
            if f.suffix == ".jsonl":
                category = "result"
            elif f.suffix == ".json" and "report" in f.name:
                category = "report"
        artifacts.append((category, f))
    return artifacts
