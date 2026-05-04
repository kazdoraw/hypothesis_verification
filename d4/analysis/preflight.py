"""Pre-run проверки перед запуском эксперимента.

Валидирует: наличие файлов, API-ключи, cache fingerprint,
collision с существующими output-файлами.
Сохраняет config snapshot в run_dir.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

from d4.analysis.artifacts import CHUNKS_FROZEN, D4_ROOT, RunPaths
from d4.config import ExperimentConfig


_CONFIG_PATH_DEFAULT = D4_ROOT / "configs" / "experiment.yaml"

# Маппинг run_name → eval set path
_EVAL_PATHS: dict[str, Path] = {
    "pilot_dev_30": D4_ROOT / "outputs" / "pilot_dev_30.yaml",
    "full_dev": D4_ROOT / "data" / "eval_set_dev_v2.yaml",
    "smoke": Path(__file__).resolve().parent.parent / "tests" / "mini_eval_set.yaml",
    "hard": Path(__file__).resolve().parent.parent / "tests" / "mini_eval_set_hard.yaml",
    "blind": Path(__file__).resolve().parent.parent / "tests" / "blind_hard_holdout.yaml",
}
_DEFAULT_RUN_NAME = "pilot_dev_30"


@dataclass
class PreflightResult:
    """Итог preflight-проверок."""

    ok: bool = True
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))
        if not passed:
            self.ok = False

    def warn(self, name: str, detail: str = "") -> None:
        """Информационное предупреждение — не влияет на ok."""
        self.checks.append((name, None, detail))  # type: ignore[arg-type]

    def summary(self) -> str:
        lines = []
        for name, passed, detail in self.checks:
            if passed is None:
                icon = "⚠"
            else:
                icon = "✓" if passed else "✗"
            line = f"  {icon} {name}"
            if detail:
                line += f"  ({detail})"
            lines.append(line)
        status = "PASS" if self.ok else "FAIL"
        lines.insert(0, f"Preflight: {status}")
        return "\n".join(lines)


def _check_file(result: PreflightResult, name: str, path: Path) -> None:
    """Проверка: файл существует и не пуст."""
    if not path.exists():
        result.add(name, False, f"не найден: {path}")
    elif path.stat().st_size == 0:
        result.add(name, False, f"пустой: {path}")
    else:
        result.add(name, True, str(path))


def _check_output_collision(
    result: PreflightResult,
    run_paths: RunPaths,
    modes: list[str],
    run_name: str = _DEFAULT_RUN_NAME,
    expect_existing: bool = False,
) -> None:
    """Проверка согласованности raw-артефактов с режимом прогона.

    expect_existing=False (run_new):
        raw-файлы НЕ должны существовать — иначе collision с предыдущим прогоном.
    expect_existing=True (analyze_existing):
        raw-файлы (jsonl + report.json) ДОЛЖНЫ существовать — иначе анализировать
        нечего.
    """
    for mode in modes:
        jsonl = run_paths.results_jsonl(mode, run_name=run_name)
        report = run_paths.report_json(mode, run_name=run_name)

        if expect_existing:
            missing = [p.name for p in (jsonl, report) if not p.exists()]
            if missing:
                result.add(
                    f"artifacts_{mode}",
                    False,
                    f"нет файлов для анализа: {', '.join(missing)}",
                )
            else:
                result.add(
                    f"artifacts_{mode}",
                    True,
                    f"{jsonl.name} + {report.name}",
                )
        else:
            if jsonl.exists():
                result.add(
                    f"collision_{mode}",
                    False,
                    f"файл уже существует: {jsonl.name}",
                )
            else:
                result.add(
                    f"collision_{mode}", True, f"нет collision ({run_name}_{mode})"
                )


def _check_cache_fingerprint(
    result: PreflightResult,
    config: ExperimentConfig,
    chunks_path: Path,
) -> None:
    """Проверка cache fingerprint: prompt version, prompt hash, chunk content hash."""
    try:
        from d4.pipeline.enrichment import _PROMPT_VERSION, _prompts_hash
        result.add(
            "cache_prompt_version",
            True,
            f"prompt_version={_PROMPT_VERSION}",
        )
        result.add(
            "cache_prompts_hash",
            True,
            f"prompts_hash={_prompts_hash()}",
        )
    except ImportError:
        result.add("cache_fingerprint", False, "не удалось импортировать enrichment")
        return

    if chunks_path.exists():
        try:
            import json
            from d4.models import KBChunk
            from d4.pipeline.enrichment import _content_hash
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunks = [KBChunk.model_validate(c) for c in raw]
            result.add(
                "cache_content_hash",
                True,
                f"content_hash={_content_hash(chunks)}, n_chunks={len(chunks)}",
            )
        except Exception as exc:
            result.add("cache_content_hash", False, f"ошибка загрузки чанков: {exc}")

    rep = config.representation
    result.add(
        "cache_params",
        True,
        f"mode={rep.mode}, c2_max_tokens={rep.c2_prompt_max_tokens}, "
        f"c2_temp={rep.c2_prompt_temperature}",
    )


def _snapshot_config(
    run_paths: RunPaths,
    config_path: Path | None = None,
) -> Path:
    """Копирует experiment.yaml в run_dir/config_snapshot.yaml."""
    src = config_path or _CONFIG_PATH_DEFAULT
    dst = run_paths.config_snapshot()
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def run_preflight(
    config: ExperimentConfig,
    modes: list[str],
    run_paths: RunPaths,
    chunks_path: Path | None = None,
    eval_path: Path | None = None,
    config_path: Path | None = None,
    run_name: str = _DEFAULT_RUN_NAME,
    expect_existing: bool = False,
) -> PreflightResult:
    """Полная pre-run валидация.

    Args:
        config: загруженный ExperimentConfig
        modes: список representation modes для прогона
        run_paths: RunPaths текущего прогона
        chunks_path: путь к chunks_frozen.json (default: D4_ROOT/data/chunks_frozen.json)
        eval_path: путь к eval set (резолвится из run_name, если не указан явно)
        config_path: путь к experiment.yaml для snapshot
        run_name: имя eval set (pilot_dev_30, full_dev, smoke, hard, blind)
        expect_existing: если True — ожидаем, что raw-файлы уже есть
            (режим analyze_existing). Collision-check инвертируется: файлы
            ДОЛЖНЫ существовать, иначе анализировать нечего.

    Returns:
        PreflightResult с результатами всех проверок
    """
    result = PreflightResult()
    _chunks = chunks_path or CHUNKS_FROZEN
    _eval = eval_path or _EVAL_PATHS.get(run_name, _EVAL_PATHS[_DEFAULT_RUN_NAME])

    # 1. Файлы данных
    _check_file(result, "chunks_frozen", _chunks)
    _check_file(result, "eval_set", _eval)

    # 2. Python / venv
    result.add(
        "python_venv",
        hasattr(sys, "prefix") and "venv" in sys.prefix,
        f"prefix={sys.prefix}",
    )

    # 3. Output collision / existing artifacts
    _check_output_collision(
        result, run_paths, modes, run_name=run_name, expect_existing=expect_existing
    )

    # 4. API key (для llm_enriched; в analyze-режиме LLM не запускается)
    if "llm_enriched" in modes and not expect_existing:
        has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
        result.add("api_key", has_key, "OPENROUTER_API_KEY")

    # 5. Config mode vs запрошенный (info-only: runner перезаписывает через --representation)
    for mode in modes:
        match = config.representation.mode == mode
        if match:
            result.add(f"config_mode_{mode}", True,
                        f"config.mode={config.representation.mode}")
        else:
            result.warn(f"config_mode_{mode}",
                        f"config.mode={config.representation.mode}, requested={mode}"
                        " — runner перезапишет через --representation")

    # 6. Cache fingerprint
    _check_cache_fingerprint(result, config, _chunks)

    # 7. Config snapshot
    snap = _snapshot_config(run_paths, config_path)
    result.add("config_snapshot", snap.exists(), str(snap))

    # 8. Dry summary
    strategy_names = [s.get("name", "?") for s in config.core_strategies]
    result.add(
        "dry_summary",
        True,
        f"modes={modes}, strategies={strategy_names}, eval_set={_eval.name}",
    )

    return result
