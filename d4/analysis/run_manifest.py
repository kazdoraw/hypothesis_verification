"""Manifest прогона: полная meta-информация для воспроизводимости.

Сохраняется в run_dir/manifest.json после каждого прогона.
Содержит: run_id, run_name, modes, eval_set_path, config snapshot,
input/output/cache файлы, enrichment prompt version/hash, git info.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from d4.analysis.artifacts import RunPaths

logger = logging.getLogger(__name__)


@dataclass
class RunManifest:
    """Полный fingerprint одного экспериментального прогона."""

    run_id: str
    timestamp: str

    # --- run-name & modes (v2) ---
    run_name: str = ""
    modes: list[str] = field(default_factory=list)
    eval_set_path: str = ""

    # legacy compat: если сохранён одиночный mode
    representation_mode: str = ""

    command: str = ""
    config_snapshot_path: str = ""
    input_files: dict[str, str] = field(default_factory=dict)
    output_files: dict[str, str] = field(default_factory=dict)
    cache_files: dict[str, str] = field(default_factory=dict)
    enrichment_prompt_version: str = ""
    enrichment_prompts_hash: str = ""
    git_info: str | None = None


def _git_info() -> str | None:
    """Branch + short SHA, если доступен git."""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        return f"{branch}@{sha}"
    except Exception:
        return None


def build_manifest(
    run_paths: RunPaths,
    *,
    modes: list[str] | None = None,
    mode: str = "",
    run_name: str = "pilot_dev_30",
    eval_set_path: str = "",
    command: str = "",
) -> RunManifest:
    """Собирает manifest из текущего состояния run_dir.

    Args:
        run_paths: пути текущего run
        modes: список representation modes (предпочтительно)
        mode: одиночный mode (legacy fallback, если modes не передан)
        run_name: имя eval set (pilot_dev_30, full_dev, smoke, ...)
        eval_set_path: фактический путь к eval set файлу
        command: CLI команда, которой запущен прогон
    """
    from d4.pipeline.enrichment import _PROMPT_VERSION, _prompts_hash

    resolved_modes = modes if modes else ([mode] if mode else [])

    input_files: dict[str, str] = {}
    output_files: dict[str, str] = {}
    cache_files: dict[str, str] = {}

    # input: chunks
    from d4.analysis.artifacts import CHUNKS_FROZEN, D4_ROOT
    if CHUNKS_FROZEN.exists():
        input_files["chunks_frozen"] = str(CHUNKS_FROZEN)

    # input: eval set
    if eval_set_path:
        input_files["eval_set"] = eval_set_path
    else:
        from d4.analysis.preflight import _EVAL_PATHS
        resolved = _EVAL_PATHS.get(run_name)
        if resolved and resolved.exists():
            input_files["eval_set"] = str(resolved)

    # input: config snapshot
    config_snap = run_paths.config_snapshot()
    if config_snap.exists():
        input_files["config_snapshot"] = str(config_snap)

    # output: per-mode results + reports
    for m in resolved_modes:
        for ext in ("jsonl", "report.json"):
            p = run_paths.run_dir / f"{run_name}_{m}.{ext}"
            if p.exists():
                output_files[p.name] = str(p)

    # cache
    cache_dir = D4_ROOT / "outputs" / "enrichment_cache"
    if cache_dir.exists():
        for f in sorted(cache_dir.glob("*.json")):
            cache_files[f.name] = str(f)

    return RunManifest(
        run_id=run_paths.run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        run_name=run_name,
        modes=resolved_modes,
        eval_set_path=input_files.get("eval_set", ""),
        representation_mode=mode or (resolved_modes[0] if len(resolved_modes) == 1 else ""),
        command=command,
        config_snapshot_path=str(config_snap) if config_snap.exists() else "",
        input_files=input_files,
        output_files=output_files,
        cache_files=cache_files,
        enrichment_prompt_version=_PROMPT_VERSION,
        enrichment_prompts_hash=_prompts_hash(),
        git_info=_git_info(),
    )


def save_manifest(manifest: RunManifest, run_paths: RunPaths) -> Path:
    """Сохраняет manifest.json в run_dir."""
    path = run_paths.manifest_json()
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Manifest saved: %s", path)
    return path


def load_manifest(run_paths: RunPaths) -> RunManifest | None:
    """Загружает manifest.json. Возвращает None если файл не найден."""
    path = run_paths.manifest_json()
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    # Обратная совместимость: старые manifest без новых полей
    data.setdefault("run_name", "")
    data.setdefault("modes", [])
    data.setdefault("eval_set_path", "")
    return RunManifest(**data)
