"""Обучение и сохранение всех D1 baselines — тонкая обёртка над TrainedBundle.

Переиспользует единую точку обучения из `d1/baselines/trained_bundle.py`
(Task 0 из roadmap v3). Скрипт нужен для:
- принудительного rebuild всех enabled baseline'ов (use_cache=False);
- human-friendly summary с размерами joblib-файлов.

Для inference consumers используйте напрямую `train_bundle(...)` с
`use_cache=True` — это дешевле и быстрее.

Запуск:
    cd study && python -m d1.scripts.save_models
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.baselines.trained_bundle import (
    BASELINE_CONFIGS,
    MODELS_DIR,
    train_bundle,
)

logger = logging.getLogger(__name__)


def save_all_models() -> dict[str, Path]:
    """Принудительно обучить и сохранить все enabled baseline'ы.

    Use-case: первый запуск эксперимента или пере-обучение после смены
    гиперпараметров / версии sklearn. Для обычной работы consumer'ы
    вызывают `train_bundle(..., use_cache=True)` напрямую.

    Returns:
        dict[baseline_name, path_to_joblib]
    """
    enabled_names = [n for n, cfg in BASELINE_CONFIGS.items() if cfg["enabled"]]
    logger.info("save_all_models: %d enabled baseline'ов → %s",
                len(enabled_names), MODELS_DIR)

    # use_cache=False — обязательный rebuild; пишем свежие joblib и bundle_metadata.json.
    bundle = train_bundle(
        names=enabled_names, use_cache=False, cache_dir=MODELS_DIR,
    )

    # Маппим имена на реальные файлы joblib (slug-преобразование делает trained_bundle).
    from d1.baselines.trained_bundle import _model_path

    saved: dict[str, Path] = {
        name: _model_path(MODELS_DIR, name) for name in bundle.models
    }

    print(f"\n✓ Сохранено {len(saved)} моделей в {MODELS_DIR}")
    for name, path in saved.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"  {name:30s} → {path.name} ({size_kb:.1f} KB)")
        else:
            print(f"  {name:30s} → (файл отсутствует: {path.name})")

    return saved


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    save_all_models()


if __name__ == "__main__":
    main()
