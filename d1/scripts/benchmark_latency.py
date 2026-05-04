"""Phase 4: Honest latency benchmark per-baseline (encode vs predict breakdown).

`run_baselines.py` уже пишет суммарный `latency_ms` в `baseline_results.csv`,
но без разбивки encode/predict. Для дипломной работы важно видеть, сколько
времени тратит encoder (e5-small / bge-m3 / fastText / TF-IDF), а сколько
classifier head — это даёт честный production-decision.

Алгоритм:
    1. Загружаем bundle (cached) для всех enabled baselines.
    2. Берём первые N текстов val (default 100).
    3. Для каждого baseline × R повторов (default 5):
         encode_ms — только эмбеддинг/векторизация (через _encode для B2,
                     vectorizer.transform для B1, no-op для B0/B1.3).
         predict_ms = total_ms − encode_ms.
    4. Аггрегация: median, p95.

Артефакт:
    d1/results/latency_breakdown.csv

Запуск:
    cd study && .venv/bin/python -m d1.scripts.benchmark_latency
    cd study && .venv/bin/python -m d1.scripts.benchmark_latency --n 100 --repeats 5
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from d1.baselines.b0_rules import B0RulesClassifier
from d1.baselines.b1_fasttext import B1FastTextClassifier
from d1.baselines.b1_tfidf import B1TfidfClassifier
from d1.baselines.b2_embedding import B2EmbeddingClassifier
from d1.baselines.trained_bundle import BASELINE_CONFIGS, train_bundle
from d1.config import DATA_DIR, DATASET_PREFIX, RESULTS_DIR

logger = logging.getLogger(__name__)


def _measure_encode_predict(
    model: Any, texts: list[str],
) -> tuple[float, float]:
    """Один проход: возвращает (encode_ms, predict_ms).

    Разбивка:
      - B2EmbeddingClassifier: encode = sentence-transformer.encode + L2.
      - B1TfidfClassifier:     encode = TfidfVectorizer.transform.
      - B1FastTextClassifier:  fastText не имеет separate encode шага
                               (linear over hashed ngrams) — encode_ms=0,
                               predict_ms = total.
      - B0RulesClassifier:     no encode → encode_ms=0.
    """
    t0 = time.perf_counter()
    if isinstance(model, B2EmbeddingClassifier):
        X = model._encode(texts)
        t_enc = time.perf_counter() - t0
        # predict только на классификаторе head (без encode)
        t1 = time.perf_counter()
        model._head.predict(X)
        t_pred = time.perf_counter() - t1
    elif isinstance(model, B1TfidfClassifier):
        tfidf_step = model.pipeline.named_steps["tfidf"]
        clf_step = model.pipeline.named_steps["clf"]
        X = tfidf_step.transform(texts)
        t_enc = time.perf_counter() - t0
        t1 = time.perf_counter()
        clf_step.predict(X)
        t_pred = time.perf_counter() - t1
    else:
        # B0/B1.3: encode не выделяется — total ≈ predict.
        model.predict(texts)
        t_enc = 0.0
        t_pred = time.perf_counter() - t0
    return t_enc * 1000, t_pred * 1000


def benchmark_baseline(
    name: str, model: Any, texts: list[str], repeats: int,
) -> dict[str, Any]:
    """Усреднение по `repeats` повторам.

    Первый прогон выкидывается как warm-up (encoder/JIT кэширование).
    """
    encode_runs: list[float] = []
    predict_runs: list[float] = []
    for r in range(repeats + 1):  # +1 warm-up
        enc_ms, pred_ms = _measure_encode_predict(model, texts)
        if r == 0:
            continue
        encode_runs.append(enc_ms)
        predict_runs.append(pred_ms)

    encode_arr = np.asarray(encode_runs)
    predict_arr = np.asarray(predict_runs)
    total_arr = encode_arr + predict_arr
    n = len(texts)
    return {
        "baseline": name,
        "n_samples": n,
        "repeats": repeats,
        "encode_ms_total_median": round(float(np.median(encode_arr)), 3),
        "encode_ms_per_text_median": round(float(np.median(encode_arr) / n), 4),
        "predict_ms_total_median": round(float(np.median(predict_arr)), 3),
        "predict_ms_per_text_median": round(float(np.median(predict_arr) / n), 4),
        "total_ms_median": round(float(np.median(total_arr)), 3),
        "total_ms_per_text_median": round(float(np.median(total_arr) / n), 4),
        "total_ms_p95": round(float(np.percentile(total_arr, 95)), 3),
        "encode_share": round(
            float(np.median(encode_arr) / max(np.median(total_arr), 1e-9)), 3,
        ),
    }


def _free_memory(has_mps: bool) -> None:
    """Принудительно освободить накопленную память между моделями.

    Для dense-моделей (BGE-M3 ~2.3 GB, e5-small ~500 MB) sequential-загрузка
    без явной очистки приводит к OOM на Apple Silicon (MPS unified memory).
    """
    import gc

    gc.collect()
    if has_mps:
        try:
            import torch

            torch.mps.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def run_benchmark(
    n: int = 100, repeats: int = 5, save: bool = True,
) -> pd.DataFrame:
    """Замер латентности per-baseline на CPU (production-aligned).

    Стратегия:
      1. Загружаем модели **по одной** (sequential), чтобы не накапливать
         dense-encoder'ы в памяти одновременно.
      2. Все sentence-transformer модели forced на CPU — production-инференс
         AI-ядра тоже работает на CPU, поэтому MPS/CUDA-замер был бы
         методологически некорректным (overestimated speed).
      3. После каждой модели — `gc.collect()` + `torch.mps.empty_cache()`.

    Args:
        n: число текстов из val для замера (default 100).
        repeats: повторов на baseline после warm-up (default 5).
        save: писать ли результат в `RESULTS_DIR / "latency_breakdown.csv"`.
            По умолчанию True (вызов из notebook сразу обновляет артефакт).

    Returns:
        DataFrame со строкой на baseline и колонками encode/predict/total.
    """
    try:
        import torch

        has_mps = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except ImportError:
        has_mps = False

    val_path = DATA_DIR / f"{DATASET_PREFIX}_val.csv"
    df = pd.read_csv(val_path, dtype=str).fillna("")
    texts = df["text"].head(n).tolist()
    if len(texts) < n:
        logger.warning(
            "Запрошено n=%d, в val %d → используем %d", n, len(texts), len(texts),
        )

    enabled = [
        name for name, cfg in BASELINE_CONFIGS.items() if cfg["enabled"]
    ]

    rows: list[dict[str, Any]] = []
    for name in enabled:
        # Загрузка по одной модели — кэш используется, переобучения нет.
        bundle_one = train_bundle(names=[name], use_cache=True)
        model = bundle_one.get(name)

        # Force CPU для dense-моделей: production-aligned latency.
        if isinstance(model, B2EmbeddingClassifier):
            model.set_device("cpu")

        if isinstance(model, B0RulesClassifier):
            # B0_rules: rule-based, _encode/_head не определены — спец-проход.
            t0 = time.perf_counter()
            for _ in range(repeats):
                model.predict(texts)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            mean_ms = elapsed_ms / repeats
            rows.append({
                "baseline": name,
                "n_samples": len(texts),
                "repeats": repeats,
                "encode_ms_total_median": 0.0,
                "encode_ms_per_text_median": 0.0,
                "predict_ms_total_median": round(mean_ms, 3),
                "predict_ms_per_text_median": round(mean_ms / len(texts), 4),
                "total_ms_median": round(mean_ms, 3),
                "total_ms_per_text_median": round(mean_ms / len(texts), 4),
                "total_ms_p95": round(mean_ms, 3),
                "encode_share": 0.0,
            })
        else:
            logger.info("Benchmark: %s", name)
            rows.append(benchmark_baseline(name, model, texts, repeats))

        # Освобождаем модель и GPU/MPS pool перед загрузкой следующей.
        del bundle_one, model
        _free_memory(has_mps)

    df_out = pd.DataFrame(rows)

    if save:
        out_path = RESULTS_DIR / "latency_breakdown.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_path, index=False)
        logger.info("Saved: %s", out_path)

    return df_out


def main() -> None:
    parser = argparse.ArgumentParser(description="D1 v6 latency breakdown benchmark")
    parser.add_argument("--n", type=int, default=100,
                        help="число текстов из val (default 100)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="повторов на baseline (default 5)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    df = run_benchmark(n=args.n, repeats=args.repeats)

    print("\n" + "=" * 100)
    print("  LATENCY BREAKDOWN (per-text median, ms)")
    print("=" * 100)
    cols = [
        "baseline", "encode_ms_per_text_median",
        "predict_ms_per_text_median", "total_ms_per_text_median",
        "encode_share",
    ]
    print(df[cols].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
