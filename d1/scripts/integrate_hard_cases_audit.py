"""Интеграция audited hard-case candidates в d1_v6_hard_cases.yaml.

Использует результаты ручного audit из `hard_cases_audit.csv` и candidate pool
из `hard_cases_candidates.yaml`, добавляя только `accept` / `rewrite`.

Ключевые гарантии:
- reject кейсы не попадают в gold YAML;
- exact duplicate texts против existing gold и внутри новой партии запрещены;
- leakage threshold (`max_sim_to_hard` / `max_sim_to_seed`) проверяется повторно;
- квоты roadmap контролируются: `grok_audited <= 30%`, `grok_rewritten >= 30%`.

Запуск:
    cd study && python -m d1.scripts.integrate_hard_cases_audit --apply
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

_STUDY_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_STUDY_ROOT) not in sys.path:
    sys.path.insert(0, str(_STUDY_ROOT))

from d1.config import DATA_DIR, HARD_CASES_FILE, LEAKAGE_COSINE_THRESHOLD

logger = logging.getLogger(__name__)

_AUDIT_FILE = DATA_DIR / "hard_cases_audit.csv"
_CANDIDATES_FILE = DATA_DIR / "hard_cases_candidates.yaml"
_TEXT_NORMALIZE_RE = re.compile(r"\s+")

_SCENARIO_HEADERS: dict[str, str] = {
    "anamnesis_faq_confusion": "10. ANAMNESIS FAQ CONFUSION — audited candidate pool",
    "post_treatment_complications": "11. POST-TREATMENT COMPLICATIONS — audited candidate pool",
    "mixed_pain_price_booking": "12. MIXED PAIN + PRICE/BOOKING — audited candidate pool",
    "vague_short_urgent": "13. VAGUE SHORT URGENT — audited candidate pool",
    "pediatric_trauma": "14. PEDIATRIC TRAUMA — audited candidate pool",
    "allergy_bleeding_swelling": "15. ALLERGY / BLEEDING / SWELLING — audited candidate pool",
    "switch_additional": "16. SWITCH ADDITIONAL — audited candidate pool",
}


def _normalize_text(text: str) -> str:
    return _TEXT_NORMALIZE_RE.sub(" ", text.strip().lower())


def _load_candidates() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with open(_CANDIDATES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        c["candidate_id"]: c for c in data["candidates"]
    }, {
        "generated_by": data.get("generated_by", ""),
        "prompt_file": data.get("prompt_file", ""),
    }


def _load_audit_rows() -> list[dict[str, str]]:
    with open(_AUDIT_FILE, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_existing_cases() -> tuple[list[dict[str, Any]], str]:
    yaml_text = HARD_CASES_FILE.read_text(encoding="utf-8")
    with open(HARD_CASES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["hard_cases"], yaml_text


def _max_case_id(cases: list[dict[str, Any]], prefix: str) -> int:
    max_id = 0
    for case in cases:
        case_id = str(case.get("id", ""))
        if not case_id.startswith(prefix):
            continue
        try:
            max_id = max(max_id, int(case_id.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_id


def _build_case(
    row: dict[str, str],
    candidate: dict[str, Any],
    candidate_meta: dict[str, Any],
) -> dict[str, Any]:
    decision = row["audit_decision"]
    text = row["final_text"].strip() or candidate["text"]
    route_domain = row["final_route_domain"].strip() or row["proposed_route_domain"].strip()
    subtype = row["final_subtype"].strip() or row["proposed_subtype"].strip()
    urgency = row["final_urgency"].strip() or row["proposed_urgency"].strip()
    source = row["source_after_audit"].strip() or (
        "grok_rewritten" if decision == "rewrite" else "grok_audited"
    )

    case: dict[str, Any] = {
        "text": text,
        "route_domain": route_domain,
        "subtype": subtype,
        "urgency": urgency,
        "source": source,
    }
    active_domain = candidate.get("active_domain", "").strip()
    if active_domain:
        case["active_domain"] = active_domain
    notes = row["notes"].strip()
    if notes:
        case["notes"] = notes

    case["provenance"] = {
        "candidate_id": row["candidate_id"],
        "scenario": row["scenario"],
        "generator": candidate_meta["generated_by"],
        "prompt_file": candidate_meta["prompt_file"],
        "audit_decision": decision,
        "max_sim_to_hard": float(row["max_sim_to_hard"] or 0.0),
        "nearest_hard_id": row["nearest_hard_id"] or "",
        "max_sim_to_seed": float(row["max_sim_to_seed"] or 0.0),
    }
    return case


def _validate_audit(
    audit_rows: list[dict[str, str]],
    candidates: dict[str, dict[str, Any]],
    existing_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing_candidate_ids = {
        str(case.get("provenance", {}).get("candidate_id", ""))
        for case in existing_cases
        if isinstance(case.get("provenance"), dict)
    }
    existing_texts = {
        _normalize_text(str(case.get("text", "")))
        for case in existing_cases
        if str(case.get("text", "")).strip()
    }

    accepted_rows = [r for r in audit_rows if r["audit_decision"] in {"accept", "rewrite"}]
    decisions = Counter(r["audit_decision"] for r in audit_rows)
    logger.info("Audit decisions: %s", dict(decisions))

    if not accepted_rows:
        raise RuntimeError("Нет accept/rewrite кейсов — нечего интегрировать.")

    n_final = len(accepted_rows)
    n_accept = decisions.get("accept", 0)
    n_rewrite = decisions.get("rewrite", 0)
    if (n_accept / n_final) > 0.30:
        raise RuntimeError(
            f"Нарушение квоты: grok_audited={n_accept}/{n_final} > 30%.",
        )
    if (n_rewrite / n_final) < 0.30:
        raise RuntimeError(
            f"Нарушение квоты: grok_rewritten={n_rewrite}/{n_final} < 30%.",
        )

    new_cases: list[dict[str, Any]] = []
    new_texts: set[str] = set()
    for row in accepted_rows:
        cid = row["candidate_id"]
        if cid in existing_candidate_ids:
            raise RuntimeError(f"Candidate {cid} уже был интегрирован в hard_cases.yaml.")
        if cid not in candidates:
            raise RuntimeError(f"Candidate {cid} отсутствует в hard_cases_candidates.yaml.")
        if max(float(row["max_sim_to_hard"] or 0.0), float(row["max_sim_to_seed"] or 0.0)) >= LEAKAGE_COSINE_THRESHOLD:
            raise RuntimeError(
                f"Candidate {cid} нарушает leakage threshold >= {LEAKAGE_COSINE_THRESHOLD}.",
            )
        if row["audit_decision"] == "rewrite" and not row["final_text"].strip():
            raise RuntimeError(f"Rewrite candidate {cid} без final_text.")

        case = _build_case(row, candidates[cid], {
            "generated_by": candidates[cid].get("generated_by", "") or "",
            "prompt_file": "",
        })
        text_norm = _normalize_text(case["text"])
        if text_norm in existing_texts:
            raise RuntimeError(f"Duplicate text vs existing gold: {cid} -> '{case['text']}'")
        if text_norm in new_texts:
            raise RuntimeError(f"Duplicate text внутри новой партии: {cid} -> '{case['text']}'")

        new_texts.add(text_norm)
        new_cases.append(case)
    return new_cases


def _assign_ids(
    new_cases: list[dict[str, Any]],
    existing_cases: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    next_hard = _max_case_id(existing_cases, "hard_") + 1
    next_switch = _max_case_id(existing_cases, "switch_") + 1

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in new_cases:
        scenario = str(case["provenance"]["scenario"])
        if scenario == "switch_additional":
            case["id"] = f"switch_{next_switch:03d}"
            next_switch += 1
        else:
            case["id"] = f"hard_{next_hard:03d}"
            next_hard += 1
        grouped[scenario].append(case)
    return grouped


def _render_case(case: dict[str, Any]) -> str:
    ordered: dict[str, Any] = {
        "id": case["id"],
        "text": case["text"],
        "route_domain": case["route_domain"],
        "subtype": case["subtype"],
        "urgency": case["urgency"],
    }
    if "active_domain" in case:
        ordered["active_domain"] = case["active_domain"]
    ordered["source"] = case["source"]
    if "notes" in case:
        ordered["notes"] = case["notes"]
    ordered["provenance"] = case["provenance"]

    dumped = yaml.safe_dump(
        ordered,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
        width=1000,
    ).strip().splitlines()
    lines = [f"  - {dumped[0]}"]
    lines.extend(f"    {line}" for line in dumped[1:])
    return "\n".join(lines)


def _render_append_block(grouped_cases: dict[str, list[dict[str, Any]]]) -> str:
    chunks: list[str] = []
    for scenario in _SCENARIO_HEADERS:
        cases = grouped_cases.get(scenario, [])
        if not cases:
            continue
        title = _SCENARIO_HEADERS[scenario]
        chunks.append("")
        chunks.append("  # =========================================================================")
        chunks.append(f"  # {title}")
        chunks.append("  # =========================================================================")
        for case in cases:
            chunks.append(_render_case(case))
    return "\n".join(chunks) + "\n"


def integrate(apply_changes: bool = False) -> dict[str, Any]:
    candidates, meta = _load_candidates()
    audit_rows = _load_audit_rows()
    existing_cases, existing_yaml = _load_existing_cases()

    new_cases = _validate_audit(audit_rows, candidates, existing_cases)
    for case in new_cases:
        prov = case["provenance"]
        prov["generator"] = meta["generated_by"]
        prov["prompt_file"] = meta["prompt_file"]

    grouped = _assign_ids(new_cases, existing_cases)

    summary = {
        "accepted_total": len(new_cases),
        "hard_added": sum(len(v) for k, v in grouped.items() if k != "switch_additional"),
        "switch_added": len(grouped.get("switch_additional", [])),
        "by_scenario": {k: len(v) for k, v in grouped.items()},
        "sources": Counter(case["source"] for case in new_cases),
    }

    logger.info("Integration summary: %s", summary)
    if apply_changes:
        append_block = _render_append_block(grouped)
        if not existing_yaml.endswith("\n"):
            existing_yaml += "\n"
        HARD_CASES_FILE.write_text(existing_yaml + append_block, encoding="utf-8")
        logger.info("Updated %s", HARD_CASES_FILE)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Integrate audited hard case candidates")
    parser.add_argument("--apply", action="store_true", help="Записать изменения в hard_cases.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    summary = integrate(apply_changes=args.apply)
    print(summary)


if __name__ == "__main__":
    main()
