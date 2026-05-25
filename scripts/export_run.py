"""
Export a recorded campaign run to a self-contained JSON file for the web dashboard.

Usage:
    python scripts/export_run.py                          # uses defaults
    python scripts/export_run.py --db mibg_demo.db --run-id 16140108
    python scripts/export_run.py --out web/src/data/run_export.json

The output JSON is read at build time by the Next.js dashboard — no backend,
no secrets, no API calls from the browser.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from autoradionuclide.domain.models import CandidateConstruct, Chelator, Radionuclide, TargetingVector
from autoradionuclide.featurization.featurizer import DESCRIPTOR_NAMES, FEATURIZER_VERSION, featurize
from autoradionuclide.featurization.registry import _TARGETING_VECTOR_REGISTRY
from frozen.benchmark_runner import run_benchmark


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a ledger run to JSON for the web dashboard.")
    p.add_argument("--db", default=str(ROOT / "mibg_demo.db"), help="Path to the SQLite ledger")
    p.add_argument("--run-id", default="16140108", help="Run ID to export")
    p.add_argument("--out", default=str(ROOT / "web" / "src" / "data" / "run_export.json"),
                   help="Output JSON path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Ledger reader
# ---------------------------------------------------------------------------

def _read_entries(db_path: str, run_id: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT entry_type, cycle_id, timestamp, data FROM entries "
        "WHERE run_id = ? ORDER BY timestamp",
        (run_id,),
    ).fetchall()
    conn.close()
    return [
        {"entry_type": r["entry_type"], "cycle_id": r["cycle_id"],
         "timestamp": r["timestamp"], "data": json.loads(r["data"])}
        for r in rows
    ]


def _count_by_type(db_path: str) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT entry_type, COUNT(*) as cnt FROM entries GROUP BY entry_type"
    ).fetchall()
    conn.close()
    return {r["entry_type"]: r["cnt"] for r in rows}


def _count_model_calls(entries: list[dict]) -> int:
    return sum(1 for e in entries if e["entry_type"] == "model_call")


# ---------------------------------------------------------------------------
# Featurize MIBG+none+I-131 directly (bypass ledger — use real RDKit)
# ---------------------------------------------------------------------------

def _featurize_mibg() -> dict:
    construct = CandidateConstruct(
        name="MIBG-none-0",
        targeting_vector=TargetingVector(name="MIBG", target="NET", vector_type="small_molecule"),
        chelator=Chelator(name="none", compatible_isotopes=["I-131", "At-211"]),
        linker=None,
        radionuclide=Radionuclide.I131,
    )
    record = featurize(construct)
    descriptors = {
        name: round(float(record.descriptor_vector[i]), 4)
        for i, name in enumerate(DESCRIPTOR_NAMES)
    }
    return {
        "quality": record.quality.value,
        "featurizer_version": record.featurizer_version,
        "rdkit_version": record.rdkit_version,
        "descriptor_names": DESCRIPTOR_NAMES,
        "descriptors": descriptors,
        "fingerprint_bits": int(record.fingerprint.sum()),
        "fingerprint_params": record.fingerprint_params,
        "resolution_reasons": record.resolution_reasons,
        "isotope_features": {
            "atomic_number": int(record.isotope_features[0]),
            "half_life_days": round(float(record.isotope_features[1]), 4),
            "decay_mode_encoded": int(record.isotope_features[2]),
        },
    }


# ---------------------------------------------------------------------------
# Build turn list from ledger entries
# ---------------------------------------------------------------------------

def _build_turns(entries: list[dict]) -> list[dict]:
    # Group cycle_summary and strategy_modification by cycle_id
    summaries: dict[str, dict] = {}
    mods: dict[str, dict] = {}
    for e in entries:
        if e["entry_type"] == "cycle_summary":
            summaries[e["cycle_id"]] = e["data"]
        elif e["entry_type"] == "strategy_modification":
            mods[e["cycle_id"]] = e["data"]

    turns = []
    for cycle_id, s in sorted(summaries.items(), key=lambda kv: kv[1]["cycle_number"]):
        mod_data = mods.get(cycle_id)
        mod = None
        if mod_data:
            m = mod_data["modification"]
            mod = {
                "description": m["modification_description"],
                "parameter": m["parameter_name"],
                "old_value": m["old_value"],
                "new_value": m["new_value"],
                "rationale": m["rationale"],
            }
        turns.append({
            "turn": s["cycle_number"] + 1,
            "cycle_id": cycle_id,
            "cycle_number": s["cycle_number"],
            "constructs_proposed": s["constructs_proposed"],
            "constructs_selected": s["constructs_selected"],
            "score_before": round(s["campaign_score_before"], 6),
            "score_after": round(s["campaign_score_after"], 6),
            "score_delta": round(s["score_delta"], 6),
            "strategy_modification": mod,
            "kept": s.get("strategy_kept"),
            "inner_loop_note": s.get("rationale", ""),
            "started_at": s["started_at"],
            "finished_at": s["finished_at"],
        })
    return turns


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def build_export(db_path: str, run_id: str) -> dict:
    entries = _read_entries(db_path, run_id)
    if not entries:
        raise SystemExit(f"No entries found for run_id={run_id!r} in {db_path}")

    total_entries_db = sum(_count_by_type(db_path).values())

    # First timestamp = run start
    run_start = entries[0]["timestamp"]
    model_id = next(
        (e["data"]["model"] for e in entries if e["entry_type"] == "model_call"), "unknown"
    )

    # Objective scores from the SCORE entry (cycle 0)
    score_entry = next(
        (e["data"] for e in entries if e["entry_type"] == "score"), {}
    )
    objectives_raw = score_entry.get("objectives", {})
    objectives = {k: round(v, 6) for k, v in objectives_raw.items()}
    objectives["aggregate"] = round(score_entry.get("aggregate_score", 0.0), 6)

    # Stub result values (with experimental noise) from the RESULT entry
    result_entry = next(
        (e["data"] for e in entries if e["entry_type"] == "result"), {}
    )
    stub_results = {}
    for cr in result_entry.get("construct_results", []):
        stub_results[cr["assay_name"]] = {
            "value": round(cr["value"], 6),
            "uncertainty": round(cr["uncertainty"], 6),
            "passed": cr["passed"],
        }

    mibg_registry = _TARGETING_VECTOR_REGISTRY["MIBG"]
    featurization = _featurize_mibg()

    construct = {
        "name": "MIBG-none-0",
        "targeting_vector": "MIBG",
        "chelator": "none",
        "isotope": "I-131",
        "smiles": mibg_registry["smiles"],
        "formula": mibg_registry["formula"],
        "registry_source": mibg_registry["source"],
        "clinical_name": "iobenguane (Azedra)",
        "mechanism": "Norepinephrine transporter (NET) ligand; directly radioiodinated",
        "featurization": featurization,
        "objectives": objectives,
        "stub_results": stub_results,
    }

    turns = _build_turns(entries)

    benchmark = run_benchmark()
    # Round all floats in rankings
    for r in benchmark["rankings"]:
        r["score"] = round(r["score"], 6)

    provenance = {
        "model_id": model_id,
        "featurizer_version": FEATURIZER_VERSION,
        "smiles_source": mibg_registry["source"],
        "total_model_calls_this_run": _count_model_calls(entries),
        "total_ledger_entries_this_run": len(entries),
        "total_ledger_entries_all_runs": total_entries_db,
        "run_id": run_id,
        "run_started_at": run_start,
    }

    honest_limits = [
        "Scoring functions are frozen heuristics — not validated predictive models.",
        "Metal coordination chemistry is NOT modeled (coordination geometry, thermodynamic stability, kinetic inertness).",
        "Radiation dose profile (LET, β⁻/α/Auger particle energy, DNA damage) is NOT captured.",
        "Benchmark rank accuracy 0.57 vs. random baseline 0.44 — confirms wiring, not predictive power.",
        "StubWetLab returns frozen-harness scores plus Gaussian noise — no real radiochemistry.",
        "DOTA and NOTA produce identical Morgan-2 fingerprints (ring-size difference invisible at radius 2).",
        "Large peptide targeting vectors (DOTATATE, PSMA-617, FAPI-46) omitted from registry pending independent verification.",
    ]

    return {
        "meta": {
            "campaign_id": "mibg-net-demo-001",
            "campaign_name": "MIBG/NET Flagship Demo Campaign",
            "run_id": run_id,
            "run_started_at": run_start,
            "export_generated_at": datetime.now(timezone.utc).isoformat(),
            "export_script": "scripts/export_run.py",
        },
        "construct": construct,
        "turns": turns,
        "benchmark": {
            "rank_accuracy": round(benchmark["rank_accuracy"], 4),
            "baseline_accuracy": round(benchmark["baseline_accuracy"], 4),
            "engine_hits": benchmark["engine_hits"],
            "n_positive": benchmark["n_positive"],
            "interpretation": benchmark["interpretation"],
            "rankings": benchmark["rankings"],
        },
        "provenance": provenance,
        "honest_limits": honest_limits,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    print(f"Reading {args.db} run_id={args.run_id} ...")
    export = build_export(args.db, args.run_id)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(export, indent=2))
    print(f"Wrote {out_path} ({out_path.stat().st_size:,} bytes)")

    # Quick sanity check
    required_keys = {"meta", "construct", "turns", "benchmark", "provenance", "honest_limits"}
    missing = required_keys - set(export.keys())
    if missing:
        raise SystemExit(f"ERROR: missing top-level keys: {missing}")
    print(f"Keys OK: {sorted(export.keys())}")
    print(f"Turns: {len(export['turns'])}")
    print(f"Benchmark rankings: {len(export['benchmark']['rankings'])}")
    print(f"Feature quality: {export['construct']['featurization']['quality']}")
    print("Done.")
