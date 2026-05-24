"""Observability — structured logging and campaign inspection (analog of check_results.py)."""
from __future__ import annotations
import json
from datetime import datetime
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.domain.models import LedgerEntryType


def inspect_campaign(ledger: LedgerStore, campaign_id: str) -> dict:
    """Return a structured summary of a campaign from its ledger entries."""
    entries = ledger.query(campaign_id=campaign_id, limit=10_000)
    by_type: dict[str, int] = {}
    cycles: set[str] = set()
    scores: list[float] = []
    model_calls = 0
    errors = 0

    for e in entries:
        by_type[e.entry_type.value] = by_type.get(e.entry_type.value, 0) + 1
        if e.cycle_id:
            cycles.add(e.cycle_id)
        if e.entry_type == LedgerEntryType.CYCLE_SUMMARY:
            score = e.data.get("campaign_score_after")
            if score is not None:
                scores.append(float(score))
        if e.entry_type == LedgerEntryType.MODEL_CALL:
            model_calls += 1
        if e.entry_type == LedgerEntryType.ERROR:
            errors += 1

    return {
        "campaign_id": campaign_id,
        "total_entries": len(entries),
        "entry_counts_by_type": by_type,
        "n_cycles": len(cycles),
        "n_model_calls": model_calls,
        "n_errors": errors,
        "score_history": scores,
        "final_score": scores[-1] if scores else None,
        "score_improvement": (scores[-1] - scores[0]) if len(scores) >= 2 else None,
    }


def print_campaign_report(ledger: LedgerStore, campaign_id: str) -> None:
    summary = inspect_campaign(ledger, campaign_id)
    print(f"\n{'='*60}")
    print(f"Campaign Report: {campaign_id}")
    print(f"{'='*60}")
    print(f"Total ledger entries : {summary['total_entries']}")
    print(f"Cycles completed     : {summary['n_cycles']}")
    print(f"Model calls          : {summary['n_model_calls']}")
    print(f"Errors               : {summary['n_errors']}")
    if summary["score_history"]:
        print(f"Score history        : {[f'{s:.3f}' for s in summary['score_history']]}")
        print(f"Final score          : {summary['final_score']:.3f}")
        if summary["score_improvement"] is not None:
            print(f"Total improvement    : {summary['score_improvement']:+.3f}")
    print(f"\nEntry counts by type:")
    for t, n in sorted(summary["entry_counts_by_type"].items()):
        print(f"  {t:<25} {n}")
    print(f"{'='*60}\n")


def get_cycle_detail(ledger: LedgerStore, campaign_id: str, cycle_id: str) -> dict:
    """Return all ledger entries for one cycle, structured for human inspection."""
    entries = ledger.query(campaign_id=campaign_id, cycle_id=cycle_id)
    return {
        "cycle_id": cycle_id,
        "entries": [
            {
                "id": e.id,
                "timestamp": (
                    e.timestamp.isoformat()
                    if hasattr(e.timestamp, "isoformat")
                    else str(e.timestamp)
                ),
                "type": e.entry_type.value,
                "data": e.data,
            }
            for e in entries
        ],
    }
