"""Observability — structured campaign inspection (analog of check_results.py).

Run-scoped reporting: every report is scoped to a single run_id by default.
Pass run_id=None explicitly to aggregate across all runs of a campaign.
"""
from __future__ import annotations
import json
from typing import Optional
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.domain.models import LedgerEntryType


def inspect_campaign(
    ledger: LedgerStore,
    campaign_id: str,
    run_id: Optional[str] = None,
) -> dict:
    """Return a structured summary of a campaign from its ledger entries.

    Args:
        ledger:      The ledger store to query.
        campaign_id: Campaign identifier.
        run_id:      If provided, scope the report to this single run.
                     If None, aggregate across all runs (shows full campaign history).
    """
    entries = ledger.query(campaign_id=campaign_id, run_id=run_id, limit=10_000)

    by_type: dict[str, int] = {}
    cycles: set[str] = set()
    cycle_summaries: list[dict] = []
    strategy_mods: dict[str, dict] = {}  # cycle_id → STRATEGY_MODIFICATION data
    model_calls = 0
    errors = 0

    for e in entries:
        by_type[e.entry_type.value] = by_type.get(e.entry_type.value, 0) + 1
        if e.cycle_id:
            cycles.add(e.cycle_id)
        if e.entry_type == LedgerEntryType.CYCLE_SUMMARY:
            cycle_summaries.append(e.data)
        if e.entry_type == LedgerEntryType.STRATEGY_MODIFICATION:
            strategy_mods[e.cycle_id] = e.data
        if e.entry_type == LedgerEntryType.MODEL_CALL:
            model_calls += 1
        if e.entry_type == LedgerEntryType.ERROR:
            errors += 1

    # Sort cycle summaries chronologically by cycle_number
    cycle_summaries.sort(key=lambda s: s.get("cycle_number", 0))

    # Build kept-score history: for each cycle, the score the outer loop actually
    # committed after the keep-or-revert decision.
    # Turn 0 (no modification): always commits the inner-cycle score.
    # Turn N > 0: commits inner-cycle score if kept, reverts to before if not.
    kept_score_history: list[float] = []
    n_turns = len(cycle_summaries)
    n_kept = 0
    n_reverted = 0

    for summary in cycle_summaries:
        cycle_id = summary.get("cycle_id", "")
        mod = strategy_mods.get(cycle_id)
        if mod is None:
            # Turn 0 or no modification proposed: score always committed
            committed = summary.get("campaign_score_after", 0.0)
        elif mod.get("kept", True):
            committed = summary.get("campaign_score_after", 0.0)
            n_kept += 1
        else:
            committed = summary.get("campaign_score_before", 0.0)
            n_reverted += 1
        kept_score_history.append(float(committed))

    total_improvement = (
        (kept_score_history[-1] - kept_score_history[0])
        if len(kept_score_history) >= 2
        else None
    )

    return {
        "campaign_id": campaign_id,
        "run_id": run_id,
        "total_entries": len(entries),
        "entry_counts_by_type": by_type,
        "n_outer_turns": n_turns,       # distinct from inner cycles for clarity
        "n_inner_cycles": len(cycles),  # unique cycle_ids (1 per turn)
        "n_strategy_modifications_proposed": len(strategy_mods),
        "n_modifications_kept": n_kept,
        "n_modifications_reverted": n_reverted,
        "n_model_calls": model_calls,
        "n_errors": errors,
        "kept_score_history": kept_score_history,
        "final_kept_score": kept_score_history[-1] if kept_score_history else None,
        "total_improvement": total_improvement,
    }


def print_campaign_report(
    ledger: LedgerStore,
    campaign_id: str,
    run_id: Optional[str] = None,
) -> None:
    """Print a human-readable campaign report, scoped to a single run by default."""
    summary = inspect_campaign(ledger, campaign_id, run_id=run_id)

    scope_label = f"run {run_id}" if run_id else "ALL runs (full campaign history)"
    width = 62

    print(f"\n{'=' * width}")
    print(f"Campaign Report: {campaign_id}")
    print(f"Scope: {scope_label}")
    print(f"{'=' * width}")
    print(f"Total ledger entries : {summary['total_entries']}")
    print(f"Outer turns          : {summary['n_outer_turns']}")
    print(f"  inner cycles run   : {summary['n_inner_cycles']}")
    print(f"  modifications proposed: {summary['n_strategy_modifications_proposed']}")
    print(f"  kept               : {summary['n_modifications_kept']}")
    print(f"  reverted           : {summary['n_modifications_reverted']}")
    print(f"Model calls          : {summary['n_model_calls']}")
    print(f"Errors               : {summary['n_errors']}")
    if summary["kept_score_history"]:
        hist = [f"{s:.3f}" for s in summary["kept_score_history"]]
        print(f"Kept score history   : {hist}")
        print(f"Final kept score     : {summary['final_kept_score']:.3f}")
        if summary["total_improvement"] is not None:
            print(f"Total improvement    : {summary['total_improvement']:+.3f}")
            if summary["total_improvement"] < -1e-9:
                print(
                    "  NOTE: negative improvement — raw inner-cycle scores are in "
                    "CYCLE_SUMMARY ledger entries; the kept score should be "
                    "non-decreasing with the current outer loop."
                )
    print(f"\nEntry counts by type:")
    for t, n in sorted(summary["entry_counts_by_type"].items()):
        print(f"  {t:<30} {n}")
    print(f"{'=' * width}\n")


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
                "run_id": e.run_id,
                "data": e.data,
            }
            for e in entries
        ],
    }
