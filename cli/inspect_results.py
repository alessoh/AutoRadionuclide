"""ar-inspect: inspect a campaign's ledger (analog of check_results.py)."""
from __future__ import annotations
import click
import json
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.observability.inspector import print_campaign_report, get_cycle_detail


@click.command()
@click.argument("db_path")
@click.option("--campaign-id", default=None, help="Campaign ID to inspect.")
@click.option("--run-id", default=None, help="Scope report to a single run (from ar-launch output).")
@click.option("--all-runs", is_flag=True, default=False, help="Show full campaign history across all runs.")
@click.option("--cycle-id", default=None, help="Show detail for one cycle.")
@click.option("--list-campaigns", is_flag=True, default=False, help="List all campaigns in this database.")
@click.option("--list-runs", is_flag=True, default=False, help="List all run IDs for the chosen campaign.")
def main(
    db_path: str,
    campaign_id: str | None,
    run_id: str | None,
    all_runs: bool,
    cycle_id: str | None,
    list_campaigns: bool,
    list_runs: bool,
):
    """Inspect a campaign ledger.

    By default, scopes to the most recent run. Use --run-id to inspect a
    specific run, or --all-runs to see the full campaign history across runs.
    """
    ledger = LedgerStore(db_path)

    if list_campaigns:
        rows = ledger._conn().execute(
            "SELECT DISTINCT campaign_id, COUNT(*) as n FROM entries GROUP BY campaign_id"
        ).fetchall()
        for r in rows:
            print(f"  {r[0]}: {r[1]} entries")
        return

    # Resolve campaign_id if not supplied
    if campaign_id is None:
        rows = ledger._conn().execute(
            "SELECT DISTINCT campaign_id FROM entries LIMIT 1"
        ).fetchall()
        if rows:
            campaign_id = rows[0][0]
        else:
            print("No campaigns found.")
            return

    if list_runs:
        run_ids = ledger.list_run_ids(campaign_id)
        if run_ids:
            print(f"Run IDs for campaign '{campaign_id}':")
            for rid in run_ids:
                print(f"  {rid}")
        else:
            print(f"No run IDs found for campaign '{campaign_id}' (legacy entries have no run_id).")
        return

    if cycle_id:
        detail = get_cycle_detail(ledger, campaign_id, cycle_id)
        print(json.dumps(detail, indent=2, default=str))
        return

    # Determine effective run_id for scoping:
    # --all-runs → run_id=None (aggregate)
    # --run-id X  → run_id=X (explicit)
    # default     → most recent run
    if all_runs:
        effective_run_id = None
    elif run_id:
        effective_run_id = run_id
    else:
        effective_run_id = ledger.get_most_recent_run_id(campaign_id)
        if effective_run_id is None:
            # Legacy database with no run_ids — fall back to all entries
            print("(No run_id found in database; showing all entries. "
                  "Re-run the campaign to get per-run scoping.)")
        else:
            print(f"(Showing most recent run: {effective_run_id}. "
                  f"Use --all-runs for full history or --list-runs to see all runs.)")

    print_campaign_report(ledger, campaign_id, run_id=effective_run_id)


if __name__ == "__main__":
    main()
