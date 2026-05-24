"""ar-inspect: inspect a campaign's ledger (analog of check_results.py)."""
from __future__ import annotations
import click
from pathlib import Path
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.observability.inspector import print_campaign_report, get_cycle_detail
import json


@click.command()
@click.argument("db_path")
@click.option("--campaign-id", default=None, help="Campaign ID to inspect.")
@click.option("--cycle-id", default=None, help="Show detail for one cycle.")
@click.option("--list-campaigns", is_flag=True, default=False)
def main(db_path: str, campaign_id: str | None, cycle_id: str | None, list_campaigns: bool):
    """Inspect a campaign ledger (analog of check_results.py)."""
    ledger = LedgerStore(db_path)
    if list_campaigns:
        rows = ledger._conn().execute(
            "SELECT DISTINCT campaign_id, COUNT(*) as n FROM entries GROUP BY campaign_id"
        ).fetchall()
        for r in rows:
            print(f"  {r[0]}: {r[1]} entries")
        return
    if campaign_id is None:
        rows = ledger._conn().execute(
            "SELECT DISTINCT campaign_id FROM entries LIMIT 1"
        ).fetchall()
        if rows:
            campaign_id = rows[0][0]
        else:
            print("No campaigns found.")
            return
    if cycle_id:
        detail = get_cycle_detail(ledger, campaign_id, cycle_id)
        print(json.dumps(detail, indent=2, default=str))
    else:
        print_campaign_report(ledger, campaign_id)


if __name__ == "__main__":
    main()
