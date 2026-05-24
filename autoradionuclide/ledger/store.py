"""Append-only SQLite ledger — every decision, proposal, and result is recorded here."""
from __future__ import annotations
import json, sqlite3, threading
from pathlib import Path
from typing import Optional
from autoradionuclide.domain.models import LedgerEntry, LedgerEntryType


class LedgerStore:
    """Thread-safe, append-only ledger backed by SQLite.

    The only DML allowed is INSERT — rows are never updated or deleted,
    satisfying the ALCOA-plus 'enduring and accurate' requirements.
    """

    _CREATE = """
    CREATE TABLE IF NOT EXISTS entries (
        id          TEXT PRIMARY KEY,
        timestamp   TEXT NOT NULL,
        entry_type  TEXT NOT NULL,
        campaign_id TEXT NOT NULL,
        run_id      TEXT NOT NULL DEFAULT '',
        cycle_id    TEXT NOT NULL DEFAULT '',
        construct_id TEXT,
        model_call_id TEXT,
        provenance_id TEXT NOT NULL DEFAULT '',
        data        TEXT NOT NULL,
        attribution TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_campaign ON entries (campaign_id);
    CREATE INDEX IF NOT EXISTS idx_run      ON entries (campaign_id, run_id);
    CREATE INDEX IF NOT EXISTS idx_type     ON entries (entry_type);
    CREATE INDEX IF NOT EXISTS idx_cycle    ON entries (campaign_id, cycle_id);
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self._db_path = str(db_path)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                self._db_path, check_same_thread=False
            )
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(self._CREATE)
        conn.commit()
        self._migrate_db(conn)

    def _migrate_db(self, conn: sqlite3.Connection) -> None:
        """Add new columns to existing databases without losing data."""
        try:
            conn.execute(
                "ALTER TABLE entries ADD COLUMN run_id TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists — normal on all but the very first migration

    def append(self, entry: LedgerEntry) -> None:
        """Write one immutable entry. Raises on duplicate id."""
        conn = self._conn()
        conn.execute(
            """INSERT INTO entries
               (id, timestamp, entry_type, campaign_id, run_id, cycle_id,
                construct_id, model_call_id, provenance_id, data, attribution)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                entry.id,
                entry.timestamp.isoformat(),
                entry.entry_type.value,
                entry.campaign_id,
                entry.run_id,
                entry.cycle_id,
                entry.construct_id,
                entry.model_call_id,
                entry.provenance_id,
                json.dumps(entry.data),
                entry.attribution,
            ),
        )
        conn.commit()

    def query(
        self,
        *,
        campaign_id: Optional[str] = None,
        entry_type: Optional[LedgerEntryType] = None,
        cycle_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[LedgerEntry]:
        clauses, params = [], []
        if campaign_id:
            clauses.append("campaign_id = ?")
            params.append(campaign_id)
        if entry_type:
            clauses.append("entry_type = ?")
            params.append(entry_type.value)
        if cycle_id:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn().execute(
            f"SELECT * FROM entries {where} ORDER BY timestamp LIMIT ?",
            params + [limit],
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def get(self, entry_id: str) -> Optional[LedgerEntry]:
        row = self._conn().execute(
            "SELECT * FROM entries WHERE id = ?", (entry_id,)
        ).fetchone()
        return _row_to_entry(row) if row else None

    def count(self, campaign_id: Optional[str] = None) -> int:
        if campaign_id:
            return self._conn().execute(
                "SELECT COUNT(*) FROM entries WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()[0]
        return self._conn().execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def get_most_recent_run_id(self, campaign_id: str) -> Optional[str]:
        """Return the run_id of the most recently started run for this campaign."""
        row = self._conn().execute(
            "SELECT run_id FROM entries "
            "WHERE campaign_id = ? AND run_id != '' "
            "ORDER BY timestamp DESC LIMIT 1",
            (campaign_id,),
        ).fetchone()
        return row[0] if row else None

    def list_run_ids(self, campaign_id: str) -> list[str]:
        """Return all distinct run_ids for this campaign, in chronological order."""
        rows = self._conn().execute(
            "SELECT run_id, MIN(timestamp) as first_seen FROM entries "
            "WHERE campaign_id = ? AND run_id != '' "
            "GROUP BY run_id ORDER BY first_seen",
            (campaign_id,),
        ).fetchall()
        return [r[0] for r in rows]


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    keys = row.keys()
    return LedgerEntry(
        id=row["id"],
        timestamp=row["timestamp"],
        entry_type=LedgerEntryType(row["entry_type"]),
        campaign_id=row["campaign_id"],
        run_id=(row["run_id"] or "") if "run_id" in keys else "",
        cycle_id=row["cycle_id"] or "",
        construct_id=row["construct_id"],
        model_call_id=row["model_call_id"],
        provenance_id=row["provenance_id"] or "",
        data=json.loads(row["data"]),
        attribution=row["attribution"],
    )
