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
        cycle_id    TEXT NOT NULL DEFAULT '',
        construct_id TEXT,
        model_call_id TEXT,
        provenance_id TEXT NOT NULL DEFAULT '',
        data        TEXT NOT NULL,
        attribution TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_campaign ON entries (campaign_id);
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

    def append(self, entry: LedgerEntry) -> None:
        """Write one immutable entry. Raises on duplicate id."""
        conn = self._conn()
        conn.execute(
            """INSERT INTO entries
               (id, timestamp, entry_type, campaign_id, cycle_id,
                construct_id, model_call_id, provenance_id, data, attribution)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                entry.id,
                entry.timestamp.isoformat(),
                entry.entry_type.value,
                entry.campaign_id,
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


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        id=row["id"],
        timestamp=row["timestamp"],
        entry_type=LedgerEntryType(row["entry_type"]),
        campaign_id=row["campaign_id"],
        cycle_id=row["cycle_id"] or "",
        construct_id=row["construct_id"],
        model_call_id=row["model_call_id"],
        provenance_id=row["provenance_id"] or "",
        data=json.loads(row["data"]),
        attribution=row["attribution"],
    )
