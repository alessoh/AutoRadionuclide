"""Tests for the LedgerStore."""
from __future__ import annotations
import tempfile
from pathlib import Path
import pytest
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.domain.models import LedgerEntry, LedgerEntryType


def _make_entry(**kwargs) -> LedgerEntry:
    defaults = dict(
        entry_type=LedgerEntryType.PROPOSAL,
        campaign_id="test-campaign",
        cycle_id="cycle-1",
        data={"test": "data"},
    )
    defaults.update(kwargs)
    return LedgerEntry(**defaults)


class TestLedgerStoreInMemory:
    def test_empty_store_count(self):
        ledger = LedgerStore(":memory:")
        assert ledger.count() == 0

    def test_append_and_count(self):
        ledger = LedgerStore(":memory:")
        entry = _make_entry()
        ledger.append(entry)
        assert ledger.count() == 1

    def test_append_multiple(self):
        ledger = LedgerStore(":memory:")
        for i in range(5):
            ledger.append(_make_entry(data={"i": i}))
        assert ledger.count() == 5

    def test_get_by_id(self):
        ledger = LedgerStore(":memory:")
        entry = _make_entry()
        ledger.append(entry)
        retrieved = ledger.get(entry.id)
        assert retrieved is not None
        assert retrieved.id == entry.id
        assert retrieved.campaign_id == entry.campaign_id

    def test_get_nonexistent_returns_none(self):
        ledger = LedgerStore(":memory:")
        assert ledger.get("nonexistent-id") is None

    def test_duplicate_id_raises(self):
        ledger = LedgerStore(":memory:")
        entry = _make_entry()
        ledger.append(entry)
        with pytest.raises(Exception):
            ledger.append(entry)  # same id

    def test_query_by_campaign_id(self):
        ledger = LedgerStore(":memory:")
        ledger.append(_make_entry(campaign_id="camp-A"))
        ledger.append(_make_entry(campaign_id="camp-B"))
        ledger.append(_make_entry(campaign_id="camp-A"))
        results = ledger.query(campaign_id="camp-A")
        assert len(results) == 2
        assert all(e.campaign_id == "camp-A" for e in results)

    def test_query_by_entry_type(self):
        ledger = LedgerStore(":memory:")
        ledger.append(_make_entry(entry_type=LedgerEntryType.PROPOSAL))
        ledger.append(_make_entry(entry_type=LedgerEntryType.SCORE))
        ledger.append(_make_entry(entry_type=LedgerEntryType.PROPOSAL))
        results = ledger.query(entry_type=LedgerEntryType.PROPOSAL)
        assert len(results) == 2

    def test_query_by_cycle_id(self):
        ledger = LedgerStore(":memory:")
        ledger.append(_make_entry(cycle_id="cyc-1"))
        ledger.append(_make_entry(cycle_id="cyc-2"))
        ledger.append(_make_entry(cycle_id="cyc-1"))
        results = ledger.query(cycle_id="cyc-1")
        assert len(results) == 2

    def test_query_combined_filters(self):
        ledger = LedgerStore(":memory:")
        ledger.append(_make_entry(campaign_id="camp-A", entry_type=LedgerEntryType.PROPOSAL))
        ledger.append(_make_entry(campaign_id="camp-A", entry_type=LedgerEntryType.SCORE))
        ledger.append(_make_entry(campaign_id="camp-B", entry_type=LedgerEntryType.PROPOSAL))
        results = ledger.query(campaign_id="camp-A", entry_type=LedgerEntryType.PROPOSAL)
        assert len(results) == 1

    def test_query_limit(self):
        ledger = LedgerStore(":memory:")
        for _ in range(10):
            ledger.append(_make_entry())
        results = ledger.query(limit=3)
        assert len(results) == 3

    def test_count_by_campaign(self):
        ledger = LedgerStore(":memory:")
        ledger.append(_make_entry(campaign_id="camp-A"))
        ledger.append(_make_entry(campaign_id="camp-A"))
        ledger.append(_make_entry(campaign_id="camp-B"))
        assert ledger.count("camp-A") == 2
        assert ledger.count("camp-B") == 1
        assert ledger.count() == 3

    def test_data_preserved(self):
        ledger = LedgerStore(":memory:")
        entry = _make_entry(data={"nested": {"key": "value"}, "number": 42})
        ledger.append(entry)
        retrieved = ledger.get(entry.id)
        assert retrieved.data["nested"]["key"] == "value"
        assert retrieved.data["number"] == 42

    def test_immutability_no_update_method(self):
        ledger = LedgerStore(":memory:")
        # Verify there's no update/delete method
        assert not hasattr(ledger, "update")
        assert not hasattr(ledger, "delete")


class TestLedgerStoreFileBased:
    def test_persist_and_reload(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Write
            ledger1 = LedgerStore(db_path)
            entry = _make_entry(campaign_id="persistent-campaign")
            ledger1.append(entry)
            assert ledger1.count() == 1
            ledger1._conn().close()

            # Read in new instance
            ledger2 = LedgerStore(db_path)
            assert ledger2.count() == 1
            retrieved = ledger2.get(entry.id)
            assert retrieved is not None
            assert retrieved.campaign_id == "persistent-campaign"
            ledger2._conn().close()

    def test_multiple_sessions_accumulate(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            for i in range(3):
                ledger = LedgerStore(db_path)
                ledger.append(_make_entry(data={"session": i}))
                ledger._conn().close()

            ledger_final = LedgerStore(db_path)
            assert ledger_final.count() == 3
            ledger_final._conn().close()
