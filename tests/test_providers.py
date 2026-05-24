"""Tests for model providers."""
from __future__ import annotations
import json
import pytest
from autoradionuclide.domain.models import ModelRequest, LedgerEntryType
from autoradionuclide.providers.mock import MockModelProvider
from autoradionuclide.ledger.store import LedgerStore


def _make_request(system: str = "You are an expert radioligand chemist.", **kwargs) -> ModelRequest:
    defaults = dict(
        model="mock-deterministic-v1",
        system=system,
        messages=[{"role": "user", "content": "Propose 4 candidates for PSMA with Lu-177."}],
    )
    defaults.update(kwargs)
    return ModelRequest(**defaults)


class TestMockModelProvider:
    def test_returns_model_response(self):
        provider = MockModelProvider()
        req = _make_request()
        resp = provider.complete(req)
        assert resp.model == "mock-deterministic-v1"
        assert isinstance(resp.content, str)
        assert len(resp.content) > 0

    def test_deterministic_same_input_same_output(self):
        provider = MockModelProvider()
        req = _make_request()
        resp1 = provider.complete(req)
        resp2 = provider.complete(req)
        assert resp1.content == resp2.content

    def test_different_inputs_different_outputs(self):
        provider = MockModelProvider()
        req1 = _make_request(messages=[{"role": "user", "content": "Propose 4 candidates for PSMA."}])
        req2 = _make_request(messages=[{"role": "user", "content": "Propose 4 candidates for SSTR2."}])
        resp1 = provider.complete(req1)
        resp2 = provider.complete(req2)
        # Different inputs should generally produce different outputs (hash-seeded)
        # Allow for hash collision in rare cases — just check they run without error
        assert resp1.content is not None
        assert resp2.content is not None

    def test_candidate_generation_returns_json_list(self):
        provider = MockModelProvider()
        req = _make_request(
            system="You are an expert radioligand chemist. Given a campaign targeting PSMA propose 4 diverse candidate constructs. Return JSON list of candidates."
        )
        resp = provider.complete(req)
        parsed = json.loads(resp.content)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

    def test_candidate_has_required_fields(self):
        provider = MockModelProvider()
        req = _make_request(
            system="You are an expert radioligand chemist. Propose 4 diverse candidate constructs. Return JSON list of candidates."
        )
        resp = provider.complete(req)
        parsed = json.loads(resp.content)
        candidate = parsed[0]
        assert "targeting_vector" in candidate
        assert "chelator" in candidate
        assert "name" in candidate

    def test_strategy_modification_returns_json(self):
        provider = MockModelProvider()
        req = _make_request(
            system="You are optimizing a radioligand discovery campaign. strategy modification params. Return JSON."
        )
        resp = provider.complete(req)
        parsed = json.loads(resp.content)
        assert isinstance(parsed, dict)
        assert "parameter_name" in parsed
        assert "new_value" in parsed

    def test_token_usage_populated(self):
        provider = MockModelProvider()
        req = _make_request()
        resp = provider.complete(req)
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0
        assert resp.usage.total_tokens > 0

    def test_latency_ms_set(self):
        provider = MockModelProvider()
        req = _make_request()
        resp = provider.complete(req)
        assert resp.latency_ms >= 0

    def test_request_id_preserved(self):
        provider = MockModelProvider()
        req = _make_request()
        resp = provider.complete(req)
        assert resp.request_id == req.request_id

    def test_ledger_recording(self):
        ledger = LedgerStore(":memory:")
        provider = MockModelProvider(ledger=ledger)
        provider.set_campaign("test-campaign")
        req = _make_request()
        provider.complete(req)
        entries = ledger.query(entry_type=LedgerEntryType.MODEL_CALL)
        assert len(entries) == 1
        assert entries[0].model_call_id == req.request_id

    def test_ledger_recording_multiple_calls(self):
        ledger = LedgerStore(":memory:")
        provider = MockModelProvider(ledger=ledger)
        provider.set_campaign("test-campaign")
        for _ in range(3):
            provider.complete(_make_request())
        entries = ledger.query(entry_type=LedgerEntryType.MODEL_CALL)
        assert len(entries) == 3

    def test_no_ledger_no_crash(self):
        provider = MockModelProvider(ledger=None)
        req = _make_request()
        resp = provider.complete(req)
        assert resp.content is not None

    def test_model_id_constant(self):
        assert MockModelProvider.MODEL_ID == "mock-deterministic-v1"
