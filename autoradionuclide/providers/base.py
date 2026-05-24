"""Model-provider abstraction — the single boundary all LLM calls cross."""
from __future__ import annotations
import time, uuid
from abc import ABC, abstractmethod
from typing import Any, Optional
from autoradionuclide.domain.models import ModelRequest, ModelResponse, LedgerEntry, LedgerEntryType
from autoradionuclide.provenance.context import ProvenanceContext


class ModelProvider(ABC):
    """Every LLM backend implements exactly this interface.

    Concrete adapters MUST call _record_call() so every exchange lands
    in the ledger before the caller processes the response.
    """

    def __init__(self, ledger=None) -> None:
        self._ledger = ledger   # LedgerStore | None
        self._run_id: str = ""

    def complete(self, request: ModelRequest) -> ModelResponse:
        t0 = time.monotonic()
        response = self._do_complete(request)
        response.latency_ms = (time.monotonic() - t0) * 1000
        if self._ledger is not None:
            self._record_call(request, response)
        return response

    @abstractmethod
    def _do_complete(self, request: ModelRequest) -> ModelResponse:
        ...

    def _record_call(self, req: ModelRequest, resp: ModelResponse) -> None:
        entry = LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type=LedgerEntryType.MODEL_CALL,
            campaign_id=getattr(self, "_campaign_id", ""),
            run_id=self._run_id,
            model_call_id=req.request_id,
            data={
                "model": req.model,
                "system": req.system,
                "messages": req.messages,
                "response": resp.content[:500],   # truncate for ledger
                "usage": resp.usage.model_dump(),
                "latency_ms": resp.latency_ms,
            },
        )
        self._ledger.append(entry)

    def set_campaign(self, campaign_id: str) -> None:
        self._campaign_id = campaign_id

    def set_run_id(self, run_id: str) -> None:
        self._run_id = run_id


PROMPT_TEMPLATES: dict[str, str] = {
    "generate_candidates": (
        "You are an expert radioligand chemist. Given a campaign targeting {target} "
        "using {isotope}, propose {n} diverse candidate constructs. Each candidate "
        "must have a targeting_vector (name, target, vector_type), a chelator (name), "
        "an optional linker, and generation_reasoning. Return JSON list of candidates."
    ),
    "strategy_modification": (
        "You are optimizing a radioligand discovery campaign targeting {target}. "
        "Current campaign score: {score:.3f}. Recent cycle deltas: {deltas}. "
        "Strategy parameters: {params}. "
        "Propose ONE specific modification to improve discovery efficiency. "
        "Return JSON: {{modification_description, parameter_name, old_value, new_value, rationale}}"
    ),
    "cycle_rationale": (
        "Summarize the key learning from discovery cycle {cycle_id}. "
        "Score delta: {delta:+.3f}. Top candidates: {top_candidates}. "
        "Return a single paragraph explanation."
    ),
}
