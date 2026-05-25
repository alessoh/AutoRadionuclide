"""Candidate generator — combines LLM-based proposal with cheminformatics filtering."""
from __future__ import annotations
import json, uuid
from typing import Any
from autoradionuclide.domain.models import (
    CandidateConstruct, TargetingVector, Chelator, Radionuclide, CandidateStatus,
    ModelRequest, LedgerEntry, LedgerEntryType
)
from autoradionuclide.providers.base import ModelProvider, PROMPT_TEMPLATES
from autoradionuclide.provenance.context import ProvenanceContext


class CandidateGenerator:
    """Proposes new candidate constructs for a campaign.

    Uses the model provider to generate diverse proposals, then validates
    and deduplicates them before returning.
    """

    def __init__(
        self,
        provider: ModelProvider,
        ledger=None,
    ) -> None:
        self._provider = provider
        self._ledger = ledger

    def generate(
        self,
        campaign_id: str,
        cycle_id: str,
        target: str,
        isotope: Radionuclide,
        n: int,
        provenance: ProvenanceContext,
        known_ids: set[str] | None = None,
        prioritized_targets: list[str] | None = None,
        run_id: str = "",
        allowed_vectors: list[str] | None = None,
        allowed_chelators: list[str] | None = None,
    ) -> list[CandidateConstruct]:
        """Ask the model provider for n candidate constructs and return validated objects.

        When allowed_vectors or allowed_chelators are provided, the generator
        (a) signals the constraint to the provider via the system prompt and
        (b) post-filters the response to enforce the declared building-block space.
        This ensures constructs outside the declared space are never returned,
        regardless of which provider is in use.
        """
        system = PROMPT_TEMPLATES["generate_candidates"].format(
            target=target, isotope=isotope.value, n=n
        )
        if prioritized_targets:
            system += f" Prioritize these targets: {', '.join(prioritized_targets)}."
        if allowed_vectors:
            system += f" Only propose these targeting vectors: {', '.join(allowed_vectors)}."
        if allowed_chelators:
            system += f" Only propose these chelators: {', '.join(allowed_chelators)}."

        request = ModelRequest(
            model=self._provider.MODEL_ID if hasattr(self._provider, "MODEL_ID") else "unknown",
            system=system,
            messages=[{"role": "user", "content": f"Propose {n} candidates for {target} with {isotope.value}."}],
            temperature=0.8,
            max_tokens=2048,
            response_format="json_object",
        )
        response = self._provider.complete(request)
        raw = _parse_list(response.content)

        constructs = []
        known_ids = known_ids or set()
        for item in raw:
            try:
                c = _item_to_construct(item, campaign_id, cycle_id, isotope, provenance)
                # Enforce declared building-block space (post-filter safety net)
                if allowed_vectors and c.targeting_vector.name not in allowed_vectors:
                    continue
                if allowed_chelators and c.chelator.name not in allowed_chelators:
                    continue
                if c.composite_key not in known_ids:
                    known_ids.add(c.composite_key)
                    constructs.append(c)
                if len(constructs) >= n:
                    break
            except Exception:
                continue

        if self._ledger is not None:
            entry = LedgerEntry(
                entry_type=LedgerEntryType.PROPOSAL,
                campaign_id=campaign_id,
                run_id=run_id,
                cycle_id=cycle_id,
                provenance_id=provenance.id,
                data={
                    "n_requested": n,
                    "n_generated": len(constructs),
                    "construct_ids": [c.id for c in constructs],
                    "model_call_id": request.request_id,
                },
            )
            self._ledger.append(entry)

        return constructs


def _parse_list(content: str) -> list[dict]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "candidates" in parsed:
            return parsed["candidates"]
        return [parsed]
    except json.JSONDecodeError:
        return []


def _item_to_construct(
    item: dict,
    campaign_id: str,
    cycle_id: str,
    isotope: Radionuclide,
    provenance: ProvenanceContext,
) -> CandidateConstruct:
    tv_data = item.get("targeting_vector", {})
    ch_data = item.get("chelator", {})
    tv = TargetingVector(
        name=tv_data.get("name", "unknown"),
        target=tv_data.get("target", "unknown"),
        vector_type=tv_data.get("vector_type", "small_molecule"),
        smiles=tv_data.get("smiles"),
        notes=tv_data.get("notes", ""),
    )
    ch = Chelator(
        name=ch_data.get("name", "DOTA"),
        smiles=ch_data.get("smiles"),
        compatible_isotopes=ch_data.get("compatible_isotopes", []),
    )
    return CandidateConstruct(
        id=str(uuid.uuid4()),
        name=item.get("name", f"{tv.name}-{ch.name}"),
        targeting_vector=tv,
        chelator=ch,
        linker=item.get("linker"),
        radionuclide=isotope,
        status=CandidateStatus.PROPOSED,
        provenance_id=provenance.id,
        generation_reasoning=item.get("generation_reasoning", ""),
        campaign_id=campaign_id,
    )
