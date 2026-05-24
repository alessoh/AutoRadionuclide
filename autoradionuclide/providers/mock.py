"""Deterministic mock provider — fully offline, no API key required."""
from __future__ import annotations
import hashlib, json, random
from typing import Any
from autoradionuclide.domain.models import ModelRequest, ModelResponse, TokenUsage
from autoradionuclide.providers.base import ModelProvider

# Canned candidate components the mock cycles through deterministically.
# Registry coverage:
#   Chelators with SMILES: DOTA, NOTA, DOTAGA (→ PARTIAL quality when paired)
#   Targeting vectors with SMILES: MIBG (→ FULL quality when chelator="none")
_VECTORS = [
    {"name": "PSMA-617",   "target": "PSMA",        "vector_type": "small_molecule"},
    {"name": "DOTATATE",   "target": "SSTR2",        "vector_type": "peptide"},
    {"name": "FAPI-46",    "target": "FAP",          "vector_type": "small_molecule"},
    {"name": "RGD-peptide","target": "integrin_avb3","vector_type": "peptide"},
    {"name": "PSMA-I&T",   "target": "PSMA",         "vector_type": "peptide"},
    {"name": "DOTATOC",    "target": "SSTR2",        "vector_type": "peptide"},
    {"name": "MIP-1072",   "target": "PSMA",         "vector_type": "small_molecule"},
    {"name": "FAPI-74",    "target": "FAP",          "vector_type": "small_molecule"},
    # MIBG: directly radioiodinated NET ligand; paired only with chelator="none"
    {"name": "MIBG",       "target": "NET",          "vector_type": "small_molecule"},
]
_CHELATORS = [
    {"name": "DOTA",   "compatible_isotopes": ["Lu-177", "Ac-225", "Y-90", "Ga-68"]},
    {"name": "NOTA",   "compatible_isotopes": ["Ga-68", "Cu-64"]},
    {"name": "DOTAGA", "compatible_isotopes": ["Lu-177", "Ac-225", "Y-90"]},
    {"name": "PSMA",   "compatible_isotopes": ["Lu-177"]},
    # Direct labeling: no separate chelator (e.g. I-131 or At-211 bonded to MIBG)
    {"name": "none",   "compatible_isotopes": ["I-131", "At-211"]},
]
_LINKERS = [None, "PEG2", "PEG4", "alkyl-C3"]


class MockModelProvider(ModelProvider):
    """Returns deterministic responses seeded from the request content hash.

    This provider is the default for offline testing. It produces plausible
    JSON that passes schema validation without any network call.
    """

    MODEL_ID = "mock-deterministic-v1"

    def __init__(self, ledger=None) -> None:
        super().__init__(ledger)

    def _do_complete(self, request: ModelRequest) -> ModelResponse:
        seed = _hash_seed(request)
        rng = random.Random(seed)
        content = self._dispatch(request, rng)
        return ModelResponse(
            request_id=request.request_id,
            model=self.MODEL_ID,
            content=content,
            usage=TokenUsage(prompt_tokens=200, completion_tokens=300, total_tokens=500),
            cached=False,
        )

    def _dispatch(self, request: ModelRequest, rng: random.Random) -> str:
        system = (request.system or "").lower()
        # Check strategy modification first — its prompt also contains "propose"
        if "modification_description" in system or (
            "modification" in system and "strategy parameters" in system
        ):
            return self._strategy_modification(request, rng)
        if "propose" in system or "candidate" in system or "generate" in system:
            return self._generate_candidates(request, rng)
        if "modification" in system or "strategy" in system:
            return self._strategy_modification(request, rng)
        return self._generic_rationale(rng)

    def _generate_candidates(self, request: ModelRequest, rng: random.Random) -> str:
        # Parse n from messages if present
        n = 4
        for msg in request.messages:
            content = msg.get("content", "")
            if isinstance(content, str) and "propose" in content.lower():
                import re
                m = re.search(r"propose\s+(\d+)", content, re.I)
                if m:
                    n = int(m.group(1))
        candidates = []
        for i in range(n):
            vec = rng.choice(_VECTORS)
            chel = rng.choice(_CHELATORS)
            linker = rng.choice(_LINKERS)
            candidates.append({
                "targeting_vector": {**vec, "smiles": None, "notes": ""},
                "chelator": {**chel},
                "linker": linker,
                "name": f"{vec['name']}-{chel['name']}-{i}",
                "generation_reasoning": (
                    f"Targeting {vec['target']} with {chel['name']} "
                    f"based on current campaign knowledge."
                ),
            })
        return json.dumps(candidates)

    def _strategy_modification(self, request: ModelRequest, rng: random.Random) -> str:
        options = [
            {
                "modification_description": "Increase exploration weight",
                "parameter_name": "exploration_weight",
                "old_value": 1.5,
                "new_value": 2.0,
                "rationale": "Insufficient diversity in recent batches; higher kappa increases exploration."
            },
            {
                "modification_description": "Focus on validated PSMA vectors",
                "parameter_name": "prioritized_targets",
                "old_value": [],
                "new_value": ["PSMA"],
                "rationale": "PSMA targeting vectors showing strongest objective improvements."
            },
            {
                "modification_description": "Switch acquisition function to EI",
                "parameter_name": "acquisition_function",
                "old_value": "UCB",
                "new_value": "EI",
                "rationale": "UCB may be over-exploring; EI focuses on high-probability improvements."
            },
        ]
        return json.dumps(rng.choice(options))

    def _generic_rationale(self, rng: random.Random) -> str:
        return rng.choice([
            "Cycle completed with measurable improvement in binding affinity estimates.",
            "Strategy modification tested; insufficient improvement observed, reverting.",
            "Top candidates demonstrate strong chelator compatibility and target selectivity.",
        ])


def _hash_seed(request: ModelRequest) -> int:
    payload = json.dumps(
        {"system": request.system, "messages": request.messages},
        sort_keys=True,
    )
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)
