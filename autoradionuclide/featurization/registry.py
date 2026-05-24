"""
SMILES registry for known radioligand building blocks.

Convention: Each registry entry stores the STANDALONE building-block moiety,
WITHOUT the conjugated chelator or covalent linker to other parts. The featurizer
combines parts via disconnected SMILES ("." separator) to get approximate
physicochemical descriptors for the resolved organic portion. This avoids
double-counting the chelator contribution when both chelator and targeting vector
are present.

Rules enforced in this file:
1. Only include structures with publicly verifiable SMILES. Cite source per entry.
2. Include the expected molecular formula; cross-check with RDKit at entry creation.
3. Never fabricate a SMILES. If a structure is uncertain, omit it and let the
   quality flag record the unresolved status.
4. New entries must include smiles, formula, and source keys.

Structure resolution priority (see ``resolve_organic_smiles``):
  1. Full construct SMILES from ``construct.smiles`` (if provided)
  2. Part-level SMILES from ``construct.chelator.smiles`` /
     ``construct.targeting_vector.smiles`` (if set on the object)
  3. Registry lookup for each part by name
"""
from __future__ import annotations

import warnings

from autoradionuclide.domain.models import CandidateConstruct
from autoradionuclide.featurization._types import FeatureQuality


# ---------------------------------------------------------------------------
# Chelator registry
# Each entry carries smiles, formula (verified with RDKit), and source citation.
# All structures verified against PubChem or cited literature.
# ---------------------------------------------------------------------------

_CHELATOR_REGISTRY: dict[str, dict[str, str]] = {
    # DOTA: 1,4,7,10-tetraazacyclododecane-1,4,7,10-tetraacetic acid
    #   12-membered ring (4 N + 8 CH₂), four -CH₂COOH arms.
    "DOTA": {
        "smiles": "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC1",
        "formula": "C16H28N4O8",
        "source": "PubChem CID 129730",
    },
    # NOTA: 1,4,7-triazacyclononane-1,4,7-triacetic acid
    #   9-membered ring (3 N + 6 CH₂), three -CH₂COOH arms.
    "NOTA": {
        "smiles": "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CC1",
        "formula": "C12H21N3O6",
        "source": "PubChem CID 5460477",
    },
    # DOTAGA: 2-(4,7,10-tris(carboxymethyl)-1,4,7,10-tetraazacyclododec-1-yl)pentanedioic acid
    #   Same 12-membered ring as DOTA; N1 bears the glutaric-acid arm
    #   (HOOC-CH(N-ring)-CH₂-CH₂-COOH) instead of an acetic arm.
    "DOTAGA": {
        "smiles": "OC(=O)C(N1CCN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC1)CCC(=O)O",
        "formula": "C19H32N4O10",
        "source": "Simecek et al. EJNMMI Res 2012, doi:10.1186/2191-219X-2-17",
    },
    # PSMA chelator (bifunctional PSMA-617) omitted: the SMILES of the full
    # bifunctional molecule includes the targeting urea pharmacophore as well as
    # the chelation functionality. Including only the chelation portion would
    # misrepresent the structure; the full molecule needs independent verification.
    # Add with source citation and verified formula when available.
}


# ---------------------------------------------------------------------------
# Targeting-vector registry
# Each entry carries smiles, formula (verified with RDKit), and source citation.
#
# Convention: store the STANDALONE moiety WITHOUT conjugated chelator.
#
# Most clinical targeting vectors (DOTATATE, DOTATOC, PSMA-617, FAPI-46,
# FAPI-74) are large peptides or bifunctional conjugates whose standalone
# SMILES require independent expert verification. Until verified, constructs
# using those vectors are PARTIAL (if the chelator resolves) or FALLBACK.
# ---------------------------------------------------------------------------

_TARGETING_VECTOR_REGISTRY: dict[str, dict[str, str]] = {
    # MIBG: meta-iodobenzylguanidine (iobenguane)
    #   Small-molecule norepinephrine transporter (NET) ligand; no chelator —
    #   the iodine is attached directly to the aromatic ring.
    #   Clinical use: I-131 MIBG therapy (Azedra), I-123 MIBG imaging.
    "MIBG": {
        "smiles": "NC(=N)NCc1cccc(I)c1",
        "formula": "C8H10IN3",
        "source": "PubChem CID 60860 (iobenguane)",
    },
}


# ---------------------------------------------------------------------------
# Backward-compatible flat views
# These expose the same interface as the original CHELATOR_SMILES /
# TARGETING_VECTOR_SMILES dicts that external consumers and tests reference.
# ---------------------------------------------------------------------------

CHELATOR_SMILES: dict[str, str] = {
    name: entry["smiles"] for name, entry in _CHELATOR_REGISTRY.items()
}

TARGETING_VECTOR_SMILES: dict[str, str] = {
    name: entry["smiles"] for name, entry in _TARGETING_VECTOR_REGISTRY.items()
}


# ---------------------------------------------------------------------------
# Warning deduplication
# Each unresolved building block emits exactly one UserWarning per Python
# session, keyed on "kind:name". Tests must call reset_registry_warning_state()
# for isolation so that a name seen in a prior test does not suppress a warning
# in a later test that asserts on warning counts.
# ---------------------------------------------------------------------------

_warned_registry_misses: set[str] = set()


def reset_registry_warning_state() -> None:
    """Clear the per-building-block warning deduplication state.

    Call this at the start of any test that asserts on UserWarning counts from
    registry resolution. Without resetting, a name already seen in a prior test
    will not re-fire its warning.
    """
    _warned_registry_misses.clear()


def _warn_missing_building_block(kind: str, name: str) -> None:
    """Emit a UserWarning the first time a building-block name is unresolved."""
    key = f"{kind}:{name}"
    if key in _warned_registry_misses:
        return
    _warned_registry_misses.add(key)
    warnings.warn(
        f"Building block '{name}' (kind={kind}) not found in registry. "
        "Feature record quality will be FALLBACK or PARTIAL for constructs "
        f"that use this {kind}. Add a verified SMILES entry with a source "
        "citation to registry.py to enable structure-based featurization.",
        UserWarning,
        stacklevel=4,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_organic_smiles(
    construct: CandidateConstruct,
) -> tuple[str | None, FeatureQuality, list[str], dict[str, str]]:
    """Resolve the organic-portion SMILES for a construct without fabrication.

    Returns a 4-tuple:
      smiles         — resolved SMILES string, or None on FALLBACK
      quality        — FeatureQuality flag
      unresolved     — list of part names that could not be resolved
      reasons        — dict mapping part name → resolution outcome string
    """
    unresolved: list[str] = []
    reasons: dict[str, str] = {}

    # Priority 1: full construct SMILES
    if construct.smiles:
        reasons["construct"] = "full_smiles_provided"
        return construct.smiles, FeatureQuality.FULL, [], reasons

    # Priority 2 + 3: resolve by parts
    chelator_smiles = _resolve_chelator(construct, reasons, unresolved)
    vector_smiles = _resolve_vector(construct, reasons, unresolved)

    resolved_parts = [s for s in (chelator_smiles, vector_smiles) if s]

    if not resolved_parts:
        return None, FeatureQuality.FALLBACK, unresolved, reasons

    # Combine resolved parts as disconnected components. This gives approximate
    # physicochemical descriptors for the available organic portions; it does NOT
    # model the covalent bonds between parts, and it does NOT model the
    # metal-organic bond. The quality flag records this incompleteness.
    combined = ".".join(resolved_parts)
    quality = FeatureQuality.PARTIAL if unresolved else FeatureQuality.FULL
    return combined, quality, unresolved, reasons


def _resolve_chelator(
    construct: CandidateConstruct,
    reasons: dict[str, str],
    unresolved: list[str],
) -> str | None:
    # Direct SMILES on the Chelator object (supplied by caller)
    if construct.chelator.smiles:
        reasons["chelator"] = "smiles_field"
        return construct.chelator.smiles

    name = construct.chelator.name

    # Special case: no chelator (e.g. direct iodination of MIBG)
    if not name or name.lower() == "none":
        reasons["chelator"] = "no_chelator_direct_labelling"
        return None

    # Registry lookup
    if name in CHELATOR_SMILES:
        reasons["chelator"] = f"registry:{name}"
        return CHELATOR_SMILES[name]

    _warn_missing_building_block("chelator", name)
    reasons["chelator"] = f"unresolved:not_in_registry:{name}"
    unresolved.append("chelator")
    return None


def _resolve_vector(
    construct: CandidateConstruct,
    reasons: dict[str, str],
    unresolved: list[str],
) -> str | None:
    # Direct SMILES on the TargetingVector object
    if construct.targeting_vector.smiles:
        reasons["targeting_vector"] = "smiles_field"
        return construct.targeting_vector.smiles

    name = construct.targeting_vector.name
    if name in TARGETING_VECTOR_SMILES:
        reasons["targeting_vector"] = f"registry:{name}"
        return TARGETING_VECTOR_SMILES[name]

    _warn_missing_building_block("targeting_vector", name)
    reasons["targeting_vector"] = f"unresolved:not_in_registry:{name}"
    unresolved.append("targeting_vector")
    return None
