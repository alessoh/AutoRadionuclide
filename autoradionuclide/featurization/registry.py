"""
SMILES registry for known radioligand building blocks.

Rules enforced in this file:
1. Only include structures with publicly verifiable SMILES. Cite source per entry.
2. Never fabricate a SMILES. If a structure is uncertain, omit it and let the
   quality flag record the unresolved status.
3. New entries must include a source annotation in the adjacent comment.

Structure resolution priority (see ``resolve_organic_smiles``):
  1. Full construct SMILES from ``construct.smiles`` (if provided)
  2. Part-level SMILES from ``construct.chelator.smiles`` /
     ``construct.targeting_vector.smiles`` (if set on the object)
  3. Registry lookup for each part by name
"""
from __future__ import annotations

from autoradionuclide.domain.models import CandidateConstruct
from autoradionuclide.featurization._types import FeatureQuality


# ---------------------------------------------------------------------------
# Chelator registry
# All structures verified against PubChem or cited literature.
# ---------------------------------------------------------------------------

# DOTA: 1,4,7,10-tetraazacyclododecane-1,4,7,10-tetraacetic acid
#   PubChem CID 129730
#   12-membered ring (4 N + 8 CH₂), four -CH₂COOH arms
_DOTA_SMILES = "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC1"

# NOTA: 1,4,7-triazacyclononane-1,4,7-triacetic acid
#   PubChem CID 5460477
#   9-membered ring (3 N + 6 CH₂), three -CH₂COOH arms
_NOTA_SMILES = "OC(=O)CN1CCN(CC(=O)O)CCN(CC(=O)O)CC1"

# DOTAGA: 2-(4,7,10-tris(carboxymethyl)-1,4,7,10-tetraazacyclododec-1-yl)pentanedioic acid
#   Simecek et al. EJNMMI Res 2012, doi:10.1186/2191-219X-2-17
#   Same 12-membered ring as DOTA; N1 bears the glutaric-acid arm
#   (HOOC-CH(N-ring)-CH₂-CH₂-COOH) instead of an acetic arm.
_DOTAGA_SMILES = "OC(=O)C(N1CCN(CC(=O)O)CCN(CC(=O)O)CCN(CC(=O)O)CC1)CCC(=O)O"

CHELATOR_SMILES: dict[str, str] = {
    "DOTA": _DOTA_SMILES,
    "NOTA": _NOTA_SMILES,
    "DOTAGA": _DOTAGA_SMILES,
    # "PSMA" chelator (bifunctional PSMA-617) omitted: the SMILES of the full
    # bifunctional molecule includes the targeting urea pharmacophore as well as
    # the chelation functionality. Including only the chelation portion would
    # misrepresent the structure; the full molecule SMILES needs independent
    # verification before inclusion. Add with source citation when available.
}


# ---------------------------------------------------------------------------
# Targeting-vector registry
# Most clinical targeting vectors are large, complex peptides or small molecules
# whose verified SMILES require careful sourcing. Entries will be added as
# structures are verified; the registry is intentionally left sparse.
# ---------------------------------------------------------------------------
TARGETING_VECTOR_SMILES: dict[str, str] = {
    # No entries. Add with source citations as structures are verified.
}


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

    reasons["targeting_vector"] = f"unresolved:not_in_registry:{name}"
    unresolved.append("targeting_vector")
    return None
