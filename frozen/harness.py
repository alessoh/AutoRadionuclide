"""
FROZEN — DO NOT MODIFY.
This file defines the benchmark scoring definitions.
The planner must not alter this file.

All scoring functions are PLACEHOLDER heuristics — not validated predictive models.
They encode domain knowledge into structured numerical scores to enable
the closed-loop reasoning system to function in simulation.
"""
from __future__ import annotations
from autoradionuclide.domain.models import (
    CandidateConstruct, ObjectiveValue, ProvenanceTag, Radionuclide, HALF_LIFE_DAYS
)


# ---------------------------------------------------------------------------
# SCORING_DEFINITIONS — frozen reference descriptions for each objective
# ---------------------------------------------------------------------------
SCORING_DEFINITIONS: dict[str, str] = {
    "binding_affinity": (
        "PLACEHOLDER heuristic. Estimated binding affinity to primary biological target. "
        "Based on known clinical data for target-vector pairs. Range 0–1. "
        "NOT a validated predictive model for novel compounds."
    ),
    "chelator_stability": (
        "PLACEHOLDER heuristic. Thermodynamic compatibility of chelator with radionuclide. "
        "Based on known coordination chemistry. Range 0–1. "
        "NOT a validated predictive model for novel combinations."
    ),
    "half_life_compatibility": (
        "Physics-based score for isotope half-life suitability for therapy context. "
        "Uses factual HALF_LIFE_DAYS values. Range 0–1. "
        "Therapy sweet spot: 4–14 days."
    ),
    "synthetic_feasibility": (
        "PLACEHOLDER heuristic. Estimated synthetic accessibility by vector type. "
        "NOT a validated predictive model. Range 0–1."
    ),
    "selectivity": (
        "PLACEHOLDER heuristic. Estimated off-target binding selectivity. "
        "Based on known clinical profiles of target-vector class. Range 0–1. "
        "NOT a validated predictive model for novel compounds."
    ),
}

# ---------------------------------------------------------------------------
# CHELATOR_ISOTOPE_COMPATIBILITY — known coordination chemistry compatibility
# ---------------------------------------------------------------------------
CHELATOR_ISOTOPE_COMPATIBILITY: dict[tuple[str, str], float] = {
    ("DOTA",   "Lu-177"): 0.95,
    ("DOTA",   "Ac-225"): 0.90,
    ("DOTA",   "Y-90"):   0.90,
    ("DOTA",   "Ga-68"):  0.65,
    ("DOTA",   "I-131"):  0.30,   # iodine doesn't chelate well via DOTA
    ("DOTA",   "Bi-213"): 0.85,
    ("DOTA",   "At-211"): 0.25,
    ("NOTA",   "Ga-68"):  0.95,
    ("NOTA",   "Lu-177"): 0.50,   # unstable — known incompatibility
    ("NOTA",   "Ac-225"): 0.20,
    ("NOTA",   "Y-90"):   0.35,
    ("NOTA",   "I-131"):  0.20,
    ("NOTA",   "Bi-213"): 0.30,
    ("NOTA",   "At-211"): 0.20,
    ("DOTAGA", "Lu-177"): 0.92,
    ("DOTAGA", "Ac-225"): 0.88,
    ("DOTAGA", "Y-90"):   0.88,
    ("DOTAGA", "Ga-68"):  0.60,
    ("DOTAGA", "I-131"):  0.25,
    ("PSMA",   "Lu-177"): 0.80,   # PSMA chelator (bifunctional)
    ("PSMA",   "Ac-225"): 0.70,
    ("PSMA",   "Ga-68"):  0.55,
    # Direct iodination (no chelator)
    ("none",   "I-131"):  0.90,
    ("none",   "At-211"): 0.75,
    ("none",   "Lu-177"): 0.10,
    ("none",   "Ga-68"):  0.10,
}

# ---------------------------------------------------------------------------
# TARGET_VALIDATION_SCORES — heuristic binding affinity by target
# ---------------------------------------------------------------------------
TARGET_VALIDATION_SCORES: dict[str, float] = {
    "PSMA":         0.90,
    "SSTR2":        0.85,
    "FAP":          0.72,
    "integrin_avb3": 0.65,
    "NET":          0.75,
    "VEGFR":        0.45,
    "unknown":      0.25,
}

# ---------------------------------------------------------------------------
# VECTOR_TYPE_FEASIBILITY — synthetic feasibility by vector class
# ---------------------------------------------------------------------------
VECTOR_TYPE_FEASIBILITY: dict[str, float] = {
    "small_molecule":    0.80,
    "peptide":           0.85,
    "antibody_fragment": 0.55,
}

# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def score_binding_affinity(construct: CandidateConstruct) -> ObjectiveValue:
    """PLACEHOLDER — not a validated predictive model.

    Estimates binding affinity based on the known clinical validation status
    of the biological target. Higher scores indicate better-validated targets
    with documented radioligand binding.
    """
    target = construct.targeting_vector.target
    base = TARGET_VALIDATION_SCORES.get(target, 0.25)

    # Small bonus for peptide vectors (generally well-characterized)
    vtype = construct.targeting_vector.vector_type
    if vtype == "peptide":
        base = min(1.0, base + 0.03)
    elif vtype == "antibody_fragment":
        base = max(0.0, base - 0.05)

    return ObjectiveValue(
        estimate=round(base, 4),
        uncertainty=0.10,
        source=ProvenanceTag.HEURISTIC,
        notes="PLACEHOLDER heuristic from target validation database. Not a validated predictive model.",
    )


def score_chelator_stability(construct: CandidateConstruct) -> ObjectiveValue:
    """PLACEHOLDER — not a validated predictive model.

    Scores chelator-isotope thermodynamic compatibility using known coordination
    chemistry. NOTA+Lu-177 and similar known-incompatible pairs score low.
    Direct iodination (no chelator) for iodine isotopes scores high.
    """
    chelator = construct.chelator.name
    isotope_val = construct.radionuclide.value

    # Handle direct iodination case (no chelator / minimal chelation)
    lookup_chelator = chelator if chelator else "none"
    score = CHELATOR_ISOTOPE_COMPATIBILITY.get((lookup_chelator, isotope_val))

    if score is None:
        # Fallback: check if isotope is in declared compatible list
        if isotope_val in construct.chelator.compatible_isotopes:
            score = 0.60
        else:
            score = 0.30

    return ObjectiveValue(
        estimate=round(score, 4),
        uncertainty=0.08,
        source=ProvenanceTag.HEURISTIC,
        notes="PLACEHOLDER heuristic from coordination chemistry compatibility table. Not a validated predictive model.",
    )


def score_half_life_compatibility(construct: CandidateConstruct) -> ObjectiveValue:
    """Physics-based half-life suitability score for therapy context.

    Uses factual HALF_LIFE_DAYS values from the domain model.
    Formula: min(1.0, half_life_days / 7.0) for therapy context,
    with a bonus for the 'sweet spot' of 4–14 days.
    Ga-68 (0.047d) scores near 0 for therapy — it is an imaging agent.
    """
    hl = HALF_LIFE_DAYS.get(construct.radionuclide, 0.0)

    # Base score: normalize to 7-day reference for therapy
    base = min(1.0, hl / 7.0)

    # Sweet-spot bonus: 4–14 days is optimal for systemic radioligand therapy
    if 4.0 <= hl <= 14.0:
        bonus = 0.05
    elif hl < 1.0:
        # Very short half-lives are poorly suited for therapy
        # Ga-68 at 0.047d → effectively 0
        base = min(base, 0.05)
        bonus = 0.0
    else:
        bonus = 0.0

    score = min(1.0, base + bonus)

    return ObjectiveValue(
        estimate=round(score, 4),
        uncertainty=0.0,   # physics value is exact (no uncertainty in formula)
        source=ProvenanceTag.PHYSICS,
        notes=(
            f"Physics-based: half_life={hl:.3f}d. "
            f"Therapy sweet spot 4–14d. Score=min(1,hl/7)+bonus."
        ),
    )


def score_synthetic_feasibility(construct: CandidateConstruct) -> ObjectiveValue:
    """PLACEHOLDER — not a validated predictive model.

    Estimates synthetic accessibility based on vector type.
    Peptides and small molecules are generally more accessible than
    antibody fragments for radioligand conjugation.
    """
    vtype = construct.targeting_vector.vector_type
    base = VECTOR_TYPE_FEASIBILITY.get(vtype, 0.60)

    # Linker adds minor complexity
    if construct.linker is not None:
        # PEG linkers are well established; reduce score only marginally
        if "PEG" in (construct.linker or ""):
            base = min(1.0, base + 0.02)
        else:
            base = max(0.0, base - 0.03)

    return ObjectiveValue(
        estimate=round(base, 4),
        uncertainty=0.12,
        source=ProvenanceTag.HEURISTIC,
        notes="PLACEHOLDER heuristic based on vector type class. Not a validated predictive model.",
    )


def score_selectivity(construct: CandidateConstruct) -> ObjectiveValue:
    """PLACEHOLDER — not a validated predictive model.

    Estimates off-target selectivity based on known clinical profiles.
    Well-validated clinical targets generally have better-characterized
    selectivity profiles from clinical experience.
    """
    target = construct.targeting_vector.target
    # Use target validation as a proxy for selectivity knowledge
    base_affinity = TARGET_VALIDATION_SCORES.get(target, 0.25)

    # Selectivity is loosely correlated with target characterization
    # but degraded for very high-affinity targets that may have off-target effects
    if base_affinity >= 0.85:
        selectivity = 0.75  # well-validated but potentially some off-target
    elif base_affinity >= 0.70:
        selectivity = 0.80  # good selectivity profile
    elif base_affinity >= 0.50:
        selectivity = 0.65
    else:
        selectivity = 0.45  # poorly characterized targets have unknown selectivity

    # Peptides tend to have better selectivity (smaller, more specific)
    vtype = construct.targeting_vector.vector_type
    if vtype == "peptide":
        selectivity = min(1.0, selectivity + 0.05)
    elif vtype == "antibody_fragment":
        selectivity = min(1.0, selectivity + 0.10)

    return ObjectiveValue(
        estimate=round(selectivity, 4),
        uncertainty=0.15,
        source=ProvenanceTag.HEURISTIC,
        notes="PLACEHOLDER heuristic from target class clinical profiles. Not a validated predictive model.",
    )


def score_all(construct: CandidateConstruct) -> dict[str, ObjectiveValue]:
    """Score all objectives and return as a dict. Called by the frozen harness."""
    return {
        "binding_affinity":       score_binding_affinity(construct),
        "chelator_stability":     score_chelator_stability(construct),
        "half_life_compatibility": score_half_life_compatibility(construct),
        "synthetic_feasibility":  score_synthetic_feasibility(construct),
        "selectivity":            score_selectivity(construct),
    }
