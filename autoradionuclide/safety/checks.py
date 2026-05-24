"""Safety checks — isotope feasibility, hard constraints, hazard flags."""
from __future__ import annotations
from dataclasses import dataclass, field
from autoradionuclide.domain.models import CandidateConstruct, Radionuclide, HALF_LIFE_DAYS


# Minimum cycle duration in days (time from synthesis to imaging/therapy start)
_MIN_CYCLE_DAYS = 0.5   # 12 hours minimum
_MAX_CYCLE_DAYS = 30.0  # 30 days maximum

# Known incompatible combinations (chelator, isotope)
_INCOMPATIBLE = {
    ("NOTA", "Lu-177"),   # NOTA forms unstable Lu complex
    ("NOTA", "Ac-225"),
    ("NOTA", "Y-90"),
}

# Isotopes requiring alpha-emitter special handling
_ALPHA_EMITTERS = {Radionuclide.AC225, Radionuclide.AT211, Radionuclide.BI213}


@dataclass
class SafetyCheckResult:
    passed: bool
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_construct(construct: CandidateConstruct, dry_run: bool = False) -> SafetyCheckResult:
    """Run all safety checks on a single construct."""
    violations: list[str] = []
    warnings: list[str] = []

    hl = HALF_LIFE_DAYS.get(construct.radionuclide, 0.0)
    if hl < _MIN_CYCLE_DAYS:
        violations.append(
            f"Half-life {hl:.3f}d for {construct.radionuclide.value} "
            f"is below minimum cycle duration {_MIN_CYCLE_DAYS}d"
        )
    if hl > _MAX_CYCLE_DAYS:
        warnings.append(
            f"Half-life {hl:.1f}d for {construct.radionuclide.value} "
            f"exceeds typical therapy window {_MAX_CYCLE_DAYS}d"
        )

    pair = (construct.chelator.name, construct.radionuclide.value)
    if pair in _INCOMPATIBLE:
        violations.append(
            f"Chelator {construct.chelator.name} is incompatible with "
            f"{construct.radionuclide.value} (known unstable complex)"
        )

    if construct.radionuclide in _ALPHA_EMITTERS:
        warnings.append(
            f"{construct.radionuclide.value} is an alpha emitter — "
            f"requires enhanced radiation safety protocols"
        )

    if construct.targeting_vector.target == "unknown":
        warnings.append("Targeting vector has unknown biological target")

    if dry_run:
        warnings.append("[DRY-RUN] This request was not submitted to the lab")

    return SafetyCheckResult(
        passed=len(violations) == 0,
        violations=violations,
        warnings=warnings,
    )


def check_batch(
    constructs: list[CandidateConstruct],
    dry_run: bool = False,
) -> dict[str, SafetyCheckResult]:
    return {c.id: check_construct(c, dry_run) for c in constructs}
