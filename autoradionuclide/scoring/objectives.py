"""Per-objective scoring — thin wrappers that call the frozen harness functions."""
from __future__ import annotations
from autoradionuclide.domain.models import CandidateConstruct, ScoredObjective, ObjectiveDirection
import frozen.harness as harness


def score_construct(construct: CandidateConstruct) -> list[ScoredObjective]:
    """Return all objectives for one construct by delegating to the frozen harness."""
    scores = harness.score_all(construct)
    return [
        ScoredObjective(
            name=name,
            value=val,
            direction=ObjectiveDirection.MAXIMIZE,
        )
        for name, val in scores.items()
    ]
