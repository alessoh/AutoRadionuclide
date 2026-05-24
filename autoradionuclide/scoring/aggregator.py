"""Multi-objective aggregation — weighted scalarization with hard-constraint filtering."""
from __future__ import annotations
from typing import Optional
from autoradionuclide.domain.models import (
    CandidateConstruct, ScoredObjective, ObjectiveSpec
)
from autoradionuclide.scoring.objectives import score_construct


def aggregate_score(
    objectives: list[ScoredObjective],
    specs: list[ObjectiveSpec],
) -> tuple[float, bool]:
    """Return (weighted_score, constraints_satisfied).

    Weights are normalized so the maximum possible score is 1.0.
    Hard constraints (spec.constraint not None) gate the score:
    if any constraint is violated, constraints_satisfied is False.
    """
    spec_map = {s.name: s for s in specs}
    obj_names = {o.name for o in objectives}
    total_weight = sum(s.weight for s in specs if s.name in obj_names)
    if total_weight == 0:
        return 0.0, True

    weighted_sum = 0.0
    satisfied = True
    for obj in objectives:
        spec = spec_map.get(obj.name)
        if spec is None:
            continue
        val = obj.value.estimate
        if spec.direction.value == "minimize":
            val = 1.0 - val
        if spec.constraint is not None and obj.value.estimate < spec.constraint:
            satisfied = False
        weighted_sum += spec.weight * val

    return weighted_sum / total_weight, satisfied


def score_and_aggregate(
    construct: CandidateConstruct,
    specs: list[ObjectiveSpec],
) -> tuple[list[ScoredObjective], float, bool]:
    """Score a construct and return (objectives, aggregate_score, feasible)."""
    objectives = score_construct(construct)
    score, feasible = aggregate_score(objectives, specs)
    return objectives, score, feasible
