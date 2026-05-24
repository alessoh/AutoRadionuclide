"""AGENT-EDITABLE: objective weights the planner may modify."""
from __future__ import annotations

OBJECTIVE_WEIGHTS = {
    "binding_affinity":        2.0,
    "chelator_stability":      1.5,
    "half_life_compatibility": 1.0,
    "synthetic_feasibility":   0.8,
    "selectivity":             1.2,
}
