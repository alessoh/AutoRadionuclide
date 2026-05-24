"""Campaign specification — the analog of program.md / the AutoResearch instruction file."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field, model_validator
from autoradionuclide.domain.models import Radionuclide, ObjectiveSpec, ObjectiveDirection


class GatingPolicy(str):
    AUTOMATIC = "automatic"   # no human approval needed (testing)
    ADVISORY  = "advisory"    # approval logged but not blocking
    MANDATORY = "mandatory"   # loop blocks until approved


class BudgetSpec(BaseModel):
    max_cycles: int = 10
    max_constructs_total: int = 100
    max_cost_usd: float = 10_000.0
    max_wall_minutes: float = 60.0
    max_model_calls: int = 500


class StoppingCriteria(BaseModel):
    target_campaign_score: float = 0.85
    min_score_delta: float = 0.005   # stall threshold per cycle
    stall_patience: int = 3          # consecutive stalls before stopping


class CampaignSpec(BaseModel):
    """Full declarative campaign specification. Validated before any cycle runs."""
    campaign_id: str
    name: str
    description: str = ""
    target: str                       # biological target, e.g. "PSMA"
    indication: str = ""              # e.g. "mCRPC"
    isotope: Radionuclide = Radionuclide.LU177
    objectives: list[ObjectiveSpec] = Field(default_factory=list)
    budget: BudgetSpec = Field(default_factory=BudgetSpec)
    stopping: StoppingCriteria = Field(default_factory=StoppingCriteria)
    gating_policy: str = GatingPolicy.AUTOMATIC
    model_provider: str = "mock"      # "mock" | "anthropic"
    model_id: str = "mock-deterministic-v1"
    batch_size: int = 4
    db_path: str = "campaign.db"
    random_seed: int = 42
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def add_default_objectives(self) -> "CampaignSpec":
        if not self.objectives:
            self.objectives = _default_objectives()
        return self

    def config_hash(self) -> str:
        d = self.model_dump(exclude={"extra"})
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: Path | str) -> "CampaignSpec":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)

    def to_yaml(self, path: Path | str) -> None:
        Path(path).write_text(yaml.dump(self.model_dump(mode="json"), sort_keys=False))


def _default_objectives() -> list[ObjectiveSpec]:
    return [
        ObjectiveSpec(
            name="binding_affinity",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=2.0,
            target=0.8,
            description="Estimated binding affinity to primary target (0–1 heuristic)",
        ),
        ObjectiveSpec(
            name="chelator_stability",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=1.5,
            constraint=0.5,
            description="Chelator-isotope thermodynamic compatibility (0–1 heuristic)",
        ),
        ObjectiveSpec(
            name="half_life_compatibility",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=1.0,
            description="Isotope half-life suitability for therapy (0–1, physics-based)",
        ),
        ObjectiveSpec(
            name="synthetic_feasibility",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=0.8,
            constraint=0.4,
            description="Estimated synthetic accessibility (0–1 heuristic)",
        ),
        ObjectiveSpec(
            name="selectivity",
            direction=ObjectiveDirection.MAXIMIZE,
            weight=1.2,
            description="Estimated off-target selectivity (0–1 heuristic)",
        ),
    ]
