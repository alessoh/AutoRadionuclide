"""Core domain model — all typed schemas that cross module boundaries."""
from __future__ import annotations
import hashlib, uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, model_validator


class Radionuclide(str, Enum):
    LU177 = "Lu-177"
    AC225 = "Ac-225"
    GA68 = "Ga-68"
    Y90 = "Y-90"
    I131 = "I-131"
    BI213 = "Bi-213"
    AT211 = "At-211"


HALF_LIFE_DAYS: dict[Radionuclide, float] = {
    Radionuclide.LU177: 6.65,
    Radionuclide.AC225: 9.92,
    Radionuclide.GA68: 68 / (60 * 24),    # 68 minutes → days
    Radionuclide.Y90: 64 / 24,             # 64 hours → days
    Radionuclide.I131: 8.02,
    Radionuclide.BI213: 45.6 / (60 * 24), # 45.6 min → days
    Radionuclide.AT211: 7.21 / 24,         # 7.21 hours → days
}


class CandidateStatus(str, Enum):
    PROPOSED = "proposed"
    SCORED = "scored"
    SELECTED = "selected"
    REQUESTED = "requested"
    MADE = "made"
    TESTED = "tested"
    CONCLUDED = "concluded"


class ObjectiveDirection(str, Enum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ProvenanceTag(str, Enum):
    PLACEHOLDER = "placeholder"
    HEURISTIC = "heuristic"
    LEARNED = "learned"
    PHYSICS = "physics"
    MEASURED = "measured"
    MODEL_REASONED = "model_reasoned"


class TargetingVector(BaseModel):
    name: str
    target: str                       # e.g. "PSMA", "SSTR2", "FAP"
    vector_type: str                  # "small_molecule", "peptide", "antibody_fragment"
    smiles: Optional[str] = None
    notes: str = ""


class Chelator(BaseModel):
    name: str                         # e.g. "DOTA", "NOTA", "PSMA", "DOTAGA"
    smiles: Optional[str] = None
    compatible_isotopes: list[str] = Field(default_factory=list)


class CandidateConstruct(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    targeting_vector: TargetingVector
    chelator: Chelator
    linker: Optional[str] = None
    radionuclide: Radionuclide
    smiles: Optional[str] = None       # full construct SMILES (optional)
    canonical_smiles: Optional[str] = None
    status: CandidateStatus = CandidateStatus.PROPOSED
    provenance_id: str = ""
    generation_reasoning: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    campaign_id: str = ""

    @property
    def composite_key(self) -> str:
        parts = [
            self.targeting_vector.name,
            self.chelator.name,
            self.linker or "nolinker",
            self.radionuclide.value,
        ]
        return "_".join(p.replace(" ", "_").replace("-", "") for p in parts)


class ObjectiveValue(BaseModel):
    estimate: float
    uncertainty: float = 0.0
    source: ProvenanceTag = ProvenanceTag.PLACEHOLDER
    notes: str = ""

    @model_validator(mode="after")
    def uncertainty_non_negative(self) -> "ObjectiveValue":
        if self.uncertainty < 0:
            raise ValueError("uncertainty must be >= 0")
        return self


class ScoredObjective(BaseModel):
    name: str
    value: ObjectiveValue
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    target: Optional[float] = None
    constraint: Optional[float] = None   # hard constraint threshold


class ObjectiveSpec(BaseModel):
    """Campaign-level objective definition."""
    name: str
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    weight: float = 1.0
    target: Optional[float] = None
    constraint: Optional[float] = None
    description: str = ""


class ExperimentRequest(BaseModel):
    """Outward message the reasoning layer emits to the make-test-analyze layer."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    campaign_id: str
    cycle_id: str
    constructs: list[CandidateConstruct]
    assays: list[str]
    isotope: Radionuclide
    quantity_gbq: float
    priority: int = 5
    estimated_cost_usd: float = 0.0
    risk_flag: bool = False
    reasoning_id: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AssayResult(BaseModel):
    assay_name: str
    construct_id: str
    value: float
    unit: str
    uncertainty: float = 0.0
    passed: bool = True
    notes: str = ""


class ResultRecord(BaseModel):
    """Message returned from the make-test-analyze layer."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    campaign_id: str
    cycle_id: str
    construct_results: list[AssayResult]
    radiochemical_yield: float = 0.0    # fraction 0–1
    radiochemical_purity: float = 0.0   # fraction 0–1
    failure_reason: Optional[str] = None
    received_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CycleResult(BaseModel):
    """Summary of one inner discovery cycle."""
    cycle_id: str
    campaign_id: str
    cycle_number: int
    constructs_proposed: int
    constructs_scored: int
    constructs_selected: int
    campaign_score_before: float
    campaign_score_after: float
    score_delta: float
    strategy_modification: Optional[str] = None
    strategy_kept: Optional[bool] = None
    rationale: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None


class LedgerEntryType(str, Enum):
    PROPOSAL = "proposal"
    SCORE = "score"
    SELECTION = "selection"
    REQUEST = "request"
    RESULT = "result"
    SURROGATE_REFIT = "surrogate_refit"
    STRATEGY_MODIFICATION = "strategy_modification"
    APPROVAL = "approval"
    CYCLE_SUMMARY = "cycle_summary"
    MODEL_CALL = "model_call"
    ERROR = "error"


class LedgerEntry(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    entry_type: LedgerEntryType
    campaign_id: str
    cycle_id: str = ""
    construct_id: Optional[str] = None
    model_call_id: Optional[str] = None
    provenance_id: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    attribution: str = "autoradionuclide-reasoning-layer"


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str
    system: Optional[str] = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 2048
    response_format: Optional[str] = None  # "json_object" or None
    seed: Optional[int] = None


class ModelResponse(BaseModel):
    request_id: str
    model: str
    content: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_ms: float = 0.0
    cached: bool = False
