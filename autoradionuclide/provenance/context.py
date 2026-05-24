"""Provenance context — pin every factor that influences a decision."""
from __future__ import annotations
import hashlib, json, uuid
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


SCORING_VERSION = "0.1.0"
SURROGATE_VERSION = "0.1.0"
PROMPT_TEMPLATE_VERSION = "0.1.0"


class ProvenanceContext(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION
    scoring_version: str = SCORING_VERSION
    surrogate_version: str = SURROGATE_VERSION
    config_hash: str = ""
    random_seed: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        model_id: str,
        config_dict: dict[str, Any],
        seed: Optional[int] = None,
        **extra: Any,
    ) -> "ProvenanceContext":
        config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True).encode()
        ).hexdigest()[:16]
        return cls(
            model_id=model_id,
            config_hash=config_hash,
            random_seed=seed,
            extra=extra,
        )

    def to_ledger_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
