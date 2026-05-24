"""AGENT-EDITABLE: candidate generation strategy."""
from __future__ import annotations

GENERATION_STRATEGY = {
    "mode":               "llm_propose",  # llm_propose | enumerate | hybrid
    "n_llm_proposals":    8,
    "n_enumerated":       4,
    "dedup_threshold":    0.95,
    "allow_novel_vectors": True,
}
