"""AGENT-EDITABLE: search and policy hyperparameters."""
from __future__ import annotations

HYPERPARAMS = {
    "acquisition_function":   "UCB",
    "exploration_weight":     1.5,
    "diversity_threshold":    0.3,
    "batch_size":             4,
    "n_candidates_generated": 12,
    "prioritized_targets":    [],
    "prioritized_chelators":  [],
}
