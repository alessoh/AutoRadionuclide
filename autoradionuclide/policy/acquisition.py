"""Active-learning policy — acquisition functions and batch selection with diversity."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from typing import Literal
from autoradionuclide.domain.models import CandidateConstruct, ObjectiveSpec, ObjectiveDirection, ScoredObjective
from autoradionuclide.surrogates.gp_surrogate import SurrogateBank, featurize
from autoradionuclide.scoring.aggregator import aggregate_score


AcquisitionFunction = Literal["UCB", "EI", "thompson"]


def ucb_score(mean: float, std: float, kappa: float = 1.5) -> float:
    return mean + kappa * std


def ei_score(mean: float, std: float, best: float, xi: float = 0.01) -> float:
    if std <= 0:
        return 0.0
    Z = (mean - best - xi) / std
    return (mean - best - xi) * norm.cdf(Z) + std * norm.pdf(Z)


class ActiveLearningPolicy:
    """Selects a diverse batch of candidates using acquisition functions.

    The batch selection is greedy with diversity enforcement: after scoring
    all candidates, we iteratively pick the highest-scoring one that is
    sufficiently different (in feature space) from already-selected ones.
    """

    def __init__(
        self,
        surrogate_bank: SurrogateBank,
        specs: list[ObjectiveSpec],
        acquisition_fn: AcquisitionFunction = "UCB",
        exploration_weight: float = 1.5,
        diversity_threshold: float = 0.3,
    ) -> None:
        self._bank = surrogate_bank
        self._specs = specs
        self._acq_fn = acquisition_fn
        self._kappa = exploration_weight
        self._div_threshold = diversity_threshold
        self._best_known: float = 0.0

    def update_best(self, score: float) -> None:
        self._best_known = max(self._best_known, score)

    def rank(
        self,
        candidates: list[CandidateConstruct],
        batch_size: int,
    ) -> list[tuple[CandidateConstruct, float]]:
        """Return up to batch_size (construct, acquisition_score) pairs, diverse."""
        if not candidates:
            return []

        scored = []
        for c in candidates:
            preds = self._bank.predict_all(c)
            objectives = []
            for spec in self._specs:
                if spec.name in preds:
                    objectives.append(ScoredObjective(
                        name=spec.name,
                        value=preds[spec.name],
                        direction=spec.direction,
                    ))

            agg_mean, _ = aggregate_score(objectives, self._specs)

            # Compute acquisition score using prediction uncertainty
            stds = [preds[s.name].uncertainty for s in self._specs if s.name in preds]
            mean_std = float(np.mean(stds)) if stds else 0.0

            if self._acq_fn == "UCB":
                acq = ucb_score(agg_mean, mean_std, self._kappa)
            elif self._acq_fn == "EI":
                acq = ei_score(agg_mean, mean_std, self._best_known)
            else:  # thompson
                acq = agg_mean + float(np.random.normal(0, mean_std))

            scored.append((c, acq))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Greedy diversity selection
        selected = []
        selected_feats: list[np.ndarray] = []
        for c, acq in scored:
            if len(selected) >= batch_size:
                break
            feat = featurize(c)
            if _is_diverse(feat, selected_feats, self._div_threshold):
                selected.append((c, acq))
                selected_feats.append(feat)

        # If diversity is too strict, fall back to top-k
        if not selected:
            selected = scored[:batch_size]

        return selected


def _is_diverse(feat: np.ndarray, existing: list[np.ndarray], threshold: float) -> bool:
    if not existing:
        return True
    dists = [float(np.linalg.norm(feat - e)) for e in existing]
    return min(dists) >= threshold
