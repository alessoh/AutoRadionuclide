"""Tests for the active learning policy."""
from __future__ import annotations
import pytest
import numpy as np
from autoradionuclide.domain.models import (
    CandidateConstruct, TargetingVector, Chelator, Radionuclide, ObjectiveSpec, ObjectiveDirection
)
from autoradionuclide.surrogates.gp_surrogate import SurrogateBank
from autoradionuclide.policy.acquisition import ActiveLearningPolicy, ucb_score, ei_score
from autoradionuclide.config.schema import _default_objectives


def make_construct(
    target: str = "PSMA",
    chelator: str = "DOTA",
    vector_type: str = "small_molecule",
    isotope: Radionuclide = Radionuclide.LU177,
    linker: str | None = None,
    name: str = "test",
) -> CandidateConstruct:
    return CandidateConstruct(
        name=name,
        targeting_vector=TargetingVector(
            name=f"{target}-vector",
            target=target,
            vector_type=vector_type,
        ),
        chelator=Chelator(name=chelator),
        radionuclide=isotope,
        linker=linker,
    )


def make_diverse_constructs() -> list[CandidateConstruct]:
    """Create a set of constructs with different feature vectors."""
    return [
        make_construct("PSMA", "DOTA", "small_molecule", Radionuclide.LU177, name="c1"),
        make_construct("SSTR2", "NOTA", "peptide", Radionuclide.GA68, name="c2"),
        make_construct("FAP", "DOTAGA", "small_molecule", Radionuclide.AC225, name="c3"),
        make_construct("integrin_avb3", "DOTA", "peptide", Radionuclide.Y90, name="c4"),
        make_construct("NET", "DOTA", "small_molecule", Radionuclide.I131, name="c5"),
        make_construct("PSMA", "DOTAGA", "peptide", Radionuclide.LU177, linker="PEG2", name="c6"),
    ]


class TestAcquisitionFunctions:
    def test_ucb_increases_with_std(self):
        """Higher uncertainty should increase UCB score."""
        s1 = ucb_score(0.5, 0.1, kappa=1.5)
        s2 = ucb_score(0.5, 0.3, kappa=1.5)
        assert s2 > s1

    def test_ucb_increases_with_mean(self):
        s1 = ucb_score(0.5, 0.1)
        s2 = ucb_score(0.7, 0.1)
        assert s2 > s1

    def test_ei_zero_when_no_std(self):
        assert ei_score(0.8, 0.0, 0.5) == 0.0

    def test_ei_zero_when_mean_below_best(self):
        """EI should be near zero when mean is well below best."""
        score = ei_score(0.3, 0.01, 0.9)
        assert score < 1e-6

    def test_ei_positive_when_mean_above_best(self):
        score = ei_score(0.8, 0.1, 0.5)
        assert score > 0


class TestActiveLearningPolicy:
    def _make_policy(
        self,
        acquisition_fn: str = "UCB",
        diversity_threshold: float = 0.1,
    ) -> ActiveLearningPolicy:
        specs = _default_objectives()
        bank = SurrogateBank([s.name for s in specs], seed=42)
        return ActiveLearningPolicy(
            surrogate_bank=bank,
            specs=specs,
            acquisition_fn=acquisition_fn,
            exploration_weight=1.5,
            diversity_threshold=diversity_threshold,
        )

    def test_rank_returns_correct_count(self):
        policy = self._make_policy()
        candidates = make_diverse_constructs()
        ranked = policy.rank(candidates, batch_size=3)
        assert len(ranked) == 3

    def test_rank_returns_fewer_if_not_enough_candidates(self):
        policy = self._make_policy()
        candidates = make_diverse_constructs()[:2]
        ranked = policy.rank(candidates, batch_size=5)
        assert len(ranked) <= 2

    def test_rank_empty_candidates(self):
        policy = self._make_policy()
        ranked = policy.rank([], batch_size=4)
        assert ranked == []

    def test_rank_returns_constructs_and_scores(self):
        policy = self._make_policy()
        candidates = make_diverse_constructs()
        ranked = policy.rank(candidates, batch_size=2)
        for c, score in ranked:
            assert isinstance(c, CandidateConstruct)
            assert isinstance(score, float)

    def test_diversity_enforced(self):
        """No two selected constructs should be identical (same feature vector)."""
        policy = self._make_policy(diversity_threshold=0.1)
        # Create duplicates
        dup1 = make_construct("PSMA", "DOTA", "small_molecule", Radionuclide.LU177, name="dup1")
        dup2 = make_construct("PSMA", "DOTA", "small_molecule", Radionuclide.LU177, name="dup2")
        diverse = make_construct("SSTR2", "NOTA", "peptide", Radionuclide.GA68, name="diverse")
        candidates = [dup1, dup2, diverse]
        ranked = policy.rank(candidates, batch_size=2)
        # Should not select both duplicates (they have identical feature vectors)
        selected_names = {c.targeting_vector.target + c.chelator.name for c, _ in ranked}
        # At most one "PSMADOTA" entry
        psma_dota_count = sum(
            1 for c, _ in ranked
            if c.targeting_vector.target == "PSMA" and c.chelator.name == "DOTA"
        )
        assert psma_dota_count <= 1

    def test_ucb_acquisition_works(self):
        policy = self._make_policy(acquisition_fn="UCB")
        candidates = make_diverse_constructs()
        ranked = policy.rank(candidates, batch_size=3)
        assert len(ranked) > 0

    def test_ei_acquisition_works(self):
        policy = self._make_policy(acquisition_fn="EI")
        candidates = make_diverse_constructs()
        ranked = policy.rank(candidates, batch_size=3)
        assert len(ranked) > 0

    def test_thompson_acquisition_works(self):
        policy = self._make_policy(acquisition_fn="thompson")
        candidates = make_diverse_constructs()
        ranked = policy.rank(candidates, batch_size=3)
        assert len(ranked) > 0

    def test_update_best(self):
        policy = self._make_policy()
        assert policy._best_known == 0.0
        policy.update_best(0.75)
        assert policy._best_known == 0.75
        policy.update_best(0.50)  # lower, should not change
        assert policy._best_known == 0.75

    def test_ranked_scores_are_sorted_descending(self):
        policy = self._make_policy(diversity_threshold=0.0)  # no diversity constraint
        candidates = make_diverse_constructs()
        ranked = policy.rank(candidates, batch_size=len(candidates))
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)
