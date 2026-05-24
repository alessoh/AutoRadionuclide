"""Tests for the frozen harness scoring functions."""
from __future__ import annotations
import pytest
from autoradionuclide.domain.models import (
    CandidateConstruct, TargetingVector, Chelator, Radionuclide, ProvenanceTag
)
import frozen.harness as harness


def make_construct(
    target: str = "PSMA",
    chelator: str = "DOTA",
    vector_type: str = "small_molecule",
    isotope: Radionuclide = Radionuclide.LU177,
    linker: str | None = None,
    name: str = "test-construct",
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


class TestScoreAll:
    def test_returns_all_objectives(self):
        c = make_construct()
        scores = harness.score_all(c)
        expected = {
            "binding_affinity",
            "chelator_stability",
            "half_life_compatibility",
            "synthetic_feasibility",
            "selectivity",
        }
        assert set(scores.keys()) == expected

    def test_all_values_in_zero_one(self):
        c = make_construct()
        scores = harness.score_all(c)
        for name, ov in scores.items():
            assert 0.0 <= ov.estimate <= 1.0, f"{name} estimate out of range: {ov.estimate}"
            assert ov.uncertainty >= 0.0, f"{name} has negative uncertainty"


class TestBindingAffinity:
    def test_psma_scores_high(self):
        c = make_construct(target="PSMA")
        ov = harness.score_binding_affinity(c)
        assert ov.estimate >= 0.85, f"PSMA binding affinity too low: {ov.estimate}"

    def test_sstr2_scores_high(self):
        c = make_construct(target="SSTR2")
        ov = harness.score_binding_affinity(c)
        assert ov.estimate >= 0.80

    def test_unknown_target_scores_low(self):
        c = make_construct(target="unknown")
        ov = harness.score_binding_affinity(c)
        assert ov.estimate <= 0.35

    def test_source_tag_is_heuristic(self):
        c = make_construct()
        ov = harness.score_binding_affinity(c)
        assert ov.source == ProvenanceTag.HEURISTIC

    def test_docstring_contains_placeholder(self):
        assert "PLACEHOLDER" in harness.score_binding_affinity.__doc__
        assert "not a validated predictive model" in harness.score_binding_affinity.__doc__

    def test_uncertainty_positive(self):
        c = make_construct()
        ov = harness.score_binding_affinity(c)
        assert ov.uncertainty > 0


class TestChelatorStability:
    def test_dota_lu177_scores_high(self):
        c = make_construct(chelator="DOTA", isotope=Radionuclide.LU177)
        ov = harness.score_chelator_stability(c)
        assert ov.estimate >= 0.90

    def test_nota_ga68_scores_high(self):
        c = make_construct(chelator="NOTA", isotope=Radionuclide.GA68)
        ov = harness.score_chelator_stability(c)
        assert ov.estimate >= 0.90

    def test_nota_lu177_scores_low(self):
        """NOTA is known to form unstable complexes with Lu-177."""
        c = make_construct(chelator="NOTA", isotope=Radionuclide.LU177)
        ov = harness.score_chelator_stability(c)
        assert ov.estimate <= 0.60

    def test_dotaga_lu177_scores_high(self):
        c = make_construct(chelator="DOTAGA", isotope=Radionuclide.LU177)
        ov = harness.score_chelator_stability(c)
        assert ov.estimate >= 0.88

    def test_source_tag_is_heuristic(self):
        c = make_construct()
        ov = harness.score_chelator_stability(c)
        assert ov.source == ProvenanceTag.HEURISTIC

    def test_docstring_contains_placeholder(self):
        assert "PLACEHOLDER" in harness.score_chelator_stability.__doc__
        assert "not a validated predictive model" in harness.score_chelator_stability.__doc__


class TestHalfLifeCompatibility:
    def test_lu177_scores_high_for_therapy(self):
        c = make_construct(isotope=Radionuclide.LU177)
        ov = harness.score_half_life_compatibility(c)
        # Lu-177: 6.65d → min(1, 6.65/7) ≈ 0.95 + sweet-spot bonus
        assert ov.estimate >= 0.90

    def test_ac225_scores_high_for_therapy(self):
        c = make_construct(isotope=Radionuclide.AC225)
        ov = harness.score_half_life_compatibility(c)
        # Ac-225: 9.92d → min(1, 9.92/7) ≈ 1.0 + sweet-spot bonus
        assert ov.estimate >= 0.95

    def test_ga68_scores_very_low_for_therapy(self):
        """Ga-68 (68 min ≈ 0.047d) is an imaging agent, not suitable for therapy."""
        c = make_construct(isotope=Radionuclide.GA68)
        ov = harness.score_half_life_compatibility(c)
        assert ov.estimate <= 0.10, f"Ga-68 therapy score should be near 0, got {ov.estimate}"

    def test_source_tag_is_physics(self):
        c = make_construct()
        ov = harness.score_half_life_compatibility(c)
        assert ov.source == ProvenanceTag.PHYSICS

    def test_zero_uncertainty_for_physics(self):
        c = make_construct()
        ov = harness.score_half_life_compatibility(c)
        assert ov.uncertainty == 0.0

    def test_i131_scores_high_for_therapy(self):
        c = make_construct(isotope=Radionuclide.I131)
        ov = harness.score_half_life_compatibility(c)
        assert ov.estimate >= 0.90


class TestSyntheticFeasibility:
    def test_peptide_scores_high(self):
        c = make_construct(vector_type="peptide")
        ov = harness.score_synthetic_feasibility(c)
        assert ov.estimate >= 0.80

    def test_small_molecule_scores_high(self):
        c = make_construct(vector_type="small_molecule")
        ov = harness.score_synthetic_feasibility(c)
        assert ov.estimate >= 0.75

    def test_antibody_fragment_scores_lower(self):
        c_ab = make_construct(vector_type="antibody_fragment")
        c_sm = make_construct(vector_type="small_molecule")
        ov_ab = harness.score_synthetic_feasibility(c_ab)
        ov_sm = harness.score_synthetic_feasibility(c_sm)
        assert ov_ab.estimate < ov_sm.estimate

    def test_source_tag_is_heuristic(self):
        c = make_construct()
        ov = harness.score_synthetic_feasibility(c)
        assert ov.source == ProvenanceTag.HEURISTIC

    def test_docstring_contains_placeholder(self):
        assert "PLACEHOLDER" in harness.score_synthetic_feasibility.__doc__
        assert "not a validated predictive model" in harness.score_synthetic_feasibility.__doc__


class TestSelectivity:
    def test_returns_value_in_range(self):
        c = make_construct()
        ov = harness.score_selectivity(c)
        assert 0.0 <= ov.estimate <= 1.0

    def test_source_tag_is_heuristic(self):
        c = make_construct()
        ov = harness.score_selectivity(c)
        assert ov.source == ProvenanceTag.HEURISTIC

    def test_docstring_contains_placeholder(self):
        assert "PLACEHOLDER" in harness.score_selectivity.__doc__
        assert "not a validated predictive model" in harness.score_selectivity.__doc__


class TestKnownGoodVsKnownBad:
    def test_psma_dota_lu177_scores_higher_than_unknown_nota_lu177(self):
        """Approved PSMA-617-DOTA-Lu177 should score higher than unknown+NOTA+Lu177."""
        good = make_construct(
            target="PSMA", chelator="DOTA", isotope=Radionuclide.LU177,
            vector_type="small_molecule"
        )
        bad = make_construct(
            target="unknown", chelator="NOTA", isotope=Radionuclide.LU177,
            vector_type="small_molecule"
        )
        from autoradionuclide.domain.models import ScoredObjective, ObjectiveDirection
        from autoradionuclide.config.schema import _default_objectives
        from autoradionuclide.scoring.aggregator import aggregate_score

        specs = _default_objectives()

        good_scores = harness.score_all(good)
        bad_scores = harness.score_all(bad)

        good_objs = [
            ScoredObjective(name=k, value=v, direction=ObjectiveDirection.MAXIMIZE)
            for k, v in good_scores.items()
        ]
        bad_objs = [
            ScoredObjective(name=k, value=v, direction=ObjectiveDirection.MAXIMIZE)
            for k, v in bad_scores.items()
        ]

        good_agg, _ = aggregate_score(good_objs, specs)
        bad_agg, _ = aggregate_score(bad_objs, specs)

        assert good_agg > bad_agg, (
            f"Good construct ({good_agg:.3f}) should score higher than bad ({bad_agg:.3f})"
        )

    def test_ga68_wrong_for_therapy_scores_low(self):
        """Ga-68 is an imaging isotope; using it for therapy should score poorly."""
        c = make_construct(target="PSMA", chelator="DOTA", isotope=Radionuclide.GA68)
        ov = harness.score_half_life_compatibility(c)
        assert ov.estimate < 0.10
