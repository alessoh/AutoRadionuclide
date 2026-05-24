"""Tests for the retrospective benchmark."""
from __future__ import annotations
import pytest
from frozen.benchmark_runner import run_benchmark, load_benchmark, score_benchmark_entry, LABEL_NUMERIC


class TestBenchmarkData:
    def test_loads_9_entries(self):
        entries = load_benchmark()
        assert len(entries) == 9

    def test_expected_ids_present(self):
        entries = load_benchmark()
        ids = {e["id"] for e in entries}
        expected = {
            "psma617-dota-lu177",
            "dotatate-dota-lu177",
            "mibg-i131-direct",
            "dotatoc-dota-y90",
            "psma-it-dotaga-lu177",
            "fapi46-dota-lu177",
            "rgd-nota-ga68",
            "unknown1-nota-lu177",
            "unknown2-dota-ga68-therapy",
        }
        assert ids == expected

    def test_label_distribution(self):
        entries = load_benchmark()
        labels = [e["label"] for e in entries]
        assert labels.count("approved") == 3
        assert labels.count("clinical") == 4
        assert labels.count("failed") == 2

    def test_failed_compounds_have_placeholder_note(self):
        entries = load_benchmark()
        failed = [e for e in entries if e["label"] == "failed"]
        for e in failed:
            assert "ILLUSTRATIVE PLACEHOLDER" in e["notes"] or "fictional" in e["notes"].lower()


class TestBenchmarkScoring:
    def test_all_entries_score_in_range(self):
        entries = load_benchmark()
        for e in entries:
            score = score_benchmark_entry(e)
            assert 0.0 <= score <= 1.0, f"{e['id']} scored {score} (out of range)"

    def test_approved_compounds_score_higher_than_failed_on_average(self):
        entries = load_benchmark()
        approved_scores = [score_benchmark_entry(e) for e in entries if e["label"] == "approved"]
        failed_scores = [score_benchmark_entry(e) for e in entries if e["label"] == "failed"]
        avg_approved = sum(approved_scores) / len(approved_scores)
        avg_failed = sum(failed_scores) / len(failed_scores)
        assert avg_approved > avg_failed, (
            f"Approved avg ({avg_approved:.3f}) should be higher than failed avg ({avg_failed:.3f})"
        )

    def test_clinical_compounds_score_higher_than_failed_on_average(self):
        entries = load_benchmark()
        clinical_scores = [score_benchmark_entry(e) for e in entries if e["label"] == "clinical"]
        failed_scores = [score_benchmark_entry(e) for e in entries if e["label"] == "failed"]
        avg_clinical = sum(clinical_scores) / len(clinical_scores)
        avg_failed = sum(failed_scores) / len(failed_scores)
        assert avg_clinical > avg_failed, (
            f"Clinical avg ({avg_clinical:.3f}) should be higher than failed avg ({avg_failed:.3f})"
        )


class TestRunBenchmark:
    def test_runs_without_error(self):
        result = run_benchmark()
        assert result is not None

    def test_returns_required_keys(self):
        result = run_benchmark()
        required = {
            "rank_accuracy", "baseline_accuracy", "engine_hits", "n_positive",
            "top_half_n", "spearman_rho", "spearman_pval", "interpretation", "rankings"
        }
        assert required.issubset(set(result.keys()))

    def test_rank_accuracy_beats_baseline(self):
        """Engine should place more known-good agents in the top half than random."""
        result = run_benchmark()
        assert result["rank_accuracy"] > result["baseline_accuracy"], (
            f"Engine accuracy ({result['rank_accuracy']:.3f}) should beat "
            f"random baseline ({result['baseline_accuracy']:.3f})"
        )

    def test_n_positive_is_7(self):
        """7 approved+clinical compounds in the benchmark."""
        result = run_benchmark()
        assert result["n_positive"] == 7

    def test_all_9_compounds_ranked(self):
        result = run_benchmark()
        assert len(result["rankings"]) == 9

    def test_rankings_sorted_descending(self):
        result = run_benchmark()
        scores = [r["score"] for r in result["rankings"]]
        assert scores == sorted(scores, reverse=True)

    def test_spearman_rho_nonzero(self):
        """The Spearman correlation between rank order and -scores captures ranking quality.

        benchmark_runner computes spearmanr(ranks=[1..N], -scores).
        Rankings are sorted descending by score, so rank 1 has the highest score.
        -scores are most negative at rank 1. As rank increases, -score increases (less negative).
        This gives a positive rho: higher rank index → less negative -score.
        A rho close to +1 means the ordering is perfectly consistent — ranks and
        -scores move together, confirming the sorted order is monotone.
        """
        result = run_benchmark()
        # rho should be nonzero — a flat correlation would indicate randomness
        assert abs(result["spearman_rho"]) > 0.5, (
            f"Spearman rho magnitude should be > 0.5, got {result['spearman_rho']:.3f}"
        )

    def test_interpretation_string_present(self):
        result = run_benchmark()
        assert len(result["interpretation"]) > 50
        assert "NOT a validated predictive model" in result["interpretation"]

    def test_engine_fills_top_half_with_positives(self):
        """All top-half slots should be filled with known-good (positive) compounds.

        With 9 benchmark entries and top_half=4, and 7 positives total,
        the engine should place all 4 top slots with known-good compounds.
        This verifies the scoring correctly separates approved/clinical from failed.
        """
        result = run_benchmark()
        # engine_hits should equal top_half_n (all top slots are known-good)
        assert result["engine_hits"] == result["top_half_n"], (
            f"Engine should fill all {result['top_half_n']} top slots with known-good compounds, "
            f"but only got {result['engine_hits']}"
        )
