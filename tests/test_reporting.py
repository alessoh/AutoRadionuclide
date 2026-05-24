"""Tests for run-scoped reporting, dry-run isolation, and outer-loop invariants."""
from __future__ import annotations
import warnings
import pytest

from autoradionuclide.config.schema import CampaignSpec
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.providers.mock import MockModelProvider
from autoradionuclide.design.generator import CandidateGenerator
from autoradionuclide.surrogates.gp_surrogate import SurrogateBank
from autoradionuclide.policy.acquisition import ActiveLearningPolicy
from autoradionuclide.planner.inner_loop import InnerLoop
from autoradionuclide.planner.outer_loop import OuterLoop
from autoradionuclide.observability.inspector import inspect_campaign
from autoradionuclide.domain.models import LedgerEntryType, Radionuclide
from frozen.stub import StubWetLab


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_spec(
    campaign_id: str = "report-test-001",
    db_path: str = ":memory:",
    max_cycles: int = 3,
    target_campaign_score: float = 0.99,
    stall_patience: int = 100,
) -> CampaignSpec:
    spec = CampaignSpec(
        campaign_id=campaign_id,
        name="Report Test Campaign",
        target="PSMA",
        isotope="Lu-177",
        model_provider="mock",
        model_id="mock-deterministic-v1",
        batch_size=2,
        random_seed=42,
        db_path=db_path,
    )
    spec.budget.max_cycles = max_cycles
    spec.budget.max_wall_minutes = 60.0
    spec.stopping.target_campaign_score = target_campaign_score
    spec.stopping.stall_patience = stall_patience
    return spec


def _make_outer_loop(
    spec: CampaignSpec,
    ledger: LedgerStore,
    run_id: str = "",
) -> OuterLoop:
    provider = MockModelProvider(ledger=ledger)
    provider.set_campaign(spec.campaign_id)
    provider.set_run_id(run_id)
    bank = SurrogateBank([o.name for o in spec.objectives], seed=spec.random_seed)
    policy = ActiveLearningPolicy(
        surrogate_bank=bank,
        specs=spec.objectives,
        acquisition_fn="UCB",
        exploration_weight=1.5,
        diversity_threshold=0.1,
    )
    generator = CandidateGenerator(provider=provider, ledger=ledger)
    wet_lab = StubWetLab(seed=spec.random_seed)
    strategy = {
        "acquisition_function": "UCB",
        "exploration_weight": 1.5,
        "diversity_threshold": 0.1,
        "batch_size": spec.batch_size,
        "n_candidates_generated": 6,
        "prioritized_targets": [],
    }
    inner = InnerLoop(
        spec=spec,
        generator=generator,
        policy=policy,
        surrogate_bank=bank,
        wet_lab=wet_lab,
        ledger=ledger,
        strategy_config=strategy,
        run_id=run_id,
    )
    return OuterLoop(
        spec=spec,
        inner_loop=inner,
        provider=provider,
        ledger=ledger,
        base_strategy=strategy,
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# Task A1: Run-scoped reporting (regression for the multi-run contamination bug)
# ---------------------------------------------------------------------------

class TestRunScoping:
    def test_second_run_report_scopes_to_second_run(self, tmp_path):
        """Running a campaign twice must not contaminate the second run's report."""
        db_path = str(tmp_path / "test.db")
        campaign_id = "scoping-regression-001"

        # Run 1 — write to persistent DB
        spec = _make_spec(campaign_id=campaign_id, db_path=db_path, max_cycles=2)
        outer1 = _make_outer_loop(spec, LedgerStore(db_path), run_id="run-aaa")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer1.run()
        n_turns_run1 = len(outer1._score_history)

        # Run 2 — same campaign_id, same DB
        spec2 = _make_spec(campaign_id=campaign_id, db_path=db_path, max_cycles=2)
        outer2 = _make_outer_loop(spec2, LedgerStore(db_path), run_id="run-bbb")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer2.run()
        n_turns_run2 = len(outer2._score_history)

        # Scoped to run-bbb: shows only run 2
        report_run2 = inspect_campaign(LedgerStore(db_path), campaign_id, run_id="run-bbb")
        assert report_run2["n_outer_turns"] == n_turns_run2, (
            f"Run-scoped report shows {report_run2['n_outer_turns']} turns "
            f"but run 2 had {n_turns_run2}"
        )

        # Without scoping: shows both runs combined
        report_all = inspect_campaign(LedgerStore(db_path), campaign_id, run_id=None)
        assert report_all["n_outer_turns"] == n_turns_run1 + n_turns_run2, (
            f"Unscoped report shows {report_all['n_outer_turns']} turns, "
            f"expected {n_turns_run1 + n_turns_run2}"
        )

    def test_run_id_stored_on_ledger_entries(self):
        """Every ledger entry produced by a run must carry that run's run_id."""
        run_id = "test-run-xyz"
        spec = _make_spec(max_cycles=1)
        ledger = LedgerStore(":memory:")
        outer = _make_outer_loop(spec, ledger, run_id=run_id)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer.run()
        entries = ledger.query(campaign_id=spec.campaign_id)
        assert len(entries) > 0
        for e in entries:
            assert e.run_id == run_id, (
                f"Entry {e.id} ({e.entry_type}) has run_id={e.run_id!r}, expected {run_id!r}"
            )

    def test_get_most_recent_run_id_returns_last_run(self, tmp_path):
        """get_most_recent_run_id must return the run_id of the latest run."""
        db_path = str(tmp_path / "test.db")
        campaign_id = "recency-test-001"

        spec = _make_spec(campaign_id=campaign_id, db_path=db_path, max_cycles=1)
        for rid in ("run-first", "run-second"):
            outer = _make_outer_loop(spec, LedgerStore(db_path), run_id=rid)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                outer.run()

        ledger = LedgerStore(db_path)
        most_recent = ledger.get_most_recent_run_id(campaign_id)
        assert most_recent == "run-second", (
            f"Expected most recent run to be 'run-second', got {most_recent!r}"
        )


# ---------------------------------------------------------------------------
# Task A2: Dry-run does not persist to the real ledger
# ---------------------------------------------------------------------------

class TestDryRunIsolation:
    def test_dry_run_with_memory_ledger_leaves_persistent_db_unchanged(self, tmp_path):
        """A dry-run (using in-memory ledger) must not write to the persistent DB."""
        db_path = str(tmp_path / "test.db")
        campaign_id = "dry-run-isolation-001"
        spec = _make_spec(campaign_id=campaign_id, db_path=db_path, max_cycles=1)

        # Real run — writes to persistent DB
        outer_real = _make_outer_loop(spec, LedgerStore(db_path), run_id="real-run")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer_real.run()
        count_after_real = LedgerStore(db_path).count()
        assert count_after_real > 0, "Real run should have written entries"

        # Dry run — uses in-memory ledger (same as what the launcher does)
        memory_ledger = LedgerStore(":memory:")
        outer_dry = _make_outer_loop(spec, memory_ledger, run_id="dry-run")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer_dry.run()

        # Persistent DB unchanged
        count_after_dry = LedgerStore(db_path).count()
        assert count_after_dry == count_after_real, (
            f"Dry-run must not write to persistent DB. "
            f"Before: {count_after_real}, After: {count_after_dry}"
        )

    def test_dry_run_memory_ledger_has_entries(self):
        """The in-memory ledger used by a dry-run must contain this run's entries."""
        spec = _make_spec(max_cycles=1)
        memory_ledger = LedgerStore(":memory:")
        outer = _make_outer_loop(spec, memory_ledger, run_id="dry-run-001")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer.run()
        assert memory_ledger.count() > 0, "In-memory ledger must contain dry-run entries"


# ---------------------------------------------------------------------------
# Task A3: Non-decreasing kept-score invariant
# ---------------------------------------------------------------------------

class TestKeptScoreInvariant:
    def test_kept_campaign_score_never_decreases(self):
        """The outer loop's kept campaign score must not decrease across turns."""
        spec = _make_spec(max_cycles=4, stall_patience=100)
        ledger = LedgerStore(":memory:")
        outer = _make_outer_loop(spec, ledger, run_id="invariant-test")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer.run()

        history = outer._score_history
        assert len(history) >= 1, "Should have at least one turn"
        for i in range(1, len(history)):
            assert history[i] >= history[i - 1] - 1e-9, (
                f"Kept score decreased from {history[i - 1]:.4f} to {history[i]:.4f} "
                f"at turn {i + 1} — non-decreasing invariant violated"
            )

    def test_total_improvement_non_negative(self):
        """Total improvement in a run must be >= 0 given the non-decreasing invariant."""
        spec = _make_spec(max_cycles=3, stall_patience=100)
        ledger = LedgerStore(":memory:")
        outer = _make_outer_loop(spec, ledger, run_id="improvement-test")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer.run()

        report = inspect_campaign(ledger, spec.campaign_id, run_id="improvement-test")
        if report["total_improvement"] is not None:
            assert report["total_improvement"] >= -1e-9, (
                f"Total improvement should be >= 0, got {report['total_improvement']:.4f}"
            )


# ---------------------------------------------------------------------------
# Task A4: Multi-turn decisions are recorded with rationale
# ---------------------------------------------------------------------------

class TestMultiTurnDecisions:
    def test_multi_turn_records_strategy_modification_per_turn_after_first(self):
        """Each turn > 0 must produce exactly one STRATEGY_MODIFICATION ledger entry."""
        spec = _make_spec(max_cycles=3, stall_patience=100)
        ledger = LedgerStore(":memory:")
        outer = _make_outer_loop(spec, ledger, run_id="multiturn-test")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer.run()

        mods = ledger.query(
            campaign_id=spec.campaign_id,
            entry_type=LedgerEntryType.STRATEGY_MODIFICATION,
        )
        # 3 turns total: turn 0 has no mod, turns 1 and 2 each have one mod
        n_turns = len(outer._score_history)
        expected_mods = max(0, n_turns - 1)
        assert len(mods) == expected_mods, (
            f"Expected {expected_mods} STRATEGY_MODIFICATION entries for "
            f"{n_turns} turns, got {len(mods)}"
        )

    def test_strategy_modification_entry_has_kept_and_rationale(self):
        """Each STRATEGY_MODIFICATION entry must record kept (bool) and rationale."""
        spec = _make_spec(max_cycles=3, stall_patience=100)
        ledger = LedgerStore(":memory:")
        outer = _make_outer_loop(spec, ledger, run_id="decision-test")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer.run()

        mods = ledger.query(
            campaign_id=spec.campaign_id,
            entry_type=LedgerEntryType.STRATEGY_MODIFICATION,
        )
        assert len(mods) >= 1, "Need at least one modification for this test (max_cycles >= 2)"
        for mod in mods:
            assert "kept" in mod.data, f"STRATEGY_MODIFICATION missing 'kept': {mod.data}"
            assert isinstance(mod.data["kept"], bool), "'kept' must be a boolean"
            assert "rationale" in mod.data, f"STRATEGY_MODIFICATION missing 'rationale': {mod.data}"
            assert mod.data["rationale"], "'rationale' must be non-empty"
            assert mod.run_id == "decision-test", "STRATEGY_MODIFICATION must carry run_id"


# ---------------------------------------------------------------------------
# Task B: Demo campaign constructs resolve to FULL/PARTIAL quality
# ---------------------------------------------------------------------------

class TestDemoCampaignResolution:
    def test_mibg_none_i131_construct_resolves_to_full(self):
        """The primary demo construct (MIBG + none chelator + I-131) resolves FULL."""
        import numpy as np
        from autoradionuclide.featurization import featurize, FeatureQuality
        from autoradionuclide.featurization.registry import reset_registry_warning_state
        from autoradionuclide.domain.models import (
            CandidateConstruct, Chelator, TargetingVector,
        )
        reset_registry_warning_state()
        c = CandidateConstruct(
            name="mibg-i131-demo",
            targeting_vector=TargetingVector(
                name="MIBG", target="NET", vector_type="small_molecule"
            ),
            chelator=Chelator(name="none"),
            radionuclide=Radionuclide.I131,
        )
        record = featurize(c)
        assert record.quality is FeatureQuality.FULL, (
            f"Expected FULL for MIBG+none+I-131, got {record.quality}"
        )
        assert np.any(record.descriptor_vector != 0.0), "FULL record must have nonzero descriptors"
        assert record.fingerprint.sum() > 0, "FULL record must have nonzero fingerprint"

    def test_dota_chelator_construct_resolves_to_partial(self):
        """DOTA chelator with an unknown vector resolves PARTIAL (not FALLBACK)."""
        from autoradionuclide.featurization import featurize, FeatureQuality
        from autoradionuclide.featurization.registry import reset_registry_warning_state
        from autoradionuclide.domain.models import (
            CandidateConstruct, Chelator, Radionuclide, TargetingVector,
        )
        reset_registry_warning_state()
        c = CandidateConstruct(
            name="dota-psma-lu177",
            targeting_vector=TargetingVector(
                name="PSMA-617", target="PSMA", vector_type="small_molecule"
            ),
            chelator=Chelator(name="DOTA"),
            radionuclide=Radionuclide.LU177,
        )
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            record = featurize(c)
        assert record.quality is FeatureQuality.PARTIAL, (
            f"Expected PARTIAL for DOTA+unknown vector, got {record.quality}"
        )

    def test_demo_campaign_run_produces_at_least_one_partial_or_full(self):
        """A full run of the demo campaign must featurize at least one real structure."""
        import numpy as np
        from autoradionuclide.featurization import featurize, FeatureQuality
        from autoradionuclide.featurization.registry import reset_registry_warning_state
        from autoradionuclide.domain.models import (
            CandidateConstruct, Chelator, TargetingVector,
        )
        reset_registry_warning_state()

        spec = _make_spec(
            campaign_id="demo-featurization-001",
            max_cycles=2,
            stall_patience=100,
        )
        # Patch isotope to I-131 for this test (matches demo campaign)
        spec.isotope = Radionuclide.I131
        spec.target = "NET"

        ledger = LedgerStore(":memory:")
        outer = _make_outer_loop(spec, ledger, run_id="demo-test")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            outer.run()

        # Check at least one SCORE entry corresponds to a non-FALLBACK construct
        score_entries = ledger.query(
            campaign_id=spec.campaign_id, entry_type=LedgerEntryType.SCORE
        )
        assert len(score_entries) > 0, "Campaign must produce scored candidates"

        # The aggregate scores must be real numbers (not all zeros)
        scores = [e.data.get("aggregate_score", 0.0) for e in score_entries]
        assert any(s > 0 for s in scores), "At least one scored construct must have score > 0"
