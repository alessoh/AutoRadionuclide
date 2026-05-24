"""Tests for InnerLoop and OuterLoop planners."""
from __future__ import annotations
import pytest
from autoradionuclide.config.schema import CampaignSpec
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.providers.mock import MockModelProvider
from autoradionuclide.design.generator import CandidateGenerator
from autoradionuclide.surrogates.gp_surrogate import SurrogateBank
from autoradionuclide.policy.acquisition import ActiveLearningPolicy
from autoradionuclide.planner.inner_loop import InnerLoop
from autoradionuclide.planner.outer_loop import OuterLoop
from autoradionuclide.provenance.context import ProvenanceContext
from autoradionuclide.domain.models import LedgerEntryType
from frozen.stub import StubWetLab


def make_test_spec(**overrides) -> CampaignSpec:
    defaults = dict(
        campaign_id="test-campaign-001",
        name="Test PSMA Campaign",
        target="PSMA",
        isotope="Lu-177",
        model_provider="mock",
        model_id="mock-deterministic-v1",
        batch_size=2,
        random_seed=42,
        db_path=":memory:",
    )
    defaults.update(overrides)
    # Parse budget/stopping sub-specs
    spec = CampaignSpec(**defaults)
    spec.budget.max_cycles = overrides.get("max_cycles", 3)
    spec.budget.max_wall_minutes = 60.0
    return spec


def make_inner_loop(spec: CampaignSpec, ledger: LedgerStore) -> InnerLoop:
    provider = MockModelProvider(ledger=ledger)
    provider.set_campaign(spec.campaign_id)
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
    return InnerLoop(
        spec=spec,
        generator=generator,
        policy=policy,
        surrogate_bank=bank,
        wet_lab=wet_lab,
        ledger=ledger,
        strategy_config=strategy,
    )


class TestInnerLoop:
    def test_run_returns_cycle_result(self):
        spec = make_test_spec()
        ledger = LedgerStore(":memory:")
        inner = make_inner_loop(spec, ledger)
        provenance = ProvenanceContext.from_config("mock", spec.model_dump(mode="json"), seed=42)
        result = inner.run(cycle_number=0, provenance=provenance, campaign_score_before=0.0)
        assert result.campaign_id == spec.campaign_id
        assert result.cycle_number == 0
        assert result.constructs_proposed > 0
        assert result.constructs_scored > 0

    def test_cycle_result_has_score(self):
        spec = make_test_spec()
        ledger = LedgerStore(":memory:")
        inner = make_inner_loop(spec, ledger)
        provenance = ProvenanceContext.from_config("mock", spec.model_dump(mode="json"), seed=42)
        result = inner.run(cycle_number=0, provenance=provenance, campaign_score_before=0.0)
        assert isinstance(result.campaign_score_after, float)
        assert 0.0 <= result.campaign_score_after <= 1.0

    def test_ledger_has_expected_entry_types(self):
        spec = make_test_spec()
        ledger = LedgerStore(":memory:")
        inner = make_inner_loop(spec, ledger)
        provenance = ProvenanceContext.from_config("mock", spec.model_dump(mode="json"), seed=42)
        inner.run(cycle_number=0, provenance=provenance, campaign_score_before=0.0)

        entries = ledger.query(campaign_id=spec.campaign_id)
        entry_types = {e.entry_type for e in entries}

        # Must have all these entry types from one cycle
        required = {
            LedgerEntryType.PROPOSAL,
            LedgerEntryType.SCORE,
            LedgerEntryType.SELECTION,
            LedgerEntryType.APPROVAL,
            LedgerEntryType.REQUEST,
            LedgerEntryType.RESULT,
            LedgerEntryType.CYCLE_SUMMARY,
        }
        assert required.issubset(entry_types), (
            f"Missing entry types: {required - entry_types}"
        )

    def test_ledger_has_surrogate_refit_after_results(self):
        spec = make_test_spec()
        ledger = LedgerStore(":memory:")
        inner = make_inner_loop(spec, ledger)
        provenance = ProvenanceContext.from_config("mock", spec.model_dump(mode="json"), seed=42)
        inner.run(cycle_number=0, provenance=provenance, campaign_score_before=0.0)
        entries = ledger.query(
            campaign_id=spec.campaign_id,
            entry_type=LedgerEntryType.SURROGATE_REFIT,
        )
        assert len(entries) >= 1

    def test_cycle_score_delta_computed(self):
        spec = make_test_spec()
        ledger = LedgerStore(":memory:")
        inner = make_inner_loop(spec, ledger)
        provenance = ProvenanceContext.from_config("mock", spec.model_dump(mode="json"), seed=42)
        result = inner.run(cycle_number=0, provenance=provenance, campaign_score_before=0.5)
        assert abs(result.score_delta - (result.campaign_score_after - 0.5)) < 1e-6

    def test_dry_run_adds_warning(self):
        spec = make_test_spec()
        ledger = LedgerStore(":memory:")
        inner = make_inner_loop(spec, ledger)
        provenance = ProvenanceContext.from_config("mock", spec.model_dump(mode="json"), seed=42)
        # Should not raise
        result = inner.run(
            cycle_number=0, provenance=provenance, campaign_score_before=0.0, dry_run=True
        )
        assert result is not None


class TestOuterLoop:
    def make_outer_loop(
        self, spec: CampaignSpec, ledger: LedgerStore
    ) -> tuple[OuterLoop, InnerLoop]:
        provider = MockModelProvider(ledger=ledger)
        provider.set_campaign(spec.campaign_id)
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
        )
        outer = OuterLoop(
            spec=spec,
            inner_loop=inner,
            provider=provider,
            ledger=ledger,
            base_strategy=strategy,
        )
        return outer, inner

    def test_run_2_cycles_produces_2_summaries(self):
        spec = make_test_spec()
        spec.budget.max_cycles = 2
        spec.stopping.stall_patience = 10  # don't stop early
        spec.stopping.target_campaign_score = 2.0  # unreachable
        ledger = LedgerStore(":memory:")
        outer, _ = self.make_outer_loop(spec, ledger)
        summaries = outer.run(dry_run=False)
        assert len(summaries) == 2

    def test_summaries_have_score_history(self):
        spec = make_test_spec()
        spec.budget.max_cycles = 2
        spec.stopping.stall_patience = 10
        spec.stopping.target_campaign_score = 2.0
        ledger = LedgerStore(":memory:")
        outer, _ = self.make_outer_loop(spec, ledger)
        summaries = outer.run(dry_run=False)
        for s in summaries:
            assert "campaign_score_before" in s
            assert "campaign_score_after" in s
            assert "cycle_id" in s

    def test_score_history_accumulates(self):
        spec = make_test_spec()
        spec.budget.max_cycles = 3
        spec.stopping.stall_patience = 10
        spec.stopping.target_campaign_score = 2.0
        ledger = LedgerStore(":memory:")
        outer, _ = self.make_outer_loop(spec, ledger)
        summaries = outer.run(dry_run=False)
        assert len(outer._score_history) == len(summaries)

    def test_ledger_has_strategy_modification_entries(self):
        spec = make_test_spec()
        spec.budget.max_cycles = 3
        spec.stopping.stall_patience = 10
        spec.stopping.target_campaign_score = 2.0
        ledger = LedgerStore(":memory:")
        outer, _ = self.make_outer_loop(spec, ledger)
        outer.run(dry_run=False)
        # After turn 0, strategy modifications are proposed
        entries = ledger.query(
            campaign_id=spec.campaign_id,
            entry_type=LedgerEntryType.STRATEGY_MODIFICATION,
        )
        # Should have at least one (from cycle 2+ when turn > 0)
        assert len(entries) >= 1

    def test_stopping_on_target_score(self):
        spec = make_test_spec()
        spec.budget.max_cycles = 10
        spec.stopping.target_campaign_score = -1.0  # immediately met on any positive score
        spec.stopping.stall_patience = 100
        ledger = LedgerStore(":memory:")
        outer, _ = self.make_outer_loop(spec, ledger)
        # Preset score so stopping condition is met immediately
        outer._campaign_score = 0.0  # target_campaign_score = -1, so already met
        summaries = outer.run(dry_run=False)
        # Should stop before running any cycles
        assert len(summaries) == 0
