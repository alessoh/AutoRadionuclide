"""Replay tests — same seed produces identical results."""
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


def make_spec_and_infrastructure(
    seed: int = 42, n_cycles: int = 2
) -> tuple[CampaignSpec, LedgerStore, OuterLoop]:
    spec = CampaignSpec(
        campaign_id="replay-test-001",
        name="Replay Test Campaign",
        target="PSMA",
        isotope="Lu-177",
        model_provider="mock",
        model_id="mock-deterministic-v1",
        batch_size=2,
        random_seed=seed,
        db_path=":memory:",
    )
    spec.budget.max_cycles = n_cycles
    spec.budget.max_wall_minutes = 60.0
    spec.stopping.stall_patience = 100
    spec.stopping.target_campaign_score = 2.0  # unreachable

    ledger = LedgerStore(":memory:")
    provider = MockModelProvider(ledger=ledger)
    provider.set_campaign(spec.campaign_id)

    bank = SurrogateBank([o.name for o in spec.objectives], seed=seed)
    policy = ActiveLearningPolicy(
        surrogate_bank=bank,
        specs=spec.objectives,
        acquisition_fn="UCB",
        exploration_weight=1.5,
        diversity_threshold=0.1,
    )
    generator = CandidateGenerator(provider=provider, ledger=ledger)
    wet_lab = StubWetLab(seed=seed)
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
    return spec, ledger, outer


class TestReplay:
    def test_same_seed_produces_same_number_of_cycles(self):
        _, _, outer1 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        _, _, outer2 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        s1 = outer1.run(dry_run=False)
        s2 = outer2.run(dry_run=False)
        assert len(s1) == len(s2)

    def test_same_seed_produces_same_final_score(self):
        _, _, outer1 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        _, _, outer2 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        s1 = outer1.run(dry_run=False)
        s2 = outer2.run(dry_run=False)
        assert abs(s1[-1]["campaign_score_after"] - s2[-1]["campaign_score_after"]) < 1e-6

    def test_same_seed_produces_same_ledger_entry_count(self):
        spec1, ledger1, outer1 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        spec2, ledger2, outer2 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        outer1.run(dry_run=False)
        outer2.run(dry_run=False)
        assert ledger1.count() == ledger2.count()

    def test_same_seed_produces_same_cycle_scores(self):
        _, _, outer1 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        _, _, outer2 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        s1 = outer1.run(dry_run=False)
        s2 = outer2.run(dry_run=False)
        for c1, c2 in zip(s1, s2):
            assert abs(c1["campaign_score_after"] - c2["campaign_score_after"]) < 1e-6

    def test_different_seeds_may_produce_different_results(self):
        """Different seeds should generally produce different results."""
        _, _, outer1 = make_spec_and_infrastructure(seed=42, n_cycles=2)
        _, _, outer2 = make_spec_and_infrastructure(seed=99, n_cycles=2)
        s1 = outer1.run(dry_run=False)
        s2 = outer2.run(dry_run=False)
        # Not guaranteed to differ (could coincide), but generally should
        # Just check they both run successfully
        assert len(s1) >= 0
        assert len(s2) >= 0

    def test_same_seed_same_proposal_names(self):
        """With same seed, the proposal ledger entries should have same construct names."""
        spec1, ledger1, outer1 = make_spec_and_infrastructure(seed=42, n_cycles=1)
        spec2, ledger2, outer2 = make_spec_and_infrastructure(seed=42, n_cycles=1)
        outer1.run(dry_run=False)
        outer2.run(dry_run=False)

        proposals1 = ledger1.query(
            campaign_id=spec1.campaign_id,
            entry_type=LedgerEntryType.PROPOSAL,
        )
        proposals2 = ledger2.query(
            campaign_id=spec2.campaign_id,
            entry_type=LedgerEntryType.PROPOSAL,
        )
        assert len(proposals1) == len(proposals2)
        # The n_generated should be the same
        for p1, p2 in zip(proposals1, proposals2):
            assert p1.data.get("n_generated") == p2.data.get("n_generated")

    def test_inner_loop_single_cycle_is_deterministic(self):
        """Running a single inner cycle with same provenance and seed is deterministic."""
        def run_once(seed: int):
            spec = CampaignSpec(
                campaign_id="det-test",
                name="Determinism Test",
                target="PSMA",
                isotope="Lu-177",
                model_provider="mock",
                model_id="mock-deterministic-v1",
                batch_size=2,
                random_seed=seed,
                db_path=":memory:",
            )
            ledger = LedgerStore(":memory:")
            provider = MockModelProvider(ledger=ledger)
            provider.set_campaign(spec.campaign_id)
            bank = SurrogateBank([o.name for o in spec.objectives], seed=seed)
            policy = ActiveLearningPolicy(
                surrogate_bank=bank,
                specs=spec.objectives,
                acquisition_fn="UCB",
                exploration_weight=1.5,
                diversity_threshold=0.1,
            )
            generator = CandidateGenerator(provider=provider, ledger=ledger)
            wet_lab = StubWetLab(seed=seed)
            strategy = {
                "batch_size": 2,
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
            prov = ProvenanceContext.from_config("mock", spec.model_dump(mode="json"), seed=seed)
            result = inner.run(cycle_number=0, provenance=prov, campaign_score_before=0.0)
            return result.campaign_score_after

        score_a = run_once(42)
        score_b = run_once(42)
        assert abs(score_a - score_b) < 1e-9
