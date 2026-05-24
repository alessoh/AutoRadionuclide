"""Tests for domain models."""
from __future__ import annotations
import pytest
from datetime import datetime, timezone
from autoradionuclide.domain.models import (
    Radionuclide, HALF_LIFE_DAYS, CandidateStatus, ObjectiveDirection,
    ProvenanceTag, TargetingVector, Chelator, CandidateConstruct,
    ObjectiveValue, ScoredObjective, ObjectiveSpec, ExperimentRequest,
    AssayResult, ResultRecord, CycleResult, LedgerEntryType, LedgerEntry,
    TokenUsage, ModelRequest, ModelResponse,
)


class TestRadionuclide:
    def test_all_enum_values(self):
        assert Radionuclide.LU177.value == "Lu-177"
        assert Radionuclide.AC225.value == "Ac-225"
        assert Radionuclide.GA68.value == "Ga-68"
        assert Radionuclide.Y90.value == "Y-90"
        assert Radionuclide.I131.value == "I-131"
        assert Radionuclide.BI213.value == "Bi-213"
        assert Radionuclide.AT211.value == "At-211"

    def test_half_life_completeness(self):
        """Every Radionuclide must have a HALF_LIFE_DAYS entry."""
        for r in Radionuclide:
            assert r in HALF_LIFE_DAYS, f"{r} missing from HALF_LIFE_DAYS"
            assert HALF_LIFE_DAYS[r] > 0

    def test_half_life_values_reasonable(self):
        assert HALF_LIFE_DAYS[Radionuclide.LU177] == pytest.approx(6.65)
        assert HALF_LIFE_DAYS[Radionuclide.AC225] == pytest.approx(9.92)
        # Ga-68 is about 0.047 days (68 minutes)
        assert HALF_LIFE_DAYS[Radionuclide.GA68] < 0.1
        assert HALF_LIFE_DAYS[Radionuclide.GA68] > 0.04


class TestCandidateConstruct:
    def _make_construct(self, **kwargs) -> CandidateConstruct:
        defaults = dict(
            name="PSMA617-DOTA-Lu177",
            targeting_vector=TargetingVector(
                name="PSMA-617", target="PSMA", vector_type="small_molecule"
            ),
            chelator=Chelator(name="DOTA"),
            radionuclide=Radionuclide.LU177,
        )
        defaults.update(kwargs)
        return CandidateConstruct(**defaults)

    def test_construction(self):
        c = self._make_construct()
        assert c.name == "PSMA617-DOTA-Lu177"
        assert c.status == CandidateStatus.PROPOSED
        assert c.radionuclide == Radionuclide.LU177

    def test_id_auto_generated(self):
        c1 = self._make_construct()
        c2 = self._make_construct()
        assert c1.id != c2.id
        assert len(c1.id) > 0

    def test_composite_key(self):
        c = self._make_construct()
        key = c.composite_key
        assert "PSMA617" in key or "PSMA" in key
        assert "DOTA" in key
        assert "nolinker" in key
        assert "Lu177" in key or "Lu" in key

    def test_composite_key_with_linker(self):
        c = self._make_construct(linker="PEG2")
        assert "PEG2" in c.composite_key

    def test_composite_key_differs_by_linker(self):
        c1 = self._make_construct(linker=None)
        c2 = self._make_construct(linker="PEG2")
        assert c1.composite_key != c2.composite_key

    def test_status_progression(self):
        c = self._make_construct()
        assert c.status == CandidateStatus.PROPOSED
        c.status = CandidateStatus.SCORED
        assert c.status == CandidateStatus.SCORED
        c.status = CandidateStatus.SELECTED
        assert c.status == CandidateStatus.SELECTED
        c.status = CandidateStatus.REQUESTED
        assert c.status == CandidateStatus.REQUESTED
        c.status = CandidateStatus.TESTED
        assert c.status == CandidateStatus.TESTED
        c.status = CandidateStatus.CONCLUDED
        assert c.status == CandidateStatus.CONCLUDED

    def test_created_at_is_utc(self):
        c = self._make_construct()
        assert c.created_at.tzinfo is not None


class TestObjectiveValue:
    def test_valid(self):
        ov = ObjectiveValue(estimate=0.8, uncertainty=0.1)
        assert ov.estimate == 0.8
        assert ov.uncertainty == 0.1

    def test_zero_uncertainty_ok(self):
        ov = ObjectiveValue(estimate=0.5, uncertainty=0.0)
        assert ov.uncertainty == 0.0

    def test_negative_uncertainty_raises(self):
        with pytest.raises(Exception):
            ObjectiveValue(estimate=0.5, uncertainty=-0.1)

    def test_default_source(self):
        ov = ObjectiveValue(estimate=0.5)
        assert ov.source == ProvenanceTag.PLACEHOLDER


class TestScoredObjective:
    def test_construction(self):
        ov = ObjectiveValue(estimate=0.7, uncertainty=0.05)
        so = ScoredObjective(name="binding_affinity", value=ov)
        assert so.name == "binding_affinity"
        assert so.direction == ObjectiveDirection.MAXIMIZE


class TestObjectiveSpec:
    def test_defaults(self):
        spec = ObjectiveSpec(name="test_obj")
        assert spec.direction == ObjectiveDirection.MAXIMIZE
        assert spec.weight == 1.0
        assert spec.constraint is None


class TestExperimentRequest:
    def test_construction(self):
        tv = TargetingVector(name="PSMA-617", target="PSMA", vector_type="small_molecule")
        ch = Chelator(name="DOTA")
        c = CandidateConstruct(
            name="test", targeting_vector=tv, chelator=ch, radionuclide=Radionuclide.LU177
        )
        req = ExperimentRequest(
            campaign_id="camp1",
            cycle_id="cyc1",
            constructs=[c],
            assays=["binding_affinity"],
            isotope=Radionuclide.LU177,
            quantity_gbq=1.0,
        )
        assert req.campaign_id == "camp1"
        assert len(req.constructs) == 1
        assert req.id  # auto-generated


class TestResultRecord:
    def test_construction(self):
        ar = AssayResult(
            assay_name="binding_affinity",
            construct_id="abc",
            value=0.85,
            unit="normalized_score",
        )
        rr = ResultRecord(
            request_id="req1",
            campaign_id="camp1",
            cycle_id="cyc1",
            construct_results=[ar],
        )
        assert rr.campaign_id == "camp1"
        assert len(rr.construct_results) == 1


class TestCycleResult:
    def test_construction(self):
        cr = CycleResult(
            cycle_id="abc123",
            campaign_id="camp1",
            cycle_number=1,
            constructs_proposed=10,
            constructs_scored=10,
            constructs_selected=4,
            campaign_score_before=0.5,
            campaign_score_after=0.6,
            score_delta=0.1,
        )
        assert cr.score_delta == pytest.approx(0.1)
        assert cr.strategy_kept is None


class TestLedgerEntry:
    def test_construction(self):
        entry = LedgerEntry(
            entry_type=LedgerEntryType.PROPOSAL,
            campaign_id="camp1",
            cycle_id="cyc1",
            data={"n_generated": 4},
        )
        assert entry.id
        assert entry.timestamp.tzinfo is not None
        assert entry.attribution == "autoradionuclide-reasoning-layer"


class TestModelSchemas:
    def test_token_usage(self):
        tu = TokenUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300)
        assert tu.total_tokens == 300

    def test_model_request(self):
        req = ModelRequest(model="mock-v1", system="test system")
        assert req.request_id
        assert req.temperature == 0.7

    def test_model_response(self):
        resp = ModelResponse(request_id="req1", model="mock-v1", content="hello")
        assert resp.content == "hello"
        assert resp.latency_ms == 0.0
