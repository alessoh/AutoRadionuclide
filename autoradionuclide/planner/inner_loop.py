"""Inner loop — one discovery cycle: generate → score → select → request → observe → update."""
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Optional
from autoradionuclide.domain.models import (
    CandidateConstruct, CandidateStatus, CycleResult,
    ExperimentRequest, LedgerEntry, LedgerEntryType,
    ObjectiveSpec, ResultRecord
)
from autoradionuclide.config.schema import CampaignSpec, GatingPolicy
from autoradionuclide.design.generator import CandidateGenerator
from autoradionuclide.scoring.aggregator import score_and_aggregate
from autoradionuclide.policy.acquisition import ActiveLearningPolicy
from autoradionuclide.surrogates.gp_surrogate import SurrogateBank
from autoradionuclide.safety.checks import check_batch
from autoradionuclide.interfaces.contract import WetLabInterface
from autoradionuclide.provenance.context import ProvenanceContext
from autoradionuclide.ledger.store import LedgerStore


class InnerLoop:
    """One discovery cycle.

    Follows the AutoResearch pattern for one 'training run':
    generate candidates → score → policy ranks → emit request →
    receive result → update surrogates → return cycle summary.
    """

    def __init__(
        self,
        spec: CampaignSpec,
        generator: CandidateGenerator,
        policy: ActiveLearningPolicy,
        surrogate_bank: SurrogateBank,
        wet_lab: WetLabInterface,
        ledger: LedgerStore,
        strategy_config: dict,
        run_id: str = "",
    ) -> None:
        self._spec = spec
        self._generator = generator
        self._policy = policy
        self._surrogates = surrogate_bank
        self._wet_lab = wet_lab
        self._ledger = ledger
        self._strategy = strategy_config
        self._run_id = run_id
        self._known_keys: set[str] = set()

    def run(
        self,
        cycle_number: int,
        provenance: ProvenanceContext,
        campaign_score_before: float,
        dry_run: bool = False,
    ) -> CycleResult:
        cycle_id = str(uuid.uuid4())[:8]
        started_at = datetime.now(timezone.utc)

        # 1. Generate candidates
        n_gen = self._strategy.get("n_candidates_generated", self._spec.batch_size * 3)
        candidates = self._generator.generate(
            campaign_id=self._spec.campaign_id,
            cycle_id=cycle_id,
            target=self._spec.target,
            isotope=self._spec.isotope,
            n=n_gen,
            provenance=provenance,
            known_ids=self._known_keys,
            prioritized_targets=self._strategy.get("prioritized_targets"),
            run_id=self._run_id,
            allowed_vectors=self._spec.allowed_vectors or None,
            allowed_chelators=self._spec.allowed_chelators or None,
        )

        # 2. Score through frozen harness
        scored = []
        for c in candidates:
            objs, score, feasible = score_and_aggregate(c, self._spec.objectives)
            c.status = CandidateStatus.SCORED
            scored.append((c, objs, score, feasible))
            self._ledger.append(LedgerEntry(
                entry_type=LedgerEntryType.SCORE,
                campaign_id=self._spec.campaign_id,
                run_id=self._run_id,
                cycle_id=cycle_id,
                construct_id=c.id,
                provenance_id=provenance.id,
                data={
                    "aggregate_score": score,
                    "feasible": feasible,
                    "objectives": {o.name: o.value.estimate for o in objs},
                },
            ))

        # 3. Policy selects batch
        feasible_candidates = [c for c, _, _, f in scored if f]
        batch_size = self._strategy.get("batch_size", self._spec.batch_size)
        ranked = self._policy.rank(feasible_candidates, batch_size)
        selected = [c for c, _ in ranked]
        for c in selected:
            c.status = CandidateStatus.SELECTED
        self._ledger.append(LedgerEntry(
            entry_type=LedgerEntryType.SELECTION,
            campaign_id=self._spec.campaign_id,
            run_id=self._run_id,
            cycle_id=cycle_id,
            provenance_id=provenance.id,
            data={
                "selected_ids": [c.id for c in selected],
                "acquisition_scores": [float(s) for _, s in ranked],
            },
        ))

        # 4. Safety gate
        safety = check_batch(selected, dry_run=dry_run)
        selected = [c for c in selected if safety[c.id].passed]
        if not selected:
            cycle_result = CycleResult(
                cycle_id=cycle_id,
                campaign_id=self._spec.campaign_id,
                cycle_number=cycle_number,
                constructs_proposed=len(candidates),
                constructs_scored=len(scored),
                constructs_selected=0,
                campaign_score_before=campaign_score_before,
                campaign_score_after=campaign_score_before,
                score_delta=0.0,
                rationale="No constructs passed safety checks.",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
            )
            self._ledger.append(LedgerEntry(
                entry_type=LedgerEntryType.CYCLE_SUMMARY,
                campaign_id=self._spec.campaign_id,
                run_id=self._run_id,
                cycle_id=cycle_id,
                provenance_id=provenance.id,
                data=cycle_result.model_dump(mode="json"),
            ))
            return cycle_result

        # 5. Human-in-the-loop gate
        approved = _apply_gate(
            selected, self._spec.gating_policy, self._ledger,
            self._spec.campaign_id, cycle_id, provenance.id, self._run_id
        )

        # 6. Emit experiment request
        req = ExperimentRequest(
            campaign_id=self._spec.campaign_id,
            cycle_id=cycle_id,
            constructs=approved,
            assays=list({spec.name for spec in self._spec.objectives}),
            isotope=self._spec.isotope,
            quantity_gbq=1.0,
            reasoning_id=provenance.id,
        )
        for c in approved:
            c.status = CandidateStatus.REQUESTED
        self._ledger.append(LedgerEntry(
            entry_type=LedgerEntryType.REQUEST,
            campaign_id=self._spec.campaign_id,
            run_id=self._run_id,
            cycle_id=cycle_id,
            provenance_id=provenance.id,
            data=req.model_dump(mode="json"),
        ))

        # 7. Receive result (stub returns immediately)
        result = self._wet_lab.submit_and_wait(req)
        for c in approved:
            c.status = CandidateStatus.TESTED
        self._ledger.append(LedgerEntry(
            entry_type=LedgerEntryType.RESULT,
            campaign_id=self._spec.campaign_id,
            run_id=self._run_id,
            cycle_id=cycle_id,
            provenance_id=provenance.id,
            data=result.model_dump(mode="json"),
        ))

        # 8. Update surrogates
        obj_values: dict[str, list[float]] = {}
        for ar in result.construct_results:
            obj_values.setdefault(ar.assay_name, []).append(ar.value)

        # Build per-objective aligned construct lists
        if obj_values and approved:
            constructs_for_update: list[CandidateConstruct] = []
            aligned_values: dict[str, list[float]] = {}
            for obj_name, vals in obj_values.items():
                n_vals = len(vals)
                padded = (approved * ((n_vals // len(approved)) + 1))[:n_vals]
                constructs_for_update = padded
                aligned_values[obj_name] = vals

            if constructs_for_update:
                self._surrogates.update(constructs_for_update, aligned_values)
                self._ledger.append(LedgerEntry(
                    entry_type=LedgerEntryType.SURROGATE_REFIT,
                    campaign_id=self._spec.campaign_id,
                    run_id=self._run_id,
                    cycle_id=cycle_id,
                    provenance_id=provenance.id,
                    data={"n_observations": {k: len(v) for k, v in obj_values.items()}},
                ))

        # 9. Compute new campaign score
        if result.construct_results:
            all_scores = [ar.value for ar in result.construct_results]
            campaign_score_after = float(sum(all_scores) / len(all_scores))
        else:
            campaign_score_after = campaign_score_before
        self._policy.update_best(campaign_score_after)

        # Track known constructs
        for c in approved:
            self._known_keys.add(c.composite_key)

        cycle_result = CycleResult(
            cycle_id=cycle_id,
            campaign_id=self._spec.campaign_id,
            cycle_number=cycle_number,
            constructs_proposed=len(candidates),
            constructs_scored=len(scored),
            constructs_selected=len(approved),
            campaign_score_before=campaign_score_before,
            campaign_score_after=campaign_score_after,
            score_delta=campaign_score_after - campaign_score_before,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
        )
        self._ledger.append(LedgerEntry(
            entry_type=LedgerEntryType.CYCLE_SUMMARY,
            campaign_id=self._spec.campaign_id,
            run_id=self._run_id,
            cycle_id=cycle_id,
            provenance_id=provenance.id,
            data=cycle_result.model_dump(mode="json"),
        ))
        return cycle_result


def _apply_gate(
    constructs: list[CandidateConstruct],
    policy: str,
    ledger: LedgerStore,
    campaign_id: str,
    cycle_id: str,
    provenance_id: str,
    run_id: str = "",
) -> list[CandidateConstruct]:
    if policy == GatingPolicy.MANDATORY:
        print(f"\n[GATE] Mandatory approval required for {len(constructs)} constructs.")
        answer = input("Approve? [y/N]: ").strip().lower()
        approved = constructs if answer == "y" else []
    elif policy == GatingPolicy.ADVISORY:
        print(f"\n[GATE] Advisory: {len(constructs)} constructs selected (auto-approved).")
        approved = constructs
    else:  # automatic
        approved = constructs

    ledger.append(LedgerEntry(
        entry_type=LedgerEntryType.APPROVAL,
        campaign_id=campaign_id,
        run_id=run_id,
        cycle_id=cycle_id,
        provenance_id=provenance_id,
        data={
            "policy": policy,
            "n_submitted": len(constructs),
            "n_approved": len(approved),
        },
    ))
    return approved
