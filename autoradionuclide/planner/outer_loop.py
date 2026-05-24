"""Outer loop — AutoResearch meta-loop: propose strategy mod → test → keep or revert."""
from __future__ import annotations
import copy, json, uuid
from datetime import datetime, timezone
from typing import Optional
from autoradionuclide.domain.models import LedgerEntry, LedgerEntryType, ModelRequest
from autoradionuclide.config.schema import CampaignSpec
from autoradionuclide.planner.inner_loop import InnerLoop
from autoradionuclide.providers.base import ModelProvider, PROMPT_TEMPLATES
from autoradionuclide.provenance.context import ProvenanceContext
from autoradionuclide.ledger.store import LedgerStore


class OuterLoop:
    """Implements the AutoResearch outer meta-loop.

    Each turn:
    1. Planner asks LLM to propose one strategy modification.
    2. Runs one inner cycle with the modified strategy.
    3. Compares campaign objective before and after.
    4. Keeps modification if improved, reverts if not.
    5. Records the decision and rationale in the ledger.
    """

    def __init__(
        self,
        spec: CampaignSpec,
        inner_loop: InnerLoop,
        provider: ModelProvider,
        ledger: LedgerStore,
        base_strategy: dict,
    ) -> None:
        self._spec = spec
        self._inner = inner_loop
        self._provider = provider
        self._ledger = ledger
        self._strategy = copy.deepcopy(base_strategy)
        self._cycle_number = 0
        self._campaign_score = 0.0
        self._score_history: list[float] = []
        self._stall_count = 0

    def run(self, dry_run: bool = False) -> list[dict]:
        """Run the full campaign until stopping criteria are met. Returns cycle summaries."""
        summaries = []
        started = datetime.now(timezone.utc)

        for turn in range(self._spec.budget.max_cycles):
            # Check stopping criteria
            if self._should_stop():
                print(f"[Planner] Stopping at turn {turn}: {self._stop_reason()}")
                break

            wall_minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60
            if wall_minutes > self._spec.budget.max_wall_minutes:
                print(f"[Planner] Wall time budget exhausted ({wall_minutes:.1f} min).")
                break

            print(f"\n[Planner] === Turn {turn + 1}/{self._spec.budget.max_cycles} "
                  f"| Campaign score: {self._campaign_score:.3f} ===")

            # 1. Propose strategy modification (outer loop)
            modification = None
            old_strategy = copy.deepcopy(self._strategy)
            if turn > 0 and len(self._score_history) >= 1:
                modification = self._propose_modification()
                if modification:
                    self._apply_modification(modification)

            # 2. Run inner cycle with current strategy
            provenance = ProvenanceContext.from_config(
                model_id=self._spec.model_id,
                config_dict=self._spec.model_dump(mode="json"),
                seed=self._spec.random_seed + turn,
            )
            self._inner._strategy = self._strategy

            cycle_result = self._inner.run(
                cycle_number=self._cycle_number,
                provenance=provenance,
                campaign_score_before=self._campaign_score,
                dry_run=dry_run,
            )
            self._cycle_number += 1

            # 3. Keep or revert modification
            kept = None
            if modification is not None:
                kept = cycle_result.score_delta > 0
                if not kept:
                    self._strategy = old_strategy
                    print(
                        f"[Planner] Strategy modification "
                        f"'{modification.get('modification_description', '')}' "
                        f"reverted (delta={cycle_result.score_delta:+.3f})"
                    )
                else:
                    print(
                        f"[Planner] Strategy modification KEPT "
                        f"(delta={cycle_result.score_delta:+.3f})"
                    )

                # Record the keep/revert decision
                self._ledger.append(LedgerEntry(
                    entry_type=LedgerEntryType.STRATEGY_MODIFICATION,
                    campaign_id=self._spec.campaign_id,
                    cycle_id=cycle_result.cycle_id,
                    provenance_id=provenance.id,
                    data={
                        "modification": modification,
                        "score_delta": cycle_result.score_delta,
                        "kept": kept,
                        "rationale": modification.get("rationale", ""),
                    },
                ))

            # Update cycle result with strategy decision
            cycle_result.strategy_modification = (
                modification.get("modification_description") if modification else None
            )
            cycle_result.strategy_kept = kept

            self._campaign_score = cycle_result.campaign_score_after
            self._score_history.append(self._campaign_score)
            if abs(cycle_result.score_delta) < self._spec.stopping.min_score_delta:
                self._stall_count += 1
            else:
                self._stall_count = 0

            summaries.append(cycle_result.model_dump(mode="json"))
            print(
                f"[Planner] Cycle {self._cycle_number}: "
                f"score {cycle_result.campaign_score_before:.3f} -> "
                f"{cycle_result.campaign_score_after:.3f} "
                f"(delta={cycle_result.score_delta:+.3f})"
            )

        print(f"\n[Planner] Campaign finished. Final score: {self._campaign_score:.3f}")
        return summaries

    def _propose_modification(self) -> Optional[dict]:
        recent_deltas = self._score_history[-3:] if self._score_history else []
        system = PROMPT_TEMPLATES["strategy_modification"].format(
            target=self._spec.target,
            score=self._campaign_score,
            deltas=recent_deltas,
            params=json.dumps(self._strategy, default=str),
        )
        request = ModelRequest(
            model=self._spec.model_id,
            system=system,
            messages=[{"role": "user", "content": "Propose one strategy modification."}],
            temperature=0.7,
            max_tokens=512,
            response_format="json_object",
        )
        response = self._provider.complete(request)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return None

    def _apply_modification(self, mod: dict) -> None:
        param = mod.get("parameter_name")
        new_val = mod.get("new_value")
        if param and new_val is not None:
            self._strategy[param] = new_val

    def _should_stop(self) -> bool:
        if self._campaign_score >= self._spec.stopping.target_campaign_score:
            return True
        if self._stall_count >= self._spec.stopping.stall_patience:
            return True
        return False

    def _stop_reason(self) -> str:
        if self._campaign_score >= self._spec.stopping.target_campaign_score:
            return f"Target score {self._spec.stopping.target_campaign_score} reached."
        return f"Progress stalled for {self._stall_count} consecutive cycles."
