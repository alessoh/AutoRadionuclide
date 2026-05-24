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
    1. Planner asks LLM to propose one strategy modification (turns > 0 only).
    2. Runs one inner cycle with the (possibly modified) strategy.
    3. Compares campaign score before and after the inner cycle.
    4. Keeps modification and new score if improved; reverts both if not.
    5. Records the decision and rationale in the ledger.

    Non-decreasing invariant: the kept campaign score never decreases across
    turns within one run. When a modification is not kept, the score reverts to
    its value at the start of that turn (not to the inner cycle's result).
    """

    def __init__(
        self,
        spec: CampaignSpec,
        inner_loop: InnerLoop,
        provider: ModelProvider,
        ledger: LedgerStore,
        base_strategy: dict,
        run_id: str = "",
    ) -> None:
        self._spec = spec
        self._inner = inner_loop
        self._provider = provider
        self._ledger = ledger
        self._strategy = copy.deepcopy(base_strategy)
        self._run_id = run_id
        self._cycle_number = 0
        self._campaign_score = 0.0
        self._score_history: list[float] = []
        self._stall_count = 0

    def run(self, dry_run: bool = False) -> list[dict]:
        """Run the full campaign until stopping criteria are met. Returns cycle summaries."""
        summaries = []
        started = datetime.now(timezone.utc)

        for turn in range(self._spec.budget.max_cycles):
            # Check stopping criteria before running this turn
            if self._should_stop():
                print(f"[Planner] Stopping at turn {turn}: {self._stop_reason()}")
                break

            wall_minutes = (datetime.now(timezone.utc) - started).total_seconds() / 60
            if wall_minutes > self._spec.budget.max_wall_minutes:
                print(f"[Planner] Wall time budget exhausted ({wall_minutes:.1f} min).")
                break

            print(f"\n[Planner] === Turn {turn + 1}/{self._spec.budget.max_cycles} "
                  f"| Campaign score: {self._campaign_score:.3f} ===")

            # Capture score at turn start (used to revert if modification not kept)
            score_before_turn = self._campaign_score

            # 1. Propose strategy modification (outer loop, turns > 0 only)
            modification = None
            old_strategy = copy.deepcopy(self._strategy)
            if turn > 0 and len(self._score_history) >= 1:
                modification = self._propose_modification()
                if modification:
                    self._apply_modification(modification)

            # 2. Run one inner cycle with current strategy
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

            # 3. Keep or revert modification (and score)
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

                # Record the keep/revert decision with full context
                self._ledger.append(LedgerEntry(
                    entry_type=LedgerEntryType.STRATEGY_MODIFICATION,
                    campaign_id=self._spec.campaign_id,
                    run_id=self._run_id,
                    cycle_id=cycle_result.cycle_id,
                    provenance_id=provenance.id,
                    data={
                        "modification": modification,
                        "score_delta": cycle_result.score_delta,
                        "kept": kept,
                        "rationale": modification.get("rationale", ""),
                        "campaign_score_before": cycle_result.campaign_score_before,
                        "campaign_score_after_raw": cycle_result.campaign_score_after,
                    },
                ))

            # Update cycle result with strategy decision
            cycle_result.strategy_modification = (
                modification.get("modification_description") if modification else None
            )
            cycle_result.strategy_kept = kept

            # 4. Commit or revert the campaign score.
            # Non-decreasing invariant: when a modification is not kept, revert the
            # campaign score to the start of the turn so that the next turn's
            # modification proposal is based on the last kept score, not a bad result.
            if modification is None or kept:
                self._campaign_score = cycle_result.campaign_score_after
            else:
                self._campaign_score = score_before_turn

            self._score_history.append(self._campaign_score)

            # 5. Update stall counter using the effective (kept) delta
            effective_delta = self._campaign_score - score_before_turn
            if abs(effective_delta) < self._spec.stopping.min_score_delta:
                self._stall_count += 1
            else:
                self._stall_count = 0

            summaries.append(cycle_result.model_dump(mode="json"))
            print(
                f"[Planner] Turn {turn + 1} complete: "
                f"score {score_before_turn:.3f} -> {self._campaign_score:.3f} "
                f"(effective delta={effective_delta:+.3f}"
                + (f", raw inner delta={cycle_result.score_delta:+.3f}" if modification else "")
                + ")"
            )

        print(f"\n[Planner] Campaign finished. "
              f"Turns run: {len(summaries)} | Final kept score: {self._campaign_score:.3f}")
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
