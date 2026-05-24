"""ar-launch: run a campaign from a YAML spec (analog of run_autoresearch.sh)."""
from __future__ import annotations
import uuid
import click
from pathlib import Path
from autoradionuclide.config.schema import CampaignSpec
from autoradionuclide.ledger.store import LedgerStore
from autoradionuclide.providers.mock import MockModelProvider
from autoradionuclide.design.generator import CandidateGenerator
from autoradionuclide.surrogates.gp_surrogate import SurrogateBank
from autoradionuclide.policy.acquisition import ActiveLearningPolicy
from autoradionuclide.interfaces.contract import WetLabInterface
from frozen.stub import StubWetLab
from autoradionuclide.planner.inner_loop import InnerLoop
from autoradionuclide.planner.outer_loop import OuterLoop
from autoradionuclide.observability.inspector import print_campaign_report
import strategy.hyperparams as hp


@click.command()
@click.argument("spec_path", default="campaigns/example_psma.yaml")
@click.option("--dry-run", is_flag=True, default=False, help="Run without persisting to the database.")
@click.option("--cycles", default=None, type=int, help="Override max_cycles from spec.")
def main(spec_path: str, dry_run: bool, cycles: int | None):
    """Launch a discovery campaign from YAML spec (analog of run_autoresearch.sh)."""
    spec = CampaignSpec.from_yaml(spec_path)
    if cycles is not None:
        spec.budget.max_cycles = cycles

    # Every launch gets a unique run_id so its ledger entries can be scoped
    # independently of all other runs that share the same campaign_id.
    run_id = str(uuid.uuid4())[:8]

    # Dry-run uses an in-memory ledger: results are never written to the
    # persistent database and the end-of-run report reflects only this run.
    if dry_run:
        ledger = LedgerStore(":memory:")
        print(f"[Launch] DRY-RUN mode — results are NOT persisted to '{spec.db_path}'.")
    else:
        ledger = LedgerStore(spec.db_path)

    provider = MockModelProvider(ledger=ledger)
    provider.set_campaign(spec.campaign_id)
    provider.set_run_id(run_id)

    surrogate_bank = SurrogateBank(
        [o.name for o in spec.objectives], seed=spec.random_seed
    )
    policy = ActiveLearningPolicy(
        surrogate_bank=surrogate_bank,
        specs=spec.objectives,
        acquisition_fn=hp.HYPERPARAMS["acquisition_function"],
        exploration_weight=hp.HYPERPARAMS["exploration_weight"],
        diversity_threshold=hp.HYPERPARAMS["diversity_threshold"],
    )
    generator = CandidateGenerator(provider=provider, ledger=ledger)
    wet_lab: WetLabInterface = StubWetLab(seed=spec.random_seed)

    strategy_config = {**hp.HYPERPARAMS}

    inner = InnerLoop(
        spec=spec,
        generator=generator,
        policy=policy,
        surrogate_bank=surrogate_bank,
        wet_lab=wet_lab,
        ledger=ledger,
        strategy_config=strategy_config,
        run_id=run_id,
    )
    outer = OuterLoop(
        spec=spec,
        inner_loop=inner,
        provider=provider,
        ledger=ledger,
        base_strategy=strategy_config,
        run_id=run_id,
    )

    print(f"[Launch] Campaign : {spec.name}")
    print(f"[Launch] Run ID   : {run_id}  (use --run-id {run_id} with ar-inspect)")
    print(f"[Launch] Target   : {spec.target} | Isotope: {spec.isotope.value}")
    print(f"[Launch] Max turns: {spec.budget.max_cycles} | Dry-run: {dry_run}")

    summaries = outer.run(dry_run=dry_run)

    # Report scoped to this run only — never contaminated by earlier runs
    print_campaign_report(ledger, spec.campaign_id, run_id=run_id)
    return summaries


if __name__ == "__main__":
    main()
