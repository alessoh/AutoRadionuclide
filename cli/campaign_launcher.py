"""ar-launch: run a campaign from a YAML spec (analog of run_autoresearch.sh)."""
from __future__ import annotations
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
@click.option("--dry-run", is_flag=True, default=False, help="Run without emitting real requests.")
@click.option("--cycles", default=None, type=int, help="Override max_cycles from spec.")
def main(spec_path: str, dry_run: bool, cycles: int | None):
    """Launch a discovery campaign from YAML spec (analog of run_autoresearch.sh)."""
    spec = CampaignSpec.from_yaml(spec_path)
    if cycles is not None:
        spec.budget.max_cycles = cycles

    ledger = LedgerStore(spec.db_path)
    provider = MockModelProvider(ledger=ledger)
    provider.set_campaign(spec.campaign_id)

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
    )
    outer = OuterLoop(
        spec=spec,
        inner_loop=inner,
        provider=provider,
        ledger=ledger,
        base_strategy=strategy_config,
    )

    print(f"[Launch] Campaign: {spec.name}")
    print(f"[Launch] Target: {spec.target} | Isotope: {spec.isotope.value}")
    print(f"[Launch] Max cycles: {spec.budget.max_cycles} | Dry-run: {dry_run}")

    summaries = outer.run(dry_run=dry_run)
    print_campaign_report(ledger, spec.campaign_id)
    return summaries


if __name__ == "__main__":
    main()
