# AutoRadionuclide — Reasoning Layer

AI-native discovery engine for radioligand cancer therapies.
**In-silico closed loop only** — no wet lab, no real isotope, no paid API required.

---

## The AutoResearch Mapping

This system implements Andrej Karpathy's AutoResearch autonomous-experiment loop,
adapted from optimizing a neural network against a single loss to optimizing molecules
against several objectives under cost, scarcity, and safety constraints.

| AutoResearch (Music) | AutoRadionuclide | Role |
|---|---|---|
| `program.md` | `CampaignSpec` / `campaigns/*.yaml` | Standing operating instructions and goals |
| Autonomous agent | `OuterLoop` + `InnerLoop` | Never stops until criterion met |
| `prepare.py` (frozen) | `frozen/` directory | Benchmark dataset, scoring definitions, stub — NOT editable |
| `train.py` (editable) | `strategy/` directory | Generation params, weights, hyperparams — agent editable |
| One timed training run | One discovery cycle (`InnerLoop.run()`) | The unit of experiment |
| Validation metric | Multi-objective campaign score | What gets kept or discarded |
| Keep-or-discard | `OuterLoop`: keep if delta > 0, revert if not | Learning from each turn |
| `results.tsv` | SQLite ledger (`LedgerStore`) | Append-only record of every decision |
| `check_results.py` | `ar-inspect` CLI | Inspect any campaign from its ledger |
| `run_autoresearch.sh` | `ar-launch` CLI | Launch a campaign |
| Per-run time budget | Per-cycle budget (compute, cost, wall-clock) | Resource constraint |
| Per-run git branch | Campaign ID + provenance context | Reproducibility unit |

### Two Nested Loops

```
OuterLoop (AutoResearch meta-loop)
  for each turn until stopping criteria:
    1. Ask LLM: propose ONE strategy modification
    2. Apply modification to in-memory StrategyConfig
    3. InnerLoop.run()  <-- one discovery cycle
    4. Compare campaign score before/after
    5. Keep if improved; revert if not
    6. Record modification + rationale + outcome in ledger

    InnerLoop (one discovery cycle)
      generate_candidates()      <- design module + LLM
      score_all()                <- frozen harness (NEVER modified by agent)
      policy.rank()              <- acquisition function + diversity
      safety_check()             <- isotope/chelator feasibility
      human_gate()               <- configurable: automatic/advisory/mandatory
      emit ExperimentRequest     <- the outward contract
      wet_lab.submit_and_wait()  <- stub (or real lab via WetLabInterface)
      update_surrogates()        <- GP refitted with new observations
      record CycleResult         <- ledger entry
```

---

## Running Offline (No API Key, No Cost)

Everything runs with the deterministic mock provider by default.

### Install

```bash
pip install -e .
# Optional: real Claude API
pip install -e ".[anthropic]"
```

### Run the retrospective benchmark

```bash
python -m cli.benchmark_runner --verbose
# or
ar-bench --verbose
```

Expected output: engine rank accuracy ~0.57, baseline ~0.44.

### Launch a campaign

```bash
python -m cli.campaign_launcher campaigns/example_psma.yaml --cycles 5
# or
ar-launch campaigns/example_psma.yaml --cycles 5 --dry-run
```

### Inspect a campaign ledger

```bash
python -m cli.inspect_results psma_campaign.db
ar-inspect psma_campaign.db --list-campaigns
ar-inspect psma_campaign.db --cycle-id <cycle_id>
```

### Run the full test suite

```bash
pytest tests/ -v
```

---

## Frozen vs. Editable Boundary

```
frozen/          # FROZEN — NOT AGENT-EDITABLE
  harness.py     # scoring function definitions (the benchmark spec)
  benchmark.json # ground-truth labels for known radioligands
  stub.py        # computational experiment stub
  benchmark_runner.py  # ranking evaluation

strategy/        # AGENT-EDITABLE (like train.py in AutoResearch)
  weights.py     # objective weights
  hyperparams.py # UCB kappa, batch size, acquisition function, etc.
  generation.py  # generation mode, n_proposals, etc.
```

**The planner may read `frozen/` but must never modify it.**
The outer loop's strategy modifications are applied only to the in-memory `StrategyConfig`
derived from `strategy/`. The frozen harness is the ground truth; the planner cannot improve
its measured score by altering the benchmark or scoring definitions.

---

## Placeholders — Honest Limits

Every scoring function that lacks a validated predictive model is:
1. Tagged `ProvenanceTag.HEURISTIC` or `ProvenanceTag.PLACEHOLDER` in the returned `ObjectiveValue`
2. Documented with `PLACEHOLDER` in its docstring
3. Listed here

| Function | Location | Type | Limitation |
|---|---|---|---|
| `score_binding_affinity` | `frozen/harness.py` | HEURISTIC | Lookup table of target validation scores; not a trained affinity model |
| `score_chelator_stability` | `frozen/harness.py` | HEURISTIC | Expert-encoded compatibility table; no DFT or thermodynamic calculation |
| `score_synthetic_feasibility` | `frozen/harness.py` | HEURISTIC | Vector-type lookup; not SAScore or RetroStar |
| `score_selectivity` | `frozen/harness.py` | HEURISTIC | Target + chelator lookup; no proteome-wide off-target model |
| `score_half_life_compatibility` | `frozen/harness.py` | PHYSICS | Uses factual IAEA half-life values; therapy suitability formula is heuristic |
| GP surrogate predictions | `autoradionuclide/surrogates/gp_surrogate.py` | LEARNED | Fitted on stub-simulated data; not real biodistribution measurements |
| `StubWetLab` results | `frozen/stub.py` | PLACEHOLDER | Returns frozen-harness scores + Gaussian noise; no real radiochemistry |
| Benchmark numeric labels | `frozen/benchmark.json` | ILLUSTRATIVE | Qualitative (approved/clinical/failed) only; no real IC50/Ki values |

### What the benchmark does and does NOT establish

The benchmark ranks 9 known radioligands (3 approved, 4 clinical, 2 illustrative failures)
by their aggregate heuristic score. Engine rank accuracy 0.57 vs. random baseline 0.44.

**This establishes**: the scoring and ranking machinery is wired correctly and behaves
sensibly on known cases — approved/clinical agents rank above the illustrative failed ones.

**This does NOT establish**: validated predictive power for novel compounds, accurate
binding affinity prediction, or any claim about real radiochemical yields.

---

## Connecting a Real Wet Lab

The reasoning engine emits `ExperimentRequest` objects and receives `ResultRecord` objects.
The only thing to implement is `WetLabInterface`:

```python
# autoradionuclide/interfaces/contract.py
class WetLabInterface(ABC):
    def submit(self, request: ExperimentRequest) -> str: ...
    def poll(self, job_id: str) -> ResultRecord | None: ...
    def submit_and_wait(self, request: ExperimentRequest) -> ResultRecord: ...
```

Replace `StubWetLab` in `cli/campaign_launcher.py` with your implementation:

```python
from my_facility import LIMSAdapter
wet_lab = LIMSAdapter(api_key=..., facility_id=...)
```

Nothing in the reasoning engine changes. The `ExperimentRequest` schema is the contract.

## Connecting a Real Model Provider

Replace `MockModelProvider` with `AnthropicProvider` in the campaign launcher:

```python
from autoradionuclide.providers.anthropic_adapter import AnthropicProvider
provider = AnthropicProvider(model="claude-sonnet-4-6", ledger=ledger)
```

Or implement `ModelProvider` for any other backend — every LLM call passes through
`autoradionuclide/providers/base.py:ModelProvider` and nothing else imports vendor SDKs.

---

## Package Architecture

```
autoradionuclide/
  domain/       Core typed schemas (Pydantic v2). No internal imports.
  ledger/       Append-only SQLite. ALCOA-plus design.
  provenance/   Pins model_id, prompt_version, scoring_version, seed per decision.
  providers/    ModelProvider ABC + MockProvider + AnthropicAdapter.
  config/       CampaignSpec (YAML) — the analog of program.md.
  interfaces/   WetLabInterface ABC + ExperimentRequest/ResultRecord contract.
  scoring/      Thin wrappers calling frozen harness scoring functions.
  design/       LLM-based candidate generation with deduplication.
  surrogates/   One sklearn GP per objective, refitted on each cycle's results.
  policy/       UCB/EI/Thompson acquisition + greedy diversity batch selection.
  safety/       Isotope half-life feasibility, chelator compatibility, alpha flags.
  planner/      InnerLoop (one cycle) + OuterLoop (AutoResearch meta-loop).
  observability/Campaign inspection and reporting (analog of check_results.py).
```

---

## Reproducing a Past Campaign

Every campaign can be replayed because the ledger records:
- `model_id` and full request/response for every LLM call
- `scoring_version`, `surrogate_version`, `prompt_template_version`
- `config_hash` of the campaign spec
- `random_seed` used for all stochastic operations

```bash
ar-inspect psma_campaign.db --campaign-id psma-lu177-example-001
```
