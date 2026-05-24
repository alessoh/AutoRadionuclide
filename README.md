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

RDKit is a required dependency (not optional). It is included in the standard
`pip install -e .` command above. On a conda environment it can alternatively
be installed from conda-forge:

```bash
conda install -c conda-forge rdkit
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

## Featurization

The engine now uses genuine molecular featurization rather than one-hot categorical
encoding. Featurization is fixed, deterministic, versioned infrastructure — it is
not an agent-editable search-strategy knob.

### What the features represent

**Organic portion (chelator + targeting vector):**
When a chemical structure (SMILES) is available for the organic parts of a construct,
the engine computes two distinct representations:

- **Descriptor vector** (8 features): molecular weight, calculated logP, topological
  polar surface area (TPSA), hydrogen-bond donor and acceptor counts, rotatable-bond
  count, ring count, and fraction of sp³ carbons. These are computed with RDKit and
  used by the Gaussian-process surrogates for regression. The set is intentionally
  small — GPs fit on very few observations, and a high-dimensional representation
  would overfit.

- **Morgan fingerprint** (2048 bits, radius 2): a binary substructure fingerprint used
  for Tanimoto-based diversity selection in the policy. Two proposed constructs are
  considered structurally similar if their Tanimoto distance is below the diversity
  threshold.

**Radionuclide:**
Represented separately by three factual physics features: atomic number (from RDKit's
periodic table), half-life in days (from the project's single `HALF_LIFE_DAYS` source),
and primary decay mode encoded as an integer (0 = β⁻, 1 = α, 2 = EC/β⁺), sourced
from the IAEA Live Chart of Nuclides.

**Structure resolution:** structures are resolved from (a) a SMILES string provided
directly on the construct or its building blocks, or (b) the building-block registry
(`autoradionuclide/featurization/registry.py`) which currently holds verified SMILES
for DOTA (PubChem CID 129730), NOTA (PubChem CID 5460477), DOTAGA
(Simecek et al. EJNMMI Res 2012), and MIBG / iobenguane (PubChem CID 60860).
When no structure can be resolved, the feature record is flagged `FALLBACK` and its
descriptor vector and fingerprint are explicit zeros — the system does not fabricate values.

**Registry convention:** each entry stores the *standalone* building-block moiety
without the covalent linker to other parts. When both chelator and targeting vector
resolve, the featurizer combines them via the SMILES "." (disconnected fragment)
notation — an approximation that captures the parts' physicochemical contributions
but does not model the covalent bond between them. The quality flag (`PARTIAL` /
`FULL`) on every `FeatureRecord` documents which parts were resolved.

**Deliberate FALLBACK entries** (known building blocks not yet in the registry, with reason):

| Building block | Reason not included |
|---|---|
| DOTATATE (Tyr³-octreotate) | Large octapeptide; standalone SMILES needs expert verification |
| DOTATOC (Tyr³-octreotide) | Large octapeptide; standalone SMILES needs expert verification |
| PSMA-617 targeting vector | Bifunctional urea pharmacophore; standalone fragment needs verification |
| FAPI-46 | Small-molecule FAP inhibitor; standalone SMILES needs verification |
| FAPI-74 | Small-molecule FAP inhibitor; standalone SMILES needs verification |
| PSMA-I&T targeting vector | Bifunctional conjugate; standalone fragment needs verification |

### What the features do NOT represent

**Metal coordination chemistry is not modeled.** The metal-organic bond between
the radionuclide and the chelator — its geometry, thermodynamic stability, kinetic
inertness, and transmetallation susceptibility — is not represented by these 2D
organic-molecule features.

**Radiation effects are not captured.** The energy and type of emitted particles
(β⁻, α, γ, Auger electrons), the linear energy transfer (LET), the dose profile in
tissue, and the capacity of high-LET particles to cause double-strand DNA breaks are
not encoded in any descriptor or fingerprint.

**Large-peptide 3D conformation is not represented.** Standard 2D physicochemical
descriptors and Morgan fingerprints were designed for drug-like small molecules.
They do not capture backbone geometry, secondary structure, or the spatial arrangement
of a large peptide targeting vector such as DOTATATE or PSMA-I&T.

**A note on macrocyclic chelators and Morgan fingerprints:** DOTA and NOTA produce
identical binary Morgan fingerprints at radius 2. Both macrocycles share all atom
environments visible at that radius (N-CH₂-COOH in a macrocyclic context). Ring-size
differences between DOTA (12-membered) and NOTA (9-membered) require a higher radius
to distinguish. This is a known property of Morgan fingerprints for macrocyclic
compounds and is documented in the test suite.

### What the benchmark measures after this change

The retrospective benchmark is scored entirely by the frozen heuristic scoring
functions (`frozen/harness.py`), which do not consume these features. The benchmark
number is **expected to be essentially unchanged** — and it is (0.571 vs. baseline
0.444, same as before featurization). The gain from this change is a more honest and
capable internal representation for the surrogates and diversity selection, not a
higher benchmark score.

---

## Placeholders — Honest Limits

Every scoring function that lacks a validated predictive model is:
1. Tagged `ProvenanceTag.HEURISTIC` or `ProvenanceTag.PLACEHOLDER` in the returned `ObjectiveValue`
2. Documented with `PLACEHOLDER` in its docstring
3. Listed here

| Function / Component | Location | Type | Limitation |
|---|---|---|---|
| `score_binding_affinity` | `frozen/harness.py` | HEURISTIC | Lookup table of target validation scores; not a trained affinity model |
| `score_chelator_stability` | `frozen/harness.py` | HEURISTIC | Expert-encoded compatibility table; no DFT or thermodynamic calculation |
| `score_synthetic_feasibility` | `frozen/harness.py` | HEURISTIC | Vector-type lookup; not SAScore or RetroStar |
| `score_selectivity` | `frozen/harness.py` | HEURISTIC | Target + chelator lookup; no proteome-wide off-target model |
| `score_half_life_compatibility` | `frozen/harness.py` | PHYSICS | Uses factual IAEA half-life values; therapy suitability formula is heuristic |
| GP surrogate predictions | `autoradionuclide/surrogates/gp_surrogate.py` | LEARNED | Fitted on RDKit descriptors from stub-simulated data; not real biodistribution |
| `StubWetLab` results | `frozen/stub.py` | PLACEHOLDER | Returns frozen-harness scores + Gaussian noise; no real radiochemistry |
| Benchmark numeric labels | `frozen/benchmark.json` | ILLUSTRATIVE | Qualitative (approved/clinical/failed) only; no real IC50/Ki values |
| Building-block SMILES registry | `autoradionuclide/featurization/registry.py` | REFERENCE | Three chelators (DOTA, NOTA, DOTAGA) and one targeting vector (MIBG) from PubChem / literature; large peptide vectors (DOTATATE, DOTATOC, PSMA-617, FAPI-46/74) omitted pending independent verification |
| Isotope decay-mode data | `autoradionuclide/featurization/isotope_data.py` | REFERENCE | Primary decay modes from IAEA Live Chart of Nuclides; Bi-213 encoded as β⁻ (its direct decay) even though its α-emitting Po-213 daughter drives therapy |
| Organic feature descriptors | `autoradionuclide/featurization/featurizer.py` | COMPUTED | Standard 2D RDKit descriptors; metal coordination, radiation effects, and 3D conformation NOT modeled |
| Morgan fingerprint diversity | `autoradionuclide/policy/acquisition.py` | COMPUTED | Tanimoto distance over 2048-bit Morgan-2 fingerprints; DOTA and NOTA have identical fingerprints at this radius (macrocycle ring-size invisible to radius-2 Morgan) |

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
