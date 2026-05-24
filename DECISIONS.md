# Design Decisions

Non-obvious choices made during implementation.

---

## D1: Strategy modifications are in-memory, not file-writes

The outer loop applies strategy modifications to an in-memory `dict` (derived from
`strategy/hyperparams.py`), not by writing to `strategy/*.py` files. This keeps the
filesystem representation stable and makes revert trivially cheap (copy the old dict
back). A future enhancement could checkpoint winning strategies to files.

## D2: Benchmark uses rank accuracy, not NDCG

The benchmark evaluates what fraction of approved+clinical compounds appear in the top
half of the engine's ranking. This is simpler to explain to a non-ML audience and
matches the question regulators care about: "does the engine focus attention on
compounds that turned out to be good?" Spearman rho is also reported as a secondary
metric.

## D3: Featurization uses real RDKit descriptors + Morgan fingerprints, with honest fallback

The featurization package (`autoradionuclide/featurization/`) computes two distinct
representations: (a) 8 RDKit physicochemical descriptors for GP surrogate regression
and (b) a 2048-bit Morgan-2 fingerprint for Tanimoto-based diversity selection.
The descriptor set is intentionally small (8 features) because Gaussian-process
surrogates fit on very few observations and a high-dimensional representation would
overfit. Structure resolution uses a three-priority scheme: full construct SMILES if
provided, then part-level SMILES from Chelator/TargetingVector objects, then a
small verified registry (DOTA, NOTA, DOTAGA). When no structure resolves, the record
is flagged FALLBACK and its vectors are explicit zeros — no values are fabricated.
FALLBACK records are excluded from the GP fit; PARTIAL records (some parts resolved)
are included. The radionuclide is represented by factual physics features (atomic
number, half-life, decay-mode encoding), not by attempting to model the coordination
complex with RDKit.

## D10: DOTA and NOTA have identical Morgan-2 fingerprints — documented, not fixed

DOTA (12-membered ring) and NOTA (9-membered ring) produce identical 2048-bit
Morgan fingerprints at radius 2. All atom environments visible at radius 2 are shared:
every nitrogen atom sees two CH₂ neighbors in the ring and one CH₂COOH arm.
Ring-size differences only become visible at radius ≥ 3 (though still identical even
there, due to the repeating N-CH₂-CH₂-N pattern). This is a documented property of
Morgan fingerprints for macrocyclic compounds. The consequence for diversity selection
is that constructs differing only in DOTA vs. NOTA are treated as structurally
identical, which is the scientifically defensible outcome — they share the same
local chemical environment. The test suite documents this explicitly. DOTAGA is
fingerprint-distinct from both because its glutaric arm creates unique atom
environments.

## D11: Descriptors-for-regression versus fingerprint-for-diversity are kept separate

The 8-feature descriptor vector and the 2048-bit fingerprint serve different purposes
and are kept as separate fields on FeatureRecord rather than being combined. The
descriptor vector is what the GP scaler and kernel operate on; the fingerprint is what
Tanimoto distance operates on. Mixing them (e.g. concatenating descriptor values with
fingerprint bits) would produce an incoherent distance metric where MW and logP values
arbitrarily compete with bit-counts. The two representations are computed once per
featurize() call and retrieved by their respective consumers (surrogates → descriptors;
policy → fingerprint).

## D12: FALLBACK records excluded from GP fit, not zero-padded

When no organic structure resolves, the featurizer returns an explicit zero vector
rather than fabricating descriptor values. In the surrogate, FALLBACK records are
silently excluded from the training set — not zero-padded — because a zero descriptor
vector does not represent a chemically meaningful point in descriptor space and would
corrupt the GP's hyperparameter optimisation. The surrogate falls back to the heuristic
prior (frozen harness) for FALLBACK predictions, maintaining the same behavior as
before any GP fitting occurs. This choice is documented in gp_surrogate.py.

## D4: The Ga-68 "failed therapy" benchmark compound is intentional

`unknown2-dota-ga68-therapy` has a 68-minute half-life, which scores near 0 on the
therapy half-life compatibility metric. This creates a clear, physics-grounded failure
mode without fabricating chemistry. It confirms the frozen harness correctly penalizes
isotope choices that are incompatible with multi-hour radiopharmacy workflows.

## D5: SQLite without WAL mode, using per-thread connections

The ledger uses `check_same_thread=False` with per-thread connection objects stored in
`threading.local()`. This is safe because each thread always uses its own connection
and SQLite's default journal mode handles the read/write mix in the test suite. WAL
mode would be the right choice for a production multi-process deployment but adds
complexity here.

## D6: Mock provider dispatches on system prompt keywords, not message structure

The mock provider inspects the lowercased system prompt for keywords
("candidate"/"propose" vs. "modification"/"strategy") rather than parsing the full
message. This keeps the mock simple and ensures it stays stable if prompt wording
evolves slightly. The hash-based seed ensures the same prompt always returns the same
response, satisfying the replay requirement.

## D7: Scoring functions are thin wrappers in the main package, definitions are frozen

`autoradionuclide/scoring/objectives.py` calls `frozen.harness.score_all()`.
The scoring *definitions* (what the numbers mean and how they are computed) live in
`frozen/harness.py`, which is frozen. The thin wrappers in the main package exist
only to convert the raw `ObjectiveValue` dict into the typed `ScoredObjective` list
expected by the aggregator. This preserves the frozen-harness invariant: the planner
cannot alter the meaning of any score.

## D8: `ExperimentRequest` includes full `CandidateConstruct` objects, not just IDs

This ensures the result can always be matched to the exact construct that was
requested, including its generation reasoning and provenance ID, without a separate
database lookup. The ledger entry for a REQUEST therefore contains a full snapshot
of the constructs at the moment of submission.

## D9: Half-life scoring uses `min(1.0, half_life_days / 7.0)` for therapy

The 7-day divisor was chosen so that Lu-177 (6.65 d) and I-131 (8.02 d) score near
1.0, and Y-90 (2.67 d) scores ~0.38. The "sweet spot" for PRRT/RLT is roughly
4–14 days: long enough to allow biodistribution but short enough to limit unnecessary
dose. This formula is labeled PHYSICS in the source tag but the 7-day threshold
is itself a heuristic judgment; see `frozen/harness.py` for the explicit caveat.

## D13: Registry stores the standalone building-block moiety, not the conjugate

Each entry in `autoradionuclide/featurization/registry.py` stores the SMILES for
the standalone building block — the chelator or targeting vector alone — without the
covalent linker to the other part. When both parts resolve, the featurizer combines
them with the SMILES "." (disconnected fragment) notation. This avoids double-counting
the chelator's contribution when both chelator and targeting vector are present, and
keeps each registry entry verifiable against a single cited structure. The consequence
is that the combined descriptor vector represents the chelator plus the targeting vector
as separate fragments in the same molecular graph, which is an approximation: the
covalent bond between parts is not modeled. The quality flag (PARTIAL when only some
parts resolved, FULL when all resolved) records this incompleteness in every FeatureRecord.

MIBG is a special case: the chelator is "none" (direct iodination, no separate chelator
moiety), so the MIBG targeting-vector SMILES IS the complete organic molecule and the
disconnected-fragment approximation does not apply.

## D14: Per-building-block warning deduplication prevents log spam

When the registry cannot resolve a building-block name, a UserWarning is emitted.
In a campaign with many constructs sharing the same unresolved targeting vector (e.g.
"PSMA-617" not yet in the registry), the old per-construct warning fired once per
construct — potentially hundreds of times, drowning out other diagnostics. The new
system fires once per unique `"kind:name"` key per Python session. A module-level set
(`_warned_registry_misses`) tracks which keys have already warned. Tests that assert
on warning counts must call `reset_registry_warning_state()` at the start to clear this
state, since a name seen in a prior test would otherwise suppress the warning in the
test under examination.
