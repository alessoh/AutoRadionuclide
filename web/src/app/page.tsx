import runData from "@/data/run_export.json";

// ---------------------------------------------------------------------------
// Types (derived from JSON structure — no fabrication)
// ---------------------------------------------------------------------------

type Turn = (typeof runData.turns)[number];
type Ranking = (typeof runData.benchmark.rankings)[number];

// ---------------------------------------------------------------------------
// Small presentational helpers
// ---------------------------------------------------------------------------

function SectionHeading({ children }: { children: React.ReactNode }) {
  return (
    <h2 className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-3">
      {children}
    </h2>
  );
}

function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-white border border-gray-200 rounded-lg p-5 ${className}`}>
      {children}
    </div>
  );
}

function ScoreBar({ value, max = 1.0 }: { value: number; max?: number }) {
  const pct = Math.round((value / max) * 100);
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
        <div
          className="h-2 rounded-full bg-blue-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs font-mono text-gray-700 w-12 text-right">
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function QualityBadge({ quality }: { quality: string }) {
  const q = quality.toUpperCase();
  const cls =
    q === "FULL"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : q === "PARTIAL"
      ? "bg-amber-50 text-amber-700 border-amber-200"
      : "bg-red-50 text-red-700 border-red-200";
  return (
    <span
      className={`inline-block border rounded px-2 py-0.5 text-xs font-semibold ${cls}`}
    >
      {q}
    </span>
  );
}

function LabelBadge({ label }: { label: string }) {
  const cls =
    label === "approved"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : label === "clinical"
      ? "bg-blue-50 text-blue-700 border-blue-200"
      : "bg-red-50 text-red-700 border-red-200";
  return (
    <span className={`inline-block border rounded px-2 py-0.5 text-xs font-medium ${cls}`}>
      {label}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Panels
// ---------------------------------------------------------------------------

function HonestyBanner() {
  return (
    <div className="bg-amber-50 border border-amber-300 rounded-lg px-5 py-4">
      <div className="flex gap-3">
        <span className="text-amber-500 text-lg flex-shrink-0 mt-0.5">⚠</span>
        <div>
          <p className="text-sm font-semibold text-amber-800 mb-1">
            In-silico demonstration only — no wet lab, no real isotope
          </p>
          <p className="text-sm text-amber-700 leading-relaxed">
            All scoring functions are frozen heuristics, not validated predictive models.
            The wet-lab step is a stub that returns heuristic scores plus Gaussian noise.
            MIBG + I-131 (Azedra) is a real FDA-approved therapy; the scores shown here
            are illustrative engine outputs, not clinical measurements.
          </p>
        </div>
      </div>
    </div>
  );
}

function HeaderPanel() {
  const { meta, provenance } = runData;
  const runDate = new Date(meta.run_started_at).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
    timeZone: "UTC",
  });
  return (
    <div className="border-b border-gray-200 pb-6 mb-6">
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-blue-600 mb-1">
            AutoRadionuclide · In-silico Discovery Engine
          </p>
          <h1 className="text-2xl font-bold text-gray-900 mb-1">
            {meta.campaign_name}
          </h1>
          <p className="text-sm text-gray-500">
            Campaign <span className="font-mono">{meta.campaign_id}</span> &middot;{" "}
            Run <span className="font-mono">{meta.run_id}</span> &middot; {runDate}
          </p>
        </div>
        <div className="flex gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-gray-900">
              {provenance.total_ledger_entries_this_run}
            </div>
            <div className="text-xs text-gray-400 mt-0.5">Ledger entries</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-900">
              {provenance.total_model_calls_this_run}
            </div>
            <div className="text-xs text-gray-400 mt-0.5">Model calls</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-gray-900">4</div>
            <div className="text-xs text-gray-400 mt-0.5">Turns</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function LoopExplanationPanel() {
  return (
    <Card>
      <SectionHeading>How the loop works</SectionHeading>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            OuterLoop (AutoResearch meta-loop)
          </p>
          <ol className="space-y-1.5 text-sm text-gray-700">
            {[
              "Ask LLM: propose ONE strategy modification",
              "Apply modification to in-memory StrategyConfig",
              "Run InnerLoop — one discovery cycle",
              "Compare campaign score before vs. after",
              "Keep if improved (Δ > 0); revert if not",
              "Record modification + rationale in append-only ledger",
            ].map((step, i) => (
              <li key={i} className="flex gap-2">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-blue-100 text-blue-700 text-xs font-semibold flex items-center justify-center">
                  {i + 1}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </div>
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            InnerLoop (one discovery cycle)
          </p>
          <ol className="space-y-1.5 text-sm text-gray-700">
            {[
              "generate_candidates() — design module + LLM",
              "score_all() — frozen harness (never agent-editable)",
              "policy.rank() — acquisition function + diversity",
              "safety_check() — isotope/chelator feasibility",
              "wet_lab.submit_and_wait() — stub in this run",
              "update_surrogates() — GP refitted with new observations",
            ].map((step, i) => (
              <li key={i} className="flex gap-2">
                <span className="flex-shrink-0 w-5 h-5 rounded-full bg-gray-100 text-gray-600 text-xs font-semibold flex items-center justify-center">
                  {i + 1}
                </span>
                <span>{step}</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
      <p className="text-xs text-gray-400 mt-4 border-t border-gray-100 pt-3">
        Frozen harness = the benchmark spec. The planner may read it but never modify it.
        Every decision is recorded in an append-only SQLite ledger — rows are never updated or deleted.
      </p>
    </Card>
  );
}

function TurnCard({ turn, index }: { turn: Turn; index: number }) {
  const isFirst = index === 0;
  const mod = turn.strategy_modification;
  const scoreLabel = isFirst
    ? `Established ${turn.score_after.toFixed(4)}`
    : turn.score_delta === 0
    ? `Plateau — score held at ${turn.score_after.toFixed(4)}`
    : `+${turn.score_delta.toFixed(4)}`;

  return (
    <div className="relative pl-8">
      {/* Timeline dot */}
      <div className="absolute left-0 top-1 w-5 h-5 rounded-full border-2 border-blue-500 bg-white flex items-center justify-center">
        <div className={`w-2 h-2 rounded-full ${isFirst ? "bg-blue-500" : "bg-gray-300"}`} />
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-start justify-between gap-2 mb-2">
          <div>
            <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Turn {turn.turn}
            </span>
            {isFirst && (
              <span className="ml-2 inline-block bg-blue-50 text-blue-700 border border-blue-200 rounded px-2 py-0.5 text-xs">
                First cycle
              </span>
            )}
          </div>
          <span className="font-mono text-xs text-gray-400">{turn.cycle_id.slice(0, 8)}</span>
        </div>

        {/* Score progress */}
        <div className="mb-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-gray-500">Campaign score</span>
            <span className={`text-xs font-mono font-semibold ${isFirst ? "text-emerald-600" : "text-gray-500"}`}>
              {scoreLabel}
            </span>
          </div>
          <ScoreBar value={turn.score_after} />
        </div>

        {/* Constructs */}
        <div className="text-xs text-gray-500 mb-2">
          {turn.constructs_proposed === 0 ? (
            <span className="text-amber-600">{turn.inner_loop_note || "No constructs proposed"}</span>
          ) : (
            <span>
              {turn.constructs_proposed} proposed · {turn.constructs_selected} selected
            </span>
          )}
        </div>

        {/* Strategy modification */}
        {mod && (
          <div className="mt-3 border-t border-gray-100 pt-3">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="text-xs font-semibold text-gray-700">{mod.description}</p>
                <p className="text-xs text-gray-500 mt-0.5 font-mono">
                  {mod.parameter}: {JSON.stringify(mod.old_value)} → {JSON.stringify(mod.new_value)}
                </p>
                <p className="text-xs text-gray-400 mt-1 italic">{mod.rationale}</p>
              </div>
              <span className="flex-shrink-0 inline-block bg-gray-100 text-gray-500 rounded px-2 py-0.5 text-xs font-semibold">
                reverted
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function TurnTimelinePanel() {
  return (
    <Card>
      <SectionHeading>Turn-by-turn outer loop</SectionHeading>
      <div className="space-y-4 relative">
        {/* Timeline line */}
        <div className="absolute left-2 top-2 bottom-2 w-px bg-gray-200" />
        {runData.turns.map((turn, i) => (
          <TurnCard key={turn.cycle_id} turn={turn} index={i} />
        ))}
      </div>
      <p className="text-xs text-gray-400 mt-4 border-t border-gray-100 pt-3">
        Score is non-decreasing by design: the outer loop reverts any modification
        that does not improve the campaign score (Δ ≤ 0). The score plateaus here
        because MIBG+none+I-131 is the only unique resolvable construct in the
        declared building-block space — this is honest scientific behaviour.
      </p>
    </Card>
  );
}

function ConstructPanel() {
  const c = runData.construct;
  const f = c.featurization;
  const descriptorLabels: Record<string, string> = {
    mw: "Molecular weight (Da)",
    logp: "Wildman-Crippen logP",
    tpsa: "TPSA (Å²)",
    hbd: "H-bond donors",
    hba: "H-bond acceptors",
    rotbonds: "Rotatable bonds",
    rings: "Ring count",
    frac_csp3: "Fraction sp³ C",
  };
  const objectiveLabels: Record<string, string> = {
    binding_affinity: "Binding affinity",
    chelator_stability: "Chelator stability",
    half_life_compatibility: "Half-life compatibility",
    synthetic_feasibility: "Synthetic feasibility",
    selectivity: "Selectivity",
  };

  return (
    <Card>
      <SectionHeading>Candidate construct</SectionHeading>
      <div className="flex items-start gap-3 mb-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-lg font-bold text-gray-900">{c.clinical_name}</span>
            <QualityBadge quality={f.quality} />
          </div>
          <p className="text-xs text-gray-500">{c.mechanism}</p>
          <p className="text-xs text-gray-400 mt-1">
            Vector: <span className="font-mono">{c.targeting_vector}</span> &middot;
            Chelator: <span className="font-mono">{c.chelator}</span> &middot;
            Isotope: <span className="font-mono">{c.isotope}</span>
          </p>
        </div>
      </div>

      {/* SMILES + formula */}
      <div className="bg-gray-50 rounded p-3 mb-4 text-xs">
        <div className="flex gap-4 flex-wrap">
          <div>
            <span className="text-gray-400 uppercase tracking-wide text-xs">SMILES</span>
            <p className="font-mono text-gray-800 mt-0.5 break-all">{c.smiles}</p>
          </div>
          <div>
            <span className="text-gray-400 uppercase tracking-wide text-xs">Formula</span>
            <p className="font-mono text-gray-800 mt-0.5">{c.formula}</p>
          </div>
          <div>
            <span className="text-gray-400 uppercase tracking-wide text-xs">Source</span>
            <p className="font-mono text-gray-800 mt-0.5">{c.registry_source}</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* RDKit Descriptors */}
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            RDKit descriptors (GP surrogate input)
          </p>
          <table className="w-full text-xs">
            <tbody>
              {f.descriptor_names.map((name) => (
                <tr key={name} className="border-b border-gray-100 last:border-0">
                  <td className="py-1.5 text-gray-500">{descriptorLabels[name] ?? name}</td>
                  <td className="py-1.5 text-right font-mono text-gray-800">
                    {(f.descriptors as Record<string, number>)[name].toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-gray-400 mt-2">
            Morgan-2 fingerprint: {f.fingerprint_bits} active bits / {f.fingerprint_params.n_bits} total
          </p>
        </div>

        {/* Objective scores */}
        <div>
          <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
            Heuristic objective scores
          </p>
          <div className="space-y-2.5">
            {Object.entries(objectiveLabels).map(([key, label]) => (
              <div key={key}>
                <div className="flex justify-between text-xs text-gray-500 mb-0.5">
                  <span>{label}</span>
                </div>
                <ScoreBar value={(c.objectives as Record<string, number>)[key]} />
              </div>
            ))}
          </div>
          <div className="mt-3 pt-3 border-t border-gray-100">
            <div className="flex justify-between items-center">
              <span className="text-xs font-semibold text-gray-600">Aggregate</span>
              <span className="font-mono text-sm font-bold text-blue-700">
                {c.objectives.aggregate.toFixed(4)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Isotope physics */}
      <div className="mt-4 pt-4 border-t border-gray-100">
        <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">
          Isotope physics (I-131)
        </p>
        <div className="flex gap-6 text-xs flex-wrap">
          <div>
            <span className="text-gray-400">Atomic number</span>
            <p className="font-mono text-gray-800">{f.isotope_features.atomic_number} (iodine)</p>
          </div>
          <div>
            <span className="text-gray-400">Half-life</span>
            <p className="font-mono text-gray-800">{f.isotope_features.half_life_days} days</p>
          </div>
          <div>
            <span className="text-gray-400">Primary decay</span>
            <p className="font-mono text-gray-800">β⁻ (encoded: {f.isotope_features.decay_mode_encoded})</p>
          </div>
          <div>
            <span className="text-gray-400">Source</span>
            <p className="font-mono text-gray-800">IAEA Live Chart of Nuclides</p>
          </div>
        </div>
      </div>
    </Card>
  );
}

function BenchmarkPanel() {
  const { benchmark } = runData;
  const topN = Math.ceil(benchmark.rankings.length / 2);
  return (
    <Card>
      <SectionHeading>Retrospective benchmark</SectionHeading>
      <div className="flex gap-6 mb-4 flex-wrap">
        <div>
          <div className="text-2xl font-bold text-gray-900">
            {(benchmark.rank_accuracy * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400">Engine rank accuracy</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-gray-400">
            {(benchmark.baseline_accuracy * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400">Random baseline</div>
        </div>
        <div>
          <div className="text-2xl font-bold text-gray-900">{benchmark.engine_hits}/{benchmark.n_positive}</div>
          <div className="text-xs text-gray-400">Known-good in top {topN}</div>
        </div>
      </div>
      <p className="text-xs text-gray-500 mb-4 italic">{benchmark.interpretation}</p>

      {/* Rankings */}
      <div className="space-y-2">
        {benchmark.rankings.map((r: Ranking, i: number) => (
          <div key={r.id} className="flex items-center gap-3">
            <span className="text-xs font-mono text-gray-400 w-4">{i + 1}</span>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-0.5">
                <span className="text-xs text-gray-700 truncate">{r.name}</span>
                {r.id === "mibg-i131-direct" && (
                  <span className="flex-shrink-0 text-xs bg-blue-50 text-blue-600 border border-blue-200 rounded px-1.5 py-0.5">
                    this run
                  </span>
                )}
                <LabelBadge label={r.label} />
              </div>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-100 rounded-full h-1.5 overflow-hidden">
                  <div
                    className={`h-1.5 rounded-full ${
                      r.label === "failed" ? "bg-red-300" : r.label === "approved" ? "bg-emerald-400" : "bg-blue-300"
                    }`}
                    style={{ width: `${Math.round(r.score * 100)}%` }}
                  />
                </div>
                <span className="text-xs font-mono text-gray-500 w-14 text-right">
                  {r.score.toFixed(4)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-400 mt-4 border-t border-gray-100 pt-3">
        9 compounds (3 approved · 4 clinical · 2 illustrative failures). Benchmark confirms
        scoring machinery ranks known-good agents above known-poor ones — this does NOT
        establish validated predictive power for novel compounds.
      </p>
    </Card>
  );
}

function ProvenancePanel() {
  const p = runData.provenance;
  const rows = [
    ["Run ID", p.run_id],
    ["Model provider", p.model_id],
    ["Featurizer version", p.featurizer_version],
    ["MIBG SMILES source", p.smiles_source],
    ["Model calls this run", String(p.total_model_calls_this_run)],
    ["Ledger entries this run", String(p.total_ledger_entries_this_run)],
    ["Ledger entries (all runs)", String(p.total_ledger_entries_all_runs)],
  ];
  return (
    <Card>
      <SectionHeading>Provenance</SectionHeading>
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k} className="border-b border-gray-100 last:border-0">
              <td className="py-1.5 text-gray-500 pr-4">{k}</td>
              <td className="py-1.5 font-mono text-gray-800">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-400 mt-3 border-t border-gray-100 pt-3">
        Every ledger row is immutable (INSERT-only). Model ID, prompt version, scoring
        version, config hash, and random seed are recorded per decision. Export generated
        from: <span className="font-mono">{runData.meta.export_script}</span>
      </p>
    </Card>
  );
}

function HonestLimitsPanel() {
  return (
    <Card>
      <SectionHeading>Honest limits</SectionHeading>
      <ul className="space-y-2">
        {runData.honest_limits.map((limit, i) => (
          <li key={i} className="flex gap-2 text-sm text-gray-600">
            <span className="flex-shrink-0 text-red-400 font-bold mt-0.5">✗</span>
            <span>{limit}</span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-gray-400 mt-4 border-t border-gray-100 pt-3">
        Limitations are encoded in the source — every scoring function that lacks a
        validated predictive model is tagged <span className="font-mono">HEURISTIC</span> or{" "}
        <span className="font-mono">PLACEHOLDER</span> in its returned{" "}
        <span className="font-mono">ObjectiveValue</span>.
      </p>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function Home() {
  return (
    <main className="max-w-5xl mx-auto px-4 py-10 space-y-6">
      <HeaderPanel />
      <HonestyBanner />
      <LoopExplanationPanel />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <TurnTimelinePanel />
        <div className="space-y-6">
          <ConstructPanel />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <BenchmarkPanel />
        <div className="space-y-6">
          <ProvenancePanel />
          <HonestLimitsPanel />
        </div>
      </div>

      <footer className="text-center text-xs text-gray-400 pt-4 border-t border-gray-100">
        AutoRadionuclide · in-silico only · no wet lab · no paid API required ·{" "}
        <a
          href="https://github.com/alessoh"
          className="underline hover:text-gray-600"
          target="_blank"
          rel="noopener noreferrer"
        >
          GitHub
        </a>
      </footer>
    </main>
  );
}
