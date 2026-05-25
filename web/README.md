# AutoRadionuclide Web Dashboard

Read-only dashboard visualizing the MIBG flagship run of the AutoRadionuclide
in-silico radioligand discovery engine.

**No backend. No secrets. No API calls from the browser.** All data is baked
into a static JSON file at build time.

## Regenerate the JSON export

Run this from the project root whenever you want to refresh the dashboard data
from the ledger (e.g. after a new campaign run):

```bash
python scripts/export_run.py
# or with custom options:
python scripts/export_run.py --db mibg_demo.db --run-id 16140108 --out web/src/data/run_export.json
```

Requires the Python package to be installed (`pip install -e .` from the repo root).

## Run locally

```bash
cd web
npm install
npm run dev       # http://localhost:3000
```

## Build static export

```bash
cd web
npm run build     # produces web/out/
```

## Deploy to Vercel

1. Push the repo to GitHub.
2. Import the repo in Vercel.
3. Set the **Root Directory** to `web` in Vercel's project settings.
4. No environment variables needed.
5. Deploy.

Vercel will run `npm run build` inside `web/`, which produces a fully static
site (`output: "export"` in `next.config.js`). The JSON at
`web/src/data/run_export.json` is committed to the repo and served as static
data — no database connection, no secrets.

## What the dashboard shows

| Panel | Content |
|---|---|
| Header | Campaign ID, run ID, run date, ledger entry counts |
| Honesty banner | Non-dismissible warning: in-silico only, stub wet lab |
| Loop explanation | OuterLoop + InnerLoop step-by-step |
| Turn timeline | 4 turns with score progression and strategy modifications |
| Construct | MIBG SMILES, RDKit descriptors, objective scores, isotope physics |
| Benchmark | 9-compound ranking, rank accuracy 0.57 vs. baseline 0.44 |
| Provenance | Model ID, featurizer version, ledger entry counts |
| Honest limits | What the engine does NOT model |
