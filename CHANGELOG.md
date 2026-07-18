# Changelog

All notable changes to this product. Format: `X.XX.XXX` (display), see `fisuralab.__version__`. Keep `0.x`
while on mock/synthetic data. Tag every release.

## [0.03.000], 2026-07-18

### Added
- The real CONTRACT 1 for the image domain (`io/image_contract.py`): image + optional binary mask +
  optional mm-per-px scale + material/source/license metadata, with hard-reject rules, soft flags
  (near-constant image, suspicious mask coverage, tiny masks) and the redistribution boundary
  (`is_redistributable`). Numpy-only core so the browser live lane reuses the exact validation.
- Standard-format IO (`io/image_formats.py`): PNG/JPG readers, mask IO, float conversion, and the
  committed-examples manifest loader.
- Curated committed example set (`data/examples/`): 4 Bridge Crack Library patches with pixel masks
  (CC0; concrete, steel, and an uncracked control) and 2 SDNET2018 patches (CC BY 4.0; cracked and
  uncracked), each attributed in `manifest.json`; CI validates every example through CONTRACT 1.
- `scripts/fetch-data.ps1` + `.sh`: idempotent acquisition of every direct-download dataset the lab
  uses (Dataverse, S3, Zenodo, Mendeley API, GitHub, Kaggle mirrors, Drive), rooted at
  `FISURA_DATA_ROOT`; gated sets documented as manual steps.
- `data/README.md` rewritten as the dataset registry: license, retrieval and redistribution ruling
  per source; the bring-your-own-data guide updated to the image contract.
- Tests: contract validation paths + committed examples through the gate (7 new tests).

## [0.02.000], 2026-07-18

### Added
- Base frontend shell wired on the shared design system (`@fasl-work/caos-app-shell` 0.3): six pages
  (App, Introduction, Methodology, Implementation, Experiments, Benchmark), EN/ES i18n, light/dark
  theming, per-panel error boundary, Pages SPA 404 redirect shim, display version derived from the
  manifest.
- Real page content transcribed from the verified research dossiers: the seven method tracks, the
  16-case matrix with honest planned statuses, KaTeX method equations (inscribed-circle width,
  Paris-Erdogan, ZNCC), the published-anchor benchmark tables with per-row primary citations, and
  the evaluation-protocol duality (2 px vs 5 px tolerance) documented up front.
- Citation spine (data/citations.ts) with verified DOIs and arXiv identifiers only.
- Screenshot verification: all six pages captured light + dark (plus ES sanity), zero console errors.

### Removed
- SIR frontend residue (App.tsx, SIRChart, Pyodide stub wired to the example engine); the SIR
  pipeline engine itself remains until the classical-ladder unit replaces it.

## [0.01.000], 2026-07-18

### Added
- Initial instantiation from the CAOS product-repo template (ADR-0057).
- Offline `data-pipeline/` (`fisuralab`): the two data contracts (ingestion + artifact), the named staged
  pipeline (preprocess → feature_extraction → train → infer → evaluate → export), the seeded RNG, the compact
  trace, the manifest, and the measured live-vs-precompute gate.
- EXAMPLE engine: a deterministic SIR epidemic (numpy-only, Pyodide-safe), **replace with the product's
  research-chosen SOTA engine**.
- Cases-by-category registry (4 regimes + 1 degenerate control); a live-lane entrypoint (`live.py`); tests for
  both contracts + pipeline determinism.
