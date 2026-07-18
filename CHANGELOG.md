# Changelog

All notable changes to this product. Format: `X.XX.XXX` (display), see `fisuralab.__version__`. Keep `0.x`
while on mock/synthetic data. Tag every release.

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
