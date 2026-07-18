# data-pipeline/, the offline engine (`fisuralab`)

Rename `fisuralab` → `<slug>lab` per product. The **single source of physics/algorithm truth**; `frontend/` and
`app/` consume it, never re-implement it. Its own venv: **`.venv-pipeline`** (heavy SOTA engines, local-only).

## Layout (the package lives directly under `data-pipeline/`)
- `fisuralab/pipeline.py`, orchestrator + CLI (`python -m fisuralab.pipeline [all|<case>] [--seed N]`)
- `fisuralab/registry.py`, cases grouped by CATEGORY · `fisuralab/live.py`, Pyodide live entrypoint
- `fisuralab/io/`, `contract.py` (**CONTRACT 1**) · `formats.py` (standard readers/writers) · `schema.py` (types)
- `fisuralab/core/`, `rng.py` (seeded determinism) · `trace.py` · `manifest.py` (**CONTRACT 2**) · `gate.py`
- `fisuralab/model/`, the shared pure-Python core (Pyodide-safe); EXAMPLE = SIR
- `fisuralab/stages/`, `preprocess → feature_extraction → train → infer → evaluate → export`
- `fisuralab/cases/`, documented cases

Setup + run: `scripts/setup.{sh,ps1}` then `scripts/precompute.{sh,ps1}`. See
[../docs/architecture/05_precompute-pipeline.md](../docs/architecture/05_precompute-pipeline.md).
