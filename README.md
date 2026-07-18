# Fisura

[![CI](https://img.shields.io/github/actions/workflow/status/fsantibanezleal/CAOS_RES_Fisura/ci.yml?branch=main&label=CI)](https://github.com/fsantibanezleal/CAOS_RES_Fisura/actions)
[![License](https://img.shields.io/github/license/fsantibanezleal/CAOS_RES_Fisura)](LICENSE)
[![Version](https://img.shields.io/github/v/tag/fsantibanezleal/CAOS_RES_Fisura?label=version&sort=semver)](https://github.com/fsantibanezleal/CAOS_RES_Fisura/tags)

> **Status: under active construction.** The classical engine is real (staged S0-S8 pipeline,
> ladder L0-L5, dual-tolerance evaluation, synthetic regression battery, two replayable cases in
> the workbench); the learned, foundation, anomaly, multi-class, monitoring and DIC units land one
> vertical slice at a time (code + tests + deep docs per unit). Build tracker:
> [issue #1](https://github.com/fsantibanezleal/CAOS_RES_Fisura/issues/1). Live demo (planned):
> `fisura.fasl-work.com`.

A public research lab on **seeing damage in built materials**. One image of a concrete wall, a
pavement, a masonry facade or an industrial surface goes in; Fisura detects the damage (cracks,
spalling, surface defects), quantifies it in engineering units (width, length, orientation, density,
growth between inspections), and shows how every method family gets there:

1. **Classical pipelines**: illumination correction, adaptive thresholding, Hessian ridge filters,
   morphological path operators, minimal-path linking, skeleton geometry.
2. **Learned SOTA**: encoder-decoder and transformer crack segmentation, patch classification,
   multi-class structural-damage models, trained and evaluated on open datasets.
3. **Beyond SOTA**: promptable foundation models and unsupervised industrial anomaly detection
   applied to material surfaces.
4. **Measurement**: pixel-to-mm calibrated crack width and length, severity context from published
   guidance, change detection across inspection epochs, and vision-based deformation (2D digital
   image correlation) on specimen sequences.

The honest comparison across that whole ladder, on the same open cases with the same metrics, is the
product. Masks are never the end result: they are the input to engineering numbers.

## What it is (and is not)

- It **is** a reproducible offline pipeline (the heavy lane), a static replay web app over committed,
  audited artifacts, and a browser live lane where a user-provided photo is analyzed client-side
  (classical pipeline plus compact ONNX models); no image ever leaves the browser.
- It **is** dataset-honest: full open datasets live outside the repo and are fetched by scripts; the
  repo commits only tiny contract-passing samples and license-checked compact artifacts.
- It is **not** a certified inspection tool. Outputs are research artifacts for method comparison,
  not structural assessments; severity references are documented context, not engineering advice.
- Subsurface modalities (thermography, GPR, impact-echo, ultrasonics) are documented as context and
  are **out of the optical-RGB scope** of this lab.

## Quickstart (the archetype lanes)

```bash
# 1. create the reproducible environment (.venv + pinned per-need requirements)
./scripts/setup.sh                      # or scripts/setup.ps1 on Windows PowerShell

# 2. run the offline pipeline over every case: data/artifacts/ + manifests/
./scripts/precompute.sh                 # or scripts/precompute.ps1

# 3. the tests (determinism, both data contracts, the gate, parity)
.venv/bin/python -m pytest              # .venv/Scripts/python.exe on Windows

# 4. the web app consumes the artifacts (copy-data enforces the artifact contract)
cd frontend && npm install && node copy-data.mjs && npm run dev
```

## Repository shape (ADR-0057 archetype)

- `data-pipeline/fisuralab/`: the staged offline pipeline (preprocess, feature_extraction, train,
  infer, evaluate, export) behind two enforced data contracts (ingestion and artifact).
- `data/`: tiny committed examples plus per-case compact artifacts; heavy raw datasets stay outside
  git and are declared in `data/README.md`.
- `frontend/`: the replay + live web app (shared CAOS shell, EN/ES, light/dark, architecture panel).
- `app/`: dormant FastAPI module (not required by this product at the moment).
- `docs/`: the navigable wiki (architecture, frameworks, cases, guides), authored as the build
  proceeds, never at the end.

## License

MIT (see [LICENSE](LICENSE)). Dataset and model licenses are tracked per source in `data/README.md`
and `docs/frameworks/`; anything whose license does not allow public redistribution stays local-only.
