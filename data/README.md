# data/, the data contracts + the dataset registry

Governed by the two data contracts of ADR-0057. The image-domain CONTRACT 1 below is Fisura's real
ingestion gate; the archetype's SIR example schema remains active for the example engine only until
the classical-ladder unit replaces it.

## Layout

| Path | What | Git |
|---|---|---|
| `raw/` and `raw-local/` | full source datasets | **git-ignored** (fetched by `scripts/fetch-data.{ps1,sh}` into the vault) |
| `examples/` | tiny curated samples that PASS Contract 1, permissive licenses only, with `manifest.json` attribution | committed |
| `derived/<case>/` | the compact, standard-format artifacts the web replays | committed |
| `derived/manifests/` | per-case `<case>.json` (Contract 2) + the flat `index.json` inventory | committed |
| `demo/` | small deterministic payload for smoke | committed |

The vault root is `FISURA_DATA_ROOT` (env or `.env`), defaulting to `data/raw-local/` in a fresh
clone. Nothing heavy ever enters git.

## CONTRACT 1, ingestion (raw to pipeline), the bring-your-own-data gate

Defined in `data-pipeline/fisuralab/io/image_contract.py` (numpy-only core, reused by the browser
live lane) with IO in `image_formats.py`. A sample is **accepted** iff it satisfies the schema;
**rejected** with explicit reasons otherwise (never silently coerced); plausible-but-suspicious
samples are **flagged**.

| Field | Type / range | Policy |
|---|---|---|
| `image` | HxW or HxWx3; uint8, or float32/64 in [0, 1]; sides 32..8192 | outside: reject; near-constant: flag |
| `mask` (optional) | HxW; bool or binary int ({0,1} or {0,255}); same HxW as image | non-binary or mismatched: reject; empty: flag (valid uncracked); coverage over 50 percent: flag (polarity check); under 10 positive px: flag |
| `mm_per_px` (optional) | finite, 0.001..50 | outside: reject; absent: physical widths are simply not produced (never invented) |
| `material` | one of concrete, asphalt, masonry, stone, steel, ceramic, synthetic, other | else reject |
| `source` | non-empty provenance id | empty: reject |
| `license_tag` | cc0, cc-by, cc-by-sa, cc-by-nc, cc-by-nc-sa, academic, competition, unknown | else reject; drives redistribution below |

**Redistribution rule (enforced by `is_redistributable`):** imagery may be committed to this public
MIT repository only under `cc0`, `cc-by` (with attribution) or `cc-by-sa` (share-alike, in a marked
data area). Everything else stays in the local vault; only metrics and plots are published.

## CONTRACT 2, artifact (pipeline to web)

Each pipeline run writes a compact artifact under `derived/<case>/` and a manifest
(`derived/manifests/<case>.json`) recording params, seed, engine + version, artifact byte size, the
measured lane/gate verdict, Contract-1 flags, and evaluation metrics.
`frontend/src/lib/contract.types.ts` mirrors these schemas so any drift fails the web build. The web
loads ONLY these committed artifacts (plus the explicit live lane).

## The dataset registry (retrieval verified 2026-07-18; full research: the Fisura dossiers)

| Dataset | Task | License | Retrieval | In this repo |
|---|---|---|---|---|
| Bridge Crack Library (BCL) | patch segmentation, concrete + steel + noise | CC0 1.0 | Harvard Dataverse DOI 10.7910/DVN/RURXSH (fetch-data) | examples + derived artifacts allowed, with credit (Ye et al. 2021) |
| CrackSeg9k | segmentation aggregation (9,255) | CC0 label; component licenses conflict | Harvard Dataverse DOI 10.7910/DVN/EGIEBY (fetch-data) | metrics + plots; imagery treated as NOT safely redistributable |
| SDNET2018 | patch classification (56k) | CC BY 4.0 | USU Digital Commons, DOI 10.15142/T3TD19 (fetch-data resolves the landing link) | examples + derived allowed, with attribution |
| Ozgenel METU 40k | patch classification | CC BY 4.0 | Mendeley Data DOI 10.17632/5y9wdsg2zt.2, public API route (fetch-data) | examples + derived allowed, with attribution |
| DeepCrack-537 | segmentation | no explicit license (cite-only) | GitHub yhlleo/DeepCrack repo zip (fetch-data) | local only; metrics fine |
| CrackForest (CFD) | segmentation | non-commercial research | git clone cuilimeng/CrackForest-dataset (fetch-data) | local only; metrics fine |
| fyangneil pavement bundle (Crack500, GAPs384, CFD, AEL, CrackTree200) | segmentation | cite-only mix; GAPs academic terms | Google Drive id 13_vDYl54Mrd34dddX9w4ppAEiuWv4MlD via gdown (fetch-data) | local only; metrics fine |
| khanhha 11.2k aggregation | segmentation | cite-only components | Kaggle lakshaymiddha/crack-segmentation-dataset (fetch-data) | local only |
| UAV75 | segmentation, UAV concrete | GPL-3.0 (repo data) | git clone ben-z-original/uav75 (fetch-data) | local only; metrics fine |
| Masonry (Dais) subset | classification + segmentation | GPL-3.0 repo; full set not public | git clone dimitrisdais/crack_detection_CNN_masonry (fetch-data) | local only |
| CODEBRIM | multi-label damage (6 classes) | non-commercial | Zenodo DOI 10.5281/zenodo.2620293 (fetch-data) | local only; metrics fine |
| dacl10k | 19-class damage segmentation | CC BY-NC 4.0 | official S3 zips (fetch-data) | local only; metrics fine |
| VisA | anomaly detection (10,821) | CC BY 4.0 | Amazon S3 tar (fetch-data) | examples + derived allowed, with attribution |
| KolektorSDD / SDD2 | defect segmentation | CC BY-NC-SA 4.0 | vicos.si direct links (fetch-data) | local only; metrics fine |
| NEU-DET | steel defect classification/detection | cite-only | Kaggle kaustubhdikshit/neu-surface-defect-database (fetch-data) | local only |
| DAGM2007 | synthetic defect detection | CC BY 4.0 | Kaggle mhskjelvareid mirror (fetch-data) or HCI form | derived allowed, with attribution |
| Magnetic Tile | defect segmentation (ceramic) | cite-only | git clone abin24/Magnetic-tile-defect-datasets. (fetch-data) | local only |
| RDD2022 | road damage detection (47k boxes) | CC BY-SA 4.0 | official S3 zip (fetch-data) | derived ship CC BY-SA in a marked area; metrics fine |
| MVTec AD / AD 2 | anomaly benchmark | CC BY-NC-SA 4.0 | registration form at mvtec.com (manual, once) | local only; metrics fine |
| GAPs (official) | pavement distress | academic-only, credential-gated | form to TU Ilmenau + gaps-dataset package (manual) | local only |
| OmniCrack30k | segmentation benchmark (30k) | per-request, non-commercial | email request (manual); its public nnU-Net checkpoint is free | local only; metrics fine |
| Severstal | steel defect segmentation | Kaggle competition rules | accept rules, then kaggle CLI (manual, once) | local only, strictly no redistribution |
| CrackVision12K | refined aggregation | non-commercial | UCL RDR article 26946472 (browser; bot-blocked) | local only |

## Committed examples

`examples/manifest.json` is the machine-readable attribution record (file, mask, source, license,
URL, citation, material). Only `cc0` / `cc-by` samples are committed, and every example must pass
CONTRACT 1 in CI (`tests/test_image_contract.py`). Current sources: BCL (CC0), SDNET2018 (CC BY 4.0).

The SIR example (`examples/params.csv`) belongs to the archetype's reference engine and leaves with it.
