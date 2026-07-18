# Case `bcl_examples`: the committed example set through the ladder

Category `classical-segmentation` · real imagery · seed 42 · 6 samples

## Data and curation

The six patches this public repository can legally ship, loaded through CONTRACT 1 from
`data/examples/` (attribution in `manifest.json`):

- **Bridge Crack Library** (CC0 1.0; Ye et al. 2021, Harvard Dataverse DOI 10.7910/DVN/RURXSH):
  two concrete crack patches, one steel crack patch, one uncracked noise control, each with a pixel
  mask. Curation note, recorded on purpose: the source labels use a black-crack-on-white
  convention; the committed masks were INVERTED once at curation to the contract's True = crack
  convention. The contract's mask-coverage flag is what caught the polarity (a 97 percent "crack"
  coverage is not a crack, it is an inverted label).
- **SDNET2018** (CC BY 4.0; Dorafshan et al. 2018, DOI 10.15142/T3TD19): one cracked and one
  uncracked concrete pavement patch, classification-style (no pixel masks), kept as controls the
  segmentation scores do not apply to.

## Measured behaviour (pinned stack, seed 42; buffered F1, both protocols in the artifact)

Per-patch, L-levels at 5 px tolerance (the numbers the workbench charts show):

- `bcl-nonsteel-c10` (thin, clean crack, true width 2.8 px): F1 0.91 to 0.94 at L2/L3: the
  classical ladder at its best.
- `bcl-nonsteel-c1` (wide diffuse crack, true width 6 px): F1 about 0.24 at L3, 0.48 at L2. The
  Field view shows why: a bright vertical feature recruits the ridge response while the true crack
  is low-contrast.
- `bcl-steel-s1` (scratched steel): F1 about 0.22: scratches are curvilinear dark structures too;
  material texture is the classical ladder's celling.
- Case rollup: mean L4 F1 at 5 px = 0.455 (expected band 0.35 to 1.0). L2 (oriented top-hat) beats
  L3 (ridge) on two of three masked patches: on real texture the morphological bank is often the
  more robust choice, a finding the research anticipated and this case makes concrete.

## What this case is for

It anchors every later rung: the learned models enter THIS workbench on THESE patches, so the
reader sees classical-vs-learned on identical pixels. It also demonstrates the licensing boundary:
these six patches are in the repo because their licenses allow it; the fetched datasets stay in the
local vault with only metrics published.

## Artifact

`data/derived/bcl_examples/artifact.json` + per-sample, per-level overlays under
`data/derived/bcl_examples/overlays/`. Regenerate with
`python -m fisuralab.pipeline bcl_examples`.
