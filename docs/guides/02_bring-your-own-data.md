# Guide, bring your own data

The product is **applicable to NEW data**, not just the baked cases, that is what makes it a tool. The door is
**CONTRACT 1** (`data-pipeline/fisuralab/io/image_contract.py`, IO in `image_formats.py`).

1. Prepare your input: an image (PNG/JPG, grayscale or RGB, sides between 32 and 8192 px), optionally a
   binary mask (same size; nonzero = crack/defect), optionally the physical scale in mm per pixel
   (measure it: a reference object of known size in the frame, or a checkerboard homography; the
   pipeline never invents scale). Drop the files under `data/raw/` (git-ignored) or your
   `FISURA_DATA_ROOT` vault.
2. Declare the metadata CONTRACT 1 requires: `material` (concrete, asphalt, masonry, stone, steel,
   ceramic, synthetic, other), `source` (any provenance id), `license_tag` (see the table in
   [`data/README.md`](../../data/README.md); it decides what may ever be redistributed).
3. Run the pipeline (`scripts/precompute.{sh,ps1}`). CONTRACT 1 validates each sample: **rejected**
   with explicit reasons on hard violations (wrong dtype or shape, out-of-range floats, non-binary
   mask, absurd scale), **flagged** if plausible-but-suspicious (near-constant image, mask coverage
   above 50 percent, under 10 positive pixels), **accepted** otherwise. Nothing is silently coerced.
4. The pipeline produces a compact artifact + manifest you can replay in the SPA, exactly like the
   built-in cases. Physical-width outputs appear only when `mm_per_px` was provided.
5. **Live:** the browser lane runs the same validation core (numpy-only on purpose) on a photo you
   drop into the app, entirely client-side.

To fetch the open datasets the lab itself uses, run `scripts/fetch-data.{ps1,sh}` (idempotent; the
registry with licenses and gated-access notes is in `data/README.md`).

If your data legitimately doesn't fit, extend CONTRACT 1 (and its tests) **deliberately**, never loosen it just
to make bad data pass.
