# Cases: taxonomy and coverage

Every case carries a CATEGORY from the validated 7-track taxonomy. The App shows one selected case
as a workbench; Experiments/Benchmark aggregate across cases by category. A case is listed here the
moment its engine unit ships (code + tests + committed artifacts + this documentation, same commit
series); planned cases live in the product plan, not here.

## Category taxonomy (the 7 tracks)

| Category | Track | What its cases demonstrate |
|---|---|---|
| `classical-segmentation` | Classical pipelines | The staged S0-S8 engine and ladder L0-L5 on real imagery |
| `learned-segmentation` | Learned segmentation | Trained encoder-decoder and transformer crack networks |
| `foundation` | Foundation models | SAM adapters, frozen-feature heads, zero-shot rows |
| `anomaly` | Anomaly detection | Good-only training, AU-PRO honesty, the concrete-transfer study |
| `multiclass-damage` | Multi-class damage | Beyond binary cracks: bridge damage classes |
| `quantification-validation` | Quantification | Exact-ground-truth batteries, width estimators, calibration |
| `monitoring-deformation` | Monitoring + deformation | Epoch change, growth curves, DIC |

## Shipped cases

| Case | Category | Data | Doc |
|---|---|---|---|
| `bcl_examples` | classical-segmentation | committed CC0/CC BY examples (BCL + SDNET2018) | [bcl_examples.md](bcl_examples.md) |
| `synthetic_battery` | quantification-validation | generated cracks, exact ground truth | [synthetic_battery.md](synthetic_battery.md) |

## Coverage matrix (shipped / validated plan)

Classical-segmentation 1 of 4 planned; quantification-validation 1 of 2 planned; every other
category 0 shipped (engines arrive per unit). The validated 16-case matrix lives in the product
plan; this table only ever lists what is real.
