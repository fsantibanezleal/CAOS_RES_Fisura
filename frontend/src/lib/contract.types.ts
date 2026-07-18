// CONTRACT 2 mirror (frontend side). MUST stay in lock-step with the Python schemas in
// data-pipeline/fisuralab/core/{artifact.py, manifest.py}. A drift here makes `tsc` fail -> the contract is
// enforced at BUILD time (the web cannot ship reading a shape the pipeline does not produce).

export interface MaskRLE {
  shape: [number, number];
  runs: number[]; // alternating run lengths over the row-major mask, starting with the 0-run
}

export interface TolMetrics {
  tolerance_px: number;
  precision: number;
  recall: number;
  f1: number;
  n_pred: number;
  n_gt: number;
}

export interface SegmentationRecord {
  iou_strict: number;
  protocol: string;
  tol2px: TolMetrics;
  tol5px: TolMetrics;
}

export interface WidthStats {
  n_points: number;
  edt_median: number | null;
  edt_p95: number | null;
  profile_median: number | null;
  profile_p95: number | null;
  disagreement_median: number | null;
}

export interface GeometryRecord {
  length_px: number;
  n_branches: number;
  n_endpoints: number;
  orientation_hist: number[]; // 18 bins of 10 degrees over [0, 180)
  width: WidthStats;
}

export interface WidthValidation {
  true_width_px: number;
  edt_on_gt_median: number | null;
  profile_on_gt_median: number | null;
  edt_abs_error: number | null;
  profile_abs_error: number | null;
}

export interface LevelRecord {
  mask_rle: MaskRLE;
  notes: string[];
  segmentation: SegmentationRecord | null;
}

export interface ArtifactSample {
  sample_id: string;
  source: string;
  license_tag: string;
  material: string;
  size: [number, number];
  mm_per_px: number | null;
  image_rel: string | null;            // pointer into data/examples for committed imagery
  synthetic_params: Record<string, number | string | boolean> | null;
  gt_rle: MaskRLE | null;
  levels: Record<string, LevelRecord>; // keys: L0..L5 (L5 absent when no classifier)
  geometry_level: string;
  geometry: GeometryRecord;
  width_validation: WidthValidation | null;
  overlays_rel?: string;               // derived/<case>/overlays/<sample_id> prefix (redistributable only)
}

export interface CaseArtifact {
  schema: string; // "fisura.artifact/v1"
  case_id: string;
  n_samples: number;
  samples: ArtifactSample[];
}

export interface ArtifactRef {
  path: string;
  format: string;
  artifact_schema: string;
  bytes: number;
}

export interface GateVerdict {
  lane: string;
  pure_python: boolean;
  wheels: string[];
  trace_bytes: number;
  run_ms_budget: number;
  trace_bytes_budget: number;
  reasons: string[];
}

export interface ExpectedBand {
  metric: string;
  min: number;
  max: number;
}

export interface CaseManifest {
  schema: string; // "fisura.manifest/v1"
  case_id: string;
  category: string;
  title: string;
  real_or_synthetic: string;
  expected_band: ExpectedBand;
  engine: { package: string; version: string; model: string };
  params: Record<string, unknown>;
  seed: number;
  artifact: ArtifactRef;
  lane: 'live' | 'precompute';
  gate: GateVerdict;
  flags: Array<{ sample_id: string; flags: string[] }>;
  metrics: Record<string, number | string>;
}

export interface CaseIndexEntry {
  case_id: string;
  category: string;
  manifest_path: string;
}

export interface CaseIndex {
  schema: string; // "fisura.index/v1"
  engine_version: string;
  n_cases: number;
  cases: CaseIndexEntry[];
}

/** Decode the row-major RLE into a Uint8Array (1 = crack). Mirrors core/artifact.py. */
export function rleDecode(rle: MaskRLE): Uint8Array {
  const [h, w] = rle.shape;
  const out = new Uint8Array(h * w);
  let pos = 0;
  let val = 0;
  for (const run of rle.runs) {
    if (val === 1) out.fill(1, pos, pos + run);
    pos += run;
    val = 1 - val;
  }
  return out;
}
