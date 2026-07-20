// Fetch the committed artifacts (copied into public/data by copy-data.mjs). Works against the static site OR,
// if app/ is activated, you can repoint these to the API, same shapes (CONTRACT 2).
import type { CaseArtifact, CaseIndex, CaseManifest } from '../lib/contract.types';

const base = import.meta.env.BASE_URL;

async function getJSON<T>(rel: string): Promise<T> {
  const res = await fetch(`${base}data/${rel}`);
  if (!res.ok) throw new Error(`fetch ${rel}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

export const loadIndex = (): Promise<CaseIndex> => getJSON<CaseIndex>('manifests/index.json');
export const loadManifest = (caseId: string): Promise<CaseManifest> =>
  getJSON<CaseManifest>(`manifests/${caseId}.json`);
export const loadArtifact = (artifactPath: string): Promise<CaseArtifact> =>
  getJSON<CaseArtifact>(artifactPath);

/** Load a derived example artifact by case slug (classical/learned/anomaly all share sample_ids). */
export const loadCaseArtifact = (caseSlug: string): Promise<CaseArtifact> =>
  getJSON<CaseArtifact>(`${caseSlug}/artifact.json`);

/** URL of the committed anomaly heat PNG (Beyond-SOTA overlay). */
export const heatUrl = (heatRel: string): string => `${base}data/${heatRel}`;

/** URL of a committed overlay PNG (redistributable imagery only). */
export const overlayUrl = (overlaysRel: string, suffix: string): string =>
  `${base}data/${overlaysRel}${suffix}`;

/** URL of any committed workbench PNG (preprocessing intermediates, SLIC variants). */
export const workbenchUrl = (rel: string): string => `${base}data/${rel}`;

// The per-image workbench index (preprocessing intermediates + the SLIC grid).
export interface WorkbenchPrep { gray: string; flatten: string; denoise: string; ridge: string; }
export interface WorkbenchSample {
  material: string;
  size: [number, number];
  prep: WorkbenchPrep;
  slic: Record<string, string>;             // key "<n>_<c>" -> png rel path
  slic_real_counts: Record<string, number>; // key "<n>_<c>" -> actual superpixel count
}
export interface WorkbenchIndex {
  schema: string;
  slic_grid: { n_segments: number[]; compactness: number[] };
  samples: Record<string, WorkbenchSample>;
}
export const loadWorkbench = (): Promise<WorkbenchIndex> => getJSON<WorkbenchIndex>('workbench/index.json');

// The per-image enrichment artifact (skeleton graph, width profile, orientation rose, per-model metrics).
export interface SkeletonNode { x: number; y: number; degree: number; }
export interface SkeletonEdge { polyline: [number, number][]; length_px: number; mean_halfwidth_px: number; }
export interface EnrichSkeleton { nodes: SkeletonNode[]; edges: SkeletonEdge[]; n_endpoints: number; n_junctions: number; }
export interface WidthProfile { s_px: number[]; w_dt_px: number[]; mm_per_px: number | null; }
export interface Rose { bins_deg: number[]; weight: number[]; }
export interface ModelMetrics {
  sweep: { tol_px: number[]; f1: number[] };
  confusion: { tp: number; fp: number; fn: number; tol_px: number };
  f1_2px: number | null;
  f1_5px: number | null;
}
export interface Enrichment {
  sample_id: string;
  size: [number, number];
  material: string;
  has_gt: boolean;
  skeleton?: EnrichSkeleton;
  width_profile?: WidthProfile;
  rose?: Rose;
  uncertainty?: { mean_std: number; disagree_px: number; n_models: number };
  models: Record<string, ModelMetrics>;
}
export const loadEnrichment = (sampleId: string): Promise<Enrichment> => getJSON<Enrichment>(`enrichment/${sampleId}.json`);
