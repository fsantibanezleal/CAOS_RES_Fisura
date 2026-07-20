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
