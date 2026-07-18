// Ties the CONTRACT 2 TS mirror to the REAL committed artifacts: the index, a manifest, and its artifact must
// parse into the mirror types and pass shape checks (incl. an RLE round-trip against the recorded size). If the
// pipeline changes the schema without updating contract.types.ts, this test (and tsc) fail.
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { rleDecode, type CaseArtifact, type CaseIndex, type CaseManifest } from '../lib/contract.types';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const read = <T>(...p: string[]): T => JSON.parse(readFileSync(join(ROOT, 'data', 'derived', ...p), 'utf-8')) as T;

describe('CONTRACT 2 mirror matches the committed artifacts', () => {
  it('index -> manifest -> artifact parse into the mirror types and are consistent', () => {
    const idx = read<CaseIndex>('manifests', 'index.json');
    expect(idx.schema.startsWith('fisura.index/')).toBe(true);
    expect(idx.cases.length).toBeGreaterThan(0);

    const m = read<CaseManifest>('manifests', `${idx.cases[0].case_id}.json`);
    expect(m.schema.startsWith('fisura.manifest/')).toBe(true);
    expect(m.artifact.bytes).toBeGreaterThan(0);
    expect(['live', 'precompute']).toContain(m.lane);

    const artifact = read<CaseArtifact>(...m.artifact.path.split('/'));
    expect(artifact.schema.startsWith('fisura.artifact/')).toBe(true);
    expect(artifact.n_samples).toBe(artifact.samples.length);

    const s0 = artifact.samples[0];
    const levels = Object.keys(s0.levels);
    expect(levels.length).toBeGreaterThanOrEqual(5);
    const mask = rleDecode(s0.levels[levels[0]].mask_rle);
    expect(mask.length).toBe(s0.size[0] * s0.size[1]);
    expect(s0.geometry.orientation_hist.length).toBe(18);
  });
});
