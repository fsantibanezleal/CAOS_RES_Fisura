// The live "bring your own image" lane (BL-013): the compact HrSegNet crack segmenter running
// entirely in the browser via onnxruntime-web on a user-supplied photo. Images never leave the
// browser. FrothSeg deploy gotchas (from the reference): wasm numThreads = 1 (COOP/COEP not set on
// Pages), probe WebGPU and fall back to wasm, load the .wasm from the app base path.
import * as ort from 'onnxruntime-web/wasm';

// HrSegNet was trained on img/255 with NO ImageNet mean/std normalization (see the training dataset:
// g = img/255.0). Applying mean/std here would massively over-segment; match the training convention.
let sessionPromise: Promise<ort.InferenceSession> | null = null;
let backend = 'wasm';

function configureOrt() {
  // single-threaded wasm: GitHub Pages does not send the cross-origin-isolation headers that
  // SharedArrayBuffer (threads) needs, so force 1 thread rather than crash.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.simd = true;
  // Serve the wasm runtime (both the .wasm binary AND the loader .mjs) from the app base path.
  // copy-data.mjs copies the ort dist files to public/, so they sit next to index.html; the map
  // form pins every artifact ort may request so none falls back to the bundler /assets path (which
  // 404s on the static Pages host under a subpath).
  const base = import.meta.env.BASE_URL; // "./" or "/<repo>/"
  const abs = base.startsWith('http') ? base : new URL(base, location.href).href;
  ort.env.wasm.wasmPaths = {
    wasm: `${abs}ort-wasm-simd-threaded.wasm`,
    mjs: `${abs}ort-wasm-simd-threaded.mjs`,
  } as unknown as Record<string, string>;
}

async function pickProviders(): Promise<string[]> {
  // wasm-only: the HrSegNet model is ~0.2 MB and runs in a few hundred ms single-threaded, so the
  // WebGPU (jsep) path is not worth the extra .jsep.mjs glue-file dependency that the Pages static
  // host does not resolve under the bundler's /assets path. Reliable everywhere.
  backend = 'wasm';
  return ['wasm'];
}

export function activeBackend(): string {
  return backend;
}

export async function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    configureOrt();
    sessionPromise = pickProviders().then((providers) =>
      ort.InferenceSession.create(`${import.meta.env.BASE_URL}models/hrsegnet_b16.onnx`, {
        executionProviders: providers,
        graphOptimizationLevel: 'all',
      }),
    );
  }
  return sessionPromise;
}

// resize an ImageBitmap/HTMLImageElement to a square `size`, return normalized CHW float32 + the
// drawn RGBA (so the caller can composite the mask over the exact pixels the model saw).
export function preprocess(img: CanvasImageSource, size: number): { tensor: Float32Array; rgba: Uint8ClampedArray } {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(img, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const tensor = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    // img/255 only (HrSegNet's training normalization); NO ImageNet mean/std.
    tensor[i] = data[i * 4] / 255;
    tensor[plane + i] = data[i * 4 + 1] / 255;
    tensor[2 * plane + i] = data[i * 4 + 2] / 255;
  }
  return { tensor, rgba: data };
}

export interface LiveResult {
  size: number;
  prob: Float32Array;   // sigmoid(logits), size*size
  crackFraction: number; // fraction of pixels above 0.5
  ms: number;
  backend: string;
}

// Run HrSegNet on an image, returning the per-pixel crack probability at `size`x`size`.
export async function segment(img: CanvasImageSource, size = 384): Promise<LiveResult> {
  const session = await getSession();
  const { tensor } = preprocess(img, size);
  const input = new ort.Tensor('float32', tensor, [1, 3, size, size]);
  const t0 = performance.now();
  const out = await session.run({ image: input });
  const ms = performance.now() - t0;
  const logits = out.logits.data as Float32Array;
  const prob = new Float32Array(logits.length);
  let pos = 0;
  for (let i = 0; i < logits.length; i++) {
    const p = 1 / (1 + Math.exp(-logits[i]));
    prob[i] = p;
    if (p > 0.5) pos++;
  }
  return { size, prob, crackFraction: pos / logits.length, ms, backend: activeBackend() };
}
