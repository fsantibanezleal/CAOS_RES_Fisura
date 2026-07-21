import { useEffect, useRef, useState } from 'react';
import { Callout } from '@fasl-work/caos-app-shell';
import { segment, type LiveResult } from '../../lib/liveEngine';
import { OverlayLegend } from '../../render/OverlayLegend';

// The live "bring your own image" lane (BL-013): the compact HrSegNet crack segmenter runs entirely
// in the browser (onnxruntime-web) on a photo the user drops. The image never leaves the browser.
// An optional mm-per-pixel scale turns the crack pixel count into a physical estimate.
export function LiveLane({ es }: { es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const [imgUrl, setImgUrl] = useState<string | null>(null);
  const [result, setResult] = useState<LiveResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [opacity, setOpacity] = useState(0.6);
  const [mmPerPx, setMmPerPx] = useState<string>('');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bitmapRef = useRef<ImageBitmap | null>(null);

  async function onFile(file: File) {
    setError(null);
    setResult(null);
    const url = URL.createObjectURL(file);
    setImgUrl(url);
    try {
      const bmp = await createImageBitmap(file);
      bitmapRef.current = bmp;
      setBusy(true);
      const r = await segment(bmp, 384);
      setResult(r);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  // composite the base image + the probability mask on the canvas
  useEffect(() => {
    const cv = canvasRef.current;
    const bmp = bitmapRef.current;
    if (!cv || !bmp || !result) return;
    const s = result.size;
    cv.width = s;
    cv.height = s;
    const ctx = cv.getContext('2d')!;
    ctx.drawImage(bmp, 0, 0, s, s);
    const ov = ctx.createImageData(s, s);
    for (let i = 0; i < result.prob.length; i++) {
      if (result.prob[i] > 0.5) {
        const o = i * 4;
        ov.data[o] = 230;
        ov.data[o + 1] = 57;
        ov.data[o + 2] = 70;
        ov.data[o + 3] = Math.round(255 * opacity * result.prob[i]);
      }
    }
    createImageBitmap(ov).then((b) => ctx.drawImage(b, 0, 0));
  }, [result, opacity]);

  const mmv = parseFloat(mmPerPx);
  const crackPx = result ? Math.round(result.crackFraction * result.size * result.size) : 0;
  // rough physical length proxy: crack area / an assumed ~3px mean width, times mm/px (stated as approximate)
  const lenMm = result && mmv > 0 ? (crackPx / 3) * mmv : null;

  return (
    <div>
      <p className="fs-hint" style={{ marginBottom: '0.8rem' }}>
        {t('Drop a crack photo below. The compact HrSegNet model segments it entirely in your browser (onnxruntime-web); the image never leaves your device. Add a millimetres-per-pixel scale for a physical estimate.', 'Suelta una foto de grieta abajo. El modelo compacto HrSegNet la segmenta por completo en tu navegador (onnxruntime-web); la imagen nunca sale de tu dispositivo. Agrega una escala de milímetros por píxel para una estimación física.')}
      </p>

      <div className="fs-wb-two">
        <div className="fs-wb-img">
          {!imgUrl ? (
            <label className="fs-drop">
              <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
              <span>{t('Click or drop a crack image', 'Haz clic o suelta una imagen de grieta')}</span>
            </label>
          ) : (
            <>
              <canvas ref={canvasRef} className="fs-tile-canvas" />
              <OverlayLegend items={[{ color: 'rgb(230,57,70)', label: t('HrSegNet crack probability', 'probabilidad de grieta HrSegNet') }]} />
              <div className="fs-chips" style={{ marginTop: '0.3rem' }}>
                <label className="fs-drop-mini">
                  <input type="file" accept="image/*" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
                  <span>{t('Try another image', 'Prueba otra imagen')}</span>
                </label>
              </div>
              <p className="fs-panel-sub">{t('The crack probability, in red, over your image. Nothing was uploaded anywhere.', 'La probabilidad de grieta, en rojo, sobre tu imagen. Nada se subió a ningún lado.')}</p>
            </>
          )}
        </div>
        <div className="fs-wb-read">
          {busy ? <Callout variant="note" title={t('Segmenting in your browser...', 'Segmentando en tu navegador...')}>{t('First run also fetches the model + wasm runtime; later runs are instant.', 'La primera corrida también descarga el modelo + runtime wasm; las siguientes son instantáneas.')}</Callout> : null}
          {error ? <Callout variant="honest" title={t('The in-browser model could not run', 'El modelo en el navegador no pudo correr')}><code>{error}</code></Callout> : null}
          {result ? (
            <>
              <div className="fs-kpis fs-kpis-2">
                <div className="fs-kpi"><div className="fs-kpi-v">{(result.crackFraction * 100).toFixed(2)}%</div><div className="fs-kpi-l">{t('crack pixels', 'píxeles de grieta')}</div></div>
                <div className="fs-kpi"><div className="fs-kpi-v">{result.ms.toFixed(0)} ms</div><div className="fs-kpi-l">{t('inference time', 'tiempo de inferencia')}</div></div>
                <div className="fs-kpi"><div className="fs-kpi-v">{result.backend}</div><div className="fs-kpi-l">{t('backend', 'backend')}</div></div>
                <div className="fs-kpi"><div className="fs-kpi-v">{lenMm != null ? `${lenMm.toFixed(0)}` : '--'}</div><div className="fs-kpi-l">{t('approx crack length (mm)', 'largo aprox. de grieta (mm)')}</div></div>
              </div>
              <label>
                <span className="fs-ctl-row"><span>{t('Overlay opacity', 'Opacidad del overlay')}</span><b>{opacity.toFixed(2)}</b></span>
                <input className="range" type="range" min={0.2} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
              </label>
              <div className="fs-ctl">
                <div className="fs-ctl-cap">{t('Scale (optional): millimetres per pixel', 'Escala (opcional): milímetros por píxel')}</div>
                <input className="fs-sel" type="number" step="0.001" min="0" placeholder="e.g. 0.08" value={mmPerPx} onChange={(e) => setMmPerPx(e.target.value)} />
                <p className="fs-hint">{t('The length estimate is a coarse area/width proxy, not a calibrated measurement. Use the classical width bench for validated widths.', 'La estimación de largo es un proxy grueso de área/ancho, no una medición calibrada. Usa el banco de ancho clásico para anchos validados.')}</p>
              </div>
            </>
          ) : null}
          <Callout variant="honest" title={t('What this lane is, honestly', 'Qué es este carril, con honestidad')}>
            {t('This runs one compact learned segmenter (HrSegNet, ~0.2 MB) client-side, the browser-shippable rung. The full classical ladder and the heavier SOTA models run in the offline lane and are replayed in the other tabs. HrSegNet trained on CrackSeg9k, so your photo is a genuine out-of-distribution test.', 'Esto corre un segmentador aprendido compacto (HrSegNet, ~0.2 MB) del lado del cliente, el peldaño que cabe en el navegador. La escalera clásica completa y los modelos SOTA más pesados corren en el carril offline y se reproducen en las otras pestañas. HrSegNet se entrenó en CrackSeg9k, así que tu foto es una prueba genuina fuera de distribución.')}
          </Callout>
        </div>
      </div>
    </div>
  );
}
