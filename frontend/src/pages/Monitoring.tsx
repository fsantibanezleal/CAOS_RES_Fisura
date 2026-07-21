import { useEffect, useState } from 'react';
import { Callout, Cite, Refs } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { OverlayLegend } from '../render/OverlayLegend';

// The monitoring track (BL-011): two-epoch registration + differential crack mapping. Two surveys of
// the same surface weeks apart have different pose + lighting; the growth signal (0.05-0.5 mm width,
// mm-cm tip extension) is smaller than the nuisance variation, so change detection runs AFTER metric
// registration, reporting per-branch width deltas + new-branch events, never raw pixel differences.
// Validated on a synthetic pair with EXACT ground truth (dossier 04 section 4).

interface Diff {
  unit: string;
  width_median_ep1: number; width_median_ep2: number; width_delta_median: number;
  width_p95_ep1: number; width_p95_ep2: number;
  length_ep1_px: number; length_ep2_px: number; length_delta_px: number;
  new_branch_px: number; grew: boolean;
}
interface Growth {
  case: string; mm_per_px: number;
  registration: { method: string; inliers: number };
  measured: Diff;
  ground_truth: { true_width_median_delta_mm: number; true_length_delta_px: number };
  overlays: { epoch1: string; epoch2_raw: string; epoch2_registered: string; change: string };
  framing: string;
}

export default function Monitoring() {
  const t = useT();
  const [d, setD] = useState<Growth | null>(null);
  const [err, setErr] = useState(false);
  const [view, setView] = useState<'change' | 'epoch1' | 'epoch2_raw' | 'epoch2_registered'>('change');
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/monitoring/growth.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setD)
      .catch(() => setErr(true));
  }, []);

  return (
    <>
      <p className="fs-kicker">{t('Monitoring', 'Monitoreo')}</p>
      <h1>{t('Two epochs, one crack: measuring growth', 'Dos épocas, una grieta: medir el crecimiento')}</h1>
      <p className="fs-lead">
        {t(
          'Detection becomes prognosis when you measure change over time. Two surveys of the same surface, weeks apart, have different camera poses and lighting, and the growth signal (0.05 to 0.5 mm of widening, millimetres to centimetres of tip extension) is smaller than that nuisance variation. So change detection runs after metric registration, and reports per-branch width deltas and new-branch events, never raw pixel differences.',
          'La detección se vuelve pronóstico cuando mides el cambio en el tiempo. Dos inspecciones de la misma superficie, con semanas de diferencia, tienen distintas poses de cámara e iluminación, y la señal de crecimiento (0.05 a 0.5 mm de ensanchamiento, milímetros a centímetros de extensión de punta) es más pequeña que esa variación de ruido. Por eso la detección de cambios corre tras un registro métrico, y reporta deltas de ancho por rama y eventos de rama nueva, nunca diferencias crudas de píxeles.',
        )}{' '}
        (<Cite id="spencer2019" />)
      </p>

      {err ? (
        <Callout variant="note" title={t('Monitoring artifact baking', 'Artefacto de monitoreo horneándose')}>
          {t('The two-epoch growth case is not committed yet.', 'El caso de crecimiento de dos épocas aún no está versionado.')}
        </Callout>
      ) : !d ? (
        <div className="fs-panel"><div className="fs-panel-t">{t('Loading...', 'Cargando...')}</div></div>
      ) : (
        <Body d={d} view={view} setView={setView} es={t('x', 'y') === 'y'} />
      )}

      <Refs label={t('Refs', 'Refs')} ids={['spencer2019', 'zhu2023crackpropnet', 'melching2022', 'paris1963']} />
    </>
  );
}

function Body({ d, view, setView, es }: { d: Growth; view: string; setView: (v: 'change' | 'epoch1' | 'epoch2_raw' | 'epoch2_registered') => void; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const m = d.measured;
  const gt = d.ground_truth;
  const wErr = Math.abs(m.width_delta_median - gt.true_width_median_delta_mm);
  const views: { id: 'change' | 'epoch1' | 'epoch2_raw' | 'epoch2_registered'; en: string; es: string }[] = [
    { id: 'change', en: 'Change map', es: 'Mapa de cambio' },
    { id: 'epoch1', en: 'Epoch 1', es: 'Época 1' },
    { id: 'epoch2_raw', en: 'Epoch 2 (new pose)', es: 'Época 2 (nueva pose)' },
    { id: 'epoch2_registered', en: 'Epoch 2 registered', es: 'Época 2 registrada' },
  ];
  const src = d.overlays[view as keyof typeof d.overlays];
  return (
    <>
      <div className="fs-kpis">
        <div className="fs-kpi"><div className="fs-kpi-v">+{m.width_delta_median.toFixed(3)}</div><div className="fs-kpi-l">{t('median width growth (mm)', 'crecimiento de ancho mediano (mm)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{m.length_delta_px}</div><div className="fs-kpi-l">{t('tip extension (px)', 'extensión de punta (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{m.new_branch_px}</div><div className="fs-kpi-l">{t('new-crack pixels', 'píxeles de grieta nueva')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{d.registration.inliers}</div><div className="fs-kpi-l">{t('ORB registration inliers', 'inliers de registro ORB')}</div></div>
      </div>

      <div className="fs-wb-two" style={{ marginTop: '1rem' }}>
        <div className="fs-wb-img">
          <img className="fs-wb-photo" src={`${import.meta.env.BASE_URL}data/${src}`} alt={view} />
          <div className="fs-chips" style={{ marginTop: '0.4rem' }}>
            {views.map((v) => <button key={v.id} className={`chip ${view === v.id ? 'on' : ''}`} onClick={() => setView(v.id)}>{t(v.en, v.es)}</button>)}
          </div>
          {view === 'change' ? (
            <OverlayLegend items={[
              { color: 'rgb(70,200,110)', label: t('crack in both epochs', 'grieta en ambas épocas') },
              { color: 'rgb(230,60,60)', label: t('new in epoch 2 (growth)', 'nueva en época 2 (crecimiento)') },
            ]} />
          ) : null}
          <p className="fs-panel-sub">
            {view === 'change'
              ? t('The change map after registration: green is the crack present in both surveys, red is the new growth (widening + tip extension) in epoch 2.', 'El mapa de cambio tras el registro: verde es la grieta presente en ambas inspecciones, rojo es el crecimiento nuevo (ensanchamiento + extensión de punta) en la época 2.')
              : view === 'epoch2_raw'
                ? t('Epoch 2 as captured from a different camera pose (translation + small rotation + scale). Comparing it to epoch 1 directly would be dominated by the pose change.', 'La época 2 capturada desde una pose de cámara distinta (traslación + rotación pequeña + escala). Compararla con la época 1 directamente estaría dominado por el cambio de pose.')
                : view === 'epoch2_registered'
                  ? t('Epoch 2 after ORB + RANSAC registration warps it back into epoch 1 frame: now the two surveys are pixel-aligned and only the real crack change remains.', 'La época 2 tras el registro ORB + RANSAC que la deforma de vuelta al marco de la época 1: ahora las dos inspecciones están alineadas y solo queda el cambio real de la grieta.')
                  : t('The reference survey (epoch 1): a shorter, thinner crack.', 'La inspección de referencia (época 1): una grieta más corta y delgada.')}
          </p>
        </div>
        <div className="fs-wb-read">
          <Callout variant="strong" title={t('The pipeline recovers the true growth', 'El pipeline recupera el crecimiento verdadero')}>
            {t(`Registration found ${d.registration.inliers} feature inliers and recovered the camera pose. The measured median width growth is ${m.width_delta_median.toFixed(3)} mm against a known ground-truth growth of ${gt.true_width_median_delta_mm.toFixed(3)} mm (absolute error ${wErr.toFixed(3)} mm), and the tip extension ${m.length_delta_px} px against ${gt.true_length_delta_px} px. The synthetic pair has exact ground truth by construction, which is the only way to validate a monitoring pipeline honestly.`, `El registro encontró ${d.registration.inliers} inliers de features y recuperó la pose de cámara. El crecimiento de ancho mediano medido es ${m.width_delta_median.toFixed(3)} mm contra un crecimiento ground-truth conocido de ${gt.true_width_median_delta_mm.toFixed(3)} mm (error absoluto ${wErr.toFixed(3)} mm), y la extensión de punta ${m.length_delta_px} px contra ${gt.true_length_delta_px} px. El par sintético tiene ground truth exacto por construcción, la única forma de validar un pipeline de monitoreo con honestidad.`)}
          </Callout>
          <div className="fs-tablewrap">
            <table className="fs-table">
              <thead><tr><th>{t('Quantity', 'Cantidad')}</th><th className="mono">{t('Epoch 1', 'Época 1')}</th><th className="mono">{t('Epoch 2', 'Época 2')}</th><th className="mono">Δ</th></tr></thead>
              <tbody>
                <tr><td>{t('median width (mm)', 'ancho mediano (mm)')}</td><td className="mono">{m.width_median_ep1.toFixed(3)}</td><td className="mono">{m.width_median_ep2.toFixed(3)}</td><td className="mono">+{m.width_delta_median.toFixed(3)}</td></tr>
                <tr><td>{t('p95 width (mm)', 'ancho p95 (mm)')}</td><td className="mono">{m.width_p95_ep1.toFixed(3)}</td><td className="mono">{m.width_p95_ep2.toFixed(3)}</td><td className="mono">+{(m.width_p95_ep2 - m.width_p95_ep1).toFixed(3)}</td></tr>
                <tr><td>{t('skeleton length (px)', 'largo esqueleto (px)')}</td><td className="mono">{m.length_ep1_px}</td><td className="mono">{m.length_ep2_px}</td><td className="mono">+{m.length_delta_px}</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <Callout variant="honest" title={t('Scope, stated plainly', 'Alcance, dicho claramente')}>
        {t('This validates the registration + differential-mapping pipeline on a synthetic pair with exact ground truth; real registered inspection pairs are the field goal. Growth-rate framing follows fracture mechanics (image tracking measures the crack-length history a(N)), but Paris-law life prediction is calibrated for metals under small-scale yielding and is out of an optical concrete lab scope: Fisura measures and publishes a(N), it does not certify remaining life.', 'Esto valida el pipeline de registro + mapeo diferencial en un par sintético con ground truth exacto; los pares reales de inspección registrados son la meta de campo. El encuadre de tasa de crecimiento sigue la mecánica de fractura (el seguimiento por imagen mide la historia de largo a(N)), pero la predicción de vida por ley de Paris está calibrada para metales bajo fluencia de pequeña escala y queda fuera del alcance de un laboratorio óptico de hormigón: Fisura mide y publica a(N), no certifica vida remanente.')}
      </Callout>
    </>
  );
}
