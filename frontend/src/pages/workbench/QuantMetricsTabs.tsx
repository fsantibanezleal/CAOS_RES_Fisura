import { useState } from 'react';
import { Callout } from '@fasl-work/caos-app-shell';
import type uPlot from 'uplot';
import type { Enrichment } from '../../api/artifacts';
import { RoseDiagram } from '../../render/RoseDiagram';
import { SkeletonOverlay } from '../../render/SkeletonOverlay';
import { UPlotChart } from '../../render/UPlotChart';

// Quantification tab: the crack turned into engineering quantities (skeleton graph on the image,
// width-along-arc-length w(s), orientation rose). Research shortlist items 1-3.
export function QuantTab({ enrich, imageUrl, es }: { enrich: Enrichment | null; imageUrl: string | null; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const [hover, setHover] = useState<number | null>(null);
  if (!enrich?.skeleton || !enrich.skeleton.nodes.length) {
    return (
      <Callout variant="note" title={t('No crack skeleton for this image', 'Sin esqueleto de grieta para esta imagen')}>
        {t('Quantification needs a crack: the uncracked control patches have no skeleton. Pick a cracked image.', 'La cuantificación necesita una grieta: los parches de control sin grieta no tienen esqueleto. Elige una imagen con grieta.')}
      </Callout>
    );
  }
  const sk = enrich.skeleton;
  const wp = enrich.width_profile;
  const mmpp = wp?.mm_per_px ?? null;
  const widths = wp?.w_dt_px ?? [];
  const wMax = widths.length ? Math.max(...widths) : 0;
  const wMed = widths.length ? [...widths].sort((a, b) => a - b)[Math.floor(widths.length / 2)] : 0;
  return (
    <div>
      <p className="fs-hint" style={{ marginBottom: '0.8rem' }}>
        {t('The crack turned into engineering quantities: its skeleton graph, the width along its length, and its dominant direction. This is what an inspector actually needs from a mask.', 'La grieta convertida en cantidades de ingeniería: su grafo de esqueleto, el ancho a lo largo de su longitud, y su dirección dominante. Esto es lo que un inspector realmente necesita de una máscara.')}
      </p>
      <div className="fs-wb-two">
        <div className="fs-wb-img">
          <SkeletonOverlay imageUrl={imageUrl} size={enrich.size} skeleton={sk} hovered={hover} onHover={setHover} />
          <p className="fs-panel-sub">{t('The crack skeleton: blue = branch segments, orange dots = junctions, small dots = endpoints. Hover a branch to trace it.', 'El esqueleto de la grieta: azul = segmentos de rama, puntos naranja = uniones, puntos pequenos = extremos. Pasa el cursor sobre una rama para trazarla.')}</p>
        </div>
        <div className="fs-wb-read">
          <div className="fs-kpis fs-kpis-2">
            <div className="fs-kpi"><div className="fs-kpi-v">{sk.edges.length}</div><div className="fs-kpi-l">{t('branch segments', 'segmentos de rama')}</div></div>
            <div className="fs-kpi"><div className="fs-kpi-v">{sk.n_junctions}</div><div className="fs-kpi-l">{t('junctions', 'uniones')}</div></div>
            <div className="fs-kpi"><div className="fs-kpi-v">{sk.n_endpoints}</div><div className="fs-kpi-l">{t('endpoints', 'extremos')}</div></div>
            <div className="fs-kpi"><div className="fs-kpi-v">{mmpp ? (wMed * mmpp).toFixed(2) : wMed.toFixed(1)}</div><div className="fs-kpi-l">{mmpp ? t('median width (mm)', 'ancho mediano (mm)') : t('median width (px)', 'ancho mediano (px)')}</div></div>
          </div>
          {enrich.rose && enrich.rose.bins_deg.length ? (
            <div className="fs-panel">
              <div className="fs-panel-t">{t('Orientation rose (length-weighted)', 'Rosa de orientacion (ponderada por longitud)')}</div>
              <div style={{ display: 'flex', justifyContent: 'center' }}><RoseDiagram rose={enrich.rose} /></div>
              <p className="fs-panel-sub">{t('The dominant crack direction. A single lobe = one oriented crack; a full ring = map-cracking.', 'La direccion dominante de la grieta. Un solo lobulo = una grieta orientada; un anillo completo = agrietamiento en mapa.')}</p>
            </div>
          ) : null}
        </div>
      </div>
      {widths.length ? (
        <div className="fs-panel" style={{ marginTop: '1rem' }}>
          <div className="fs-panel-t">{t('Width along the crack, w(s)', 'Ancho a lo largo de la grieta, w(s)')}</div>
          <UPlotChart
            data={[wp!.s_px, mmpp ? widths.map((x) => x * mmpp) : widths]}
            series={[{}, { label: mmpp ? 'width (mm)' : 'width (px)', stroke: 'var(--fs-quantify)', width: 2 }]}
            axes={[{ label: t('arc length s (px)', 'longitud de arco s (px)') }, { label: mmpp ? 'mm' : 'px' }]}
            scales={{ x: { time: false } }}
            height={200}
          />
          <p className="fs-panel-sub">
            {t('Inscribed-circle width traced along the crack centerline. Peak width ', 'Ancho de circulo inscrito trazado por la linea central. Ancho pico ')}
            <b>{mmpp ? `${(wMax * mmpp).toFixed(2)} mm` : `${wMax.toFixed(1)} px`}</b>.{' '}
            {t('This is the crack-opening-displacement profile the severity bands are read against.', 'Este es el perfil de apertura de grieta contra el que se leen las bandas de severidad.')}
          </p>
        </div>
      ) : null}
    </div>
  );
}

const METRIC_COLORS: Record<string, string> = {
  segformer_b2: '#cf222e', deeplabv3p_r18: '#fb8500', unet_r18: '#2da44e', hrsegnet_b16: '#8250df', dinov2s14_linear: '#0969da',
};
const METRIC_LABEL: Record<string, string> = {
  segformer_b2: 'SegFormer-B2', deeplabv3p_r18: 'DeepLabV3+', unet_r18: 'U-Net', hrsegnet_b16: 'HrSegNet', dinov2s14_linear: 'DINOv2',
};

// Metrics tab: tolerance-sweep F1 per model, confusion at 2px, ensemble disagreement. Shortlist 4-6,10.
export function MetricsTab({ enrich, es }: { enrich: Enrichment | null; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  if (!enrich || !Object.keys(enrich.models).length) {
    return (
      <Callout variant="note" title={t('No per-model metrics for this image', 'Sin metricas por modelo para esta imagen')}>
        {t('The tolerance sweep needs pixel ground truth. The classification-style samples (no mask) have none; pick a cracked image with a mask.', 'El barrido de tolerancia necesita ground truth de pixeles. Las muestras estilo clasificacion (sin mascara) no lo tienen; elige una imagen con grieta y mascara.')}
      </Callout>
    );
  }
  const models = Object.keys(enrich.models);
  const tol = enrich.models[models[0]].sweep.tol_px;
  return (
    <div>
      <p className="fs-hint" style={{ marginBottom: '0.8rem' }}>
        {t('The honest scoring: how each model behaves as the tolerance protocol changes, and how much the models disagree. The tolerance axis is exactly why two papers can report 0.85 and 0.23 for the same method.', 'La puntuacion honesta: como se comporta cada modelo al cambiar el protocolo de tolerancia, y cuanto discrepan los modelos. El eje de tolerancia es exactamente por que dos papers pueden reportar 0.85 y 0.23 para el mismo metodo.')}
      </p>
      <div className="fs-panel">
        <div className="fs-panel-t">{t('F1 vs tolerance (px) per model', 'F1 vs tolerancia (px) por modelo')}</div>
        <UPlotChart
          data={[tol, ...models.map((m) => enrich.models[m].sweep.f1)] as unknown as uPlot.AlignedData}
          series={[{}, ...models.map((m) => ({ label: METRIC_LABEL[m] ?? m, stroke: METRIC_COLORS[m] ?? '#888', width: 2, points: { show: true, size: 4 } }))]}
          axes={[{ label: t('tolerance (px)', 'tolerancia (px)') }, { label: 'F1' }]}
          scales={{ x: { time: false }, y: { range: [0, 1] } }}
          height={240}
        />
        <p className="fs-panel-sub">{t('Every model climbs as the tolerance loosens. Reading a single number without its tolerance is meaningless; Fisura always prints both 2 px and 5 px.', 'Todo modelo sube al aflojar la tolerancia. Leer un solo numero sin su tolerancia no significa nada; Fisura siempre imprime 2 px y 5 px.')}</p>
      </div>
      <div className="fs-panel" style={{ marginTop: '1rem' }}>
        <div className="fs-panel-t">{t('Confusion at 2 px tolerance (TP / FP / FN)', 'Confusion a tolerancia 2 px (TP / FP / FN)')}</div>
        <div className="fs-tablewrap">
          <table className="fs-table">
            <thead><tr><th>{t('Model', 'Modelo')}</th><th className="mono">TP</th><th className="mono">FP</th><th className="mono">FN</th><th className="mono">F1@2px</th></tr></thead>
            <tbody>
              {models.map((m) => {
                const c = enrich.models[m].confusion;
                return (
                  <tr key={m}>
                    <td><span className="fs-dot" style={{ background: METRIC_COLORS[m], display: 'inline-block', marginRight: 6 }} />{METRIC_LABEL[m] ?? m}</td>
                    <td className="mono">{c.tp}</td><td className="mono">{c.fp}</td><td className="mono">{c.fn}</td>
                    <td className="mono">{enrich.models[m].f1_2px?.toFixed(3) ?? '--'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="fs-panel-sub">{t('TP = predicted crack near true crack; FP = predicted where none is; FN = true crack missed. High FP with low FN is over-segmentation (the classical trap on texture).', 'TP = grieta predicha cerca de la real; FP = predicha donde no hay; FN = grieta real perdida. FP alto con FN bajo es sobre-segmentacion (la trampa clasica en textura).')}</p>
      </div>
      {enrich.uncertainty ? (
        <div className="fs-panel" style={{ marginTop: '1rem' }}>
          <div className="fs-panel-t">{t('Ensemble disagreement (free uncertainty)', 'Desacuerdo del ensemble (incertidumbre gratis)')}</div>
          <div className="fs-kpis fs-kpis-2">
            <div className="fs-kpi"><div className="fs-kpi-v">{enrich.uncertainty.mean_std.toFixed(3)}</div><div className="fs-kpi-l">{t('mean per-pixel stdev across models', 'stdev media por pixel entre modelos')}</div></div>
            <div className="fs-kpi"><div className="fs-kpi-v">{enrich.uncertainty.disagree_px.toLocaleString()}</div><div className="fs-kpi-l">{t('pixels where the models disagree', 'pixeles donde los modelos discrepan')}</div></div>
          </div>
          <p className="fs-panel-sub">
            {t('Where the ', 'Donde los ')}{enrich.uncertainty.n_models}
            {t(' learned models disagree is exactly where a mask is least trustworthy, a free uncertainty signal at no extra training cost.', ' modelos aprendidos discrepan es justo donde una mascara es menos confiable, una senal de incertidumbre gratis sin costo extra de entrenamiento.')}
          </p>
        </div>
      ) : null}
    </div>
  );
}
