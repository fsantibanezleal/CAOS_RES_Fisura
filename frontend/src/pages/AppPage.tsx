import { useEffect, useMemo, useState } from 'react';
import { Callout, Tabs } from '@fasl-work/caos-app-shell';
import { heatUrl, loadCaseArtifact, overlayUrl } from '../api/artifacts';
import type { ArtifactSample, CaseArtifact, LevelRecord } from '../lib/contract.types';
import { useT } from '../lib/i18n';
import { MaskCanvas } from '../render/MaskCanvas';
import { HeatCanvas } from '../render/HeatCanvas';
import { PanelBoundary } from '../render/PanelBoundary';
import { UPlotChart } from '../render/UPlotChart';

// The App workbench (ADR-0017 section 3). The LEFT COLUMN is parametrization: pick the data source
// (synthetic knobs / real cases / your own image), pick the sample, set the controls that drive every
// method, and read a live diagnosis. The TOP TABS are the METHODS, in ladder order: Classical, SOTA
// (learned), Beyond SOTA (anomaly), Quantification, Context. Every method tab shows its prediction ON
// THE IMAGE with a value read-out. No method is a table (tables live on Benchmark).

type Source = 'real' | 'synthetic' | 'byoi';

// Each real/synthetic case pulls three artifacts that share sample_ids, one per method family.
const CLASSICAL: Record<Source, string> = { real: 'bcl_examples', synthetic: 'synthetic_battery', byoi: 'bcl_examples' };
const LEARNED_SLUG = 'learned_on_examples'; // SOTA masks, only for the real example set
const ANOMALY_SLUG = 'anomaly_examples';    // Beyond-SOTA heat, only for the real example set

const CLASSICAL_LEVELS = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'];
const CLASSICAL_LABEL: Record<string, [string, string]> = {
  L0: ['L0 Otsu floor', 'L0 piso Otsu'],
  L1: ['L1 Sauvola', 'L1 Sauvola'],
  L2: ['L2 oriented top-hat', 'L2 top-hat orientado'],
  L3: ['L3 Hessian ridge', 'L3 cresta Hessiana'],
  L4: ['L4 path bridging', 'L4 puente de caminos'],
  L5: ['L5 RF fusion', 'L5 fusión RF'],
};
const ARCH_LABEL: Record<string, [string, string]> = {
  unet_r18: ['U-Net (R18)', 'U-Net (R18)'],
  deeplabv3p_r18: ['DeepLabV3+ (R18)', 'DeepLabV3+ (R18)'],
  segformer_b2: ['SegFormer-B2', 'SegFormer-B2'],
  hrsegnet_b16: ['HrSegNet-B16', 'HrSegNet-B16'],
  dinov2s14_linear: ['DINOv2 linear probe', 'sonda lineal DINOv2'],
};

export default function AppPage() {
  const t = useT();
  const es = t('x', 'y') === 'y';

  const [source, setSource] = useState<Source>('real');
  const [classical, setClassical] = useState<CaseArtifact | null>(null);
  const [learned, setLearned] = useState<CaseArtifact | null>(null);
  const [anomaly, setAnomaly] = useState<CaseArtifact | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // shared controls that drive every method view
  const [showGt, setShowGt] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  // per-method parameters (the left column controls the tabs)
  const [level, setLevel] = useState('L3');
  const [arch, setArch] = useState('segformer_b2');
  const [heatOpacity, setHeatOpacity] = useState(0.7);

  useEffect(() => {
    setError(null);
    setClassical(null);
    setLearned(null);
    setAnomaly(null);
    loadCaseArtifact(CLASSICAL[source])
      .then((c) => {
        setClassical(c);
        setSampleId(c.samples[0]?.sample_id ?? null);
      })
      .catch((e) => setError(String(e)));
    // SOTA + Beyond-SOTA masks exist for the real example set; synthetic has classical only
    if (source === 'real' || source === 'byoi') {
      loadCaseArtifact(LEARNED_SLUG).then(setLearned).catch(() => setLearned(null));
      loadCaseArtifact(ANOMALY_SLUG).then(setAnomaly).catch(() => setAnomaly(null));
    }
  }, [source]);

  const cSample: ArtifactSample | null = useMemo(
    () => classical?.samples.find((s) => s.sample_id === sampleId) ?? null,
    [classical, sampleId],
  );
  const lSample = useMemo(() => learned?.samples.find((s) => s.sample_id === sampleId) ?? null, [learned, sampleId]);
  const aSample = useMemo(() => anomaly?.samples.find((s) => s.sample_id === sampleId) ?? null, [anomaly, sampleId]);

  const imageUrl = cSample?.overlays_rel ? overlayUrl(cSample.overlays_rel, '_image.png') : null;

  // live diagnosis: the best classical F1 vs the best SOTA F1 vs the anomaly verdict, for the readout
  const diag = useMemo(() => {
    const cF1 = cSample ? bestF1(cSample.levels, CLASSICAL_LEVELS) : null;
    const lF1 = lSample ? bestF1(lSample.levels, Object.keys(ARCH_LABEL)) : null;
    const aScore = aSample?.levels.patchcore?.anomaly_score_norm ?? null;
    return { cF1, lF1, aScore };
  }, [cSample, lSample, aSample]);

  if (error) {
    return (
      <div className="page-body">
        <Callout variant="honest" title={t('Artifacts unavailable', 'Artefactos no disponibles')}>
          {t('The committed artifacts could not be loaded: ', 'No se pudieron cargar los artefactos versionados: ')}
          <code>{error}</code>
        </Callout>
      </div>
    );
  }

  const activeLevel = cSample && cSample.levels[level] ? level : 'L3';
  const activeArch = lSample && lSample.levels[arch] ? arch : 'segformer_b2';

  return (
    <div className="page-body fs-app-layout">
      <aside className="fs-side">
        <div className="fs-side-h">{t('Data and controls', 'Datos y controles')}</div>

        <div className="fs-ctl">
          <span>{t('Data source', 'Fuente de datos')}</span>
          <div className="fs-seg">
            <button className={`fs-seg-b ${source === 'synthetic' ? 'on' : ''}`} onClick={() => setSource('synthetic')}>
              {t('Synthetic', 'Sintético')}
            </button>
            <button className={`fs-seg-b ${source === 'real' ? 'on' : ''}`} onClick={() => setSource('real')}>
              {t('Real cases', 'Casos reales')}
            </button>
            <button className={`fs-seg-b ${source === 'byoi' ? 'on' : ''}`} onClick={() => setSource('byoi')}>
              {t('Your image', 'Tu imagen')}
            </button>
          </div>
          <p className="fs-hint">
            {source === 'synthetic'
              ? t('Generated cracks with EXACT ground truth (width and centerline known by construction).', 'Grietas generadas con ground truth EXACTO (ancho y línea central conocidos por construcción).')
              : source === 'real'
                ? t('Open-licensed concrete and steel patches (BCL CC0, SDNET2018 CC BY).', 'Parches de hormigón y acero con licencia abierta (BCL CC0, SDNET2018 CC BY).')
                : t('Bring your own photo: the in-browser lane arrives with BL-013; for now this loads the real example set so every method is visible.', 'Trae tu propia foto: el carril en el navegador llega con BL-013; por ahora carga el set real de ejemplos para que cada método sea visible.')}
          </p>
        </div>

        <div className="fs-ctl">
          <span>{t('Sample', 'Muestra')}</span>
          <select className="fs-sel" value={sampleId ?? ''} onChange={(e) => setSampleId(e.target.value)}>
            {classical?.samples.map((s) => (
              <option key={s.sample_id} value={s.sample_id}>
                {s.sample_id} ({s.material})
              </option>
            ))}
          </select>
        </div>

        <div className="fs-ctl">
          <span title={t('The classical ladder rung shown in the Classical tab', 'El peldaño clásico mostrado en la pestaña Clásico')}>
            {t('Classical rung', 'Peldaño clásico')}
          </span>
          <div className="fs-chips">
            {(cSample ? Object.keys(cSample.levels).filter((l) => CLASSICAL_LEVELS.includes(l)) : CLASSICAL_LEVELS).map((l) => (
              <button key={l} className={`chip ${l === activeLevel ? 'on' : ''}`} onClick={() => setLevel(l)} title={t(...(CLASSICAL_LABEL[l] ?? [l, l]))}>
                {l}
              </button>
            ))}
          </div>
        </div>

        {(source === 'real' || source === 'byoi') && (
          <div className="fs-ctl">
            <span title={t('The learned architecture shown in the SOTA tab', 'La arquitectura aprendida mostrada en la pestaña SOTA')}>
              {t('SOTA model', 'Modelo SOTA')}
            </span>
            <select className="fs-sel" value={activeArch} onChange={(e) => setArch(e.target.value)}>
              {Object.keys(ARCH_LABEL).map((a) => (
                <option key={a} value={a}>{t(...ARCH_LABEL[a])}</option>
              ))}
            </select>
          </div>
        )}

        <div className="fs-ctl">
          <label className="fs-ctl-row">
            <span>{t('Show ground truth', 'Mostrar ground truth')}</span>
            <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
          </label>
          <label>
            <span className="fs-ctl-row">
              <span>{t('Overlay opacity', 'Opacidad del overlay')}</span>
              <b>{opacity.toFixed(2)}</b>
            </span>
            <input className="range" type="range" min={0.1} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          </label>
          <label>
            <span className="fs-ctl-row">
              <span>{t('Anomaly heat opacity', 'Opacidad de calor de anomalía')}</span>
              <b>{heatOpacity.toFixed(2)}</b>
            </span>
            <input className="range" type="range" min={0.1} max={1} step={0.05} value={heatOpacity} onChange={(e) => setHeatOpacity(Number(e.target.value))} />
          </label>
        </div>

        <div className="fs-readout">
          <div className="fs-readout-h">{t('Live diagnosis', 'Diagnóstico en vivo')}</div>
          <ReadoutRow label={t('Classical best F1 @ 2px', 'Clásico mejor F1 @ 2px')} value={diag.cF1} tone="classical" />
          <ReadoutRow label={t('SOTA best F1 @ 2px', 'SOTA mejor F1 @ 2px')} value={diag.lF1} tone="learned" />
          <ReadoutRow label={t('Anomaly score (0..1)', 'Puntaje de anomalía (0..1)')} value={diag.aScore} tone="anomaly" />
          <p className="fs-hint">
            {diag.cF1 != null && diag.lF1 != null
              ? diag.lF1 > diag.cF1
                ? t('On this sample the learned model beats the classical ladder by ', 'En esta muestra el modelo aprendido supera a la escalera clásica por ') + (diag.lF1 - diag.cF1).toFixed(2) + t(' F1.', ' de F1.')
                : t('On this sample the classical ladder holds up: learned gains only ', 'En esta muestra la escalera clásica aguanta: lo aprendido gana solo ') + (diag.lF1 - diag.cF1).toFixed(2) + t(' F1.', ' de F1.')
              : t('Select a sample with a pixel mask to compare methods.', 'Selecciona una muestra con máscara de píxeles para comparar métodos.')}
          </p>
        </div>
      </aside>

      <main className="fs-main">
        <PanelBoundary label={t('Workbench', 'Banco de trabajo')} es={es}>
          {cSample ? (
            <Tabs
              ariaLabel="methods"
              initial="classical"
              tabs={[
                {
                  id: 'classical',
                  label: t('Classical', 'Clásico'),
                  content: <MethodView title={t(...(CLASSICAL_LABEL[activeLevel] ?? [activeLevel, activeLevel]))} image={imageUrl} sample={cSample} level={cSample.levels[activeLevel]} showGt={showGt} opacity={opacity} es={es} kind="classical" />,
                },
                {
                  id: 'sota',
                  label: t('SOTA (learned)', 'SOTA (aprendido)'),
                  content: lSample && lSample.levels[activeArch]
                    ? <MethodView title={t(...(ARCH_LABEL[activeArch] ?? [activeArch, activeArch]))} image={imageUrl} sample={lSample} level={lSample.levels[activeArch]} showGt={showGt} opacity={opacity} es={es} kind="sota" />
                    : <NoData es={es} what={t('learned masks', 'máscaras aprendidas')} />,
                },
                {
                  id: 'beyond',
                  label: t('Beyond SOTA (anomaly)', 'Más allá de SOTA (anomalía)'),
                  content: aSample && aSample.levels.patchcore
                    ? <AnomalyView image={imageUrl} sample={aSample} showGt={showGt} heatOpacity={heatOpacity} es={es} />
                    : <NoData es={es} what={t('anomaly maps', 'mapas de anomalía')} />,
                },
                {
                  id: 'quant',
                  label: t('Quantification', 'Cuantificación'),
                  content: <QuantView sample={cSample} es={es} />,
                },
                {
                  id: 'context',
                  label: t('Context', 'Contexto'),
                  content: <ContextView source={source} sample={cSample} es={es} />,
                },
              ]}
            />
          ) : (
            <div className="fs-panel"><div className="fs-panel-t">{t('Loading artifacts...', 'Cargando artefactos...')}</div></div>
          )}
        </PanelBoundary>
      </main>
    </div>
  );
}

function bestF1(levels: Record<string, LevelRecord>, keys: string[]): number | null {
  let best: number | null = null;
  for (const k of keys) {
    const f = levels[k]?.segmentation?.tol2px.f1;
    if (typeof f === 'number' && (best === null || f > best)) best = f;
  }
  return best;
}

function ReadoutRow({ label, value, tone }: { label: string; value: number | null; tone: string }) {
  return (
    <div className="fs-readout-row">
      <span className={`fs-dot ${tone}`} />
      <span className="fs-readout-l">{label}</span>
      <span className="fs-readout-v">{value != null ? value.toFixed(3) : '--'}</span>
    </div>
  );
}

function NoData({ es, what }: { es: boolean; what: string }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  return (
    <Callout variant="note" title={t('Not available for this source', 'No disponible para esta fuente')}>
      {t('This sample has no ', 'Esta muestra no tiene ')}{what}{t('. The synthetic battery runs the classical ladder and quantification only; the learned and anomaly methods run on the real example set (switch the data source to Real cases).', '. La batería sintética corre solo la escalera clásica y la cuantificación; los métodos aprendidos y de anomalía corren sobre el set real (cambia la fuente a Casos reales).')}
    </Callout>
  );
}

// A single method's PREDICTION on the image + its value read-out. Classical and SOTA share this shape.
function MethodView({ title, image, sample, level, showGt, opacity, es, kind }: {
  title: string; image: string | null; sample: ArtifactSample; level: LevelRecord;
  showGt: boolean; opacity: number; es: boolean; kind: 'classical' | 'sota';
}) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const seg = level.segmentation;
  return (
    <div className="fs-method">
      <div className="fs-method-view">
        <div className="fs-panel-t">{title} · {sample.sample_id}</div>
        <MaskCanvas imageUrl={image} size={sample.size} mask={level.mask_rle} gt={sample.gt_rle} showGt={showGt} opacity={opacity} />
        <p className="fs-panel-sub">
          {t('Prediction in red; ground truth in green (overlap reads yellow). Decoded client-side from the committed RLE.', 'Predicción en rojo; ground truth en verde (el solape se lee amarillo). Decodificado en el cliente desde el RLE versionado.')}
        </p>
        {level.notes.map((n, i) => <p key={i} className="fs-note">{n}</p>)}
      </div>
      <div className="fs-method-read">
        {seg ? (
          <>
            <div className="fs-kpis fs-kpis-2">
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.f1.toFixed(3)}</div><div className="fs-kpi-l">F1 @ 2 px</div></div>
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol5px.f1.toFixed(3)}</div><div className="fs-kpi-l">F1 @ 5 px</div></div>
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.precision.toFixed(3)}</div><div className="fs-kpi-l">{t('precision @ 2px', 'precisión @ 2px')}</div></div>
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.recall.toFixed(3)}</div><div className="fs-kpi-l">{t('recall @ 2px', 'recall @ 2px')}</div></div>
            </div>
            <p className="fs-hint">{t('Protocol travels with the number: ', 'El protocolo viaja con el número: ')}{seg.protocol}</p>
          </>
        ) : (
          <Callout variant="note" title={t('No pixel ground truth', 'Sin ground truth de píxeles')}>
            {t('This is a classification-style sample; segmentation F1 is not defined. The prediction is still shown on the image.', 'Esta es una muestra estilo clasificación; el F1 de segmentación no está definido. La predicción se muestra igual sobre la imagen.')}
          </Callout>
        )}
        {kind === 'sota' && (
          <p className="fs-hint">{t('This mask is the trained model run offline on this exact image and baked to an artifact; the same ONNX runs in the browser in the live lane.', 'Esta máscara es el modelo entrenado corrido offline sobre esta imagen exacta y horneado a un artefacto; el mismo ONNX corre en el navegador en el carril en vivo.')}</p>
        )}
      </div>
    </div>
  );
}

// Beyond-SOTA: the anomaly heatmap ON the image, plus the honest anomaly score.
function AnomalyView({ image, sample, showGt, heatOpacity, es }: {
  image: string | null; sample: ArtifactSample; showGt: boolean; heatOpacity: number; es: boolean;
}) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const lv = sample.levels.patchcore;
  return (
    <div className="fs-method">
      <div className="fs-method-view">
        <div className="fs-panel-t">{t('PatchCore anomaly heat', 'Calor de anomalía PatchCore')} · {sample.sample_id}</div>
        <HeatCanvas imageUrl={image} heatUrl={sample.heat_rel ? heatUrl(sample.heat_rel) : null} size={sample.size} mask={lv.mask_rle} gt={sample.gt_rle} showGt={showGt} showHeat heatOpacity={heatOpacity} />
        <p className="fs-panel-sub">
          {t('Warm = far from the uncracked memory bank (more anomalous). White outline = region above 0.6 of the shared scale. Green = ground truth crack.', 'Cálido = lejos del banco de memoria sin grietas (más anómalo). Contorno blanco = región sobre 0.6 de la escala compartida. Verde = grieta ground truth.')}
        </p>
        {lv.notes.map((n, i) => <p key={i} className="fs-note">{n}</p>)}
      </div>
      <div className="fs-method-read">
        <div className="fs-kpis fs-kpis-2">
          <div className="fs-kpi"><div className="fs-kpi-v">{lv.anomaly_score_norm?.toFixed(2) ?? '--'}</div><div className="fs-kpi-l">{t('anomaly score (0..1)', 'puntaje anomalía (0..1)')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{lv.anomaly_score?.toFixed(2) ?? '--'}</div><div className="fs-kpi-l">{t('raw kNN distance', 'distancia kNN cruda')}</div></div>
        </div>
        <Callout variant="honest" title={t('What this method is, honestly', 'Qué es este método, con honestidad')}>
          {t('The memory bank saw only UNCRACKED concrete. A high score means the surface looks unlike healthy concrete, not necessarily a crack. Across the concrete-transfer study this reaches image AUROC 0.72, far below the 0.996 the same method reaches on industrial MVTec AD. It is a screen, not a detector.', 'El banco de memoria vio solo hormigón SIN grietas. Un puntaje alto significa que la superficie no se parece al hormigón sano, no necesariamente una grieta. En el estudio de transferencia esto alcanza AUROC por imagen de 0.72, muy por debajo del 0.996 que el mismo método alcanza en el industrial MVTec AD. Es un tamiz, no un detector.')}
        </Callout>
      </div>
    </div>
  );
}

function QuantView({ sample, es }: { sample: ArtifactSample; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const g = sample.geometry;
  const w = g.width;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      <div className="fs-kpis">
        <div className="fs-kpi"><div className="fs-kpi-v">{w.edt_median?.toFixed(2) ?? 'n/a'}</div><div className="fs-kpi-l">{t('width median, inscribed circle (px)', 'ancho mediano, círculo inscrito (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{w.profile_median?.toFixed(2) ?? 'n/a'}</div><div className="fs-kpi-l">{t('width median, orthogonal profile (px)', 'ancho mediano, perfil ortogonal (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{g.length_px.toFixed(0)}</div><div className="fs-kpi-l">{t('skeleton length (px)', 'largo del esqueleto (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{g.n_branches}</div><div className="fs-kpi-l">{t('branch points', 'puntos de rama')}</div></div>
      </div>
      <div className="fs-panel">
        <div className="fs-panel-t">{t('Orientation histogram (10-degree bins over 0..180)', 'Histograma de orientación (bins de 10 grados sobre 0..180)')}</div>
        <UPlotChart
          data={[g.orientation_hist.map((_, i) => i * 10 + 5), g.orientation_hist]}
          series={[{}, { label: t('skeleton px', 'px de esqueleto'), stroke: '#8250df', width: 2, points: { show: true, size: 5 } }]}
          axes={[{ label: t('angle (deg)', 'ángulo (grados)') }, {}]}
          scales={{ x: { time: false } }}
          height={180}
        />
      </div>
      {sample.width_mm ? (
        <div className="fs-kpis">
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.width_mm.median.toFixed(2)}</div><div className="fs-kpi-l">{t('width median (mm)', 'ancho mediano (mm)')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.width_mm.p95.toFixed(2)}</div><div className="fs-kpi-l">{t('width p95 (mm)', 'ancho p95 (mm)')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.width_mm.mm_per_px.toFixed(3)}</div><div className="fs-kpi-l">mm/px</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{(g.length_px * sample.width_mm.mm_per_px).toFixed(0)}</div><div className="fs-kpi-l">{t('length (mm)', 'largo (mm)')}</div></div>
        </div>
      ) : null}
      {sample.width_validation ? (
        <div className="fs-panel">
          <div className="fs-panel-t">{t('Width validation against exact synthetic truth', 'Validación de ancho contra verdad sintética exacta')}</div>
          <div className="fs-tablewrap">
            <table className="fs-table">
              <tbody>
                <tr><td>{t('true mask width (median)', 'ancho verdadero (mediana)')}</td><td className="mono">{sample.width_validation.true_width_px.toFixed(2)} px</td></tr>
                <tr><td>{t('inscribed-circle on GT', 'círculo inscrito sobre GT')}</td><td className="mono">{sample.width_validation.edt_on_gt_median?.toFixed(2) ?? 'n/a'} px</td></tr>
                <tr><td>{t('absolute error', 'error absoluto')}</td><td className="mono">{sample.width_validation.edt_abs_error?.toFixed(3) ?? 'n/a'} px</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      ) : null}
      {sample.severity ? (
        <div className="fs-panel">
          <div className="fs-panel-t">{t('Severity CONTEXT: measured width vs published guidance', 'CONTEXTO de severidad: ancho medido vs guías publicadas')}</div>
          <div className="fs-tablewrap">
            <table className="fs-table">
              <thead><tr><th>{t('Source', 'Fuente')}</th><th>{t('Exposure', 'Exposición')}</th><th className="mono">{t('limit (mm)', 'límite (mm)')}</th><th>{t('median', 'mediana')}</th></tr></thead>
              <tbody>
                {sample.severity.bands.map((b, i) => (
                  <tr key={i}><td>{b.source}</td><td>{b.exposure}</td><td className="mono">{b.limit_mm.toFixed(2)}</td>
                    <td><span className={`fs-badge ${b.median_within ? 'real' : 'tr-monitor'}`}>{b.median_within ? t('within', 'dentro') : t('exceeds', 'excede')}</span></td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="fs-panel-sub"><b>{sample.severity.framing}</b></p>
        </div>
      ) : null}
    </div>
  );
}

function ContextView({ source, sample, es }: { source: Source; sample: ArtifactSample; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const steel = sample.material === 'steel';
  return (
    <div className="fs-doc" style={{ maxWidth: '46rem', margin: 0 }}>
      <h3>{t('What you are comparing', 'Qué estás comparando')}</h3>
      <p>
        {t('Every tab runs a different method on the SAME image, so the difference you see is the method, not the picture. Classical is the transparent L0-L5 ladder (a global Otsu threshold up to a Hessian-ridge plus path-bridging fusion). SOTA is a trained network (U-Net, DeepLabV3+, SegFormer-B2, HrSegNet, or a DINOv2 linear probe) run offline and baked. Beyond SOTA is unsupervised anomaly detection: a memory bank fit on uncracked concrete only, which never saw a crack.',
          'Cada pestaña corre un método distinto sobre la MISMA imagen, así que la diferencia que ves es el método, no la foto. Clásico es la escalera transparente L0-L5 (desde un umbral global de Otsu hasta una fusión de cresta Hessiana con puente de caminos). SOTA es una red entrenada (U-Net, DeepLabV3+, SegFormer-B2, HrSegNet, o una sonda lineal DINOv2) corrida offline y horneada. Más allá de SOTA es detección de anomalías no supervisada: un banco de memoria ajustado solo sobre hormigón sin grietas, que nunca vio una grieta.')}
      </p>
      <h3>{t('This sample', 'Esta muestra')}</h3>
      <p>
        {source === 'synthetic'
          ? t('A generated crack with exact ground truth: the centerline, mask and width are known by construction, so this is the regression gate of the classical stack and the reference for the width estimators (inscribed-circle lands within about 0.2 px of the true width here).',
              'Una grieta generada con ground truth exacto: la línea central, la máscara y el ancho se conocen por construcción, así que esta es la compuerta de regresión del stack clásico y la referencia de los estimadores de ancho (el círculo inscrito cae a unos 0.2 px del ancho verdadero aquí).')
          : steel
            ? t('A steel surface full of scratches: the hardest case for every method, because scratch texture mimics cracks. Watch precision fall here even for the learned models.',
                'Una superficie de acero llena de rayas: el caso más difícil para todo método, porque la textura de rayas imita grietas. Observa cómo cae la precisión aquí incluso para los modelos aprendidos.')
            : t('A real concrete or pavement patch with weak contrast and genuine texture. The thin clean crack reaches high F1 mid-ladder; the wide diffuse crack and the uncracked controls show where each method false-fires.',
                'Un parche real de hormigón o pavimento con contraste débil y textura genuina. La grieta fina y limpia alcanza F1 alto a mitad de escalera; la grieta ancha y difusa y los controles sin grieta muestran dónde cada método dispara en falso.')}
      </p>
      <h3>{t('How to read it', 'Cómo leerlo')}</h3>
      <p>
        {t('Toggle ground truth to see where each method agrees (yellow) and misses. Sweep the classical rung and the SOTA model in the left column and watch the prediction change on the image and the F1 change in the read-out. The live diagnosis in the sidebar tells you, per sample, whether the learned model actually earns its complexity over the classical ladder.',
          'Activa el ground truth para ver dónde cada método acierta (amarillo) y falla. Barre el peldaño clásico y el modelo SOTA en la columna izquierda y observa cómo cambia la predicción sobre la imagen y el F1 en la lectura. El diagnóstico en vivo de la barra lateral te dice, por muestra, si el modelo aprendido realmente justifica su complejidad frente a la escalera clásica.')}
      </p>
    </div>
  );
}
