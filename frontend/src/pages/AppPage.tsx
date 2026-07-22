import { useEffect, useMemo, useState } from 'react';
import { Callout, Tabs } from '@fasl-work/caos-app-shell';
import {
  heatUrl, loadCaseArtifact, loadDinoPca, loadEnrichment, loadGradCam, loadWorkbench, overlayUrl, workbenchUrl,
  type DinoPca, type Enrichment, type GradCam, type WorkbenchIndex, type WorkbenchSample,
} from '../api/artifacts';
import type { ArtifactSample, CaseArtifact, LevelRecord } from '../lib/contract.types';
import { useT } from '../lib/i18n';
import { MethodTile } from '../render/MethodTile';
import { OverlayLegend, predictionLegend } from '../render/OverlayLegend';
import { PanelBoundary } from '../render/PanelBoundary';
import { UPlotChart } from '../render/UPlotChart';
import { MetricsTab, QuantTab } from './workbench/QuantMetricsTabs';
import { LiveLane } from './workbench/LiveLane';

// The App is a per-case interactive WORKBENCH (Felipe's spec, 2026-07-20):
//   LEFT COLUMN: pick the case source (real samples / synthetic battery / upload your own), the image,
//     and set the PARAMETERS that drive the tabs (e.g. the SLIC superpixel count + compactness).
//   TABS: navigate rich, interactive views of the methods ON the selected image, in order:
//     Overview -> Preprocessing -> Semantic segmentation -> SLIC (param-driven) -> Classical ->
//     SOTA -> Beyond SOTA -> Summary. Every method is VISIBLE ON THE IMAGE; nothing is a bare table.

// The case source is a DATA axis: which images the whole workbench runs on. It is deliberately NOT a
// method axis. An earlier "Prebaked / Pretrained" split tried to separate baked classical outputs from
// trained-model outputs, but that is exactly what the TABS already do (Classical vs SOTA vs Beyond
// SOTA), so both modes loaded the same artifact and rendered the same thing: a control that changed one
// sentence of copy and nothing else.
type Source = 'real' | 'synthetic' | 'fundus' | 'upload';
const CLASSICAL_SLUG = 'bcl_examples';
const LEARNED_SLUG = 'learned_on_examples';
// The synthetic samples whose overlay PNGs are baked. The battery holds 12 cases and every one of them
// carries per-level metrics, but only these four were rendered to imagery; the degradation view below
// uses all 12, the thumbnails only these.
// Which committed sources feed each case-source button. The generated battery and the FIVES fundus
// images are materialized into data/examples like every other sample, so they live in the SAME
// artifacts and every tab covers them; this is only a filter over one list.
const SOURCE_FILTER: Record<'real' | 'synthetic' | 'fundus', (src: string) => boolean> = {
  real: (src) => src === 'bcl' || src === 'sdnet2018',
  synthetic: (src) => src === 'fisuralab-synthetic',
  fundus: (src) => src === 'fives',
};
const ANOMALY_SLUG = 'anomaly_examples';

type Family = 'classical' | 'learned' | 'anomaly';
interface MethodDef { id: string; family: Family; label: string; en: string; es: string; }

const CLASSICAL_METHODS: MethodDef[] = [
  { id: 'L0', family: 'classical', label: 'L0', en: 'Otsu global floor', es: 'Piso global de Otsu' },
  { id: 'L1', family: 'classical', label: 'L1', en: 'Sauvola local threshold', es: 'Umbral local Sauvola' },
  { id: 'L2', family: 'classical', label: 'L2', en: 'Oriented top-hat + hysteresis', es: 'Top-hat orientado + histéresis' },
  { id: 'L3', family: 'classical', label: 'L3', en: 'Hessian ridge (sato)', es: 'Cresta Hessiana (sato)' },
  { id: 'L4', family: 'classical', label: 'L4', en: 'Minimal-path bridging', es: 'Puente de caminos mínimos' },
  { id: 'L5', family: 'classical', label: 'L5', en: 'Random-forest fusion', es: 'Fusión random-forest' },
];
const LEARNED_METHODS: MethodDef[] = [
  { id: 'segformer_b2', family: 'learned', label: 'SegFormer-B2', en: 'Transformer segmenter (CrackSeg9k)', es: 'Segmentador transformer (CrackSeg9k)' },
  { id: 'deeplabv3p_r18', family: 'learned', label: 'DeepLabV3+', en: 'ASPP encoder-decoder (R18)', es: 'Encoder-decoder ASPP (R18)' },
  { id: 'unet_r18', family: 'learned', label: 'U-Net', en: 'Classic encoder-decoder (R18)', es: 'Encoder-decoder clásico (R18)' },
  { id: 'hrsegnet_b16', family: 'learned', label: 'HrSegNet', en: 'Real-time crack net (in-repo)', es: 'Red de grietas en tiempo real (en repo)' },
  { id: 'dinov2s14_linear', family: 'learned', label: 'DINOv2', en: 'Frozen ViT-S/14 + linear head', es: 'ViT-S/14 congelado + cabeza lineal' },
];
const ANOMALY_METHODS: MethodDef[] = [
  { id: 'patchcore', family: 'anomaly', label: 'PatchCore', en: 'Memory bank on uncracked concrete', es: 'Banco de memoria en hormigón sano' },
];

const FAMILY_TONE: Record<Family, string> = { classical: 'var(--fs-classical)', learned: 'var(--fs-learned)', anomaly: 'var(--fs-anomaly)' };
const FAMILY_RGB: Record<Family, [number, number, number]> = { classical: [47, 129, 247], learned: [163, 113, 247], anomaly: [219, 109, 40] };
const rgbStr = (c: [number, number, number]) => `rgb(${c[0]},${c[1]},${c[2]})`;
const FAMILY_LABEL: Record<Family, [string, string]> = {
  classical: ['Classical ladder', 'Escalera clásica'],
  learned: ['SOTA learned', 'Aprendido SOTA'],
  anomaly: ['Beyond SOTA (anomaly)', 'Más allá de SOTA (anomalía)'],
};

export default function AppPage() {
  const t = useT();
  const es = t('x', 'y') === 'y';

  const [source, setSource] = useState<Source>('real');
  const [classical, setClassical] = useState<CaseArtifact | null>(null);
  const [learned, setLearned] = useState<CaseArtifact | null>(null);
  const [anomaly, setAnomaly] = useState<CaseArtifact | null>(null);
  const [wb, setWb] = useState<WorkbenchIndex | null>(null);
  const [enrich, setEnrich] = useState<Enrichment | null>(null);
  const [dinoPca, setDinoPca] = useState<DinoPca | null>(null);
  const [gradCam, setGradCam] = useState<GradCam | null>(null);
  // SOTA tab view mode: the prediction mask, the frozen-feature PCA, or the Grad-CAM evidence map
  const [sotaView, setSotaView] = useState<'pred' | 'features' | 'cam'>('pred');
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // shared + per-tab parameters (the left column drives the tabs)
  const [showGt, setShowGt] = useState(true);
  const [opacity, setOpacity] = useState(0.6);
  const [slicN, setSlicN] = useState(150);
  const [slicC, setSlicC] = useState(10);
  const [detail, setDetail] = useState('segformer_b2'); // the method enlarged in Summary

  const classicalSlug = CLASSICAL_SLUG;

  useEffect(() => {
    setError(null);
    loadCaseArtifact(classicalSlug).then((c) => { setClassical(c); setSampleId(c.samples[0]?.sample_id ?? null); }).catch((e) => setError(String(e)));
    loadCaseArtifact(LEARNED_SLUG).then(setLearned).catch(() => setLearned(null));
    loadCaseArtifact(ANOMALY_SLUG).then(setAnomaly).catch(() => setAnomaly(null));
    loadWorkbench().then(setWb).catch(() => setWb(null));
    loadDinoPca().then(setDinoPca).catch(() => setDinoPca(null));
    loadGradCam().then(setGradCam).catch(() => setGradCam(null));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!sampleId) return;
    setEnrich(null);
    loadEnrichment(sampleId).then(setEnrich).catch(() => setEnrich(null));
  }, [sampleId]);

  // The synthetic battery is baked WITHOUT overlays_rel, but its overlay files follow the same
  // "<case>/overlays/<sample_id>" + "_L0.png" / "_image.png" / "_gt.png" convention as the real set,
  // so the path is derived here rather than re-baking the artifact. Only the samples whose overlays
  // were actually rendered are offered: the rest carry metrics but no imagery, and a thumbnail that
  // 404s would be a worse lie than not listing it.
  const shownSamples = useMemo(() => {
    const all = classical?.samples ?? [];
    if (source === 'upload') return all;
    return all.filter((s) => SOURCE_FILTER[source](s.source));
  }, [classical, source]);

  // keep the selected image valid when the source changes
  useEffect(() => {
    if (!shownSamples.length) return;
    if (!shownSamples.some((s) => s.sample_id === sampleId)) setSampleId(shownSamples[0].sample_id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [source, shownSamples.length]);

  const cSample = useMemo(
    () => classical?.samples.find((s) => s.sample_id === sampleId) ?? null,
    [classical, sampleId],
  );
  const lSample = useMemo(() => learned?.samples.find((s) => s.sample_id === sampleId) ?? null, [learned, sampleId]);
  const aSample = useMemo(() => anomaly?.samples.find((s) => s.sample_id === sampleId) ?? null, [anomaly, sampleId]);
  const wbSample: WorkbenchSample | null = sampleId ? wb?.samples[sampleId] ?? null : null;
  const imageUrl = cSample?.overlays_rel ? overlayUrl(cSample.overlays_rel, '_image.png') : null;

  if (error) {
    return (
      <div className="page-body">
        <Callout variant="honest" title={t('Artifacts unavailable', 'Artefactos no disponibles')}>
          {t('The committed artifacts could not be loaded: ', 'No se pudieron cargar los artefactos versionados: ')}<code>{error}</code>
        </Callout>
      </div>
    );
  }

  // snap the SLIC sliders to the nearest baked grid variant
  const gridN = wb?.slic_grid.n_segments ?? [150];
  const gridC = wb?.slic_grid.compactness ?? [10];
  const snapN = gridN.reduce((a, b) => (Math.abs(b - slicN) < Math.abs(a - slicN) ? b : a), gridN[0]);
  const snapC = gridC.reduce((a, b) => (Math.abs(b - slicC) < Math.abs(a - slicC) ? b : a), gridC[0]);

  return (
    <div className="page-body fs-app-layout">
      <aside className="fs-side">
        <div className="fs-side-h">{t('1 · Case source', '1 · Fuente del caso')}</div>
        <div className="fs-seg">
          <button className={`fs-seg-b ${source === 'real' ? 'on' : ''}`} onClick={() => setSource('real')}>{t('Real samples', 'Muestras reales')}</button>
          <button className={`fs-seg-b ${source === 'synthetic' ? 'on' : ''}`} onClick={() => setSource('synthetic')}>{t('Synthetic', 'Sintético')}</button>
          <button className={`fs-seg-b ${source === 'fundus' ? 'on' : ''}`} onClick={() => setSource('fundus')}>{t('Fundus', 'Fondo de ojo')}</button>
          <button className={`fs-seg-b ${source === 'upload' ? 'on' : ''}`} onClick={() => setSource('upload')}>{t('Upload', 'Subir')}</button>
        </div>
        <p className="fs-hint">
          {source === 'real'
            ? t('Open-licensed concrete and steel photographs, with hand-labelled pixel ground truth. Every method in the tabs runs on the image you pick.', 'Fotografías de hormigón y acero con licencia abierta, con ground truth de píxeles etiquetado a mano. Cada método de las pestañas corre sobre la imagen que elijas.')
            : source === 'fundus'
              ? t('Retinal fundus photographs (FIVES, CC BY 4.0), carried as a control on the thesis behind this lab. A vessel and a crack are the same kind of object to a camera, so the whole ladder runs here unchanged and you can read off which rungs actually transfer.', 'Fotografías de fondo de ojo (FIVES, CC BY 4.0), como control de la tesis de este laboratorio. Un vaso y una grieta son el mismo tipo de objeto para una cámara, así que toda la escalera corre aquí sin cambios y se puede leer qué peldaños transfieren de verdad.')
            : source === 'synthetic'
              ? t('Generated cracks with knobs: width, contrast, angle, noise. The ground truth is exact by construction, not hand-drawn, so this is where you can read off exactly when a method stops working.', 'Grietas generadas con perillas: ancho, contraste, ángulo, ruido. El ground truth es exacto por construcción, no dibujado a mano, así que aquí se puede leer con precisión cuándo un método deja de funcionar.')
              : t('Bring your own crack photo: a compact model segments it entirely in your browser (onnxruntime-web). The image never leaves your device. Drop it in the panel on the right.', 'Trae tu propia foto de grieta: un modelo compacto la segmenta por completo en tu navegador (onnxruntime-web). La imagen nunca sale de tu dispositivo. Suéltala en el panel de la derecha.')}
        </p>

        <div className="fs-side-h">{t('2 · Image', '2 · Imagen')}</div>
        <div className="fs-thumbs">
          {shownSamples.map((s) => {
            const url = s.overlays_rel ? overlayUrl(s.overlays_rel, '_image.png') : null;
            return (
              <button key={s.sample_id} className={`fs-thumb ${s.sample_id === sampleId ? 'on' : ''}`} onClick={() => setSampleId(s.sample_id)} title={`${s.sample_id} (${s.material})`}>
                {url ? <img src={url} alt={s.sample_id} loading="lazy" /> : <span className="fs-thumb-ph" />}
                <span className="fs-thumb-l">{s.material}</span>
              </button>
            );
          })}
        </div>

        <div className="fs-side-h">{t('3 · Parameters', '3 · Parámetros')}</div>
        <div className="fs-ctl">
          <label className="fs-ctl-row">
            <span>{t('Show ground truth (green)', 'Mostrar ground truth (verde)')}</span>
            <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} />
          </label>
          <label>
            <span className="fs-ctl-row"><span>{t('Prediction opacity', 'Opacidad de predicción')}</span><b>{opacity.toFixed(2)}</b></span>
            <input className="range" type="range" min={0.2} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
          </label>
        </div>
        <div className="fs-ctl">
          <div className="fs-ctl-cap">{t('SLIC superpixels (drives the SLIC tab)', 'Superpíxeles SLIC (controla la pestaña SLIC)')}</div>
          <label>
            <span className="fs-ctl-row"><span>{t('Target superpixels', 'Superpíxeles objetivo')}</span><b>{snapN}</b></span>
            <input className="range" type="range" min={Math.min(...gridN)} max={Math.max(...gridN)} step={1} value={slicN} onChange={(e) => setSlicN(Number(e.target.value))} />
          </label>
          <label>
            <span className="fs-ctl-row"><span>{t('Compactness', 'Compacidad')}</span><b>{snapC}</b></span>
            <input className="range" type="range" min={Math.min(...gridC)} max={Math.max(...gridC)} step={1} value={slicC} onChange={(e) => setSlicC(Number(e.target.value))} />
          </label>
          <p className="fs-hint">{t('Higher compactness = squarer, more regular superpixels; lower = they hug edges (and the crack).', 'Mayor compacidad = superpíxeles más cuadrados y regulares; menor = se pegan a los bordes (y a la grieta).')}</p>
        </div>
      </aside>

      <main className="fs-main">
        <p className="fs-kicker">{t('The workbench', 'El banco de trabajo')}</p>
        <h1 style={{ fontSize: '1.4rem', margin: '0.15rem 0 0.7rem' }}>
          {t('Every stage and method, on the image you pick', 'Cada etapa y método, sobre la imagen que elijas')}
        </h1>

        <PanelBoundary label={t('Workbench', 'Banco de trabajo')} es={es}>
          {source === 'upload' ? (
            <LiveLane es={es} />
          ) : !cSample ? (
            <div className="fs-panel"><div className="fs-panel-t">{t('Loading artifacts...', 'Cargando artefactos...')}</div></div>
          ) : (
            <Tabs
              ariaLabel="workbench stages"
              initial="overview"
              tabs={[
                { id: 'overview', label: t('Overview', 'Vista general'), content: <OverviewTab sample={cSample} imageUrl={imageUrl} es={es} /> },
                { id: 'prep', label: t('Preprocessing', 'Preprocesamiento'), content: <PrepTab wb={wbSample} es={es} /> },
                { id: 'semantic', label: t('Semantic seg', 'Segm. semántica'), content: <SemanticTab cSample={cSample} lSample={lSample} imageUrl={imageUrl} showGt={showGt} opacity={opacity} es={es} /> },
                { id: 'slic', label: t('SLIC', 'SLIC'), content: <SlicTab wb={wbSample} n={snapN} c={snapC} es={es} /> },
                { id: 'quant', label: t('Quantification', 'Cuantificación'), content: <QuantTab enrich={enrich} imageUrl={imageUrl} es={es} /> },
                { id: 'classical', label: t('Classical', 'Clásico'), content: <FamilyTab methods={CLASSICAL_METHODS} sample={cSample} base={cSample} imageUrl={imageUrl} showGt={showGt} opacity={opacity} detail={detail} setDetail={setDetail} es={es} /> },
                { id: 'sota', label: t('SOTA', 'SOTA'), content: <FamilyTab methods={LEARNED_METHODS} sample={lSample} base={cSample} imageUrl={imageUrl} showGt={showGt} opacity={opacity} detail={detail} setDetail={setDetail} es={es} dinoPca={dinoPca} gradCam={gradCam} sotaView={sotaView} setSotaView={setSotaView} /> },
                { id: 'beyond', label: t('Beyond SOTA', 'Más allá SOTA'), content: <FamilyTab methods={ANOMALY_METHODS} sample={aSample} base={cSample} imageUrl={imageUrl} showGt={showGt} opacity={opacity} detail={detail} setDetail={setDetail} es={es} anomalyHeat={aSample?.heat_rel ? heatUrl(aSample.heat_rel) : null} /> },
                { id: 'metrics', label: t('Metrics', 'Métricas'), content: <MetricsTab enrich={enrich} es={es} /> },
                { id: 'summary', label: t('Summary', 'Resumen'), content: <SummaryTab cSample={cSample} lSample={lSample} aSample={aSample} imageUrl={imageUrl} showGt={showGt} opacity={opacity} es={es} /> },
                ...(source === 'synthetic'
                  ? [{ id: 'degradation', label: t('Degradation', 'Degradación'),
                       content: <DegradationTab samples={shownSamples} es={es} /> }]
                  : []),
              ]}
            />
          )}
        </PanelBoundary>
      </main>
    </div>
  );
}

// ---- Overview ----------------------------------------------------------------------------------

function OverviewTab({ sample, imageUrl, es }: { sample: ArtifactSample; imageUrl: string | null; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const g = sample.geometry;
  return (
    <div className="fs-wb-two">
      <div className="fs-wb-img">
        {imageUrl ? <img src={imageUrl} alt={sample.sample_id} className="fs-wb-photo" /> : <div className="fs-thumb-ph" />}
        <p className="fs-panel-sub">{sample.sample_id} · {sample.source} · {sample.license_tag}</p>
      </div>
      <div className="fs-wb-read">
        <h3>{t('What you are looking at', 'Qué estás mirando')}</h3>
        <p className="fs-detail-desc">
          {t('The raw image before any method touches it. The workbench applies each stage and method to THIS picture; walk the tabs left to right to see the pipeline unfold.', 'La imagen cruda antes de que ningún método la toque. El banco aplica cada etapa y método a ESTA imagen; recorre las pestañas de izquierda a derecha para ver el pipeline desplegarse.')}
        </p>
        <div className="fs-kpis fs-kpis-2">
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.size[1]}x{sample.size[0]}</div><div className="fs-kpi-l">{t('pixels', 'píxeles')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.material}</div><div className="fs-kpi-l">{t('material', 'material')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.gt_rle ? t('yes', 'sí') : t('no', 'no')}</div><div className="fs-kpi-l">{t('pixel ground truth', 'ground truth de píxeles')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{g.length_px.toFixed(0)}</div><div className="fs-kpi-l">{t('GT skeleton length (px)', 'largo esqueleto GT (px)')}</div></div>
        </div>
      </div>
    </div>
  );
}

// ---- Preprocessing (the real classical stage outputs) ------------------------------------------

function PrepTab({ wb, es }: { wb: WorkbenchSample | null; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  if (!wb) return <Callout variant="note" title={t('No preprocessing for this image', 'Sin preprocesamiento para esta imagen')}>{t('The preprocessing intermediates are baked for the committed example set.', 'Los intermedios de preprocesamiento están baked para el set de ejemplos versionado.')}</Callout>;
  const steps: { key: keyof WorkbenchSample['prep']; en: string; es: string; desc_en: string; desc_es: string }[] = [
    { key: 'gray', en: 'Grayscale', es: 'Escala de grises', desc_en: 'The starting point: intensity only.', desc_es: 'El punto de partida: solo intensidad.' },
    { key: 'flatten', en: 'Illumination flattened', es: 'Iluminación aplanada', desc_en: 'Median-subtract removes uneven lighting so a global threshold can work.', desc_es: 'La resta de mediana quita la iluminación despareja para que un umbral global funcione.' },
    { key: 'denoise', en: 'NLM denoised', es: 'Denoise NLM', desc_en: 'Non-local means suppresses texture noise while preserving thin edges.', desc_es: 'Non-local means suprime el ruido de textura preservando bordes finos.' },
    { key: 'ridge', en: 'Hessian ridge response', es: 'Respuesta cresta Hessiana', desc_en: 'The multiscale sato filter lights up dark curvilinear ridges (the crack).', desc_es: 'El filtro sato multiescala enciende crestas curvilíneas oscuras (la grieta).' },
  ];
  return (
    <div>
      <p className="fs-hint" style={{ marginBottom: '0.8rem' }}>{t('The real classical preprocessing stages, each applied to the selected image. This is how the pipeline turns a photo into something a threshold can segment.', 'Las etapas reales de preprocesamiento clásico, cada una aplicada a la imagen seleccionada. Así el pipeline convierte una foto en algo que un umbral puede segmentar.')}</p>
      <div className="fs-prep-grid">
        {steps.map((s, i) => (
          <figure key={s.key} className="fs-prep-cell">
            <div className="fs-prep-n">{i}</div>
            <img src={workbenchUrl(wb.prep[s.key])} alt={t(s.en, s.es)} />
            <figcaption>
              <b>{t(s.en, s.es)}</b>
              <span>{t(s.desc_en, s.desc_es)}</span>
            </figcaption>
          </figure>
        ))}
      </div>
      {wb.scale_space ? <ScaleSpace ss={wb.scale_space} es={es} /> : null}
    </div>
  );
}

// The Hessian ridge scale-space: sweep the per-sigma response + the argmax-sigma map (which scale wins
// per pixel), so the reader sees WHY the ridge filter is multi-scale (fine texture vs the wide crack).
function ScaleSpace({ ss, es }: { ss: NonNullable<WorkbenchSample['scale_space']>; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const [sig, setSig] = useState<string>(String(ss.sigmas[Math.floor(ss.sigmas.length / 2)]));
  const showArgmax = sig === 'argmax';
  return (
    <div className="fs-panel" style={{ marginTop: '1.1rem' }}>
      <div className="fs-panel-t">{t('Hessian ridge scale-space (why multi-scale matters)', 'Espacio de escalas de la cresta Hessiana (por qué importa multiescala)')}</div>
      <div className="fs-wb-two">
        <div className="fs-wb-img">
          <img className="fs-wb-photo" src={workbenchUrl(showArgmax ? ss.maps.argmax : ss.maps[sig])} alt="scale-space" />
          <p className="fs-panel-sub">
            {showArgmax
              ? t('Winning-scale map: red = a wide ridge (large sigma) fires strongest here, blue = fine texture (small sigma). The crack is red; the noise is blue.', 'Mapa de escala ganadora: rojo = una cresta ancha (sigma grande) responde más fuerte aquí, azul = textura fina (sigma pequeño). La grieta es roja; el ruido es azul.')
              : t('The ridge response at a single scale. Small sigma catches thin cracks and noise alike; large sigma only the wide ridge.', 'La respuesta de cresta a una sola escala. Sigma pequeño atrapa grietas finas y ruido por igual; sigma grande solo la cresta ancha.')}
          </p>
        </div>
        <div className="fs-wb-read">
          <div className="fs-chips">
            {ss.sigmas.map((s) => (
              <button key={s} className={`chip ${sig === String(s) ? 'on' : ''}`} onClick={() => setSig(String(s))}>{`σ=${s}`}</button>
            ))}
            <button className={`chip ${showArgmax ? 'on' : ''}`} onClick={() => setSig('argmax')}>{t('winning scale', 'escala ganadora')}</button>
          </div>
          <p className="fs-detail-desc">
            {t('A single Gaussian scale only responds to ridges near its own half-width. The pipeline runs several sigmas and keeps the max, so it catches both hairline and wide cracks. Sweep the scales, then look at the winning-scale map: the crack occupies the large scales, the texture the small ones. That separation is exactly what a single-scale filter cannot do.', 'Una sola escala gaussiana solo responde a crestas cercanas a su medio-ancho. El pipeline corre varios sigmas y guarda el máximo, así atrapa grietas capilares y anchas. Barre las escalas, luego mira el mapa de escala ganadora: la grieta ocupa las escalas grandes, la textura las pequeñas. Esa separación es justo lo que un filtro de una sola escala no puede hacer.')}
          </p>
        </div>
      </div>
    </div>
  );
}

// ---- Semantic segmentation (the learned masks) -------------------------------------------------

function SemanticTab({ cSample, lSample, imageUrl, showGt, opacity, es }: { cSample: ArtifactSample; lSample: ArtifactSample | null; imageUrl: string | null; showGt: boolean; opacity: number; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const [pick, setPick] = useState('segformer_b2');
  if (!lSample) return <Callout variant="note" title={t('No learned segmentation for this image', 'Sin segmentación aprendida para esta imagen')}>{t('The learned models are baked on the committed example set.', 'Los modelos aprendidos están baked sobre el set de ejemplos versionado.')}</Callout>;
  const methods = LEARNED_METHODS.filter((m) => lSample.levels[m.id]);
  const active = methods.find((m) => m.id === pick) ? pick : methods[0].id;
  const lev = lSample.levels[active];
  const seg = lev?.segmentation;
  return (
    <div className="fs-wb-two">
      <div className="fs-wb-img">
        <MethodTile imageUrl={imageUrl} size={cSample.size} mask={lev?.mask_rle ?? null} gt={cSample.gt_rle} showGt={showGt} opacity={opacity} color={FAMILY_RGB.learned} />
        <OverlayLegend items={predictionLegend(rgbStr(FAMILY_RGB.learned), t('semantic segmentation', 'segmentación semántica'), t('ground truth', 'ground truth'), t('overlap', 'solape'), showGt)} />
        <p className="fs-panel-sub">{t('The learned semantic segmentation of the crack, in purple, over the image (green = ground truth).', 'La segmentación semántica aprendida de la grieta, en púrpura, sobre la imagen (verde = ground truth).')}</p>
      </div>
      <div className="fs-wb-read">
        <div className="fs-chips">
          {methods.map((m) => <button key={m.id} className={`chip ${m.id === active ? 'on' : ''}`} onClick={() => setPick(m.id)}>{m.label}</button>)}
        </div>
        {seg ? (
          <div className="fs-kpis fs-kpis-2">
            <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.f1.toFixed(3)}</div><div className="fs-kpi-l">F1 @ 2 px</div></div>
            <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol5px.f1.toFixed(3)}</div><div className="fs-kpi-l">F1 @ 5 px</div></div>
            <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.precision.toFixed(3)}</div><div className="fs-kpi-l">{t('precision @ 2px', 'precisión @ 2px')}</div></div>
            <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.recall.toFixed(3)}</div><div className="fs-kpi-l">{t('recall @ 2px', 'recall @ 2px')}</div></div>
          </div>
        ) : null}
        {(lev?.notes ?? []).map((n, i) => <p key={i} className="fs-note">{n}</p>)}
      </div>
    </div>
  );
}

// ---- SLIC (param-driven superpixels) -----------------------------------------------------------

function SlicTab({ wb, n, c, es }: { wb: WorkbenchSample | null; n: number; c: number; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  if (!wb) return <Callout variant="note" title={t('No SLIC for this image', 'Sin SLIC para esta imagen')}>{t('The SLIC grid is baked for the committed example set.', 'La grilla SLIC está baked para el set de ejemplos versionado.')}</Callout>;
  const key = `${n}_${c}`;
  const rel = wb.slic[key];
  const real = wb.slic_real_counts[key];
  return (
    <div className="fs-wb-two">
      <div className="fs-wb-img">
        {rel ? <img src={workbenchUrl(rel)} alt="SLIC" className="fs-wb-photo" /> : <div className="fs-thumb-ph" />}
        <p className="fs-panel-sub">{t('SLIC superpixel boundaries (yellow) over the image. Change the superpixel count and compactness in the left column.', 'Fronteras de superpíxeles SLIC (amarillo) sobre la imagen. Cambia el número de superpíxeles y la compacidad en la columna izquierda.')}</p>
      </div>
      <div className="fs-wb-read">
        <h3>{t('SLIC superpixels', 'Superpíxeles SLIC')}</h3>
        <p className="fs-detail-desc">{t('SLIC groups pixels into perceptually uniform regions (superpixels) by clustering in colour + space. It is the standard pre-segmentation for region-based crack methods and for cutting annotation cost: label superpixels, not pixels.', 'SLIC agrupa píxeles en regiones perceptualmente uniformes (superpíxeles) agrupando en color + espacio. Es la pre-segmentación estándar para métodos de grietas por regiones y para reducir el costo de anotación: etiqueta superpíxeles, no píxeles.')}</p>
        <div className="fs-kpis fs-kpis-2">
          <div className="fs-kpi"><div className="fs-kpi-v">{real ?? '--'}</div><div className="fs-kpi-l">{t('actual superpixels', 'superpíxeles reales')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{c}</div><div className="fs-kpi-l">{t('compactness', 'compacidad')}</div></div>
        </div>
        <Callout variant="note" title={t('Reacts to the left column', 'Reacciona a la columna izquierda')}>
          {t('The superpixel-count and compactness sliders drive this view. Low compactness lets the superpixels bend around the crack; high compactness keeps them square and ignores it.', 'Los deslizadores de número de superpíxeles y compacidad controlan esta vista. Baja compacidad deja que los superpíxeles se doblen alrededor de la grieta; alta los mantiene cuadrados y la ignora.')}
        </Callout>
      </div>
    </div>
  );
}

// ---- a method family applied to the image (Classical / SOTA / Beyond) --------------------------

function FamilyTab({ methods, sample, base, imageUrl, showGt, opacity, detail, setDetail, es, anomalyHeat, dinoPca, gradCam, sotaView, setSotaView }: {
  methods: MethodDef[]; sample: ArtifactSample | null; base: ArtifactSample; imageUrl: string | null;
  showGt: boolean; opacity: number; detail: string; setDetail: (s: string) => void; es: boolean; anomalyHeat?: string | null;
  dinoPca?: DinoPca | null; gradCam?: GradCam | null;
  sotaView?: 'pred' | 'features' | 'cam'; setSotaView?: (v: 'pred' | 'features' | 'cam') => void;
}) {
  const t = (en: string, esx: string) => (es ? esx : en);
  if (!sample) return <Callout variant="note" title={t('Not available for this image', 'No disponible para esta imagen')}>{t('This method family is baked on the committed example set.', 'Esta familia de métodos está baked sobre el set de ejemplos versionado.')}</Callout>;
  const avail = methods.filter((m) => sample.levels[m.id]);
  const active = avail.find((m) => m.id === detail) ? detail : avail[0].id;
  const fam = methods[0].family;
  const lev = sample.levels[active];
  const seg = lev?.segmentation;
  const anom = (lev as (LevelRecord & { anomaly_score_norm?: number }) | undefined)?.anomaly_score_norm;
  const pcaUrl = fam === 'learned' && sotaView === 'features' && dinoPca
    ? dinoPca.samples.find((s) => s.id === base.sample_id)?.pca ?? null
    : null;
  const camRow = fam === 'learned' && sotaView === 'cam' && gradCam
    ? gradCam.samples.find((s) => s.id === base.sample_id && s.arch === active) ?? null
    : null;
  return (
    <div>
      <div className="fs-matrix-h"><span className="fs-dot" style={{ background: FAMILY_TONE[fam] }} />{t(...FAMILY_LABEL[fam])}</div>
      <div className="fs-wb-two">
        <div className="fs-wb-img">
          {camRow ? (
            <>
              <img className="fs-wb-photo" src={`${import.meta.env.BASE_URL}data/${camRow.cam}`} alt="Grad-CAM" />
              <OverlayLegend items={[{ color: 'linear-gradient(90deg,rgb(40,90,220),rgb(235,60,40))', label: t('Grad-CAM evidence (low to high)', 'evidencia Grad-CAM (baja a alta)'), kind: 'gradient' }]} />
              <p className="fs-panel-sub">
                {camRow.note
                  ? t(`Grad-CAM for this image: ${camRow.note}`, `Grad-CAM para esta imagen: ${camRow.note}`)
                  : t(`The evidence behind the mask, not the mask itself: gradients of the crack logit weight the encoder features. ${camRow.cam_mass_on_crack != null ? `${(camRow.cam_mass_on_crack * 100).toFixed(0)} percent of the CAM mass falls on the true crack.` : ''} Grad-CAM was built for whole-image classification, so on a 1-5 px crack these maps localise the evidence to a region, not to the outline.`, `La evidencia detrás de la máscara, no la máscara: los gradientes del logit de grieta ponderan las features del codificador. ${camRow.cam_mass_on_crack != null ? `${(camRow.cam_mass_on_crack * 100).toFixed(0)} por ciento de la masa del CAM cae sobre la grieta real.` : ''} Grad-CAM se diseñó para clasificación de imagen completa, así que sobre una grieta de 1-5 px estos mapas localizan la evidencia a una región, no al contorno.`)}
              </p>
            </>
          ) : pcaUrl ? (
            <>
              <img className="fs-wb-photo" src={`${import.meta.env.BASE_URL}data/${pcaUrl}`} alt="DINOv2 feature PCA" />
              <OverlayLegend items={[{ color: 'linear-gradient(90deg,rgb(230,60,160),rgb(60,200,120),rgb(60,120,230))', label: t('DINOv2 feature space (PCA to RGB)', 'espacio de features DINOv2 (PCA a RGB)'), kind: 'gradient' }]} />
              <p className="fs-panel-sub">
                {t('What the FROZEN foundation model encodes, with no crack supervision at all: each 14 px patch becomes a 384-dim descriptor, and its first three principal components map to RGB. The crack separates as its own hue, which is exactly why a 385-parameter linear head on these features is competitive.', 'Lo que el modelo fundacional CONGELADO codifica, sin ninguna supervisión de grietas: cada parche de 14 px se vuelve un descriptor de 384 dimensiones, y sus tres primeras componentes principales mapean a RGB. La grieta se separa como su propio tono, que es exactamente por qué una cabeza lineal de 385 parámetros sobre estas features es competitiva.')}
              </p>
            </>
          ) : (
            <>
              <MethodTile imageUrl={imageUrl} size={(fam === 'anomaly' ? sample : base).size} mask={lev?.mask_rle ?? null} gt={base.gt_rle} showGt={showGt} opacity={opacity} color={FAMILY_RGB[fam]} heatUrl={fam === 'anomaly' ? anomalyHeat ?? null : null} />
              <OverlayLegend items={fam === 'anomaly'
                ? [{ color: 'linear-gradient(90deg,rgb(40,90,220),rgb(235,60,40))', label: t('anomaly heat (low to high)', 'calor de anomalía (bajo a alto)'), kind: 'gradient' }, { color: 'rgb(255,255,255)', label: t('flagged region outline', 'contorno de región marcada'), kind: 'outline' }, ...(showGt ? [{ color: 'rgb(46,204,113)', label: t('ground truth crack', 'grieta ground truth') }] : [])]
                : predictionLegend(rgbStr(FAMILY_RGB[fam]), t('prediction', 'predicción'), t('ground truth', 'ground truth'), t('overlap', 'solape'), showGt)} />
              <p className="fs-panel-sub">{t('The method prediction over the image. Pick a method to compare.', 'La predicción del método sobre la imagen. Elige un método para comparar.')}</p>
            </>
          )}
        </div>
        <div className="fs-wb-read">
          <div className="fs-chips">
            {avail.map((m) => <button key={m.id} className={`chip ${m.id === active ? 'on' : ''}`} onClick={() => setDetail(m.id)} title={t(m.en, m.es)}>{m.label}</button>)}
            {fam === 'learned' && setSotaView ? (
              <>
                <span className="fs-chip-sep" aria-hidden="true" />
                <button className={`chip ${sotaView === 'pred' ? 'on' : ''}`} onClick={() => setSotaView('pred')} title={t('The predicted crack mask', 'La máscara de grieta predicha')}>
                  {t('mask', 'máscara')}
                </button>
                {dinoPca ? (
                  <button className={`chip ${sotaView === 'features' ? 'on' : ''}`} onClick={() => setSotaView('features')} title={t('What the frozen DINOv2 features encode', 'Lo que codifican las features congeladas de DINOv2')}>
                    {t('DINOv2 features', 'features DINOv2')}
                  </button>
                ) : null}
                {gradCam && gradCam.archs.includes(active) ? (
                  <button className={`chip ${sotaView === 'cam' ? 'on' : ''}`} onClick={() => setSotaView('cam')} title={t('What evidence drove the decision', 'Qué evidencia impulsó la decisión')}>
                    {t('Grad-CAM', 'Grad-CAM')}
                  </button>
                ) : null}
              </>
            ) : null}
          </div>
          <p className="fs-detail-desc">{t(avail.find((m) => m.id === active)!.en, avail.find((m) => m.id === active)!.es)}</p>
          {seg ? (
            <div className="fs-kpis fs-kpis-2">
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.f1.toFixed(3)}</div><div className="fs-kpi-l">F1 @ 2 px</div></div>
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol5px.f1.toFixed(3)}</div><div className="fs-kpi-l">F1 @ 5 px</div></div>
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.precision.toFixed(3)}</div><div className="fs-kpi-l">{t('precision @ 2px', 'precisión @ 2px')}</div></div>
              <div className="fs-kpi"><div className="fs-kpi-v">{seg.tol2px.recall.toFixed(3)}</div><div className="fs-kpi-l">{t('recall @ 2px', 'recall @ 2px')}</div></div>
            </div>
          ) : anom != null ? (
            <div className="fs-kpis fs-kpis-2"><div className="fs-kpi"><div className="fs-kpi-v">{(anom * 100).toFixed(0)}%</div><div className="fs-kpi-l">{t('anomaly score', 'puntaje anomalía')}</div></div></div>
          ) : null}
          {(lev?.notes ?? []).map((n, i) => <p key={i} className="fs-note">{n}</p>)}
          {fam === 'anomaly' ? (
            <Callout variant="honest" title={t('Read honestly', 'Léelo con honestidad')}>
              {t('The memory bank saw only UNCRACKED concrete. High = unlike healthy concrete, not necessarily a crack. Transfer AUROC is 0.72, far below the 0.996 the same method reaches on industrial MVTec AD.', 'El banco de memoria vio solo hormigón SIN grietas. Alto = distinto del hormigón sano, no necesariamente una grieta. El AUROC de transferencia es 0.72, muy por debajo del 0.996 del mismo método en el industrial MVTec AD.')}
            </Callout>
          ) : null}
        </div>
      </div>
    </div>
  );
}

// ---- Summary (the full comparison matrix + ranking) --------------------------------------------

interface Cell { method: MethodDef; level: LevelRecord | null; f1_2: number | null; anom: number | null; }

function SummaryTab({ cSample, lSample, aSample, imageUrl, showGt, opacity, es }: {
  cSample: ArtifactSample; lSample: ArtifactSample | null; aSample: ArtifactSample | null;
  imageUrl: string | null; showGt: boolean; opacity: number; es: boolean;
}) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const cells: Cell[] = [
    ...CLASSICAL_METHODS.map((m) => ({ method: m, level: cSample.levels[m.id] ?? null })),
    ...LEARNED_METHODS.map((m) => ({ method: m, level: lSample?.levels[m.id] ?? null })),
    ...ANOMALY_METHODS.map((m) => ({ method: m, level: (aSample?.levels[m.id] as LevelRecord) ?? null })),
  ].map(({ method, level }) => ({
    method, level,
    f1_2: level?.segmentation?.tol2px.f1 ?? null,
    anom: (level as (LevelRecord & { anomaly_score_norm?: number }) | null)?.anomaly_score_norm ?? null,
  }));
  const bestId = cells.reduce<{ id: string; f1: number } | null>((b, c) => (c.f1_2 != null && (!b || c.f1_2 > b.f1) ? { id: c.method.id, f1: c.f1_2 } : b), null)?.id ?? null;
  const scored = cells.filter((c) => c.f1_2 != null);
  const xs = scored.map((_, i) => i);

  return (
    <div>
      <p className="fs-hint" style={{ marginBottom: '0.5rem' }}>{t('Every method applied to this image at once. The winner (highest F1@2px) is starred; click nothing, just compare.', 'Cada método aplicado a esta imagen a la vez. El ganador (mayor F1@2px) lleva estrella; no hagas clic, solo compara.')}</p>
      <OverlayLegend items={[
        { color: rgbStr(FAMILY_RGB.classical), label: t('classical prediction', 'predicción clásica') },
        { color: rgbStr(FAMILY_RGB.learned), label: t('learned prediction', 'predicción aprendida') },
        { color: 'linear-gradient(90deg,rgb(40,90,220),rgb(235,60,40))', label: t('anomaly heat', 'calor de anomalía'), kind: 'gradient' },
        ...(showGt ? [{ color: 'rgb(46,204,113)', label: t('ground truth', 'ground truth') }, { color: 'rgb(240,210,70)', label: t('overlap', 'solape') }] : []),
      ]} />
      {(['classical', 'learned', 'anomaly'] as Family[]).map((fam) => {
        const famCells = cells.filter((c) => c.method.family === fam);
        return (
          <section key={fam} className="fs-matrix-sec">
            <div className="fs-matrix-h"><span className="fs-dot" style={{ background: FAMILY_TONE[fam] }} />{t(...FAMILY_LABEL[fam])}</div>
            <div className="fs-grid">
              {famCells.map((c) => (
                <div key={c.method.id} className="fs-cell" style={{ ['--tone' as string]: FAMILY_TONE[fam] }}>
                  <MethodTile imageUrl={imageUrl} size={(fam === 'anomaly' ? aSample : cSample)?.size ?? cSample.size} mask={c.level?.mask_rle ?? null} gt={cSample.gt_rle} showGt={showGt} opacity={opacity} color={FAMILY_RGB[fam]} heatUrl={fam === 'anomaly' && aSample?.heat_rel ? heatUrl(aSample.heat_rel) : null} />
                  <div className="fs-cell-b">
                    <span className="fs-cell-l">{c.method.id === bestId ? '★ ' : ''}{c.method.label}</span>
                    <span className="fs-cell-v">{c.f1_2 != null ? c.f1_2.toFixed(2) : c.anom != null ? `${(c.anom * 100).toFixed(0)}%` : '--'}</span>
                  </div>
                </div>
              ))}
            </div>
          </section>
        );
      })}
      {scored.length > 1 ? (
        <div className="fs-panel" style={{ marginTop: '1rem' }}>
          <div className="fs-panel-t">{t('Every method ranked on THIS image (both tolerance protocols)', 'Cada método rankeado en ESTA imagen (ambos protocolos)')}</div>
          <UPlotChart
            data={[xs, scored.map((c) => c.f1_2), scored.map((c) => c.level?.segmentation?.tol5px.f1 ?? null)]}
            series={[{}, { label: 'F1 @ 2 px', stroke: '#cf222e', width: 2, points: { show: true, size: 6 } }, { label: 'F1 @ 5 px', stroke: '#2da44e', width: 2, points: { show: true, size: 6 } }]}
            axes={[{ values: (_u, ticks) => ticks.map((v) => (Number.isInteger(v) ? scored[v]?.method.label ?? '' : '')), splits: () => xs, rotate: -35 }, { label: 'F1' }]}
            scales={{ x: { time: false }, y: { range: [0, 1] } }}
            height={230}
          />
          <p className="fs-panel-sub">{t('The same masks score differently at 2 px and 5 px; the protocol always travels with the number.', 'Las mismas máscaras puntúan distinto a 2 px y 5 px; el protocolo siempre viaja con el número.')}</p>
        </div>
      ) : null}
    </div>
  );
}

// ---- Degradation (synthetic only) --------------------------------------------------------------
// The reason the synthetic battery is worth having. On a photograph the ground truth is a human
// tracing, so "the method missed the crack" and "the label was drawn a pixel wide" are not separable.
// Here the mask is exact by construction, the only thing that changes across cases is one generator
// knob, and the classical ladder was scored on every one of them. So the F1 collapse as the crack
// narrows is a measurement of the METHOD, with the labelling error removed.

function DegradationTab({ samples, es }: { samples: ArtifactSample[]; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);

  // the controlled sweep: same generator, same contrast, only the crack width moves
  const sweep = useMemo(() => {
    const rows = samples
      .filter((s) => {
        const p = (s.synthetic_params ?? {}) as Record<string, unknown>;
        return p.kind === 'straight_bar' && p.contrast === 0.35 && typeof p.width_px === 'number';
      })
      .map((s) => ({ w: Number((s.synthetic_params as Record<string, unknown>).width_px), s }))
      .sort((a, b) => a.w - b.w);
    // several seeds share a width: average them so each width is one honest point
    const byWidth = new Map<number, ArtifactSample[]>();
    for (const r of rows) byWidth.set(r.w, [...(byWidth.get(r.w) ?? []), r.s]);
    return [...byWidth.entries()].sort((a, b) => a[0] - b[0]);
  }, [samples]);

  const f1 = (s: ArtifactSample, lvl: string): number | null => {
    const rec = s.levels?.[lvl] as LevelRecord | undefined;
    const v = rec?.segmentation?.tol2px?.f1;
    return typeof v === 'number' ? v : null;
  };

  const widths = sweep.map(([w]) => w);
  const levels = ['L0', 'L1', 'L2', 'L3', 'L4', 'L5'];
  const labels: Record<string, string> = {
    L0: 'L0 Otsu', L1: 'L1 Sauvola', L2: 'L2 top-hat', L3: 'L3 Hessian', L4: 'L4 min-path', L5: 'L5 RF fusion',
  };
  const colors = ['#b0304f', '#d1663a', '#c9a227', '#3f8f5f', '#2f6f9f', '#6a4c93'];

  const series = levels.map((lvl, i) => {
    const vals = sweep.map(([, ss]) => {
      const got = ss.map((s) => f1(s, lvl)).filter((v): v is number => v !== null);
      return got.length ? got.reduce((a, b) => a + b, 0) / got.length : null;
    });
    return { lvl, vals, color: colors[i] };
  });

  const data = [widths, ...series.map((s) => s.vals)] as unknown as import('uplot').AlignedData;
  const uSeries = [
    {},
    ...series.map((s) => ({ label: labels[s.lvl], stroke: s.color, width: 2, points: { show: true, size: 6 } })),
  ];

  const controls = samples.filter((s) => (s.synthetic_params as Record<string, unknown> | undefined)?.kind === 'uncracked');
  const lowContrast = samples.find((s) => {
    const p = (s.synthetic_params ?? {}) as Record<string, unknown>;
    return p.kind === 'straight_bar' && p.contrast === 0.12;
  });

  return (
    <div className="fs-wb-read">
      <h3>{t('Where each classical method stops working', 'Dónde cada método clásico deja de funcionar')}</h3>
      <p className="fs-detail-desc">
        {t('Same generator, same contrast (0.35), same angle: only the crack width changes, from 9 px down to 2 px. Because these cracks are generated, the mask is exact rather than hand-traced, so a drop in F1 measures the method and not the labelling. Scored at the strict 2 px tolerance.',
           'Mismo generador, mismo contraste (0.35), mismo ángulo: sólo cambia el ancho de la grieta, de 9 px a 2 px. Como estas grietas son generadas, la máscara es exacta y no trazada a mano, así que una caída de F1 mide el método y no el etiquetado. Evaluado con la tolerancia estricta de 2 px.')}
      </p>
      <p className="fs-detail-desc">
        {t('Read the ordering carefully, because it inverts here: the simple thresholds sit at F1 1.0 across the whole sweep, while the two most elaborate rungs, minimal-path bridging and the random-forest fusion, score BELOW them. That is not a defect. A generated crack is a high-contrast mark on a nearly uniform field, which is the exact case a global threshold solves perfectly, and the later rungs pay for machinery built to survive staining, shadow and texture that is simply not present. The ladder only starts to order itself the way you would expect once the images get hard. You will also count fewer curves than methods: L0 and L1 are both pinned at 1.0, and L3 and L4 land on identical values at every width, because minimal-path bridging has nothing to bridge when the generated crack is already one unbroken stroke.',
           'Lee el orden con cuidado, porque aquí se invierte: los umbrales simples se quedan en F1 1.0 durante todo el barrido, mientras que los dos peldaños más elaborados, el puente de caminos mínimos y la fusión random-forest, puntúan POR DEBAJO. No es un defecto. Una grieta generada es una marca de alto contraste sobre un fondo casi uniforme, que es justo el caso que un umbral global resuelve a la perfección, y los peldaños posteriores pagan por maquinaria construida para sobrevivir manchas, sombras y textura que aquí no existen. La escalera sólo empieza a ordenarse como esperarías cuando las imágenes se ponen difíciles. También contarás menos curvas que métodos: L0 y L1 quedan fijos en 1.0, y L3 y L4 caen en valores idénticos en cada ancho, porque el puente de caminos mínimos no tiene nada que unir cuando la grieta generada ya es un trazo continuo.')}
      </p>
      {widths.length > 1 ? (
        <UPlotChart
          data={data}
          series={uSeries}
          height={300}
          axes={[
            // without an explicit formatter uPlot renders the x values as clock times (":02.000"),
            // because its default x scale is a time scale; these are pixel widths
            { label: t('crack width (px)', 'ancho de grieta (px)'), values: (_u, vs) => vs.map((v) => `${v}`) },
            { label: 'F1 @ 2 px' },
          ]}
          scales={{ x: { time: false }, y: { range: [0, 1] } }}
        />
      ) : (
        <Callout variant="honest" title={t('Sweep unavailable', 'Barrido no disponible')}>
          {t('The width sweep needs at least two widths in the committed battery.', 'El barrido de ancho necesita al menos dos anchos en la batería versionada.')}
        </Callout>
      )}
      <div className="fs-kpis fs-kpis-2" style={{ marginTop: '0.8rem' }}>
        <div className="fs-kpi"><div className="fs-kpi-v">{widths.length}</div><div className="fs-kpi-l">{t('widths swept', 'anchos barridos')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{samples.length}</div><div className="fs-kpi-l">{t('cases in the battery', 'casos en la batería')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{controls.length}</div><div className="fs-kpi-l">{t('uncracked controls', 'controles sin grieta')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{lowContrast ? '0.12' : '-'}</div><div className="fs-kpi-l">{t('low-contrast case', 'caso de bajo contraste')}</div></div>
      </div>
      <Callout variant="honest" title={t('What this view cannot tell you', 'Lo que esta vista no puede decirte')}>
        {t('Synthetic cracks are clean: no shadows, no surface staining, no camera blur, no joints mistaken for damage. A method that survives this sweep has cleared the easy half of the problem. That is why the real samples sit next to it, and why the uncracked controls matter, since a method can score well here purely by firing everywhere.',
           'Las grietas sintéticas son limpias: sin sombras, sin manchas de superficie, sin desenfoque de cámara, sin juntas confundidas con daño. Un método que sobrevive este barrido superó la mitad fácil del problema. Por eso las muestras reales están al lado, y por eso importan los controles sin grieta, ya que un método puede puntuar bien aquí simplemente disparando en todas partes.')}
      </Callout>
    </div>
  );
}
