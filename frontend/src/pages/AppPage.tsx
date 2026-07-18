import { useEffect, useMemo, useState } from 'react';
import { Callout, SubTabs } from '@fasl-work/caos-app-shell';
import { loadArtifact, loadIndex, loadManifest, overlayUrl } from '../api/artifacts';
import type { ArtifactSample, CaseArtifact, CaseIndex, CaseManifest } from '../lib/contract.types';
import { useT } from '../lib/i18n';
import { MaskCanvas } from '../render/MaskCanvas';
import { PanelBoundary } from '../render/PanelBoundary';
import { UPlotChart } from '../render/UPlotChart';

// The App workbench (ADR-0016 section 9): one selected case, a variant bar over the ladder levels,
// Field / Live / Charts / Context sub-tabs. Replay-first: everything shown here is a committed,
// audited artifact; the Live tab states its own status honestly until the browser lane ships.
export default function AppPage() {
  const t = useT();
  const es = t('x', 'y') === 'y';

  const [index, setIndex] = useState<CaseIndex | null>(null);
  const [caseId, setCaseId] = useState<string | null>(null);
  const [manifest, setManifest] = useState<CaseManifest | null>(null);
  const [artifact, setArtifact] = useState<CaseArtifact | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [level, setLevel] = useState<string>('L3');
  const [showGt, setShowGt] = useState(true);
  const [opacity, setOpacity] = useState(0.55);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadIndex()
      .then((idx) => {
        setIndex(idx);
        if (idx.cases.length > 0) setCaseId(idx.cases[0].case_id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!caseId) return;
    setManifest(null);
    setArtifact(null);
    loadManifest(caseId)
      .then((m) => {
        setManifest(m);
        return loadArtifact(m.artifact.path);
      })
      .then((a) => {
        setArtifact(a);
        setSampleId(a.samples[0]?.sample_id ?? null);
      })
      .catch((e) => setError(String(e)));
  }, [caseId]);

  const sample: ArtifactSample | null = useMemo(
    () => artifact?.samples.find((s) => s.sample_id === sampleId) ?? null,
    [artifact, sampleId],
  );
  const levels = useMemo(() => (sample ? Object.keys(sample.levels) : []), [sample]);
  const activeLevel = levels.includes(level) ? level : levels[levels.length - 1] ?? 'L3';
  const lev = sample?.levels[activeLevel] ?? null;

  if (error) {
    return (
      <div className="fs-doc">
        <Callout variant="honest" title={t('Artifacts unavailable', 'Artefactos no disponibles')}>
          {t('The committed artifacts could not be loaded: ', 'No se pudieron cargar los artefactos versionados: ')}
          <code>{error}</code>
        </Callout>
      </div>
    );
  }

  return (
    <div>
      <p className="fs-kicker">{t('The workbench', 'El banco de trabajo')}</p>
      <h1 style={{ fontSize: '1.5rem', margin: '0.2rem 0 0.8rem' }}>
        {t('Classical crack analysis, replayed from audited artifacts', 'Análisis clásico de grietas, replay de artefactos auditados')}
      </h1>

      <div className="fs-layout">
        <aside className="fs-controls">
          <div className="fs-ctl">
            <span>{t('Case', 'Caso')}</span>
            <select className="fs-sel" value={caseId ?? ''} onChange={(e) => setCaseId(e.target.value)}>
              {index?.cases.map((c) => (
                <option key={c.case_id} value={c.case_id}>
                  {c.case_id} [{c.category}]
                </option>
              ))}
            </select>
          </div>
          <div className="fs-ctl">
            <span>{t('Sample', 'Muestra')}</span>
            <select className="fs-sel" value={sampleId ?? ''} onChange={(e) => setSampleId(e.target.value)}>
              {artifact?.samples.map((s) => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.sample_id} ({s.material})
                </option>
              ))}
            </select>
          </div>
          <div className="fs-ctl">
            <span>{t('Ladder level (the variant bar)', 'Nivel de la escalera (barra de variantes)')}</span>
            <div className="fs-chips">
              {levels.map((l) => (
                <button key={l} className={`chip ${l === activeLevel ? 'on' : ''}`} onClick={() => setLevel(l)}>
                  {l}
                </button>
              ))}
            </div>
          </div>
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
          </div>
          {manifest ? (
            <div className="fs-panel">
              <div className="fs-panel-t">{t('Provenance', 'Procedencia')}</div>
              <p className="fs-panel-sub">
                {manifest.engine.model} · v{manifest.engine.version} · seed {manifest.seed} ·{' '}
                <span className={`fs-badge ${manifest.lane === 'live' ? 'live' : 'replay'}`}>{manifest.lane}</span>
              </p>
              <p className="fs-panel-sub">{manifest.title}</p>
            </div>
          ) : null}
        </aside>

        <main className="fs-main">
          <PanelBoundary label={t('Workbench', 'Banco de trabajo')} es={es}>
            {sample && lev ? (
              <SubTabs
                ariaLabel="workbench views"
                tabs={[
                  {
                    id: 'field',
                    label: t('Field', 'Campo'),
                    content: (
                      <div className="fs-panel">
                        <div className="fs-panel-t">
                          {sample.sample_id} · {activeLevel} · {lev.notes.join('; ')}
                        </div>
                        <MaskCanvas
                          imageUrl={sample.overlays_rel ? overlayUrl(sample.overlays_rel, '_image.png') : null}
                          size={sample.size}
                          mask={lev.mask_rle}
                          gt={sample.gt_rle}
                          showGt={showGt}
                          opacity={opacity}
                        />
                        <p className="fs-panel-sub">
                          {t('Prediction in red; ground truth in green (overlap reads yellow). All masks decoded client-side from the committed RLE artifact.', 'Predicción en rojo; ground truth en verde (el solape se lee amarillo). Todas las máscaras se decodifican en el cliente desde el artefacto RLE versionado.')}
                        </p>
                      </div>
                    ),
                  },
                  {
                    id: 'live',
                    label: t('Live', 'En vivo'),
                    content: (
                      <Callout variant="note" title={t('The live lane arrives with its own unit', 'El carril en vivo llega con su propia unidad')}>
                        {t(
                          'This tab will analyze a photo you drop here, entirely in your browser (the same validation and ladder code, compiled to WebAssembly, plus compact ONNX models). Until that unit ships, everything you see is the replay lane: committed, audited artifacts.',
                          'Esta pestaña analizará una foto que arrastres aquí, por completo en tu navegador (el mismo código de validación y escalera, compilado a WebAssembly, más modelos ONNX compactos). Hasta que esa unidad llegue, todo lo que ves es el carril replay: artefactos versionados y auditados.',
                        )}
                      </Callout>
                    ),
                  },
                  {
                    id: 'charts',
                    label: t('Charts', 'Gráficos'),
                    content: <ChartsView sample={sample} es={es} />,
                  },
                  {
                    id: 'context',
                    label: t('Context', 'Contexto'),
                    content: <ContextView caseId={caseId ?? ''} es={es} />,
                  },
                ]}
              />
            ) : (
              <div className="fs-panel">
                <div className="fs-panel-t">{t('Loading artifacts...', 'Cargando artefactos...')}</div>
              </div>
            )}
          </PanelBoundary>
        </main>
      </div>
    </div>
  );
}

function ChartsView({ sample, es }: { sample: ArtifactSample; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const levels = Object.keys(sample.levels);
  const f1s2 = levels.map((l) => sample.levels[l].segmentation?.tol2px.f1 ?? null);
  const f1s5 = levels.map((l) => sample.levels[l].segmentation?.tol5px.f1 ?? null);
  const xs = levels.map((_, i) => i);
  const w = sample.geometry.width;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
      {sample.gt_rle ? (
        <div className="fs-panel">
          <div className="fs-panel-t">{t('F1 per ladder level, BOTH tolerance protocols', 'F1 por nivel de la escalera, AMBOS protocolos de tolerancia')}</div>
          <UPlotChart
            data={[xs, f1s2, f1s5]}
            series={[
              {},
              { label: 'F1 @ 2 px', stroke: '#2f81f7', width: 2, points: { show: true, size: 6 } },
              { label: 'F1 @ 5 px', stroke: '#2da44e', width: 2, points: { show: true, size: 6 } },
            ]}
            axes={[
              { values: (_u, ticks) => ticks.map((v) => (Number.isInteger(v) ? levels[v] ?? '' : '')), splits: () => xs },
              { label: 'F1' },
            ]}
            scales={{ x: { time: false }, y: { range: [0, 1] } }}
            height={220}
          />
          <p className="fs-panel-sub">
            {t('The protocol travels with the number: the same masks score differently at 2 px and 5 px tolerance.', 'El protocolo viaja con el número: las mismas máscaras puntúan distinto con tolerancia de 2 px y de 5 px.')}
          </p>
        </div>
      ) : (
        <Callout variant="note" title={t('No ground truth for this sample', 'Sin ground truth para esta muestra')}>
          {t('Classification-only source (no pixel mask): segmentation scores are not defined here.', 'Fuente solo de clasificación (sin máscara de píxeles): los puntajes de segmentación no están definidos aquí.')}
        </Callout>
      )}

      <div className="fs-kpis">
        <div className="fs-kpi"><div className="fs-kpi-v">{w.edt_median?.toFixed(2) ?? 'n/a'}</div><div className="fs-kpi-l">{t('width median, inscribed circle (px)', 'ancho mediano, círculo inscrito (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{w.profile_median?.toFixed(2) ?? 'n/a'}</div><div className="fs-kpi-l">{t('width median, orthogonal profile (px)', 'ancho mediano, perfil ortogonal (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{sample.geometry.length_px.toFixed(0)}</div><div className="fs-kpi-l">{t('skeleton length (px)', 'largo del esqueleto (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{sample.geometry.n_branches}</div><div className="fs-kpi-l">{t('branch points', 'puntos de rama')}</div></div>
      </div>

      <div className="fs-panel">
        <div className="fs-panel-t">{t('Orientation histogram (10-degree bins over 0..180)', 'Histograma de orientación (bins de 10 grados sobre 0..180)')}</div>
        <UPlotChart
          data={[sample.geometry.orientation_hist.map((_, i) => i * 10 + 5), sample.geometry.orientation_hist]}
          series={[{}, { label: t('skeleton px', 'px de esqueleto'), stroke: '#8250df', width: 2, points: { show: true, size: 5 } }]}
          axes={[{ label: t('angle (deg)', 'ángulo (grados)') }, {}]}
          scales={{ x: { time: false } }}
          height={180}
        />
      </div>

      {sample.width_validation ? (
        <div className="fs-panel">
          <div className="fs-panel-t">{t('Width validation against exact synthetic truth', 'Validación de ancho contra verdad sintética exacta')}</div>
          <div className="fs-tablewrap">
            <table className="fs-table">
              <tbody>
                <tr><td>{t('true mask width (median)', 'ancho verdadero de máscara (mediana)')}</td><td className="mono">{sample.width_validation.true_width_px.toFixed(2)} px</td></tr>
                {sample.width_validation.true_fwhm_px != null ? (
                  <tr><td>{t('true optical FWHM (mask + edge softness)', 'FWHM óptico verdadero (máscara + suavidad de borde)')}</td><td className="mono">{sample.width_validation.true_fwhm_px.toFixed(2)} px</td></tr>
                ) : null}
                <tr><td>{t('inscribed-circle estimate on GT', 'estimación círculo inscrito sobre GT')}</td><td className="mono">{sample.width_validation.edt_on_gt_median?.toFixed(2) ?? 'n/a'} px</td></tr>
                <tr><td>{t('orthogonal-profile estimate on GT', 'estimación perfil ortogonal sobre GT')}</td><td className="mono">{sample.width_validation.profile_on_gt_median?.toFixed(2) ?? 'n/a'} px</td></tr>
                {sample.width_validation.intensity_on_gt_median != null ? (
                  <tr><td>{t('intensity sub-pixel FWHM estimate', 'estimación FWHM subpíxel por intensidad')}</td><td className="mono">{sample.width_validation.intensity_on_gt_median.toFixed(2)} px</td></tr>
                ) : null}
                <tr><td>{t('absolute error (inscribed circle vs mask width)', 'error absoluto (círculo inscrito vs ancho de máscara)')}</td><td className="mono">{sample.width_validation.edt_abs_error?.toFixed(3) ?? 'n/a'} px</td></tr>
                {sample.width_validation.intensity_fwhm_abs_error != null ? (
                  <tr><td>{t('absolute error (intensity vs optical FWHM)', 'error absoluto (intensidad vs FWHM óptico)')}</td><td className="mono">{sample.width_validation.intensity_fwhm_abs_error.toFixed(3)} px</td></tr>
                ) : null}
              </tbody>
            </table>
          </div>
          <p className="fs-panel-sub">
            {t('Two width definitions coexist on purpose: mask-boundary width and optical full-width-at-half-maximum. Each estimator is validated against its own definition; the gap between them is the edge-softness physics, not an error.', 'Dos definiciones de ancho coexisten a propósito: ancho por borde de máscara y ancho óptico a media altura. Cada estimador se valida contra su propia definición; la brecha entre ambas es la física de la suavidad del borde, no un error.')}
          </p>
        </div>
      ) : null}

      {sample.width_mm ? (
        <div className="fs-kpis">
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.width_mm.median.toFixed(2)}</div><div className="fs-kpi-l">{t('width median (mm)', 'ancho mediano (mm)')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.width_mm.p95.toFixed(2)}</div><div className="fs-kpi-l">{t('width p95 (mm)', 'ancho p95 (mm)')}</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{sample.width_mm.mm_per_px.toFixed(3)}</div><div className="fs-kpi-l">mm/px</div></div>
          <div className="fs-kpi"><div className="fs-kpi-v">{(sample.geometry.length_px * sample.width_mm.mm_per_px).toFixed(0)}</div><div className="fs-kpi-l">{t('length (mm)', 'largo (mm)')}</div></div>
        </div>
      ) : null}

      {sample.severity ? (
        <div className="fs-panel">
          <div className="fs-panel-t">{t('Severity CONTEXT: measured width vs published guidance bands', 'CONTEXTO de severidad: ancho medido vs bandas de guías publicadas')}</div>
          <div className="fs-tablewrap">
            <table className="fs-table">
              <thead>
                <tr><th>{t('Source', 'Fuente')}</th><th>{t('Exposure', 'Exposición')}</th><th className="mono">{t('limit (mm)', 'límite (mm)')}</th><th>{t('median', 'mediana')}</th><th>p95</th></tr>
              </thead>
              <tbody>
                {sample.severity.bands.map((b, i) => (
                  <tr key={i}>
                    <td>{b.source}</td>
                    <td>{b.exposure}</td>
                    <td className="mono">{b.limit_mm.toFixed(2)}</td>
                    <td><span className={`fs-badge ${b.median_within ? 'real' : 'tr-monitor'}`}>{b.median_within ? t('within', 'dentro') : t('exceeds', 'excede')}</span></td>
                    <td><span className={`fs-badge ${b.p95_within ? 'real' : 'tr-monitor'}`}>{b.p95_within ? t('within', 'dentro') : t('exceeds', 'excede')}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {sample.severity.caveats.map((c, i) => (
            <p key={i} className="fs-panel-sub">{c}</p>
          ))}
          <p className="fs-panel-sub"><b>{sample.severity.framing}</b></p>
        </div>
      ) : null}
    </div>
  );
}

function ContextView({ caseId, es }: { caseId: string; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  if (caseId === 'synthetic_battery') {
    return (
      <div className="fs-doc" style={{ maxWidth: '46rem', margin: 0 }}>
        <h3>{t('Why a synthetic battery', 'Por qué una batería sintética')}</h3>
        <p>
          {t(
            'Generated cracks are the only place where ground truth is EXACT: the centerline, the mask and the width are known by construction. That makes this case the regression gate of the whole classical stack (the pinned scientific-Python versions must reproduce these scores) and the reference for the dual width estimators (the inscribed-circle estimator lands within about 0.2 px of the true width here).',
            'Las grietas generadas son el único lugar donde el ground truth es EXACTO: la línea central, la máscara y el ancho se conocen por construcción. Eso convierte este caso en la compuerta de regresión de todo el stack clásico (las versiones fijadas de Python científico deben reproducir estos puntajes) y en la referencia de los dos estimadores de ancho (el estimador de círculo inscrito cae a unos 0.2 px del ancho verdadero aquí).',
          )}
        </p>
        <p>
          {t(
            'Read the battery as a story: the L0 floor fires on half the texture (honest failure), the mid-ladder cleans it up, and the uncracked controls show what remains: percentile thresholds always mark SOME texture maxima. The joint trap (a straight dark formwork line) is deliberately included because joints are the classic classical-pipeline false positive.',
            'Lee la batería como una historia: el piso L0 dispara sobre la mitad de la textura (falla honesta), la mitad de la escalera lo limpia, y los controles sin grieta muestran lo que queda: los umbrales por percentil siempre marcan ALGUNOS máximos de textura. La trampa de junta (una línea recta oscura de encofrado) está incluida a propósito porque las juntas son el falso positivo clásico de estos pipelines.',
          )}
        </p>
      </div>
    );
  }
  return (
    <div className="fs-doc" style={{ maxWidth: '46rem', margin: 0 }}>
      <h3>{t('The committed example set', 'El set de ejemplos versionado')}</h3>
      <p>
        {t(
          'Six patches the repository can legally ship: four from the Bridge Crack Library (CC0; two concrete cracks, one steel crack, one uncracked noise control, each with a pixel mask, inverted at curation from the source\'s black-on-white convention) and two SDNET2018 patches (CC BY 4.0; cracked and uncracked, classification-style, no masks). They are small and hard: real texture, weak contrast, a steel surface full of scratches.',
          'Seis parches que el repositorio puede publicar legalmente: cuatro de la Bridge Crack Library (CC0; dos grietas en hormigón, una en acero, un control sin grieta, cada uno con máscara de píxeles, invertida en curaduría desde la convención negro-sobre-blanco de la fuente) y dos parches SDNET2018 (CC BY 4.0; con y sin grieta, estilo clasificación, sin máscaras). Son pequeños y difíciles: textura real, contraste débil, una superficie de acero llena de rayas.',
        )}
      </p>
      <p>
        {t(
          'The honest reading of the scores: the thin clean crack reaches F1 above 0.9 at mid-ladder, the wide diffuse crack and the scratched steel patch stay much lower, and the oriented top-hat level (L2) often beats the ridge level (L3) on real texture. Classical pipelines are transparent and fast, and this case shows exactly where their ceiling sits; the learned rungs enter the same workbench in the next units.',
          'La lectura honesta de los puntajes: la grieta fina y limpia supera F1 0.9 a mitad de escalera, la grieta ancha y difusa y el parche de acero rayado quedan mucho más abajo, y el nivel de top-hat orientado (L2) a menudo supera al de crestas (L3) en textura real. Los pipelines clásicos son transparentes y rápidos, y este caso muestra exactamente dónde está su techo; los peldaños aprendidos entran a este mismo banco en las próximas unidades.',
        )}
      </p>
    </div>
  );
}
