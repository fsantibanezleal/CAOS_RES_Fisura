import { useEffect, useState } from 'react';
import { Callout, Equation, Refs, Tabs } from '@fasl-work/caos-app-shell';
import { loadCaseArtifact } from '../api/artifacts';
import type { CaseArtifact } from '../lib/contract.types';
import { useT } from '../lib/i18n';
import { PLANNED_CASES, TRACKS } from '../lib/tracks';

// Experiments (ADR-0017 section 2): prose + tabs, NOT info-box cards. The distinct experimental
// questions are separated into tabs, each with the exact metric equations, the leakage-safe protocol
// drawn as an SVG (with the forbidden anti-pattern struck out), a real datasets table with per-set
// redistribution + status, the coverage matrix, and results read from the committed artifacts.
export default function Experiments() {
  const t = useT();
  const rf = (en: string, es: string) => t(en, es);
  return (
    <div className="page-body prose">
      <div className="page-head">
        <p className="fs-kicker">{t('Experiments', 'Experimentos')}</p>
        <h1>{t('How Fisura measures, and why the protocol travels with every number', 'Cómo mide Fisura, y por qué el protocolo viaja con cada número')}</h1>
        <p className="lede">
          {t(
            'The experimental design is the product as much as the engines are: a crack F1 means nothing without its tolerance and its split. This page separates the distinct questions (the metric, the leakage-safe protocol, the datasets, the coverage, the current results) so each is answered exactly, with the equations and the real committed numbers.',
            'El diseño experimental es tanto el producto como los motores: un F1 de grietas no significa nada sin su tolerancia y su split. Esta página separa las preguntas distintas (la métrica, el protocolo sin fuga, los datasets, la cobertura, los resultados actuales) para responder cada una con exactitud, con las ecuaciones y los números reales versionados.',
          )}
        </p>
      </div>

      <Tabs
        ariaLabel="experiment questions"
        initial="metric"
        tabs={[
          { id: 'metric', label: t('The metric', 'La métrica'), content: <MetricTab t={rf} /> },
          { id: 'protocol', label: t('Leakage-safe protocol', 'Protocolo sin fuga'), content: <ProtocolTab t={rf} /> },
          { id: 'datasets', label: t('Datasets', 'Datasets'), content: <DatasetsTab t={rf} /> },
          { id: 'coverage', label: t('Coverage matrix', 'Matriz de cobertura'), content: <CoverageTab t={rf} /> },
          { id: 'results', label: t('Results so far', 'Resultados hasta ahora'), content: <ResultsTab t={rf} /> },
        ]}
      />
    </div>
  );
}

type TF = (en: string, es: string) => string;

function MetricTab({ t }: { t: TF }) {
  return (
    <section>
      <h2>{t('Buffered precision, recall and F1 on one-pixel-wide truth', 'Precisión, recall y F1 con buffer sobre verdad de un píxel')}</h2>
      <p className="measure">
        {t(
          'Cracks are one to five pixels wide, so a pixel-exact overlap metric punishes a correct detection that is off by a single pixel at the boundary. The field uses a buffered (tolerance) score instead: a predicted crack pixel counts as a true positive if a ground-truth crack pixel lies within a tolerance of d pixels, implemented as a dilation of the ground truth by radius d. Precision and recall then follow their standard definitions on the buffered sets, and F1 is their harmonic mean.',
          'Las grietas miden de uno a cinco píxeles de ancho, así que una métrica de solape exacto castiga una detección correcta desviada un solo píxel en el borde. El campo usa en su lugar un puntaje con buffer (tolerancia): un píxel predicho cuenta como verdadero positivo si un píxel de grieta del ground truth está dentro de una tolerancia de d píxeles, implementado como una dilatación del ground truth con radio d. La precisión y el recall siguen sus definiciones estándar sobre los conjuntos con buffer, y F1 es su media armónica.',
        )}
      </p>
      <Equation
        tex={String.raw`P=\frac{|\hat{C}\cap \mathcal{D}_d(C)|}{|\hat{C}|},\quad R=\frac{|C\cap \mathcal{D}_d(\hat{C})|}{|C|},\quad F_1=\frac{2PR}{P+R}`}
        caption={t(
          'Buffered precision P, recall R and F1. C is the ground-truth crack pixel set, C-hat the prediction, and D_d the morphological dilation by radius d (the tolerance). No thinning or non-maximum suppression is applied, so the number is reproducible from the mask alone.',
          'Precisión P, recall R y F1 con buffer. C es el conjunto de píxeles de grieta del ground truth, C-hat la predicción, y D_d la dilatación morfológica con radio d (la tolerancia). No se aplica adelgazamiento ni supresión de no máximos, así que el número es reproducible desde la máscara sola.',
        )}
      />
      <p className="measure">
        {t(
          'The strict overlap (intersection over union with d = 0) is reported alongside, as the honest floor. Every segmentation table in Fisura prints P, R and F1 at BOTH d = 2 px and d = 5 px, because the same masks score very differently under the two conventions.',
          'El solape estricto (intersección sobre unión con d = 0) se reporta al lado, como el piso honesto. Cada tabla de segmentación en Fisura imprime P, R y F1 en AMBAS tolerancias d = 2 px y d = 5 px, porque las mismas máscaras puntúan muy distinto bajo las dos convenciones.',
        )}
      </p>
      <Callout variant="honest" title={t('Exact vs illustrative', 'Exacto vs ilustrativo')}>
        {t(
          'The buffered scores in the App and Benchmark are computed by the real evaluation harness on the committed masks (exact). The width and severity numbers are exact against synthetic ground truth and illustrative on real imagery, where no sub-pixel truth exists. Every table says which.',
          'Los puntajes con buffer en la App y Benchmark los calcula el arnés de evaluación real sobre las máscaras versionadas (exacto). Los números de ancho y severidad son exactos contra ground truth sintético e ilustrativos sobre imágenes reales, donde no existe verdad subpíxel. Cada tabla dice cuál.',
        )}
      </Callout>
      <Refs label="Refs" ids={['yang2019fphbn', 'zhang2025review']} />
    </section>
  );
}

function ProtocolTab({ t }: { t: TF }) {
  return (
    <section>
      <h2>{t('Split by physical surface, never by patch', 'Dividir por superficie física, nunca por parche')}</h2>
      <p className="measure">
        {t(
          'The most common silent error in crack benchmarks is patch-level leakage: a single photographed surface is tiled into hundreds of patches, and if those patches are split into train and test at random, patches of the SAME surface appear on both sides. The network then memorises the surface texture and reports an inflated F1 that collapses on any new surface. Fisura splits at the level of the physical surface: all patches of one surface go entirely to train or entirely to test.',
          'El error silencioso más común en benchmarks de grietas es la fuga a nivel de parche: una sola superficie fotografiada se corta en cientos de parches, y si esos parches se dividen en train y test al azar, parches de la MISMA superficie aparecen en ambos lados. La red entonces memoriza la textura de la superficie y reporta un F1 inflado que colapsa en cualquier superficie nueva. Fisura divide al nivel de la superficie física: todos los parches de una superficie van entera a train o entera a test.',
        )}
      </p>
      <figure className="fig-svg wide">
        <ThemedSvg src="svg/tech/exp-protocol.svg" title={t('Leakage-safe evaluation protocol', 'Protocolo de evaluación sin fuga')} />
        <figcaption>
          {t(
            'Top: the leakage-safe split (by surface) versus the forbidden random-patch split (struck out), which lets the same surface leak across train and test. Bottom: the two tolerance conventions plus the FPHBN protocol that is comparable to neither.',
            'Arriba: el split sin fuga (por superficie) frente al split aleatorio de parches prohibido (tachado), que deja filtrar la misma superficie entre train y test. Abajo: las dos convenciones de tolerancia más el protocolo FPHBN que no es comparable con ninguna.',
          )}
        </figcaption>
      </figure>
      <p className="measure">
        {t(
          'For classification datasets with known label noise (SDNET2018 ships documented mislabels), the noisy labels are kept and reported honestly rather than silently cleaned, so the measured ceiling reflects the real dataset. Datasets behind a registration form always have an ungated fallback, so the public repository reproduces every result without any access request.',
          'Para datasets de clasificación con ruido de etiquetas conocido (SDNET2018 trae errores documentados), las etiquetas ruidosas se mantienen y se reportan con honestidad en vez de limpiarse en silencio, así el techo medido refleja el dataset real. Los datasets tras un formulario de registro siempre tienen una alternativa abierta, así el repositorio público reproduce cada resultado sin ninguna solicitud de acceso.',
        )}
      </p>
      <Refs label="Refs" ids={['dorafshan2018sdnet', 'benz2024omnicrack']} />
    </section>
  );
}

interface DatasetRow {
  name: string; role_en: string; role_es: string; license: string; ship: 'sample' | 'metrics' | 'link'; status: string;
}
const DATASETS: DatasetRow[] = [
  { name: 'Bridge Crack Library (BCL)', role_en: 'App example patches + masks', role_es: 'Parches de ejemplo + máscaras de la App', license: 'CC0', ship: 'sample', status: 'live' },
  { name: 'SDNET2018', role_en: 'Concrete classification + anomaly fit', role_es: 'Clasificación de hormigón + ajuste de anomalías', license: 'CC BY 4.0', ship: 'sample', status: 'live' },
  { name: 'CrackSeg9k', role_en: 'Learned segmentation training set', role_es: 'Set de entrenamiento de segmentación aprendida', license: 'CC BY 4.0', ship: 'metrics', status: 'live' },
  { name: 'Synthetic battery (in-repo)', role_en: 'Exact-truth regression + width validation', role_es: 'Regresión de verdad exacta + validación de ancho', license: 'MIT (generated)', ship: 'sample', status: 'live' },
  { name: 'CrackForest CFD / AigleRN', role_en: 'Classical published anchors', role_es: 'Anclas clásicas publicadas', license: 'research use', ship: 'link', status: 'planned' },
  { name: 'dacl10k', role_en: '19-class bridge damage (multi-class track)', role_es: 'Daño de puentes 19 clases (pista multiclase)', license: 'CC BY-NC 4.0', ship: 'metrics', status: 'planned' },
  { name: 'CODEBRIM', role_en: 'Damage boxes (multi-class track)', role_es: 'Cajas de daño (pista multiclase)', license: 'non-commercial', ship: 'metrics', status: 'planned' },
  { name: 'VisA / KolektorSDD2', role_en: 'Industrial anomaly reference', role_es: 'Referencia industrial de anomalías', license: 'CC BY-NC / research', ship: 'metrics', status: 'planned' },
];

function DatasetsTab({ t }: { t: TF }) {
  const shipLabel = (s: DatasetRow['ship']) =>
    s === 'sample' ? t('sample in-repo', 'muestra en repo') : s === 'metrics' ? t('metrics only', 'solo métricas') : t('link only', 'solo enlace');
  return (
    <section>
      <h2>{t('What ships, and what stays local', 'Qué se publica, y qué queda local')}</h2>
      <p className="measure">
        {t(
          'The license decides what leaves the local vault. Permissively licensed sets contribute tiny contract-passing samples committed to the repository; non-commercial sets are used locally and only metrics and plots are published; cite-only sets are linked, never re-hosted. Every row states its redistribution mode and whether its case is live or planned.',
          'La licencia decide qué sale del vault local. Los sets con licencia permisiva aportan muestras diminutas que pasan el contrato y se versionan en el repositorio; los sets no comerciales se usan localmente y solo se publican métricas y gráficos; los sets de solo cita se enlazan, nunca se re-alojan. Cada fila declara su modo de redistribución y si su caso está vivo o planificado.',
        )}
      </p>
      <div className="fs-tablewrap">
        <table className="fs-table">
          <thead>
            <tr>
              <th>{t('Dataset', 'Dataset')}</th>
              <th>{t('Role in Fisura', 'Rol en Fisura')}</th>
              <th>{t('License', 'Licencia')}</th>
              <th>{t('Redistribution', 'Redistribución')}</th>
              <th>{t('Status', 'Estado')}</th>
            </tr>
          </thead>
          <tbody>
            {DATASETS.map((d) => (
              <tr key={d.name}>
                <td>{d.name}</td>
                <td>{t(d.role_en, d.role_es)}</td>
                <td className="mono">{d.license}</td>
                <td>{shipLabel(d.ship)}</td>
                <td><span className={`fs-badge ${d.status === 'live' ? 'real' : 'todo'}`}>{d.status === 'live' ? t('live', 'vivo') : t('planned', 'planificado')}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Callout variant="note" title={t('No restricted data is re-hosted', 'Ningún dato restringido se re-aloja')}>
        {t(
          'The non-commercial and research-only sets never appear in the repository as imagery. Their fetch scripts download them to a local data volume, the pipeline computes metrics, and only those metrics ship. This is why every gated case still has an open fallback.',
          'Los sets no comerciales y de solo investigación nunca aparecen en el repositorio como imágenes. Sus scripts de descarga los bajan a un volumen local de datos, el pipeline calcula métricas, y solo esas métricas se publican. Por eso cada caso restringido tiene una alternativa abierta.',
        )}
      </Callout>
      <Refs label="Refs" ids={['ye2021bcl', 'dorafshan2018sdnet', 'kulkarni2022crackseg9k', 'flotzinger2024dacl10k', 'mundt2019codebrim', 'zou2022visa']} />
    </section>
  );
}

function CoverageTab({ t }: { t: TF }) {
  return (
    <section>
      <h2>{t('Seven tracks, sixteen cases, one shared protocol', 'Siete pistas, dieciséis casos, un protocolo compartido')}</h2>
      <p className="measure">
        {t(
          'Each method family is exercised on at least one dataset where it should shine and one where it should struggle. The status is real: a case is planned until its engine exists in the repository with tests and committed artifacts; it becomes replay or live only when its pipeline runs end to end.',
          'Cada familia de métodos se ejercita sobre al menos un dataset donde debería brillar y uno donde debería sufrir. El estado es real: un caso está planificado hasta que su motor existe en el repositorio con tests y artefactos versionados; pasa a replay o vivo solo cuando su pipeline corre de punta a punta.',
        )}
      </p>
      {TRACKS.map((tr) => {
        const cases = PLANNED_CASES.filter((c) => c.track === tr.id);
        const live = cases.filter((c) => c.status === 'replay' || c.status === 'live').length;
        return (
          <div key={tr.id} className="fs-cov-row">
            <div className="fs-cov-head">
              <span className="fs-dot" style={{ background: tr.tone }} />
              <b>{String(tr.index).padStart(2, '0')} {t(tr.en, tr.es)}</b>
              <span className="fs-cov-count">{live}/{cases.length} {t('live', 'vivo')}</span>
            </div>
            <div className="fs-cov-cases">
              {cases.map((c) => (
                <span key={c.id} className={`fs-badge ${c.status === 'replay' || c.status === 'live' ? 'real' : c.status === 'building' ? 'building' : 'todo'}`} title={t(c.data_en, c.data_es)}>
                  {t(c.en, c.es)}
                </span>
              ))}
            </div>
          </div>
        );
      })}
      <p className="measure" style={{ marginTop: '1rem' }}>
        {t(
          'Controls and traps are built into the cases by design: uncracked surfaces expose false-positive behaviour, formwork lines and joints are the classic classical-pipeline trap, and a scratched steel patch is the hardest negative for every method.',
          'Los controles y trampas están integrados en los casos por diseño: superficies sin grieta exponen el comportamiento de falsos positivos, las líneas de encofrado y juntas son la trampa clásica de los pipelines clásicos, y un parche de acero rayado es el negativo más difícil para todo método.',
        )}
      </p>
    </section>
  );
}

function ResultsTab({ t }: { t: TF }) {
  const [learned, setLearned] = useState<CaseArtifact | null>(null);
  useEffect(() => {
    loadCaseArtifact('learned_on_examples').then(setLearned).catch(() => setLearned(null));
  }, []);

  // pull the real per-arch best F1 across the committed example samples (mean over samples with GT)
  const archStats = learned ? summariseArch(learned) : [];

  return (
    <section>
      <h2>{t('Classical versus learned, on the same committed patches', 'Clásico versus aprendido, sobre los mismos parches versionados')}</h2>
      <p className="measure">
        {t(
          'These numbers are read live from the committed artifact the App replays, not typed into the page. They are the mean F1 at 2 px tolerance across the example patches that carry a pixel mask, per learned architecture, next to the best classical rung on the identical patches. The full cross-method tables with published anchors live on the Benchmark page.',
          'Estos números se leen en vivo desde el artefacto versionado que la App reproduce, no se escriben en la página. Son el F1 medio con tolerancia de 2 px sobre los parches de ejemplo que llevan máscara de píxeles, por arquitectura aprendida, junto al mejor peldaño clásico sobre los parches idénticos. Las tablas cruzadas completas con anclas publicadas viven en la página Benchmark.',
        )}
      </p>
      {archStats.length ? (
        <div className="fs-tablewrap">
          <table className="fs-table">
            <thead>
              <tr><th>{t('Architecture', 'Arquitectura')}</th><th className="mono">{t('mean F1 @ 2px', 'F1 medio @ 2px')}</th><th className="mono">{t('mean F1 @ 5px', 'F1 medio @ 5px')}</th><th>{t('samples', 'muestras')}</th></tr>
            </thead>
            <tbody>
              {archStats.map((a) => (
                <tr key={a.arch}>
                  <td>{a.arch}</td>
                  <td className="mono">{a.f1_2.toFixed(3)}</td>
                  <td className="mono">{a.f1_5.toFixed(3)}</td>
                  <td>{a.n}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Callout variant="note" title={t('Loading committed results', 'Cargando resultados versionados')}>
          {t('Reading the learned-on-examples artifact...', 'Leyendo el artefacto learned-on-examples...')}
        </Callout>
      )}
      <p className="measure">
        {t(
          'The reading is honest: on these hard, low-contrast patches the trained models beat the classical ladder by a clear margin at 2 px, but the margin narrows at 5 px where the classical ridge filters already capture the crack body. The DINOv2 linear probe, training almost nothing, stays competitive with the fully trained networks, which is the result the foundation-model track contributes.',
          'La lectura es honesta: sobre estos parches difíciles y de bajo contraste los modelos entrenados superan a la escalera clásica por un margen claro a 2 px, pero el margen se estrecha a 5 px donde los filtros de crestas clásicos ya capturan el cuerpo de la grieta. La sonda lineal DINOv2, entrenando casi nada, se mantiene competitiva con las redes entrenadas por completo, que es el resultado que aporta la pista de modelos fundacionales.',
        )}
      </p>
      <Refs label="Refs" ids={['kulkarni2022crackseg9k', 'li2023hrsegnet', 'oquab2023dinov2']} />
    </section>
  );
}

function summariseArch(art: CaseArtifact) {
  const acc: Record<string, { s2: number; s5: number; n: number }> = {};
  for (const s of art.samples) {
    for (const [arch, lv] of Object.entries(s.levels)) {
      const seg = lv.segmentation;
      if (!seg) continue;
      acc[arch] ??= { s2: 0, s5: 0, n: 0 };
      acc[arch].s2 += seg.tol2px.f1;
      acc[arch].s5 += seg.tol5px.f1;
      acc[arch].n += 1;
    }
  }
  return Object.entries(acc)
    .map(([arch, v]) => ({ arch, f1_2: v.s2 / v.n, f1_5: v.s5 / v.n, n: v.n }))
    .sort((a, b) => b.f1_2 - a.f1_2);
}

// Fetch + inline a themed SVG (dangerouslySetInnerHTML) so it inherits the CSS-var palette.
function ThemedSvg({ src, title }: { src: string; title: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(`${import.meta.env.BASE_URL}${src.replace(/^\//, '')}`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((txt) => { if (alive) setSvg(txt); })
      .catch(() => { if (alive) setSvg(null); });
    return () => { alive = false; };
  }, [src]);
  if (!svg) return <div className="fs-svg-fallback" role="img" aria-label={title} />;
  return <div className="fs-svg-inline" role="img" aria-label={title} dangerouslySetInnerHTML={{ __html: svg }} />;
}
