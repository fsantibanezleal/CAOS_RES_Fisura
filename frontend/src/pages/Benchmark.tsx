import { useEffect, useState } from 'react';
import { Callout, Cite, ReferenceList } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { UPlotChart } from '../render/UPlotChart';

interface ConcreteTransfer {
  method: string;
  study: string;
  dataset: string;
  n_fit_uncracked: number;
  n_test_cracked: number;
  n_test_uncracked: number;
  image_auroc: number;
  tpr_at_median: number;
  tnr_at_median: number;
  score_hist: { centers: number[]; cracked: number[]; uncracked: number[] };
  minutes: number;
}

function AnomalyTransfer({ es }: { es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const [d, setD] = useState<ConcreteTransfer | null>(null);
  const [err, setErr] = useState(false);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/anomaly/concrete_transfer.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setD)
      .catch(() => setErr(true));
  }, []);
  if (err || !d) return null;
  return (
    <div className="fs-panel" style={{ marginTop: '0.8rem' }}>
      <div className="fs-panel-t">{t('Measured: the concrete-transfer study (PatchCore, fit on uncracked concrete only)', 'Medido: el estudio de transferencia a hormigón (PatchCore, ajustado solo en hormigón sin grieta)')}</div>
      <div className="fs-kpis">
        <div className="fs-kpi"><div className="fs-kpi-v">{d.image_auroc.toFixed(3)}</div><div className="fs-kpi-l">{t('image AUROC (cracked vs uncracked)', 'AUROC de imagen (con vs sin grieta)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{(d.tpr_at_median * 100).toFixed(0)}%</div><div className="fs-kpi-l">{t('TPR at median threshold', 'TPR al umbral mediano')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{(d.tnr_at_median * 100).toFixed(0)}%</div><div className="fs-kpi-l">{t('TNR at median threshold', 'TNR al umbral mediano')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{d.n_fit_uncracked}</div><div className="fs-kpi-l">{t('uncracked fit images', 'imágenes sin grieta de ajuste')}</div></div>
      </div>
      <div className="fs-chart" style={{ marginTop: '0.6rem' }}>
        <UPlotChart
          data={[d.score_hist.centers, d.score_hist.uncracked, d.score_hist.cracked]}
          series={[
            {},
            { label: t('uncracked', 'sin grieta'), stroke: '#2da44e', width: 2 },
            { label: t('cracked', 'con grieta'), stroke: '#cf222e', width: 2 },
          ]}
          axes={[{ label: t('anomaly score', 'puntaje de anomalía') }, { label: t('count', 'conteo') }]}
          scales={{ x: { time: false } }}
          height={200}
        />
      </div>
      <p className="fs-panel-sub">
        {t(
          `PatchCore (reimplemented in-repo) fit on ${d.n_fit_uncracked} uncracked SDNET2018 concrete patches, never seeing a crack, then scored ${d.n_test_cracked} cracked + ${d.n_test_uncracked} uncracked test patches. AUROC ${d.image_auroc.toFixed(3)}: modest-but-real transfer, far below the 0.996 the same method reaches on industrial MVTec AD. The head-to-head the literature lacked, measured. Metrics only; SDNET2018 imagery stays local (CC BY 4.0).`,
          `PatchCore (reimplementado en el repo) ajustado en ${d.n_fit_uncracked} parches de hormigón SDNET2018 sin grieta, sin ver nunca una grieta, luego evaluó ${d.n_test_cracked} parches con grieta + ${d.n_test_uncracked} sin grieta. AUROC ${d.image_auroc.toFixed(3)}: transferencia modesta pero real, muy por debajo del 0.996 que el mismo método alcanza en MVTec AD industrial. El head-to-head que faltaba en la literatura, medido. Solo métricas; la imagenería SDNET2018 se queda local (CC BY 4.0).`,
        )}
      </p>
    </div>
  );
}

// The benchmark page starts as the PUBLISHED-ANCHOR record: verified numbers from the primary
// literature that Fisura's own runs must reproduce or be measured against. The lab's own numbers
// join these tables per engine unit, always at both tolerance protocols. No number without source.
export default function Benchmark() {
  const t = useT();
  return (
    <div className="fs-doc">
      <p className="fs-kicker">Benchmark</p>
      <h1>{t('Published anchors, and the protocol problem', 'Anclas publicadas, y el problema del protocolo')}</h1>
      <p className="fs-lead">
        {t(
          'Before this lab reports a single number of its own, this page fixes the published record it will be measured against, and the incompatibility that makes naive comparisons meaningless. Every value below was verified against its primary source during the research pass; every future Fisura value will appear next to these, with its protocol printed.',
          'Antes de que este laboratorio reporte un solo número propio, esta página fija el registro publicado contra el que será medido, y la incompatibilidad que vuelve sin sentido las comparaciones ingenuas. Cada valor de abajo fue verificado contra su fuente primaria durante la investigación; cada valor futuro de Fisura aparecerá junto a estos, con su protocolo impreso.',
        )}
      </p>

      <h2>{t('The classical record (5 px tolerance, F1)', 'El registro clásico (tolerancia 5 px, F1)')}</h2>
      <p>
        {t(
          'The pre-deep-learning ladder on the two classic benchmarks, as published in the structured-forest paper ',
          'La escalera previa al aprendizaje profundo en los dos benchmarks clásicos, como se publicó en el paper de bosques estructurados ',
        )}
        (<Cite id="shi2016crackforest" />):
      </p>
      <div className="fs-tablewrap">
        <table className="fs-table">
          <thead>
            <tr>
              <th>{t('Method', 'Método')}</th>
              <th>{t('Family', 'Familia')}</th>
              <th>CFD</th>
              <th>AigleRN</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>Canny</td><td>{t('edge floor', 'piso de bordes')}</td><td className="mono">15.76</td><td className="mono">n/a</td></tr>
            <tr><td>CrackTree <Cite id="zou2012cracktree" /></td><td>{t('graph tracing', 'trazado en grafo')}</td><td className="mono">70.80</td><td className="mono">n/a</td></tr>
            <tr><td>CrackIT</td><td>{t('toolbox pipeline', 'pipeline de toolbox')}</td><td className="mono">71.64</td><td className="mono">n/a</td></tr>
            <tr><td>FFA</td><td>{t('free-form anisotropy', 'anisotropía de forma libre')}</td><td className="mono">73.15</td><td className="mono">n/a</td></tr>
            <tr><td>MPS <Cite id="amhaz2016mps" /></td><td>{t('minimal path', 'camino mínimo')}</td><td className="mono">n/a</td><td className="mono">88.33</td></tr>
            <tr><td>CrackForest (SVM)</td><td>{t('structured forest', 'bosque estructurado')}</td><td className="mono">85.71</td><td className="mono">88.39</td></tr>
          </tbody>
        </table>
      </div>
      <p>
        {t(
          'These are the marks Fisura’s classical ladder reproduces first: the Canny floor is the honesty anchor, the structured forest is the classical ceiling.',
          'Estas son las marcas que la escalera clásica de Fisura reproduce primero: el piso de Canny es el ancla de honestidad, el bosque estructurado es el techo clásico.',
        )}
      </p>

      <h2>{t('The protocol trap, quantified', 'La trampa del protocolo, cuantificada')}</h2>
      <p>
        {t(
          'The FPHBN benchmark re-evaluated classical and deep methods under a different protocol (non-maximum suppression, thinning, and a tolerance proportional to the image diagonal) and reported, for example, ODS values of 0.604 on CRACK500, 0.683 on CFD, 0.517 on Cracktree200, 0.492 on AEL and 0.220 on GAPs384 for its own deep network ',
          'El benchmark FPHBN reevaluó métodos clásicos y profundos bajo otro protocolo (supresión de no máximos, adelgazamiento, y una tolerancia proporcional a la diagonal de la imagen) y reportó, por ejemplo, valores ODS de 0.604 en CRACK500, 0.683 en CFD, 0.517 en Cracktree200, 0.492 en AEL y 0.220 en GAPs384 para su propia red profunda ',
        )}
        (<Cite id="yang2019fphbn" />).
        {t(
          ' Under that protocol the same structured forest that scores 85.71 above collapses to fractions of its 5 px value. Neither table is wrong; they measure different things. This is why every Fisura table carries its protocol inline and why the lab computes both conventions for every method.',
          ' Bajo ese protocolo el mismo bosque estructurado que arriba marca 85.71 colapsa a fracciones de su valor a 5 px. Ninguna tabla está mal; miden cosas distintas. Por eso cada tabla de Fisura lleva su protocolo en línea y por eso el laboratorio calcula ambas convenciones para cada método.',
        )}
      </p>

      <h2>{t('The learned record (as published, protocols vary)', 'El registro aprendido (como se publicó, protocolos varían)')}</h2>
      <div className="fs-tablewrap">
        <table className="fs-table">
          <thead>
            <tr>
              <th>{t('Method', 'Método')}</th>
              <th>{t('Benchmark', 'Benchmark')}</th>
              <th>{t('Published result', 'Resultado publicado')}</th>
              <th>{t('Source', 'Fuente')}</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>DeepCrack (SegNet-style)</td><td>{t('own multi-set', 'multi-set propio')}</td><td className="mono">F &gt; 0.87</td><td><Cite id="zou2019deepcrack" /></td></tr>
            <tr><td>DeepCrack (hierarchical)</td><td>DeepCrack-537</td><td className="mono">mIoU 85.9, F 86.5</td><td><Cite id="liu2019deepcrack" /></td></tr>
            <tr><td>U-Net (matched harness)</td><td>{t('refined CrackSeg9k', 'CrackSeg9k refinado')}</td><td className="mono">mIoU 76.71</td><td><Cite id="li2023hrsegnet" /></td></tr>
            <tr><td>DeepLabV3+ (R18)</td><td>{t('refined CrackSeg9k', 'CrackSeg9k refinado')}</td><td className="mono">mIoU 78.29</td><td><Cite id="li2023hrsegnet" /></td></tr>
            <tr><td>HrSegNet-B48</td><td>{t('refined CrackSeg9k', 'CrackSeg9k refinado')}</td><td className="mono">mIoU 80.32 @ 140 FPS</td><td><Cite id="li2023hrsegnet" /></td></tr>
            <tr><td>CrackFormer-II</td><td>{t('four thin-crack sets', 'cuatro sets de grietas finas')}</td><td className="mono">ODS 0.869-0.914</td><td><Cite id="liu2023crackformer2" /></td></tr>
            <tr><td>{t('SAM norm-only (SAC)', 'SAM solo-normalización (SAC)')}</td><td>OmniCrack30k</td><td className="mono">F1 61.22</td><td><Cite id="sac2025" /></td></tr>
            <tr><td>nnU-Net</td><td>OmniCrack30k</td><td className="mono">{t('wins by 10+ points', 'gana por 10+ puntos')}</td><td><Cite id="benz2024omnicrack" /></td></tr>
            <tr><td>{t('dacl10k baseline / challenge best', 'línea base dacl10k / mejor del challenge')}</td><td>dacl10k</td><td className="mono">mIoU 0.42 / 0.51</td><td><Cite id="flotzinger2024dacl10k" /></td></tr>
          </tbody>
        </table>
      </div>
      <p>
        {t(
          'Two structural lessons sit in this table. First, the strongest published crack segmenter is a disciplined U-Net variant, not an exotic architecture: training discipline beats architecture novelty on this problem ',
          'Dos lecciones estructurales viven en esta tabla. Primero, el segmentador de grietas publicado más fuerte es una variante disciplinada de U-Net, no una arquitectura exótica: la disciplina de entrenamiento vence a la novedad arquitectónica en este problema ',
        )}
        (<Cite id="benz2024omnicrack" />).
        {t(
          ' Second, foundation models are competitive but not dominant on thin structures; their honest current ceiling on the broadest benchmark is ten points below the specialist.',
          ' Segundo, los modelos fundacionales son competitivos pero no dominantes en estructuras finas; su techo honesto actual en el benchmark más amplio está diez puntos bajo el especialista.',
        )}
      </p>

      <h2>{t('The anomaly record', 'El registro de anomalías')}</h2>
      <p>
        {t(
          'On the standard industrial benchmark, memory-bank methods essentially saturated image-level detection (PatchCore reports 99.6 percent AUROC on MVTec AD ',
          'En el benchmark industrial estándar, los métodos de banco de memoria esencialmente saturaron la detección a nivel de imagen (PatchCore reporta 99.6 por ciento AUROC en MVTec AD ',
        )}
        (<Cite id="roth2022patchcore" />)
        {t(
          '), which is exactly why the field built a harder successor: on MVTec AD 2, published state of the art stays below 58.7 percent AU-PRO ',
          '), que es exactamente la razón por la que el campo construyó un sucesor más duro: en MVTec AD 2, el estado del arte publicado se mantiene bajo 58.7 por ciento AU-PRO ',
        )}
        (<Cite id="heckler2026mvtec2" />).
        {t(
          ' No published head-to-head of industrial anomaly detection on civil surfaces exists, so the lab measures it directly (below).',
          ' No existe un head-to-head publicado de detección de anomalías industrial sobre superficies civiles, así que el laboratorio lo mide directamente (abajo).',
        )}
      </p>

      <AnomalyTransfer es={t('x', 'y') === 'y'} />

      <Callout variant="note" title={t('Where the Fisura numbers will appear', 'Dónde aparecerán los números de Fisura')}>
        {t(
          'As each engine unit lands, its cross-method table joins this page: every value at both 2 px and 5 px tolerances for segmentation, AU-PRO for anomaly, mIoU for multi-class, millimetre error against ground truth for quantification, always with dataset, split and protocol printed beside the number.',
          'A medida que cada unidad de motor aterriza, su tabla cruzada se suma a esta página: cada valor en tolerancias de 2 px y 5 px para segmentación, AU-PRO para anomalías, mIoU para multiclase, error en milímetros contra ground truth para cuantificación, siempre con el dataset, el split y el protocolo impresos junto al número.',
        )}
      </Callout>

      <ReferenceList ids={['shi2016crackforest', 'zou2012cracktree', 'amhaz2016mps', 'yang2019fphbn', 'zou2019deepcrack', 'liu2019deepcrack', 'li2023hrsegnet', 'liu2023crackformer2', 'sac2025', 'benz2024omnicrack', 'flotzinger2024dacl10k', 'roth2022patchcore', 'heckler2026mvtec2']} />
    </div>
  );
}
