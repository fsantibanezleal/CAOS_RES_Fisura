import { useEffect, useState } from 'react';
import { Callout, Cite, Refs } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { UPlotChart } from '../render/UPlotChart';

// The multi-class damage track (BL-010): dacl10k 19-class multi-label semantic segmentation. The App's
// binary crack ladder becomes real bridge-inspection grading (spalling, efflorescence, exposed rebars,
// corrosion, ...). dacl10k is CC BY-NC 4.0: the images stay local, only metrics + low-res coloured
// overlays ship. Every number is measured against the WACV 2024 paper baseline (0.424 mIoU).

interface Sample { id: string; overlay: string; present_classes: string[]; per_class_iou: Record<string, number>; }
interface Dacl {
  dataset: string; arch: string; classes: string[]; damage_classes: string[]; palette: number[][];
  val_mIoU: number; baseline_mIoU: number; baseline_source: string;
  per_class_IoU: Record<string, number>; n_train: number; n_val: number; epochs: number; samples: Sample[];
}

export default function MultiClass() {
  const t = useT();
  const [d, setD] = useState<Dacl | null>(null);
  const [err, setErr] = useState(false);
  const [pick, setPick] = useState(0);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/multiclass/dacl10k.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setD)
      .catch(() => setErr(true));
  }, []);

  return (
    <div className="fs-doc">
      <p className="fs-kicker">{t('Multi-class damage', 'Daño multiclase')}</p>
      <h1>{t('Beyond the binary crack: 19-class bridge damage', 'Más allá de la grieta binaria: daño de puentes en 19 clases')}</h1>
      <p className="fs-lead">
        {t(
          'Real bridge inspection grades far more than cracks. This track trains a multi-label semantic segmenter on dacl10k, nearly ten thousand inspection images with nineteen overlapping damage and component classes (a single pixel can be several at once), and measures it against the published baseline.',
          'La inspección real de puentes gradúa mucho más que grietas. Esta pista entrena un segmentador semántico multietiqueta sobre dacl10k, casi diez mil imágenes de inspección con diecinueve clases superpuestas de daño y componentes (un píxel puede ser varias a la vez), y lo mide contra la línea base publicada.',
        )}{' '}
        (<Cite id="flotzinger2024dacl10k" />)
      </p>

      {err ? (
        <Callout variant="note" title={t('Metrics baking', 'Métricas horneándose')}>
          {t('The dacl10k metrics artifact is not committed yet. The model trains on the local GPU (the images are CC BY-NC 4.0 and never leave the vault); the measured numbers and low-resolution coloured overlays land here when the run bakes.', 'El artefacto de métricas de dacl10k aún no está versionado. El modelo entrena en la GPU local (las imágenes son CC BY-NC 4.0 y nunca salen del vault); los números medidos y las superposiciones de baja resolución aterrizan aquí cuando la corrida hornea.')}
        </Callout>
      ) : !d ? (
        <div className="fs-panel"><div className="fs-panel-t">{t('Loading...', 'Cargando...')}</div></div>
      ) : (
        <MultiClassBody d={d} pick={pick} setPick={setPick} es={t('x', 'y') === 'y'} />
      )}

      <Refs label={t('Refs', 'Refs')} ids={['flotzinger2024dacl10k', 'mundt2019codebrim', 'benz2024omnicrack']} />
    </div>
  );
}

function MultiClassBody({ d, pick, setPick, es }: { d: Dacl; pick: number; setPick: (i: number) => void; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const beat = d.val_mIoU >= d.baseline_mIoU;
  const rgb = (c: string) => { const i = d.classes.indexOf(c); const p = d.palette[i] ?? [136, 136, 136]; return `rgb(${p[0]},${p[1]},${p[2]})`; };
  // per-class IoU bar (sorted desc), the honest imbalance view
  const entries = Object.entries(d.per_class_IoU).sort((a, b) => b[1] - a[1]);
  const xs = entries.map((_, i) => i);
  const sample = d.samples[pick] ?? d.samples[0];

  return (
    <>
      <div className="fs-kpis">
        <div className="fs-kpi"><div className="fs-kpi-v">{d.val_mIoU.toFixed(3)}</div><div className="fs-kpi-l">{t('measured val mIoU', 'mIoU val medido')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{d.baseline_mIoU.toFixed(3)}</div><div className="fs-kpi-l">{t('WACV 2024 baseline', 'línea base WACV 2024')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{d.classes.length}</div><div className="fs-kpi-l">{t('classes (13 damage + 6 object)', 'clases (13 daño + 6 objeto)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{d.n_train.toLocaleString()}</div><div className="fs-kpi-l">{t('train images (local)', 'imágenes de entrenamiento (local)')}</div></div>
      </div>

      <Callout variant={beat ? 'strong' : 'honest'} title={beat ? t('Beats the published baseline', 'Supera la línea base publicada') : t('Measured against the published baseline', 'Medido contra la línea base publicada')}>
        {beat
          ? t(`The model reaches mIoU ${d.val_mIoU.toFixed(3)}, above the ${d.baseline_mIoU} of the WACV 2024 paper baseline (${d.baseline_source}). Trained on ${d.n_train} images for ${d.epochs} epochs on the local GPU.`, `El modelo alcanza mIoU ${d.val_mIoU.toFixed(3)}, sobre el ${d.baseline_mIoU} de la línea base del paper WACV 2024. Entrenado en ${d.n_train} imágenes por ${d.epochs} épocas en la GPU local.`)
          : t(`The model reaches mIoU ${d.val_mIoU.toFixed(3)} against the ${d.baseline_mIoU} paper baseline (${d.baseline_source}), a single-model run on ${d.n_train} images for ${d.epochs} epochs. The published baseline uses the full 6,935-image train set; the gap is the honest cost of the reduced budget, stated rather than hidden.`, `El modelo alcanza mIoU ${d.val_mIoU.toFixed(3)} contra el ${d.baseline_mIoU} de la línea base del paper, una corrida de un solo modelo sobre ${d.n_train} imágenes por ${d.epochs} épocas. La línea base publicada usa el set completo de 6,935 imágenes; la brecha es el costo honesto del presupuesto reducido, declarado y no escondido.`)}
      </Callout>

      {sample ? (
        <div className="fs-wb-two" style={{ marginTop: '1rem' }}>
          <div className="fs-wb-img">
            <img className="fs-wb-photo" src={`${import.meta.env.BASE_URL}data/${sample.overlay}`} alt={sample.id} />
            <div className="fs-chips" style={{ marginTop: '0.4rem' }}>
              {d.samples.map((s, i) => <button key={s.id} className={`chip ${i === pick ? 'on' : ''}`} onClick={() => setPick(i)}>{`#${i + 1}`}</button>)}
            </div>
            <p className="fs-panel-sub">{t('The model prediction, class-coloured, on a low-resolution dacl10k inspection crop (CC BY-NC 4.0; full images stay local).', 'La predicción del modelo, coloreada por clase, sobre un recorte de baja resolución de inspección dacl10k (CC BY-NC 4.0; las imágenes completas quedan locales).')}</p>
          </div>
          <div className="fs-wb-read">
            <div className="fs-panel-t">{t('Classes present in this image', 'Clases presentes en esta imagen')}</div>
            <ul className="fs-legend" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
              {sample.present_classes.map((c) => (
                <li key={c} className="fs-legend-item">
                  <span className="fs-legend-sw" style={{ background: rgb(c) }} aria-hidden="true" />
                  <span className="fs-legend-l">{c}{sample.per_class_iou[c] != null ? ` · IoU ${sample.per_class_iou[c].toFixed(2)}` : ''}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ) : null}

      {entries.length ? (
        <div className="fs-panel" style={{ marginTop: '1rem' }}>
          <div className="fs-panel-t">{t('Per-class IoU (the honest imbalance)', 'IoU por clase (el desbalance honesto)')}</div>
          <UPlotChart
            data={[xs, entries.map((e) => e[1])]}
            series={[{}, { label: 'IoU', stroke: 'var(--fs-multiclass)', width: 2, points: { show: true, size: 5 } }]}
            axes={[{ values: (_u, ticks) => ticks.map((v) => (Number.isInteger(v) ? entries[v]?.[0] ?? '' : '')), splits: () => xs, rotate: -40 }, { label: 'IoU', size: 50 }]}
            scales={{ x: { time: false }, y: { range: [0, 1] } }}
            height={240}
          />
          <p className="fs-panel-sub">{t('Common, large-area classes (graffiti, weathering, bearings) segment well; rare thin classes (cracks, exposed rebars) are far harder. The mean over the present classes is the reported mIoU. This spread is why the multi-label task is hard and why a single headline number hides most of the story.', 'Las clases comunes de área grande (graffiti, weathering, apoyos) se segmentan bien; las clases raras y finas (grietas, barras expuestas) son mucho más difíciles. La media sobre las clases presentes es el mIoU reportado. Esta dispersión es por qué la tarea multietiqueta es difícil y por qué un solo número esconde casi toda la historia.')}</p>
        </div>
      ) : null}

      <Callout variant="note" title={t('License and honesty', 'Licencia y honestidad')}>
        {t('dacl10k is CC BY-NC 4.0: non-commercial, no redistribution of the raw images. Fisura trains locally and publishes only the metrics and small transformative low-resolution overlays. Trained dacl10k weights carry the same non-commercial notice, never the MIT license of the lab code. CODEBRIM bounding-box detection (RT-DETR, Faster R-CNN) is the next rung of this track.', 'dacl10k es CC BY-NC 4.0: no comercial, sin redistribución de las imágenes crudas. Fisura entrena localmente y publica solo las métricas y pequeñas superposiciones transformadas de baja resolución. Los pesos entrenados de dacl10k llevan el mismo aviso no comercial, nunca la licencia MIT del código del laboratorio. La detección por cajas de CODEBRIM (RT-DETR, Faster R-CNN) es el siguiente peldaño de esta pista.')}
      </Callout>
    </>
  );
}
