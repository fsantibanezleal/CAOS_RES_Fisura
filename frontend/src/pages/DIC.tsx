import { useEffect, useState } from 'react';
import { Callout, Cite, Equation, Refs } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { OverlayLegend } from '../render/OverlayLegend';

// The DIC track (BL-012): 2D digital image correlation. A reference speckle image deformed by a KNOWN
// field (uniform stretch + a crack-opening jump) validates the in-repo subset-ZNCC engine against
// exact ground truth; the crack reads as the displacement discontinuity. Natural concrete texture is
// measured on the same field to show the accuracy cost the literature reports (~3x worse). Dossier 04 s5.

interface Dic {
  method: string;
  known_field: { uniform_strain_exx: number; crack_opening_px: number; crack_x: number };
  speckle: { u_mae_px: number; measured_mean_exx: number; measured_cod_px: number };
  natural_texture: { u_mae_px: number };
  texture_vs_speckle_error_ratio: number;
  overlays: { ref_speckle: string; u_field: string; exx_field: string; ref_texture: string };
  framing: string;
}

export default function DIC() {
  const t = useT();
  const [d, setD] = useState<Dic | null>(null);
  const [err, setErr] = useState(false);
  const [view, setView] = useState<'u_field' | 'exx_field' | 'ref_speckle' | 'ref_texture'>('u_field');
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/dic/dic.json`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then(setD)
      .catch(() => setErr(true));
  }, []);

  return (
    <>
      <p className="fs-kicker">{t('Deformation (DIC)', 'Deformación (DIC)')}</p>
      <h1>{t('The crack as a displacement discontinuity', 'La grieta como una discontinuidad de desplazamiento')}</h1>
      <p className="fs-lead">
        {t(
          'Digital image correlation reads full-field displacement and strain from images: a reference subset of pixels around each point is located in the deformed image by maximizing the zero-normalized cross-correlation. A crack reveals itself as a jump in the displacement field, often before it is visually resolvable, and the crack-opening displacement is exactly that jump.',
          'La correlación digital de imágenes lee desplazamiento y deformación de campo completo desde imágenes: un subconjunto de referencia de píxeles alrededor de cada punto se localiza en la imagen deformada maximizando la correlación cruzada normalizada a media cero. Una grieta se revela como un salto en el campo de desplazamiento, a menudo antes de ser visible, y la apertura de grieta es exactamente ese salto.',
        )}{' '}
        (<Cite id="pan2009dic" />)
      </p>

      <Equation
        tex={String.raw`C_{ZNCC} \;=\; \frac{\sum_i (f_i-\bar f)(g_i-\bar g)}{\sqrt{\sum_i (f_i-\bar f)^2}\;\sqrt{\sum_i (g_i-\bar g)^2}}`}
        caption={t(
          'The ZNCC matching criterion, invariant to affine intensity changes. Fisura implements the real subset method in-repo (integer search + sub-pixel quadratic peak); strain comes from local polynomial fits of the displacement field, never raw pointwise differentiation.',
          'El criterio de correspondencia ZNCC, invariante a cambios afines de intensidad. Fisura implementa el método de subconjuntos real en el repo (búsqueda entera + pico cuadrático subpíxel); la deformación viene de ajustes polinomiales locales del campo, nunca de diferenciación puntual cruda.',
        )}
      />

      {err ? (
        <Callout variant="note" title={t('DIC artifact baking', 'Artefacto DIC horneándose')}>
          {t('The DIC virtual experiment is not committed yet.', 'El experimento virtual DIC aún no está versionado.')}
        </Callout>
      ) : !d ? (
        <div className="fs-panel"><div className="fs-panel-t">{t('Loading...', 'Cargando...')}</div></div>
      ) : (
        <Body d={d} view={view} setView={setView} es={t('x', 'y') === 'y'} />
      )}

      <Refs label={t('Refs', 'Refs')} ids={['pan2009dic', 'olufsen2020mudic', 'jiang2023opencorr']} />
    </>
  );
}

function Body({ d, view, setView, es }: { d: Dic; view: string; setView: (v: 'u_field' | 'exx_field' | 'ref_speckle' | 'ref_texture') => void; es: boolean }) {
  const t = (en: string, esx: string) => (es ? esx : en);
  const codErr = Math.abs(d.speckle.measured_cod_px - d.known_field.crack_opening_px);
  const strainErrPct = Math.abs(d.speckle.measured_mean_exx - d.known_field.uniform_strain_exx) / d.known_field.uniform_strain_exx * 100;
  const views: { id: 'u_field' | 'exx_field' | 'ref_speckle' | 'ref_texture'; en: string; es: string }[] = [
    { id: 'u_field', en: 'u displacement', es: 'desplazamiento u' },
    { id: 'exx_field', en: 'strain e_xx', es: 'deformación e_xx' },
    { id: 'ref_speckle', en: 'speckle pattern', es: 'patrón de moteado' },
    { id: 'ref_texture', en: 'natural texture', es: 'textura natural' },
  ];
  const src = d.overlays[view as keyof typeof d.overlays];
  const isField = view === 'u_field' || view === 'exx_field';
  return (
    <>
      <div className="fs-kpis">
        <div className="fs-kpi"><div className="fs-kpi-v">{d.speckle.measured_cod_px.toFixed(2)}</div><div className="fs-kpi-l">{t('measured crack opening (px)', 'apertura de grieta medida (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{(d.speckle.measured_mean_exx * 100).toFixed(2)}%</div><div className="fs-kpi-l">{t('measured strain (known 1.00%)', 'deformación medida (conocida 1.00%)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{d.speckle.u_mae_px.toFixed(3)}</div><div className="fs-kpi-l">{t('displacement error, speckle (px)', 'error de desplazamiento, moteado (px)')}</div></div>
        <div className="fs-kpi"><div className="fs-kpi-v">{d.texture_vs_speckle_error_ratio.toFixed(1)}x</div><div className="fs-kpi-l">{t('natural texture worse than speckle', 'textura natural peor que moteado')}</div></div>
      </div>

      <div className="fs-wb-two" style={{ marginTop: '1rem' }}>
        <div className="fs-wb-img">
          <img className="fs-wb-photo" src={`${import.meta.env.BASE_URL}data/${src}`} alt={view} />
          <div className="fs-chips" style={{ marginTop: '0.4rem' }}>
            {views.map((v) => <button key={v.id} className={`chip ${view === v.id ? 'on' : ''}`} onClick={() => setView(v.id)}>{t(v.en, v.es)}</button>)}
          </div>
          {isField ? (
            <OverlayLegend items={[{ color: 'linear-gradient(90deg,rgb(40,90,220),rgb(235,60,40))', label: view === 'u_field' ? t('horizontal displacement (low to high)', 'desplazamiento horizontal (bajo a alto)') : t('strain e_xx (low to high)', 'deformación e_xx (baja a alta)'), kind: 'gradient' }]} />
          ) : null}
          <p className="fs-panel-sub">
            {view === 'u_field'
              ? t('The recovered horizontal displacement field. The smooth gradient is the 1 percent stretch; the sharp jump down the middle is the crack-opening displacement, a discontinuity the eye would miss.', 'El campo de desplazamiento horizontal recuperado. El gradiente suave es el 1 por ciento de estiramiento; el salto abrupto en el medio es la apertura de grieta, una discontinuidad que el ojo no vería.')
              : view === 'exx_field'
                ? t('The horizontal strain field from local polynomial fits. It is near-uniform at 1 percent away from the crack, and spikes at the crack line where the displacement is discontinuous.', 'El campo de deformación horizontal de ajustes polinomiales locales. Es casi uniforme al 1 por ciento lejos de la grieta, y se dispara en la línea de la grieta donde el desplazamiento es discontinuo.')
                : view === 'ref_speckle'
                  ? t('The synthetic speckle pattern: high-contrast, random, isotropic. This is the DIC gold standard, painted onto specimens for quantitative work.', 'El patrón de moteado sintético: alto contraste, aleatorio, isotrópico. Es el estándar de oro de DIC, pintado sobre las probetas para trabajo cuantitativo.')
                  : t('Natural concrete-like texture: low intensity gradients and self-similar subsets. Usable, but the same known deformation is measured far less accurately on it.', 'Textura tipo hormigón natural: gradientes de intensidad bajos y subconjuntos auto-similares. Usable, pero la misma deformación conocida se mide mucho menos exacta sobre ella.')}
          </p>
        </div>
        <div className="fs-wb-read">
          <Callout variant="strong" title={t('Recovered against exact ground truth', 'Recuperado contra ground truth exacto')}>
            {t(`On the speckle pattern the engine recovers the crack opening as ${d.speckle.measured_cod_px.toFixed(2)} px against the known ${d.known_field.crack_opening_px} px (error ${codErr.toFixed(2)} px), and the uniform strain as ${(d.speckle.measured_mean_exx * 100).toFixed(2)} percent against the known 1.00 percent (${strainErrPct.toFixed(1)} percent relative error), with a mean displacement error of ${d.speckle.u_mae_px.toFixed(3)} px. A known deformation is the only way to validate a DIC engine honestly.`, `Sobre el patrón de moteado el motor recupera la apertura de grieta como ${d.speckle.measured_cod_px.toFixed(2)} px contra los ${d.known_field.crack_opening_px} px conocidos (error ${codErr.toFixed(2)} px), y la deformación uniforme como ${(d.speckle.measured_mean_exx * 100).toFixed(2)} por ciento contra el 1.00 por ciento conocido (${strainErrPct.toFixed(1)} por ciento de error relativo), con un error medio de desplazamiento de ${d.speckle.u_mae_px.toFixed(3)} px. Una deformación conocida es la única forma de validar un motor DIC con honestidad.`)}
          </Callout>
          <Callout variant="honest" title={t('Speckle versus natural texture', 'Moteado versus textura natural')}>
            {t(`The same known field measured on natural concrete-like texture has a displacement error of ${d.natural_texture.u_mae_px.toFixed(3)} px, ${d.texture_vs_speckle_error_ratio.toFixed(1)}x worse than the painted speckle, matching the DIC literature (natural texture costs roughly a factor of three). 2D DIC also assumes a planar specimen, a perpendicular camera axis and negligible out-of-plane motion; an out-of-plane shift fakes an in-plane strain.`, `El mismo campo conocido medido sobre textura tipo hormigón natural tiene un error de desplazamiento de ${d.natural_texture.u_mae_px.toFixed(3)} px, ${d.texture_vs_speckle_error_ratio.toFixed(1)}x peor que el moteado pintado, coincidiendo con la literatura DIC (la textura natural cuesta cerca de un factor de tres). El DIC 2D además asume probeta plana, eje de cámara perpendicular y movimiento fuera de plano despreciable; un corrimiento fuera de plano finge una deformación en plano.`)}
          </Callout>
        </div>
      </div>
    </>
  );
}
