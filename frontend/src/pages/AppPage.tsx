import { Callout } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { PLANNED_CASES, TRACKS, trackById } from '../lib/tracks';
import { PanelBoundary } from '../render/PanelBoundary';

// The App landing while the engines are wired in, unit by unit. HONEST by design: it shows the real
// case matrix with real statuses and promises nothing that does not exist yet. The first workbench
// (classical ladder on a real image) replaces the placeholder area when BL-004 lands.
export default function AppPage() {
  const t = useT();
  const es = t('x', 'y') === 'y';
  return (
    <div className="fs-doc">
      <p className="fs-kicker">{t('The lab', 'El laboratorio')}</p>
      <h1>{t('Seeing damage in built materials', 'Ver el daño en materiales construidos')}</h1>
      <p className="fs-lead">
        {t(
          'One image of a concrete wall, a pavement or an industrial surface goes in. Fisura detects the damage, measures it in engineering units, and shows how every method family gets there: the same cases, the same metrics, the whole ladder.',
          'Entra una imagen de un muro de hormigón, un pavimento o una superficie industrial. Fisura detecta el daño, lo mide en unidades de ingeniería, y muestra cómo llega ahí cada familia de métodos: los mismos casos, las mismas métricas, la escalera completa.',
        )}
      </p>

      <div className="fs-wip">
        <b>{t('Under construction, honestly.', 'En construcción, honestamente.')}</b>{' '}
        {t(
          'The base shell is wired and the research layer is complete; the engines land one vertical unit at a time (code, tests and deep docs together). The classical ladder workbench arrives first, then quantification, then the learned and foundation rungs. Every case below flips its badge only when its engine is real.',
          'La base está conectada y la capa de investigación está completa; los motores aterrizan de a una unidad vertical (código, tests y documentación profunda juntos). Primero llega el banco de la escalera clásica, luego cuantificación, luego los peldaños aprendidos y fundacionales. Cada caso de abajo cambia su insignia solo cuando su motor es real.',
        )}
      </div>

      <h2>{t('The seven method tracks', 'Las siete pistas de métodos')}</h2>
      <div className="fs-tracks">
        {TRACKS.map((tr) => (
          <div key={tr.id} className="fs-tr-cell" style={{ ['--tone' as string]: tr.tone }}>
            <div className="n">{String(tr.index).padStart(2, '0')}</div>
            <div className="t">{t(tr.en, tr.es)}</div>
            <div className="d">{t(tr.blurb_en, tr.blurb_es)}</div>
          </div>
        ))}
      </div>

      <h2>{t('The case matrix (16 planned)', 'La matriz de casos (16 planificados)')}</h2>
      <PanelBoundary label={t('Case matrix', 'Matriz de casos')} es={es}>
        <div className="fs-cases">
          {PLANNED_CASES.map((c) => {
            const tr = trackById(c.track);
            return (
              <div key={c.id} className="fs-case" style={{ ['--tone' as string]: tr.tone }}>
                <div className="id">{c.id}</div>
                <div className="t">{t(c.en, c.es)}</div>
                <div className="d">{t(c.data_en, c.data_es)}</div>
                <span className={`fs-badge tr-${c.track}`}>{t(tr.en, tr.es)}</span>{' '}
                <span className={`fs-badge ${c.status}`}>
                  {c.status === 'todo' ? t('planned', 'planificado') : c.status === 'building' ? t('building', 'en construcción') : c.status}
                </span>
              </div>
            );
          })}
        </div>
      </PanelBoundary>

      <Callout variant="note" title={t('Your images stay yours', 'Tus imágenes siguen siendo tuyas')}>
        {t(
          'The live lane analyzes a photo you provide entirely in your browser: the classical pipeline runs client-side and the learned models are compact ONNX networks executed locally. Nothing is uploaded, ever.',
          'El carril en vivo analiza una foto que tú entregas por completo en tu navegador: el pipeline clásico corre del lado del cliente y los modelos aprendidos son redes ONNX compactas ejecutadas localmente. Nada se sube, nunca.',
        )}
      </Callout>
    </div>
  );
}
