import { Callout } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { PLANNED_CASES, TRACKS, trackById } from '../lib/tracks';

// The experiment matrix: 16 cases across 7 tracks, each becoming a full workbench (variant bar +
// Field / Live / Charts / Context sub-tabs) as its engine unit lands. Statuses are real, never
// aspirational: a case is "planned" until its engine exists in the repo with tests and docs.
export default function Experiments() {
  const t = useT();
  return (
    <div className="fs-doc">
      <p className="fs-kicker">{t('Experiments', 'Experimentos')}</p>
      <h1>{t('The case matrix', 'La matriz de casos')}</h1>
      <p className="fs-lead">
        {t(
          'Sixteen cases across the seven tracks, chosen so that every method family faces at least one dataset where it is expected to shine and one where it is expected to struggle. Each case becomes a workbench: a variant bar over the applicable ladder rungs, the image field with overlays, the live lane where applicable, reactive charts with value readouts, and a deep bilingual context write-up.',
          'Dieciséis casos a lo largo de las siete pistas, elegidos para que cada familia de métodos enfrente al menos un dataset donde se espera que brille y uno donde se espera que sufra. Cada caso se convierte en un banco de trabajo: una barra de variantes sobre los peldaños aplicables, el campo de imagen con superposiciones, el carril en vivo donde aplica, gráficos reactivos con lectura de valores, y un contexto bilingüe profundo.',
        )}
      </p>

      {TRACKS.map((tr) => {
        const cases = PLANNED_CASES.filter((c) => c.track === tr.id);
        if (cases.length === 0) return null;
        return (
          <section key={tr.id}>
            <h2>
              {String(tr.index).padStart(2, '0')} {t(tr.en, tr.es)}
            </h2>
            <p>{t(tr.blurb_en, tr.blurb_es)}</p>
            <div className="fs-cases">
              {cases.map((c) => (
                <div key={c.id} className="fs-case" style={{ ['--tone' as string]: trackById(c.track).tone }}>
                  <div className="id">{c.id}</div>
                  <div className="t">{t(c.en, c.es)}</div>
                  <div className="d">{t(c.data_en, c.data_es)}</div>
                  <span className={`fs-badge ${c.status}`}>
                    {c.status === 'todo' ? t('planned', 'planificado') : c.status === 'building' ? t('building', 'en construcción') : c.status}
                  </span>
                </div>
              ))}
            </div>
          </section>
        );
      })}

      <h2>{t('Controls and traps, by design', 'Controles y trampas, por diseño')}</h2>
      <p>
        {t(
          'Every workbench includes negative and sanity material: uncracked surfaces (so false-positive behaviour is visible), formwork lines and joints (the classic classical-pipeline trap), shadows and stains (the texture trap), and where the dataset provides it, acquisition masks that exclude invalid regions from every metric. Cases that use gated datasets have open fallbacks so the public repository reproduces everything without any registration.',
          'Cada banco incluye material negativo y de cordura: superficies sin grietas (para que el comportamiento de falsos positivos sea visible), líneas de encofrado y juntas (la trampa clásica de los pipelines clásicos), sombras y manchas (la trampa de textura), y donde el dataset lo provee, máscaras de adquisición que excluyen regiones inválidas de toda métrica. Los casos que usan datasets con acceso restringido tienen alternativas abiertas para que el repositorio público reproduzca todo sin registro alguno.',
        )}
      </p>

      <Callout variant="note" title={t('How a case goes live', 'Cómo un caso pasa a estar vivo')}>
        {t(
          'A case flips from planned to building when its engine unit starts, and to live or replay only when its pipeline runs end to end in the repository, its tests pass, its artifacts are committed with manifests, and its write-up is transcribed from the persisted research. The badge is the contract.',
          'Un caso pasa de planificado a en construcción cuando su unidad de motor comienza, y a vivo o replay solo cuando su pipeline corre de punta a punta en el repositorio, sus tests pasan, sus artefactos quedan versionados con manifiestos, y su texto está transcrito desde la investigación persistida. La insignia es el contrato.',
        )}
      </Callout>
    </div>
  );
}
