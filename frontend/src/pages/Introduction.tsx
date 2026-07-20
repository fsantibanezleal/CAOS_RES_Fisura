import { Callout, Equation, InlineMath, Cite, Refs } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { TRACKS } from '../lib/tracks';

export default function Introduction() {
  const t = useT();
  return (
    <div className="fs-doc">
      <p className="fs-kicker">{t('Introduction', 'Introducción')}</p>
      <h1>{t('From a photo of a wall to an engineering number', 'De la foto de un muro a un número de ingeniería')}</h1>

      <p className="fs-lead">
        {t(
          'Cracks are the most universal symptom of distress in built materials: concrete shrinks and settles, pavements fatigue under traffic, masonry moves, steel corrodes under its coating. Inspection practice still finds most of them the same way it did a century ago, a person looking at a surface. Computer vision promises to look instead: earlier, everywhere, repeatably. Fisura is a research lab about what that promise is actually worth today.',
          'Las grietas son el síntoma más universal de deterioro en materiales construidos: el hormigón se contrae y asienta, los pavimentos se fatigan bajo el tráfico, la mampostería se mueve, el acero se corroe bajo su recubrimiento. La práctica de inspección todavía las encuentra igual que hace un siglo, una persona mirando una superficie. La visión por computador promete mirar por nosotros: antes, en todas partes, repetiblemente. Fisura es un laboratorio de investigación sobre cuánto vale realmente esa promesa hoy.',
        )}
      </p>
      <p>
        {t(
          'The field splits into two halves: inspection (find and classify visible defects) and monitoring (measure displacement and change over time) ',
          'El campo se divide en dos mitades: inspección (encontrar y clasificar defectos visibles) y monitoreo (medir desplazamiento y cambio en el tiempo) ',
        )}
        (<Cite id="spencer2019" />).
        {t(
          ' Fisura covers both, with one deliberate emphasis: a segmentation mask is never the end result. The lab always pushes one step further, to the quantities an engineer can use: width in millimetres, length, orientation, density, growth between visits.',
          ' Fisura cubre ambas, con un énfasis deliberado: una máscara de segmentación nunca es el resultado final. El laboratorio siempre empuja un paso más, hacia las cantidades que un ingeniero puede usar: ancho en milímetros, largo, orientación, densidad, crecimiento entre visitas.',
        )}
      </p>
      <Refs label={t('Refs','Refs')} ids={['spencer2019']} />

      <h2>{t('The whole ladder, on the same cases', 'La escalera completa, sobre los mismos casos')}</h2>
      <p>
        {t(
          'Most crack-detection repositories demonstrate one model on one dataset. Fisura is built as a comparison instrument instead: seven method tracks, from a global threshold that fails honestly to promptable foundation models, all run on the same open cases with the same metrics and the same evaluation protocol, so the reader sees what each rung of the ladder actually buys.',
          'La mayoría de los repositorios de detección de grietas demuestran un modelo sobre un dataset. Fisura está construido como un instrumento de comparación: siete pistas de métodos, desde un umbral global que falla honestamente hasta modelos fundacionales con prompts, todas corriendo sobre los mismos casos abiertos con las mismas métricas y el mismo protocolo de evaluación, para que el lector vea qué compra realmente cada peldaño de la escalera.',
        )}
      </p>
      <div className="fs-tracks">
        {TRACKS.map((tr) => (
          <div key={tr.id} className="fs-tr-cell" style={{ ['--tone' as string]: tr.tone }}>
            <div className="n">{String(tr.index).padStart(2, '0')}</div>
            <div className="t">{t(tr.en, tr.es)}</div>
            <div className="d">{t(tr.blurb_en, tr.blurb_es)}</div>
          </div>
        ))}
      </div>
      <p>
        {t(
          'The ladder has real history at both ends. The classical era peaked with structured random forests and minimal-path methods that still set respectable marks ',
          'La escalera tiene historia real en ambos extremos. La era clásica culminó con bosques aleatorios estructurados y métodos de caminos mínimos que todavía marcan registros respetables ',
        )}
        (<Cite id="shi2016crackforest" />, <Cite id="amhaz2016mps" />).
        {t(
          ' The deep era brought encoder-decoder networks purpose-built for thin structures ',
          ' La era profunda trajo redes codificador-decodificador construidas para estructuras finas ',
        )}
        (<Cite id="zou2019deepcrack" />, <Cite id="liu2019deepcrack" />).
        {t(
          ' And the foundation era is actively being sorted out: adapting the Segment Anything Model to cracks is a real research front, and the honest current answer is that it does not solve the problem: a normalization-tuned SAM reaches an F1 of 61.22 on the broadest crack benchmark, where a disciplined U-Net still wins by more than ten points ',
          ' Y la era fundacional se está decantando ahora mismo: adaptar el Segment Anything Model a grietas es un frente real de investigación, y la respuesta honesta actual es que no resuelve el problema: un SAM ajustado solo en sus normalizaciones alcanza un F1 de 61.22 en el benchmark de grietas más amplio, donde una U-Net disciplinada todavía gana por más de diez puntos ',
        )}
        (<Cite id="kirillov2023sam" />, <Cite id="sac2025" />, <Cite id="benz2024omnicrack" />).
      </p>
      <Refs label={t('Refs','Refs')} ids={['shi2016crackforest', 'amhaz2016mps', 'zou2019deepcrack', 'liu2019deepcrack', 'kirillov2023sam', 'sac2025', 'benz2024omnicrack']} />

      <h2>{t('Masks become numbers', 'Las máscaras se vuelven números')}</h2>
      <p>
        {t(
          'The flagship of the lab is quantification. Once a crack is segmented, its skeleton S and the Euclidean distance transform D of the mask give a first width estimate at every skeleton point: the diameter of the largest inscribed circle,',
          'El buque insignia del laboratorio es la cuantificación. Una vez segmentada la grieta, su esqueleto S y la transformada de distancia euclidiana D de la máscara entregan una primera estimación de ancho en cada punto del esqueleto: el diámetro del mayor círculo inscrito,',
        )}
      </p>
      <Equation
        tex={String.raw`w(s) \;=\; 2\,D(s), \qquad s \in S`}
        caption={t(
          'Inscribed-circle width from the skeleton and the distance transform. Fisura computes a second, independent estimate from profiles cast orthogonal to the local crack tangent, and treats disagreement between the two as a per-point quality flag.',
          'Ancho por círculo inscrito desde el esqueleto y la transformada de distancia. Fisura calcula una segunda estimación independiente con perfiles ortogonales a la tangente local de la grieta, y trata el desacuerdo entre ambas como una bandera de calidad por punto.',
        )}
      />
      <p>
        {t('Pixels become millimetres only through calibration. With a known standoff distance ', 'Los píxeles se vuelven milímetros solo mediante calibración. Con una distancia conocida ')}
        <InlineMath tex={String.raw`Z`} />
        {t(', focal length ', ', distancia focal ')}
        <InlineMath tex={String.raw`f`} />
        {t(' and pixel pitch ', ' y paso de píxel ')}
        <InlineMath tex={String.raw`p`} />
        {t(', the ground sampling distance is ', ', la distancia de muestreo en superficie es ')}
        <InlineMath tex={String.raw`GSD = pZ/f`} />
        {t(' and a fronto-parallel width reads ', ' y un ancho fronto-paralelo se lee ')}
        <InlineMath tex={String.raw`w_{mm} = w_{px}\cdot GSD`} />
        {t(
          '. The lab implements the two calibrations that need no special hardware (a known reference object, and a checkerboard homography for oblique planar views) and documents the laser and depth-sensor routes as literature.',
          '. El laboratorio implementa las dos calibraciones que no requieren hardware especial (un objeto de referencia conocido, y una homografía de tablero para vistas planas oblicuas) y documenta las rutas con láser y sensores de profundidad como literatura.',
        )}
      </p>
      <p>
        {t(
          'Measured widths gain meaning against published guidance: the ACI 224R-01 guide table of tolerable crack widths runs from 0.41 mm in dry exposure down to 0.10 mm for water-retaining structures, and Eurocode 2 recommends limiting calculated widths of 0.3 to 0.4 mm depending on exposure class ',
          'Los anchos medidos cobran sentido frente a guías publicadas: la tabla guía de anchos tolerables de ACI 224R-01 va de 0.41 mm en exposición seca hasta 0.10 mm para estructuras que retienen agua, y el Eurocódigo 2 recomienda anchos calculados límite de 0.3 a 0.4 mm según la clase de exposición ',
        )}
        (<Cite id="aci224r01" />, <Cite id="en1992" />).
        {t(
          ' Fisura reports width percentiles against those bands as context, and repeats the caveat both documents carry: width alone is not a reliable indicator of corrosion or structural condition. Nothing in this lab is a safety verdict.',
          ' Fisura reporta percentiles de ancho contra esas bandas como contexto, y repite el caveat que ambos documentos llevan: el ancho por sí solo no es un indicador confiable de corrosión ni de condición estructural. Nada en este laboratorio es un veredicto de seguridad.',
        )}
      </p>
      <Refs label={t('Refs','Refs')} ids={['aci224r01', 'en1992']} />

      <h2>{t('Honesty as method', 'La honestidad como método')}</h2>
      <p>
        {t(
          'Three disciplines run through everything. First, protocol transparency: crack papers report tolerance-based F-measures under incompatible protocols, so every number in Fisura states its tolerance and split next to the value. Second, dataset honesty: label noise is documented where it exists (the classification workhorse SDNET2018 ships known mislabels ',
          'Tres disciplinas cruzan todo. Primero, transparencia de protocolo: los papers de grietas reportan medidas F con tolerancias incompatibles, así que cada número en Fisura declara su tolerancia y su split junto al valor. Segundo, honestidad de datos: el ruido de etiquetas se documenta donde existe (el caballo de batalla de clasificación SDNET2018 trae errores conocidos ',
        )}
        (<Cite id="dorafshan2018sdnet" />)
        {t(
          '), licenses decide what ships in the repo, and gated datasets always have ungated fallbacks. Third, scope honesty: an optical lab sees surface openings, never depth; subsurface methods (thermography, radar, impact-echo) are documented as context and not promised.',
          '), las licencias deciden qué viaja en el repo, y los datasets con acceso restringido siempre tienen alternativas abiertas. Tercero, honestidad de alcance: un laboratorio óptico ve aperturas superficiales, nunca profundidad; los métodos de subsuperficie (termografía, radar, impact-echo) se documentan como contexto y no se prometen.',
        )}
      </p>
      <Refs label={t('Refs','Refs')} ids={['dorafshan2018sdnet']} />

      <Callout variant="note" title={t('Where to go next', 'Dónde seguir')}>
        {t(
          'Methodology explains each track and the evaluation protocol in depth. Implementation shows how the repository, the pipeline and the browser lanes are built. Experiments holds the case matrix, and Benchmark the cross-method tables with published anchors.',
          'Metodología explica cada pista y el protocolo de evaluación en profundidad. Implementación muestra cómo están construidos el repositorio, el pipeline y los carriles del navegador. Experimentos contiene la matriz de casos, y Benchmark las tablas cruzadas con las anclas publicadas.',
        )}
      </Callout>
    </div>
  );
}
