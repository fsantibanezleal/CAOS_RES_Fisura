import { Callout, Equation, Cite, ReferenceList } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';

export default function Methodology() {
  const t = useT();
  return (
    <div className="fs-doc">
      <p className="fs-kicker">{t('Methodology', 'Metodología')}</p>
      <h1>{t('Seven tracks, one instrument', 'Siete pistas, un instrumento')}</h1>
      <p className="fs-lead">
        {t(
          'Every method family in Fisura is implemented against the same staged pipeline, the same data contracts and the same evaluation harness, so differences in results reflect the methods, not the plumbing. This page explains each track at the level the lab implements it; the docs wiki in the repository carries the full derivations.',
          'Cada familia de métodos en Fisura se implementa contra el mismo pipeline por etapas, los mismos contratos de datos y el mismo arnés de evaluación, de modo que las diferencias en resultados reflejen los métodos, no la fontanería. Esta página explica cada pista al nivel en que el laboratorio la implementa; la wiki de docs del repositorio lleva las derivaciones completas.',
        )}
      </p>

      <h2>{t('1. Classical pipelines: the staged ladder', '1. Pipelines clásicos: la escalera por etapas')}</h2>
      <p>
        {t(
          'The classical engine is a graph of nine pure, seeded stages: ingest (with calibration metadata), illumination flattening, denoising, curvilinear enhancement, binarization, structure filtering, fragment linking, skeleton topology, and quantification. Each stage has switchable implementations, and six named ladder levels compose them from a deliberately weak floor (a global Otsu threshold on the raw image) through oriented morphology with hysteresis, up to multiscale Hessian ridge filters combined with morphological path openings, endpoint linking and full geometry extraction.',
          'El motor clásico es un grafo de nueve etapas puras y con semilla: ingesta (con metadatos de calibración), aplanado de iluminación, eliminación de ruido, realce curvilíneo, binarización, filtrado estructural, enlace de fragmentos, topología del esqueleto, y cuantificación. Cada etapa tiene implementaciones intercambiables, y seis niveles nombrados de la escalera las componen desde un piso deliberadamente débil (un umbral global de Otsu sobre la imagen cruda) pasando por morfología orientada con histéresis, hasta filtros de crestas Hessianos multiescala combinados con aperturas de camino morfológicas, enlace de extremos y extracción completa de geometría.',
        )}
      </p>
      <p>
        {t(
          'The rungs are anchored to the published classical record: tree-structure tracing, minimal path selection, and the structured-forest detector that closed the pre-deep era ',
          'Los peldaños están anclados al registro clásico publicado: trazado con estructura de árbol, selección de caminos mínimos, y el detector de bosques estructurados que cerró la era pre-profunda ',
        )}
        (<Cite id="zou2012cracktree" />, <Cite id="amhaz2016mps" />, <Cite id="shi2016crackforest" />).
        {t(
          ' Methods whose reference implementations no longer exist in maintained open form (percolation, minimal-path selection, free-form anisotropy) are rebuilt from their papers, validated against the behaviour and numbers those papers report. Two engineering rules from the research are binding: the scientific-Python ridge filters are version-pinned and regression-tested on synthetic bars at every upgrade, and every license-gated classical method lives behind an optional plugin boundary so the core ladder stays fully permissive.',
          ' Los métodos cuyas implementaciones de referencia ya no existen en forma abierta mantenida (percolación, selección de caminos mínimos, anisotropía de forma libre) se reconstruyen desde sus papers, validados contra el comportamiento y los números que esos papers reportan. Dos reglas de ingeniería de la investigación son vinculantes: los filtros de crestas de Python científico se fijan por versión y se someten a pruebas de regresión sobre barras sintéticas en cada actualización, y cada método clásico con licencia restringida vive tras una frontera de plugin opcional para que la escalera central sea totalmente permisiva.',
        )}
      </p>

      <h2>{t('2. Learned segmentation', '2. Segmentación aprendida')}</h2>
      <p>
        {t(
          'The learned track trains its own models, in-repo, on open datasets: a U-Net family baseline, the strong generic segmenters DeepLabV3+ and SegFormer-B2, and a faithful reimplementation of the real-time crack specialist HrSegNet with its published recipe ',
          'La pista aprendida entrena sus propios modelos, en el repo, sobre datasets abiertos: una línea base familia U-Net, los segmentadores genéricos fuertes DeepLabV3+ y SegFormer-B2, y una reimplementación fiel del especialista en tiempo real HrSegNet con su receta publicada ',
        )}
        (<Cite id="li2023hrsegnet" />).
        {t(
          ' The transformer specialist CrackFormer-II is evaluated from its released weights (its repository carries no license, so it is never vendored, only measured) ',
          ' El especialista transformer CrackFormer-II se evalúa desde sus pesos publicados (su repositorio no tiene licencia, así que nunca se incorpora, solo se mide) ',
        )}
        (<Cite id="liu2021crackformer" />, <Cite id="liu2023crackformer2" />).
        {t(
          ' Thin structures under extreme class imbalance drive the loss design: weighted cross-entropy, Dice and Tversky variants, and the topology-preserving clDice loss as an ablation case ',
          ' Las estructuras finas bajo desbalance extremo de clases gobiernan el diseño de pérdidas: entropía cruzada ponderada, variantes Dice y Tversky, y la pérdida topológica clDice como caso de ablación ',
        )}
        (<Cite id="shit2021cldice" />).
        {t(
          ' Detection-style damage localization (bounding boxes over spalling and exposed reinforcement) follows the region-based lineage that entered structural inspection with Faster R-CNN ',
          ' La localización de daño estilo detección (cajas sobre descascaramiento y armadura expuesta) sigue el linaje regional que entró a la inspección estructural con Faster R-CNN ',
        )}
        (<Cite id="cha2018damage" />).
      </p>

      <h2>{t('3. Foundation models, without the hype', '3. Modelos fundacionales, sin humo')}</h2>
      <p>
        {t(
          'The lab evaluates three honest uses of foundation vision models. As promptable annotators and mask refiners, SAM and SAM 2.1 accelerate ground-truth work. As adaptation targets, a frozen SAM encoder with LoRA adapters or normalization-only tuning is fine-tuned on crack data, following the published adapter literature. And as frozen feature extractors, DINOv2 backbones with a linear head give the cheapest credible foundation baseline, a configuration whose crack-specific numbers the literature has not established, which makes it a result the lab can contribute.',
          'El laboratorio evalúa tres usos honestos de los modelos fundacionales de visión. Como anotadores con prompt y refinadores de máscaras, SAM y SAM 2.1 aceleran el trabajo de ground truth. Como blancos de adaptación, un codificador SAM congelado con adaptadores LoRA o ajuste solo de normalizaciones se afina sobre datos de grietas, siguiendo la literatura publicada de adaptadores. Y como extractores de features congeladas, los backbones DINOv2 con una cabeza lineal dan la línea base fundacional creíble más barata, una configuración cuyos números específicos en grietas la literatura no ha establecido, lo que la vuelve un resultado que el laboratorio puede aportar.',
        )}{' '}
        (<Cite id="kirillov2023sam" />, <Cite id="ravi2024sam2" />, <Cite id="ge2024cracksam" />, <Cite id="sac2025" />, <Cite id="oquab2023dinov2" />)
      </p>

      <h2>{t('4. Unsupervised anomaly detection', '4. Detección de anomalías no supervisada')}</h2>
      <p>
        {t(
          'Industrial anomaly detection trains only on defect-free surfaces and flags deviations, which matches the economics of inspection (good surface is abundant, labeled damage is scarce). Fisura runs the established ladder (PatchCore memory banks, PaDiM, FastFlow, and the millisecond-latency EfficientAD) through the anomalib framework, reports the region-overlap metric AU-PRO rather than inflated image-level scores, and adds the study the literature is missing: how well these methods transfer from industrial textures to concrete surfaces, trained on uncracked patches from open crack datasets. The honest counterweight is stated up front: on the hardest current industrial benchmark, state of the art remains below 60 percent AU-PRO.',
          'La detección de anomalías industrial entrena solo con superficies sin defecto y marca desviaciones, lo que calza con la economía de la inspección (la superficie sana abunda, el daño etiquetado escasea). Fisura corre la escalera establecida (bancos de memoria PatchCore, PaDiM, FastFlow, y EfficientAD de latencia de milisegundos) a través del framework anomalib, reporta la métrica de solape por regiones AU-PRO en vez de puntajes inflados a nivel de imagen, y agrega el estudio que falta en la literatura: qué tan bien transfieren estos métodos desde texturas industriales a superficies de hormigón, entrenados en parches sin grieta de datasets abiertos. El contrapeso honesto se declara de entrada: en el benchmark industrial más duro actual, el estado del arte sigue bajo 60 por ciento AU-PRO.',
        )}{' '}
        (<Cite id="roth2022patchcore" />, <Cite id="batzner2024efficientad" />, <Cite id="akcay2022anomalib" />, <Cite id="bergmann2019mvtec" />, <Cite id="heckler2026mvtec2" />)
      </p>

      <h2>{t('5. Multi-class structural damage', '5. Daño estructural multiclase')}</h2>
      <p>
        {t(
          'Real inspections grade more than cracks. The multi-class track works the two verified benchmarks: dacl10k, nearly ten thousand bridge-inspection images with nineteen overlapping damage and component classes, and CODEBRIM, multi-label defect boxes over thirty bridges (crack, spallation, efflorescence, exposed bars, corrosion). Both carry non-commercial licenses, so they are used locally and only metrics are published; the detectors themselves come from permissive stacks.',
          'Las inspecciones reales gradúan más que grietas. La pista multiclase trabaja los dos benchmarks verificados: dacl10k, casi diez mil imágenes de inspección de puentes con diecinueve clases superpuestas de daño y componentes, y CODEBRIM, cajas multietiqueta de defectos sobre treinta puentes (grieta, descascaramiento, eflorescencia, barras expuestas, corrosión). Ambos tienen licencias no comerciales, así que se usan localmente y solo se publican métricas; los detectores vienen de stacks permisivos.',
        )}{' '}
        (<Cite id="flotzinger2024dacl10k" />, <Cite id="mundt2019codebrim" />)
      </p>

      <h2>{t('6. Quantification: the measurement bench', '6. Cuantificación: el banco de medición')}</h2>
      <p>
        {t(
          'Two independent width estimators run on every detected crack: the inscribed-circle width from the skeleton and distance transform, and orthogonal profile sampling with sub-pixel boundary interpolation. Their disagreement is a per-point quality flag, and junction neighbourhoods are excluded from width statistics. Length is calibrated arc length along the pruned skeleton; the skeleton graph yields orientation histograms, branch counts and crack density. Calibration uses a known reference object or a checkerboard homography; published validation studies with in-image metric references report absolute width errors around 0.05 to 0.2 mm at close range, which the lab treats as the realistic accuracy envelope.',
          'Dos estimadores de ancho independientes corren sobre cada grieta detectada: el ancho de círculo inscrito desde el esqueleto y la transformada de distancia, y el muestreo de perfiles ortogonales con interpolación subpíxel del borde. Su desacuerdo es una bandera de calidad por punto, y los vecindarios de uniones se excluyen de las estadísticas de ancho. El largo es la longitud de arco calibrada sobre el esqueleto podado; el grafo del esqueleto entrega histogramas de orientación, conteo de ramas y densidad de grietas. La calibración usa un objeto de referencia conocido o una homografía de tablero; los estudios publicados de validación con referencias métricas en imagen reportan errores absolutos de ancho de 0.05 a 0.2 mm a corta distancia, lo que el laboratorio trata como la envolvente realista de exactitud.',
        )}
      </p>

      <h2>{t('7. Monitoring and deformation', '7. Monitoreo y deformación')}</h2>
      <p>
        {t(
          'Change between inspection epochs is measured after metric registration: crack maps from two visits are aligned by homography on planar surfaces, and the honest output is per-branch width change and new-branch events, not a pixel difference heat map. On the bench, crack growth is tracked frame by frame; the published discontinuity-tracking network CrackPropNet and DIC-based crack-tip detection define the reproducible reference points ',
          'El cambio entre épocas de inspección se mide tras un registro métrico: los mapas de grietas de dos visitas se alinean por homografía en superficies planas, y la salida honesta es el cambio de ancho por rama y los eventos de rama nueva, no un mapa de calor de diferencias de píxeles. En el banco, el crecimiento de grieta se sigue cuadro a cuadro; la red publicada de seguimiento de discontinuidades CrackPropNet y la detección de punta de grieta basada en DIC definen las referencias reproducibles ',
        )}
        (<Cite id="zhu2023crackpropnet" />, <Cite id="melching2022" />).
        {t(
          ' Growth-rate framing follows fracture mechanics: image tracking measures the crack length history a(N), exactly the quantity the Paris-Erdogan law relates to the stress-intensity range,',
          ' El encuadre de tasa de crecimiento sigue la mecánica de fractura: el seguimiento por imagen mide la historia de largo de grieta a(N), exactamente la cantidad que la ley de Paris-Erdogan relaciona con el rango de intensidad de esfuerzos,',
        )}
      </p>
      <Equation
        tex={String.raw`\frac{da}{dN} \;=\; C\,(\Delta K)^m, \qquad \Delta K = K_{max}-K_{min}`}
        caption={t(
          'The Paris-Erdogan law. In Fisura it is context and bench demonstration only: it is calibrated for metals under small-scale yielding, and quantitative life prediction for quasi-brittle concrete is explicitly out of scope.',
          'La ley de Paris-Erdogan. En Fisura es contexto y demostración de banco solamente: está calibrada para metales bajo fluencia de pequeña escala, y la predicción cuantitativa de vida para hormigón cuasifrágil queda explícitamente fuera de alcance.',
        )}
      />
      <p>
        {t(
          'Deformation is measured with two-dimensional digital image correlation: a reference subset around each point is located in the deformed image by maximizing the zero-normalized cross-correlation,',
          'La deformación se mide con correlación digital de imágenes bidimensional: un subconjunto de referencia alrededor de cada punto se localiza en la imagen deformada maximizando la correlación cruzada normalizada a media cero,',
        )}
      </p>
      <Equation
        tex={String.raw`C_{ZNCC} \;=\; \frac{\sum_i (f_i-\bar f)(g_i-\bar g)}{\sqrt{\sum_i (f_i-\bar f)^2}\;\sqrt{\sum_i (g_i-\bar g)^2}}`}
        caption={t(
          'The ZNCC matching criterion, invariant to affine intensity changes. Strains come from local polynomial fits of the displacement field, never from raw pointwise differentiation.',
          'El criterio de correspondencia ZNCC, invariante a cambios afines de intensidad. Las deformaciones vienen de ajustes polinomiales locales del campo de desplazamiento, nunca de diferenciación puntual cruda.',
        )}
      />
      <p>
        {t(
          'The lab uses the MIT-licensed muDIC for virtual experiments (synthetic speckle plus a known deformation field gives closed-loop validation with exact ground truth) and the actively maintained OpenCorr as the performance reference, and quantifies honestly how much accuracy natural concrete texture costs compared to painted speckle ',
          'El laboratorio usa muDIC (licencia MIT) para experimentos virtuales (moteado sintético más un campo de deformación conocido da validación de lazo cerrado con ground truth exacto) y OpenCorr, activamente mantenido, como referencia de rendimiento, y cuantifica honestamente cuánta exactitud cuesta la textura natural del hormigón comparada con moteado pintado ',
        )}
        (<Cite id="pan2009dic" />, <Cite id="olufsen2020mudic" />, <Cite id="jiang2023opencorr" />).
      </p>

      <h2>{t('The evaluation protocol (read this before any table)', 'El protocolo de evaluación (lee esto antes de cualquier tabla)')}</h2>
      <p>
        {t(
          'Crack segmentation is evaluated with tolerance-based precision and recall: a predicted crack pixel counts as correct if ground truth lies within d pixels, which forgives sub-pixel boundary ambiguity on structures often one to five pixels wide. The field never standardized d. The classical literature reports at five pixels; the deep era popularized two; and one influential benchmark evaluates with a tolerance proportional to the image diagonal after non-maximum suppression, which is not comparable to either ',
          'La segmentación de grietas se evalúa con precisión y recall con tolerancia: un píxel predicho cuenta como correcto si el ground truth está a menos de d píxeles, lo que perdona la ambigüedad subpíxel del borde en estructuras que suelen medir de uno a cinco píxeles de ancho. El campo nunca estandarizó d. La literatura clásica reporta a cinco píxeles; la era profunda popularizó dos; y un benchmark influyente evalúa con una tolerancia proporcional a la diagonal de la imagen tras supresión de no máximos, que no es comparable con ninguna de las dos ',
        )}
        (<Cite id="yang2019fphbn" />).
        {t(
          ' The same method can read 0.85 under one protocol and 0.23 under another. Fisura therefore computes every segmentation number at BOTH the 2 px and 5 px tolerances and prints the protocol next to every value, in the app and in the docs. A recent field review confirms this incomparability as one of the open problems of the area ',
          ' El mismo método puede leer 0.85 bajo un protocolo y 0.23 bajo otro. Por eso Fisura calcula cada número de segmentación en AMBAS tolerancias, 2 px y 5 px, e imprime el protocolo junto a cada valor, en la app y en la documentación. Una revisión reciente del campo confirma esta incomparabilidad como uno de los problemas abiertos del área ',
        )}
        (<Cite id="zhang2025review" />).
      </p>

      <Callout variant="honest" title={t('What this lab does not claim', 'Lo que este laboratorio no afirma')}>
        {t(
          'No structural safety verdicts. No crack depth from a photograph (optical methods see surface openings). No field bridge monitoring claims from consumer hardware. No foundation-model magic: the numbers decide.',
          'Sin veredictos de seguridad estructural. Sin profundidad de grieta desde una fotografía (los métodos ópticos ven aperturas superficiales). Sin promesas de monitoreo de puentes en terreno con hardware de consumo. Sin magia de modelos fundacionales: deciden los números.',
        )}
      </Callout>

      <ReferenceList ids={['zou2012cracktree', 'amhaz2016mps', 'shi2016crackforest', 'li2023hrsegnet', 'liu2021crackformer', 'liu2023crackformer2', 'shit2021cldice', 'cha2018damage', 'kirillov2023sam', 'ravi2024sam2', 'ge2024cracksam', 'sac2025', 'oquab2023dinov2', 'roth2022patchcore', 'batzner2024efficientad', 'akcay2022anomalib', 'bergmann2019mvtec', 'heckler2026mvtec2', 'flotzinger2024dacl10k', 'mundt2019codebrim', 'zhu2023crackpropnet', 'melching2022', 'paris1963', 'pan2009dic', 'olufsen2020mudic', 'jiang2023opencorr', 'yang2019fphbn', 'zhang2025review']} />
    </div>
  );
}
