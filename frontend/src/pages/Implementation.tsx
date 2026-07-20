import { Callout, Cite, Refs } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import { ThemedSvg } from '../render/ThemedSvg';

export default function Implementation() {
  const t = useT();
  return (
    <div className="fs-doc">
      <p className="fs-kicker">{t('Implementation', 'Implementación')}</p>
      <h1>{t('How the lab is built', 'Cómo está construido el laboratorio')}</h1>
      <p className="fs-lead">
        {t(
          'Fisura follows the CAOS product archetype: an offline staged pipeline is the product, and the web app is a read-only projection of committed, audited artifacts plus a live in-browser lane. Everything is reproducible from the repository; nothing heavy lives in git.',
          'Fisura sigue el arquetipo de producto CAOS: un pipeline offline por etapas es el producto, y la aplicación web es una proyección de solo lectura de artefactos auditados y versionados, más un carril en vivo en el navegador. Todo es reproducible desde el repositorio; nada pesado vive en git.',
        )}
      </p>

      <h2>{t('Three lanes', 'Tres carriles')}</h2>
      <ul>
        <li>
          <b>{t('Offline (precompute).', 'Offline (precómputo).')}</b>{' '}
          {t(
            'The heavy lane, on a local consumer GPU (8 GB): dataset preparation into fixed crop shards, the classical ladder at scale, training of the learned rungs with mixed precision and gradient accumulation, parameter-efficient fine-tuning for foundation encoders, anomaly-model fitting, ONNX export, evaluation, and the baking of every committed artifact with its manifest.',
            'El carril pesado, en una GPU local de consumo (8 GB): preparación de datasets en shards de recortes fijos, la escalera clásica a escala, entrenamiento de los peldaños aprendidos con precisión mixta y acumulación de gradientes, fine-tuning eficiente en parámetros para codificadores fundacionales, ajuste de modelos de anomalías, exportación ONNX, evaluación, y el precálculo de cada artefacto versionado con su manifiesto.',
          )}
        </li>
        <li>
          <b>{t('Live (your browser).', 'En vivo (tu navegador).')}</b>{' '}
          {t(
            'A user-provided photo is analyzed client-side: the classical pipeline runs on scientific Python compiled to WebAssembly, and compact learned models run through ONNX Runtime Web. An optional scale input (millimetres per pixel, or a reference object) turns pixel widths into physical widths. Images never leave the browser.',
            'Una foto entregada por el usuario se analiza del lado del cliente: el pipeline clásico corre sobre Python científico compilado a WebAssembly, y modelos aprendidos compactos corren por ONNX Runtime Web. Una entrada opcional de escala (milímetros por píxel, o un objeto de referencia) convierte anchos en píxeles en anchos físicos. Las imágenes nunca salen del navegador.',
          )}
        </li>
        <li>
          <b>{t('Replay (always).', 'Replay (siempre).')}</b>{' '}
          {t(
            'Every case ships deterministic committed artifacts; the first paint is always a replay of audited results, and the live lane declares itself explicitly when it runs.',
            'Cada caso trae artefactos deterministas versionados; el primer render siempre es un replay de resultados auditados, y el carril en vivo se declara explícitamente cuando corre.',
          )}
        </li>
      </ul>

      <figure className="fig-svg wide">
        <ThemedSvg src="svg/tech/impl-architecture.svg" title={t('Offline pipeline, committed artifact, browser lane', 'Pipeline offline, artefacto versionado, carril del navegador')} />
        <figcaption>
          {t(
            'The offline lane (local GPU) shards datasets, trains and fits the models, evaluates at both tolerances, exports ONNX with parity checks, and bakes the compact artifacts. The web app reads ONLY those committed artifacts, gated by a TypeScript mirror of the manifest schema that fails the build on any drift, and replays them; the live in-browser lane (BL-013) runs the classical subset and compact ONNX on a photo that never leaves the browser.',
            'El carril offline (GPU local) fragmenta datasets, entrena y ajusta los modelos, evalúa en ambas tolerancias, exporta ONNX con chequeos de paridad, y hornea los artefactos compactos. La aplicación web lee SOLO esos artefactos versionados, con una compuerta de espejo TypeScript del esquema del manifiesto que rompe el build ante cualquier deriva, y los reproduce; el carril en vivo en el navegador (BL-013) corre el subconjunto clásico y ONNX compacto sobre una foto que nunca sale del navegador.',
          )}
        </figcaption>
      </figure>

      <h2>{t('Two data contracts', 'Dos contratos de datos')}</h2>
      <p>
        {t(
          'An ingestion contract (image, optional mask, optional scale metadata, with declared units, ranges and an explicit outlier policy) is the bring-your-own-data gate: a dataset enters the pipeline only if it satisfies the contract. An artifact contract (a compact standard-format result plus a manifest with parameters, seed, runtime, sizes and the lane verdict) is the only thing the web app reads, and a TypeScript mirror of the manifest schema makes any drift fail the build.',
          'Un contrato de ingesta (imagen, máscara opcional, metadatos de escala opcionales, con unidades declaradas, rangos y una política explícita de outliers) es la puerta de traer-tus-propios-datos: un dataset entra al pipeline solo si satisface el contrato. Un contrato de artefactos (un resultado compacto en formato estándar más un manifiesto con parámetros, semilla, tiempo de ejecución, tamaños y el veredicto de carril) es lo único que lee la aplicación web, y un espejo TypeScript del esquema del manifiesto hace que cualquier deriva rompa el build.',
        )}
      </p>

      <h2>{t('Storage and reproducibility', 'Almacenamiento y reproducibilidad')}</h2>
      <p>
        {t(
          'Full open datasets (tens of gigabytes) live outside git on a local data volume and are fetched by idempotent scripts committed to the repository; the repo itself commits only tiny contract-passing samples from permissively licensed sets, plus the compact derived artifacts whose licenses allow publication. Model weights and training checkpoints live on a local model volume. Two pinned Python environments separate the slim runtime from the heavy pipeline. Every run is a pure function of its parameters and seed.',
          'Los datasets abiertos completos (decenas de gigabytes) viven fuera de git en un volumen local de datos y se descargan con scripts idempotentes versionados en el repositorio; el repo solo versiona muestras diminutas que pasan el contrato, de sets con licencia permisiva, más los artefactos derivados compactos cuya licencia permite publicar. Los pesos de modelos y checkpoints de entrenamiento viven en un volumen local de modelos. Dos entornos Python fijados separan el runtime liviano del pipeline pesado. Cada corrida es una función pura de sus parámetros y semilla.',
        )}
      </p>

      <h2>{t('The license architecture', 'La arquitectura de licencias')}</h2>
      <ul>
        <li>{t('The lab code is MIT, and the default ladder on every track uses only permissive engines (BSD, Apache, MIT).', 'El código del laboratorio es MIT, y la escalera por defecto de cada pista usa solo motores permisivos (BSD, Apache, MIT).')}</li>
        <li>{t('Non-commercial datasets (the multi-label bridge benchmarks, the industrial anomaly sets) are used locally; only metrics and plots are published.', 'Los datasets no comerciales (los benchmarks multietiqueta de puentes, los sets industriales de anomalías) se usan localmente; solo se publican métricas y gráficos.')}</li>
        <li>{t('License-gated classical methods live behind optional plugin boundaries; unlicensed released weights are evaluated, never vendored.', 'Los métodos clásicos con licencia restringida viven tras fronteras de plugin opcionales; los pesos publicados sin licencia se evalúan, nunca se incorporan.')}</li>
        <li>{t('AGPL detection stacks are an isolated optional extra executed as a subprocess in their own environment, never linked into the MIT core and never redistributed.', 'Los stacks de detección AGPL son un extra opcional aislado ejecutado como subproceso en su propio entorno, nunca enlazados al núcleo MIT y nunca redistribuidos.')}</li>
      </ul>

      <h2>{t('Quality machinery', 'Maquinaria de calidad')}</h2>
      <p>
        {t(
          'Continuous integration runs linting, the test suite (contracts, determinism, stages, the lane gate), a pipeline smoke that regenerates one case end to end, and content guards (no tracked secrets or heavy data, no leaked local paths, no template residue, content-standard checks). The evaluation harness prints its protocol next to every number. Every panel of the app is screenshot-verified in both themes before a deploy is called done.',
          'La integración continua corre linting, la suite de tests (contratos, determinismo, etapas, la compuerta de carril), un smoke del pipeline que regenera un caso de punta a punta, y guardias de contenido (sin secretos ni datos pesados versionados, sin rutas locales filtradas, sin residuo de plantilla, chequeos del estándar de contenido). El arnés de evaluación imprime su protocolo junto a cada número. Cada panel de la app se verifica con capturas en ambos temas antes de declarar listo un despliegue.',
        )}
      </p>

      <Callout variant="note" title={t('Current state, stated plainly', 'Estado actual, dicho claramente')}>
        {t(
          'Wired today: the shell and the six pages, the research layer, the dataset and model acquisition, and the real engines. The classical ladder (S0 to S8 stages, L0 to L5 levels) and the quantification flagship (dual width estimators, sub-pixel FWHM, ACI and Eurocode severity bands) are live. The learned ladder ships U-Net, DeepLabV3+ and SegFormer-B2 trained on CrackSeg9k, an in-repo HrSegNet reimplementation, and a DINOv2 frozen-features linear probe, all exported to ONNX with parity checks. The anomaly track ships an in-repo PatchCore and the concrete-transfer study (image AUROC 0.72). Five of sixteen cases are replayable in the workbench. Remaining rungs (PaDiM, FastFlow, EfficientAD, multi-class dacl10k and CODEBRIM, monitoring, DIC, the live browser lane) land in their own units, each flipping its cases from planned to real with tests and docs in the same commit.',
          'Conectado hoy: la carcasa y las seis páginas, la capa de investigación, la adquisición de datasets y modelos, y los motores reales. La escalera clásica (etapas S0 a S8, niveles L0 a L5) y el buque insignia de cuantificación (estimadores duales de ancho, FWHM subpíxel, bandas de severidad ACI y Eurocódigo) están vivos. La escalera aprendida trae U-Net, DeepLabV3+ y SegFormer-B2 entrenados sobre CrackSeg9k, una reimplementación de HrSegNet en el repo, y una sonda lineal sobre features congeladas de DINOv2, todos exportados a ONNX con chequeos de paridad. La pista de anomalías trae un PatchCore en el repo y el estudio de transferencia a hormigón (AUROC por imagen 0.72). Cinco de dieciséis casos son reproducibles en el banco. Los peldaños restantes (PaDiM, FastFlow, EfficientAD, multiclase dacl10k y CODEBRIM, monitoreo, DIC, el carril en vivo del navegador) landan en sus propias unidades, cada una cambiando sus casos de planificado a real con tests y docs en el mismo commit.',
        )}
      </Callout>

      <p>
        {t('Datasets used across the lab include, among others, ', 'Los datasets usados a lo largo del laboratorio incluyen, entre otros, ')}
        <Cite id="kulkarni2022crackseg9k" />, <Cite id="ye2021bcl" />, <Cite id="dorafshan2018sdnet" />, <Cite id="ozgenel2018" />, <Cite id="zou2022visa" />
        {t(', each documented with its license and retrieval mechanism in the data registry.', ', cada uno documentado con su licencia y mecanismo de descarga en el registro de datos.')}
      </p>
      <Refs label={t('Refs','Refs')} ids={['kulkarni2022crackseg9k', 'ye2021bcl', 'dorafshan2018sdnet', 'ozgenel2018', 'zou2022visa']} />
    </div>
  );
}
