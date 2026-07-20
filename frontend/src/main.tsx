import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import { ScanSearch } from 'lucide-react';
import { AppShell, applyTheme, readTheme, CitationsProvider, type ShellConfig } from '@fasl-work/caos-app-shell';
import '@fasl-work/caos-app-shell/styles.css';
import 'katex/dist/katex.min.css';
import './fisura.css';
import { CITATIONS } from './data/citations';
import { EXTERNAL_LINKS } from './lib/links';
import pkg from '../package.json';

import AppPage from './pages/AppPage';
import Introduction from './pages/Introduction';
import Methodology from './pages/Methodology';
import Implementation from './pages/Implementation';
import Experiments from './pages/Experiments';
import Benchmark from './pages/Benchmark';

// Display version X.XX.XXX derived from the semver manifest (single source, no drift).
const displayVersion = pkg.version
  .split('.')
  .map((p, i) => (i === 0 ? p : p.padStart(i === 1 ? 2 : 3, '0')))
  .join('.');

applyTheme(readTheme());

// Restore a deep link captured by the Pages 404 shim (public/404.html) before the router mounts.
const redirect = sessionStorage.getItem('fs-redirect');
if (redirect && redirect !== location.pathname + location.search) {
  sessionStorage.removeItem('fs-redirect');
  history.replaceState(null, '', redirect);
}

// ADR-0058 in-app Architecture / "How it works" modal. Five themed SVGs authored in
// public/svg/tech/ (floor-compliant: CSS-var tokens, no hardcoded hex, typed boxes).
const architecture: ShellConfig['architecture'] = {
  tabs: [
    {
      id: 'app',
      en: 'The app',
      es: 'La app',
      svg: 'svg/tech/01-the-app.svg',
      body_en: 'Fisura is a public research lab for seeing damage in built materials: detection, segmentation, classification and quantification of cracks and surface damage on concrete, asphalt, masonry and steel. The product is the honest cross-ladder comparison on open data, with engineering numbers (width in millimetres, length, orientation, growth) as the output, never a segmentation mask alone.',
      body_es: 'Fisura es un laboratorio público de investigación para ver daño en materiales construidos: detección, segmentación, clasificación y cuantificación de grietas y daños superficiales en hormigón, asfalto, mampostería y acero. El producto es la comparación honesta entre peldaños de la escalera sobre datos abiertos, con números de ingeniería (ancho en milímetros, largo, orientación, crecimiento) como salida, nunca solo una máscara de segmentación.',
    },
    {
      id: 'lanes',
      en: 'Lanes: web / offline / compute',
      es: 'Carriles: web / offline / cómputo',
      svg: 'svg/tech/02-lanes.svg',
      body_en: 'Three lanes. REPLAY (always): every case ships deterministic, committed artifacts with manifests; first paint is an audited result. LIVE (your browser): a user image is analyzed client-side, classical pipeline on scientific Python compiled to WebAssembly plus compact ONNX models; images never leave the browser. OFFLINE/COMPUTE (local GPU): dataset shards, training, anomalib-free PatchCore fits, ONNX export, evaluation, artifact baking. The gate decides per case.',
      body_es: 'Tres carriles. REPLAY (siempre): cada caso trae artefactos deterministas versionados con manifiestos; el primer render es un resultado auditado. EN VIVO (tu navegador): una imagen del usuario se analiza en el cliente, pipeline clásico sobre Python científico compilado a WebAssembly más modelos ONNX compactos; las imágenes nunca salen del navegador. OFFLINE/CÓMPUTO (GPU local): shards de datos, entrenamiento, ajuste de PatchCore sin anomalib, exportación ONNX, evaluación, baking de artefactos. La compuerta decide por caso.',
    },
    {
      id: 'webflow',
      en: 'The web-app flow',
      es: 'El flujo de la app web',
      svg: 'svg/tech/03-web-flow.svg',
      body_en: 'The SPA reads only committed artifacts through two data contracts: an ingestion contract (image, optional mask, optional scale) gates what enters the pipeline, and an artifact contract (compact result plus manifest with parameters, seed, runtime, lane verdict) is the single thing the app reads. A TypeScript mirror of the manifest schema fails the build on any drift. The App workbench replays masks, charts and quantification from these artifacts.',
      body_es: 'La SPA lee solo artefactos versionados a través de dos contratos de datos: un contrato de ingesta (imagen, máscara y escala opcionales) decide qué entra al pipeline, y un contrato de artefactos (resultado compacto más manifiesto con parámetros, semilla, tiempo, veredicto de carril) es lo único que lee la app. Un espejo TypeScript del esquema del manifiesto rompe el build ante cualquier deriva. El banco App reproduce máscaras, gráficos y cuantificación desde estos artefactos.',
    },
    {
      id: 'science',
      en: 'The science',
      es: 'La ciencia',
      svg: 'svg/tech/04-the-science.svg',
      body_en: 'The classical engine is a graph of nine pure seeded stages (S0 ingest, S1 flatten, S2 denoise, S3 enhance, S4 binarize, S5 structure filter, S6 link, S7 topology, S8 quantify) composed into six ladder levels L0 to L5 (Otsu floor, Sauvola, oriented top-hat with hysteresis, multiscale Hessian ridges, minimal-path bridging, fusion). Quantification runs two independent width estimators on every crack: the inscribed-circle width w(s) = 2D(s) from skeleton and distance transform, and an orthogonal-profile estimate whose disagreement is a per-point quality flag.',
      body_es: 'El motor clásico es un grafo de nueve etapas puras y con semilla (S0 ingesta, S1 aplanado, S2 denoise, S3 realce, S4 binarizado, S5 filtrado estructural, S6 enlace, S7 topología, S8 cuantificación) compuestas en seis niveles de escalera L0 a L5 (piso de Otsu, Sauvola, top-hat orientado con histéresis, crestas Hessianas multiescala, puente de caminos mínimos, fusión). La cuantificación corre dos estimadores de ancho independientes sobre cada grieta: el ancho de círculo inscrito w(s) = 2D(s) desde esqueleto y transformada de distancia, y un perfil ortogonal cuyo desacuerdo es una bandera de calidad por punto.',
    },
    {
      id: 'contracts',
      en: 'Data contracts and design',
      es: 'Contratos de datos y diseño',
      svg: 'svg/tech/05-data-contracts.svg',
      body_en: 'Sixteen cases across seven tracks (classical, learned, foundation, anomaly, multi-class, quantification, monitoring), each a workbench with a variant bar over the applicable ladder rungs and Field, Live, Charts and Context sub-tabs. Full datasets (tens of GB) live outside git on a local data volume; the repo commits only tiny contract-passing samples from permissively licensed sets (BCL CC0, SDNET2018 CC BY) plus compact derived artifacts. Non-commercial sets (dacl10k, CODEBRIM, MVTec, Kolektor) stay local; only metrics are published. Every gated set has an ungated fallback so the public repo reproduces without registration.',
      body_es: 'Dieciséis casos en siete pistas (clásico, aprendido, fundacional, anomalía, multiclase, cuantificación, monitoreo), cada uno un banco con una barra de variantes sobre los peldaños aplicables y sub-pestañas Campo, En vivo, Gráficos y Contexto. Los datasets completos (decenas de GB) viven fuera de git en un volumen local de datos; el repo versiona solo muestras diminutas que pasan el contrato de sets permisivos (BCL CC0, SDNET2018 CC BY) más artefactos derivados compactos. Los sets no comerciales (dacl10k, CODEBRIM, MVTec, Kolektor) quedan locales; solo se publican métricas. Cada set restringido tiene una alternativa abierta para que el repo público reproduzca sin registro.',
    },
  ],
};

const config: ShellConfig = {
  product: { name: 'Fisura', mark: <ScanSearch size={18} aria-hidden="true" /> },
  routes: [
    { path: '/', en: 'App', es: 'App' },
    { path: '/introduction', en: 'Introduction', es: 'Introducción' },
    { path: '/methodology', en: 'Methodology', es: 'Metodología' },
    { path: '/implementation', en: 'Implementation', es: 'Implementación' },
    { path: '/experiments', en: 'Experiments', es: 'Experimentos' },
    { path: '/benchmark', en: 'Benchmark', es: 'Benchmark' },
  ],
  links: {
    github: EXTERNAL_LINKS.github,
    personal: EXTERNAL_LINKS.personal,
    portfolio: EXTERNAL_LINKS.portfolio,
  },
  version: displayVersion,
  architecture,
  footer: {
    provenance: {
      en: 'Data: open crack datasets (CC0/CC BY in-repo, non-commercial local). Engines: scikit-image, PyTorch offline; ONNX in the browser.',
      es: 'Datos: datasets abiertos de grietas (CC0/CC BY en el repo, no comerciales locales). Motores: scikit-image, PyTorch offline; ONNX en el navegador.',
    },
    disclaimer: {
      en: 'A research lab, not a certified inspection tool. No structural safety verdict. Your images never leave the browser.',
      es: 'Un laboratorio de investigación, no una herramienta certificada. Sin veredicto de seguridad estructural. Tus imágenes nunca salen del navegador.',
    },
  },
};

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <CitationsProvider items={CITATIONS}>
        <AppShell config={config}>
          <Routes>
            <Route path="/" element={<AppPage />} />
            <Route path="/introduction" element={<Introduction />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="/implementation" element={<Implementation />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="*" element={<AppPage />} />
          </Routes>
        </AppShell>
      </CitationsProvider>
    </BrowserRouter>
  </StrictMode>,
);
