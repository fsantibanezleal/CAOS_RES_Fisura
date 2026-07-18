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
  footer: {
    provenance: {
      en: 'Data: open crack and damage datasets, each attributed with its license (CC0 and CC BY sets sampled in-repo; non-commercial sets used locally, metrics only). Engines: scikit-image, OpenCV, PyTorch and anomalib offline on a local GPU; the classical pipeline and compact ONNX models run live in the browser.',
      es: 'Datos: datasets abiertos de grietas y daños, cada uno atribuido con su licencia (muestras CC0 y CC BY en el repo; los no comerciales se usan localmente, solo métricas). Motores: scikit-image, OpenCV, PyTorch y anomalib offline en una GPU local; el pipeline clásico y modelos ONNX compactos corren en vivo en el navegador.',
    },
    disclaimer: {
      en: 'A research lab, not a certified inspection tool: outputs are reproducible research artifacts for method comparison. Severity references quote published guidance as context; nothing here is a structural safety verdict. Your images never leave the browser.',
      es: 'Un laboratorio de investigación, no una herramienta certificada de inspección: las salidas son artefactos reproducibles para comparar métodos. Las referencias de severidad citan guías publicadas como contexto; nada aquí es un veredicto de seguridad estructural. Tus imágenes nunca salen del navegador.',
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
