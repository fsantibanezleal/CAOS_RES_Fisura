import { useEffect, useState } from 'react';

// Fetch + inline a themed SVG (dangerouslySetInnerHTML) so it inherits the app's CSS-var palette and
// follows light/dark. An <img> would not inherit the variables (ADR-0058). Fetched via BASE_URL so it
// resolves under the GitHub-Pages subpath.
export function ThemedSvg({ src, title }: { src: string; title: string }) {
  const [svg, setSvg] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    fetch(`${import.meta.env.BASE_URL}${src.replace(/^\//, '')}`)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(String(r.status)))))
      .then((txt) => { if (alive) setSvg(txt); })
      .catch(() => { if (alive) setSvg(null); });
    return () => { alive = false; };
  }, [src]);
  if (!svg) return <div className="fs-svg-fallback" role="img" aria-label={title} />;
  return <div className="fs-svg-inline" role="img" aria-label={title} dangerouslySetInnerHTML={{ __html: svg }} />;
}
