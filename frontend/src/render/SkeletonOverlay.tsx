import { useEffect, useRef, useState } from 'react';
import type { EnrichSkeleton } from '../api/artifacts';

// The crack skeleton as a graph ON the image: branch polylines + nodes coloured by degree (endpoints
// vs junctions). Rendered as an SVG over the base photo so the nodes/edges stay crisp and hoverable.
export function SkeletonOverlay({ imageUrl, size, skeleton, hovered, onHover }: {
  imageUrl: string | null;
  size: [number, number];
  skeleton: EnrichSkeleton;
  hovered: number | null;
  onHover: (i: number | null) => void;
}) {
  const [h, w] = size;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    cv.width = w; cv.height = h;
    const ctx = cv.getContext('2d');
    if (!ctx) return;
    if (imageUrl) {
      const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0, w, h); setLoaded(true); };
      img.onerror = () => { ctx.fillStyle = '#777'; ctx.fillRect(0, 0, w, h); setLoaded(true); };
      img.src = imageUrl;
    } else { ctx.fillStyle = '#777'; ctx.fillRect(0, 0, w, h); setLoaded(true); }
  }, [imageUrl, w, h]);

  return (
    <div className="fs-skel-wrap" style={{ position: 'relative' }}>
      <canvas ref={canvasRef} className="fs-tile-canvas" />
      {loaded ? (
        <svg viewBox={`0 0 ${w} ${h}`} className="fs-skel-svg" preserveAspectRatio="xMidYMid meet">
          {skeleton.edges.map((e, i) => (
            <polyline
              key={i}
              points={e.polyline.map((p) => p.join(',')).join(' ')}
              className={`fs-skel-edge ${hovered === i ? 'on' : ''}`}
              onMouseEnter={() => onHover(i)}
              onMouseLeave={() => onHover(null)}
            />
          ))}
          {skeleton.nodes.map((n, i) => (
            <circle key={i} cx={n.x} cy={n.y} r={n.degree === 1 ? 3.2 : 4} className={`fs-skel-node ${n.degree >= 3 ? 'junction' : 'endpoint'}`} />
          ))}
        </svg>
      ) : null}
    </div>
  );
}
