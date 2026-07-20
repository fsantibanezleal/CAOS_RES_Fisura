import { useEffect, useRef } from 'react';
import { rleDecode, type MaskRLE } from '../lib/contract.types';

// The Beyond-SOTA (anomaly) view: base image + a committed anomaly heat PNG blended on top, plus an
// optional thresholded anomaly-region outline (the mask) and ground truth. The heat is the per-patch
// nearest-memory distance upsampled to the image, so the reader SEES where the model finds the surface
// unlike its uncracked training set. Nothing here is a table.
export interface HeatCanvasProps {
  imageUrl: string | null;
  heatUrl: string | null;
  size: [number, number];
  mask: MaskRLE | null; // thresholded anomaly region
  gt: MaskRLE | null;
  showGt: boolean;
  showHeat: boolean;
  heatOpacity: number;
}

const GT_COLOR: [number, number, number] = [46, 204, 113];

export function HeatCanvas({ imageUrl, heatUrl, size, mask, gt, showGt, showHeat, heatOpacity }: HeatCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [h, w] = size;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawVectors = () => {
      const overlay = ctx.createImageData(w, h);
      const buf = overlay.data;
      if (mask) {
        const m = rleDecode(mask);
        // outline the thresholded anomaly region in white so it reads on any heat colour
        for (let y = 0; y < h; y++) {
          for (let x = 0; x < w; x++) {
            const i = y * w + x;
            if (!m[i]) continue;
            const edge =
              x === 0 || y === 0 || x === w - 1 || y === h - 1 ||
              !m[i - 1] || !m[i + 1] || !m[i - w] || !m[i + w];
            if (edge) {
              const o = i * 4;
              buf[o] = 255; buf[o + 1] = 255; buf[o + 2] = 255; buf[o + 3] = 235;
            }
          }
        }
      }
      if (showGt && gt) {
        const g = rleDecode(gt);
        for (let i = 0; i < g.length; i++) {
          if (g[i]) {
            const o = i * 4;
            if (buf[o + 3]) continue;
            buf[o] = GT_COLOR[0]; buf[o + 1] = GT_COLOR[1]; buf[o + 2] = GT_COLOR[2]; buf[o + 3] = 150;
          }
        }
      }
      createImageBitmap(overlay).then((bmp) => ctx.drawImage(bmp, 0, 0));
    };

    const drawBaseThen = (after: () => void) => {
      if (imageUrl) {
        const img = new Image();
        img.onload = () => { ctx.drawImage(img, 0, 0, w, h); after(); };
        img.onerror = () => { ctx.fillStyle = '#777'; ctx.fillRect(0, 0, w, h); after(); };
        img.src = imageUrl;
      } else {
        ctx.fillStyle = '#777'; ctx.fillRect(0, 0, w, h); after();
      }
    };

    drawBaseThen(() => {
      if (showHeat && heatUrl) {
        const heat = new Image();
        heat.onload = () => {
          ctx.globalAlpha = heatOpacity;
          ctx.drawImage(heat, 0, 0, w, h);
          ctx.globalAlpha = 1;
          drawVectors();
        };
        heat.onerror = () => drawVectors();
        heat.src = heatUrl;
      } else {
        drawVectors();
      }
    });
  }, [imageUrl, heatUrl, mask, gt, showGt, showHeat, heatOpacity, h, w]);

  return <canvas ref={ref} className="fs-canvas" style={{ width: '100%', maxWidth: 480 }} />;
}
