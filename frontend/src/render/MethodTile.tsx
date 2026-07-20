import { useEffect, useRef } from 'react';
import { rleDecode, type MaskRLE } from '../lib/contract.types';

// A single method's prediction rendered small, ON the image: base photo + prediction mask (family
// colour) + optional ground truth (green). Used to tile ALL methods over ONE image at once, so the
// reader compares every method on the same picture. The heat PNG (anomaly) blends under the mask.
export interface MethodTileProps {
  imageUrl: string | null;
  size: [number, number];
  mask: MaskRLE | null;
  gt: MaskRLE | null;
  showGt: boolean;
  opacity: number;
  color: [number, number, number]; // family colour for the prediction
  heatUrl?: string | null;          // optional anomaly heat under the mask
  heatOpacity?: number;
}

const GT: [number, number, number] = [46, 204, 113];

export function MethodTile({ imageUrl, size, mask, gt, showGt, opacity, color, heatUrl, heatOpacity = 0.7 }: MethodTileProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [h, w] = size;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawMask = () => {
      const ov = ctx.createImageData(w, h);
      const buf = ov.data;
      if (mask) {
        const m = rleDecode(mask);
        for (let i = 0; i < m.length; i++) {
          if (m[i]) {
            const o = i * 4;
            buf[o] = color[0]; buf[o + 1] = color[1]; buf[o + 2] = color[2];
            buf[o + 3] = Math.round(255 * opacity);
          }
        }
      }
      if (showGt && gt) {
        const g = rleDecode(gt);
        for (let i = 0; i < g.length; i++) {
          if (g[i]) {
            const o = i * 4;
            if (buf[o + 3]) { buf[o] = 240; buf[o + 1] = 210; buf[o + 2] = 70; }
            else { buf[o] = GT[0]; buf[o + 1] = GT[1]; buf[o + 2] = GT[2]; buf[o + 3] = Math.round(190 * opacity); }
          }
        }
      }
      createImageBitmap(ov).then((b) => ctx.drawImage(b, 0, 0));
    };

    const paint = () => {
      if (heatUrl) {
        const heat = new Image();
        heat.onload = () => { ctx.globalAlpha = heatOpacity; ctx.drawImage(heat, 0, 0, w, h); ctx.globalAlpha = 1; drawMask(); };
        heat.onerror = drawMask;
        heat.src = heatUrl;
      } else drawMask();
    };

    if (imageUrl) {
      const img = new Image();
      img.onload = () => { ctx.drawImage(img, 0, 0, w, h); paint(); };
      img.onerror = () => { ctx.fillStyle = '#777'; ctx.fillRect(0, 0, w, h); paint(); };
      img.src = imageUrl;
    } else { ctx.fillStyle = '#777'; ctx.fillRect(0, 0, w, h); paint(); }
  }, [imageUrl, mask, gt, showGt, opacity, color, heatUrl, heatOpacity, h, w]);

  return <canvas ref={ref} className="fs-tile-canvas" />;
}
