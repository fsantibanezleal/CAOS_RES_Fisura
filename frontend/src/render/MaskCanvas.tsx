import { useEffect, useRef } from 'react';
import { rleDecode, type MaskRLE } from '../lib/contract.types';

// The Field view: base image (committed overlay PNG) + client-side RLE mask overlays rendered on a
// canvas, so switching ladder levels and toggling ground truth is instant (no per-level image loads).
export interface MaskCanvasProps {
  imageUrl: string | null;      // the *_image.png overlay base (null -> flat background)
  size: [number, number];       // [h, w] of the mask space
  mask: MaskRLE | null;         // the selected level's mask
  gt: MaskRLE | null;           // ground truth (toggleable)
  showGt: boolean;
  opacity: number;              // 0..1 overlay opacity
}

const PRED_COLOR: [number, number, number] = [230, 57, 70];   // red family
const GT_COLOR: [number, number, number] = [46, 204, 113];    // green family

export function MaskCanvas({ imageUrl, size, mask, gt, showGt, opacity }: MaskCanvasProps) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [h, w] = size;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const drawOverlays = () => {
      const overlay = ctx.createImageData(w, h);
      const buf = overlay.data;
      if (mask) {
        const m = rleDecode(mask);
        for (let i = 0; i < m.length; i++) {
          if (m[i]) {
            const o = i * 4;
            buf[o] = PRED_COLOR[0]; buf[o + 1] = PRED_COLOR[1]; buf[o + 2] = PRED_COLOR[2];
            buf[o + 3] = Math.round(255 * opacity);
          }
        }
      }
      if (showGt && gt) {
        const g = rleDecode(gt);
        for (let i = 0; i < g.length; i++) {
          if (g[i]) {
            const o = i * 4;
            // GT drawn as a green tint; where prediction overlaps, blend toward yellow
            buf[o] = buf[o + 3] ? 240 : GT_COLOR[0];
            buf[o + 1] = buf[o + 3] ? 200 : GT_COLOR[1];
            buf[o + 2] = 60;
            buf[o + 3] = Math.max(buf[o + 3], Math.round(200 * opacity));
          }
        }
      }
      createImageBitmap(overlay).then((bmp) => {
        ctx.drawImage(bmp, 0, 0);
      });
    };

    if (imageUrl) {
      const img = new Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, w, h);
        drawOverlays();
      };
      img.onerror = () => {
        ctx.fillStyle = '#777';
        ctx.fillRect(0, 0, w, h);
        drawOverlays();
      };
      img.src = imageUrl;
    } else {
      ctx.fillStyle = '#777';
      ctx.fillRect(0, 0, w, h);
      drawOverlays();
    }
  }, [imageUrl, mask, gt, showGt, opacity, h, w]);

  return <canvas ref={ref} className="fs-canvas" style={{ width: '100%', maxWidth: 480 }} />;
}
