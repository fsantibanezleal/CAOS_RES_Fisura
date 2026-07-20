import type { Rose } from '../api/artifacts';

// A length-weighted orientation rose: a polar histogram of crack-segment directions over [0,180) deg,
// mirrored to a full circle so the dominant crack direction reads at a glance. Themed via CSS vars.
export function RoseDiagram({ rose, size = 220 }: { rose: Rose; size?: number }) {
  const cx = size / 2;
  const cy = size / 2;
  const R = size / 2 - 14;
  const max = Math.max(1, ...rose.weight);
  const n = rose.bins_deg.length;
  if (!n) return null;
  const wedge = (i: number, mirror: boolean) => {
    const wgt = rose.weight[i] / max;
    const r = 6 + wgt * (R - 6);
    // orientation angle -> screen angle (0 deg = horizontal East); mirror adds 180
    const a0 = ((rose.bins_deg[i] - 90 / n * 0 - (180 / n) / 2) + (mirror ? 180 : 0)) * Math.PI / 180;
    const a1 = ((rose.bins_deg[i] + (180 / n) / 2) + (mirror ? 180 : 0)) * Math.PI / 180;
    const p = (ang: number, rad: number) => `${(cx + rad * Math.cos(ang)).toFixed(1)},${(cy - rad * Math.sin(ang)).toFixed(1)}`;
    return `M ${cx},${cy} L ${p(a0, r)} A ${r},${r} 0 0 0 ${p(a1, r)} Z`;
  };
  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="fs-rose" width={size} height={size} role="img" aria-label="orientation rose">
      {[0.33, 0.66, 1].map((f) => <circle key={f} cx={cx} cy={cy} r={6 + f * (R - 6)} className="fs-rose-ring" />)}
      <line x1={cx - R} y1={cy} x2={cx + R} y2={cy} className="fs-rose-axis" />
      <line x1={cx} y1={cy - R} x2={cx} y2={cy + R} className="fs-rose-axis" />
      {rose.bins_deg.map((_, i) => <path key={`a${i}`} d={wedge(i, false)} className="fs-rose-wedge" />)}
      {rose.bins_deg.map((_, i) => <path key={`b${i}`} d={wedge(i, true)} className="fs-rose-wedge" />)}
      <text x={cx + R - 2} y={cy - 4} className="fs-rose-lbl" textAnchor="end">0&#176;</text>
      <text x={cx + 3} y={cy - R + 10} className="fs-rose-lbl">90&#176;</text>
    </svg>
  );
}
