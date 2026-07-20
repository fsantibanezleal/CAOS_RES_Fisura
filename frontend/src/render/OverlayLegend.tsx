// A compact legend for the colored overlays on the images (accessibility: identity is never
// color-alone). Each item is a swatch + label. Used under every prediction/segmentation/heat image.
export interface LegendItem { color: string; label: string; kind?: 'swatch' | 'gradient' | 'outline'; }

export function OverlayLegend({ items }: { items: LegendItem[] }) {
  return (
    <ul className="fs-legend" aria-label="overlay legend">
      {items.map((it, i) => (
        <li key={i} className="fs-legend-item">
          <span
            className={`fs-legend-sw ${it.kind ?? 'swatch'}`}
            style={it.kind === 'gradient' ? { background: it.color } : it.kind === 'outline' ? { borderColor: it.color } : { background: it.color }}
            aria-hidden="true"
          />
          <span className="fs-legend-l">{it.label}</span>
        </li>
      ))}
    </ul>
  );
}

// The standard prediction-overlay legend (family-coloured prediction + green GT + yellow overlap).
export function predictionLegend(familyColor: string, predLabel: string, gtLabel: string, overlapLabel: string, showGt: boolean): LegendItem[] {
  const items: LegendItem[] = [{ color: familyColor, label: predLabel }];
  if (showGt) {
    items.push({ color: 'rgb(46,204,113)', label: gtLabel });
    items.push({ color: 'rgb(240,210,70)', label: overlapLabel });
  }
  return items;
}
