import { useEffect, useRef } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

// A small reactive uPlot wrapper: recreates on option/size change, setData on data change. Reads the shell
// tokens for axis/grid colours so it follows light/dark. Mirrors the validated ImageLab wrapper.
export interface UPlotChartProps {
  data: uPlot.AlignedData;
  series: uPlot.Series[];
  axes?: uPlot.Axis[];
  scales?: uPlot.Scales;
  height?: number;
  legend?: boolean;
}

function tokenColor(name: string, fallback: string): string {
  if (typeof window === 'undefined') return fallback;
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function UPlotChart({ data, series, axes, scales, height = 260, legend = true }: UPlotChartProps) {
  const hostRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<uPlot | null>(null);

  useEffect(() => {
    if (!hostRef.current) return;
    const grid = tokenColor('--color-border', '#3336');
    const fg = tokenColor('--color-fg-subtle', '#888');
    const width = hostRef.current.clientWidth || 520;
    const baseAxis: Partial<uPlot.Axis> = {
      stroke: fg,
      grid: { stroke: grid, width: 1 },
      ticks: { stroke: grid, width: 1 },
    };
    const opts: uPlot.Options = {
      width,
      height,
      series,
      axes: (axes ?? [{}, {}]).map((a) => ({ ...baseAxis, ...a })),
      scales: scales ?? {},
      legend: { show: legend },
      cursor: { focus: { prox: 16 } },
    };
    plotRef.current = new uPlot(opts, data, hostRef.current);
    const onResize = () => plotRef.current?.setSize({ width: hostRef.current!.clientWidth, height });
    window.addEventListener('resize', onResize);
    return () => {
      window.removeEventListener('resize', onResize);
      plotRef.current?.destroy();
      plotRef.current = null;
    };
    // recreate only on structural change (series identity / height)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, height]);

  useEffect(() => {
    plotRef.current?.setData(data);
  }, [data]);

  return <div ref={hostRef} className="fs-uplot fs-chart" />;
}
