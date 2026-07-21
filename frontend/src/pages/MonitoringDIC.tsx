import { Tabs } from '@fasl-work/caos-app-shell';
import { useT } from '../lib/i18n';
import Monitoring from './Monitoring';
import DIC from './DIC';

// The temporal / deformation track (G): monitoring (two-epoch growth) + DIC (full-field deformation),
// hosted as two top-level tabs under one nav item so the header stays compact (ADR-0016 section 6).
export default function MonitoringDIC() {
  const t = useT();
  return (
    <div className="fs-doc" style={{ paddingTop: 0 }}>
      <Tabs
        ariaLabel={t('monitoring and deformation', 'monitoreo y deformación')}
        initial="monitoring"
        tabs={[
          { id: 'monitoring', label: t('Growth monitoring', 'Monitoreo de crecimiento'), content: <Monitoring /> },
          { id: 'dic', label: t('Deformation (DIC)', 'Deformación (DIC)'), content: <DIC /> },
        ]}
      />
    </div>
  );
}
