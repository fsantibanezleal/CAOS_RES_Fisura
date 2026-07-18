// Per-panel error boundary: one broken panel must never blank the whole workbench.
// Class component because React error boundaries have no hook equivalent. Mirrors the exemplar guard.
import { Component, type ReactNode } from 'react';

interface Props {
  /** Panel name shown in the fallback so a failure is identifiable in a screenshot. */
  label: string;
  es?: boolean;
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export class PanelBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidUpdate(prev: Props) {
    // A new child tree (tab or case switch) gets a fresh chance instead of a sticky fallback.
    if (prev.children !== this.props.children && this.state.error) this.setState({ error: null });
  }

  render() {
    const { error } = this.state;
    if (error) {
      return (
        <div className="fs-panel" role="alert">
          <div className="fs-panel-t">{this.props.es ? 'Panel no disponible' : 'Panel unavailable'}</div>
          <p className="fs-panel-sub">
            {this.props.es
              ? `El panel "${this.props.label}" fallo al renderizar; el resto de la app sigue operativo. Detalle: `
              : `The "${this.props.label}" panel failed to render; the rest of the app keeps working. Detail: `}
            <code>{error.message}</code>
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}
