import { useEffect } from "react";
import { Toolbar } from "./components/Toolbar";
import { GanttCanvas } from "./components/GanttCanvas";
import { Tooltip } from "./components/Tooltip";
import { DetailPanel } from "./components/DetailPanel";
import { useGanttStore } from "./store";

export default function App() {
  const loading = useGanttStore((s) => s.loading);
  const traceRows = useGanttStore((s) => s.traceRows);
  const error = useGanttStore((s) => s.error);
  const loadResponses = useGanttStore((s) => s.loadResponses);

  useEffect(() => {
    void loadResponses();
  }, [loadResponses]);

  return (
    <div className="flex h-screen flex-col bg-background">
      <Toolbar />
      <div className="relative min-h-0 flex-1 overflow-hidden">
        {loading && traceRows.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-border border-t-accent" />
              <p className="text-sm text-text-secondary">Loading traces...</p>
            </div>
          </div>
        ) : error && traceRows.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <p className="mb-1 text-sm text-red-400">{error}</p>
              <p className="text-xs text-muted">
                Try expanding the time range or sending some requests first.
              </p>
            </div>
          </div>
        ) : traceRows.length === 0 ? (
          <div className="flex h-full items-center justify-center">
            <div className="text-center">
              <p className="mb-1 text-sm text-text-secondary">No traces found</p>
              <p className="text-xs text-muted">
                Send some requests, then hit Refresh.
              </p>
            </div>
          </div>
        ) : (
          <GanttCanvas />
        )}
      </div>
      <DetailPanel />
      <Tooltip />
    </div>
  );
}
