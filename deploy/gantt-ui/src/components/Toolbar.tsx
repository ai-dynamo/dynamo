import { useGanttStore } from "../store";
import { useCallback, useEffect, useRef, useState } from "react";
import { LEGEND_ITEMS } from "../lib/colors";
import { fmtTime } from "../lib/time";

const PRESETS = [
  { label: "5m", sec: 300 },
  { label: "15m", sec: 900 },
  { label: "1h", sec: 3600 },
  { label: "3h", sec: 10800 },
];

export function Toolbar() {
  const rangeSec = useGanttStore((s) => s.rangeSec);
  const setRange = useGanttStore((s) => s.setRange);
  const fetchTraces = useGanttStore((s) => s.fetchTraces);
  const loading = useGanttStore((s) => s.loading);
  const traceRows = useGanttStore((s) => s.traceRows);
  const viewMin = useGanttStore((s) => s.viewMin);
  const viewMax = useGanttStore((s) => s.viewMax);
  const error = useGanttStore((s) => s.error);

  const [auto, setAuto] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined);

  const refresh = useCallback(() => {
    void fetchTraces();
  }, [fetchTraces]);

  useEffect(() => {
    refresh();
  }, [rangeSec, refresh]);

  useEffect(() => {
    if (auto) {
      intervalRef.current = setInterval(refresh, 10000);
    }
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [auto, refresh]);

  const totalSpans = traceRows.reduce((n, r) => n + r.spans.length, 0);

  return (
    <div className="flex-shrink-0 border-b border-border bg-surface">
      {/* top row */}
      <div className="flex items-center gap-3 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <div className="h-4 w-4 rounded bg-accent opacity-80" />
          <span className="text-sm font-semibold text-text-primary tracking-tight">
            KVBM Trace Timeline
          </span>
        </div>

        <div className="ml-auto flex items-center gap-1.5">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => setRange(p.sec)}
              className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                rangeSec === p.sec
                  ? "bg-accent text-white"
                  : "bg-transparent text-text-secondary hover:bg-surface-alt hover:text-text-primary"
              }`}
            >
              {p.label}
            </button>
          ))}

          <div className="mx-2 h-4 w-px bg-border" />

          <button
            onClick={refresh}
            disabled={loading}
            className="rounded-md px-3 py-1 text-xs font-medium text-text-secondary transition-colors hover:bg-surface-alt hover:text-text-primary disabled:opacity-40"
          >
            {loading ? (
              <span className="animate-pulse-subtle">Loading...</span>
            ) : (
              "Refresh"
            )}
          </button>

          <label className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-text-secondary">
            <input
              type="checkbox"
              checked={auto}
              onChange={(e) => setAuto(e.target.checked)}
              className="h-3 w-3 rounded border-border accent-accent"
            />
            Auto 10s
          </label>
        </div>
      </div>

      {/* bottom row: legend + status */}
      <div className="flex items-center gap-4 border-t border-border/50 px-4 py-1.5">
        <div className="flex items-center gap-3">
          {LEGEND_ITEMS.map((it) => (
            <span
              key={it.label}
              className="flex items-center gap-1 text-[11px] text-text-secondary"
            >
              <span
                className="inline-block h-2 w-2 rounded-sm"
                style={{ backgroundColor: it.color }}
              />
              {it.label}
            </span>
          ))}
        </div>

        <div className="ml-auto text-[11px] text-muted">
          {error ? (
            <span className="text-red-400">{error}</span>
          ) : traceRows.length > 0 ? (
            <>
              {traceRows.length} traces, {totalSpans} spans
              <span className="mx-1.5 text-border">|</span>
              {fmtTime(viewMin)} - {fmtTime(viewMax)}
            </>
          ) : loading ? (
            "Fetching traces..."
          ) : (
            "No data"
          )}
        </div>
      </div>
    </div>
  );
}
