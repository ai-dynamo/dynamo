import { create } from "zustand";
import type { TraceRow, Span, ResponseEntry } from "./types";
import { searchTraces, fetchTrace, parseTraceRows, fetchResponses as apiFetchResponses } from "./api";

export interface HoverData {
  span: Span;
  row: TraceRow;
  x: number;
  y: number;
}

interface GanttState {
  traceRows: TraceRow[];
  dataMinNs: bigint;
  dataMaxNs: bigint;
  viewMin: bigint;
  viewMax: bigint;
  loading: boolean;
  error: string | null;
  rangeSec: number;
  hoverData: HoverData | null;
  selectedRowIndex: number | null;
  responses: Map<string, ResponseEntry>;

  setRange: (sec: number) => void;
  fetchTraces: () => Promise<void>;
  loadResponses: () => Promise<void>;
  zoomAt: (frac: number, factor: number) => void;
  pan: (deltaNsFrac: number) => void;
  resetZoom: () => void;
  brushZoom: (fracLo: number, fracHi: number) => void;
  setHover: (data: HoverData | null) => void;
  selectRow: (index: number | null) => void;
}

export const useGanttStore = create<GanttState>()((set, get) => ({
  traceRows: [],
  dataMinNs: 0n,
  dataMaxNs: 1n,
  viewMin: 0n,
  viewMax: 1n,
  loading: false,
  error: null,
  rangeSec: 3600,
  hoverData: null,
  selectedRowIndex: null,
  responses: new Map(),

  setRange(sec) {
    set({ rangeSec: sec });
  },

  setHover(data) {
    set({ hoverData: data });
  },

  selectRow(index) {
    set({ selectedRowIndex: index });
  },

  async loadResponses() {
    const map = await apiFetchResponses();
    set({ responses: map });
  },

  async fetchTraces() {
    set({ loading: true, error: null });
    try {
      const now = Math.floor(Date.now() / 1000);
      const start = now - get().rangeSec;
      const search = await searchTraces(start, now);
      const traces = search.traces ?? [];

      if (traces.length === 0) {
        set({ traceRows: [], loading: false, error: "No traces found in selected range" });
        return;
      }

      const recent = traces.slice(0, 15);
      const all: Array<{ traceID: string; data: Awaited<ReturnType<typeof fetchTrace>> }> = [];
      for (const t of recent) {
        try {
          const data = await fetchTrace(t.traceID);
          all.push({ traceID: t.traceID, data });
        } catch {
          /* skip */
        }
      }

      const rows = parseTraceRows(all);
      if (rows.length === 0) {
        set({ traceRows: [], loading: false, error: "No spans to display" });
        return;
      }

      const dMin = rows.reduce((m, r) => (r.minNs < m ? r.minNs : m), rows[0].minNs);
      const dMax = rows.reduce((m, r) => (r.maxNs > m ? r.maxNs : m), rows[0].maxNs);

      set({
        traceRows: rows,
        dataMinNs: dMin,
        dataMaxNs: dMax,
        viewMin: dMin,
        viewMax: dMax,
        loading: false,
        error: null,
        selectedRowIndex: null,
      });
    } catch (e) {
      set({ loading: false, error: (e as Error).message });
    }
  },

  zoomAt(frac, factor) {
    const { viewMin, viewMax, dataMinNs, dataMaxNs } = get();
    const curDur = viewMax - viewMin;
    const newDurF = Math.max(1e6, Number(curDur) * factor);
    if (newDurF > 1e18) return;
    const newDur = BigInt(Math.round(newDurF));
    const pivot = viewMin + BigInt(Math.round(Number(curDur) * frac));
    let nMin = pivot - BigInt(Math.round(Number(newDur) * frac));
    let nMax = nMin + newDur;
    const margin = (dataMaxNs - dataMinNs) / 10n || 1000000n;
    if (nMin < dataMinNs - margin) { nMin = dataMinNs - margin; nMax = nMin + newDur; }
    if (nMax > dataMaxNs + margin) { nMax = dataMaxNs + margin; nMin = nMax - newDur; }
    set({ viewMin: nMin, viewMax: nMax });
  },

  pan(deltaNsFrac) {
    const { viewMin, viewMax, dataMinNs, dataMaxNs } = get();
    const dur = viewMax - viewMin;
    const shift = BigInt(Math.round(Number(dur) * deltaNsFrac));
    let nMin = viewMin + shift;
    let nMax = nMin + dur;
    const margin = (dataMaxNs - dataMinNs) / 10n || 1000000n;
    if (nMin < dataMinNs - margin) { nMin = dataMinNs - margin; nMax = nMin + dur; }
    if (nMax > dataMaxNs + margin) { nMax = dataMaxNs + margin; nMin = nMax - dur; }
    set({ viewMin: nMin, viewMax: nMax });
  },

  resetZoom() {
    const { dataMinNs, dataMaxNs } = get();
    set({ viewMin: dataMinNs, viewMax: dataMaxNs });
  },

  brushZoom(fracLo, fracHi) {
    const { viewMin, viewMax } = get();
    const dur = viewMax - viewMin;
    const nMin = viewMin + BigInt(Math.round(Number(dur) * fracLo));
    const nMax = viewMin + BigInt(Math.round(Number(dur) * fracHi));
    if (nMax - nMin > 1000000n) {
      set({ viewMin: nMin, viewMax: nMax });
    }
  },
}));
