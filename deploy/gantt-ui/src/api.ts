import type { TraceRow, TempoSearchResult, TempoTraceData, ResponseEntry } from "./types";

export async function searchTraces(
  startSec: number,
  endSec: number,
  limit = 20,
): Promise<TempoSearchResult> {
  const q = encodeURIComponent('{ name = "llm_request" }');
  const url = `/api/search?start=${startSec}&end=${endSec}&q=${q}&limit=${limit}`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Tempo search ${resp.status}: ${await resp.text()}`);
  return resp.json();
}

export async function fetchTrace(traceID: string): Promise<TempoTraceData> {
  const resp = await fetch(`/api/traces/${traceID}`);
  if (!resp.ok) throw new Error(`Tempo trace ${resp.status}`);
  return resp.json();
}

export async function fetchResponses(): Promise<Map<string, ResponseEntry>> {
  try {
    const resp = await fetch("/data/responses.json");
    if (!resp.ok) return new Map();
    const entries: ResponseEntry[] = await resp.json();
    const map = new Map<string, ResponseEntry>();
    for (const entry of entries) {
      const parts = entry.traceparent.split("-");
      if (parts.length >= 3) {
        map.set(parts[1], entry);
      }
    }
    return map;
  } catch {
    return new Map();
  }
}

function extractAttrs(
  attrs: Array<{ key: string; value: { stringValue?: string; intValue?: string } }> | undefined,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const a of attrs ?? []) {
    out[a.key] = a.value?.stringValue ?? a.value?.intValue ?? "";
  }
  return out;
}

export function parseTraceRows(
  allTraceData: Array<{ traceID: string; data: TempoTraceData }>,
): TraceRow[] {
  const rows: TraceRow[] = [];

  for (const { traceID, data } of allTraceData) {
    const spans: TraceRow["spans"] = [];
    let reqId = "";
    let rootStartNs = 0n;
    let rootEndNs = 0n;
    const mergedAttrs: Record<string, string> = {};

    const batches = data.batches ?? data.resourceSpans ?? [];
    for (const batch of batches) {
      for (const ss of batch.scopeSpans ?? []) {
        for (const s of ss.spans ?? []) {
          const startNs = BigInt(s.startTimeUnixNano ?? "0");
          const endNs = BigInt(s.endTimeUnixNano ?? "0");
          const name = s.name ?? "";
          const attrs = extractAttrs(s.attributes);

          if (attrs["request_id"]) reqId = attrs["request_id"];
          Object.assign(mergedAttrs, attrs);

          if (name === "llm_request") {
            rootStartNs = startNs;
            rootEndNs = endNs;
          } else if (endNs > startNs) {
            spans.push({ name, startNs, endNs, durMs: Number(endNs - startNs) / 1e6, attributes: attrs });
          }
        }
      }
    }

    if (spans.length > 0) {
      spans.sort((a, b) => (a.startNs < b.startNs ? -1 : 1));
      const minNs = spans[0].startNs;
      const maxNs = spans.reduce((mx, s) => (s.endNs > mx ? s.endNs : mx), spans[0].endNs);
      const totalStart = rootStartNs > 0n ? rootStartNs : minNs;
      const totalEnd = rootEndNs > 0n ? rootEndNs : maxNs;
      rows.push({
        traceID,
        label: reqId || traceID.substring(0, 16),
        spans,
        minNs,
        maxNs,
        totalDurMs: Number(totalEnd - totalStart) / 1e6,
        attributes: mergedAttrs,
      });
    }
  }

  rows.sort((a, b) => (a.minNs < b.minNs ? -1 : 1));
  return rows;
}
