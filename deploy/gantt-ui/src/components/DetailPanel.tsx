import { useState, useRef, useEffect, useCallback } from "react";
import { useGanttStore } from "../store";
import { spanColor } from "../lib/colors";
import { fmtDuration } from "../lib/time";
import type { TraceRow, Span, ResponseEntry } from "../types";

type Tab = "response" | "spans" | "blocks" | "attributes";

const MIN_H = 120;
const MAX_H = 800;
const DEFAULT_H = 320;

const PROMPT_PATTERNS: Record<string, string> = {
  v0: '"hi ok" repeated',
  v1: '"hey if" repeated',
  v2: '"yo so" repeated',
  v3: '"ok go" repeated',
};

export function DetailPanel() {
  const selectedRowIndex = useGanttStore((s) => s.selectedRowIndex);
  const traceRows = useGanttStore((s) => s.traceRows);
  const responses = useGanttStore((s) => s.responses);
  const selectRow = useGanttStore((s) => s.selectRow);
  const [tab, setTab] = useState<Tab>("response");
  const [panelH, setPanelH] = useState(DEFAULT_H);
  const dragging = useRef(false);
  const dragStartY = useRef(0);
  const dragStartH = useRef(0);

  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragging.current = true;
    dragStartY.current = e.clientY;
    dragStartH.current = panelH;
  }, [panelH]);

  useEffect(() => {
    if (!dragging.current) return;

    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return;
      const dy = dragStartY.current - e.clientY;
      setPanelH(Math.max(MIN_H, Math.min(MAX_H, dragStartH.current + dy)));
    };
    const onUp = () => { dragging.current = false; };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  });

  if (selectedRowIndex === null) return null;
  const row = traceRows[selectedRowIndex];
  if (!row) return null;

  const resp = responses.get(row.traceID);

  const blockSpans = row.spans.filter((s) => {
    const a = s.attributes;
    return a["num_blocks"] || a["host_matched"] || a["disk_matched"] || a["g4_matched"]
      || a["num_external_blocks"] || a["block_ids"] || a["device_block_ids"];
  });

  const tabs: { id: Tab; label: string }[] = [
    { id: "response", label: "I/O" },
    { id: "spans", label: `Spans (${row.spans.length})` },
    { id: "blocks", label: `Blocks${blockSpans.length ? ` (${blockSpans.length})` : ""}` },
    { id: "attributes", label: "Attributes" },
  ];

  return (
    <div className="flex-shrink-0 border-t border-border bg-surface" style={{ height: panelH }}>
      {/* drag handle */}
      <div
        onMouseDown={onDragStart}
        className="group flex h-2 cursor-row-resize items-center justify-center hover:bg-accent/10"
      >
        <div className="h-0.5 w-12 rounded-full bg-border transition-colors group-hover:bg-accent" />
      </div>

      {/* header */}
      <div className="flex items-center gap-3 border-b border-border/50 px-4 py-1.5">
        <div className="flex items-center gap-1.5">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                tab === t.id
                  ? "bg-accent text-white"
                  : "text-text-secondary hover:bg-surface-alt hover:text-text-primary"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>

        <div className="ml-auto flex items-center gap-3">
          <span className="font-mono text-[11px] text-muted">{row.traceID}</span>
          <span className="text-xs text-text-secondary">
            Total: <span className="text-text-primary font-medium">{fmtDuration(row.totalDurMs)}</span>
          </span>
          <button
            onClick={() => selectRow(null)}
            className="rounded p-1 text-muted transition-colors hover:bg-surface-alt hover:text-text-primary"
          >
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <path d="M3.5 3.5L10.5 10.5M10.5 3.5L3.5 10.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>

      {/* body */}
      <div className="overflow-auto px-4 py-3" style={{ height: panelH - 42 }}>
        {tab === "response" && <ResponseTab row={row} resp={resp} />}
        {tab === "spans" && <SpansTab row={row} />}
        {tab === "blocks" && <BlocksTab row={row} blockSpans={blockSpans} />}
        {tab === "attributes" && <AttributesTab row={row} resp={resp} />}
      </div>
    </div>
  );
}

function ResponseTab({ row, resp }: { row: TraceRow; resp: ResponseEntry | undefined }) {
  if (!resp) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <p className="text-sm text-text-secondary">No response data available</p>
          <p className="mt-1 text-xs text-muted">
            Serve <span className="font-mono">responses.json</span> at{" "}
            <span className="font-mono">/data/responses.json</span> to see I/O data.
          </p>
        </div>
      </div>
    );
  }

  const promptDesc = PROMPT_PATTERNS[resp.prompt_key] ?? resp.prompt_key;

  return (
    <div className="flex flex-col gap-4">
      {/* metadata bar */}
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <Badge label="Scenario" value={resp.scenario} />
        <Badge label="ISL" value={resp.isl.toLocaleString()} />
        <Badge label="Prompt" value={resp.prompt_key} />
        <Badge label="Finish" value={resp.finish_reason} />
        <Badge label="Prompt Tokens" value={resp.prompt_tokens.toLocaleString()} />
        <Badge label="Completion Tokens" value={resp.completion_tokens.toLocaleString()} />
        {resp.has_reasoning && <Badge label="Reasoning" value="Yes" />}
        <Badge label="Seed" value={String(resp.seed)} />
      </div>

      {/* input */}
      <div>
        <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-muted">Input</p>
        <div className="rounded-lg border border-border/60 bg-background p-3 text-sm leading-relaxed text-text-secondary">
          <span className="text-text-primary font-medium">[{resp.prompt_key}]</span>{" "}
          {promptDesc}, ~{resp.isl.toLocaleString()} tokens
          <span className="ml-2 text-muted">({resp.prompt_tokens.toLocaleString()} prompt tokens)</span>
        </div>
      </div>

      {/* output */}
      <div>
        <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wider text-muted">Output</p>
        <div className="whitespace-pre-wrap rounded-lg border border-border/60 bg-background p-3 text-sm leading-relaxed text-text-primary">
          {resp.completion}
        </div>
      </div>
    </div>
  );
}

function SpansTab({ row }: { row: TraceRow }) {
  const sorted = [...row.spans].sort((a, b) => (a.startNs < b.startNs ? -1 : 1));
  const earliest = sorted[0]?.startNs ?? 0n;

  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-border/50 text-left text-[11px] uppercase tracking-wider text-muted">
          <th className="pb-2 pr-4 font-medium">Span</th>
          <th className="pb-2 pr-4 font-medium">Offset</th>
          <th className="pb-2 pr-4 font-medium">Duration</th>
          <th className="pb-2 pr-4 font-medium">Key Attrs</th>
          <th className="pb-2 font-medium">Timeline</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((span, i) => {
          const offsetMs = Number(span.startNs - earliest) / 1e6;
          const totalMs = Number((sorted[sorted.length - 1]?.endNs ?? span.endNs) - earliest) / 1e6;
          const leftPct = totalMs > 0 ? (offsetMs / totalMs) * 100 : 0;
          const widthPct = totalMs > 0 ? Math.max((span.durMs / totalMs) * 100, 0.5) : 100;
          const keyAttrs = formatKeyAttrs(span);

          return (
            <tr key={i} className="border-b border-border/30 hover:bg-surface-alt/50">
              <td className="py-1.5 pr-4">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: spanColor(span.name) }} />
                  <span className="font-mono text-text-primary">{span.name}</span>
                </span>
              </td>
              <td className="py-1.5 pr-4 font-mono text-text-secondary">+{fmtDuration(offsetMs)}</td>
              <td className="py-1.5 pr-4 font-mono text-text-primary">{fmtDuration(span.durMs)}</td>
              <td className="max-w-xs truncate py-1.5 pr-4 font-mono text-[11px] text-text-secondary">{keyAttrs}</td>
              <td className="w-40 py-1.5">
                <div className="relative h-3 rounded-sm bg-background">
                  <div
                    className="absolute top-0 h-full rounded-sm"
                    style={{
                      left: `${leftPct}%`,
                      width: `${widthPct}%`,
                      backgroundColor: spanColor(span.name),
                      opacity: 0.8,
                    }}
                  />
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function BlocksTab({ row, blockSpans }: { row: TraceRow; blockSpans: Span[] }) {
  if (blockSpans.length === 0) {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="text-center">
          <p className="text-sm text-text-secondary">No block data in this trace</p>
          <p className="mt-1 text-xs text-muted">
            Block counts appear on spans like <span className="font-mono">stage_local_matches</span>,{" "}
            <span className="font-mono">r2h</span>, <span className="font-mono">h2d</span>, etc.
          </p>
        </div>
      </div>
    );
  }

  const sorted = [...blockSpans].sort((a, b) => (a.startNs < b.startNs ? -1 : 1));

  return (
    <div className="flex flex-col gap-4">
      {/* block flow summary */}
      <div>
        <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted">Block Flow</p>
        <div className="flex flex-wrap gap-3">
          {summarizeBlockFlow(sorted).map((item, i) => (
            <div key={i} className="rounded-lg border border-border/60 bg-background px-3 py-2">
              <p className="text-[10px] uppercase tracking-wider text-muted">{item.label}</p>
              <p className="text-lg font-semibold text-text-primary">{item.value}</p>
              {item.detail && <p className="text-[11px] text-text-secondary">{item.detail}</p>}
            </div>
          ))}
        </div>
      </div>

      {/* per-span block details */}
      <div>
        <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted">Per-Span Detail</p>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border/50 text-left text-[11px] uppercase tracking-wider text-muted">
              <th className="pb-2 pr-4 font-medium">Span</th>
              <th className="pb-2 pr-4 font-medium">Duration</th>
              <th className="pb-2 font-medium">Block Info</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((span, i) => (
              <tr key={i} className="border-b border-border/30 hover:bg-surface-alt/50">
                <td className="py-1.5 pr-4">
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block h-2 w-2 rounded-sm" style={{ backgroundColor: spanColor(span.name) }} />
                    <span className="font-mono text-text-primary">{span.name}</span>
                  </span>
                </td>
                <td className="py-1.5 pr-4 font-mono text-text-primary">{fmtDuration(span.durMs)}</td>
                <td className="py-1.5">
                  <div className="flex flex-wrap gap-x-4 gap-y-0.5">
                    {formatBlockAttrs(span).map(([k, v]) => (
                      <span key={k} className="text-text-secondary">
                        {k}: <span className="font-mono text-text-primary">{v}</span>
                      </span>
                    ))}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AttributesTab({ row, resp }: { row: TraceRow; resp: ResponseEntry | undefined }) {
  const traceAttrs = Object.entries(row.attributes).filter(([, v]) => v !== "");
  const respAttrs = resp
    ? [
        ["scenario", resp.scenario],
        ["isl", String(resp.isl)],
        ["request", String(resp.request)],
        ["prompt_key", resp.prompt_key],
        ["response_id", resp.response_id],
        ["finish_reason", resp.finish_reason],
        ["stop_reason", resp.stop_reason ?? "null"],
        ["tool_calls", String(resp.tool_calls)],
        ["refusal", resp.refusal ?? "null"],
        ["has_reasoning", String(resp.has_reasoning)],
        ["prompt_tokens", String(resp.prompt_tokens)],
        ["completion_tokens", String(resp.completion_tokens)],
        ["seed", String(resp.seed)],
      ] as [string, string][]
    : [];

  return (
    <div className="flex gap-8">
      <div className="min-w-0 flex-1">
        <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted">Trace Attributes</p>
        <AttrTable entries={traceAttrs} />
      </div>
      {respAttrs.length > 0 && (
        <div className="min-w-0 flex-1">
          <p className="mb-2 text-[11px] font-medium uppercase tracking-wider text-muted">Response Metadata</p>
          <AttrTable entries={respAttrs} />
        </div>
      )}
    </div>
  );
}

// --- helpers ---

const BLOCK_ATTR_KEYS = [
  "num_blocks", "host_matched", "disk_matched", "g4_matched",
  "num_external_blocks", "backend", "src_pool",
  "block_ids", "device_block_ids",
];

const SKIP_ATTRS = new Set(["request_id", "otel.name", "num_computed_tokens", "num_external_tokens", "current_position"]);

function formatKeyAttrs(span: Span): string {
  const parts: string[] = [];
  for (const [k, v] of Object.entries(span.attributes)) {
    if (!SKIP_ATTRS.has(k) && v) parts.push(`${k}=${v}`);
  }
  return parts.join(", ");
}

function formatBlockAttrs(span: Span): [string, string][] {
  const out: [string, string][] = [];
  for (const k of BLOCK_ATTR_KEYS) {
    const v = span.attributes[k];
    if (v) out.push([k, v]);
  }
  return out;
}

function summarizeBlockFlow(spans: Span[]): { label: string; value: string; detail?: string }[] {
  const items: { label: string; value: string; detail?: string }[] = [];

  for (const s of spans) {
    const a = s.attributes;
    if (s.name === "kvbm.stage_local_matches" || s.name.endsWith("stage_local_matches")) {
      const host = a["host_matched"] || "0";
      const disk = a["disk_matched"] || "0";
      const g4 = a["g4_matched"] || "0";
      const total = Number(host) + Number(disk) + Number(g4);
      items.push({
        label: "Matched Blocks",
        value: String(total),
        detail: `host=${host} disk=${disk} g4=${g4}`,
      });
    }
    if (s.name === "kvbm.r2h" || s.name.endsWith("r2h")) {
      items.push({
        label: "R2H Transfer",
        value: `${a["num_blocks"] || "?"} blocks`,
        detail: a["backend"] ? `backend: ${a["backend"]}` : undefined,
      });
    }
    if (s.name === "kvbm.h2d" || s.name.endsWith("h2d")) {
      items.push({
        label: "H2D Transfer",
        value: `${a["num_blocks"] || "?"} blocks`,
      });
    }
    if (s.name === "kvbm.trigger_onboarding" || s.name.endsWith("trigger_onboarding")) {
      items.push({
        label: "Onboarding",
        value: `${a["num_external_blocks"] || "?"} blocks`,
      });
    }
    if (s.name === "kvbm.onboard_from_g4" || s.name.endsWith("onboard_from_g4")) {
      items.push({
        label: "G4 Onboard",
        value: `${a["num_blocks"] || "?"} blocks`,
      });
    }
    if (s.name === "kvbm.update_state_after_alloc" || s.name.endsWith("update_state_after_alloc")) {
      if (a["num_blocks"]) {
        items.push({
          label: "Device Alloc",
          value: `${a["num_blocks"]} blocks`,
          detail: a["device_block_ids"] ? `ids: ${a["device_block_ids"]}` : undefined,
        });
      }
    }
    if (s.name === "kvbm.request_finished" || s.name.endsWith("request_finished")) {
      if (a["num_blocks"]) {
        items.push({
          label: "Freed",
          value: `${a["num_blocks"]} blocks`,
          detail: a["block_ids"] ? `ids: ${a["block_ids"]}` : undefined,
        });
      }
    }
  }

  return items;
}

function Badge({ label, value }: { label: string; value: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded border border-border/60 bg-background px-2 py-0.5 text-xs">
      <span className="text-muted">{label}:</span>
      <span className="font-mono text-text-primary">{value}</span>
    </span>
  );
}

function AttrTable({ entries }: { entries: [string, string][] }) {
  if (entries.length === 0) {
    return <p className="text-xs text-muted">No attributes</p>;
  }
  return (
    <div className="space-y-0.5">
      {entries.map(([k, v]) => (
        <div key={k} className="flex gap-2 text-xs">
          <span className="w-44 flex-shrink-0 truncate font-mono text-text-secondary">{k}</span>
          <span className="min-w-0 truncate font-mono text-text-primary">{v}</span>
        </div>
      ))}
    </div>
  );
}
