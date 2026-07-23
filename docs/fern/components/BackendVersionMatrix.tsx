/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * BackendVersionMatrix — Dynamo release × backend-pin table for the
 * Compatibility reference page. mode="current" shows main (ToT) and the
 * current stable release; mode="all" shows the full release history with a
 * diff-strip highlight on every pin that changed versus the chronologically
 * previous non-model-build release (RELEASES is newest-first, so "previous"
 * is the next comparable entry in the array; model-build side branches are
 * skipped both as rows-to-compare-against and never break the main lineage).
 *
 * Server component (no "use client"); shares .dynref-* base classes from
 * ReferenceStyles.tsx and carries only its own .dynref-vm-* layout rules.
 */


import {
  RELEASES,
  MAIN_TOT,
  CURRENT_VERSION,
  type BackendPins,
  type Release,
  type ReleaseKind,
} from "./releases.data";

const VM_CSS = `
.dynref-vm-scroll {
    overflow-x: auto;
    margin: 16px 0 8px;
}

.dynref-vm-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.dynref-vm-table th {
    padding: 6px 8px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--pst-color-text-muted);
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    white-space: nowrap;
}

.dynref-vm-table td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    color: var(--pst-color-text-base);
    vertical-align: top;
}

.dynref-vm-version { white-space: nowrap; }

.dynref-vm-date {
    display: block;
    margin-top: 2px;
    font-size: 12px;
    color: var(--pst-color-text-muted);
}

.dynref-vm-partial {
    display: block;
    margin-top: 2px;
    font-size: 12px;
    font-style: italic;
    color: var(--pst-color-text-muted);
}

.dynref-vm-changed {
    display: inline-block;
    background: rgba(118, 185, 0, 0.1);
    padding-inline: 4px;
    border-left: 2px solid #76B900;
    border-radius: 0 4px 4px 0;
}

.dynref-vm-nixl { display: inline-block; }

/* One line per backend, always — the label column keeps the three rows
   aligned into a compact sub-table. */
.dynref-vm-nixl-entry {
    display: block;
    white-space: nowrap;
    line-height: 1.5;
}

.dynref-vm-nixl-label {
    margin-left: 5px;
    font-size: 12px;
    color: var(--pst-color-text-muted);
}

.dynref-vm-kind { white-space: nowrap; }

.dynref-vm-dash { color: var(--pst-color-text-muted); }
`;

type PinKey = "sglang" | "trtllm" | "vllm";
type NixlKey = "nixlSglang" | "nixlTrtllm" | "nixlVllm";

const PIN_COLUMNS: { key: PinKey; label: string }[] = [
  { key: "sglang", label: "SGLang" },
  { key: "trtllm", label: "TensorRT-LLM" },
  { key: "vllm", label: "vLLM" },
];

/* Sub-entry labels are abbreviated (SGL / TRT / vLLM) so the stacked NIXL
   sub-rows stay compact. */
const NIXL_COLUMNS: { key: NixlKey; label: string }[] = [
  { key: "nixlSglang", label: "SGL" },
  { key: "nixlTrtllm", label: "TRT" },
  { key: "nixlVllm", label: "vLLM" },
];

const KIND_BADGE: Record<ReleaseKind, { variant: "green" | "gray" | "amber"; label: string }> = {
  stable: { variant: "green", label: "GA release" },
  patch: { variant: "gray", label: "Patch" },
  "platform-preview": { variant: "amber", label: "Early access" },
  "model-build": { variant: "amber", label: "Model build" },
};

/** Chronologically previous non-model-build release's pins (RELEASES is newest-first). */
function previousComparablePins(index: number): BackendPins | undefined {
  for (let j = index + 1; j < RELEASES.length; j++) {
    if (RELEASES[j].kind !== "model-build") return RELEASES[j].pins;
  }
  return undefined;
}

function Pin({ value, changed }: { value?: string; changed?: boolean }) {
  if (!value) return <span className="dynref-vm-dash">&mdash;</span>;
  const pin = <span className="dynref-mono">{value}</span>;
  return changed ? <span className="dynref-vm-changed">{pin}</span> : pin;
}

function NixlCell({ pins, prev }: { pins?: BackendPins; prev?: BackendPins }) {
  const entries = NIXL_COLUMNS.filter(({ key }) => pins?.[key]);
  if (!pins || entries.length === 0) return <span className="dynref-vm-dash">&mdash;</span>;

  const changedFor = (key: NixlKey) => (prev ? prev[key] !== pins[key] : false);

  return (
    <span className="dynref-vm-nixl">
      {entries.map(({ key, label }) => (
        <span className="dynref-vm-nixl-entry" key={key}>
          <Pin value={pins[key]} changed={changedFor(key)} />
          <span className="dynref-vm-nixl-label">{label}</span>
        </span>
      ))}
    </span>
  );
}

function PinCells({ pins, prev }: { pins?: BackendPins; prev?: BackendPins }) {
  return (
    <>
      {PIN_COLUMNS.map(({ key }) => (
        <td key={key}>
          <Pin value={pins?.[key]} changed={prev && pins?.[key] ? prev[key] !== pins[key] : false} />
        </td>
      ))}
      <td>
        <NixlCell pins={pins} prev={prev} />
      </td>
    </>
  );
}

function ReleaseRow({ release, prev }: { release: Release; prev?: BackendPins }) {
  const badge = KIND_BADGE[release.kind];
  return (
    <tr>
      <td className="dynref-vm-version">
        <span className="dynref-mono">{release.version}</span>
        {release.date && <span className="dynref-vm-date">{release.date}</span>}
        {release.partial && <span className="dynref-vm-partial">partial coverage</span>}
      </td>
      <td className="dynref-vm-kind">
        <span className={`dynref-badge dynref-badge--${badge.variant}`}>{badge.label}</span>
      </td>
      <PinCells pins={release.pins} prev={prev} />
    </tr>
  );
}

export function BackendVersionMatrix({ mode = "current" }: { mode?: "current" | "all" }) {
  const current = RELEASES.find((release) => release.version === CURRENT_VERSION);

  return (
    <div>
      <style>{VM_CSS}</style>
      <div className="dynref-vm-scroll">
        <table className="dynref-vm-table">
          <thead>
            <tr>
              <th>Dynamo</th>
              <th>Type</th>
              {PIN_COLUMNS.map(({ key, label }) => (
                <th key={key}>{label}</th>
              ))}
              <th>NIXL</th>
            </tr>
          </thead>
          <tbody>
            {mode === "current" ? (
              <>
                <tr>
                  <td className="dynref-vm-version">main (ToT)</td>
                  <td className="dynref-vm-kind">
                    <span className="dynref-badge dynref-badge--gray">development</span>
                  </td>
                  <PinCells pins={MAIN_TOT} />
                </tr>
                {current && (
                  <tr>
                    <td className="dynref-vm-version">
                      <span className="dynref-mono">{current.version}</span>
                      {current.date && <span className="dynref-vm-date">{current.date}</span>}
                    </td>
                    <td className="dynref-vm-kind">
                      <span className="dynref-badge dynref-badge--green">GA release</span>
                    </td>
                    <PinCells pins={current.pins} />
                  </tr>
                )}
              </>
            ) : (
              RELEASES.map((release, index) => (
                <ReleaseRow
                  key={release.version}
                  release={release}
                  prev={previousComparablePins(index)}
                />
              ))
            )}
          </tbody>
        </table>
      </div>
      {mode === "all" && (
        <>
          <p className="dynref-grid-note">
            <span className="dynref-vm-changed">
              <span className="dynref-mono">Highlighted</span>
            </span>{" "}
            pins changed relative to the previous release; unmarked pins are unchanged (patch
            releases typically re-ship their base release&rsquo;s pins).
          </p>
          <p className="dynref-grid-note">
            Early access rows show branch build pins from container/context.yaml; not every backend
            ships a published container for those tags.
          </p>
          <p className="dynref-grid-note">
            Backend versions listed are the only versions tested and supported for each release.
            TensorRT-LLM does not support Python 3.11.
          </p>
        </>
      )}
    </div>
  );
}
