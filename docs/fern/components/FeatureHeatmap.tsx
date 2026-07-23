/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FeatureHeatmap — at-a-glance feature-by-backend support grid for the
 * Compatibility reference page. Renders entirely from FEATURES in
 * releases.data.ts; per-backend coverage scores are computed, never
 * hardcoded.
 *
 * Status-cell scheme: "yes" and "caveat" cells use the shared tinted-chip
 * treatment (translucent fill + 1px border, matching .dynref-badge--green /
 * --amber in ReferenceStyles.tsx) so green stays an accent, never a solid
 * wallpaper. Experimental keeps the dashed amber outline; not-supported stays
 * dim neutral. Cells with a note carry a superscript footnote marker
 * (numbered in row-major grid order) resolved in an ordered list below the
 * grid; the title attribute is kept as a hover bonus.
 *
 * Server component (no "use client"); shares .dynref-* base classes from
 * ReferenceStyles.tsx and carries only its own .dynref-heat-* layout rules.
 */

import { Fragment } from "react";

import { FEATURES, type FeatureCell } from "./releases.data";

const HEAT_CSS = `
.dynref-heat-legend {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    font-size: 12px;
    color: var(--pst-color-text-muted);
}

.dynref-heat-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

.dynref-heat-swatch {
    display: inline-block;
    box-sizing: border-box;
    width: 10px;
    height: 10px;
    border-radius: 3px;
}

.dynref-heat-swatch--yes {
    background: rgba(118, 185, 0, 0.16);
    border: 1px solid rgba(118, 185, 0, 0.38);
}
.dark .dynref-heat-swatch--yes {
    background: rgba(118, 185, 0, 0.22);
    border-color: rgba(118, 185, 0, 0.46);
}

.dynref-heat-swatch--caveat {
    background: rgba(239, 159, 39, 0.14);
    border: 1px solid rgba(239, 159, 39, 0.35);
}
.dark .dynref-heat-swatch--caveat {
    background: rgba(239, 159, 39, 0.16);
    border-color: rgba(239, 159, 39, 0.4);
}

.dynref-heat-swatch--wip {
    background: transparent;
    border: 1.5px dashed #b97a17;
}

.dynref-heat-swatch--no { background: #ececec; }
.dark .dynref-heat-swatch--no { background: #242424; }

.dynref-heat-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.6fr) repeat(3, minmax(64px, 1fr));
    gap: 6px;
    font-size: 13px;
}

.dynref-heat-colhead {
    align-self: end;
    text-align: center;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--pst-color-text-base);
}

.dynref-heat-score {
    display: block;
    margin-top: 2px;
    font-size: 11.5px;
    font-weight: 400;
    color: #5a8c00;
}
.dark .dynref-heat-score { color: #76B900; }

.dynref-heat-feature {
    align-self: center;
    min-width: 0;
    color: var(--pst-color-text-base);
}

.dynref-heat-cell {
    box-sizing: border-box;
    height: 26px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.dynref-heat-cell--titled { cursor: help; }

.dynref-heat-cell--yes {
    background: rgba(118, 185, 0, 0.16);
    border: 1px solid rgba(118, 185, 0, 0.38);
    color: #4D7C0F;
}
.dark .dynref-heat-cell--yes {
    background: rgba(118, 185, 0, 0.22);
    border-color: rgba(118, 185, 0, 0.46);
    color: #A3E635;
}

.dynref-heat-cell--caveat {
    background: rgba(239, 159, 39, 0.14);
    border: 1px solid rgba(239, 159, 39, 0.35);
    color: #854F0B;
}
.dark .dynref-heat-cell--caveat {
    background: rgba(239, 159, 39, 0.16);
    border-color: rgba(239, 159, 39, 0.4);
    color: #FAC775;
}

.dynref-heat-cell--wip {
    background: transparent;
    border: 1.5px dashed #b97a17;
    color: #B97A17;
}
.dark .dynref-heat-cell--wip { color: #EF9F27; }

.dynref-heat-cell--no { background: #ececec; }
.dark .dynref-heat-cell--no { background: #242424; }

.dynref-heat-dash { color: var(--pst-color-text-muted); }

.dynref-heat-fn {
    margin-left: 2px;
    font-size: 11px;
    font-weight: 600;
    line-height: 1;
}

.dynref-heat-footnotes {
    margin: 12px 0 0;
    padding-left: 20px;
    font-size: 12px;
    color: var(--pst-color-text-muted);
}

.dynref-heat-footnotes li { margin: 2px 0; }
`;

const BACKENDS = [
  { key: "sglang", label: "SGLang" },
  { key: "trtllm", label: "TRT-LLM" },
  { key: "vllm", label: "vLLM" },
] as const;

type BackendKey = (typeof BACKENDS)[number]["key"];

const STATUS_LABEL: Record<FeatureCell["status"], string> = {
  yes: "Supported",
  caveat: "Caveat",
  wip: "Experimental",
  no: "Not supported",
};

interface Footnote {
  feature: string;
  backend: string;
  note: string;
}

/** Every noted cell, numbered in row-major grid order (derived, never hardcoded). */
const FOOTNOTES: Footnote[] = FEATURES.flatMap((feature) =>
  BACKENDS.flatMap((backend) => {
    const note = feature[backend.key].note;
    return note ? [{ feature: feature.name, backend: backend.label, note }] : [];
  }),
);

/** 1-based footnote number for a noted cell; undefined when the cell has no note. */
function footnoteIndex(feature: string, backend: string): number | undefined {
  const i = FOOTNOTES.findIndex((fn) => fn.feature === feature && fn.backend === backend);
  return i === -1 ? undefined : i + 1;
}

function coverageScore(key: BackendKey): string {
  const supported = FEATURES.filter((feature) => {
    const status = feature[key].status;
    return status === "yes" || status === "caveat";
  }).length;
  return `${supported} / ${FEATURES.length}`;
}

function CheckGlyph() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M5 12l5 5L20 7" />
    </svg>
  );
}

function AlertGlyph() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M12 4 2.8 19.5h18.4L12 4z" />
      <path d="M12 10v4" />
      <path d="M12 17h.01" />
    </svg>
  );
}

function FlaskGlyph() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M10 3v6l-5.2 9a2 2 0 0 0 1.7 3h11a2 2 0 0 0 1.7-3L14 9V3" />
      <path d="M8.5 3h7" />
    </svg>
  );
}

function StatusCellContent({ status }: { status: FeatureCell["status"] }) {
  if (status === "yes") return <CheckGlyph />;
  if (status === "caveat") return <AlertGlyph />;
  if (status === "wip") return <FlaskGlyph />;
  return <span className="dynref-heat-dash">&mdash;</span>;
}

function StatusCell({ cell, feature, backend }: { cell: FeatureCell; feature: string; backend: string }) {
  const classes = [
    "dynref-heat-cell",
    `dynref-heat-cell--${cell.status}`,
    cell.note ? "dynref-heat-cell--titled" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const label = `${feature} on ${backend}: ${STATUS_LABEL[cell.status]}${cell.note ? ` — ${cell.note}` : ""}`;
  const index = cell.note ? footnoteIndex(feature, backend) : undefined;
  return (
    <div className={classes} title={cell.note} role="img" aria-label={label}>
      <StatusCellContent status={cell.status} />
      {index !== undefined && <sup className="dynref-heat-fn">{index}</sup>}
    </div>
  );
}

export function FeatureHeatmap() {
  return (
    <div className="dynref-panel">
      <style>{HEAT_CSS}</style>
      <div className="dynref-panel-header">
        <span className="dynref-h">Feature support by backend</span>
        <div className="dynref-heat-legend">
          <span className="dynref-heat-legend-item">
            <span className="dynref-heat-swatch dynref-heat-swatch--yes" />
            Supported
          </span>
          <span className="dynref-heat-legend-item">
            <span className="dynref-heat-swatch dynref-heat-swatch--caveat" />
            Caveat
          </span>
          <span className="dynref-heat-legend-item">
            <span className="dynref-heat-swatch dynref-heat-swatch--wip" />
            Experimental
          </span>
          <span className="dynref-heat-legend-item">
            <span className="dynref-heat-swatch dynref-heat-swatch--no" />
            Not supported
          </span>
        </div>
      </div>
      <div className="dynref-heat-grid">
        <div />
        {BACKENDS.map((backend) => (
          <div key={backend.key} className="dynref-heat-colhead">
            {backend.label}
            <span className="dynref-heat-score dynref-mono">{coverageScore(backend.key)}</span>
          </div>
        ))}
        {FEATURES.map((feature) => (
          <Fragment key={feature.name}>
            <div className="dynref-heat-feature">{feature.name}</div>
            {BACKENDS.map((backend) => (
              <StatusCell
                key={backend.key}
                cell={feature[backend.key]}
                feature={feature.name}
                backend={backend.label}
              />
            ))}
          </Fragment>
        ))}
      </div>
      {FOOTNOTES.length > 0 && (
        <ol className="dynref-heat-footnotes">
          {FOOTNOTES.map((fn) => (
            <li key={`${fn.feature}-${fn.backend}`}>
              {fn.feature} &middot; {fn.backend}: {fn.note}
            </li>
          ))}
        </ol>
      )}
      <p className="dynref-grid-note">Superscripts reference the numbered notes above; full per-backend detail follows.</p>
    </div>
  );
}
