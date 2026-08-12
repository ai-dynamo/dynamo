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
 * dim neutral. Cells with a note carry an information marker and reveal the
 * note in a CSS-only tooltip on hover or keyboard focus.
 *
 * CSS is injected via dangerouslySetInnerHTML, not as a <style> text child:
 * a text child is escaped on render, so a `>` child combinator becomes &gt;
 * and the double quotes in the [data-backend="..."] selectors below become
 * &quot;, silently dropping those rules (see #12402).
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
    background: var(--dynref-green-bg);
    border: 1px solid var(--dynref-green-border);
}
.dark .dynref-heat-swatch--yes {
    background: var(--dynref-green-bg);
    border-color: var(--dynref-green-border);
}

.dynref-heat-swatch--caveat {
    background: var(--dynref-amber-bg);
    border: 1px solid var(--dynref-amber-border);
}
.dark .dynref-heat-swatch--caveat {
    background: var(--dynref-amber-bg);
    border-color: var(--dynref-amber-border);
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
    position: relative;
    box-sizing: border-box;
    height: 26px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.dynref-heat-cell--titled {
    cursor: help;
}

.dynref-heat-cell--titled:focus-visible {
    outline: 2px solid var(--dynref-blue-fg);
    outline-offset: 2px;
}

.dynref-heat-cell--yes {
    background: var(--dynref-green-bg);
    border: 1px solid var(--dynref-green-border);
    color: var(--dynref-green-fg);
}
.dark .dynref-heat-cell--yes {
    background: var(--dynref-green-bg);
    border-color: var(--dynref-green-border);
    color: var(--dynref-green-fg);
}

.dynref-heat-cell--caveat {
    background: var(--dynref-amber-bg);
    border: 1px solid var(--dynref-amber-border);
    color: var(--dynref-amber-fg);
}
.dark .dynref-heat-cell--caveat {
    background: var(--dynref-amber-bg);
    border-color: var(--dynref-amber-border);
    color: var(--dynref-amber-fg);
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

.dynref-heat-note-mark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 12px;
    height: 12px;
    margin-left: 3px;
    border: 1px solid currentColor;
    border-radius: 999px;
    font-size: 8px;
    font-weight: 700;
    line-height: 1;
}

.dynref-heat-tooltip {
    position: absolute;
    z-index: 10;
    bottom: calc(100% + 8px);
    left: 50%;
    width: min(280px, calc(100vw - 48px));
    padding: 8px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 7px;
    background: var(--pst-color-surface);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.16);
    color: var(--pst-color-text-base);
    font-size: 12px;
    font-weight: 400;
    line-height: 1.4;
    text-align: left;
    transform: translate(-50%, 3px);
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
}

.dark .dynref-heat-tooltip {
    background: #202020;
    border-color: #3a3a3a;
}

.dynref-heat-tooltip::after {
    position: absolute;
    top: 100%;
    left: 0;
    width: 100%;
    height: 8px;
    content: "";
}

.dynref-heat-cell[data-backend="SGLang"] .dynref-heat-tooltip {
    left: 0;
    transform: translate(0, 3px);
}

.dynref-heat-cell[data-backend="vLLM"] .dynref-heat-tooltip {
    right: 0;
    left: auto;
    transform: translate(0, 3px);
}

.dynref-heat-cell--titled:hover,
.dynref-heat-cell--titled:focus-visible {
    z-index: 9;
}

.dynref-heat-cell--titled:hover .dynref-heat-tooltip,
.dynref-heat-cell--titled:focus-visible .dynref-heat-tooltip {
    opacity: 1;
    visibility: visible;
    transform: translate(-50%, 0);
    pointer-events: auto;
}

.dynref-heat-cell--titled[data-backend="SGLang"]:hover .dynref-heat-tooltip,
.dynref-heat-cell--titled[data-backend="SGLang"]:focus-visible .dynref-heat-tooltip,
.dynref-heat-cell--titled[data-backend="vLLM"]:hover .dynref-heat-tooltip,
.dynref-heat-cell--titled[data-backend="vLLM"]:focus-visible .dynref-heat-tooltip {
    transform: translate(0, 0);
}

@media (prefers-reduced-motion: reduce) {
    .dynref-heat-tooltip { transition: none; }
}
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
  return (
    <div
      className={classes}
      data-backend={backend}
      role="img"
      aria-label={label}
      tabIndex={cell.note ? 0 : undefined}
    >
      <StatusCellContent status={cell.status} />
      {cell.note && (
        <>
          <span className="dynref-heat-note-mark" aria-hidden="true">
            i
          </span>
          <span className="dynref-heat-tooltip" aria-hidden="true">
            {cell.note}
          </span>
        </>
      )}
    </div>
  );
}

export function FeatureHeatmap() {
  return (
    <div className="dynref-panel">
      <style dangerouslySetInnerHTML={{ __html: HEAT_CSS }} />
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
      <p className="dynref-grid-note">Hover a noted cell or focus it with the keyboard to view its compatibility note.</p>
    </div>
  );
}
