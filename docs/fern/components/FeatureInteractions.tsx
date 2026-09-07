/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FeatureInteractions — pairwise feature-by-feature compatibility matrix for
 * one backend, rendered in the same visual vocabulary as FeatureHeatmap:
 * tinted status cells, a glyph per state, and an "i" marker whose note opens
 * on hover or keyboard focus.
 *
 * Cells come from FEATURE_INTERACTIONS in releases.data.ts, which stores only
 * the lower triangle (rows[i] carries i+1 cells, ending on the diagonal). The
 * upper triangle is that triangle mirrored, so it is rendered blank rather
 * than authored twice. gen_llms_tables.py emits the same cells as markdown
 * into the page's <llms-only> twin, so no pairwise status can be visible here
 * yet missing from an agent export.
 *
 * CSS is injected via dangerouslySetInnerHTML, not as a <style> text child: a
 * text child is escaped on render, which turns any `>` child combinator into
 * &gt; and silently drops the rule (see #12402).
 *
 * Server component (no "use client"); disclosure is CSS-only, so no client JS.
 */

import { FEATURE_INTERACTIONS, type InteractionCell } from "./releases.data";

const FI_CSS = `
.dynref-fi-legend {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px 14px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
}

.dynref-fi-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 5px;
}

.dynref-fi-swatch {
    display: inline-block;
    width: 13px;
    height: 13px;
    border-radius: 4px;
}

.dynref-fi-swatch--yes {
    background: var(--dynref-green-bg);
    border: 1px solid var(--dynref-green-border);
}

.dynref-fi-swatch--wip {
    background: transparent;
    border: 1.5px dashed #b97a17;
}

.dynref-fi-swatch--no {
    background: #ececec;
    border: 1px solid #d8d8d8;
}

.dark .dynref-fi-swatch--no {
    background: #242424;
    border-color: #333;
}

.dynref-fi-swatch--na { background: rgba(120, 120, 120, 0.16); }

.dynref-fi-scroll {
    overflow-x: auto;
    padding-bottom: 4px;
}

.dynref-fi-table {
    width: max-content;
    min-width: 100%;
    margin: 0;
    border-collapse: separate;
    border-spacing: 6px;
    font-size: 13px;
}

.dynref-fi-table th,
.dynref-fi-table td {
    padding: 0;
    border: 0;
    background: none;
}

.dynref-fi-colhead {
    width: 74px;
    min-width: 74px;
    vertical-align: bottom;
    color: var(--pst-color-text-base);
    font-size: 11.5px;
    font-weight: 600;
    line-height: 1.25;
    text-align: center;
}

.dynref-fi-rowhead {
    position: sticky;
    left: 0;
    z-index: 3;
    min-width: 152px;
    padding-right: 8px;
    background: var(--pst-color-surface);
    box-shadow: 4px 0 0 var(--pst-color-surface);
    color: var(--pst-color-text-base);
    font-size: 12.5px;
    font-weight: 500;
    text-align: left;
    vertical-align: middle;
}

.dark .dynref-fi-rowhead {
    background: #161616;
    box-shadow: 4px 0 0 #161616;
}

.dynref-fi-cell {
    position: relative;
    box-sizing: border-box;
    height: 26px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.dynref-fi-cell--titled { cursor: help; }

.dynref-fi-cell--titled:focus-visible {
    outline: 2px solid var(--dynref-blue-fg);
    outline-offset: 2px;
}

.dynref-fi-cell--yes {
    background: var(--dynref-green-bg);
    border: 1px solid var(--dynref-green-border);
    color: var(--dynref-green-fg);
}

.dynref-fi-cell--wip {
    background: transparent;
    border: 1.5px dashed #b97a17;
    color: #B97A17;
}
.dark .dynref-fi-cell--wip { color: #EF9F27; }

.dynref-fi-cell--no {
    background: #ececec;
    border: 1px solid #d8d8d8;
    color: #8a8a8a;
}
.dark .dynref-fi-cell--no {
    background: #242424;
    border-color: #333;
    color: #8a8a8a;
}

.dynref-fi-cell--na {
    background: rgba(120, 120, 120, 0.08);
    color: var(--pst-color-text-muted);
}

.dynref-fi-note-mark {
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

.dynref-fi-tooltip {
    position: absolute;
    z-index: 10;
    bottom: calc(100% + 8px);
    left: 50%;
    width: min(260px, calc(100vw - 48px));
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

.dark .dynref-fi-tooltip {
    background: #202020;
    border-color: #3a3a3a;
}

/* Bridges the 8px gap so the pointer can travel from cell to tooltip without
   the tooltip closing -- the Source link inside it has to be clickable. */
.dynref-fi-tooltip::after {
    position: absolute;
    top: 100%;
    left: 0;
    width: 100%;
    height: 8px;
    content: "";
}

/* The matrix scrolls horizontally, so a centred tooltip on an edge column is
   clipped by the scroll box. Anchor the first columns left and the last
   columns right instead. */
.dynref-fi-cell[data-edge="start"] .dynref-fi-tooltip {
    left: 0;
    transform: translate(0, 3px);
}

.dynref-fi-cell[data-edge="end"] .dynref-fi-tooltip {
    right: 0;
    left: auto;
    transform: translate(0, 3px);
}

.dynref-fi-cell--titled:hover,
.dynref-fi-cell--titled:focus-visible {
    z-index: 9;
}

.dynref-fi-cell--titled:hover .dynref-fi-tooltip,
.dynref-fi-cell--titled:focus-visible .dynref-fi-tooltip {
    opacity: 1;
    visibility: visible;
    transform: translate(-50%, 0);
    pointer-events: auto;
}

.dynref-fi-cell--titled[data-edge="start"]:hover .dynref-fi-tooltip,
.dynref-fi-cell--titled[data-edge="start"]:focus-visible .dynref-fi-tooltip,
.dynref-fi-cell--titled[data-edge="end"]:hover .dynref-fi-tooltip,
.dynref-fi-cell--titled[data-edge="end"]:focus-visible .dynref-fi-tooltip {
    transform: translate(0, 0);
}

.dynref-fi-tooltip a {
    color: inherit;
    text-decoration: underline;
}

@media (prefers-reduced-motion: reduce) {
    .dynref-fi-tooltip { transition: none; }
}
`;

const STATUS_LABEL: Record<InteractionCell["status"], string> = {
  yes: "Supported",
  wip: "Work in progress, experimental, or limited",
  no: "Not supported",
  na: "Not applicable",
};

// Columns whose tooltip is anchored to a cell edge rather than centred.
const EDGE_START = 2;
const EDGE_END = 2;

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

function CrossGlyph() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="13"
      height="13"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M6 6l12 12M18 6L6 18" />
    </svg>
  );
}

function StatusGlyph({ status }: { status: InteractionCell["status"] }) {
  if (status === "yes") return <CheckGlyph />;
  if (status === "wip") return <FlaskGlyph />;
  if (status === "no") return <CrossGlyph />;
  return <span aria-hidden="true">&mdash;</span>;
}

function edgeOf(column: number, total: number): "start" | "end" | undefined {
  if (column < EDGE_START) return "start";
  if (column >= total - EDGE_END) return "end";
  return undefined;
}

export function FeatureInteractions({ backend }: { backend: string }) {
  const matrix = FEATURE_INTERACTIONS.find((entry) => entry.backend === backend);
  if (!matrix) {
    throw new Error(
      `FeatureInteractions: no FEATURE_INTERACTIONS entry for backend "${backend}" ` +
        `(have ${FEATURE_INTERACTIONS.map((e) => e.backend).join(", ")})`,
    );
  }
  const features = matrix.features;
  return (
    <div className="dynref-panel">
      <style dangerouslySetInnerHTML={{ __html: FI_CSS }} />
      <div className="dynref-panel-header">
        <span className="dynref-h">{matrix.backend} Feature Interactions</span>
        <div className="dynref-fi-legend">
          <span className="dynref-fi-legend-item">
            <span className="dynref-fi-swatch dynref-fi-swatch--yes" />
            Supported
          </span>
          <span className="dynref-fi-legend-item">
            <span className="dynref-fi-swatch dynref-fi-swatch--wip" />
            Experimental
          </span>
          <span className="dynref-fi-legend-item">
            <span className="dynref-fi-swatch dynref-fi-swatch--no" />
            Not supported
          </span>
          <span className="dynref-fi-legend-item">
            <span className="dynref-fi-swatch dynref-fi-swatch--na" />
            Not applicable
          </span>
        </div>
      </div>
      <div className="dynref-fi-scroll">
        <table className="dynref-fi-table">
          <thead>
            <tr>
              <td />
              {features.map((feature) => (
                <th className="dynref-fi-colhead" key={feature} scope="col">
                  {feature}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.rows.map((cells, rowIndex) => (
              <tr key={features[rowIndex]}>
                <th className="dynref-fi-rowhead" scope="row">
                  {features[rowIndex]}
                </th>
                {features.map((column, columnIndex) => {
                  const cell = cells[columnIndex];
                  return (
                    <td key={column}>
                      {cell && (
                        <div
                          className={[
                            "dynref-fi-cell",
                            `dynref-fi-cell--${cell.status}`,
                            cell.note ? "dynref-fi-cell--titled" : "",
                          ]
                            .filter(Boolean)
                            .join(" ")}
                          data-edge={edgeOf(columnIndex, features.length)}
                          role="img"
                          aria-label={`${features[rowIndex]} with ${column}: ${
                            cell.label
                              ? `${STATUS_LABEL[cell.status]} — ${cell.label}`
                              : STATUS_LABEL[cell.status]
                          }${cell.note ? `. ${cell.note}` : ""}`}
                          tabIndex={cell.note ? 0 : undefined}
                        >
                          <StatusGlyph status={cell.status} />
                          {cell.note && (
                            <>
                              <span className="dynref-fi-note-mark" aria-hidden="true">
                                i
                              </span>
                              <span className="dynref-fi-tooltip" aria-hidden="true">
                                {cell.note}
                                {cell.source && (
                                  <>
                                    {" "}
                                    <a href={cell.source}>Source</a>
                                  </>
                                )}
                              </span>
                            </>
                          )}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="dynref-grid-note">
        Each cell reports whether the row feature works together with the column feature. Blank cells
        mirror the populated lower triangle. Hover a noted cell or focus it with the keyboard to view
        its note.
      </p>
    </div>
  );
}
