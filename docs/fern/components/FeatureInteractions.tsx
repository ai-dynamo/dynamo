/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FeatureInteractions renders only non-obvious feature combinations. Base
 * backend support belongs to FeatureHeatmap; this component owns requirements
 * and limitations that appear only when two otherwise available features are
 * used together.
 *
 * Server component (no "use client").
 */

import { FEATURE_INTERACTIONS } from "./releases.data";

const FI_CSS = `
.dynref-fi-table {
    width: 100%;
    margin: 0;
    border-collapse: collapse;
    font-size: 13px;
}

.dynref-fi-table th,
.dynref-fi-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    text-align: left;
    vertical-align: top;
}

.dynref-fi-table th {
    color: var(--pst-color-text-base);
    font-weight: 600;
}

.dynref-fi-combination {
    min-width: 190px;
    font-weight: 500;
}

.dynref-fi-status {
    display: inline-block;
    white-space: nowrap;
    padding: 2px 7px;
    border-radius: 999px;
    font-size: 11.5px;
    font-weight: 600;
}

.dynref-fi-status--yes {
    background: var(--dynref-green-bg);
    border: 1px solid var(--dynref-green-border);
    color: var(--dynref-green-fg);
}

.dynref-fi-status--wip {
    background: var(--dynref-amber-bg);
    border: 1px solid var(--dynref-amber-border);
    color: var(--dynref-amber-fg);
}

.dynref-fi-status--no {
    background: #ececec;
    border: 1px solid #d8d8d8;
    color: #666;
}

.dark .dynref-fi-status--no {
    background: #242424;
    border-color: #333;
    color: #aaa;
}
`;

export function FeatureInteractions({ backend }: { backend: string }) {
  const entry = FEATURE_INTERACTIONS.find((candidate) => candidate.backend === backend);
  if (!entry) {
    throw new Error(
      `FeatureInteractions: no FEATURE_INTERACTIONS entry for backend "${backend}" ` +
        `(have ${FEATURE_INTERACTIONS.map((candidate) => candidate.backend).join(", ")})`,
    );
  }

  return (
    <div className="dynref-panel">
      <style dangerouslySetInnerHTML={{ __html: FI_CSS }} />
      <div className="dynref-panel-header">
        <span className="dynref-h">{entry.backend} Constraints</span>
      </div>
      <table className="dynref-fi-table">
        <thead>
          <tr>
            <th scope="col">Feature combination</th>
            <th scope="col">Status</th>
            <th scope="col">Constraint</th>
          </tr>
        </thead>
        <tbody>
          {entry.constraints.map((constraint) => (
            <tr key={constraint.features.join("+")}>
              <th className="dynref-fi-combination" scope="row">
                {constraint.features.join(" + ")}
              </th>
              <td>
                <span className={`dynref-fi-status dynref-fi-status--${constraint.status}`}>
                  {constraint.label}
                </span>
              </td>
              <td>
                {constraint.note}
                {constraint.source && (
                  <>
                    {" "}
                    <a href={constraint.source}>Source</a>
                  </>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="dynref-grid-note">
        This table lists only combinations whose behavior is not clear from the base feature support
        table.
      </p>
    </div>
  );
}
