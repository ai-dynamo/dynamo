/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CudaDriverMatrix — CUDA toolkit / minimum-driver support for the
 * Compatibility reference page. mode="current" is a compact per-backend
 * Backend | CUDA toolkit | Min driver table for the current release
 * (CURRENT_TAG). mode="all" renders the CUDA driver ladder (toolkit lanes
 * derived from the data, one row per version, newest first — the empty
 * CUDA 12 lanes above v1.3.0 visualize the CUDA 12 → 13 cutoff; each pill
 * carries the lane's minimum driver, e.g. ≥575) followed by the full history
 * table and CUDA_NOTES.
 *
 * Server component (no "use client"); shares .dynref-* base classes from
 * ReferenceStyles.tsx and carries only its own .dynref-cuda-* layout rules.
 */

import { Fragment } from "react";

import { CUDA_HISTORY, CUDA_NOTES, CURRENT_TAG, type CudaRow } from "./releases.data";

const CUDA_CSS = `
.dynref-cuda-ladder {
    display: grid;
    gap: 6px 8px;
    align-items: center;
    margin: 16px 0 6px;
}

.dynref-cuda-lane-head {
    text-align: center;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--pst-color-text-muted);
    padding-bottom: 2px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-cuda-corner {
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-cuda-vlabel {
    font-size: 12.5px;
    color: var(--pst-color-text-base);
    white-space: nowrap;
}

.dynref-cuda-lanecell {
    display: flex;
    justify-content: center;
}

.dynref-cuda-exp { border: 1px dashed currentColor; }

.dynref-cuda-caption {
    margin: 6px 0 20px;
    font-size: 12px;
    color: var(--pst-color-text-muted);
}

.dynref-cuda-scroll {
    overflow-x: auto;
    margin: 12px 0 8px;
}

.dynref-cuda-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}

.dynref-cuda-table th {
    padding: 8px 10px;
    text-align: left;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--pst-color-text-muted);
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    white-space: nowrap;
}

.dynref-cuda-table td {
    padding: 8px 10px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    color: var(--pst-color-text-base);
    vertical-align: top;
}

.dynref-cuda-vcell { white-space: nowrap; }

.dynref-cuda-note-cell {
    font-size: 12px;
    color: var(--pst-color-text-muted);
}
`;

/** Unique toolkits, ascending (lanes are derived from the data, never hardcoded). */
function toolkitLanes(): string[] {
  return [...new Set(CUDA_HISTORY.map((row) => row.toolkit))].sort(
    (a, b) => parseFloat(a) - parseFloat(b),
  );
}

/** Unique versions in data order (CUDA_HISTORY is newest-first). */
function uniqueVersions(): string[] {
  return [...new Set(CUDA_HISTORY.map((row) => row.version))];
}

function rowsFor(version: string): CudaRow[] {
  return CUDA_HISTORY.filter((row) => row.version === version);
}

/** "575.xx+" → "≥575": ladder-pill label derived from CudaRow.minDriver. */
function minDriverLabel(minDriver: string): string {
  return `≥${minDriver.replace(/\.xx\+$/, "").replace(/\+$/, "")}`;
}

function CudaChip({ label, experimental }: { label: string; experimental?: boolean }) {
  const classes = `dynref-chip dynref-chip--cuda${experimental ? " dynref-cuda-exp" : ""}`;
  return <span className={classes}>{label}</span>;
}

function CudaLadder() {
  const lanes = toolkitLanes();
  const versions = uniqueVersions();
  const gridColumns = { gridTemplateColumns: `60px repeat(${lanes.length}, minmax(64px, 1fr))` };

  return (
    <>
      <div className="dynref-cuda-ladder" style={gridColumns} role="presentation">
        <div className="dynref-cuda-corner" />
        {lanes.map((lane) => (
          <div key={lane} className="dynref-cuda-lane-head">
            CUDA {lane}
          </div>
        ))}
        {versions.map((version) => {
          const rows = rowsFor(version);
          return (
            <Fragment key={version}>
              <div className="dynref-cuda-vlabel dynref-mono">v{version}</div>
              {lanes.map((lane) => {
                const laneRows = rows.filter((row) => row.toolkit === lane);
                const experimental =
                  laneRows.length > 0 && laneRows.every((row) => row.note === "Experimental");
                const drivers = [...new Set(laneRows.map((row) => minDriverLabel(row.minDriver)))];
                return (
                  <div key={lane} className="dynref-cuda-lanecell">
                    {laneRows.length > 0 && (
                      <CudaChip label={drivers.join("/")} experimental={experimental} />
                    )}
                  </div>
                );
              })}
            </Fragment>
          );
        })}
      </div>
      <p className="dynref-cuda-caption">
        Pills show the minimum driver for each version &times; toolkit lane. CUDA 12 lanes end at
        v{CURRENT_TAG} &mdash; CUDA 12 container images are discontinued.
      </p>
    </>
  );
}

function CurrentTable() {
  const rows = rowsFor(CURRENT_TAG);
  return (
    <div className="dynref-cuda-scroll">
      <table className="dynref-cuda-table">
        <thead>
          <tr>
            <th>Backend</th>
            <th>CUDA toolkit</th>
            <th>Min driver</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.backend}-${row.toolkit}`}>
              <td>{row.backend}</td>
              <td>
                <CudaChip label={row.toolkit} experimental={row.note === "Experimental"} />
              </td>
              <td className="dynref-mono">{row.minDriver}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HistoryTable() {
  const versions = uniqueVersions();
  return (
    <div className="dynref-cuda-scroll">
      <table className="dynref-cuda-table">
        <thead>
          <tr>
            <th>Version</th>
            <th>Backend</th>
            <th>CUDA toolkit</th>
            <th>Min driver</th>
            <th>Notes</th>
          </tr>
        </thead>
        <tbody>
          {versions.map((version) => {
            const rows = rowsFor(version);
            return rows.map((row, index) => (
              <tr key={`${version}-${row.backend}-${row.toolkit}`}>
                {index === 0 && (
                  <td className="dynref-cuda-vcell dynref-mono" rowSpan={rows.length}>
                    v{version}
                  </td>
                )}
                <td>{row.backend}</td>
                <td>
                  <CudaChip label={row.toolkit} experimental={row.note === "Experimental"} />
                </td>
                <td className="dynref-mono">{row.minDriver}</td>
                <td className="dynref-cuda-note-cell">{row.note ?? ""}</td>
              </tr>
            ));
          })}
        </tbody>
      </table>
    </div>
  );
}

export function CudaDriverMatrix({ mode = "current" }: { mode?: "current" | "all" }) {
  return (
    <div>
      <style>{CUDA_CSS}</style>
      {mode === "current" ? (
        <CurrentTable />
      ) : (
        <>
          <span className="dynref-label">CUDA driver ladder</span>
          <CudaLadder />
          <span className="dynref-label">Full history</span>
          <HistoryTable />
          {CUDA_NOTES.map((note) => (
            <p key={note} className="dynref-grid-note">
              {note}
            </p>
          ))}
        </>
      )}
    </div>
  );
}
