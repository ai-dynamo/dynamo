/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * DriverCheck — PROPOSAL, NOT MOUNTED ON ANY PUBLISHED PAGE.
 *
 * Minimal replacement sketch for RunsWhereWizard. The reader types the one
 * fact they can read off `nvidia-smi` (their driver version) and gets the
 * newest release each backend can run on it, plus how many releases qualify.
 *
 * The difference from the wizard it replaces is the direction of the
 * question. The wizard asked the reader to name their CUDA driver
 * *generation* — an abstraction they had to derive, and often the very thing
 * the page exists to tell them — and then hid all but one of the
 * backend/generation combinations it had already rendered. This asks for a
 * number the reader already has and does the comparison work for them across
 * every row of CUDA_HISTORY, which is the one job a static table cannot do.
 *
 * Everything is derived from releases.data.ts. With no input, the component
 * states what it needs and shows nothing speculative; it never hides data the
 * reader could otherwise see, because the full matrix is on the page anyway.
 *
 * Client component: the comparison depends on reader input, so unlike the
 * other Reference components this one cannot be server-rendered.
 */

"use client";

import { useMemo, useState } from "react";

import { CUDA_HISTORY, RELEASES, type CudaRow } from "./releases.data";

const BACKENDS = ["SGLang", "TensorRT-LLM", "vLLM"] as const;

/** "580.xx+" -> 580. Returns null for anything without a leading integer. */
function floorOf(row: CudaRow): number | null {
  const match = /^(\d+)/.exec(row.minDriver);
  return match ? Number(match[1]) : null;
}

/** Accepts what `nvidia-smi` prints: "580", "580.65", "580.65.06". */
function parseDriver(raw: string): number | null {
  const match = /^\s*(\d{2,4})(?:\.\d+)*\s*$/.exec(raw);
  return match ? Number(match[1]) : null;
}

/** Release recency, newest first — RELEASES is already in that order. */
const RELEASE_ORDER = new Map(
  RELEASES.map((release, index) => [release.version.replace(/^v/, ""), index]),
);

interface BackendResult {
  backend: string;
  newest?: { version: string; toolkits: string[] };
  qualifying: number;
  total: number;
}

function resultsFor(driver: number): BackendResult[] {
  return BACKENDS.map((backend) => {
    const rows = CUDA_HISTORY.filter((row) => row.backend === backend);
    const usable = rows.filter((row) => {
      const floor = floorOf(row);
      return floor !== null && floor <= driver;
    });
    const versions = [...new Set(usable.map((row) => row.version))].sort(
      (a, b) => (RELEASE_ORDER.get(a) ?? Infinity) - (RELEASE_ORDER.get(b) ?? Infinity),
    );
    const newest = versions[0];
    return {
      backend,
      newest: newest
        ? {
            version: newest,
            toolkits: [
              ...new Set(usable.filter((row) => row.version === newest).map((r) => r.toolkit)),
            ],
          }
        : undefined,
      qualifying: versions.length,
      total: new Set(rows.map((row) => row.version)).size,
    };
  });
}

const DC_CSS = `
.dynref-dc-ask {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 8px 12px;
    margin: 0 0 14px;
}

.dynref-dc-input {
    min-height: 36px;
    width: 140px;
    padding: 6px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--pst-color-surface);
    color: var(--pst-color-text-base);
    font-family: var(--font-mono, monospace);
    font-size: 14px;
}

.dynref-dc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 12px;
}

.dynref-dc-card {
    display: grid;
    gap: 6px;
    padding: 12px 14px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 10px;
}

.dynref-dc-version {
    font-size: 18px;
    font-weight: 600;
    line-height: 1.2;
}
`;

/** Seed from ?driver= so a result can be linked to. Empty during SSR. */
function initialDriver(): string {
  if (typeof window === "undefined") return "";
  return new URLSearchParams(window.location.search).get("driver") ?? "";
}

export function DriverCheck() {
  const [raw, setRaw] = useState(initialDriver);
  const driver = parseDriver(raw);
  const results = useMemo(() => (driver === null ? [] : resultsFor(driver)), [driver]);
  const touched = raw.trim().length > 0;

  return (
    <>
      <style>{DC_CSS}</style>
      <section className="dynref-panel">
        <div className="dynref-panel-header">
          <div>
            <p className="dynref-eyebrow">Preview component — not published</p>
            <h3 className="dynref-h">What can I run on my driver?</h3>
          </div>
        </div>

        <div className="dynref-dc-ask">
          <label className="dynref-label" htmlFor="dynref-dc-driver">
            Driver version
          </label>
          <input
            id="dynref-dc-driver"
            className="dynref-dc-input"
            type="text"
            inputMode="decimal"
            placeholder="580.65"
            value={raw}
            onChange={(event) => setRaw(event.target.value)}
            aria-describedby="dynref-dc-hint"
          />
          <span className="dynref-grid-note" id="dynref-dc-hint">
            Run <span className="dynref-mono">nvidia-smi</span> and copy the driver version.
          </span>
        </div>

        {touched && driver === null && (
          <p className="dynref-muted">
            Enter a driver version such as <span className="dynref-mono">580.65</span>.
          </p>
        )}

        {driver !== null && (
          <div className="dynref-dc-grid">
            {results.map((result) => (
              <div className="dynref-dc-card" key={result.backend}>
                <span className="dynref-label">{result.backend}</span>
                {result.newest ? (
                  <>
                    <span className="dynref-dc-version dynref-mono">v{result.newest.version}</span>
                    <span className="dynref-cuda-wrap">
                      {result.newest.toolkits.map((toolkit) => (
                        <span className="dynref-chip dynref-chip--cuda" key={toolkit}>
                          CUDA {toolkit}
                        </span>
                      ))}
                    </span>
                    <span className="dynref-grid-note">
                      newest of {result.qualifying} of {result.total} releases you can run
                    </span>
                  </>
                ) : (
                  <span className="dynref-muted">
                    No release ships {result.backend} for a driver this old.
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </section>
    </>
  );
}
