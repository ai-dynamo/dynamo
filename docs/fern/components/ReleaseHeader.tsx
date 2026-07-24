/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ReleaseHeader — masthead panel for a Release Notes page.
 *
 * Renders the release title row (version, GA badge, date from releases.data),
 * GitHub / Artifacts link chips, and a stat-tile strip (PRs merged,
 * contributors, first-time contributors, breaking changes, known issues)
 * driven entirely by RELEASE_STATS[version] in releases.data — pages pass
 * only the version. Optional stats simply drop their tiles when absent;
 * a version without a RELEASE_STATS entry renders the header with no tiles.
 * Breaking-changes and known-issues tiles deep-link into the Deprecations /
 * Known Issues pages.
 *
 * Server component (no "use client"); shared vocabulary (panel, eyebrow,
 * badges, muted text) comes from ReferenceStyles — place <ReferenceStyles />
 * on the page alongside this component. Only the .dynref-rh-* layout classes
 * are defined here. All stat tiles share the neutral border; semantic color
 * lives only in the number (amber for breaking changes / known issues).
 */

import { RELEASES, RELEASE_STATS } from "./releases.data";

const RH_CSS = `
.dynref-rh-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 8px 16px;
}

.dynref-rh-title {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    margin: 0;
    color: var(--pst-color-text-base);
    font-size: 21px;
    font-weight: 600;
    line-height: 1.2;
}

.dynref-rh-links {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}

.dynref-rh-link {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 6px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    white-space: nowrap;
}

.dark .dynref-rh-link {
    border-color: #383838;
}

.dynref-rh-link:hover {
    border-color: var(--pst-color-text-muted);
    color: var(--pst-color-text-base);
}

.dynref-rh-tiles {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(105px, 1fr));
    gap: 10px;
    margin-top: 16px;
}

/* All tiles share the neutral border — semantic color lives in the number
   only. Rendered as <a> when the tile deep-links; same box either way. */
.dynref-rh-tile {
    display: block;
    padding: 10px 12px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 10px;
    text-decoration: none;
}

.dark .dynref-rh-tile {
    background: #1d1d1d;
    border-color: #2e2e2e;
}

a.dynref-rh-tile:hover {
    border-color: var(--pst-color-text-muted);
}

.dynref-rh-num {
    display: block;
    color: var(--pst-color-text-base);
    font-size: 18px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1.3;
}

.dynref-rh-num--amber {
    color: var(--dynref-amber-fg);
}

.dynref-rh-tilelabel {
    display: block;
    margin-top: 2px;
    color: var(--pst-color-text-muted);
    font-size: 11px;
}
`;

interface StatTile {
  label: string;
  value: number;
  amber: boolean;
  href?: string;
}

export function ReleaseHeader(props: { version: string }) {
  const release = RELEASES.find((r) => r.version === props.version);
  /* All counts come from RELEASE_STATS in releases.data — the single source
     shared with UpgradePanel and the ledger accordion titles. */
  const stats = RELEASE_STATS[props.version];
  /* v1.3.0 -> "v130" — anchor shape shared with the Deprecations and Known
     Issues pages' per-version section ids. */
  const versionAnchor = props.version.replace(/\./g, "");

  const tiles: StatTile[] = [];
  if (stats) {
    if (typeof stats.prs === "number") {
      tiles.push({ label: "PRs merged", value: stats.prs, amber: false });
    }
    if (typeof stats.contributors === "number") {
      tiles.push({ label: "Contributors", value: stats.contributors, amber: false });
    }
    if (typeof stats.firstTimers === "number") {
      tiles.push({ label: "First-time contributors", value: stats.firstTimers, amber: false });
    }
    tiles.push({
      label: "Breaking changes",
      value: stats.breaking,
      amber: true,
      href: `/dynamo/dev/reference/releases/deprecations#${versionAnchor}`,
    });
    tiles.push({
      label: "Known issues",
      value: stats.knownIssues,
      amber: true,
      href:
        stats.knownIssues > 0
          ? `/dynamo/dev/reference/releases/known-issues#${versionAnchor}`
          : undefined,
    });
  }

  return (
    <>
      <style>{RH_CSS}</style>
      <section className="dynref-panel">
        <div className="dynref-rh-header">
          <div>
            <p className="dynref-eyebrow">Release notes</p>
            <div className="dynref-rh-title">
              Dynamo {props.version}
              <span className="dynref-badge dynref-badge--green">GA release</span>
              {release?.date && <span className="dynref-muted">{release.date}</span>}
            </div>
          </div>
          <div className="dynref-rh-links">
            {release?.github && (
              <a className="dynref-rh-link" href={release.github}>
                GitHub &#8599;
              </a>
            )}
            <a className="dynref-rh-link" href="/dynamo/dev/reference/release-artifacts">
              Artifacts
            </a>
          </div>
        </div>

        {tiles.length > 0 && (
        <div className="dynref-rh-tiles">
          {tiles.map((tile) => {
            const body = (
              <>
                <span className={tile.amber ? "dynref-rh-num dynref-rh-num--amber" : "dynref-rh-num"}>
                  {tile.value}
                </span>
                <span className="dynref-rh-tilelabel">{tile.label}</span>
              </>
            );
            return tile.href ? (
              <a className="dynref-rh-tile" href={tile.href} key={tile.label}>
                {body}
              </a>
            ) : (
              <div className="dynref-rh-tile" key={tile.label}>
                {body}
              </div>
            );
          })}
        </div>
        )}
      </section>
    </>
  );
}
