/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * UpgradeSelector — "coming from" upgrade picker for the Deprecations ledger.
 *
 * Answers "I run v1.1.x — what must I read to get to the current release?".
 * Self-configuring from releases.data: the target is CURRENT_VERSION and the
 * from-candidates are every older stable release that has a RELEASE_STATS
 * entry, labeled by line ("v1.2.x"). Each line maps to its LATEST release
 * including patches (v1.2.x -> v1.2.1) so the migration strip reflects the
 * pins the user actually runs. Each panel reuses UpgradePanel's internals
 * (buildRows + MigrationStrip + ReadingListFooter + UpgradePanelStyles); the
 * reading list is one breaking-changes chip for every stable release strictly
 * after the from line's base up to and including the current release, plus
 * the current release's known-issues chip (oldest -> newest, known issues
 * last).
 *
 * Switching is CSS-only, same mechanism as ArtifactBrowser: hidden radios
 * sit first inside the panel, the label pills live in the header rail, and
 * :checked general-sibling selectors reveal the matching pre-rendered panel
 * body (newest line default-checked). Server component; shared vocabulary
 * comes from ReferenceStyles — place <ReferenceStyles /> on the page. Only
 * the .dynref-us-* layout classes are defined here.
 */

import { CURRENT_VERSION, RELEASES, RELEASE_STATS, type Release } from "./releases.data";
import {
  buildRows,
  MigrationStrip,
  ReadingListFooter,
  UpgradePanelStyles,
  type ReadingItem,
} from "./UpgradePanel";

const US_CSS = `
/* Inputs are hidden by the shared .dynref-vh (visually hidden, focusable)
   class so the from-line rail stays keyboard-operable; per-line
   :focus-visible rules are generated in wiringCss alongside :checked. */

.dynref-us-rail {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}

.dynref-us-raillabel {
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
}

/* Filter pills — same treatment as the artifact browser rail. */
.dynref-us-pill {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 5px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded, 6px);
    background: transparent;
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    cursor: pointer;
}

.dynref-us-pill:hover {
    border-color: var(--nv-color-green, #76B900);
}

.dynref-us-panel {
    display: none;
}

.dynref-us-fromnote {
    margin: 0 0 2px;
    font-size: 12px;
}
`;

/* "v1.2.0" -> "us-v120" / "v1.2.x" — shared radio-id and line-label rules. */
function lineId(baseVersion: string): string {
  return `us-${baseVersion.replace(/\./g, "")}`;
}

function lineLabel(baseVersion: string): string {
  return baseVersion.replace(/\.\d+$/, ".x");
}

interface FromLine {
  /** Stable release the line is named after (v1.2.0 -> "v1.2.x"). */
  base: Release;
  /** Latest release on the line including patches — the pins the user runs. */
  latest: Release;
  id: string;
  label: string;
  readingList: ReadingItem[];
}

function buildFromLines(): FromLine[] {
  const currentIdx = RELEASES.findIndex((r) => r.version === CURRENT_VERSION);
  if (currentIdx < 0) return [];

  /* All stable releases carrying RELEASE_STATS, in RELEASES (newest-first)
     order, with their array index for older/newer comparisons. */
  const statStables = RELEASES.map((release, index) => ({ release, index })).filter(
    ({ release }) => release.kind === "stable" && RELEASE_STATS[release.version] !== undefined,
  );

  return statStables
    .filter(({ index }) => index > currentIdx)
    .map(({ release: base, index: baseIdx }) => {
      /* Latest stable/patch on the line: first RELEASES entry (array order is
         newest-first) sharing the "vA.B." prefix. Dev/model builds share the
         prefix but are not what operators run — exclude by kind. */
      const linePrefix = base.version.replace(/\d+$/, "");
      const latest =
        RELEASES.find(
          (r) =>
            (r.kind === "stable" || r.kind === "patch") && r.version.startsWith(linePrefix),
        ) ?? base;

      /* Every stable strictly newer than the line's base, up to and including
         current — oldest first — then current's known issues last. */
      const readingList: ReadingItem[] = statStables
        .filter(({ index }) => index >= currentIdx && index < baseIdx)
        .map(({ release }) => ({ version: release.version, kind: "breaking" as const }))
        .reverse();
      readingList.push({ version: CURRENT_VERSION, kind: "known-issues" });

      return {
        base,
        latest,
        id: lineId(base.version),
        label: lineLabel(base.version),
        readingList,
      };
    });
}

export function UpgradeSelector() {
  const current = RELEASES.find((r) => r.version === CURRENT_VERSION);
  const fromLines = buildFromLines();
  if (!current || fromLines.length === 0) return null;

  /* :checked and :focus-visible wiring is per-line, so the selectors are
     generated from the same derived list that renders the radios, pills, and
     panels. */
  const wiringCss = fromLines
    .map(
      (line) => `
#${line.id}:checked ~ .dynref-panel-header label[for="${line.id}"] {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}

#${line.id}:focus-visible ~ .dynref-panel-header label[for="${line.id}"] {
    outline: 2px solid var(--nv-color-green, #76B900);
    outline-offset: 1px;
}

#${line.id}:checked ~ #${line.id}-panel {
    display: block;
}
`,
    )
    .join("");

  return (
    <>
      <UpgradePanelStyles />
      <style>{US_CSS + wiringCss}</style>
      <section className="dynref-panel">
        {fromLines.map((line, index) => (
          <input
            className="dynref-us-radio dynref-vh"
            type="radio"
            id={line.id}
            name="dynref-us-from"
            defaultChecked={index === 0}
            key={line.id}
          />
        ))}

        <div className="dynref-panel-header">
          <p className="dynref-h">Upgrade to {CURRENT_VERSION}</p>
          <div className="dynref-us-rail">
            <span className="dynref-us-raillabel">from</span>
            {fromLines.map((line) => (
              <label className="dynref-us-pill" htmlFor={line.id} key={line.id}>
                {line.label}
              </label>
            ))}
          </div>
        </div>

        {fromLines.map((line) => (
          <div className="dynref-us-panel" id={`${line.id}-panel`} key={line.id}>
            <p className="dynref-muted dynref-us-fromnote">
              Pins shown from <span className="dynref-mono">{line.latest.version}</span>, the
              latest release on the {line.label} line.
            </p>
            <MigrationStrip
              rows={buildRows(line.latest.version, CURRENT_VERSION, line.latest.pins, current.pins)}
            />
            <ReadingListFooter items={line.readingList} />
          </div>
        ))}
      </section>
    </>
  );
}
