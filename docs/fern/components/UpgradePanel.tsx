/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * UpgradePanel — "upgrade to this release" panel for a Release Notes page.
 *
 * Renders the upgrade target, an informational from-version (plain muted
 * mono text — deliberately NOT chip-styled, so it does not read as an
 * interactive filter), a migration strip computed from releases.data
 * (backend pins, NIXL pins, CUDA toolkits + discontinuation badge, minimum
 * driver), and a "Read before upgrading" reading list of link chips. Reading
 * list items are (version, kind) pairs — labels and counts derive from
 * RELEASE_STATS in releases.data, and hrefs from the shared vXYZ anchor rule;
 * items whose version has no RELEASE_STATS entry are skipped.
 *
 * Server component (no "use client"); shared vocabulary comes from
 * ReferenceStyles — place <ReferenceStyles /> on the page alongside this
 * component. Only the .dynref-up-* layout classes are defined here. Source
 * pills are neutral, target pills green — except when a pin pair is
 * identical, where the target pill stays neutral with an "unchanged" note
 * (no false green).
 *
 * The internals — buildRows, the migration strip, the reading-list footer,
 * and the .dynref-up-* stylesheet (UpgradePanelStyles) — are exported for
 * reuse by UpgradeSelector, which renders one pre-built panel body per
 * from-line. The MDX-facing surface of UpgradePanel itself is unchanged.
 */

import { RELEASES, CUDA_HISTORY, RELEASE_STATS, type BackendPins } from "./releases.data";

const UP_CSS = `
.dynref-up-from {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 6px;
}

/* Informational, not interactive — no border or background. */
.dynref-up-fromver {
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
}

.dynref-up-strip {
    margin-top: 4px;
}

.dynref-up-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 0;
    padding: 5px 0;
}

/* Single shared label column — pin rows, flattened NIXL rows, CUDA, and
   Driver all align on it. */
.dynref-up-rowlabel {
    flex: 0 0 112px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
}

.dynref-up-pill {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border: 1px solid transparent;
    border-radius: 6px;
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}

/* Source — always neutral; also the target style for unchanged pairs. */
.dynref-up-pill--src {
    background: rgba(120, 120, 120, 0.08);
    color: var(--pst-color-text-muted);
    border-color: rgba(120, 120, 120, 0.25);
}

.dark .dynref-up-pill--src {
    background: #242424;
    color: #a8a8a8;
    border-color: #383838;
}

/* Target — green tint, only when the value actually changes. */
.dynref-up-pill--dst {
    background: var(--dynref-green-bg);
    color: var(--dynref-green-fg);
    border-color: var(--dynref-green-border);
}

.dynref-up-arrow {
    margin: 0 6px;
    color: var(--pst-color-text-muted);
}

.dynref-up-unchanged {
    margin-left: 8px;
    color: var(--pst-color-text-muted);
    font-size: 11px;
}

.dynref-up-rowbadge {
    margin-left: 8px;
}

.dynref-up-footer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px 10px;
    margin-top: 14px;
    padding-top: 12px;
    border-top: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-up-footerlabel {
    color: var(--pst-color-text-base);
    font-size: 12.5px;
    font-weight: 500;
}

/* Reading-list link chips — blue family, matching the badge--blue palette. */
.dynref-up-read {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border: 1px solid var(--dynref-blue-border);
    border-radius: 6px;
    background: var(--dynref-blue-bg);
    color: var(--dynref-blue-fg);
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
    white-space: nowrap;
}

.dynref-up-read:hover {
    border-color: currentColor;
}
`;

type PinKey = "sglang" | "trtllm" | "vllm" | "nixlSglang" | "nixlTrtllm" | "nixlVllm";

/* TRT-LLM abbreviation keeps the 112px label column tight; NIXL sub-rows are
   flattened into the same single column. */
const PIN_ROWS: { key: PinKey; label: string }[] = [
  { key: "sglang", label: "SGLang" },
  { key: "trtllm", label: "TRT-LLM" },
  { key: "vllm", label: "vLLM" },
  { key: "nixlSglang", label: "NIXL · SGLang" },
  { key: "nixlTrtllm", label: "NIXL · TRT-LLM" },
  { key: "nixlVllm", label: "NIXL · vLLM" },
];

export interface MigrationRow {
  label: string;
  from: string;
  to: string;
  badge?: string;
}

function uniq(values: string[]): string[] {
  return values.filter((value, index) => values.indexOf(value) === index);
}

/* "v1.3.0" -> CUDA_HISTORY rows keyed "1.3.0". */
function cudaRowsFor(versionTag: string) {
  const version = versionTag.replace(/^v/, "");
  return CUDA_HISTORY.filter((row) => row.version === version);
}

export function buildRows(
  fromTag: string,
  toTag: string,
  fromPins: BackendPins | undefined,
  toPins: BackendPins | undefined,
): MigrationRow[] {
  const rows: MigrationRow[] = [];

  for (const { key, label } of PIN_ROWS) {
    const fromPin = fromPins?.[key];
    const toPin = toPins?.[key];
    if (fromPin && toPin) rows.push({ label, from: fromPin, to: toPin });
  }

  const fromCuda = cudaRowsFor(fromTag);
  const toCuda = cudaRowsFor(toTag);
  const fromToolkits = uniq(fromCuda.map((row) => row.toolkit));
  const toToolkits = uniq(toCuda.map((row) => row.toolkit));
  if (fromToolkits.length > 0 && toToolkits.length > 0) {
    const has12 = (toolkits: string[]) => toolkits.some((toolkit) => /^12\./.test(toolkit));
    rows.push({
      label: "CUDA",
      from: fromToolkits.join(" / "),
      to: toToolkits.join(" / "),
      badge: has12(fromToolkits) && !has12(toToolkits) ? "CUDA 12 ends" : undefined,
    });
  }

  const fromDrivers = uniq(fromCuda.map((row) => row.minDriver));
  const toDrivers = uniq(toCuda.map((row) => row.minDriver));
  if (fromDrivers.length > 0 && toDrivers.length > 0) {
    rows.push({ label: "Driver", from: fromDrivers.join(" / "), to: toDrivers.join(" / ") });
  }

  return rows;
}

/* The .dynref-up-* stylesheet as a component, so UpgradeSelector can carry
   the same visual vocabulary without duplicating the CSS. */
export function UpgradePanelStyles() {
  return <style>{UP_CSS}</style>;
}

export interface ReadingItem {
  version: string;
  kind: "breaking" | "known-issues";
}

/* Labels and hrefs derive from RELEASE_STATS + the shared "v1.3.0" -> "v130"
   anchor rule (same as ReleaseHeader); versions without stats are skipped. */
function buildReadingChips(readingList: ReadingItem[]): { label: string; href: string }[] {
  return readingList.flatMap((item) => {
    const stats = RELEASE_STATS[item.version];
    if (!stats) return [];
    const anchor = item.version.replace(/\./g, "");
    return item.kind === "breaking"
      ? [
          {
            label: `${item.version} breaking changes (${stats.breaking})`,
            href: `/dynamo/dev/reference/releases/deprecations#${anchor}`,
          },
        ]
      : [
          {
            label: `${item.version} known issues (${stats.knownIssues})`,
            href: `/dynamo/dev/reference/releases/known-issues#${anchor}`,
          },
        ];
  });
}

/* Migration strip — one from -> to row per pin, CUDA, and driver. Renders
   nothing when there are no rows. */
export function MigrationStrip({ rows }: { rows: MigrationRow[] }) {
  if (rows.length === 0) return null;
  return (
    <div className="dynref-up-strip">
      {rows.map((row) => {
        const unchanged = row.from === row.to;
        return (
          <div className="dynref-up-row" key={row.label}>
            <span className="dynref-up-rowlabel">{row.label}</span>
            <span className="dynref-up-pill dynref-up-pill--src">{row.from}</span>
            <span className="dynref-up-arrow">&#8594;</span>
            <span
              className={
                unchanged
                  ? "dynref-up-pill dynref-up-pill--src"
                  : "dynref-up-pill dynref-up-pill--dst"
              }
            >
              {row.to}
            </span>
            {unchanged && <span className="dynref-up-unchanged">unchanged</span>}
            {row.badge && (
              <span className="dynref-badge dynref-badge--red dynref-up-rowbadge">
                {row.badge}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* "Read before upgrading" footer — link chips derived from RELEASE_STATS.
   Renders nothing when every item was skipped (no stats). */
export function ReadingListFooter({ items }: { items: ReadingItem[] }) {
  const readingChips = buildReadingChips(items);
  if (readingChips.length === 0) return null;
  return (
    <div className="dynref-up-footer">
      <span className="dynref-up-footerlabel">Read before upgrading</span>
      {readingChips.map((item) => (
        <a className="dynref-up-read" href={item.href} key={item.href}>
          {item.label}
        </a>
      ))}
    </div>
  );
}

export function UpgradePanel(props: {
  toVersion: string;
  fromVersion: { version: string; label: string };
  readingList: ReadingItem[];
}) {
  const from = RELEASES.find((r) => r.version === props.fromVersion.version);
  const to = RELEASES.find((r) => r.version === props.toVersion);

  const rows = buildRows(props.fromVersion.version, props.toVersion, from?.pins, to?.pins);

  return (
    <>
      <UpgradePanelStyles />
      <section className="dynref-panel">
        <div className="dynref-panel-header">
          <p className="dynref-h">Upgrade to {props.toVersion}</p>
          <div className="dynref-up-from">
            <span className="dynref-muted">from</span>
            <span className="dynref-mono dynref-up-fromver">{props.fromVersion.label}</span>
          </div>
        </div>

        <MigrationStrip rows={rows} />

        <ReadingListFooter items={props.readingList} />
      </section>
    </>
  );
}
