/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * TagLookup — "Have a tag? Look it up." Reverse lookup from a container tag
 * to what it is: the release or early-access build it belongs to, the
 * runtimes it applies to, ship date, GA-path status (EA builds), and the
 * breaking/known-issue chips (stable releases).
 *
 * The tag set is derived from data only: unique container tag labels for the
 * current release from ARTIFACTS, plus every early-access tag in
 * MODEL_EA_BUILDS. Lookup is CSS-only: one hidden radio per tag (plus a
 * default-checked "none" radio) sits first inside the panel; :checked
 * general-sibling selectors highlight the picked pill and reveal its
 * pre-rendered detail card. No tag selected shows a muted hint row.
 *
 * Server component; shared vocabulary (panel, badges, mono, muted) comes
 * from ReferenceStyles — place <ReferenceStyles /> on the page alongside
 * this component. Only the .dynref-tg-* layout classes are defined here.
 */

import {
  ARTIFACTS,
  CURRENT_VERSION,
  MODEL_EA_BUILDS,
  RELEASES,
  RELEASE_STATS,
  type GaPath,
  type ModelEaBuild,
} from "./releases.data";

interface StableTagEntry {
  kind: "stable";
  tag: string;
  /** Container image names that publish this tag. */
  images: string[];
}

interface EaTagEntry {
  kind: "ea";
  tag: string;
  build: ModelEaBuild;
}

type TagEntry = StableTagEntry | EaTagEntry;

/* Current-release container tags from ARTIFACTS, deduped to unique tag
   strings (e.g. one "1.3.0" entry across seven images, one "1.3.0-efa"
   across the three runtimes), then the EA tags in data order. */
function deriveEntries(): TagEntry[] {
  const stable: StableTagEntry[] = [];
  for (const artifact of ARTIFACTS) {
    if (artifact.category !== "container") continue;
    for (const tag of artifact.tags) {
      const existing = stable.find((entry) => entry.tag === tag.label);
      if (existing) {
        existing.images.push(artifact.name);
      } else {
        stable.push({ kind: "stable", tag: tag.label, images: [artifact.name] });
      }
    }
  }
  const ea: EaTagEntry[] = MODEL_EA_BUILDS.map((build) => ({ kind: "ea", tag: build.tag, build }));
  return [...stable, ...ea];
}

const ENTRIES = deriveEntries();

/* Same GA-path → badge-variant mapping as ModelEABuildCards, so a build's
   status reads identically on both pages. */
const GA_BADGE_VARIANT: Record<GaPath, string> = {
  promoted: "green",
  "recipe-in-ga": "green",
  "dev-only": "amber",
  superseded: "gray",
};

/* Shared "v1.3.0" -> "v130" anchor rule (same as UpgradePanel/ReleaseHeader). */
function versionAnchor(version: string): string {
  return version.replace(/\./g, "");
}

/* Per-tag :checked selectors are generated from the entry list so the CSS
   always matches the rendered radios. */
const TG_CSS = `
.dynref-tg-input {
    /* display:none, not visually-hidden positioning — focusable inputs would
       scroll the page on pill click; label activation still toggles them.
       Same rationale as .dynref-ab-filter in ArtifactBrowser. */
    display: none;
}

.dynref-tg-rail {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0 0 12px;
}

.dynref-tg-pill {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    padding: 4px 9px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded, 6px);
    background: transparent;
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    cursor: pointer;
}

.dynref-tg-pill:hover {
    border-color: var(--nv-color-green, #76B900);
}

${ENTRIES.map((_, i) => `#tg-t${i}:checked ~ .dynref-tg-rail label[for="tg-t${i}"]`).join(",\n")} {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}

.dynref-tg-card {
    display: none;
    padding: 14px 16px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: #fcfcfc;
}

.dark .dynref-tg-card {
    background: #0f0f0f;
    border-color: #2b2b2b;
}

${ENTRIES.map((_, i) => `#tg-t${i}:checked ~ .dynref-tg-cards > [data-tg="t${i}"]`).join(",\n")} {
    display: block;
}

/* Default state: the hidden "none" radio is checked and only the hint shows. */
.dynref-tg-hint {
    display: none;
    margin: 0;
    padding: 14px 16px;
    border: 1px dashed var(--border, var(--grayscale-a5));
    border-radius: 8px;
}

#tg-none:checked ~ .dynref-tg-cards > .dynref-tg-hint {
    display: block;
}

.dynref-tg-cardhead {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.dynref-tg-tagname {
    color: var(--pst-color-text-base);
    font-size: 13px;
    font-weight: 700;
}

.dynref-tg-rows {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.dynref-tg-row {
    display: flex;
    align-items: baseline;
    gap: 10px;
    font-size: 12.5px;
    line-height: 1.45;
}

.dynref-tg-key {
    flex: 0 0 76px;
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.dynref-tg-val {
    color: var(--pst-color-text-base);
    overflow-wrap: anywhere;
}

.dynref-tg-link {
    color: var(--nv-color-green-2, #538300);
    font-weight: 600;
    text-decoration: none;
}

.dark .dynref-tg-link {
    color: var(--nv-color-green, #76B900);
}

.dynref-tg-link:hover {
    text-decoration: underline;
}

.dynref-tg-note {
    margin: 0;
}

.dynref-tg-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}

.dynref-tg-chip {
    display: inline-flex;
    align-items: center;
    padding: 3px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    line-height: 1.2;
    text-decoration: none;
}

.dynref-tg-chip:hover {
    border-color: var(--nv-color-green, #76B900);
    color: var(--pst-color-text-base);
}

.dark .dynref-tg-chip {
    border-color: #333;
    background: #1c1c1c;
}
`;

function StableCard({ entry }: { entry: StableTagEntry }) {
  const release = RELEASES.find((r) => r.version === CURRENT_VERSION);
  const stats = RELEASE_STATS[CURRENT_VERSION];
  const anchor = versionAnchor(CURRENT_VERSION);
  return (
    <>
      <div className="dynref-tg-cardhead">
        <span className="dynref-mono dynref-tg-tagname">{entry.tag}</span>
        <span className="dynref-badge dynref-badge--green">Stable · {CURRENT_VERSION}</span>
      </div>
      <div className="dynref-tg-rows">
        <div className="dynref-tg-row">
          <span className="dynref-tg-key">Release</span>
          <span className="dynref-tg-val">
            {release?.notesHref ? (
              <a className="dynref-tg-link" href={release.notesHref}>
                {CURRENT_VERSION} release notes
              </a>
            ) : (
              CURRENT_VERSION
            )}
          </span>
        </div>
        <div className="dynref-tg-row">
          <span className="dynref-tg-key">Images</span>
          <span className="dynref-tg-val dynref-mono">{entry.images.join(", ")}</span>
        </div>
        {release?.date && (
          <div className="dynref-tg-row">
            <span className="dynref-tg-key">Shipped</span>
            <span className="dynref-tg-val">{release.date}</span>
          </div>
        )}
      </div>
      {stats && (
        <div className="dynref-tg-chips">
          <a className="dynref-tg-chip" href={`/dynamo/dev/reference/deprecations#${anchor}`}>
            {stats.breaking} breaking changes
          </a>
          <a className="dynref-tg-chip" href={`/dynamo/dev/reference/known-issues#${anchor}`}>
            {stats.knownIssues} known issues
          </a>
        </div>
      )}
    </>
  );
}

function EaCard({ entry }: { entry: EaTagEntry }) {
  const { build } = entry;
  const link = build.recipeHref
    ? { href: build.recipeHref, label: build.recipeLabel ?? "Recipe" }
    : build.github
      ? { href: build.github, label: "Release tag on GitHub" }
      : null;
  return (
    <>
      <div className="dynref-tg-cardhead">
        <span className="dynref-mono dynref-tg-tagname">{entry.tag}</span>
        <span
          className={`dynref-badge dynref-badge--${GA_BADGE_VARIANT[build.gaPath]}${build.gaPath === "recipe-in-ga" ? " dynref-badge--outline" : ""}`}
        >
          {build.gaLabel}
        </span>
      </div>
      <div className="dynref-tg-rows">
        <div className="dynref-tg-row">
          <span className="dynref-tg-key">Build</span>
          <span className="dynref-tg-val">
            {build.model} early access · {build.releaseLine} line
            {link && (
              <>
                {" — "}
                <a className="dynref-tg-link" href={link.href}>
                  {link.label}
                </a>
              </>
            )}
          </span>
        </div>
        <div className="dynref-tg-row">
          <span className="dynref-tg-key">Runtimes</span>
          <span className="dynref-tg-val dynref-mono">{build.runtimes.join(", ")}</span>
        </div>
        <div className="dynref-tg-row">
          <span className="dynref-tg-key">Shipped</span>
          <span className="dynref-tg-val">{build.shipped}</span>
        </div>
        <div className="dynref-tg-row">
          <span className="dynref-tg-key">Status</span>
          <span className="dynref-tg-val">{build.statusLine}</span>
        </div>
      </div>
    </>
  );
}

export function TagLookup() {
  return (
    <>
      <style>{TG_CSS}</style>
      <section className="dynref-panel">
        <input className="dynref-tg-input" type="radio" id="tg-none" name="dynref-tg-tag" defaultChecked />
        {ENTRIES.map((entry, i) => (
          <input
            key={entry.tag}
            className="dynref-tg-input"
            type="radio"
            id={`tg-t${i}`}
            name="dynref-tg-tag"
          />
        ))}

        <div className="dynref-panel-header">
          <div>
            <p className="dynref-eyebrow">Tag lookup</p>
            <h3 className="dynref-h">Have a tag? Look it up.</h3>
          </div>
          <p className="dynref-muted dynref-tg-note">
            {ENTRIES.length} known tags · current release + early access
          </p>
        </div>

        <div className="dynref-tg-rail">
          {ENTRIES.map((entry, i) => (
            <label key={entry.tag} className="dynref-tg-pill" htmlFor={`tg-t${i}`}>
              {entry.tag}
            </label>
          ))}
        </div>

        <div className="dynref-tg-cards">
          <p className="dynref-muted dynref-tg-hint">Select a tag.</p>
          {ENTRIES.map((entry, i) => (
            <div key={entry.tag} className="dynref-tg-card" data-tg={`t${i}`}>
              {entry.kind === "stable" ? <StableCard entry={entry} /> : <EaCard entry={entry} />}
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
