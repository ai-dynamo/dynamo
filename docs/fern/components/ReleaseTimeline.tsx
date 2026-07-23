/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ReleaseTimeline — vertical timeline of every tracked release (newest first,
 * data order from releases.data.ts), followed by the crates.io first-
 * publication table. Node shape encodes the release kind: solid green =
 * stable, solid neutral = patch, dashed amber = platform preview /
 * model build.
 *
 * variant="artifacts" (default) keeps the artifact-delta blurbs and the
 * crates table — the Release Artifacts page. variant="notes" swaps stable
 * releases to their feature-voice notesSummary and suppresses the crates
 * table — the Release Notes overview.
 *
 * Server component; shared vocabulary (headings, badges, mono) comes from
 * ReferenceStyles — place <ReferenceStyles /> on the page alongside this
 * component. Only the .dynref-tl-* layout classes are defined here.
 */

import {
  CRATES_FIRST_PUBLISHED,
  RELEASES,
  type Release,
  type ReleaseKind,
} from "./releases.data";

const TL_CSS = `
.dynref-tl {
    margin: 20px 0;
}

.dynref-tl-item {
    display: grid;
    grid-template-columns: 14px minmax(0, 1fr);
    gap: 0 12px;
}

.dynref-tl-rail {
    display: flex;
    flex-direction: column;
    align-items: center;
}

.dynref-tl-node {
    flex: 0 0 auto;
    width: 12px;
    height: 12px;
    margin-top: 4px;
    border-radius: 50%;
}

.dynref-tl-node--stable {
    background: #76B900;
}

.dynref-tl-node--patch {
    background: #b5b5b5;
}

.dark .dynref-tl-node--patch {
    background: #3d3d3d;
}

.dynref-tl-node--preview {
    background: transparent;
    border: 2px dashed #C77E1B;
}

.dark .dynref-tl-node--preview {
    border-color: #FAC775;
}

.dynref-tl-line {
    flex: 1;
    width: 2px;
    margin-top: 4px;
    background: var(--border, var(--grayscale-a5));
}

.dynref-tl-body {
    min-width: 0;
    padding-bottom: 18px;
}

.dynref-tl-item:last-child .dynref-tl-body {
    padding-bottom: 0;
}

.dynref-tl-head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
}

.dynref-tl-version {
    color: var(--pst-color-text-base);
    font-size: 13.5px;
    font-weight: 600;
    text-decoration: none;
}

.dynref-tl-version:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}

.dynref-tl-date {
    color: var(--pst-color-text-muted);
    font-size: 12px;
}

.dynref-tl-gh {
    color: var(--pst-color-text-muted);
    font-size: 11.5px;
    text-decoration: none;
}

.dynref-tl-gh:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}

.dynref-tl-sum {
    margin: 4px 0 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.45;
}

.dynref-tl-crates {
    margin-top: 24px;
    padding-top: 18px;
    border-top: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-tl-table {
    width: 100%;
    margin-top: 10px;
    border-collapse: collapse;
    font-size: 12.5px;
}

.dynref-tl-table th {
    padding: 4px 16px 6px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-align: left;
    text-transform: uppercase;
}

.dynref-tl-table td {
    padding: 6px 16px 6px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    color: var(--pst-color-text-base);
}

.dynref-tl-table tbody tr:last-child td {
    border-bottom: 0;
}

.dynref-tl-table td.dynref-tl-cell-muted {
    color: var(--pst-color-text-muted);
}
`;

const KIND_BADGE: Record<ReleaseKind, { label: string; variant: string }> = {
  stable: { label: "GA release", variant: "green" },
  patch: { label: "Patch", variant: "gray" },
  "platform-preview": { label: "Early access", variant: "amber" },
  "model-build": { label: "Model build", variant: "amber" },
};

const NODE_CLASS: Record<ReleaseKind, string> = {
  stable: "dynref-tl-node--stable",
  patch: "dynref-tl-node--patch",
  "platform-preview": "dynref-tl-node--preview",
  "model-build": "dynref-tl-node--preview",
};

type TimelineVariant = "artifacts" | "notes";

function TimelineEntry({
  release,
  isLast,
  variant,
}: {
  release: Release;
  isLast: boolean;
  variant: TimelineVariant;
}) {
  const badge = KIND_BADGE[release.kind];
  const summary =
    variant === "notes" && release.kind === "stable"
      ? release.notesSummary ?? release.note ?? release.delta
      : release.delta ?? release.note;
  return (
    <div className="dynref-tl-item">
      <div className="dynref-tl-rail">
        <span className={`dynref-tl-node ${NODE_CLASS[release.kind]}`} />
        {!isLast && <span className="dynref-tl-line" />}
      </div>
      <div className="dynref-tl-body">
        <div className="dynref-tl-head">
          {release.notesHref || release.github ? (
            <a
              className="dynref-mono dynref-tl-version"
              href={release.notesHref ?? release.github}
            >
              {release.version}
            </a>
          ) : (
            <span className="dynref-mono dynref-tl-version">{release.version}</span>
          )}
          <span className={`dynref-badge dynref-badge--${badge.variant}`}>{badge.label}</span>
          {release.notesHref && release.github && (
            <a className="dynref-tl-gh" href={release.github}>
              GitHub ↗
            </a>
          )}
          {release.date && <span className="dynref-tl-date">{release.date}</span>}
        </div>
        {summary && <p className="dynref-tl-sum">{summary}</p>}
      </div>
    </div>
  );
}

export function ReleaseTimeline({ variant = "artifacts" }: { variant?: TimelineVariant }) {
  return (
    <>
      <style>{TL_CSS}</style>
      <section className="dynref-tl">
        {RELEASES.map((release, index) => (
          <TimelineEntry
            key={release.version}
            release={release}
            isLast={index === RELEASES.length - 1}
            variant={variant}
          />
        ))}
      </section>
    </>
  );
}

/* Crates.io first-publication table — crate-publishing metadata that lives on
   the Release Artifacts page (extracted from the timeline so it renders once,
   not duplicated alongside the release history). Reuses the .dynref-tl-* table
   styles. */
export function CratesFirstPublished() {
  return (
    <>
      <style>{TL_CSS}</style>
      <div className="dynref-tl-crates">
        <table className="dynref-tl-table">
          <thead>
            <tr>
              <th>Crate</th>
              <th>First version</th>
              <th>Published</th>
            </tr>
          </thead>
          <tbody>
            {CRATES_FIRST_PUBLISHED.map((entry) => (
              <tr key={entry.crate}>
                <td className="dynref-mono">{entry.crate}</td>
                <td className="dynref-mono">{entry.version}</td>
                <td className="dynref-tl-cell-muted">{entry.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="dynref-grid-note">
          dynamo-async-openai is deprecated; 1.0.2 is its final release. Use dynamo-protocols for new
          dependencies.
        </p>
      </div>
    </>
  );
}
