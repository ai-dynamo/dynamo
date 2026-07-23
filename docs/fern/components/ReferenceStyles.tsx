/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Reference page shared component styles.
 *
 * Shared style vocabulary for the Reference custom components
 * (CompatibilityHero and the other reference-page components). Each component
 * carries only its own layout classes; the panel, eyebrow, label, mono, chip,
 * badge, and click-to-copy treatments all live here so the pages read as one
 * system. Chip variants replicate the .dynamo-chip-* palette in main.css.
 *
 * Delivered as a page-level <style> block (NOT via the docs.yml `css:` field)
 * so it survives the shared NVIDIA global theme, which replaces project `css`
 * at publish. Mirrors the prod-proven pattern in NVIDIA-NeMo/DataDesigner
 * (fern/components/BlogCard.tsx) — same global-theme: nvidia, same no-css
 * constraint, CSS injected this exact way.
 *
 * Server component (no "use client"); registered via docs.yml
 * `experimental.mdx-components: ./components`. IMPORT it (ambient use is
 * unsupported — renders "Unsupported JSX tag"); the @/ prefix resolves to the
 * fern/ root and is rewritten to a relative path at publish time:
 *   import { ReferenceStyles } from "@/components/ReferenceStyles";
 * Then place <ReferenceStyles /> once, right after the frontmatter, on every
 * Reference page that uses these components.
 */
const REFERENCE_CSS = `
/* Dark-mode variable re-bind.
   The shared NVIDIA theme defines the dark values of --pst-color-text-base,
   --pst-color-text-muted, and --pst-color-surface only under
   html[data-theme="dark"], but Fern's theme toggle flips dark mode with the
   .dark *class* and does NOT set data-theme. So in real dark mode these three
   resolve to their LIGHT values (#1a1a1a text, #666 muted, #f7f7f7 surface),
   which our components use for text/panels — rendering dark-on-dark. Re-bind
   them under .dark so they track the class. !important is required: the
   theme's light-default selector outranks a bare .dark. Scoped safely because
   this stylesheet only loads on the Reference pages. (--nv-color-bg-default
   never flips even with data-theme; those surfaces keep their own .dark
   background overrides below.) */
.dark {
    --pst-color-text-base: #eee !important;
    --pst-color-text-muted: #999 !important;
    --pst-color-surface: #1a1a1a !important;
}

/* Card container shared by every reference component. */
.dynref-panel {
    margin: 24px 0;
    padding: 20px 22px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--pst-color-surface);
}

.dark .dynref-panel {
    background: #161616;
    border-color: #2b2b2b;
}

/* Light mode uses a darkened green — 12px bold #76B900 on white is ~2.2:1,
   below AA. Dark mode restores the brand green. */
.dynref-eyebrow {
    margin: 0 0 6px;
    color: #527D00;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.dark .dynref-eyebrow {
    color: var(--nv-color-green, #76B900);
}

.dynref-label {
    color: var(--pst-color-text-muted);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.dynref-mono {
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 0.9em;
    font-variant-numeric: tabular-nums;
}

.dynref-muted {
    color: var(--pst-color-text-muted);
    font-size: 13px;
}

/* Section header helpers. */
.dynref-panel-header {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 14px;
}

.dynref-h {
    margin: 0;
    color: var(--pst-color-text-base);
    font-size: 15px;
    font-weight: 600;
}

/* Category chips — palette replicated from main.css .dynamo-chip-*. */
.dynref-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    margin: 0.15rem 0.1rem;
    padding: 0.1rem 0.5rem;
    border: 1px solid transparent;
    border-radius: 6px;
    font-size: 0.8125rem;
    font-weight: 600;
}

/* NVIDIA green — GPU generations. */
.dynref-chip--gpu {
    background: rgba(118, 185, 0, 0.16);
    color: #4D7C0F;
    border-color: rgba(118, 185, 0, 0.38);
}

.dark .dynref-chip--gpu {
    background: rgba(118, 185, 0, 0.22);
    color: #A3E635;
    border-color: rgba(118, 185, 0, 0.46);
}

/* Ubuntu — brand orange. */
.dynref-chip--ubuntu {
    background: rgba(233, 84, 32, 0.12);
    color: #C7401C;
    border-color: rgba(233, 84, 32, 0.3);
}

.dark .dynref-chip--ubuntu {
    background: rgba(233, 84, 32, 0.2);
    color: #FF9068;
    border-color: rgba(233, 84, 32, 0.42);
}

/* CentOS — violet. */
.dynref-chip--centos {
    background: rgba(124, 58, 237, 0.12);
    color: #6D28D9;
    border-color: rgba(124, 58, 237, 0.3);
}

.dark .dynref-chip--centos {
    background: rgba(139, 92, 246, 0.2);
    color: #C4B5FD;
    border-color: rgba(139, 92, 246, 0.42);
}

/* CPU architecture — blue. */
.dynref-chip--arch {
    background: rgba(37, 99, 235, 0.1);
    color: #1D4ED8;
    border-color: rgba(37, 99, 235, 0.28);
}

.dark .dynref-chip--arch {
    background: rgba(59, 130, 246, 0.18);
    color: #93C5FD;
    border-color: rgba(59, 130, 246, 0.42);
}

/* Experimental modifier — dashed border marks experimental/preview entries,
   matching the heatmap's dashed-amber "experimental" encoding. Composes with
   any chip color variant. */
.dynref-chip--exp {
    border-style: dashed;
}

/* Amber — experimental/preview entries (pairs with --exp for the dashed
   treatment so "experimental" reads identically everywhere). */
.dynref-chip--amber {
    background: rgba(239, 159, 39, 0.14);
    color: #854F0B;
    border-color: rgba(239, 159, 39, 0.35);
}

.dark .dynref-chip--amber {
    background: rgba(239, 159, 39, 0.16);
    color: #FAC775;
    border-color: rgba(239, 159, 39, 0.4);
}

/* CUDA toolkit — teal. */
.dynref-chip--cuda {
    background: rgba(13, 148, 136, 0.12);
    color: #0F766E;
    border-color: rgba(13, 148, 136, 0.3);
}

.dark .dynref-chip--cuda {
    background: rgba(13, 148, 136, 0.2);
    color: #5EEAD4;
    border-color: rgba(13, 148, 136, 0.45);
}

/* Semantic badges — green = stable/promoted/supported, amber = early-access/
   caveat, gray = neutral/patch, red = deprecated, blue = info/version-tag. */
.dynref-badge {
    display: inline-flex;
    align-items: center;
    padding: 1px 8px;
    border: 1px solid transparent;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 600;
    white-space: nowrap;
}

.dynref-badge--green {
    background: rgba(118, 185, 0, 0.16);
    color: #4D7C0F;
    border-color: rgba(118, 185, 0, 0.38);
}

.dark .dynref-badge--green {
    background: rgba(118, 185, 0, 0.22);
    color: #A3E635;
    border-color: rgba(118, 185, 0, 0.46);
}

.dynref-badge--amber {
    background: rgba(239, 159, 39, 0.14);
    color: #854F0B;
    border-color: rgba(239, 159, 39, 0.35);
}

.dark .dynref-badge--amber {
    background: rgba(239, 159, 39, 0.16);
    color: #FAC775;
    border-color: rgba(239, 159, 39, 0.4);
}

.dynref-badge--gray {
    background: rgba(120, 120, 120, 0.1);
    color: #5F5E5A;
    border-color: rgba(120, 120, 120, 0.28);
}

.dark .dynref-badge--gray {
    background: #242424;
    color: #a8a8a8;
    border-color: #383838;
}

.dynref-badge--red {
    background: rgba(226, 75, 74, 0.1);
    color: #A32D2D;
    border-color: rgba(226, 75, 74, 0.3);
}

.dark .dynref-badge--red {
    background: rgba(226, 75, 74, 0.16);
    color: #F09595;
    border-color: rgba(226, 75, 74, 0.4);
}

/* Work-in-progress / experimental — dashed amber, matching the heatmap's
   experimental cell encoding. */
.dynref-badge--wip {
    background: transparent;
    color: #854F0B;
    border: 1.5px dashed rgba(185, 122, 23, 0.7);
}

.dark .dynref-badge--wip {
    color: #FAC775;
    border-color: rgba(239, 159, 39, 0.55);
}

.dynref-badge--blue {
    background: rgba(37, 99, 235, 0.1);
    color: #1D4ED8;
    border-color: rgba(37, 99, 235, 0.28);
}

.dark .dynref-badge--blue {
    background: rgba(59, 130, 246, 0.18);
    color: #93C5FD;
    border-color: rgba(59, 130, 246, 0.42);
}

/* Click-to-copy affordance. Used as
   <button className="dynref-copy dynref-badge dynref-badge--blue" type="button"
           data-dynref-copy="...">label</button>
   custom.js copies data-dynref-copy to the clipboard on click and toggles
   .dynref-copied for 1.2s; the state flip to the green palette is the only
   feedback — no icons. Background/border/color inherit from the badge variant. */
.dynref-copy {
    cursor: pointer;
    font: inherit;
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 11.5px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    line-height: 1;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.dynref-copy:hover {
    border-color: currentColor;
}

.dynref-copy:focus-visible {
    outline: 2px solid var(--nv-color-green, #76B900);
    outline-offset: 1px;
}

/* Clipboard glyph makes the copy affordance discoverable; flips to a check
   while the .dynref-copied state (set by custom.js) is active. */
.dynref-copy::before {
    content: "⧉";
    margin-right: 5px;
    font-size: 11px;
    opacity: 0.55;
}

.dynref-copy:hover::before {
    opacity: 1;
}

.dynref-copy.dynref-copied::before {
    content: "✓";
    opacity: 1;
}

.dynref-copy.dynref-copied {
    background: rgba(118, 185, 0, 0.16);
    color: #4D7C0F;
    border-color: rgba(118, 185, 0, 0.38);
}

.dark .dynref-copy.dynref-copied {
    background: rgba(118, 185, 0, 0.22);
    color: #A3E635;
    border-color: rgba(118, 185, 0, 0.46);
}

.dynref-grid-note {
    margin-top: 12px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
}

/* Outline badge modifier — same tint family, transparent fill. Distinguishes
   adjacent-but-different states (e.g. Promoted vs Recipe-in-GA greens). */
.dynref-badge--outline {
    background: transparent;
}

/* Panels rendered as the first content of an accordion sit too far below the
   summary with their default 24px margin. */
details .dynref-panel,
details .dynref-vm-scroll,
details .dynref-cuda-wrap {
    margin-top: 8px;
}
`;

export function ReferenceStyles() {
  return <style>{REFERENCE_CSS}</style>;
}
