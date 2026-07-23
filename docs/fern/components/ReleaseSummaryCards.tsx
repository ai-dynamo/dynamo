/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ReleaseSummaryCards — "what changed" area cards for a Release Notes page.
 *
 * Renders a responsive grid of per-area summary cards: colored area chip,
 * title, plain-text body, and an optional "changes in detail" anchor link
 * into the detailed section further down the page. Body text may arrive with
 * markdown bold/backticks from the notes pipeline; it is rendered as plain
 * text (the ** pairs and backticks are stripped mechanically, not parsed).
 *
 * Server component (no "use client"); shared vocabulary comes from
 * ReferenceStyles — place <ReferenceStyles /> on the page alongside this
 * component. Only the .dynref-sc-* layout classes are defined here. The area
 * chip palette reuses the exact chip rgba values from ReferenceStyles
 * (arch blue, centos violet, cuda teal, ubuntu orange, gray badge) plus the
 * shared --dynref-green-* tokens.
 */

export interface SummaryCard {
  area: string;
  title: string;
  body: string;
  anchor?: string;
}

const SC_CSS = `
.dynref-sc-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 12px;
    margin: 24px 0;
}

/* Panel recipe, card-sized. */
.dynref-sc-card {
    display: flex;
    flex-direction: column;
    padding: 16px 18px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--pst-color-surface);
}

.dark .dynref-sc-card {
    background: #161616;
    border-color: #2b2b2b;
}

.dynref-sc-title {
    margin: 8px 0 0;
    color: var(--pst-color-text-base);
    font-size: 14px;
    font-weight: 600;
}

.dynref-sc-body {
    margin: 6px 0 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.55;
}

.dark .dynref-sc-body {
    color: #a8a8a8;
}

.dynref-sc-more {
    display: inline-block;
    margin-top: 10px;
    color: var(--nv-color-green-2, #538300);
    font-size: 12px;
    font-weight: 600;
    text-decoration: none;
}

.dark .dynref-sc-more {
    color: #76B900;
}

.dynref-sc-more:hover {
    text-decoration: underline;
}

/* Area chip — badge-sized, self-contained. Color variants replicate the
   ReferenceStyles chip palettes (values duplicated deliberately; the green
   variant uses the shared tokens). */
.dynref-sc-chip {
    display: inline-flex;
    align-items: center;
    align-self: flex-start;
    padding: 1px 8px;
    border: 1px solid transparent;
    border-radius: 6px;
    font-size: 11.5px;
    font-weight: 600;
    white-space: nowrap;
}

/* Blue — arch palette (Frontend, Multimodal). */
.dynref-sc-chip--blue {
    background: rgba(37, 99, 235, 0.1);
    color: #1D4ED8;
    border-color: rgba(37, 99, 235, 0.28);
}

.dark .dynref-sc-chip--blue {
    background: rgba(59, 130, 246, 0.18);
    color: #93C5FD;
    border-color: rgba(59, 130, 246, 0.42);
}

/* Violet — centos palette (Router). */
.dynref-sc-chip--violet {
    background: rgba(124, 58, 237, 0.12);
    color: #6D28D9;
    border-color: rgba(124, 58, 237, 0.3);
}

.dark .dynref-sc-chip--violet {
    background: rgba(139, 92, 246, 0.2);
    color: #C4B5FD;
    border-color: rgba(139, 92, 246, 0.42);
}

/* Teal — cuda palette (RL, KVBM). */
.dynref-sc-chip--teal {
    background: rgba(13, 148, 136, 0.12);
    color: #0F766E;
    border-color: rgba(13, 148, 136, 0.3);
}

.dark .dynref-sc-chip--teal {
    background: rgba(13, 148, 136, 0.2);
    color: #5EEAD4;
    border-color: rgba(13, 148, 136, 0.45);
}

/* Orange — ubuntu palette (Kubernetes). */
.dynref-sc-chip--orange {
    background: rgba(233, 84, 32, 0.12);
    color: #C7401C;
    border-color: rgba(233, 84, 32, 0.3);
}

.dark .dynref-sc-chip--orange {
    background: rgba(233, 84, 32, 0.2);
    color: #FF9068;
    border-color: rgba(233, 84, 32, 0.42);
}

/* Green — shared tokens (Planner). */
.dynref-sc-chip--green {
    background: var(--dynref-green-bg);
    color: var(--dynref-green-fg);
    border-color: var(--dynref-green-border);
}

/* Gray — badge--gray palette (Mocker, Hardware, General, fallback). */
.dynref-sc-chip--gray {
    background: rgba(120, 120, 120, 0.1);
    color: #5F5E5A;
    border-color: rgba(120, 120, 120, 0.28);
}

.dark .dynref-sc-chip--gray {
    background: #242424;
    color: #a8a8a8;
    border-color: #383838;
}
`;

type ChipVariant = "blue" | "violet" | "teal" | "orange" | "green" | "gray";

/* Area -> chip color. Unknown areas fall back to gray. */
const AREA_VARIANT: Record<string, ChipVariant> = {
  Frontend: "blue",
  Router: "violet",
  RL: "teal",
  Kubernetes: "orange",
  Multimodal: "blue",
  Planner: "green",
  Mocker: "gray",
  KVBM: "teal",
  Hardware: "gray",
  General: "gray",
};

/* Bodies may carry markdown bold/code/links from the notes pipeline; render
   as plain text — strip the markers mechanically, never parse. Links keep
   their visible text ([text](url) → text); the card's anchor link is the
   only navigation affordance. */
function stripInlineMarkdown(text: string): string {
  return text
    .replace(/\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\*\*/g, "")
    .replace(/`/g, "");
}

export function ReleaseSummaryCards({ cards }: { cards: SummaryCard[] }) {
  return (
    <>
      <style>{SC_CSS}</style>
      <div className="dynref-sc-grid">
        {cards.map((card) => {
          const variant = AREA_VARIANT[card.area] ?? "gray";
          const paragraphs = stripInlineMarkdown(card.body)
            .split(/\n{2,}/)
            .map((p) => p.trim())
            .filter((p) => p.length > 0);
          return (
            <div className="dynref-sc-card" key={`${card.area} ${card.title}`}>
              <span className={`dynref-sc-chip dynref-sc-chip--${variant}`}>{card.area}</span>
              <p className="dynref-sc-title">{card.title}</p>
              {paragraphs.map((paragraph) => (
                <p className="dynref-sc-body" key={paragraph}>
                  {paragraph}
                </p>
              ))}
              {card.anchor && (
                <a className="dynref-sc-more" href={`#${card.anchor}`}>
                  {card.area} changes in detail &#8595;
                </a>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}
