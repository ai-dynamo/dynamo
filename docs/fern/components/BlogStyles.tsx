/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Blog (digest) component styles.
 *
 * Styles for BlogLanding and BlogArticleMeta, including the blog-scoped
 * sidebar treatments (body:has(.dynamo-blog-home) #fern-sidebar ...) and the
 * article media reveals.
 *
 * Delivered as a page-level <style> block (NOT via the docs.yml `css:` field)
 * so it survives the shared NVIDIA global theme, which replaces project `css`
 * at publish. The same theme also replaces the custom `footer:`, so the
 * SITE_CSS block in CustomFooter.tsx does not reach these pages either.
 * Same pattern as ReferenceStyles.tsx, RecipeStyles.tsx and LandingStyles.tsx.
 *
 * Injected via dangerouslySetInnerHTML, like RecipeStyles.tsx, not as a text
 * child like ReferenceStyles.tsx. A text child is escaped on render, which
 * turns every `>` child combinator into `&gt;` and silently drops the rule.
 * ReferenceStyles gets away with it because it has no child combinators;
 * this bundle has dozens.
 *
 * Server component (no "use client"); registered via docs.yml
 * `experimental.mdx-components: ./components`. IMPORT it (ambient use is
 * unsupported -- renders "Unsupported JSX tag"); the @/ prefix resolves to the
 * fern/ root and is rewritten to a relative path at publish time:
 *
 *   import { BlogStyles } from `@/components/BlogStyles`;
 *
 * The backticks above stand in for the double quotes the real page uses, for
 * the reason spelled out in RecipeStyles.tsx: Fern's mdx-components bundler
 * regex-scans this file for imports without skipping comments, and a quoted
 * non-relative specifier makes it shell out to `npx rolldown` on every build.
 *
 * Then place <BlogStyles /> once, right after the imports, on the digest
 * landing page and every article page.
 */
const BLOG_CSS = `
/* ===================== Dynamo Blog ===================== */

body:has(.dynamo-blog-home),
body:has(.dynamo-blog-article) {
  --dynamo-blog-green: #76b900;
  --dynamo-blog-green-bright: #8ed000;
  --dynamo-blog-ink: var(--grayscale-a12);
  --dynamo-blog-muted: var(--grayscale-a10);
  --dynamo-blog-rule: rgba(127, 127, 127, 0.2);
}

body:has(.dynamo-blog-home) [data-testid="table-of-contents"] {
  display: none;
}

/* Give the Blog navigation the cadence of an archive rather than a docs tree. */
body:has(.dynamo-blog-home) aside nav,
body:has(.dynamo-blog-article) aside nav {
  font-size: 0.86rem;
}

body:has(.dynamo-blog-home) aside nav a,
body:has(.dynamo-blog-article) aside nav a {
  line-height: 1.35;
}

body:has(.dynamo-blog-home) aside nav button,
body:has(.dynamo-blog-article) aside nav button {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.7rem;
  font-weight: 700;
}

article:has(.dynamo-blog-home) {
  width: 100% !important;
  max-width: 1060px !important;
  padding: 0 clamp(0.25rem, 2vw, 1.5rem) 5rem;
}

article:has(.dynamo-blog-home) > header {
  display: none;
}

.dynamo-blog-home {
  position: relative;
  isolation: isolate;
  color: var(--dynamo-blog-ink);
}

.dynamo-blog-kicker {
  color: var(--dynamo-blog-green);
  font-size: 0.75rem;
  font-weight: 750;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.dynamo-blog-button {
  display: inline-flex;
  min-height: 2.65rem;
  align-items: center;
  justify-content: center;
  gap: 0.55rem;
  padding: 0.65rem 1rem;
  border: 1px solid transparent;
  border-radius: 10px;
  font-size: 0.86rem;
  font-weight: 700;
  text-decoration: none !important;
  transition: background 160ms ease, border-color 160ms ease;
}

.dynamo-blog-button svg,
.dynamo-blog-card__footer svg {
  width: 1.05rem;
  height: 1.05rem;
  flex: none;
}

.dynamo-blog-button--secondary {
  border-color: var(--dynamo-blog-rule);
  background: transparent;
  color: var(--dynamo-blog-ink) !important;
}

.dynamo-blog-button--secondary:hover {
  border-color: rgba(118, 185, 0, 0.5);
  background: rgba(118, 185, 0, 0.06);
}

.dynamo-blog-card__art-link {
  display: block;
  color: inherit;
  text-decoration: none !important;
}

.dynamo-blog-card h3 a {
  color: inherit;
  text-decoration: none !important;
}

.dynamo-blog-card h3 a:hover {
  color: var(--dynamo-blog-green);
}

.dynamo-blog-meta-line {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.48rem;
  color: var(--dynamo-blog-muted);
  font-size: 0.8rem;
}

.dynamo-blog-art {
  position: relative;
  height: 100%;
  min-height: 190px;
  overflow: hidden;
  filter: saturate(0.82);
  background:
    radial-gradient(circle at 25% 25%, rgba(255, 255, 255, 0.14), transparent 28%),
    linear-gradient(135deg, #172200, #365700 50%, #0b1300);
}

.dynamo-blog-art::after {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, transparent 40%, rgba(0, 0, 0, 0.35));
}

.dynamo-blog-art--snapshot { background: linear-gradient(145deg, #031d28, #006b78 52%, #001318); }

.dynamo-blog-art--tokens { background: linear-gradient(145deg, #211600, #9b5b00 52%, #1b0e00); }

.dynamo-blog-art--agents { background: linear-gradient(145deg, #210d31, #6d2693 52%, #14051d); }

.dynamo-blog-art--stack { background: linear-gradient(145deg, #0c1e37, #245aa6 52%, #050d18); }

.dynamo-blog-art--indexer { background: linear-gradient(145deg, #25120b, #a43d13 52%, #180903); }

.dynamo-blog-art--hillclimb { background: linear-gradient(145deg, #210d31, #6d2693 52%, #14051d); }

.dynamo-blog-art__climb {
  position: absolute;
  z-index: 1;
  inset: 12% 8% 14% 8%;
  opacity: 0.92;
}

.dynamo-blog-art__grid {
  position: absolute;
  opacity: 0.72;
  inset: -10%;
  transform: perspective(480px) rotateX(58deg) rotateZ(-12deg) scale(1.2);
  background:
    linear-gradient(rgba(255, 255, 255, 0.13) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.13) 1px, transparent 1px);
  background-size: 38px 38px;
  mask-image: radial-gradient(circle at 50% 50%, black, transparent 70%);
}

.dynamo-blog-art__orb {
  position: absolute;
  opacity: 0.72;
  z-index: 1;
  display: block;
  border: 1px solid rgba(255, 255, 255, 0.55);
  border-radius: 50%;
  box-shadow: inset 0 0 28px rgba(255, 255, 255, 0.16), 0 0 45px rgba(255, 255, 255, 0.13);
}

.dynamo-blog-art__orb--one {
  top: 18%;
  right: 15%;
  width: 7.5rem;
  height: 7.5rem;
}

.dynamo-blog-art__orb--two {
  right: 36%;
  bottom: 15%;
  width: 3.5rem;
  height: 3.5rem;
}

.dynamo-blog-art__line {
  position: absolute;
  z-index: 1;
  display: block;
  height: 1px;
  transform-origin: left center;
  background: rgba(255, 255, 255, 0.58);
}

.dynamo-blog-art__line--one {
  top: 42%;
  left: 17%;
  width: 62%;
  transform: rotate(-13deg);
}

.dynamo-blog-art__line--two {
  top: 63%;
  left: 27%;
  width: 48%;
  transform: rotate(19deg);
}

.dynamo-blog-art__mark {
  position: absolute;
  left: 9%;
  bottom: 5%;
  z-index: 2;
  color: rgba(255, 255, 255, 0.14);
  font-size: clamp(7rem, 14vw, 13rem);
  font-weight: 800;
  line-height: 0.8;
  letter-spacing: -0.08em;
}

.dynamo-blog-latest {
  padding-top: clamp(1.25rem, 2vw, 2rem);
  scroll-margin-top: 7rem;
}

.dynamo-blog-section-heading {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.dynamo-blog-section-heading h2 {
  margin: 0.55rem 0 0;
  font-size: clamp(2.25rem, 3.6vw, 3rem);
  font-weight: 600;
  line-height: 1;
  letter-spacing: -0.045em;
}

.dynamo-blog-section-heading__copy p {
  max-width: 560px;
  margin: 0.9rem 0 0;
  color: var(--dynamo-blog-muted);
  line-height: 1.6;
}

.dynamo-blog-section-heading__actions {
  flex: none;
}

.dynamo-blog-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.dynamo-blog-card {
  display: flex;
  min-width: 0;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--dynamo-blog-rule);
  border-radius: 16px;
  background: var(--grayscale-a1);
  transition: border-color 180ms ease, box-shadow 180ms ease;
}

.dark .dynamo-blog-card {
  background: #090909;
}

.dynamo-blog-card:hover {
  border-color: rgba(118, 185, 0, 0.45);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.09);
}

.dark .dynamo-blog-card:hover {
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.32);
}

.dynamo-blog-card__body {
  display: flex;
  flex: 1;
  flex-direction: column;
  padding: 1.6rem;
}

.dynamo-blog-card__topline,
.dynamo-blog-card__footer {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  color: var(--dynamo-blog-muted);
  font-size: 0.76rem;
}

.dynamo-blog-card__topline span:first-child {
  color: var(--dynamo-blog-green);
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.dynamo-blog-card h3 {
  margin: 1rem 0 0;
  font-size: clamp(1.35rem, 2vw, 1.75rem);
  font-weight: 600;
  line-height: 1.15;
  letter-spacing: -0.025em;
  text-wrap: balance;
}

.dynamo-blog-card__body > p {
  margin: 0.9rem 0 1.5rem;
  color: var(--dynamo-blog-muted);
  line-height: 1.65;
}

.dynamo-blog-card__footer {
  margin-top: auto;
  padding-top: 1.15rem;
  border-top: 1px solid var(--dynamo-blog-rule);
}

.dynamo-blog-card__footer a {
  display: grid;
  width: 2rem;
  height: 2rem;
  place-items: center;
  border-radius: 50%;
  background: rgba(118, 185, 0, 0.1);
  color: var(--dynamo-blog-green) !important;
}

/* Article pages: retain Markdown authoring, but present it as long-form editorial content. */
article:has(.dynamo-blog-article) {
  width: 100%;
  max-width: none;
}

article:has(.dynamo-blog-article) > header {
  position: relative;
  margin-bottom: 0;
  padding: 1.25rem 0 0;
}

article:has(.dynamo-blog-article) > header::before {
  content: "DYNAMO BLOG";
  display: block;
  margin-bottom: 1rem;
  color: var(--dynamo-blog-green);
  font-size: 0.72rem;
  font-weight: 750;
  letter-spacing: 0.14em;
}

article:has(.dynamo-blog-article) > header .fern-breadcrumb,
article:has(.dynamo-blog-article) > header .fern-page-subtitle {
  display: none;
}

article:has(.dynamo-blog-article) > header .fern-page-heading {
  max-width: 800px;
  font-size: clamp(2.75rem, 4.8vw, 4.35rem);
  font-weight: 600;
  line-height: 1.02;
  letter-spacing: -0.045em;
  text-wrap: balance;
}

article:has(.dynamo-blog-article) > header div:has(.fern-page-heading) {
  width: 100%;
}

.dynamo-blog-article {
  margin: 1.55rem 0 clamp(3rem, 6vw, 5rem);
  padding: 1.4rem 0;
  border-top: 1px solid var(--dynamo-blog-rule);
  border-bottom: 1px solid var(--dynamo-blog-rule);
}

.dynamo-blog-article__byline,
.dynamo-blog-article__details,
.dynamo-blog-article__actions {
  display: flex;
  align-items: center;
}

.dynamo-blog-article__byline {
  flex-wrap: wrap;
  gap: 0.7rem 1rem;
}

.dynamo-blog-article__category {
  padding: 0.3rem 0.65rem;
  border-radius: 999px;
  background: rgba(118, 185, 0, 0.12);
  color: var(--dynamo-blog-green);
  font-size: 0.72rem;
  font-weight: 750;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.dynamo-blog-article__authors {
  color: var(--dynamo-blog-ink);
  font-size: 0.88rem;
  font-weight: 600;
}

.dynamo-blog-article__details {
  justify-content: space-between;
  gap: 1rem;
  margin-top: 1.1rem;
}

.dynamo-blog-article__actions {
  gap: 0.55rem;
}

.dynamo-blog-article__actions button,
.dynamo-blog-article__actions a {
  display: inline-flex;
  min-height: 2.3rem;
  align-items: center;
  justify-content: center;
  gap: 0.42rem;
  padding: 0.45rem 0.8rem;
  border: 1px solid var(--dynamo-blog-rule);
  border-radius: 999px;
  background: transparent;
  color: var(--dynamo-blog-ink) !important;
  font: inherit;
  font-size: 0.78rem;
  font-weight: 700;
  line-height: 1;
  text-decoration: none !important;
  cursor: pointer;
  transition: border-color 150ms ease, background 150ms ease;
}

.dynamo-blog-article__actions button:hover,
.dynamo-blog-article__actions a:hover {
  border-color: rgba(118, 185, 0, 0.55);
  background: rgba(118, 185, 0, 0.08);
}

.dynamo-blog-article__actions svg {
  width: 0.95rem;
  height: 0.95rem;
  fill: currentColor;
}

article:has(.dynamo-blog-article) > p,
article:has(.dynamo-blog-article) > ul,
article:has(.dynamo-blog-article) > ol,
article:has(.dynamo-blog-article) > blockquote {
  font-size: 1.04rem;
  line-height: 1.82;
}

article:has(.dynamo-blog-article) > p:first-of-type {
  color: var(--dynamo-blog-ink);
  font-size: clamp(1.15rem, 2vw, 1.32rem);
  line-height: 1.7;
}

article:has(.dynamo-blog-article) h2 {
  margin-top: 3.75rem;
  font-size: clamp(2rem, 3.8vw, 2.8rem);
  font-weight: 600;
  line-height: 1.1;
  letter-spacing: -0.035em;
}

article:has(.dynamo-blog-article) h3 {
  margin-top: 2.6rem;
  font-size: 1.55rem;
  font-weight: 600;
  line-height: 1.2;
  letter-spacing: -0.02em;
}

article:has(.dynamo-blog-article) img {
  border: 1px solid var(--dynamo-blog-rule);
  border-radius: 16px;
  box-shadow: 0 18px 50px rgba(0, 0, 0, 0.1);
}

article:has(.dynamo-blog-article) figcaption {
  color: var(--dynamo-blog-muted);
  font-size: 0.8rem;
  line-height: 1.5;
  text-align: center;
}

article:has(.dynamo-blog-article) pre {
  border-radius: 14px;
}

@media (max-width: 900px) {

  .dynamo-blog-section-heading {
    align-items: flex-start;
    flex-direction: column;
  }
}

@media (max-width: 680px) {

  article:has(.dynamo-blog-home),
  article:has(.dynamo-blog-article) {
    padding-inline: 0.5rem;
  }

  .dynamo-blog-section-heading__actions,
  .dynamo-blog-section-heading__actions .dynamo-blog-button {
    width: 100%;
  }

  .dynamo-blog-grid {
    grid-template-columns: 1fr;
  }

  .dynamo-blog-section-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  article:has(.dynamo-blog-article) > header {
    padding-top: 1rem;
  }

  article:has(.dynamo-blog-article) > header .fern-page-heading {
    font-size: clamp(2.55rem, 11vw, 3.45rem);
  }

  .dynamo-blog-article__details {
    align-items: flex-start;
    flex-direction: column;
  }
}

@media (prefers-reduced-motion: reduce) {

  .dynamo-blog-button,
  .dynamo-blog-card {
    transition: none;
  }
}

/* Date-first archive labels, inspired by editorial blog indexes. */
body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/agent-optimization-skills"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/agent-optimization-skills"]::before { content: "AUG 21"; }

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/dynosim-pareto-frontier"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/dynosim-pareto-frontier"]::before { content: "MAY 29"; }

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/dynamo-snapshot-fast-startup"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/dynamo-snapshot-fast-startup"]::before { content: "MAY 28"; }

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/tokenspeed-day-0"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/tokenspeed-day-0"]::before { content: "MAY 06"; }

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/agentic-harnesses"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/agentic-harnesses"]::before { content: "APR 30"; }

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/agentic-inference"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/agentic-inference"]::before { content: "MAR 2026"; }

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/flash-indexer"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href$="/flash-indexer"]::before { content: "FEB 23"; }

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[href*="/digest/"]::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[href*="/digest/"]::before {
  display: block;
  margin-bottom: 0.18rem;
  color: var(--dynamo-blog-green);
  font-size: 0.61rem;
  font-weight: 750;
  letter-spacing: 0.1em;
  line-height: 1;
}

@media (min-width: 901px) {

  .dynamo-blog-grid > .dynamo-blog-card:first-child {
    grid-column: 1 / -1;
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(300px, 0.85fr);
  }

  .dynamo-blog-grid > .dynamo-blog-card:first-child .dynamo-blog-card__art-link,
  .dynamo-blog-grid > .dynamo-blog-card:first-child .dynamo-blog-art {
    min-height: 250px;
    height: 100%;
  }

  .dynamo-blog-grid > .dynamo-blog-card:first-child .dynamo-blog-card__body {
    justify-content: center;
    padding: clamp(1.75rem, 3vw, 2.4rem);
  }

  .dynamo-blog-grid > .dynamo-blog-card:first-child h3 {
    font-size: clamp(1.7rem, 2.6vw, 2.3rem);
  }
}

/* Blog archive sidebar: compact, date-led, and intentionally distinct from docs navigation. */
body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group {
  gap: 0.35rem;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"],
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"] {
  position: relative;
  min-height: 4.2rem;
  align-items: flex-start;
  margin: 0.35rem 0.35rem 0.9rem;
  padding: 0.85rem 0.9rem;
  overflow: hidden;
  border: 1px solid rgba(118, 185, 0, 0.28);
  border-radius: 15px;
  background:
    radial-gradient(circle at 90% 0%, rgba(118, 185, 0, 0.22), transparent 45%),
    linear-gradient(135deg, rgba(118, 185, 0, 0.12), rgba(118, 185, 0, 0.035));
  color: var(--dynamo-blog-ink);
}

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/digest"] > svg,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/digest"] > svg {
  width: 1.15rem;
  height: 1.15rem;
  margin-top: 0.08rem;
  padding: 0.24rem;
  box-sizing: content-box;
  border-radius: 8px;
  background: var(--dynamo-blog-green);
  color: #0b1400;
}

/* The stacked title-and-kicker treatment, and the "2026 · ENGINEERING STORIES"
   label under it, belonged to the tile. The entry is a plain sidebar link now,
   and the label had also gone stale — the section it sits on is "Dynamo blogs",
   which is not year-scoped.

   The type is levelled further down, where the entry's own font rule lives. */

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 {
  position: relative;
  margin: 0 0.35rem;
  padding-left: 0.55rem;
  border-left: 1px solid rgba(118, 185, 0, 0.22);
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 > li,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 > li {
  position: relative;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 > li::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 > li::before {
  content: "";
  position: absolute;
  top: 1.12rem;
  left: -0.72rem;
  z-index: 2;
  width: 0.32rem;
  height: 0.32rem;
  border: 2px solid var(--grayscale-a1);
  border-radius: 50%;
  background: rgba(118, 185, 0, 0.55);
  box-sizing: content-box;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a {
  display: flex;
  min-height: 3.45rem;
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
  gap: 0.22rem;
  margin: 0.12rem 0;
  padding: 0.58rem 0.68rem;
  border: 1px solid transparent;
  border-radius: 10px;
  transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a:hover,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a:hover {
  border-color: rgba(118, 185, 0, 0.18);
  background: rgba(118, 185, 0, 0.055);
  transform: translateX(2px);
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[data-state="active"],
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[data-state="active"] {
  border-color: rgba(118, 185, 0, 0.3);
  background: linear-gradient(90deg, rgba(118, 185, 0, 0.15), rgba(118, 185, 0, 0.04));
  box-shadow: inset 2px 0 0 var(--dynamo-blog-green);
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 .fern-sidebar-link-title,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 .fern-sidebar-link-title {
  width: 100%;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 .fern-sidebar-link-title-inner {
  display: -webkit-box;
  overflow: hidden;
  color: var(--dynamo-blog-ink);
  font-size: 0.79rem;
  font-weight: 620;
  line-height: 1.22;
  letter-spacing: -0.005em;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a::before {
  order: -1;
  margin: 0 0 0.08rem;
  color: var(--dynamo-blog-green);
  font-size: 0.57rem;
  font-weight: 780;
  letter-spacing: 0.11em;
  line-height: 1;
}

.dynamo-blog-article__authors a {
  color: inherit;
  text-decoration-color: rgba(118, 185, 0, 0.45);
  text-underline-offset: 0.18em;
}

.dynamo-blog-article__authors a:hover {
  color: var(--dynamo-blog-green);
}

/* The custom Share and Subscribe controls replace Fern's docs-oriented page toolbar. */
article:has(.dynamo-blog-article) > header {
  border-bottom: 0 !important;
}

/* Blog archive refinement: remove the literal timeline and use soft editorial rows. */
body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"],
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"] {
  border-color: transparent;
  background:
    radial-gradient(circle at 92% 5%, rgba(118, 185, 0, 0.18), transparent 46%),
    linear-gradient(135deg, rgba(118, 185, 0, 0.1), rgba(255, 255, 255, 0.02));
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.38),
    0 10px 30px rgba(0, 0, 0, 0.045);
  backdrop-filter: blur(14px);
}

.dark body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"],
.dark body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"] {
  background:
    radial-gradient(circle at 92% 5%, rgba(118, 185, 0, 0.2), transparent 46%),
    linear-gradient(135deg, rgba(118, 185, 0, 0.1), rgba(255, 255, 255, 0.025));
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    0 12px 34px rgba(0, 0, 0, 0.24);
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 {
  margin-inline: 0.35rem;
  padding-left: 0;
  border-left: 0;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 > li::before,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 > li::before {
  display: none;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a {
  min-height: 3.7rem;
  margin: 0.18rem 0;
  padding: 0.62rem 0.78rem;
  border: 0;
  border-radius: 11px;
  background: transparent;
  box-shadow: none;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a:hover,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a:hover {
  border-color: transparent;
  background:
    linear-gradient(90deg, rgba(118, 185, 0, 0.09), rgba(118, 185, 0, 0.025));
  box-shadow: inset 0 1px 0 rgba(118, 185, 0, 0.12);
  transform: translateX(2px);
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group-level-1 a[data-state="active"],
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group-level-1 a[data-state="active"] {
  border-color: transparent;
  background:
    radial-gradient(circle at 100% 0%, rgba(118, 185, 0, 0.13), transparent 50%),
    linear-gradient(90deg, rgba(118, 185, 0, 0.13), rgba(118, 185, 0, 0.035));
  box-shadow:
    inset 2px 0 0 rgba(118, 185, 0, 0.82),
    inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

/* Existing article media reveals with a soft focus-to-sharp transition. */
article:has(.dynamo-blog-article) img.dynamo-blog-image-reveal {
  opacity: 0;
  transform: translateY(34px) scale(0.965);
  filter: blur(9px) saturate(0.82);
  transition:
    opacity 720ms cubic-bezier(0.2, 0.75, 0.2, 1) var(--dynamo-blog-image-delay, 0ms),
    transform 820ms cubic-bezier(0.2, 0.75, 0.2, 1) var(--dynamo-blog-image-delay, 0ms),
    filter 760ms ease var(--dynamo-blog-image-delay, 0ms);
  will-change: opacity, transform, filter;
}

article:has(.dynamo-blog-article) img.dynamo-blog-image-reveal[data-revealed="true"] {
  opacity: 1;
  transform: translateY(0) scale(1);
  filter: blur(0) saturate(1);
}

@media (prefers-reduced-motion: reduce) {

  article:has(.dynamo-blog-article) img.dynamo-blog-image-reveal {
    animation: none;
    transition: none;
  }

  article:has(.dynamo-blog-article) img.dynamo-blog-image-reveal {
    opacity: 1;
    transform: none;
    filter: none;
  }
}

/* The /digest entry is a normal nav link, and now a peer of the "External
   publications" page entry directly above it in the Blog tab. The inset margin
   and min-height left over from its days as the bespoke "Latest" tile put the
   two at different indents and different heights, so both go: it takes the
   ordinary top-level sidebar box, like its neighbour. Padding and radius are
   already settled by main.css, which declares them !important.

   No backticks in this block: it sits inside the BLOG_CSS template literal, so
   one would close the string and hand the rest of the file to the JS parser. */
body:has(.dynamo-blog-home) #fern-sidebar a[href$="/digest"],
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/digest"] {
  align-items: center;
  margin: 0.18rem 0;
  border: 0;
  background: transparent;
  box-shadow: none;
  backdrop-filter: none;
}

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/digest"]:hover,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/digest"]:hover,
body:has(.dynamo-blog-home) #fern-sidebar a[href$="/digest"][data-state="active"],
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/digest"][data-state="active"] {
  background: linear-gradient(90deg, rgba(118, 185, 0, 0.12), rgba(118, 185, 0, 0.025));
  box-shadow: inset 2px 0 0 rgba(118, 185, 0, 0.8);
}

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/digest"] > svg,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/digest"] > svg {
  display: none;
}

/* "Dynamo blogs" is a section and the other two are pages, and Fern weights
   section headers more heavily. All three sit side by side at the top of this
   tab as siblings, so they take the same type rather than reading as two
   different ranks. */
body:has(.dynamo-blog-home) #fern-sidebar a[href$="/digest"] .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/digest"] .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-home) #fern-sidebar a[href$="/external-publications"] .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/external-publications"] .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-home) #fern-sidebar a[href$="/research-publications"] .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/research-publications"] .fern-sidebar-link-title-inner {
  display: block;
  font-size: 0.8rem;
  font-weight: 720;
  line-height: 1.3;
  letter-spacing: 0;
}

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/digest"] .fern-sidebar-link-title-inner::after,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/digest"] .fern-sidebar-link-title-inner::after {
  content: none;
}

/* Avoid clipping descenders such as the g in Agentic. */
body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-link-title-inner {
  padding-bottom: 0.1rem;
  line-height: 1.34;
}

/* Keep Latest visibly clickable without presenting it as another section masthead. */
body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"],
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"] {
  background: transparent !important;
  box-shadow: none !important;
}

body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"]:hover,
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"]:hover,
body:has(.dynamo-blog-home) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"][data-state="active"],
body:has(.dynamo-blog-article) #fern-sidebar .fern-sidebar-group > li > a[href$="/digest"][data-state="active"] {
  background: linear-gradient(90deg, rgba(118, 185, 0, 0.11), rgba(118, 185, 0, 0.02)) !important;
  box-shadow: inset 2px 0 0 rgba(118, 185, 0, 0.78) !important;
}

/* Nested Fern links use their anchor ::before for the active indicator; dates live with the title. */
body:has(.dynamo-blog-home) #fern-sidebar a[href*="/digest/"]::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href*="/digest/"]::before {
  content: "" !important;
}

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/agent-optimization-skills"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/agent-optimization-skills"] .fern-sidebar-link-title-inner::before { content: "AUG 21"; }

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/dynosim-pareto-frontier"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/dynosim-pareto-frontier"] .fern-sidebar-link-title-inner::before { content: "MAY 29"; }

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/dynamo-snapshot-fast-startup"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/dynamo-snapshot-fast-startup"] .fern-sidebar-link-title-inner::before { content: "MAY 28"; }

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/tokenspeed-day-0"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/tokenspeed-day-0"] .fern-sidebar-link-title-inner::before { content: "MAY 06"; }

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/agentic-harnesses"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/agentic-harnesses"] .fern-sidebar-link-title-inner::before { content: "APR 30"; }

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/agentic-inference"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/agentic-inference"] .fern-sidebar-link-title-inner::before { content: "MAR 2026"; }

body:has(.dynamo-blog-home) #fern-sidebar a[href$="/flash-indexer"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href$="/flash-indexer"] .fern-sidebar-link-title-inner::before { content: "FEB 23"; }

body:has(.dynamo-blog-home) #fern-sidebar a[href*="/digest/"] .fern-sidebar-link-title-inner,
body:has(.dynamo-blog-article) #fern-sidebar a[href*="/digest/"] .fern-sidebar-link-title-inner {
  display: flex !important;
  overflow: visible !important;
  flex-direction: column;
  gap: 0.24rem;
  padding-bottom: 0.14rem;
  line-height: 1.36;
  -webkit-line-clamp: unset;
}

body:has(.dynamo-blog-home) #fern-sidebar a[href*="/digest/"] .fern-sidebar-link-title-inner::before,
body:has(.dynamo-blog-article) #fern-sidebar a[href*="/digest/"] .fern-sidebar-link-title-inner::before {
  display: block;
  color: var(--dynamo-blog-green);
  font-size: 0.57rem;
  font-weight: 780;
  letter-spacing: 0.11em;
  line-height: 1;
}
`;

export function BlogStyles() {
  return <style dangerouslySetInnerHTML={{ __html: BLOG_CSS }} />;
}
