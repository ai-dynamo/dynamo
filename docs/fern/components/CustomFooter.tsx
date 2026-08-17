/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * Custom footer for Dynamo docs (Fern native header/footer).
 *
 * SITE_CSS mirrors docs/fern/main.css, which stays canonical and is served via
 * docs.yml `css:` for local and unthemed builds. The inline copy keeps those
 * styles available whenever Fern renders this project footer; it cannot cover
 * hosted builds where the shared NVIDIA global theme replaces both project CSS
 * and the footer. Hosted theme-chrome corrections instead live in SiteStyles
 * and are injected into the disposable composed pages before Fern generates
 * them. Inlined rather than imported because Fern custom footer components
 * cannot import local modules.
 *
 * Do not edit the block by hand: run
 * `python3 docs/fern/scripts/sync_site_css.py` after changing main.css.
 * Pre-commit enforces the mirror with `--check`.
 */
// sync-site-css:begin (generated from ../main.css)
const SITE_CSS = `
/*!
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

/* Color themes for light and dark modes */
:root {
    /* Brand Colors */
    --nv-color-green: #76B900;
    --nv-color-green-2: #004B31;
    --nv-color-black: #000000;
    --nv-color-white: #FFFFFF;

    /* Grey Scale - Light */
    --nv-light-grey-1: #f7f7f7;
    --nv-light-grey-2: #EEEEEE;
    --nv-light-grey-3: #DDDDDD;
    --nv-light-grey-4: #CCCCCC;
    --nv-light-grey-5: #999999;

    /* Grey Scale - Dark */
    --nv-dark-grey-1: #111111;
    --nv-dark-grey-2: #1A1A1A;
    --nv-dark-grey-3: #222222;
    --nv-dark-grey-4: #333333;
    --nv-dark-grey-5: #666666;

    /* Colors by Usage */
    --nv-color-text: #000000;
    --nv-color-bg-default: #FFFFFF;
    --nv-color-bg-alt: #f7f7f7;
    --nv-color-success: #76B900;
    --nv-color-error: #f44336;

    /* Theme-independent settings */
    --rounded: 999px;
}
main {
    min-height: calc(100vh - 200px);
  }
/* Typography - Headers */
h1 {
    font-size: 36px;
    font-weight: 700;
    line-height: 1.25em; /* 45px */
}

h2 {
    font-size: 28px;
    font-weight: 700;
    line-height: 1.25em; /* 35px */
}

h3 {
    font-size: 24px;
    font-weight: 700;
    line-height: 1.25em; /* 30px */
}

h4 {
    font-size: 20px;
    font-weight: 700;
    line-height: 1.25em; /* 25px */
}

/* Typography - Paragraphs */
.prose{
    color: var(--nv-dark-grey-2) !important;
}
.dark .prose{
    color: var(--nv-light-grey-2) !important;
}
p {
    text-decoration-thickness: 3px;
}
.fern-mdx-link {
    color: var(--tw-prose-body);
    text-decoration-color: var(--accent);
    font-weight: var(--font-weight-normal);
}

/* Light theme (default) */
html:not([data-theme]),html[data-theme=light] {
    --pst-color-background: #fff;
    --pst-color-on-background: #fff;
    --pst-color-shadow: #ccc;
    --pst-color-heading: #000;
    --pst-color-text-base: #1a1a1a;
    --pst-color-text-muted: #666;
    --pst-color-surface: #f7f7f7;
    --pst-color-on-surface: #333;
    --pst-color-primary: var(--nv-color-green-2);
    --pst-color-table-row-hover-bg: var(--nv-color-green);
    --pst-color-link: var(--pst-color-text-base);
    --pst-color-link-hover: var(--pst-color-text-base);
    --pst-color-inline-code: var(--pst-color-primary);
    --pst-color-inline-code-links: var(--pst-color-primary);
    --pst-color-secondary: var(--pst-color-primary);
    --pst-color-secondary-bg: var(--nv-color-green);
    --pst-color-accent: var(--nv-color-green);
}

/* Dark theme */
html[data-theme=dark],
html.dark,
.dark {
    --nv-color-text: #eeeeee;
    --nv-color-bg-default: #111111;
    --nv-color-bg-alt: #1a1a1a;
    --pst-color-background: #111;
    --pst-color-on-background: #000;
    --pst-color-shadow: #000;
    --pst-color-heading: #fff;
    --pst-color-text-base: #eee;
    --pst-color-text-muted: #999;
    --pst-color-surface: #1a1a1a;
    --pst-color-on-surface: #ddd;
    --pst-color-primary: var(--nv-color-green);
    --pst-color-table-row-hover-bg: var(--nv-color-green-2);
    --pst-color-link: var(--pst-color-text-base);
    --pst-color-link-hover: var(--pst-color-text-base);
    --pst-color-inline-code: var(--pst-color-primary);
    --pst-color-inline-code-links: var(--pst-color-primary);
    --pst-color-secondary: var(--pst-color-primary);
    --pst-color-secondary-bg: var(--nv-color-green-2);
    --pst-color-accent: var(--nv-color-green);
}

/* Version selector styling */

/* Fern simulates emphasis on active navigation labels with a text stroke.
   NVIDIA Sans renders that stroke as a doubled glyph, so use font weight instead. */
#fern-header [data-radix-collection-item][data-state="active"] > span,
.fern-sidebar-link[data-state="active"] .fern-sidebar-link-title-inner {
    -webkit-text-stroke: 0 !important;
    font-weight: 600;
}

.fern-version-selector {
    margin-inline-start: 0.5rem;
    transform: none;
}

.fern-version-selector .version-dropdown-trigger {
    outline: 1px solid var(--border, var(--grayscale-a5)) !important;
    border-radius: 5px;
    transition: box-shadow 0.3s ease, outline 0.3s ease;
}

.version-dropdown-trigger {
    background-color: transparent !important;
}

.version-dropdown-trigger:hover {
    box-shadow: 0 0 0 1px var(--nv-color-green) !important;
}

/* Sidebar styling */
#fern-sidebar {
    border-right: 1px solid var(--border, var(--grayscale-a5)) !important;
}
.fern-sidebar-link:not(:hover) {
    background-color: transparent !important;
}
.fern-sidebar-link:hover {
    background-color: rgba(118, 185, 0, 0.10) !important;
}
.dark .fern-sidebar-link:hover {
    background-color: rgba(118, 185, 0, 0.16) !important;
}
.fern-sidebar-link {
    position: relative;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
    border-radius: 0px !important;
    &.nested {
        padding-left: 1rem !important;
    }
}
/* Section-level sidebar links (pages that have children) should match sidebar heading padding */
.fern-sidebar-group > li > .fern-sidebar-link:has(+ .fern-sidebar-group) {
    padding-left: 0.25rem !important;
}
/* User Guide variants use collapsible top-level sections. Keep those section
   titles visually distinct without changing flat entries in Reference. */
#fern-sidebar:has(.fern-variant-selector)
    .fern-sidebar-link.fern-sidebar-level-1:has(.expand-indicator)
    .fern-sidebar-link-title {
    font-weight: 600 !important;
}
.fern-sidebar-group{
    padding: 0 !important
}
#fern-sidebar-scroll-area{
    padding-right: 0 !important
}

/* header styling */
.fern-header-content{
    padding-left: 18.5px;
    margin-top: -5px;
    margin-bottom: -5px;
}
#fern-header {
    border-color: var(--border, var(--grayscale-a5)) !important;
}
@keyframes header-background-fade {
    0% {
      background-color: transparent;
    }
    100% {
      background-color: var(--header-background);
    }
  }

[data-theme=default]#fern-header {
animation: header-background-fade linear;
animation-timeline: scroll();
animation-range: 0 50px;
}
.fern-header-navbar-links .fern-button{
    background-color: transparent !important;
}
.fern-header-navbar-links > button{
    background-color: transparent !important;
}
.fern-header-logo-container > div > div > a > img {
    padding-right: 0;
}

/* Echo the hero wordmark with a compact semibold product name, without the
   divider bars used by the previous header treatment. */
.fern-header-logo-container .font-heading {
    margin: 0 0 0 0.75rem !important;
    padding: 0 !important;
    border: 0 !important;
    color: var(--grayscale-a12) !important;
    font-size: 20px !important;
    font-weight: 600 !important;
    line-height: 1 !important;
    letter-spacing: -0.035em;
}
@media (max-width: 1024px) {
    .fern-header-logo-container .font-heading{
        display: none !important;
    }
}
/* Search bar styling */
#fern-search-button{
    background-color: transparent !important;
    border-radius: var(--rounded);
    transition: box-shadow 0.3s ease, outline 0.3s ease;
}
#fern-search-button:hover{
    box-shadow: 0 0 0 1px var(--nv-color-green) !important;
}
#fern-search-button .fern-kbd{
    display: none;
}

.fern-layout-footer-toolbar button{
    background-color: transparent !important;
    border-color: transparent !important;
    padding-inline: 0px !important;
}

/* ========== Custom footer (native React component) – 1:1 with original ========== */
.bd-footer {
  border-top: 1px solid var(--border, var(--grayscale-a5)) !important;
  font-family: NVIDIA, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
  font-size: 0.875rem;
  padding: 2rem 0;
  width: 100%;
}
.bd-footer * {
  font-family: inherit;
}
.bd-footer__inner {
  padding: 0 2rem;
}
.footer-items__start {
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
}
.footer-logos-container {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
  gap: 1rem;
}
.footer-brand {
  display: inline-block;
  text-decoration: none;
}
.footer-brand .logo__image {
  height: 24px;
  width: auto;
  transition: opacity 0.2s ease;
}
.footer-brand:hover .logo__image {
  opacity: 0.8;
}
.footer-brand-fern {
  display: flex;
  align-items: center;
  margin-left: auto;
}
/* Logo theme visibility – .dark is on ancestor in Fern */
.only-light {
  display: block;
  filter: invert(1);
}
.only-dark {
  display: none;
}
.dark .only-light {
  display: none;
}
.dark .only-dark {
  display: block;
  filter: none;
}
.footer-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 0.5rem;
  line-height: 1.65;
  margin: 0;
  padding: 0;
}
.footer-links a {
  color: var(--grayscale-a11);
  text-decoration: none;
  transition: color 0.2s ease;
  white-space: nowrap;
}
.pipe-separator {
  color: var(--grayscale-a11);
  white-space: nowrap;
}
.copyright {
  color: var(--grayscale-a11);
  font-size: 0.875rem;
  line-height: 1.65;
  margin: 0;
}
@media (max-width: 768px) {
  .bd-footer { padding: 1.5rem 0; }
  .bd-footer__inner { padding: 0 1.5rem; }
  .footer-items__start { gap: 1rem; }
  .footer-links { flex-direction: row; gap: 0.5rem 0.75rem; }
  .footer-links a { white-space: normal; word-break: break-word; }
}
@media (max-width: 480px) {
  .footer-links { gap: 0.5rem; }
  .footer-links a { font-size: 0.8125rem; }
  .copyright { font-size: 0.8125rem; }
}
/* Built with Fern link + tooltip */
.built-with-fern-link {
  display: flex;
  align-items: baseline;
  gap: 0.25rem;
  text-decoration: none;
  position: relative;
}
.built-with-fern-logo {
  height: 1rem;
  margin: 0;
  transition: filter 150ms ease;
}
.built-with-fern-logo path { fill: var(--grayscale-a12); }
.built-with-fern-link:hover .built-with-fern-logo { filter: saturate(1) opacity(1); }
.built-with-fern-link:hover .built-with-fern-logo path:nth-child(2) { fill: #51C233; }
.built-with-fern-tooltip {
  position: absolute;
  top: 50%;
  right: calc(100%);
  bottom: auto;
  left: auto;
  transform: translateY(-50%);
  margin: 0;
  margin-right: 0.5rem;
  padding: 0.5rem 0.75rem;
  background-color: #FFFFFF;
  color: #000000;
  font-size: 0.85rem;
  border-radius: 0.375rem;
  border: 1px solid var(--grayscale-a5);
  white-space: nowrap;
  pointer-events: none;
  opacity: 0;
  transition: opacity 150ms ease;
  transition-delay: 0s;
  z-index: 50;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  width: max-content;
}
.built-with-fern-link:hover .built-with-fern-tooltip {
  opacity: 1;
  transition-delay: 0.75s;
}
.dark .built-with-fern-tooltip {
  background-color: #000000;
  color: #FFFFFF;
}
.built-with-fern-logo-dark { display: none; }
.dark .built-with-fern-logo-light { display: none; }
.dark .built-with-fern-logo-dark { display: block; }
@media (prefers-color-scheme: dark) {
  .built-with-fern-logo-light { display: none; }
  .built-with-fern-logo-dark { display: block; }
}

/* Footer styling */
.fern-footer-nav{
    border-radius: var(--rounded);
    background-color: transparent !important;
    transition: box-shadow 0.3s ease, outline 0.3s ease;
}
/* Figure caption callouts */
.fig-caption {
    --callout-border: transparent !important;
    --callout-icon-color: var(--grayscale-a8) !important;
    background-color: var(--nv-light-grey-1) !important;
    font-size: 0.875rem;
}
.dark .fig-caption {
    background-color: var(--nv-dark-grey-2) !important;
}
.fig-caption p {
    font-style: italic;
    color: var(--tw-prose-body) !important;
}

/* Hide line numbers */
.code-block-line-gutter {
    display: none !important;
}
.fern-footer-prev h4, .fern-footer-next h4{
    font-size: inherit !important;
}
/* The active page uses a persistent NVIDIA-green bar and the same subtle
   green gradient treatment as active entries in the Blog navigation. */
.fern-sidebar-link[data-state="active"]::before {
    content: "";
    position: absolute;
    left: 0 !important;
    bottom: 0 !important;
    top: 0 !important;
    width: 3px !important;
    background-color: var(--nv-color-green) !important;
}
.fern-sidebar-link[data-state="active"] {
    color: unset !important;
    background: linear-gradient(
        90deg,
        rgba(118, 185, 0, 0.15),
        rgba(118, 185, 0, 0.04)
    ) !important;
}
.dark .fern-sidebar-link[data-state="active"] {
    background: linear-gradient(
        90deg,
        rgba(118, 185, 0, 0.20),
        rgba(118, 185, 0, 0.06)
    ) !important;
}

.fern-selection-item .fern-selection-item-icon{
    border-color: transparent !important;
}
/* Button styling */
.fern-button{
    border-radius: var(--rounded);
    font-weight: bold;
}
.fern-button.filled.primary{
    color: var(--nv-color-black);
}
.dark .fern-button.filled.primary{
    background-color: var(--nv-color-white);
}
.dark .fern-button.filled.primary:hover{
    background-color: var(--nv-light-grey-2);
}
.fern-button.outlined.normal{
    background-color: transparent;
    --tw-ring-color: transparent;
    color: var(--nv-color-black);
}
.fern-button.outlined.normal:hover{
    color: var(--nv-color-green)
}
.dark .fern-button.outlined.normal{
    color: var(--nv-color-white);
}
.dark .fern-button.outlined.normal:hover{
    color: var(--nv-color-green);
}
/* Fern content tabs: use the same neutral, outlined card treatment as the
   Install Selector, Compatibility, and Recipes widgets. The tab strip already
   provides the card header, so individual <Tabs> blocks do not need an extra
   title or eyebrow. Stable direction/orientation attributes keep this independent
   of Fern utility-class changes while the .fern-prose scope excludes
   site-navigation tabs. */
.fern-prose div[dir="ltr"][data-orientation="horizontal"] {
    margin: 1.5rem 0;
    overflow: hidden;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 12px;
    background: var(--nv-color-bg-default);
}

.fern-prose div[dir="ltr"][data-orientation="horizontal"]
    > div:first-child {
    margin: 0 !important;
    padding: 0 18px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    background: var(--pst-color-surface);
}

.fern-prose div[dir="ltr"][data-orientation="horizontal"]
    > div:first-child > div {
    gap: 1.25rem;
}

.fern-prose div[dir="ltr"][data-orientation="horizontal"]
    [role="tab"] {
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
}

.fern-prose div[dir="ltr"][data-orientation="horizontal"]
    > [role="tabpanel"] {
    padding: 18px;
}

.fern-prose div[dir="ltr"][data-orientation="horizontal"]
    > [role="tabpanel"]::before {
    display: none;
}

.fern-prose div[dir="ltr"][data-orientation="horizontal"]
    > [role="tabpanel"] > :first-child {
    margin-top: 0 !important;
}

.fern-prose div[dir="ltr"][data-orientation="horizontal"]
    > [role="tabpanel"] > :last-child {
    margin-bottom: 0 !important;
}

@media (max-width: 640px) {
    .fern-prose div[dir="ltr"][data-orientation="horizontal"]
        > div:first-child {
        padding-inline: 14px;
    }

    .fern-prose div[dir="ltr"][data-orientation="horizontal"]
        > [role="tabpanel"] {
        padding: 14px;
    }
}

/* Card styling */
.fern-card{
    transition: box-shadow 0.3s ease, outline 0.3s ease;
}
svg.card-icon{
    height: 24px !important;
    width: 24px !important;
}
.card-icon{
    background-color: transparent !important;
}
/* Only linked cards (rendered as <a>) get the hover highlight. Cards without an
   href render as <div class="fern-card"> — e.g. the "Why Dynamo" cards — and must
   not look clickable. */
a.fern-card:hover{
    box-shadow: 0 0 0 1px var(--nv-color-green) !important;
}
.fern-docs-badge{
    border-radius: var(--rounded);
}
/* Enum "Allowed values" chips: match Fern's Schema renderer, which builds each
   chip as <Badge size="sm"> with the default subtle variant. We can't invoke
   that renderer (it needs an OpenAPI-backed <Schema>), so we restyle the MDX
   <Badge> to the same tokens Fern uses: label in shrink-0 text-sm grayscale-a11,
   chips as .subtle.small (grayscale-a3 fill, grayscale-a11 text, radius-1). The
   forced background/color neutralize whatever intent the MDX <Badge> requires.
   Scoped to .enum-values so the pill <Badge> used elsewhere is unaffected. */
.enum-values {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem;
}
.enum-values .enum-label {
    flex-shrink: 0;
    color: var(--grayscale-a11);
    font-size: var(--text-sm);
}
.enum-values .fern-docs-badge {
    border-radius: var(--radius-1) !important;
    height: 1.25rem;
    padding: 0 0.375rem;
    font-size: var(--text-xs);
    font-weight: 500;
    background-color: var(--grayscale-a3) !important;
    color: var(--grayscale-a11) !important;
    text-transform: none;
}

/* ============================================================
   Compatibility page: category chips for the Requirement table.
   Splits comma-separated OS / Arch / GPU / CUDA values into
   color-coded chips so supported platforms read at a glance.
   Colored by category (distro brand, CPU arch, GPU generation,
   CUDA toolkit); arch is always blue across rows so x86_64 /
   ARM64 stay visually consistent wherever they appear. Tints
   are rgba so they hold up in both light and dark themes.
   Scoped to .dynamo-chip; nothing else is affected.
   ============================================================ */
.dynamo-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.3em;
    margin: 0.15rem 0.1rem;
    min-height: 24px;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.8125rem;
    font-weight: 600;
    line-height: 1.2;
    white-space: nowrap;
    border: 1px solid transparent;
    vertical-align: middle;
}

/* Ubuntu — brand orange */
.dynamo-chip-ubuntu {
    background-color: rgba(233, 84, 32, 0.12);
    color: #C7401C;
    border-color: rgba(233, 84, 32, 0.3);
}
.dark .dynamo-chip-ubuntu {
    background-color: rgba(233, 84, 32, 0.2);
    color: #FF9068;
    border-color: rgba(233, 84, 32, 0.42);
}

/* CentOS — violet */
.dynamo-chip-centos {
    background-color: rgba(124, 58, 237, 0.12);
    color: #6D28D9;
    border-color: rgba(124, 58, 237, 0.3);
}
.dark .dynamo-chip-centos {
    background-color: rgba(139, 92, 246, 0.2);
    color: #C4B5FD;
    border-color: rgba(139, 92, 246, 0.42);
}

/* CPU architecture — blue (consistent across every row) */
.dynamo-chip-arch {
    background-color: rgba(37, 99, 235, 0.1);
    color: #1D4ED8;
    border-color: rgba(37, 99, 235, 0.28);
}
.dark .dynamo-chip-arch {
    background-color: rgba(59, 130, 246, 0.18);
    color: #93C5FD;
    border-color: rgba(59, 130, 246, 0.42);
}

/* GPU generation — NVIDIA green */
.dynamo-chip-gpu {
    background-color: rgba(118, 185, 0, 0.16);
    color: #4D7C0F;
    border-color: rgba(118, 185, 0, 0.38);
}
.dark .dynamo-chip-gpu {
    background-color: rgba(118, 185, 0, 0.22);
    color: #A3E635;
    border-color: rgba(118, 185, 0, 0.46);
}

/* CUDA toolkit — teal */
.dynamo-chip-cuda {
    background-color: rgba(8, 145, 178, 0.12);
    color: #0E7490;
    border-color: rgba(8, 145, 178, 0.3);
}
.dark .dynamo-chip-cuda {
    background-color: rgba(34, 211, 238, 0.16);
    color: #67E8F9;
    border-color: rgba(34, 211, 238, 0.4);
}

/* Caveat qualifier (e.g. "experimental") — amber */
.dynamo-chip-note {
    background-color: rgba(217, 119, 6, 0.12);
    color: #B45309;
    border-color: rgba(217, 119, 6, 0.3);
    font-weight: 500;
}
.dark .dynamo-chip-note {
    background-color: rgba(245, 158, 11, 0.16);
    color: #FCD34D;
    border-color: rgba(245, 158, 11, 0.4);
}

/* Neutral qualifier — subtle grey */
.dynamo-chip-neutral {
    background-color: var(--grayscale-a3);
    color: var(--grayscale-a11);
    border-color: var(--grayscale-a3);
    font-weight: 500;
}

/* ============================================================
   Release Artifacts page: the Tags column renders each version
   as a click-to-copy Badge (Fern's <Copy> wraps a <Badge> so the
   click copies the full image reference). <Copy> paints its own
   filled, padded chrome around the child, which shows as a grey
   box flanking the badge. Strip that chrome inside .dynamo-tag-copy
   so only the Badge shows; the wrapper stays clickable, so the
   copy behavior is unchanged. Scoped to .dynamo-tag-copy so the
   Icon-based <Copy> buttons in the Install columns keep their
   button chrome.
   ============================================================ */
.dynamo-tag-copy {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.375rem;
}
/* Reset fill/padding/border/shadow on every wrapper the <Copy>
   renders — matched depth-independently (any descendant that is
   not the Badge and not inside it) so the grey box is gone no
   matter how Fern nests the copy affordance. The Badge and its
   own children are excluded and keep their styling. */
.dynamo-tag-copy *:not(.fern-docs-badge):not(.fern-docs-badge *) {
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.dynamo-tag-copy .fern-docs-badge {
    cursor: pointer;
}

/* ============================================================
   Release Artifacts page: the Install column renders a
   click-to-copy button as <Copy><Icon icon="copy"/></Copy>.
   Fern's default chrome is wider than it is tall, so the icon
   looks cramped. Give the button uniform padding and a square
   footprint so the icon sits centered — the compact copy-button
   shape. Scoped to .dynamo-install-copy so the Tags-column tag
   copies (.dynamo-tag-copy) are untouched.
   ============================================================ */
.dynamo-install-copy {
    display: inline-flex;
}
/* Shape the button the <Copy> renders. Matched depth-independently
   (the interactive element plus any wrapper it nests in) so the
   uniform padding lands regardless of how Fern structures the DOM. */
.dynamo-install-copy button,
.dynamo-install-copy [role="button"],
.dynamo-install-copy > * {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0.375rem !important;
    aspect-ratio: 1 / 1;
    border-radius: 6px !important;
}

.fern-page-actions button:hover{
    background-color: transparent !important;
}
.fern-page-actions a:hover{
    background-color: transparent !important;
}
/* Moving logo to footer */
#builtwithfern, #builtwithfern * {
    display: none !important;
}

/* Landing Page Gradients */
/* Top: Simple radial gradient (no mask, responsive) */
.landing-gradient-top {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 800px;
    background: radial-gradient(ellipse 100% 100% at 50% 10%,
        rgba(191, 242, 48, 0.15) 0%,
        rgba(158, 228, 179, 0.12) 30%,
        rgba(124, 215, 254, 0.12) 50%,
        rgba(124, 215, 254, 0.06) 75%,
        transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* Bottom: Gradient for organic transition */
.landing-gradient-bottom {
    position: absolute;
    bottom: -282px;
    left: 0;
    right: 0;
    height: 1232px;
    background: linear-gradient(85deg, #BFF230 41.98%, #7CD7FE 99.52%);
    opacity: 0.05;
    pointer-events: none;
    z-index: 5;
}

/* Landing Page Gradients Wrapper */
.landing-page-gradients {
    position: relative;
    width: 100%;
    margin-top: -100px;
    padding-top: 100px;
    overflow: visible;
    background: #181818;
}

/* Hero Section (Landing page only) */
.hero-section {
    position: relative;
    width: 100%;
    padding: 3rem 6rem;
    margin: 0 auto;
    overflow: visible;
    display: flex;
    flex-direction: column;
    align-items: center;
    z-index: 10;
}

/* Hero Section Content - constrain width */
.hero-section > * {
    position: relative;
    z-index: 100;
    max-width: 1440px;
    width: 100%;
}

/* Tablet and Mobile: fix spacing and layout */
@media (max-width: 1024px) {
    /* Extend dark background behind header */
    .landing-page body, .landing-page html, .landing-page main {
        background: #181818 !important;
    }

    .landing-page-gradients {
        margin-top: -100px;
        padding-top: 100px;
    }

    .hero-section {
        padding: 2rem 2rem;
    }

    .hero-section > * {
        max-width: none;
    }

    .hero-content-grid {
        grid-template-columns: 1fr;
        gap: 2rem;
    }

    .hero-heading {
        font-size: 36px;
    }

    .hero-subtitle {
        font-size: 16px;
    }

    .hero-title-section {
        margin-bottom: 2rem;
    }
}

/* Small mobile only */
@media (max-width: 600px) {
    .hero-heading {
        font-size: 28px;
    }

    .hero-section {
        padding: 1.5rem 1.5rem;
    }
}

.hero-section h1,
.hero-section h2,
.hero-section h3,
.hero-section h4,
.hero-section h5,
.hero-section h6 {
    pointer-events: none !important;
}
/* Hero Title Section */
.hero-title-section {
    text-align: center;
    margin-bottom: 4rem;
    position: relative;
    z-index: 100;
}

.hero-heading {
    font-size: 48px;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 1rem 0;
    color: var(--nv-color-white);
}

.hero-subtitle {
    font-size: 18px;
    line-height: 1.5;
    margin: 0;
    color: var(--nv-color-white);
}

/* Hero Content Grid */
.hero-content-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 3rem;
    align-items: start;
    position: relative;
    z-index: 100;
}

.hero-column {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

.hero-column-title {
    font-size: 24px;
    font-weight: 700;
    margin: 0;
    color: var(--nv-color-white);
}

.hero-column-subtitle {
    font-size: 16px;
    margin: 0 0 1rem 0;
    color: var(--nv-color-white);
}

/* Hero Card Container (Left Column) */
.hero-card-container {
    display: flex;
    flex-direction: column;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border, var(--grayscale-a5));
    margin-top: 1.5rem !important;
    background: rgba(26, 26, 26, 0.2);
    backdrop-filter: blur(6px);
}

.hero-card-image {
    width: 100%;
    height: auto;
    display: block;
}

.hero-card-content {
    padding: 1.5rem;
    display: flex;
    flex-direction: row;
    gap: 1rem;
    align-items: center;
    justify-content: space-between;
    background: rgba(26, 26, 26, 0.2);
    backdrop-filter: blur(6px);
}

.hero-card-text-wrapper {
    flex: 1;
}

.hero-card-text {
    margin: 0;
    font-size: 14px;
    line-height: 1.5;
    color: var(--nv-color-white);
}

.hero-card-button-wrapper {
    flex-shrink: 0;
}
.hero-card-button-wrapper .fern-mdx-link{
    text-decoration: none !important;
}

.hero-card-button {
    white-space: nowrap;
}

/* Hero Cards */

.hero-column .fern-card {
    padding: 9px 17px;
    background-color: rgba(26, 26, 26, 0.2) !important;
    backdrop-filter: blur(6px);
}

.hero-section .fern-card{
    color: white !important;
}

.hero-column .card-icon {
    font-size: 64px !important;
    width: 64px !important;
    height: 64px !important;
}

.hero-column .card-icon svg,
.hero-column .card-icon i {
    font-size: 64px !important;
    width: 64px !important;
    height: 64px !important;
}

.hero-column .fern-card-title {
    font-size: 16px;
    font-weight: 500;
    line-height: 24px;
}

.hero-column .fern-card p {
    font-size: 14px;
    line-height: 20px;
    color: white !important;
}

/* Body Section */
.body-section {
    display: flex;
    padding: 4rem 16rem;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    gap: 4rem;
    align-self: stretch;
    position: relative;
    z-index: 1;
    background: #181818;
}

/* Body Section Content - constrain width */
.body-section > * {
    max-width: 1440px;
    width: 100%;
    position: relative;
    z-index: 10;
}

.code-block .fern-code-link{
    text-decoration: underline !important;
    text-decoration-color: var(--accent) !important;
    text-underline-offset: 1px !important;
    text-decoration-style: underline !important;
}

/* Rounded Mermaid canvas aligned with the Dynamo story-stage visual language. */
.mermaid-container {
  --mermaid-label-backdrop: #f1f5e9;
  position: relative;
  display: grid;
  min-height: 180px;
  place-items: center;
  margin: 1.75rem 0 2rem;
  padding: clamp(2rem, 5vw, 3.5rem) clamp(1.25rem, 4vw, 3rem);
  overflow: hidden;
  border: 1px solid rgba(118, 185, 0, 0.25);
  border-radius: 24px;
  background:
    radial-gradient(circle at 16% 8%, rgba(118, 185, 0, 0.2), transparent 36%),
    linear-gradient(145deg, #fbfcf8 0%, #f1f5e9 52%, #e8eee2 100%);
  box-shadow: 0 18px 48px rgba(23, 42, 0, 0.1);
}

.dark .mermaid-container {
  --mermaid-label-backdrop: #0d1209;
  border-color: rgba(118, 185, 0, 0.3);
  background:
    radial-gradient(circle at 16% 8%, rgba(118, 185, 0, 0.18), transparent 36%),
    linear-gradient(145deg, #12170e 0%, #0d1209 55%, #070906 100%);
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.3);
}

.mermaid-container-expanded {
  --mermaid-label-backdrop: #f1f5e9;
  padding: clamp(1.5rem, 4vw, 3rem);
  border: 1px solid rgba(118, 185, 0, 0.25);
  border-radius: 24px;
  background:
    radial-gradient(circle at 16% 8%, rgba(118, 185, 0, 0.18), transparent 36%),
    linear-gradient(145deg, #fbfcf8 0%, #f1f5e9 52%, #e8eee2 100%);
}

.dark .mermaid-container-expanded {
  --mermaid-label-backdrop: #0d1209;
  border-color: rgba(118, 185, 0, 0.3);
  background:
    radial-gradient(circle at 16% 8%, rgba(118, 185, 0, 0.16), transparent 36%),
    linear-gradient(145deg, #12170e 0%, #0d1209 55%, #070906 100%);
}

/* Lightweight NVIDIA-green overrides for Mermaid's default blue-gray theme. */
.mermaid-container svg .cluster > rect:not([style*="fill"]),
.mermaid-container svg .cluster > rect[style*="#e3f2fd" i],
.mermaid-container svg .cluster > rect[style*="#2196f3" i],
.mermaid-container-expanded svg .cluster > rect:not([style*="fill"]),
.mermaid-container-expanded svg .cluster > rect[style*="#e3f2fd" i],
.mermaid-container-expanded svg .cluster > rect[style*="#2196f3" i] {
  fill: rgba(255, 255, 255, 0.5) !important;
  stroke: rgba(118, 185, 0, 0.72) !important;
  rx: 14px;
  ry: 14px;
}

.mermaid-container
  svg
  .node
  > :is(rect.label-container, polygon.label-container, path.label-container, circle.label-container, ellipse.label-container):not(
    [style*="fill"]
  ),
.mermaid-container-expanded
  svg
  .node
  > :is(rect.label-container, polygon.label-container, path.label-container, circle.label-container, ellipse.label-container):not(
    [style*="fill"]
  ) {
  fill: #fcfdf9 !important;
  stroke: #76b900 !important;
}

.mermaid-container svg .node > rect.label-container:not([style*="fill"]),
.mermaid-container-expanded
  svg
  .node
  > rect.label-container:not([style*="fill"]) {
  rx: 8px;
  ry: 8px;
}

/* Flowcharts. */
.mermaid-container svg :is(.flowchart-link, .edgePath .path),
.mermaid-container-expanded svg :is(.flowchart-link, .edgePath .path) {
  stroke: #76b900 !important;
}

/* Sequence diagrams use actor boxes instead of flowchart nodes. */
.mermaid-container svg rect.actor,
.mermaid-container-expanded svg rect.actor {
  fill: #fcfdf9 !important;
  stroke: #76b900 !important;
  rx: 8px;
  ry: 8px;
}

.mermaid-container svg :is(.actor-line, .messageLine0, .messageLine1),
.mermaid-container-expanded svg :is(.actor-line, .messageLine0, .messageLine1) {
  stroke: #76b900 !important;
}

/* Keep sequence labels legible when self-message arrows run behind them. */
.mermaid-container svg .messageText,
.mermaid-container-expanded svg .messageText {
  stroke: var(--mermaid-label-backdrop) !important;
  stroke-width: 8px;
  stroke-linecap: round;
  stroke-linejoin: round;
  paint-order: stroke fill;
}

.mermaid-container svg :is(.marker, marker path, marker polygon),
.mermaid-container-expanded svg :is(.marker, marker path, marker polygon) {
  fill: #76b900 !important;
  stroke: #76b900 !important;
}

.dark .mermaid-container svg .cluster > rect:not([style*="fill"]),
.dark .mermaid-container svg .cluster > rect[style*="#e3f2fd" i],
.dark .mermaid-container svg .cluster > rect[style*="#2196f3" i],
.dark .mermaid-container-expanded svg .cluster > rect:not([style*="fill"]),
.dark .mermaid-container-expanded svg .cluster > rect[style*="#e3f2fd" i],
.dark .mermaid-container-expanded svg .cluster > rect[style*="#2196f3" i] {
  fill: rgba(118, 185, 0, 0.06) !important;
  stroke: rgba(118, 185, 0, 0.72) !important;
}

.dark
  .mermaid-container
  svg
  .node
  > :is(rect.label-container, polygon.label-container, path.label-container, circle.label-container, ellipse.label-container):not(
    [style*="fill"]
  ),
.dark
  .mermaid-container-expanded
  svg
  .node
  > :is(rect.label-container, polygon.label-container, path.label-container, circle.label-container, ellipse.label-container):not(
    [style*="fill"]
  ) {
  fill: #141c0e !important;
  stroke: #76b900 !important;
}

.dark .mermaid-container svg :is(.flowchart-link, .edgePath .path),
.dark .mermaid-container-expanded svg :is(.flowchart-link, .edgePath .path) {
  stroke: #76b900 !important;
}

.dark .mermaid-container svg rect.actor,
.dark .mermaid-container-expanded svg rect.actor {
  fill: #141c0e !important;
  stroke: #76b900 !important;
}

.dark .mermaid-container svg :is(.actor-line, .messageLine0, .messageLine1),
.dark
  .mermaid-container-expanded
  svg
  :is(.actor-line, .messageLine0, .messageLine1) {
  stroke: #76b900 !important;
}

.dark .mermaid-container svg :is(.marker, marker path, marker polygon),
.dark .mermaid-container-expanded svg :is(.marker, marker path, marker polygon) {
  fill: #76b900 !important;
  stroke: #76b900 !important;
}


/* ===================== Community page ===================== */

:root {
  --dynamo-community-green: #76b900;
  --dynamo-community-green-bright: #8ed600;
  --dynamo-community-ink: var(--grayscale-a12);
  --dynamo-community-muted: var(--grayscale-a10);
  --dynamo-community-rule: color-mix(in srgb, var(--grayscale-a12) 14%, transparent);
  --dynamo-community-soft: #f3f4f3;
  --dynamo-community-titlebar-text: #555755;
}
`;
// sync-site-css:end

export default function CustomFooter() {
  const currentYear = new Date().getFullYear();
  const logoUrl =
    "https://fern-image-hosting.s3.us-east-1.amazonaws.com/nvidia/NVIDIA_Logo_0.svg";

  return (
    <>
    <style dangerouslySetInnerHTML={{ __html: SITE_CSS }} />
    <footer className="bd-footer">
      <div className="bd-footer__inner">
        <div className="footer-items__start">
          <div className="footer-item">
            <div className="footer-logos-container">
              <a
                className="footer-brand"
                href="https://www.nvidia.com"
                target="_blank"
                rel="noopener"
              >
                <img src={logoUrl} className="logo__image only-light" alt="NVIDIA" />
                <img src={logoUrl} className="logo__image only-dark" alt="NVIDIA" />
              </a>
              <div className="footer-brand-fern">
                <a
                  href="https://buildwithfern.com"
                  className="built-with-fern-link"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <svg
                    width="145"
                    height="16"
                    viewBox="0 0 145 16"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="built-with-fern-logo built-with-fern-logo-light"
                    aria-hidden
                  >
                    <path d="M9.79656 4.8H14.5006C15.5139 4.8 16.3192 5.05067 16.9166 5.552C17.5139 6.04267 17.8126 6.71467 17.8126 7.568C17.8126 8.112 17.6739 8.608 17.3966 9.056C17.1192 9.504 16.7512 9.84 16.2926 10.064C16.8579 10.2667 17.3059 10.608 17.6366 11.088C17.9672 11.5573 18.1326 12.1173 18.1326 12.768C18.1326 13.7387 17.8286 14.5227 17.2206 15.12C16.6126 15.7067 15.7752 16 14.7086 16H9.79656V4.8ZM14.4846 14.528C15.1246 14.528 15.6206 14.3627 15.9726 14.032C16.3246 13.7013 16.5006 13.2373 16.5006 12.64C16.5006 12.0427 16.3246 11.5893 15.9726 11.28C15.6312 10.96 15.1352 10.8 14.4846 10.8H11.3966V14.528H14.4846ZM14.2766 9.424C14.8846 9.424 15.3539 9.28533 15.6846 9.008C16.0152 8.72 16.1806 8.32533 16.1806 7.824C16.1806 7.32267 16.0152 6.93867 15.6846 6.672C15.3539 6.40533 14.8846 6.272 14.2766 6.272H11.3966V9.424H14.2766ZM22.5778 16.224C21.6285 16.224 20.8871 15.9413 20.3538 15.376C19.8205 14.8107 19.5538 14 19.5538 12.944V8.304H21.1058V12.8C21.1058 13.472 21.2551 13.9787 21.5538 14.32C21.8631 14.6507 22.3005 14.816 22.8658 14.816C23.4525 14.816 23.9165 14.6293 24.2578 14.256C24.6098 13.872 24.7858 13.3707 24.7858 12.752V8.304H26.3378V16H24.9618V15.12C24.7165 15.4827 24.3858 15.76 23.9698 15.952C23.5538 16.1333 23.0898 16.224 22.5778 16.224ZM28.0746 8.304H29.6266V16H28.0746V8.304ZM27.9786 4.912H29.7066V6.752H27.9786V4.912ZM33.0334 16C32.4894 16 32.0948 15.888 31.8494 15.664C31.6041 15.44 31.4814 15.0667 31.4814 14.544V4.8H33.0334V14.064C33.0334 14.2667 33.0761 14.416 33.1614 14.512C33.2468 14.5973 33.3854 14.64 33.5774 14.64H34.5534V16H33.0334ZM37.9539 16C37.2819 16 36.7966 15.856 36.4979 15.568C36.1993 15.28 36.0499 14.8053 36.0499 14.144V9.664H34.0339V8.304H36.0499V6H37.6019V8.304H40.0179V9.664H37.6019V13.84C37.6019 14.1173 37.6659 14.32 37.7939 14.448C37.9219 14.576 38.1299 14.64 38.4179 14.64H40.0179V16H37.9539ZM43.5709 8.304H45.1869L46.8989 14.272L48.6109 8.304H50.3869L52.0989 14.272L53.8109 8.304H55.4269L53.0429 16H51.2189L49.5069 10.064L47.7789 16H45.9549L43.5709 8.304ZM56.3746 8.304H57.9266V16H56.3746V8.304ZM56.2786 4.912H58.0066V6.752H56.2786V4.912ZM62.5971 16C61.9251 16 61.4397 15.856 61.1411 15.568C60.8424 15.28 60.6931 14.8053 60.6931 14.144V9.664H58.6771V8.304H60.6931V6H62.2451V8.304H64.6611V9.664H62.2451V13.84C62.2451 14.1173 62.3091 14.32 62.4371 14.448C62.5651 14.576 62.7731 14.64 63.0611 14.64H64.6611V16H62.5971ZM65.6727 4.8H67.2247V9.056C67.4807 8.736 67.8007 8.496 68.1847 8.336C68.5794 8.16533 69.0114 8.08 69.4807 8.08C70.4407 8.08 71.1927 8.368 71.7367 8.944C72.2807 9.50933 72.5527 10.3147 72.5527 11.36V16H71.0007V11.504C71.0007 10.832 70.8407 10.3307 70.5207 10C70.2114 9.65867 69.7687 9.488 69.1927 9.488C68.5954 9.488 68.1154 9.68 67.7527 10.064C67.4007 10.4373 67.2247 10.9333 67.2247 11.552V16H65.6727V4.8Z" fill="#1E1F24" />
                    <path d="M92.3849 7.82856C91.3321 6.93847 89.746 6.58166 88.3403 7.62074C88.2756 7.66779 88.1952 7.58741 88.2442 7.52468C88.5775 7.09532 88.9638 6.63263 89.2755 6.16798C89.5931 5.69157 90.0675 5.35044 90.6145 5.18379C93.5259 4.30155 92.6515 0.00012207 92.6515 0.00012207C92.6515 0.00012207 88.154 0.290282 88.7089 4.17019C88.801 4.81913 88.6285 5.47983 88.2227 5.99545C87.7247 6.62479 87.1463 7.22667 86.7268 7.66191C86.6385 7.7521 86.4895 7.66583 86.5248 7.54428C86.9307 6.17778 87.2267 4.06432 85.821 2.70175L83.8428 1.05881L83.4625 1.56071C82.3312 3.05268 82.6626 5.15634 84.1565 6.28561C85.0132 6.93259 85.4014 7.63643 85.3407 8.40888C85.3034 8.87157 85.0936 9.30485 84.7799 9.64794C84.1898 10.2949 83.6389 10.9889 83.2135 11.7928C83.1546 11.9045 82.9841 11.8614 82.99 11.734C83.0507 10.4067 82.9233 7.41489 80.6883 6.34639L78.1866 5.37984L77.9925 5.9582C77.3632 7.82464 78.3925 9.81851 80.257 10.4518C81.8783 11.0027 82.4567 12.0476 82.0665 13.6141C82.0489 13.671 81.7666 15.2845 81.8058 16.0001H83.6036C83.6644 14.8904 84.829 14.1611 85.8386 14.614C86.1229 14.7414 86.415 14.9238 86.715 15.159C88.3227 16.4255 90.691 16.1256 91.9555 14.516L92.3163 14.0572L90.0421 12.4241C88.4815 11.1968 86.3994 11.7516 84.8584 12.8024C84.729 12.8907 84.5643 12.7495 84.6368 12.6084C86.4993 8.95391 88.9206 8.96175 89.8695 9.77341C91.0204 10.7576 92.7633 10.5812 93.7396 9.4264L94.02 9.09507L92.3829 7.82856H92.3849Z" fill="#51C233" />
                    <path d="M111.257 4.27539C114.524 4.27557 116.739 6.46855 116.739 9.98145C116.739 10.3833 116.718 10.788 116.673 11.2568H108.84C108.974 12.6434 109.892 13.4053 111.391 13.4053C112.398 13.4052 113.045 12.9803 113.338 12.375H116.538C115.888 14.5682 114.189 16 111.37 16C107.991 15.9998 105.754 13.6502 105.754 10.0703H105.751C105.751 6.55739 107.99 4.27539 111.257 4.27539ZM132.095 4.27539C134.801 4.2756 136.503 6.02159 136.503 8.95117V15.665H133.369V9.28613C133.369 7.81028 132.697 7.09379 131.444 7.09375C130.192 7.09375 129.362 7.96679 129.362 9.37598V15.6621H126.23V4.61035H128.984V5.72852C129.634 4.76615 130.82 4.27539 132.095 4.27539ZM106.379 2.72949H103.313C102.663 2.72949 102.305 2.99745 102.305 3.64746V4.60938H105.706V7.33887H102.305V15.6621H99.171V7.33887H96.42V4.60938H99.171V3.26758C99.171 1.11907 100.402 0 102.528 0H106.379V2.72949ZM120.583 6.55371C120.851 5.30087 121.747 4.60645 123.156 4.60645H125.126V4.98535C125.126 6.28287 124.074 7.33493 122.776 7.33496C121.546 7.33496 120.963 7.96297 120.963 9.21582V15.6611H117.829V4.60645H120.583V6.55371ZM111.257 6.73633C109.736 6.73633 108.907 7.58722 108.818 8.88477H113.584V8.83984C113.584 7.58713 112.778 6.73647 111.257 6.73633Z" fill="#1E1F24" />
                  </svg>
                  <svg
                    width="145"
                    height="16"
                    viewBox="0 0 145 16"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    className="built-with-fern-logo built-with-fern-logo-dark"
                    aria-hidden
                  >
                    <path d="M9.79656 4.8H14.5006C15.5139 4.8 16.3192 5.05067 16.9166 5.552C17.5139 6.04267 17.8126 6.71467 17.8126 7.568C17.8126 8.112 17.6739 8.608 17.3966 9.056C17.1192 9.504 16.7512 9.84 16.2926 10.064C16.8579 10.2667 17.3059 10.608 17.6366 11.088C17.9672 11.5573 18.1326 12.1173 18.1326 12.768C18.1326 13.7387 17.8286 14.5227 17.2206 15.12C16.6126 15.7067 15.7752 16 14.7086 16H9.79656V4.8ZM14.4846 14.528C15.1246 14.528 15.6206 14.3627 15.9726 14.032C16.3246 13.7013 16.5006 13.2373 16.5006 12.64C16.5006 12.0427 16.3246 11.5893 15.9726 11.28C15.6312 10.96 15.1352 10.8 14.4846 10.8H11.3966V14.528H14.4846ZM14.2766 9.424C14.8846 9.424 15.3539 9.28533 15.6846 9.008C16.0152 8.72 16.1806 8.32533 16.1806 7.824C16.1806 7.32267 16.0152 6.93867 15.6846 6.672C15.3539 6.40533 14.8846 6.272 14.2766 6.272H11.3966V9.424H14.2766ZM22.5778 16.224C21.6285 16.224 20.8871 15.9413 20.3538 15.376C19.8205 14.8107 19.5538 14 19.5538 12.944V8.304H21.1058V12.8C21.1058 13.472 21.2551 13.9787 21.5538 14.32C21.8631 14.6507 22.3005 14.816 22.8658 14.816C23.4525 14.816 23.9165 14.6293 24.2578 14.256C24.6098 13.872 24.7858 13.3707 24.7858 12.752V8.304H26.3378V16H24.9618V15.12C24.7165 15.4827 24.3858 15.76 23.9698 15.952C23.5538 16.1333 23.0898 16.224 22.5778 16.224ZM28.0746 8.304H29.6266V16H28.0746V8.304ZM27.9786 4.912H29.7066V6.752H27.9786V4.912ZM33.0334 16C32.4894 16 32.0948 15.888 31.8494 15.664C31.6041 15.44 31.4814 15.0667 31.4814 14.544V4.8H33.0334V14.064C33.0334 14.2667 33.0761 14.416 33.1614 14.512C33.2468 14.5973 33.3854 14.64 33.5774 14.64H34.5534V16H33.0334ZM37.9539 16C37.2819 16 36.7966 15.856 36.4979 15.568C36.1993 15.28 36.0499 14.8053 36.0499 14.144V9.664H34.0339V8.304H36.0499V6H37.6019V8.304H40.0179V9.664H37.6019V13.84C37.6019 14.1173 37.6659 14.32 37.7939 14.448C37.9219 14.576 38.1299 14.64 38.4179 14.64H40.0179V16H37.9539ZM43.5709 8.304H45.1869L46.8989 14.272L48.6109 8.304H50.3869L52.0989 14.272L53.8109 8.304H55.4269L53.0429 16H51.2189L49.5069 10.064L47.7789 16H45.9549L43.5709 8.304ZM56.3746 8.304H57.9266V16H56.3746V8.304ZM56.2786 4.912H58.0066V6.752H56.2786V4.912ZM62.5971 16C61.9251 16 61.4397 15.856 61.1411 15.568C60.8424 15.28 60.6931 14.8053 60.6931 14.144V9.664H58.6771V8.304H60.6931V6H62.2451V8.304H64.6611V9.664H62.2451V13.84C62.2451 14.1173 62.3091 14.32 62.4371 14.448C62.5651 14.576 62.7731 14.64 63.0611 14.64H64.6611V16H62.5971ZM65.6727 4.8H67.2247V9.056C67.4807 8.736 67.8007 8.496 68.1847 8.336C68.5794 8.16533 69.0114 8.08 69.4807 8.08C70.4407 8.08 71.1927 8.368 71.7367 8.944C72.2807 9.50933 72.5527 10.3147 72.5527 11.36V16H71.0007V11.504C71.0007 10.832 70.8407 10.3307 70.5207 10C70.2114 9.65867 69.7687 9.488 69.1927 9.488C68.5954 9.488 68.1154 9.68 67.7527 10.064C67.4007 10.4373 67.2247 10.9333 67.2247 11.552V16H65.6727V4.8Z" fill="#EEEEF0" />
                    <path d="M92.3848 7.82856C91.332 6.93847 89.7459 6.58166 88.3402 7.62074C88.2755 7.66779 88.1952 7.58741 88.2442 7.52468C88.5775 7.09532 88.9637 6.63263 89.2754 6.16798C89.593 5.69157 90.0675 5.35044 90.6145 5.18379C93.5259 4.30155 92.6515 0.00012207 92.6515 0.00012207C92.6515 0.00012207 88.154 0.290282 88.7088 4.17019C88.801 4.81913 88.6284 5.47983 88.2226 5.99545C87.7246 6.62479 87.1463 7.22667 86.7267 7.66191C86.6385 7.7521 86.4895 7.66583 86.5248 7.54428C86.9306 6.17778 87.2266 4.06432 85.8209 2.70175L83.8427 1.05881L83.4624 1.56071C82.3312 3.05268 82.6625 5.15634 84.1564 6.28561C85.0132 6.93259 85.4014 7.63643 85.3406 8.40888C85.3033 8.87157 85.0936 9.30485 84.7799 9.64794C84.1898 10.2949 83.6388 10.9889 83.2134 11.7928C83.1546 11.9045 82.984 11.8614 82.9899 11.734C83.0507 10.4067 82.9232 7.41489 80.6882 6.34639L78.1866 5.37984L77.9925 5.9582C77.3631 7.82464 78.3924 9.81851 80.2569 10.4518C81.8783 11.0027 82.4566 12.0476 82.0665 13.6141C82.0488 13.671 81.7665 15.2845 81.8057 16.0001H83.6036C83.6643 14.8904 84.8289 14.1611 85.8386 14.614C86.1229 14.7414 86.415 14.9238 86.7149 15.159C88.3226 16.4255 90.6909 16.1256 91.9555 14.516L92.3162 14.0572L90.042 12.4241C88.4814 11.1968 86.3993 11.7516 84.8583 12.8024C84.7289 12.8907 84.5642 12.7495 84.6368 12.6084C86.4993 8.95391 88.9206 8.96175 89.8695 9.77341C91.0203 10.7576 92.7632 10.5812 93.7396 9.4264L94.0199 9.09507L92.3829 7.82856H92.3848Z" fill="#51C233" />
                    <path d="M111.257 4.27539C114.524 4.27557 116.739 6.46855 116.739 9.98145C116.739 10.3833 116.718 10.788 116.673 11.2568H108.84C108.974 12.6434 109.892 13.4053 111.391 13.4053C112.398 13.4052 113.045 12.9803 113.338 12.375H116.538C115.888 14.5682 114.189 16 111.37 16C107.991 15.9998 105.754 13.6502 105.754 10.0703H105.751C105.751 6.55739 107.989 4.27539 111.257 4.27539ZM132.095 4.27539C134.801 4.2756 136.503 6.02159 136.503 8.95117V15.665H133.369V9.28613C133.369 7.81028 132.697 7.09379 131.444 7.09375C130.191 7.09375 129.362 7.96679 129.362 9.37598V15.6621H126.229V4.61035H128.983V5.72852C129.633 4.76615 130.82 4.27539 132.095 4.27539ZM106.379 2.72949H103.312C102.662 2.72949 102.305 2.99745 102.305 3.64746V4.60938H105.706V7.33887H102.305V15.6621H99.1709V7.33887H96.4199V4.60938H99.1709V3.26758C99.1709 1.11907 100.402 0 102.528 0H106.379V2.72949ZM120.583 6.55371C120.851 5.30087 121.747 4.60645 123.156 4.60645H125.126V4.98535C125.126 6.28287 124.074 7.33493 122.776 7.33496C121.546 7.33496 120.963 7.96297 120.963 9.21582V15.6611H117.829V4.60645H120.583V6.55371ZM111.257 6.73633C109.736 6.73633 108.907 7.58722 108.817 8.88477H113.584V8.83984C113.584 7.58713 112.777 6.73647 111.257 6.73633Z" fill="#EEEEF0" />
                  </svg>
                  <span className="built-with-fern-tooltip">Developer-friendly docs for your API</span>
                </a>
              </div>
            </div>
          </div>
          <div className="footer-item">
            <div className="footer-links">
              <a href="https://www.nvidia.com/en-us/about-nvidia/privacy-policy/" target="_blank" rel="noopener">Privacy Policy</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/about-nvidia/privacy-center/" target="_blank" rel="noopener">Your Privacy Choices</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/about-nvidia/terms-of-service/" target="_blank" rel="noopener">Terms of Service</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/about-nvidia/accessibility/" target="_blank" rel="noopener">Accessibility</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/about-nvidia/company-policies/" target="_blank" rel="noopener">Corporate Policies</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/product-security/" target="_blank" rel="noopener">Product Security</a>
              <span className="pipe-separator"> | </span>
              <a href="https://www.nvidia.com/en-us/contact/" target="_blank" rel="noopener">Contact</a>
            </div>
          </div>
          <div className="footer-item">
            <p className="copyright">Copyright &#169; {currentYear}, NVIDIA Corporation.</p>
          </div>
        </div>
      </div>
    </footer>
    </>
  );
}
