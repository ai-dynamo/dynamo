/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hosted-site layout corrections that must survive the NVIDIA global theme.
 *
 * The global theme replaces docs.yml `css:` and owns the custom footer, so
 * main.css cannot override theme chrome in hosted previews or production.
 * The publishing workflow renders this component once on every page; React
 * hoists the style element into the document head.
 */
const SITE_CSS = `
/* Preserve Fern's footer-aware sticky-sidebar sizing. The NVIDIA theme carries
   an older !important height that ignores --custom-footer-visible-height. The
   state/viewport attributes make this rule more specific than both the theme's
   default and preview-banner sidebar rules. */
#fern-sidebar[data-viewport="desktop"][data-state="sticky"] {
  height: calc(
    100dvh - var(--header-height) - var(--custom-footer-visible-height, 0px)
  ) !important;
}
`;

export function SiteStyles() {
  return <style>{SITE_CSS}</style>;
}
