/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * dep-status-pills.js — right-aligned lifecycle status pill for each DEP
 * link in the Proposals sidebar.
 *
 * WHY THIS FILE EXISTS
 *   The Proposals nav previously routed DEPs through a `Draft` section,
 *   which forced the DEP's status into the URL (/proposals/draft/<slug>).
 *   That meant a DEP's public URL changed when its status changed — bad
 *   for links, bad for canonical URLs, bad for search.
 *
 *   The fix flattens the nav (docs/index.yml) back to stable
 *   /proposals/<slug> URLs and instead injects a right-aligned status
 *   pill into each DEP's sidebar link at runtime. The pill's colours are
 *   cloned from the on-page <DepMetadata> pill (fern/main.css
 *   .dep-status-pill--{draft,proposed,accepted,rejected,muted}) and the
 *   status→variant mapping copies DepMetadata.statusVariant() VERBATIM so
 *   the sidebar and on-page pills always agree.
 *
 * DATA SOURCE
 *   `window.__DEP_STATUS` is a `{ slug → status }` map emitted at build
 *   time by fern/scripts/sync_deps.py into fern/js/dep-status-data.js
 *   (loaded first via docs.yml `js:` order). Sourced from:
 *     - synced DEPs — the parsed `**Status**:` from the enhancements-repo
 *       markdown, keyed by the manifest slug.
 *     - hand-authored DEPs — the `status="..."` prop on the page's
 *       `<DepMetadata>` component, keyed by the file basename which
 *       matches the slug in docs/index.yml.
 *   If the data file is missing (e.g. a bare `fern check` without running
 *   sync_deps.py first) the runtime silently no-ops instead of crashing.
 *
 * SPA NAVIGATION
 *   Fern is a Next.js app; client-side nav swaps the sidebar subtree
 *   without a full reload. A MutationObserver + popstate handler
 *   re-scans on every mutation; a per-link `data-dep-pill` flag keeps
 *   the injection idempotent so we never stack two pills. See the twin
 *   pattern in fern/js/dep-pr-comments.js for the origin of this shape.
 */
(function () {
  "use strict";

  if (typeof window === "undefined" || typeof document === "undefined") return;
  if (window.__depStatusPillsLoaded) return;
  window.__depStatusPillsLoaded = true;

  var DONE_ATTR = "data-dep-pill";
  var PILL_CLASS = "dep-status-pill";
  // Sidebar link selector (verified against the live preview per the Fern
  // CSS-selectors reference): sidebar links are `#fern-sidebar
  // a.fern-sidebar-link`, and the anchor is `display: flex; gap: 0.75rem`.
  // The pill uses `margin-left: auto` (see fern/main.css) to push itself
  // to the right end of that flex container.
  var LINK_SELECTOR = "#fern-sidebar a.fern-sidebar-link";

  /**
   * Map a free-text lifecycle status to a pill variant.
   *
   * MUST match `statusVariant()` in fern/components/DepMetadata.tsx
   * VERBATIM. If you change one, change the other or the sidebar and
   * on-page pills will disagree for the same DEP.
   */
  function variant(status) {
    var s = String(status == null ? "" : status).toLowerCase();
    if (/accept|approv|final|active|ratif/.test(s)) return "accepted";
    if (/propos|review/.test(s)) return "proposed";
    if (/reject|withdraw/.test(s)) return "rejected";
    if (/replac|supersed|deferr|defer/.test(s)) return "muted";
    return "draft";
  }

  /**
   * Extract the DEP slug from a sidebar link's href.
   *
   * Accepts absolute (https://...) and rooted paths; strips a trailing
   * slash, a query string, and a URL fragment before matching. Returns
   * the last path segment when the URL ends in `/proposals/<slug>`, and
   * "" for anything else (including `/proposals/` with no slug or a
   * `.../proposals/nested/<slug>` shape).
   */
  function slugFromHref(href) {
    if (!href) return "";
    var s = String(href);
    var q = s.indexOf("?");
    if (q !== -1) s = s.slice(0, q);
    var h = s.indexOf("#");
    if (h !== -1) s = s.slice(0, h);
    if (s.length > 1 && s.charAt(s.length - 1) === "/") s = s.slice(0, -1);
    var m = s.match(/\/proposals\/([^\/]+)$/);
    return m ? m[1] : "";
  }

  /** Read the build-time status map, or {} if the data file did not load. */
  function statusMap() {
    var m = window.__DEP_STATUS;
    return m && typeof m === "object" ? m : {};
  }

  /**
   * Remove pills whose surrounding anchor no longer resolves to a DEP the
   * status map knows about (e.g. after upstream removes a synced DEP), and
   * clear the per-link flag on any stripped anchor so a fresh scan can
   * re-inject with the current label. Cheap; runs once per debounced scan.
   */
  function cleanupStale(map) {
    var pills = document.querySelectorAll("." + PILL_CLASS);
    for (var i = 0; i < pills.length; i++) {
      var pill = pills[i];
      var anchor = pill.closest ? pill.closest("a.fern-sidebar-link") : null;
      if (!anchor) {
        if (pill.parentNode) pill.parentNode.removeChild(pill);
        continue;
      }
      var slug = slugFromHref(anchor.getAttribute("href") || "");
      if (!slug || !map[slug]) {
        if (pill.parentNode) pill.parentNode.removeChild(pill);
        anchor.removeAttribute(DONE_ATTR);
      }
    }
  }

  function scan() {
    var map = statusMap();
    // Nothing to render if the data file didn't load — silently no-op.
    // (fern check with a missing dep-status-data.js still validates.)
    if (!map || Object.keys(map).length === 0) return;

    cleanupStale(map);

    var links = document.querySelectorAll(LINK_SELECTOR);
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var href = a.getAttribute("href") || "";
      var slug = slugFromHref(href);
      if (!slug) continue;
      var status = map[slug];
      if (!status) continue;

      // Idempotency guard: skip if already labeled AND the pill still
      // exists. If the flag is set but the pill was stripped (React
      // remount), fall through and re-inject.
      if (
        a.getAttribute(DONE_ATTR) === "1" &&
        a.querySelector("." + PILL_CLASS)
      ) {
        continue;
      }

      // Defensive: strip any lingering pills before re-injecting so we
      // never stack two on one link.
      var stale = a.querySelectorAll("." + PILL_CLASS);
      for (var j = 0; j < stale.length; j++) {
        if (stale[j].parentNode) stale[j].parentNode.removeChild(stale[j]);
      }

      var pill = document.createElement("span");
      pill.className = PILL_CLASS + " " + PILL_CLASS + "--" + variant(status);
      // textContent, never innerHTML: the status label originates from
      // sync_deps.py which parses DEP metadata; even so, we keep the
      // injection surface zero.
      pill.textContent = String(status);
      pill.setAttribute("aria-hidden", "false");
      a.appendChild(pill);
      a.setAttribute(DONE_ATTR, "1");
    }
  }

  var scanScheduled = false;
  function scheduleScan() {
    if (scanScheduled) return;
    scanScheduled = true;
    // 60ms debounce mirrors dep-pr-comments.js; coalesces the burst of
    // mutations Fern emits during hydration + client-side nav.
    setTimeout(function () {
      scanScheduled = false;
      scan();
    }, 60);
  }

  function cleanupAll() {
    var pills = document.querySelectorAll("." + PILL_CLASS);
    for (var i = 0; i < pills.length; i++) {
      if (pills[i].parentNode) pills[i].parentNode.removeChild(pills[i]);
    }
    var flagged = document.querySelectorAll(
      "a.fern-sidebar-link[" + DONE_ATTR + "='1']",
    );
    for (var k = 0; k < flagged.length; k++) {
      flagged[k].removeAttribute(DONE_ATTR);
    }
  }

  function onNav() {
    // popstate fires on back/forward and on Fern's own routing. Wipe
    // the injection state and re-scan against the new sidebar.
    cleanupAll();
    scheduleScan();
  }

  function boot() {
    scan();
    var mo = new MutationObserver(scheduleScan);
    mo.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("popstate", onNav);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
