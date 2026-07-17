/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * dep-index.js — Proposals index "registry" grid.
 *
 * WHAT
 *   Renders a filterable / sortable card grid of every Dynamo Enhancement
 *   Proposal into the <div id="dep-index"> mount emitted by
 *   fern/components/DepIndex.tsx on docs/proposals/index.mdx. Each card links
 *   to the DEP's stable /proposals/<slug> page.
 *
 * DATA SOURCE
 *   `window.__DEP_INDEX` — a list of per-DEP records emitted at build time by
 *   fern/scripts/sync_deps.py into fern/js/dep-index-data.js (loaded first via
 *   docs.yml `js:` order). Each record:
 *     { slug, dep, title, status, sig?, category?,
 *       authors?: [{label, handle}], submitter? }
 *   If the data file is missing / empty, the grid shows an empty state.
 *
 * STATUS COLOURS
 *   variant() copies DepMetadata.statusVariant() / dep-status-pills.js VERBATIM
 *   so a DEP's pill reads the same on the card, the index, and the sidebar.
 *
 * SPA NAVIGATION
 *   Fern is a Next.js app; client-side nav recreates the mount without a full
 *   reload. A MutationObserver + popstate handler rebuild the grid whenever a
 *   fresh, unbuilt mount appears. A per-mount `data-dep-index-ready` flag keeps
 *   the build idempotent. Mirrors the shape of fern/js/dep-status-pills.js.
 */
(function () {
  "use strict";

  if (typeof window === "undefined" || typeof document === "undefined") return;
  if (window.__depIndexLoaded) return;
  window.__depIndexLoaded = true;

  var MOUNT_ID = "dep-index";
  var READY_ATTR = "data-dep-index-ready";

  // Chip order for the status filter (leading "All" is the reset).
  var CHIP_STATUSES = [
    "All",
    "Draft",
    "Under Review",
    "Accepted",
    "Implemented",
    "Rejected",
    "Deferred",
    "Replaced",
  ];
  // Lifecycle sort order (index = rank). Unknown statuses sort last.
  var STATUS_RANK = [
    "Under Review",
    "Draft",
    "Accepted",
    "Implemented",
    "Deferred",
    "Replaced",
    "Rejected",
  ];
  // Watermark tint per variant (matches the v4 registry mockup).
  var ACCENT = {
    draft: "#e0a800",
    proposed: "#5b8def",
    accepted: "#76b900",
    rejected: "#dc4848",
    muted: "#9a9a9a",
  };

  // Per-mount view state (reset each time a fresh mount is built).
  var state = { status: "All", sig: "All SIGs", sort: "num" };
  var gridEl = null;
  var countEl = null;

  /**
   * Map a free-text lifecycle status to a pill variant.
   * MUST match statusVariant() in fern/components/DepMetadata.tsx and
   * variant() in fern/js/dep-status-pills.js VERBATIM.
   */
  function variant(status) {
    var s = String(status == null ? "" : status).toLowerCase();
    if (/accept|approv|implement|final|active|ratif/.test(s)) return "accepted";
    if (/propos|review/.test(s)) return "proposed";
    if (/reject|withdraw/.test(s)) return "rejected";
    if (/replac|supersed|deferr|defer/.test(s)) return "muted";
    return "draft";
  }

  /** Live read of the build-time dataset, or [] if it did not load. */
  function records() {
    var d = window.__DEP_INDEX;
    return Array.isArray(d) ? d : [];
  }

  /**
   * Base path for DEP page links: everything up to and including
   * `/proposals/` in the current URL, so links stay version-correct
   * (e.g. /dynamo/dev/proposals/<slug>).
   */
  function proposalsBase() {
    var p = window.location.pathname || "";
    var marker = "/proposals/";
    var i = p.indexOf(marker);
    if (i !== -1) return p.slice(0, i + marker.length);
    var j = p.indexOf("/proposals");
    if (j !== -1) return p.slice(0, j + "/proposals".length) + "/";
    return "/proposals/";
  }

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = String(text);
    return node;
  }

  function statusRank(status) {
    var i = STATUS_RANK.indexOf(String(status || ""));
    return i === -1 ? STATUS_RANK.length : i;
  }

  function compare(a, b) {
    if (state.sort === "status") {
      var ra = statusRank(a.status);
      var rb = statusRank(b.status);
      if (ra !== rb) return ra - rb;
    }
    // default + status tie-break: by DEP number/slug, ascending.
    var ka = String(a.dep || a.slug || "");
    var kb = String(b.dep || b.slug || "");
    return ka.localeCompare(kb, undefined, { numeric: true, sensitivity: "base" });
  }

  function uniqueSigs(items) {
    var seen = {};
    var out = [];
    for (var i = 0; i < items.length; i++) {
      var sig = items[i].sig;
      if (sig && !seen[sig]) {
        seen[sig] = true;
        out.push(sig);
      }
    }
    out.sort();
    return out;
  }

  /** Build one DEP card anchor. Text goes through textContent (never
   * innerHTML); the only attribute-sourced string is the avatar handle,
   * which the build restricts to a GitHub login shape and we encode. */
  function card(d) {
    var base = proposalsBase();
    var v = variant(d.status);
    var a = el("a", "dep-index-card");
    a.setAttribute("href", base + encodeURIComponent(d.slug));
    var num = d.dep ? String(d.dep) : "";
    a.setAttribute("data-num", num);
    a.style.setProperty("--dep-accent", ACCENT[v] || ACCENT.draft);

    var top = el("div", "dep-index-top");
    top.appendChild(el("span", "dep-index-num", num ? "DEP-" + num : "DEP"));
    top.appendChild(el("span", "dep-index-pill dep-index-pill--" + v, d.status));
    a.appendChild(top);

    a.appendChild(el("h3", "dep-index-cardtitle", d.title || d.slug));

    var tags = el("div", "dep-index-tags");
    var hasTag = false;
    if (d.sig) {
      tags.appendChild(el("span", "dep-index-tag dep-index-tag--sig", d.sig));
      hasTag = true;
    }
    if (d.category) {
      tags.appendChild(el("span", "dep-index-tag", d.category));
      hasTag = true;
    }
    if (hasTag) a.appendChild(tags);

    var foot = el("div", "dep-index-foot");
    var who = el("div", "dep-index-who");
    var authors = Array.isArray(d.authors) ? d.authors : [];
    var avatars = el("div", "dep-index-authors");
    var shown = 0;
    for (var i = 0; i < authors.length && shown < 3; i++) {
      var handle = authors[i] && authors[i].handle;
      if (!handle) continue;
      var img = document.createElement("img");
      img.setAttribute("loading", "lazy");
      img.setAttribute("alt", "@" + handle);
      img.setAttribute(
        "src",
        "https://github.com/" + encodeURIComponent(handle) + ".png?size=48",
      );
      avatars.appendChild(img);
      shown++;
    }
    if (shown > 0) who.appendChild(avatars);
    if (d.submitter) {
      who.appendChild(el("span", "dep-index-submitter", "@" + d.submitter));
    }
    foot.appendChild(who);
    a.appendChild(foot);

    return a;
  }

  function renderGrid() {
    if (!gridEl) return;
    var all = records();

    var filtered = [];
    for (var i = 0; i < all.length; i++) {
      var d = all[i];
      if (state.status !== "All" && d.status !== state.status) continue;
      if (state.sig !== "All SIGs" && d.sig !== state.sig) continue;
      filtered.push(d);
    }
    filtered.sort(compare);

    while (gridEl.firstChild) gridEl.removeChild(gridEl.firstChild);

    if (countEl) {
      countEl.textContent =
        filtered.length + (filtered.length === 1 ? " proposal" : " proposals");
    }

    if (filtered.length === 0) {
      var msg =
        all.length === 0
          ? "No proposals found. DEP metadata is generated at build time."
          : "No proposals match the current filters.";
      gridEl.appendChild(el("div", "dep-index-empty", msg));
      return;
    }
    for (var j = 0; j < filtered.length; j++) {
      gridEl.appendChild(card(filtered[j]));
    }
  }

  function buildSkeleton(mount) {
    // Reset view state for the fresh mount.
    state = { status: "All", sig: "All SIGs", sort: "num" };

    var head = el("div", "dep-index-head");
    head.appendChild(el("h2", "dep-index-title", "Dynamo Enhancement Proposals"));
    countEl = el("span", "dep-index-count", "");
    head.appendChild(countEl);
    mount.appendChild(head);

    var filters = el("div", "dep-index-filters");

    // Status chips.
    for (var i = 0; i < CHIP_STATUSES.length; i++) {
      (function (label) {
        var chip = el(
          "button",
          "dep-index-chip" + (label === state.status ? " is-active" : ""),
        );
        chip.setAttribute("type", "button");
        if (label !== "All") {
          var sw = el("span", "dep-index-sw");
          sw.style.background = ACCENT[variant(label)] || ACCENT.draft;
          chip.appendChild(sw);
        }
        chip.appendChild(document.createTextNode(label));
        chip.addEventListener("click", function () {
          state.status = label;
          var chips = filters.querySelectorAll(".dep-index-chip");
          for (var k = 0; k < chips.length; k++) {
            chips[k].classList.remove("is-active");
          }
          chip.classList.add("is-active");
          renderGrid();
        });
        filters.appendChild(chip);
      })(CHIP_STATUSES[i]);
    }

    // SIG filter (only when at least one DEP declares an owning SIG).
    var sigs = uniqueSigs(records());
    if (sigs.length > 0) {
      var sigSelect = el("select", "dep-index-select dep-index-select--sig");
      sigSelect.setAttribute("aria-label", "Filter by owning SIG");
      var allOpt = el("option", null, "All SIGs");
      allOpt.value = "All SIGs";
      sigSelect.appendChild(allOpt);
      for (var s = 0; s < sigs.length; s++) {
        var opt = el("option", null, sigs[s]);
        opt.value = sigs[s];
        sigSelect.appendChild(opt);
      }
      sigSelect.addEventListener("change", function () {
        state.sig = sigSelect.value;
        renderGrid();
      });
      filters.appendChild(sigSelect);
    }

    // Sort select.
    var sortSelect = el(
      "select",
      "dep-index-select dep-index-select--sort" + (sigs.length ? "" : " dep-index-select--sig"),
    );
    sortSelect.setAttribute("aria-label", "Sort proposals");
    var sortNum = el("option", null, "Sort \u00B7 Number");
    sortNum.value = "num";
    var sortStatus = el("option", null, "Sort \u00B7 Status");
    sortStatus.value = "status";
    sortSelect.appendChild(sortNum);
    sortSelect.appendChild(sortStatus);
    sortSelect.addEventListener("change", function () {
      state.sort = sortSelect.value;
      renderGrid();
    });
    filters.appendChild(sortSelect);

    mount.appendChild(filters);

    gridEl = el("div", "dep-index-grid");
    mount.appendChild(gridEl);

    renderGrid();
  }

  function scan() {
    var mount = document.getElementById(MOUNT_ID);
    if (!mount) return;
    if (mount.getAttribute(READY_ATTR) === "1") {
      // Same mount, but data may have arrived after the first build.
      renderGrid();
      return;
    }
    mount.setAttribute(READY_ATTR, "1");
    buildSkeleton(mount);
  }

  var scanScheduled = false;
  function scheduleScan() {
    if (scanScheduled) return;
    scanScheduled = true;
    setTimeout(function () {
      scanScheduled = false;
      scan();
    }, 60);
  }

  function boot() {
    scan();
    // The dataset script loads afterInteractive too; re-run a few times so a
    // late-arriving window.__DEP_INDEX still populates the grid.
    var delays = [120, 400, 1000];
    for (var i = 0; i < delays.length; i++) setTimeout(scan, delays[i]);
    var mo = new MutationObserver(scheduleScan);
    mo.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("popstate", scheduleScan);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
