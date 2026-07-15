/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * dep-pr-comments.js — read-only overlay of a DEP's GitHub PR review line
 * comments, anchored to the referenced text on the rendered Fern page.
 *
 * WHY THIS FILE IS DELIVERED VIA docs.yml js: (NOT MDX):
 *   Two prior attempts to ship this runtime as part of the PrInlineComments MDX
 *   component both failed in ways that only surfaced on the live preview:
 *     1) Inline <script dangerouslySetInnerHTML>. React writes the <script>
 *        into the DOM but does NOT execute it after hydration.
 *     2) "use client" + useEffect in the MDX component. Happens to work today,
 *        but MDX components in Fern are re-bundled through the docs pipeline and
 *        client-effect execution is not part of Fern's stability contract.
 *   The docs.yml `js:` mechanism is the DOCUMENTED customization surface for
 *   Fern (see https://buildwithfern.com/learn/docs/customization/custom-css-js).
 *   Fern injects this file as a <script> at page load, so the runtime is
 *   independent of MDX bundling and survives Fern-pipeline changes.
 *
 * MOUNT POINT (SSR): the MDX component <PrInlineComments pr={N} /> renders a
 *   plain <div id="dep-pr-comments" data-pr="N" data-owner="..." data-repo="..."
 *        data-path="..."> ... </div>
 * SSR of a plain <div> is reliable; this file finds it, fetches the PR's line
 * comments from the public GitHub REST API, and anchors them into `.fern-prose`.
 *
 * ANCHORING (honest summary): each PR review comment carries a `diff_hunk`; its
 * last content line is the commented source line. We strip inline markdown from
 * it (`**Status**: Draft` -> `Status: Draft`) to form a TextQuoteSelector-style
 * quote, then locate that quote in the rendered DOM (exact, then whitespace-
 * normalized fuzzy) using the preceding line as a prefix hint to disambiguate
 * repeats, and wrap the match in <mark>. Source line numbers are NEVER used —
 * they do not survive the .mdx -> HTML transform. Comments whose text can't be
 * located (anchor drift after an edit) are listed in an "unanchored" panel.
 *
 * FETCH: client-side, UNAUTHENTICATED GitHub REST
 *   GET /repos/{owner}/{repo}/pulls/{pr}/comments
 * Public repo, no token. Subject to GitHub's 60-req/hr/IP unauthenticated
 * limit; a 403 degrades to a small notice + PR link. A production version
 * should bake comments in at build time to remove that.
 *
 * SPA NAVIGATION: Fern is a Next.js app; the browser does not fully reload
 * between DEP pages. We run once on load and then observe DOM mutations for a
 * fresh mount point (and popstate as a belt-and-suspenders). Every mount div
 * gets a `data-dep-pr-rendered` flag to make re-runs idempotent.
 */
(function () {
  "use strict";

  if (typeof window === "undefined" || typeof document === "undefined") return;
  if (window.__depPrCommentsLoaded) return;
  window.__depPrCommentsLoaded = true;

  var STYLE_ID = "dep-pr-styles";
  var MOUNT_ID = "dep-pr-comments";
  var DONE_ATTR = "data-dep-pr-done";

  /* Single-flight + cache so the observer/route re-scans below can never run
   * two anchoring passes at once (which previously left an inconsistent mix of
   * one pass's <mark>s and another pass's summary). `running` is held across
   * the one async yield point (the GitHub fetch); `commentCache` makes re-runs
   * after a React node swap reuse the fetched comments instead of refetching. */
  var running = false;
  var commentCache = {};
  var retryScheduled = false;
  var scanScheduled = false;
  var retryCount = 0;
  var MAX_RETRIES = 40;

  var BLOCK_TAGS = {
    P: true, LI: true, BLOCKQUOTE: true, H1: true, H2: true, H3: true, H4: true,
    H5: true, H6: true, TD: true, TH: true, DD: true, DT: true, FIGCAPTION: true, PRE: true
  };

  var CSS =
    ".dep-pr-mark{background:rgba(118,185,0,.22);border-bottom:2px solid var(--nv-color-green,#76b900);border-radius:2px;padding:0 1px;cursor:pointer;}" +
    ".dep-pr-badge{display:inline-flex;align-items:center;gap:3px;vertical-align:super;margin:0 2px;padding:0 6px;height:17px;border:0;border-radius:9px;background:var(--nv-color-green,#76b900);color:#111;font:700 10px/17px system-ui,sans-serif;cursor:pointer;user-select:none;}" +
    ".dep-pr-badge:hover{filter:brightness(1.05);}" +
    ".dep-pr-card{margin:8px 0 16px;padding:12px 14px;border:1px solid var(--border,#dcdcdc);border-left:3px solid var(--nv-color-green,#76b900);border-radius:8px;background:var(--pst-color-surface,#f7f7f7);}" +
    ".dark .dep-pr-card{background:var(--nv-dark-grey-2,#1a1a1a);border-color:#333;}" +
    ".dep-pr-card-eyebrow{font:700 10px/1 system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--nv-color-green,#76b900);margin-bottom:8px;}" +
    ".dep-pr-comment + .dep-pr-comment{margin-top:12px;padding-top:12px;border-top:1px solid var(--border,#e2e2e2);}" +
    ".dep-pr-comment-head{display:flex;align-items:center;gap:8px;margin-bottom:4px;}" +
    ".dep-pr-avatar{border-radius:50%;}" +
    ".dep-pr-author{font-weight:700;font-size:13px;}" +
    ".dep-pr-date{font-size:11px;color:var(--pst-color-text-muted,#777);}" +
    ".dep-pr-body{font-size:13.5px;line-height:1.5;}" +
    ".dep-pr-body p{margin:0 0 6px;}" +
    ".dep-pr-link{display:inline-block;margin-top:6px;font-size:12px;font-weight:600;color:var(--nv-color-green,#5a8f00);text-decoration:none;}" +
    ".dep-pr-link:hover{text-decoration:underline;}" +
    "#dep-pr-comments{display:block;margin-top:2.5rem;padding-top:1.25rem;border-top:1px solid var(--grayscale-a4,#e5e5e5);}" +
    ".dep-pr-heading{font-size:1.15rem;font-weight:600;margin-bottom:.15rem;}" +
    ".dep-pr-note{font-size:.85rem;color:var(--pst-color-text-muted,#777);margin-bottom:1rem;}" +
    ".dep-pr-unanchored{margin-top:1rem;padding:12px 14px;border:1px dashed var(--border,#cfcfcf);border-radius:8px;}" +
    ".dep-pr-unanchored h4{margin:0 0 8px;font-size:.9rem;}" +
    ".dep-pr-unanchored li{margin:0 0 8px;font-size:13px;}" +
    ".dep-pr-quote{color:var(--pst-color-text-muted,#777);font-style:italic;}";

  function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    var st = document.createElement("style");
    st.id = STYLE_ID;
    st.appendChild(document.createTextNode(CSS));
    document.head.appendChild(st);
  }

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function normWs(s) {
    return String(s == null ? "" : s).replace(/\s+/g, " ").trim();
  }

  function stripMarkdown(line) {
    var s = String(line || "");
    s = s.replace(/^\s*>+\s?/, "");
    s = s.replace(/^\s*#{1,6}\s+/, "");
    s = s.replace(/^\s*(?:[-*+]|\d+\.)\s+/, "");
    s = s.replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1");
    s = s.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");
    s = s.replace(/(\*\*|__)([\s\S]*?)\1/g, "$2");
    s = s.replace(/(\*|_)([\s\S]*?)\1/g, "$2");
    s = s.replace(/~~([\s\S]*?)~~/g, "$1");
    s = s.replace(/`([^`]+)`/g, "$1");
    s = s.replace(/&middot;/g, "\u00B7").replace(/&nbsp;/g, " ")
         .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
    return s.trim();
  }

  function selectorFromComment(c) {
    var lines = String(c.diff_hunk || "").split("\n");
    var content = [];
    for (var i = 0; i < lines.length; i++) {
      var ln = lines[i];
      if (ln.indexOf("@@") === 0) { content = []; continue; }
      if (ln === "") continue;
      var marker = ln.charAt(0);
      var text = (marker === "+" || marker === "-" || marker === " ") ? ln.slice(1) : ln;
      var stripped = stripMarkdown(text);
      if (stripped) content.push(stripped);
    }
    return {
      quote: content.length ? content[content.length - 1] : "",
      prefix: content.length > 1 ? content[content.length - 2] : ""
    };
  }

  /* Resolve the rendered prose container. `.fern-prose` is Fern's canonical
   * wrapper for MDX prose (verified against the live preview). Falls back to
   * article/main if the class name ever changes upstream. */
  function resolveRoot(mount) {
    var candidates = [".fern-prose", ".prose", "article", "main", "[role=main]"];
    for (var i = 0; i < candidates.length; i++) {
      var el = document.querySelector(candidates[i]);
      if (el && normWs(el.textContent).length > 120) return el;
    }
    var up = mount ? mount.closest("article,main,section") : null;
    return up || document.body;
  }

  function collectTextNodes(root) {
    var nodes = [];
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        if (!n.nodeValue || !n.nodeValue.replace(/\s+/g, "")) return NodeFilter.FILTER_REJECT;
        var p = n.parentNode;
        while (p && p !== root.parentNode) {
          var el = p;
          var tag = (el.nodeName || "").toUpperCase();
          if (tag === "SCRIPT" || tag === "STYLE" || tag === "MARK") return NodeFilter.FILTER_REJECT;
          if (el.classList && el.classList.contains("dep-pr-ui")) return NodeFilter.FILTER_REJECT;
          if (el.id === MOUNT_ID) return NodeFilter.FILTER_REJECT;
          p = p.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var node;
    while ((node = walker.nextNode())) nodes.push(node);
    return nodes;
  }

  function pickByPrefix(raw, indices, prefix) {
    if (indices.length === 1 || !prefix) return indices[0];
    var np = normWs(prefix);
    if (!np) return indices[0];
    for (var i = 0; i < indices.length; i++) {
      var before = normWs(raw.slice(Math.max(0, indices[i] - np.length - 16), indices[i]));
      if (before.slice(-np.length) === np) return indices[i];
    }
    return indices[0];
  }

  function locateQuote(root, quote, prefix) {
    var nodes = collectTextNodes(root);
    var raw = "";
    var map = [];
    for (var i = 0; i < nodes.length; i++) {
      var v = nodes[i].nodeValue || "";
      for (var j = 0; j < v.length; j++) map.push({ node: nodes[i], local: j });
      raw += v;
    }
    var indices = [];
    var from = 0;
    var hit;
    while ((hit = raw.indexOf(quote, from)) !== -1) {
      indices.push(hit);
      from = hit + Math.max(1, quote.length);
    }
    if (indices.length) {
      var idx = pickByPrefix(raw, indices, prefix);
      return { map: map, start: idx, end: idx + quote.length };
    }
    var norm = "";
    var n2r = [];
    var prevWs = false;
    for (var k = 0; k < raw.length; k++) {
      var ch = raw.charAt(k);
      if (/\s/.test(ch)) { if (prevWs) continue; norm += " "; n2r.push(k); prevWs = true; }
      else { norm += ch; n2r.push(k); prevWs = false; }
    }
    var nq = normWs(quote);
    if (!nq) return null;
    var nidx = norm.indexOf(nq);
    if (nidx === -1) return null;
    var rawStart = n2r[nidx];
    var rawEnd = n2r[Math.min(nidx + nq.length - 1, n2r.length - 1)] + 1;
    return { map: map, start: rawStart, end: rawEnd };
  }

  function wrapRange(loc, groupId) {
    var map = loc.map, start = loc.start, end = loc.end;
    var segs = [];
    var cur = null;
    for (var i = start; i < end; i++) {
      var m = map[i];
      if (!cur || cur.node !== m.node) {
        cur = { node: m.node, lo: m.local, hi: m.local + 1 };
        segs.push(cur);
      } else {
        cur.hi = m.local + 1;
      }
    }
    var marks = [];
    for (var s = 0; s < segs.length; s++) {
      var seg = segs[s];
      var target = seg.node;
      if (!target.parentNode) continue;
      if (seg.lo > 0) target = target.splitText(seg.lo);
      if (seg.hi - seg.lo < (target.nodeValue || "").length) target.splitText(seg.hi - seg.lo);
      var mk = document.createElement("mark");
      mk.className = "dep-pr-mark";
      mk.setAttribute("data-dep-pr-group", groupId);
      target.parentNode.insertBefore(mk, target);
      mk.appendChild(target);
      marks.push(mk);
    }
    return marks;
  }

  function blockAncestor(node, root) {
    var el = node.nodeType === 3 ? node.parentNode : node;
    while (el && el !== root) {
      if (BLOCK_TAGS[(el.nodeName || "").toUpperCase()]) return el;
      el = el.parentNode;
    }
    return node.nodeType === 3 ? node.parentNode : node;
  }

  function fmtDate(iso) {
    try {
      return new Date(iso || "").toLocaleDateString(undefined, {
        year: "numeric", month: "short", day: "numeric"
      });
    } catch (e) {
      return iso || "";
    }
  }

  function commentHtml(c) {
    var login = (c.user && c.user.login) || "unknown";
    var avatar = (c.user && c.user.avatar_url) || "";
    var body = String(c.body || "").split(/\n{2,}/)
      .map(function (p) { return "<p>" + esc(p).replace(/\n/g, "<br>") + "</p>"; }).join("");
    var av = avatar
      ? '<img class="dep-pr-avatar" src="' + esc(avatar) + '" alt="" width="20" height="20">'
      : "";
    return '<div class="dep-pr-comment">' +
      '<div class="dep-pr-comment-head">' + av +
      '<span class="dep-pr-author">' + esc(login) + '</span>' +
      '<span class="dep-pr-date">' + esc(fmtDate(c.created_at)) + '</span></div>' +
      '<div class="dep-pr-body">' + body + '</div>' +
      '<a class="dep-pr-link" href="' + esc(c.html_url) + '" target="_blank" rel="noopener">View / reply on GitHub &rarr;</a>' +
      '</div>';
  }

  function cleanup() {
    var ui = document.querySelectorAll(".dep-pr-ui");
    for (var i = 0; i < ui.length; i++) {
      if (ui[i].parentNode) ui[i].parentNode.removeChild(ui[i]);
    }
    var marks = document.querySelectorAll("mark.dep-pr-mark");
    for (var j = 0; j < marks.length; j++) {
      var mk = marks[j];
      var p = mk.parentNode;
      if (!p) continue;
      while (mk.firstChild) p.insertBefore(mk.firstChild, mk);
      p.removeChild(mk);
      if (p.normalize) p.normalize();
    }
  }

  function toggleCard(card, badge) {
    var open = card.style.display !== "none";
    card.style.display = open ? "none" : "";
    badge.setAttribute("aria-expanded", open ? "false" : "true");
  }

  function readConfig(mount) {
    var pr = parseInt(mount.getAttribute("data-pr") || "0", 10);
    var owner = mount.getAttribute("data-owner") || "ai-dynamo";
    var repo = mount.getAttribute("data-repo") || "dynamo";
    var path = mount.getAttribute("data-path") || "";
    var autoOpen = mount.getAttribute("data-auto-open") === "true";
    return { pr: pr, owner: owner, repo: repo, path: path, autoOpen: autoOpen };
  }

  function renderSection(mount, cfg, comments, groups, anchoredCount, unanchored) {
    var prUrl = "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + cfg.pr;
    var total = comments.length;
    var head =
      '<p class="dep-pr-heading">Inline review comments</p>' +
      '<p class="dep-pr-note">Read-only mirror of the ' + total + " review comment" +
      (total === 1 ? "" : "s") +
      ' on this file from <a href="' + esc(prUrl) + '" target="_blank" rel="noopener">PR #' +
      esc(cfg.pr) + "</a>" +
      " (" + anchoredCount + " anchored to text above). Authoring stays on GitHub &mdash; " +
      '<a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener">discuss inline on the PR</a>.</p>';
    var un = "";
    if (unanchored.length) {
      un = '<div class="dep-pr-unanchored"><h4>Comments not anchored to current text (' +
        unanchored.length + ")</h4>" +
        '<p class="dep-pr-note">The commented line was edited or could not be located in the rendered page. Shown here so nothing is lost.</p><ul>' +
        unanchored.map(function (g) {
          return g.comments.map(function (c) {
            return '<li><span class="dep-pr-quote">&ldquo;' +
              esc(g.quote || "(unknown line)") + "&rdquo;</span> &mdash; " +
              esc((c.user && c.user.login) || "unknown") + ": " +
              esc(normWs(c.body).slice(0, 140)) +
              ' <a class="dep-pr-link" href="' + esc(c.html_url) +
              '" target="_blank" rel="noopener">on GitHub &rarr;</a></li>';
          }).join("");
        }).join("") + "</ul></div>";
    }
    mount.innerHTML = head + un;
  }

  function renderError(mount, cfg, err) {
    var prUrl = "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + cfg.pr;
    var msg = err && /\b403\b/.test(String(err.message || err))
      ? "GitHub's unauthenticated API rate limit (60/hr per IP) was hit."
      : "Could not load PR comments.";
    mount.innerHTML = '<p class="dep-pr-heading">Inline review comments</p>' +
      '<p class="dep-pr-note">' + esc(msg) + " View them directly on " +
      '<a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener">PR #' +
      esc(cfg.pr) + "</a>.</p>";
  }

  function fetchComments(cfg) {
    var url = "https://api.github.com/repos/" + cfg.owner + "/" + cfg.repo +
      "/pulls/" + cfg.pr + "/comments?per_page=100";
    return fetch(url, { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) {
        if (!r.ok) throw new Error("GitHub API " + r.status);
        return r.json();
      });
  }

  function cacheKey(cfg) {
    return cfg.owner + "/" + cfg.repo + "#" + cfg.pr;
  }

  /* Anchor + render synchronously (no await inside). cleanup() first so a
   * re-render (React node swap / SPA nav) never stacks marks on top of a prior
   * pass. Called only while `running` is held, so it can't interleave. */
  function renderInto(mount, cfg, root, raw) {
    cleanup();
    var comments = (raw || []).filter(function (c) {
      return !cfg.path || c.path === cfg.path;
    });

    var groups = [];
    var byQuote = {};
    comments.forEach(function (c) {
      var sel = selectorFromComment(c);
      var key = sel.quote + "\u0000" + sel.prefix;
      if (!byQuote[key]) {
        byQuote[key] = { quote: sel.quote, prefix: sel.prefix, comments: [] };
        groups.push(byQuote[key]);
      }
      byQuote[key].comments.push(c);
    });

    var anchored = 0;
    var unanchored = [];
    groups.forEach(function (g, gi) {
      var loc = g.quote ? locateQuote(root, g.quote, g.prefix) : null;
      if (!loc) { unanchored.push(g); return; }
      var marks = wrapRange(loc, "g" + gi);
      if (!marks.length) { unanchored.push(g); return; }
      anchored++;

      var badge = document.createElement("button");
      badge.className = "dep-pr-badge dep-pr-ui";
      badge.type = "button";
      badge.setAttribute("aria-expanded", "false");
      badge.innerHTML = "\uD83D\uDCAC " + g.comments.length;
      marks[marks.length - 1].insertAdjacentElement("afterend", badge);

      var card = document.createElement("div");
      card.className = "dep-pr-card dep-pr-ui";
      card.style.display = cfg.autoOpen ? "" : "none";
      card.innerHTML = '<div class="dep-pr-card-eyebrow">PR #' + esc(cfg.pr) +
        " review \u00B7 line comment</div>" +
        g.comments.map(commentHtml).join("");
      var block = blockAncestor(marks[0], root);
      block.insertAdjacentElement("afterend", card);

      badge.addEventListener("click", function () { toggleCard(card, badge); });
    });

    renderSection(mount, cfg, comments, groups, anchored, unanchored);
  }

  function scheduleRetry() {
    if (retryScheduled || retryCount >= MAX_RETRIES) return;
    retryScheduled = true;
    retryCount++;
    setTimeout(function () { retryScheduled = false; scan(); }, 250);
  }

  function run(mount) {
    if (running) return;                                  // single-flight
    if (mount.getAttribute(DONE_ATTR) === "1") return;    // already rendered
    var cfg = readConfig(mount);
    if (!cfg.pr) return;

    ensureStyles();

    var root = resolveRoot(mount);
    if (!root || normWs(root.textContent).length < 60) {
      scheduleRetry();  // prose not hydrated yet; retry without re-entrancy
      return;
    }

    running = true;
    var key = cacheKey(cfg);
    var pending = commentCache[key]
      ? Promise.resolve(commentCache[key])
      : fetchComments(cfg).then(function (raw) {
          commentCache[key] = raw || [];
          return commentCache[key];
        });

    pending.then(function (raw) {
      try {
        renderInto(mount, cfg, root, raw);
        mount.setAttribute(DONE_ATTR, "1");
      } finally {
        running = false;
      }
    }).catch(function (err) {
      try {
        renderError(mount, cfg, err);
        mount.setAttribute(DONE_ATTR, "1");
      } finally {
        running = false;
      }
    });
  }

  function scan() {
    var mounts = document.querySelectorAll(
      "#" + MOUNT_ID + "[data-pr]:not([" + DONE_ATTR + "='1'])"
    );
    for (var i = 0; i < mounts.length; i++) run(mounts[i]);
  }

  /* Debounce observer-driven scans: our own DOM writes (marks/cards/section)
   * and Fern's hydration both fire the observer; coalesce into one scan. */
  function scheduleScan() {
    if (scanScheduled) return;
    scanScheduled = true;
    setTimeout(function () { scanScheduled = false; scan(); }, 60);
  }

  function onNav() {
    /* SPA navigation: the previous mount div is detached. Clear stale marks and
     * force a fresh anchoring pass against the new page's mount (if any). */
    cleanup();
    retryCount = 0;
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
