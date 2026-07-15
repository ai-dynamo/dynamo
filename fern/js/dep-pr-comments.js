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

  /* Theme-aware: every background is paired with an EXPLICIT foreground so
   * nothing depends on inherited color (the comment card is anchored next to a
   * [!WARNING] callout, whose amber text was leaking into the card in dark
   * mode). Colors come from Fern's NVIDIA theme vars (--pst-color-*), which are
   * defined for BOTH light and dark in fern/main.css, so a single var adapts to
   * either theme without a per-theme selector. Light-mode literals are only
   * fallbacks. */
  var CSS =
    /* Highlight: keep the wrapped prose's own color (inherit) instead of the
     * UA <mark> default of black-on-yellow, which was unreadable on dark. */
    ".dep-pr-mark{background:rgba(118,185,0,.22);color:inherit;border-bottom:2px solid var(--nv-color-green,#76b900);border-radius:2px;padding:0 1px;cursor:pointer;}" +
    ".dark .dep-pr-mark,html[data-theme=dark] .dep-pr-mark{background:rgba(118,185,0,.34);}" +
    ".dep-pr-badge{display:inline-flex;align-items:center;gap:3px;vertical-align:super;margin:0 2px;padding:0 6px;height:17px;border:0;border-radius:9px;background:var(--nv-color-green,#76b900);color:#111;font:700 10px/17px system-ui,sans-serif;cursor:pointer;user-select:none;}" +
    ".dep-pr-badge:hover{filter:brightness(1.05);}" +
    /* Density: cards were previously sized for 2-3 lines of plaintext.
     * Now that body_html renders markdown (headings, code, lists,
     * blockquotes) the card body carries its own internal rhythm, so the
     * outer chrome (card padding, eyebrow gap, comment separators, link
     * margin) is aggressively tightened here. Kept theme-aware via the
     * same --pst-color-* vars used elsewhere. See fern/main.css for the
     * NVIDIA-theme variable definitions. */
    ".dep-pr-card{margin:2px 0 8px;padding:8px 11px 7px;border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));border-left:3px solid var(--nv-color-green,#76b900);border-radius:8px;background:var(--pst-color-surface,#f7f7f7);color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-card-eyebrow{font:700 10px/1 system-ui,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--nv-color-green,#76b900);margin-bottom:3px;}" +
    ".dep-pr-comment{color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-comment + .dep-pr-comment{margin-top:6px;padding-top:6px;border-top:1px solid var(--border,var(--grayscale-a5,#e2e2e2));}" +
    ".dep-pr-comment-head{display:flex;align-items:center;gap:8px;margin-bottom:1px;}" +
    ".dep-pr-avatar{border-radius:50%;}" +
    ".dep-pr-author{font-weight:700;font-size:13px;color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-date{font-size:11px;color:var(--pst-color-text-muted,#777);}" +
    ".dep-pr-body{font-size:13px;line-height:1.4;color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-body p{margin:0 0 3px;}" +
    ".dep-pr-body p:last-child{margin-bottom:0;}" +
    /* Rendered markdown from body_html: keep headings and code compact so
     * a comment card stays visually distinct from the DEP prose around it.
     * Themed via --pst-color-* so light/dark both work without a per-theme
     * selector. */
    ".dep-pr-body h1,.dep-pr-body h2,.dep-pr-body h3,.dep-pr-body h4,.dep-pr-body h5,.dep-pr-body h6{margin:6px 0 3px;font-weight:700;line-height:1.25;color:var(--pst-color-heading,inherit);}" +
    ".dep-pr-body h1{font-size:15px;}.dep-pr-body h2{font-size:14px;}.dep-pr-body h3,.dep-pr-body h4,.dep-pr-body h5,.dep-pr-body h6{font-size:13px;}" +
    ".dep-pr-body ul,.dep-pr-body ol{margin:0 0 3px;padding-left:1.25em;}" +
    ".dep-pr-body li{margin:1px 0;}" +
    ".dep-pr-body li p{margin:0;}" +
    ".dep-pr-body blockquote{margin:2px 0 3px;padding:0 0 0 9px;border-left:3px solid var(--border,var(--grayscale-a5,#cfcfcf));color:var(--pst-color-text-muted,#555);}" +
    ".dep-pr-body blockquote p{margin:0 0 2px;}" +
    ".dep-pr-body code{padding:0 4px;border-radius:3px;background:rgba(118,185,0,.10);color:var(--pst-color-inline-code,inherit);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.92em;}" +
    ".dark .dep-pr-body code,html[data-theme=dark] .dep-pr-body code{background:rgba(118,185,0,.16);}" +
    ".dep-pr-body pre{margin:3px 0 4px;padding:8px 10px;overflow:auto;border-radius:6px;background:var(--pst-color-on-background,#0f0f0f);color:var(--pst-color-text-base,#eee);font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.4;}" +
    ".dep-pr-body pre code{padding:0;background:none;color:inherit;font-size:inherit;}" +
    ".dep-pr-body a{color:var(--nv-color-green,#76b900);text-decoration:underline;text-underline-offset:2px;}" +
    ".dep-pr-body img{max-width:100%;height:auto;}" +
    /* GitHub's fenced-code wrapper: .highlight is emitted around <pre>
     * with per-language variants. Keep it flush so the internal <pre>
     * carries the padding. */
    ".dep-pr-body .highlight{margin:3px 0 4px;}" +
    ".dep-pr-body .highlight pre{margin:0;}" +
    ".dep-pr-link{display:inline-block;margin-top:2px;font-size:12px;font-weight:600;color:var(--nv-color-green,#76b900);text-decoration:none;}" +
    ".dep-pr-link:hover{text-decoration:underline;}" +
    "#dep-pr-comments{display:block;margin-top:2.5rem;padding-top:1.25rem;border-top:1px solid var(--border,var(--grayscale-a4,#e5e5e5));}" +
    ".dep-pr-heading{font-size:1.15rem;font-weight:600;margin-bottom:.15rem;color:var(--pst-color-heading,inherit);}" +
    ".dep-pr-note{font-size:.85rem;color:var(--pst-color-text-muted,#777);margin-bottom:1rem;}" +
    ".dep-pr-unanchored{margin-top:1rem;padding:10px 12px;border:1px dashed var(--border,var(--grayscale-a5,#cfcfcf));border-radius:8px;color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-unanchored h4{margin:0 0 8px;font-size:.9rem;color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-unanchored li{margin:0 0 8px;font-size:13px;}" +
    /* Discussion sections (revision + design), read-only threaded lists. */
    ".dep-pr-thread{margin-top:1.75rem;color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-thread-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;flex-wrap:wrap;margin-bottom:.5rem;}" +
    ".dep-pr-thread-title{font-size:1.05rem;font-weight:600;margin:0;color:var(--pst-color-heading,inherit);}" +
    ".dep-pr-thread-sub{font-size:.8rem;color:var(--pst-color-text-muted,#777);margin:.1rem 0 0;}" +
    ".dep-pr-thread-item{padding:9px 11px;border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));border-radius:8px;background:var(--pst-color-surface,#f7f7f7);color:var(--pst-color-text-base,#1a1a1a);}" +
    ".dep-pr-thread-item + .dep-pr-thread-item{margin-top:8px;}" +
    ".dep-pr-action{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border:1px solid var(--nv-color-green,#76b900);border-radius:7px;background:transparent;color:var(--pst-color-text-base,#1a1a1a);font-size:12px;font-weight:600;line-height:1;text-decoration:none;white-space:nowrap;}" +
    ".dep-pr-action:hover{background:rgba(118,185,0,.12);text-decoration:none;}" +
    ".dep-pr-empty{padding:11px 13px;border:1px dashed var(--border,var(--grayscale-a5,#cfcfcf));border-radius:8px;font-size:13px;color:var(--pst-color-text-muted,#777);}" +
    ".dep-pr-empty a{color:var(--nv-color-green,#76b900);font-weight:600;text-decoration:none;}" +
    ".dep-pr-empty a:hover{text-decoration:underline;}" +
    ".dep-pr-quote{color:var(--pst-color-text-muted,#777);font-style:italic;}" +
    /* Highlight-to-comment: a small floating button that appears at the
     * end of a text selection inside .fern-prose. On click it copies a
     * blockquote of the selection to the clipboard and opens the DEP's
     * GitHub review surface in a new tab. The toast is a transient
     * confirmation. Both use theme vars so light + dark stay consistent
     * with the rest of the comment mirror. High z-index so Fern's own
     * floating toolbars don't cover the button. */
    ".dep-pr-quote-btn{position:fixed;z-index:9999;display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border:1px solid var(--nv-color-green,#76b900);border-radius:999px;background:var(--pst-color-surface,#f7f7f7);color:var(--pst-color-text-base,#1a1a1a);font:600 12px/1 system-ui,sans-serif;box-shadow:0 3px 10px rgba(0,0,0,.15);cursor:pointer;user-select:none;}" +
    ".dep-pr-quote-btn:hover{background:rgba(118,185,0,.12);}" +
    ".dark .dep-pr-quote-btn,html[data-theme=dark] .dep-pr-quote-btn{box-shadow:0 3px 12px rgba(0,0,0,.5);}" +
    ".dep-pr-toast{position:fixed;z-index:10000;padding:8px 14px;border:1px solid var(--border,var(--grayscale-a5,#dcdcdc));border-radius:8px;background:var(--pst-color-surface,#f7f7f7);color:var(--pst-color-text-base,#1a1a1a);font:600 12.5px/1.35 system-ui,sans-serif;box-shadow:0 3px 12px rgba(0,0,0,.18);opacity:0;transform:translateY(6px);transition:opacity 120ms ease,transform 120ms ease;pointer-events:none;}" +
    ".dep-pr-toast.on{opacity:1;transform:translateY(0);}" +
    ".dark .dep-pr-toast,html[data-theme=dark] .dep-pr-toast{box-shadow:0 3px 14px rgba(0,0,0,.6);}";

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

  function commentHtml(c, replyLabel) {
    var login = (c.user && c.user.login) || "unknown";
    var avatar = (c.user && c.user.avatar_url) || "";
    /* Prefer GitHub's server-rendered `body_html` (requested via the
     * "full" media type in ghGet) so markdown — bold, inline code, fenced
     * code, blockquotes, lists, headings, links — renders as HTML. It
     * matches what commenters see on github.com and needs no in-tree
     * markdown parser. Sanitize before injection; hardenLinks() below
     * runs post-parse to add target/rel to every anchor. Fall back to a
     * conservative plaintext-in-<p> wrapper when `body_html` is absent
     * (older/edge payloads, offline fixtures, etc.). */
    var body;
    if (typeof c.body_html === "string" && c.body_html) {
      body = sanitizeHtml(c.body_html);
    } else {
      body = String(c.body || "").split(/\n{2,}/)
        .map(function (p) { return "<p>" + esc(p).replace(/\n/g, "<br>") + "</p>"; }).join("");
    }
    var av = avatar
      ? '<img class="dep-pr-avatar" src="' + esc(avatar) + '" alt="" width="20" height="20">'
      : "";
    var label = replyLabel || "View / reply on GitHub";
    return '<div class="dep-pr-comment">' +
      '<div class="dep-pr-comment-head">' + av +
      '<span class="dep-pr-author">' + esc(login) + '</span>' +
      '<span class="dep-pr-date">' + esc(fmtDate(c.created_at)) + '</span></div>' +
      '<div class="dep-pr-body">' + body + '</div>' +
      '<a class="dep-pr-link" href="' + esc(c.html_url) + '" target="_blank" rel="noopener noreferrer">' + esc(label) + ' &rarr;</a>' +
      '</div>';
  }

  /* Post-innerHTML DOM pass: force every anchor rendered from body_html
   * to open in a new tab with a safe rel value. Doing this after the
   * browser has parsed the HTML is stricter than string surgery — every
   * <a> in the subtree ends up with the same policy regardless of how
   * GitHub emitted it (some anchors have rel="nofollow", some are user
   * mentions with data-*, etc.). Called from renderInto() at the end of
   * each render pass, scoped to the mount + inline cards. */
  function hardenLinks(root) {
    if (!root || !root.querySelectorAll) return;
    var anchors = root.querySelectorAll(".dep-pr-body a");
    var UNSAFE = { "javascript:": 1, "data:": 1, "vbscript:": 1 };
    for (var i = 0; i < anchors.length; i++) {
      var a = anchors[i];
      // Two-stage scheme check: (1) raw attribute for the string-only
      // path — catches typical UA-normalised cases; (2) the reflected
      // `a.protocol` property, which the URL parser canonicalises
      // (whitespace/control-char stripped, case-normalised). The
      // property-based check closes the whitespace-obfuscated-scheme
      // gap (`ja\tvascript:...`) that a raw-attribute regex misses.
      var raw = (a.getAttribute("href") || "").toLowerCase();
      var scheme = (a.protocol || "").toLowerCase();
      if (
        /^\s*(javascript|data|vbscript)\s*:/i.test(raw) ||
        UNSAFE[scheme] === 1
      ) {
        a.setAttribute("href", "#");
      }
      a.setAttribute("target", "_blank");
      a.setAttribute("rel", "noopener noreferrer");
    }
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
    var issue = parseInt(mount.getAttribute("data-issue") || "0", 10);
    var owner = mount.getAttribute("data-owner") || "ai-dynamo";
    var repo = mount.getAttribute("data-repo") || "dynamo";
    var path = mount.getAttribute("data-path") || "";
    var autoOpen = mount.getAttribute("data-auto-open") === "true";
    return { pr: pr, issue: issue, owner: owner, repo: repo, path: path, autoOpen: autoOpen };
  }

  /* Inline review-comment summary + the unanchored fallback list. Returns HTML;
   * renderInto() composes it with the discussion sections. */
  function inlineSummaryHtml(cfg, comments, anchoredCount, unanchored, reviewErr) {
    var prUrl = "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + cfg.pr;
    if (reviewErr) {
      return '<p class="dep-pr-heading">Inline review comments</p>' +
        '<p class="dep-pr-note">' + esc(rateMsg(reviewErr, "review comments")) +
        ' View them on <a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener noreferrer">PR #' +
        esc(cfg.pr) + "</a>.</p>";
    }
    var total = comments.length;
    var head =
      '<p class="dep-pr-heading">Inline review comments</p>' +
      '<p class="dep-pr-note">Read-only mirror of the ' + total + " review comment" +
      (total === 1 ? "" : "s") +
      ' on this file from <a href="' + esc(prUrl) + '" target="_blank" rel="noopener noreferrer">PR #' +
      esc(cfg.pr) + "</a>" +
      " (" + anchoredCount + " anchored to text above). Authoring stays on GitHub &mdash; " +
      '<a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener noreferrer">reply inline on the PR</a>.</p>';
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
              '" target="_blank" rel="noopener noreferrer">reply on the PR &rarr;</a></li>';
          }).join("");
        }).join("") + "</ul></div>";
    }
    return head + un;
  }

  function rateMsg(err, what) {
    return err && /\b403\b/.test(String(err.message || err))
      ? "GitHub's unauthenticated API rate limit (60/hr per IP) was hit, so the " + what + " could not load."
      : "The " + what + " could not load.";
  }

  /* A read-only threaded discussion section (PR conversation or issue thread).
   *   kind: "revision" (PR #pr conversation) or "design" (tracking issue).
   * Empty + error states deep-link back to the GitHub thread. */
  function threadHtml(cfg, kind, comments, err) {
    var isDesign = kind === "design";
    var title = isDesign ? "Design discussion" : "Revision discussion";
    var num = isDesign ? cfg.issue : cfg.pr;
    var threadUrl = isDesign
      ? "https://github.com/" + cfg.owner + "/" + cfg.repo + "/issues/" + num
      : "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + num;
    var replyLabel = isDesign ? "Comment on the issue" : "Comment on the PR";
    var sub = isDesign
      ? "Durable, open-ended debate on the tracking issue &mdash; the anchor for this DEP."
      : "General conversation on the revision (pull request), outside the line comments above.";

    // Design discussion with no tracking issue configured: prompt to start one.
    if (isDesign && !num) {
      var pageTitle = (document.title || "this DEP").replace(/\s*[|\u2013\-].*$/, "");
      var newIssue = "https://github.com/" + cfg.owner + "/" + cfg.repo +
        "/issues/new?title=" + encodeURIComponent("DEP: " + pageTitle);
      return '<div class="dep-pr-thread dep-pr-ui"><div class="dep-pr-thread-head">' +
        '<div><p class="dep-pr-thread-title">' + esc(title) + '</p>' +
        '<p class="dep-pr-thread-sub">' + sub + "</p></div></div>" +
        '<div class="dep-pr-empty">No tracking issue is linked yet. ' +
        '<a href="' + esc(newIssue) + '" target="_blank" rel="noopener noreferrer">Open a tracking issue &rarr;</a>' +
        " to start the design discussion, then set it on the DEP.</div></div>";
    }

    var head = '<div class="dep-pr-thread dep-pr-ui"><div class="dep-pr-thread-head">' +
      '<div><p class="dep-pr-thread-title">' + esc(title) + '</p>' +
      '<p class="dep-pr-thread-sub">' + sub + "</p></div>" +
      '<a class="dep-pr-action" href="' + esc(threadUrl) + '" target="_blank" rel="noopener noreferrer">' +
      esc(replyLabel) + " &rarr;</a></div>";

    if (err) {
      return head + '<div class="dep-pr-empty">' + esc(rateMsg(err, "discussion")) +
        ' <a href="' + esc(threadUrl) + '" target="_blank" rel="noopener noreferrer">Open the thread &rarr;</a></div></div>';
    }
    var humans = filterHumans(comments);
    if (!humans.length) {
      return head + '<div class="dep-pr-empty">No discussion yet. ' +
        '<a href="' + esc(threadUrl) + '" target="_blank" rel="noopener noreferrer">' + esc(replyLabel) +
        " &rarr;</a></div></div>";
    }
    var list = humans.map(function (c) {
      return '<div class="dep-pr-thread-item">' + commentHtml(c, replyLabel) + "</div>";
    }).join("");
    return head + list + "</div>";
  }

  /* Drop automation accounts so a discussion shows human debate, not CI chatter.
   * Two clean, generic rules cover the observed shapes:
   *   - `user.type === "Bot"` — GitHub sets this on every automation account,
   *     including GitHub Copilot review (login "Copilot", no [bot] suffix).
   *   - login ending in "[bot]" — copy-pr-bot, github-actions, dependabot, and
   *     the like, for cases where a legacy payload lacks `type`. */
  function filterHumans(comments) {
    return (comments || []).filter(function (c) {
      var user = c.user || {};
      if (user.type === "Bot") return false;
      var login = user.login || "";
      return !/\[bot\]$/.test(login);
    });
  }

  function ghGet(url) {
    /* Request the "full" media type so each comment payload carries
     * `body_html` alongside `body`. GitHub does the markdown -> HTML render
     * on their end (server-side, same renderer as their UI), which keeps
     * this runtime dependency-free and guarantees fidelity with what the
     * commenter sees on github.com. `body_html` is sanitized before we
     * inject it (see sanitizeHtml + hardenLinks). */
    return fetch(url, { headers: { Accept: "application/vnd.github.full+json" } })
      .then(function (r) {
        if (!r.ok) throw new Error("GitHub API " + r.status);
        return r.json();
      });
  }

  /* ------------------------------------------------------------------ *
   * sanitizeHtml — defense-in-depth pass over GitHub's server-rendered  *
   * `body_html` before it hits innerHTML. GitHub already emits well-    *
   * formed HTML from a trusted renderer; this pass hardens against a   *
   * compromised upstream or MITM by stripping executable surfaces:     *
   *   - <script>, <iframe>, <object>, <embed>, <style>, <link>, <meta>,*
   *     <base>, <svg>, <math>, <form>                                  *
   *   - inline event-handler attributes (on*=...)                      *
   *   - javascript:/data: URIs on href / src                           *
   * target="_blank" rel="noopener noreferrer" is added to <a> tags in  *
   * hardenLinks() below (that pass touches the parsed DOM, not the     *
   * string, so external links open correctly and non-http links stay   *
   * safe).                                                             *
   * ------------------------------------------------------------------ */
  function sanitizeHtml(html) {
    if (html == null) return "";
    var s = String(html);
    // Strip whole dangerous elements including any body they carry. The
    // outer group covers both `<x>...</x>` and self-closing `<x .../>`.
    var BLOCK_ELEMENTS = [
      "script", "iframe", "object", "embed", "style", "link",
      "meta", "base", "svg", "math", "form"
    ];
    for (var i = 0; i < BLOCK_ELEMENTS.length; i++) {
      var tag = BLOCK_ELEMENTS[i];
      // paired: <tag ...>...</tag> (non-greedy body)
      s = s.replace(new RegExp("<" + tag + "\\b[\\s\\S]*?</" + tag + "\\s*>", "gi"), "");
      // self-closing or unclosed: <tag ...> up to the next '>'
      s = s.replace(new RegExp("<" + tag + "\\b[^>]*>", "gi"), "");
    }
    // Strip inline event-handler attributes: on* with =. Value can be
    // double-quoted, single-quoted, or unquoted (up to whitespace or >).
    s = s.replace(/\son[a-z0-9_-]+\s*=\s*"[^"]*"/gi, "");
    s = s.replace(/\son[a-z0-9_-]+\s*=\s*'[^']*'/gi, "");
    s = s.replace(/\son[a-z0-9_-]+\s*=\s*[^\s>]+/gi, "");
    // Neutralize javascript: / data: / vbscript: URIs on href/src.
    // Three shapes matter — quoted (double or single) and unquoted:
    //   href="javascript:..."    href='javascript:...'    href=javascript:...
    // Prior version only caught the quoted forms; HTML5 lets unquoted
    // attribute values through, and a compromised upstream could inject
    // that shape. hardenLinks() below is the second line of defense for
    // <a> at DOM time, but for <img>/<track>/<source> the string pass is
    // the only pre-injection guard.
    var badUriQuoted = /(\s(?:href|src|xlink:href)\s*=\s*)("|')\s*(?:javascript|data|vbscript)\s*:[^"']*\2/gi;
    s = s.replace(badUriQuoted, '$1$2#$2');
    var badUriUnquoted = /(\s(?:href|src|xlink:href)\s*=\s*)(?:javascript|data|vbscript)\s*:[^\s>]*/gi;
    s = s.replace(badUriUnquoted, '$1"#"');
    return s;
  }

  /* ------------------------------------------------------------------ *
   * Pure helpers for the "highlight and click to comment on GitHub"    *
   * affordance. Kept at module scope (not inside the event handlers)   *
   * so the Node test file can extract and unit-test them directly —    *
   * the DOM plumbing that uses them is exercised only in the browser.  *
   * ------------------------------------------------------------------ */

  /** Turn selected plain text into a markdown blockquote.
   *
   *   "hello" -> "> hello"
   *   "a\nb"  -> "> a\n> b"
   *
   * CRLF and stray CR are normalised to LF, runs of blank lines are
   * collapsed to a single quoted blank ("> \n>\n> " -> "> \n>\n> "),
   * and the whole thing is truncated at 1500 chars with an ellipsis
   * so pasting a huge selection doesn't dump 20 KB into the reply.
   * Content is plain text; markdown/HTML metacharacters are preserved
   * verbatim so GitHub still renders exactly what the reader selected.
   */
  function buildBlockquote(text) {
    var BLOCKQUOTE_MAX = 1500;
    if (text == null) return "";
    var s = String(text).replace(/\r\n?/g, "\n");
    if (!s.replace(/\s/g, "")) return "";
    var truncated = false;
    if (s.length > BLOCKQUOTE_MAX) {
      s = s.slice(0, BLOCKQUOTE_MAX);
      truncated = true;
    }
    // Collapse runs of blank lines to a single blank so the resulting
    // blockquote reads cleanly on GitHub.
    s = s.replace(/\n{2,}/g, "\n\n");
    var lines = s.split("\n").map(function (line) {
      // A blank line inside a blockquote is "> " with only whitespace
      // -- render as ">" (no trailing space) so mobile clients don't
      // strip the empty quoted line.
      return line.length === 0 ? ">" : "> " + line;
    });
    var out = lines.join("\n");
    if (truncated) out += "…";
    return out;
  }

  /** Deep-link URL for the GitHub review surface associated with a mount.
   *
   * Prefers the PR "Files changed" view (`/pull/N/files`) because that is
   * the surface that supports line-level comments. Falls back to the
   * tracking issue when no PR is configured. Returns "" when neither
   * signal is available — the caller uses that to hide the button.
   */
  function buildGitHubUrl(cfg) {
    if (!cfg || !cfg.owner || !cfg.repo) return "";
    var base = "https://github.com/" + cfg.owner + "/" + cfg.repo;
    if (cfg.pr) return base + "/pull/" + cfg.pr + "/files";
    if (cfg.issue) return base + "/issues/" + cfg.issue;
    return "";
  }

  /* Fetch the three read-only sources in parallel; one failing source (e.g. a
   * 403 rate-limit) must not blank the others, so use allSettled and carry a
   * per-source error. */
  function fetchAll(cfg) {
    var base = "https://api.github.com/repos/" + cfg.owner + "/" + cfg.repo;
    var jobs = [
      ghGet(base + "/pulls/" + cfg.pr + "/comments?per_page=100"),   // inline review
      ghGet(base + "/issues/" + cfg.pr + "/comments?per_page=100"),  // PR conversation
      cfg.issue ? ghGet(base + "/issues/" + cfg.issue + "/comments?per_page=100")
                : Promise.resolve([])                                // tracking issue
    ];
    return Promise.allSettled(jobs).then(function (res) {
      var val = function (i) { return res[i].status === "fulfilled" ? res[i].value : []; };
      var err = function (i) { return res[i].status === "rejected" ? res[i].reason : null; };
      return {
        review: val(0), reviewErr: err(0),
        prConv: val(1), prConvErr: err(1),
        issue: val(2), issueErr: err(2)
      };
    });
  }

  function cacheKey(cfg) {
    return cfg.owner + "/" + cfg.repo + "#" + cfg.pr + "@" + cfg.issue;
  }

  /* Anchor + render synchronously (no await inside). cleanup() first so a
   * re-render (React node swap / SPA nav) never stacks marks on top of a prior
   * pass. Called only while `running` is held, so it can't interleave. */
  function renderInto(mount, cfg, root, data) {
    cleanup();
    // Uniform bot filter: strip automation accounts (Copilot review, etc.)
    // from the inline review set BEFORE anchoring/grouping so both the
    // <mark> highlights and the unanchored fallback panel see the same
    // human-only set as the two discussion threads below.
    var comments = filterHumans(data.review || []).filter(function (c) {
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
      /* Anchor the badge as a sibling of any enclosing <a> when the last mark
       * lives inside a hyperlink (e.g. a quote that overlaps `[@grahamking]
       * (...)`). Nesting a <button> inside <a> is invalid HTML and, because
       * clicks bubble, tapping the badge would open the link instead of
       * toggling this card. The wrapping itself is unchanged — per-text-node
       * marks still land inside the <a>, so the highlight and href survive. */
      var badgeAnchor = marks[marks.length - 1];
      var badgeBlock = blockAncestor(badgeAnchor, root);
      var probe = badgeAnchor.parentNode;
      while (probe && probe !== badgeBlock && probe !== root) {
        if ((probe.nodeName || "").toUpperCase() === "A") { badgeAnchor = probe; break; }
        probe = probe.parentNode;
      }
      badgeAnchor.insertAdjacentElement("afterend", badge);

      var card = document.createElement("div");
      card.className = "dep-pr-card dep-pr-ui";
      card.style.display = cfg.autoOpen ? "" : "none";
      card.innerHTML = '<div class="dep-pr-card-eyebrow">PR #' + esc(cfg.pr) +
        " review \u00B7 line comment</div>" +
        g.comments.map(function (c) { return commentHtml(c, "Reply on the PR"); }).join("");
      var block = blockAncestor(marks[0], root);
      block.insertAdjacentElement("afterend", card);

      badge.addEventListener("click", function () { toggleCard(card, badge); });
    });

    // The mount section holds: inline-review summary, then the two read-only
    // discussion threads (revision = PR conversation, design = tracking issue).
    mount.innerHTML =
      inlineSummaryHtml(cfg, comments, anchored, unanchored, data.reviewErr) +
      threadHtml(cfg, "revision", data.prConv, data.prConvErr) +
      threadHtml(cfg, "design", data.issue, data.issueErr);

    // Force target/rel on every anchor rendered from body_html — both in
    // the mount (revision + design threads) and in the per-anchor inline
    // cards that were inserted into the prose above.
    hardenLinks(mount);
    var inlineCards = document.querySelectorAll(".dep-pr-card.dep-pr-ui");
    for (var ci = 0; ci < inlineCards.length; ci++) hardenLinks(inlineCards[ci]);
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
      : fetchAll(cfg).then(function (data) {
          commentCache[key] = data;
          return data;
        });

    pending.then(function (data) {
      try {
        renderInto(mount, cfg, root, data);
        mount.setAttribute(DONE_ATTR, "1");
      } finally {
        running = false;
        // Any scan() calls that arrived while we were fetching early-returned
        // on the single-flight lock. If SPA nav swapped the mount mid-fetch,
        // the new mount is undone and would only render on the next
        // incidental DOM mutation; sweep it now (idempotent under
        // scanScheduled, so this cannot double-fire with pending observers).
        scheduleScan();
      }
    }).catch(function () {
      // fetchAll uses allSettled and never rejects; this only guards an
      // unexpected render error so the single-flight lock is always released.
      running = false;
      scheduleScan();
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
    // The quote button is infrastructure (not tagged .dep-pr-ui, so
    // cleanup doesn't touch it), but any stashed selection from the
    // outgoing page is no longer valid — dismiss.
    if (typeof hideQuoteBtn === "function") hideQuoteBtn();
    retryCount = 0;
    scheduleScan();
  }

  /* ================================================================== *
   *  Highlight-to-comment affordance (MVP)                              *
   *                                                                     *
   * A DEP page is a READ-ONLY mirror; we cannot post to GitHub from     *
   * here. But we can shorten the reviewer's path from "spot the line I  *
   * want to comment on" to "reply on GitHub with the quote already in   *
   * my clipboard". On mouseup with a non-empty selection inside the    *
   * anchoring root, we pop up a floating "💬 Comment on GitHub" button. *
   * Clicking it: builds a markdown blockquote of the selection, copies  *
   * it to the clipboard, opens the DEP's GitHub review surface in a     *
   * new tab, and shows a transient toast so the reader knows the       *
   * clipboard is loaded. All state is per-page-load; no persistence.    *
   *                                                                     *
   * Scoping is deliberate: the button only appears when the selection  *
   * lives INSIDE the anchoring root (`resolveRoot`) and outside our    *
   * own comment cards / mount. That keeps it from firing on nav text,  *
   * on the sidebar, or on the rendered comments themselves.            *
   * ================================================================== */

  var quoteBtnEl = null;
  var toastEl = null;
  var quoteBtnScrollListener = null;
  var quoteBtnHideTimeout = null;

  /** Return the mount config from the first (only) mount on the page,
   *  or null if there is no DEP mount rendered. The highlight-to-comment
   *  affordance is only meaningful on pages that already show comments. */
  function activeMountConfig() {
    var mount = document.getElementById(MOUNT_ID);
    if (!mount) return null;
    return readConfig(mount);
  }

  function ensureQuoteBtn() {
    // Reattach if a prior cleanup() blew the node away, or if Fern's
    // hydration swapped the body element. The button is INFRASTRUCTURE
    // (not tagged .dep-pr-ui) so cleanup() doesn't remove it by default,
    // but SPA teardown can still detach it.
    if (quoteBtnEl && quoteBtnEl.isConnected) return quoteBtnEl;
    var b = document.createElement("button");
    b.type = "button";
    b.className = "dep-pr-quote-btn";
    b.setAttribute("aria-label", "Comment on GitHub with this quote");
    b.style.display = "none";
    // The label carries the 💬 glyph inline (no innerHTML with user data).
    b.appendChild(document.createTextNode("\uD83D\uDCAC Comment on GitHub"));
    b.addEventListener("click", onQuoteBtnClick);
    document.body.appendChild(b);
    quoteBtnEl = b;
    return b;
  }

  function ensureToast() {
    if (toastEl && toastEl.isConnected) return toastEl;
    var t = document.createElement("div");
    t.className = "dep-pr-toast";
    t.setAttribute("role", "status");
    document.body.appendChild(t);
    toastEl = t;
    return t;
  }

  function showToast(msg) {
    var t = ensureToast();
    // Text-only content: never innerHTML user-derived data.
    t.textContent = String(msg == null ? "" : msg);
    // Anchor to viewport bottom-center; simple, avoids collision with
    // the floating quote button which sits inline with the selection.
    t.style.left = "50%";
    t.style.bottom = "24px";
    t.style.top = "auto";
    t.style.transform = "translateX(-50%)";
    // Force reflow so the "on" transition always plays even for
    // back-to-back toast calls.
    /* eslint-disable-next-line no-unused-expressions */
    t.offsetWidth;
    t.classList.add("on");
    if (t._hideTimer) clearTimeout(t._hideTimer);
    t._hideTimer = setTimeout(function () {
      t.classList.remove("on");
    }, 2200);
  }

  function hideQuoteBtn() {
    if (!quoteBtnEl) return;
    quoteBtnEl.style.display = "none";
    quoteBtnEl.__depBtnSelection = null;
    if (quoteBtnHideTimeout) {
      clearTimeout(quoteBtnHideTimeout);
      quoteBtnHideTimeout = null;
    }
  }

  /** Copy text to the clipboard. Prefers the async Clipboard API;
   *  falls back to a temp <textarea> + execCommand("copy") when the
   *  Clipboard API is unavailable or blocked (older browsers, HTTP
   *  contexts, iframes with restrictive permissions). Returns a
   *  Promise<boolean> that resolves to `true` on success. */
  function copyToClipboard(text) {
    if (
      typeof navigator !== "undefined" &&
      navigator.clipboard &&
      typeof navigator.clipboard.writeText === "function"
    ) {
      return navigator.clipboard.writeText(text).then(
        function () { return true; },
        function () { return _copyFallback(text); }
      );
    }
    return Promise.resolve(_copyFallback(text));
  }
  function _copyFallback(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand && document.execCommand("copy");
      document.body.removeChild(ta);
      return !!ok;
    } catch (e) {
      return false;
    }
  }

  /** Is the given DOM node inside our own comment UI? We must never fire
   *  the highlight button when the reader is selecting text inside an
   *  already-rendered card (that would create a confusing loop). */
  function insideOwnUi(node) {
    var el = node && node.nodeType === 3 ? node.parentNode : node;
    while (el && el.nodeType === 1) {
      if (el.id === MOUNT_ID) return true;
      if (el.classList && el.classList.contains("dep-pr-ui")) return true;
      el = el.parentNode;
    }
    return false;
  }

  function selectionInsideProse(sel) {
    if (!sel || sel.rangeCount === 0 || sel.isCollapsed) return null;
    var range = sel.getRangeAt(0);
    if (insideOwnUi(range.startContainer)) return null;
    if (insideOwnUi(range.endContainer)) return null;
    // Resolve the same root the anchoring uses. If we can't find a
    // prose container, the button has no meaningful scope — skip.
    var root = resolveRoot(null);
    if (!root) return null;
    // Require the selection to be contained in the prose root; otherwise
    // it's in the sidebar / nav / footer, which is not commentable.
    if (!root.contains(range.commonAncestorContainer)) return null;
    return range;
  }

  function positionQuoteBtn(range) {
    var rects = range.getClientRects();
    var last = rects && rects.length ? rects[rects.length - 1] : range.getBoundingClientRect();
    if (!last || (!last.width && !last.height)) return false;
    var btn = ensureQuoteBtn();
    btn.style.display = "";
    // Measure after making it visible.
    var bw = btn.offsetWidth || 160;
    var bh = btn.offsetHeight || 30;
    var top = last.bottom + 8;
    var left = last.right - bw / 2;
    var vw = window.innerWidth || document.documentElement.clientWidth;
    var vh = window.innerHeight || document.documentElement.clientHeight;
    if (left < 8) left = 8;
    if (left + bw > vw - 8) left = vw - bw - 8;
    if (top + bh > vh - 8) top = last.top - bh - 8;
    btn.style.top = top + "px";
    btn.style.left = left + "px";
    btn.style.transform = "";
    return true;
  }

  function onQuoteBtnClick() {
    var btn = ensureQuoteBtn();
    var stored = btn.__depBtnSelection;
    var text = stored && stored.text ? stored.text : "";
    var cfg = stored && stored.cfg ? stored.cfg : null;
    hideQuoteBtn();
    if (!text || !cfg) return;
    var url = buildGitHubUrl(cfg);
    var quote = buildBlockquote(text);
    copyToClipboard(quote).then(function (copied) {
      if (url) {
        window.open(url, "_blank", "noopener,noreferrer");
      }
      showToast(
        copied
          ? "Quote copied \u2014 paste it in your GitHub comment."
          : "Opened GitHub \u2014 clipboard copy blocked; select and copy manually."
      );
    });
  }

  function onDocumentMouseUp(evt) {
    // A click on the button itself is not a "select-in-prose" event;
    // let the click handler do its thing without immediately hiding.
    if (quoteBtnEl && evt && quoteBtnEl.contains(evt.target)) return;
    // Give the browser a beat to finalise the selection object.
    setTimeout(function () {
      var sel = window.getSelection && window.getSelection();
      var range = selectionInsideProse(sel);
      if (!range) {
        hideQuoteBtn();
        return;
      }
      var cfg = activeMountConfig();
      if (!cfg) {
        hideQuoteBtn();
        return;
      }
      var url = buildGitHubUrl(cfg);
      if (!url) {
        // No PR and no tracking issue — no meaningful place to hand
        // off to. Silently skip.
        hideQuoteBtn();
        return;
      }
      var text = sel.toString();
      if (!text || !text.trim()) {
        hideQuoteBtn();
        return;
      }
      if (!positionQuoteBtn(range)) {
        hideQuoteBtn();
        return;
      }
      quoteBtnEl.__depBtnSelection = { text: text, cfg: cfg };
    }, 10);
  }

  function onDocumentMouseDown(evt) {
    // Click outside the button clears the selection UI. Do NOT hide when
    // clicking the button itself (its own click handler manages state).
    if (quoteBtnEl && evt && quoteBtnEl.contains(evt.target)) return;
    hideQuoteBtn();
  }

  function onSelectionChange() {
    var sel = window.getSelection && window.getSelection();
    if (!sel || sel.isCollapsed) {
      // Debounce a bit: a click on the button collapses the selection
      // between mousedown and click; hiding immediately would kill the
      // handler. onQuoteBtnClick runs its own cleanup.
      if (quoteBtnHideTimeout) return;
      quoteBtnHideTimeout = setTimeout(function () {
        quoteBtnHideTimeout = null;
        var s = window.getSelection && window.getSelection();
        if (!s || s.isCollapsed) hideQuoteBtn();
      }, 60);
    }
  }

  function onQuoteBtnEscape(evt) {
    if (evt && evt.key === "Escape") hideQuoteBtn();
  }

  function attachQuoteBtn() {
    if (window.__depQuoteBtnBound) return;
    window.__depQuoteBtnBound = true;
    // ensureQuoteBtn() wires the button's own click handler on first
    // create; document-level listeners live here.
    ensureQuoteBtn();
    document.addEventListener("mouseup", onDocumentMouseUp);
    document.addEventListener("mousedown", onDocumentMouseDown);
    document.addEventListener("selectionchange", onSelectionChange);
    document.addEventListener("keydown", onQuoteBtnEscape);
    // Scroll invalidates the anchored position; simpler to hide than
    // to reflow.
    quoteBtnScrollListener = hideQuoteBtn;
    window.addEventListener("scroll", quoteBtnScrollListener, true);
    window.addEventListener("resize", quoteBtnScrollListener);
  }

  function boot() {
    scan();
    var mo = new MutationObserver(scheduleScan);
    mo.observe(document.body, { childList: true, subtree: true });
    window.addEventListener("popstate", onNav);
    attachQuoteBtn();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
