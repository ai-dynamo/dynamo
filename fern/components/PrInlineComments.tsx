/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PrInlineComments — read-only inline overlay of a DEP's GitHub PR review line
 * comments, anchored to the referenced text on the rendered page.
 *
 * ONE-WAY / READ-ONLY. There is no login, no OAuth, no posting. Authoring stays
 * on the pull request; this component only DISPLAYS the PR's existing review
 * line comments and deep-links each back to GitHub ("View / reply on GitHub").
 * It complements GiscusComments (page-level thread) — this is the anchored,
 * line-level layer.
 *
 * Why this shape (mirrors GiscusComments.tsx / RecipeStyles.tsx):
 *   - Server component (no "use client"). Fern registers components via docs.yml
 *     `experimental.mdx-components: ./components`; import per page:
 *       import { PrInlineComments } from "@/components/PrInlineComments";
 *     The @/ prefix resolves to the fern/ root and is rewritten at publish.
 *   - All behavior ships as a single inline <script> (dependency-free vanilla JS,
 *     injected via dangerouslySetInnerHTML) that runs client-side, so there is no
 *     npm-bundling requirement and nothing runs during SSR. This is the same
 *     delivery mechanism GiscusComments uses for giscus's client.js, except the
 *     script body is inlined rather than loaded from a CDN.
 *   - The <style> is injected by the runtime (ensureStyles) so it survives the
 *     shared NVIDIA global theme at publish, per the RecipeStyles constraint.
 *
 * HOW ANCHORING WORKS (honest summary):
 *   Each PR review comment carries a `diff_hunk`; the last content line of the
 *   hunk is the commented source line. We strip inline markdown from it
 *   (`**Status**: Draft` -> `Status: Draft`) to get a TextQuoteSelector-style
 *   quote, then search the rendered DEP DOM for that quote (exact match first,
 *   then whitespace-normalized fuzzy), disambiguating repeated matches with the
 *   preceding line as a prefix hint, and wrap the match in <mark>. Source line
 *   numbers are NEVER used to anchor — they do not survive the .mdx -> HTML
 *   transform. If a quote can't be located (the DEP text was edited: "anchor
 *   drift"), the comment is NOT dropped — it is listed in an "unanchored" panel
 *   with its quote and a GitHub link.
 *
 * FETCH: runtime, client-side, UNAUTHENTICATED GitHub REST
 *   (GET /repos/{owner}/{repo}/pulls/{pr}/comments). Public repo, so no token.
 *   Subject to GitHub's 60-req/hr/IP unauthenticated limit; a 403/again error
 *   degrades to a small notice + a link to the PR. A production version should
 *   bake comments in at build time to remove the rate-limit and load flakiness.
 *
 * NOTE: custom Fern components do NOT render under local `fern docs dev`;
 * verification is on the hosted Fern PR preview build.
 *
 * The inline runtime below is generated verbatim from the prototype's shared
 * runtime (also exercised by a standalone local harness). Do not hand-edit the
 * embedded string; regenerate it from the source runtime.
 */

interface PrInlineCommentsProps {
  /** Pull request number whose review line comments are mirrored. */
  pr: number;
  /** Repository owner (default "ai-dynamo"). */
  owner?: string;
  /** Repository name (default "dynamo"). */
  repo?: string;
  /** Only show comments on this file path (default the example DEP). */
  path?: string;
}

const PR_INLINE_RUNTIME = `/*
 * PR Inline Comments — read-only runtime (dependency-free, vanilla JS).
 *
 * Mirrors GitHub PR *review line comments* onto the rendered DEP page, anchored
 * to the referenced text via a TextQuoteSelector-style match (quote + fuzzy
 * fallback). READ-ONLY: no auth, no OAuth, no posting. Authoring stays on the PR.
 *
 * Single source of truth: this file is loaded by the local harness
 * (.dep-preview/harness.html) AND embedded verbatim into the Fern component
 * fern/components/PrInlineComments.tsx (generated from this file). Keep it free
 * of backticks and "\${" so it embeds cleanly inside a TSX template literal.
 *
 * Entry point: window.__depPrInit(config)
 *   config = { pr, owner, repo, path, comments?, rootSelector?, autoOpen? }
 *   - comments: optional pre-loaded array (build-time bake / harness). If absent,
 *     the PR's comments are fetched unauthenticated from the public GitHub REST API.
 */
(function () {
  "use strict";
  if (typeof window === "undefined" || typeof document === "undefined") return;
  if (window.__depPrInit) return;

  var GH_API = "https://api.github.com";
  var BLOCK_TAGS = { P: 1, LI: 1, BLOCKQUOTE: 1, H1: 1, H2: 1, H3: 1, H4: 1, H5: 1, H6: 1, TD: 1, TH: 1, DD: 1, DT: 1, FIGCAPTION: 1, PRE: 1 };

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  function normWs(s) { return String(s == null ? "" : s).replace(/\\s+/g, " ").trim(); }

  /* Strip inline markdown so a SOURCE line matches the RENDERED text. */
  function stripMarkdown(line) {
    var s = String(line || "");
    s = s.replace(/^\\s*>+\\s?/, "");            // blockquote marker
    s = s.replace(/^\\s*#{1,6}\\s+/, "");          // ATX heading
    s = s.replace(/^\\s*(?:[-*+]|\\d+\\.)\\s+/, ""); // list bullet / ordered marker
    s = s.replace(/!\\[([^\\]]*)\\]\\([^)]*\\)/g, "$1"); // image -> alt
    s = s.replace(/\\[([^\\]]+)\\]\\([^)]*\\)/g, "$1");   // link -> text
    s = s.replace(/(\\*\\*|__)([\\s\\S]*?)\\1/g, "$2");   // bold
    s = s.replace(/(\\*|_)([\\s\\S]*?)\\1/g, "$2");      // italic
    s = s.replace(/~~([\\s\\S]*?)~~/g, "$1");          // strikethrough
    s = s.replace(/\`([^\`]+)\`/g, "$1");               // inline code
    s = s.replace(/&middot;/g, "\\u00B7").replace(/&nbsp;/g, " ")
         .replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">");
    return s.trim();
  }

  /* The commented line is the LAST content line of the diff hunk; the prior
     content line becomes a prefix hint to disambiguate repeated quotes. */
  function selectorFromComment(c) {
    var hunk = c.diff_hunk || "";
    var lines = hunk.split("\\n");
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
    return { quote: content.length ? content[content.length - 1] : "", prefix: content.length > 1 ? content[content.length - 2] : "" };
  }

  function resolveRoot(rootSelector) {
    var candidates = [];
    if (rootSelector) candidates.push(rootSelector);
    candidates = candidates.concat([".fern-prose", ".prose", "article", "main", "[role=main]"]);
    for (var i = 0; i < candidates.length; i++) {
      var el = document.querySelector(candidates[i]);
      if (el && normWs(el.textContent).length > 120) return el;
    }
    var sec = document.getElementById("dep-pr-comments");
    if (sec) { var up = sec.closest("article,main,section"); if (up) return up; }
    return document.body;
  }

  /* Collect visible text nodes under root, skipping our own UI + code/script/style. */
  function collectTextNodes(root) {
    var nodes = [];
    var walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        if (!n.nodeValue || !n.nodeValue.replace(/\\s+/g, "")) return NodeFilter.FILTER_REJECT;
        var p = n.parentNode;
        while (p && p !== root.parentNode) {
          var tag = (p.nodeName || "").toUpperCase();
          if (tag === "SCRIPT" || tag === "STYLE" || tag === "MARK") return NodeFilter.FILTER_REJECT;
          if (p.classList && p.classList.contains("dep-pr-ui")) return NodeFilter.FILTER_REJECT;
          if (p.id === "dep-pr-comments") return NodeFilter.FILTER_REJECT;
          p = p.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    });
    var node;
    while ((node = walker.nextNode())) nodes.push(node);
    return nodes;
  }

  /* Choose among multiple occurrences using a normalized prefix hint (the text
     immediately preceding the quote), mirroring a TextQuoteSelector prefix. */
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

  /* Locate quote across text nodes: exact match (prefix-disambiguated), then
     whitespace-normalized fuzzy fallback. */
  function locateQuote(root, quote, prefix) {
    var nodes = collectTextNodes(root);
    var raw = "";
    var map = [];
    for (var i = 0; i < nodes.length; i++) {
      var v = nodes[i].nodeValue;
      for (var j = 0; j < v.length; j++) map.push({ node: nodes[i], local: j });
      raw += v;
    }
    var indices = [], from = 0, hit;
    while ((hit = raw.indexOf(quote, from)) !== -1) { indices.push(hit); from = hit + Math.max(1, quote.length); }
    if (indices.length) { var idx = pickByPrefix(raw, indices, prefix); return { map: map, start: idx, end: idx + quote.length }; }

    var norm = "", n2r = [], prevWs = false;
    for (var k = 0; k < raw.length; k++) {
      var ch = raw.charAt(k);
      if (/\\s/.test(ch)) { if (prevWs) continue; norm += " "; n2r.push(k); prevWs = true; }
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

  /* Wrap [start,end) in <mark>, splitting across text-node boundaries as needed. */
  function wrapRange(loc, groupId) {
    var map = loc.map, start = loc.start, end = loc.end, segs = [], cur = null;
    for (var i = start; i < end; i++) {
      var m = map[i];
      if (!cur || cur.node !== m.node) { cur = { node: m.node, lo: m.local, hi: m.local + 1 }; segs.push(cur); }
      else cur.hi = m.local + 1;
    }
    var marks = [];
    for (var s = 0; s < segs.length; s++) {
      var seg = segs[s], target = seg.node;
      if (!target.parentNode) continue;
      if (seg.lo > 0) target = target.splitText(seg.lo);
      if (seg.hi - seg.lo < target.nodeValue.length) target.splitText(seg.hi - seg.lo);
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
    try { return new Date(iso).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }); }
    catch (e) { return iso || ""; }
  }

  function commentHtml(c) {
    var login = (c.user && c.user.login) || "unknown";
    var avatar = (c.user && c.user.avatar_url) || "";
    var body = String(c.body || "").split(/\\n{2,}/).map(function (p) { return "<p>" + esc(p).replace(/\\n/g, "<br>") + "</p>"; }).join("");
    var av = avatar ? '<img class="dep-pr-avatar" src="' + esc(avatar) + '" alt="" width="20" height="20">' : "";
    return '<div class="dep-pr-comment">' +
      '<div class="dep-pr-comment-head">' + av +
      '<span class="dep-pr-author">' + esc(login) + '</span>' +
      '<span class="dep-pr-date">' + esc(fmtDate(c.created_at)) + '</span></div>' +
      '<div class="dep-pr-body">' + body + '</div>' +
      '<a class="dep-pr-link" href="' + esc(c.html_url) + '" target="_blank" rel="noopener">View / reply on GitHub &rarr;</a>' +
      '</div>';
  }

  function ensureStyles() {
    if (document.getElementById("dep-pr-styles")) return;
    var css =
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
      "#dep-pr-comments{margin-top:2.5rem;padding-top:1.25rem;border-top:1px solid var(--grayscale-a4,#e5e5e5);}" +
      ".dep-pr-heading{font-size:1.15rem;font-weight:600;margin-bottom:.15rem;}" +
      ".dep-pr-note{font-size:.85rem;color:var(--pst-color-text-muted,#777);margin-bottom:1rem;}" +
      ".dep-pr-note code{font-size:.8rem;}" +
      ".dep-pr-unanchored{margin-top:1rem;padding:12px 14px;border:1px dashed var(--border,#cfcfcf);border-radius:8px;}" +
      ".dep-pr-unanchored h4{margin:0 0 8px;font-size:.9rem;}" +
      ".dep-pr-unanchored li{margin:0 0 8px;font-size:13px;}" +
      ".dep-pr-quote{color:var(--pst-color-text-muted,#777);font-style:italic;}";
    var st = document.createElement("style");
    st.id = "dep-pr-styles";
    st.appendChild(document.createTextNode(css));
    document.head.appendChild(st);
  }

  function toggleCard(card, badge) {
    var open = card.style.display !== "none";
    card.style.display = open ? "none" : "";
    badge.setAttribute("aria-expanded", open ? "false" : "true");
  }

  function run(cfg, root, allComments) {
    ensureStyles();
    var section = document.getElementById("dep-pr-comments");
    var prUrl = "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + cfg.pr;
    var comments = (allComments || []).filter(function (c) { return !cfg.path || c.path === cfg.path; });

    // group by derived quote, preserving comment order
    var groups = [], byQuote = {};
    comments.forEach(function (c) {
      var sel = selectorFromComment(c);
      var key = sel.quote + "\\u0000" + sel.prefix;
      if (!byQuote[key]) { byQuote[key] = { quote: sel.quote, prefix: sel.prefix, comments: [] }; groups.push(byQuote[key]); }
      byQuote[key].comments.push(c);
    });

    var anchored = 0, unanchored = [];
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
      badge.innerHTML = "\\uD83D\\uDCAC " + g.comments.length; // speech balloon + count
      marks[marks.length - 1].insertAdjacentElement("afterend", badge);

      var card = document.createElement("div");
      card.className = "dep-pr-card dep-pr-ui";
      card.style.display = cfg.autoOpen ? "" : "none";
      card.innerHTML = '<div class="dep-pr-card-eyebrow">PR #' + esc(cfg.pr) + ' review \\u00B7 line comment</div>' +
        g.comments.map(commentHtml).join("");
      var block = blockAncestor(marks[0], root);
      block.insertAdjacentElement("afterend", card);

      badge.addEventListener("click", function () { toggleCard(card, badge); });
    });

    // section summary + honest "unanchored" handling
    if (!section) return;
    var total = comments.length;
    var head =
      '<p class="dep-pr-heading">Inline review comments</p>' +
      '<p class="dep-pr-note">Read-only mirror of the ' + total + ' review comment' + (total === 1 ? "" : "s") +
      ' on this file from <a href="' + esc(prUrl) + '" target="_blank" rel="noopener">PR #' + esc(cfg.pr) + '</a>' +
      ' (' + anchored + ' anchored to text above). Authoring stays on GitHub &mdash; ' +
      '<a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener">discuss inline on the PR</a>.</p>';
    var un = "";
    if (unanchored.length) {
      un = '<div class="dep-pr-unanchored"><h4>Comments not anchored to current text (' + unanchored.length + ')</h4>' +
        '<p class="dep-pr-note">The commented line was edited or could not be located in the rendered page. Shown here so nothing is lost.</p><ul>' +
        unanchored.map(function (g) {
          return g.comments.map(function (c) {
            return '<li><span class="dep-pr-quote">&ldquo;' + esc(g.quote || "(unknown line)") + '&rdquo;</span> &mdash; ' +
              esc((c.user && c.user.login) || "unknown") + ': ' + esc(normWs(c.body).slice(0, 140)) +
              ' <a class="dep-pr-link" href="' + esc(c.html_url) + '" target="_blank" rel="noopener">on GitHub &rarr;</a></li>';
          }).join("");
        }).join("") + "</ul></div>";
    }
    section.innerHTML = head + un;
  }

  function renderError(cfg, err) {
    ensureStyles();
    var section = document.getElementById("dep-pr-comments");
    if (!section) return;
    var prUrl = "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + cfg.pr;
    var msg = (err && /\\b403\\b/.test(String(err.message))) ?
      "GitHub's unauthenticated API rate limit (60/hr per IP) was hit." :
      "Could not load PR comments.";
    section.innerHTML = '<p class="dep-pr-heading">Inline review comments</p>' +
      '<p class="dep-pr-note">' + esc(msg) + ' View them directly on ' +
      '<a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener">PR #' + esc(cfg.pr) + '</a>.</p>';
  }

  function fetchComments(cfg) {
    var url = GH_API + "/repos/" + cfg.owner + "/" + cfg.repo + "/pulls/" + cfg.pr + "/comments?per_page=100";
    return fetch(url, { headers: { Accept: "application/vnd.github+json" } }).then(function (r) {
      if (!r.ok) throw new Error("GitHub API " + r.status);
      return r.json();
    });
  }

  window.__depPrInit = function (config) {
    var cfg = {
      pr: (config && config.pr) || 0,
      owner: (config && config.owner) || "ai-dynamo",
      repo: (config && config.repo) || "dynamo",
      path: config && config.path,
      comments: config && config.comments,
      rootSelector: config && config.rootSelector,
      autoOpen: !!(config && config.autoOpen)
    };
    var tries = 0;
    function boot() {
      var root = resolveRoot(cfg.rootSelector);
      if ((!root || normWs(root.textContent).length < 60) && tries++ < 40) { setTimeout(boot, 250); return; }
      if (cfg.comments) { run(cfg, root, cfg.comments); return; }
      fetchComments(cfg).then(function (cs) { run(cfg, root, cs); }).catch(function (e) { renderError(cfg, e); });
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
    else boot();
  };
})();
`;

export function PrInlineComments({
  pr,
  owner = "ai-dynamo",
  repo = "dynamo",
  path = "docs/proposals/0000-example-dep.mdx",
}: PrInlineCommentsProps) {
  const boot =
    PR_INLINE_RUNTIME +
    "\nwindow.__depPrInit(" +
    JSON.stringify({ pr, owner, repo, path }) +
    ");";
  return (
    <section id="dep-pr-comments">
      <script dangerouslySetInnerHTML={{ __html: boot }} />
    </section>
  );
}
