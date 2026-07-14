"use client";
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * PrInlineComments — read-only inline overlay of a DEP's GitHub PR review line
 * comments, anchored to the referenced text on the rendered page.
 *
 * ONE-WAY / READ-ONLY. No login, no OAuth, no posting. Authoring stays on the
 * pull request; this only DISPLAYS the PR's existing review line comments and
 * deep-links each back to GitHub ("View / reply on GitHub"). It complements
 * GiscusComments (page-level thread) with the anchored, line-level layer.
 *
 * WHY THIS IS A CLIENT COMPONENT (the fix):
 *   The first cut shipped the logic as an inline <script dangerouslySetInnerHTML>.
 *   Fern server-renders custom components, and React renders such a <script> into
 *   the DOM but NEVER EXECUTES it after hydration. Verified on the live preview:
 *   the <section id="dep-pr-comments"> rendered, but its innerHTML was still the
 *   inert <script> text and there were zero <mark> highlights. Fern supports
 *   client components, so the fetch + anchor logic now runs directly in a
 *   useEffect (a real client execution path). Because useEffect re-runs on mount,
 *   this also re-anchors after SPA navigation; run() is idempotent (cleanup()).
 *
 * Registered via docs.yml `experimental.mdx-components: ./components`; import per
 * page: import { PrInlineComments } from "@/components/PrInlineComments";
 * NOTE: custom components do NOT render under local `fern docs dev` — verify on
 * the hosted Fern PR preview build.
 *
 * ANCHORING (honest summary): each PR review comment carries a `diff_hunk`; its
 * last content line is the commented source line. We strip inline markdown from
 * it (`**Status**: Draft` -> `Status: Draft`) to form a TextQuoteSelector-style
 * quote, then locate that quote in the rendered DOM (exact, then
 * whitespace-normalized fuzzy) using the preceding line as a prefix hint to
 * disambiguate repeats, and wrap the match in <mark>. Source line numbers are
 * NEVER used to anchor — they do not survive the .mdx -> HTML transform.
 * Comments whose text can't be located (anchor drift after an edit) are listed
 * in an "unanchored" panel rather than dropped.
 *
 * FETCH: client-side, UNAUTHENTICATED GitHub REST
 *   (GET /repos/{owner}/{repo}/pulls/{pr}/comments). Public repo, no token; the
 *   preview CSP allows it (connect-src https:). Subject to GitHub's 60-req/hr/IP
 *   unauthenticated limit; a 403 degrades to a small notice + PR link. A
 *   production version should bake comments in at build time to remove that.
 */
import { useEffect } from "react";

type GhUser = { login?: string; avatar_url?: string };
type GhComment = {
  path?: string;
  diff_hunk?: string;
  body?: string;
  html_url?: string;
  created_at?: string;
  user?: GhUser;
};
type Config = { pr: number; owner: string; repo: string; path?: string; autoOpen?: boolean };
type Anchor = { map: Array<{ node: Text; local: number }>; start: number; end: number };
type Group = { quote: string; prefix: string; comments: GhComment[] };

const STYLE_ID = "dep-pr-styles";
const BLOCK_TAGS: Record<string, boolean> = {
  P: true, LI: true, BLOCKQUOTE: true, H1: true, H2: true, H3: true, H4: true,
  H5: true, H6: true, TD: true, TH: true, DD: true, DT: true, FIGCAPTION: true, PRE: true,
};

function esc(s: unknown): string {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function normWs(s: unknown): string {
  return String(s == null ? "" : s).replace(/\s+/g, " ").trim();
}

/* Strip inline markdown so a SOURCE line matches the RENDERED text. */
function stripMarkdown(line: string): string {
  let s = String(line || "");
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

/* Commented line = last content line of the diff hunk; prior line = prefix hint. */
function selectorFromComment(c: GhComment): { quote: string; prefix: string } {
  const lines = String(c.diff_hunk || "").split("\n");
  let content: string[] = [];
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    if (ln.indexOf("@@") === 0) { content = []; continue; }
    if (ln === "") continue;
    const marker = ln.charAt(0);
    const text = (marker === "+" || marker === "-" || marker === " ") ? ln.slice(1) : ln;
    const stripped = stripMarkdown(text);
    if (stripped) content.push(stripped);
  }
  return {
    quote: content.length ? content[content.length - 1] : "",
    prefix: content.length > 1 ? content[content.length - 2] : "",
  };
}

function resolveRoot(rootSelector?: string): Element {
  const candidates = ([] as string[]).concat(
    rootSelector ? [rootSelector] : [],
    [".fern-prose", ".prose", "article", "main", "[role=main]"]
  );
  for (let i = 0; i < candidates.length; i++) {
    const el = document.querySelector(candidates[i]);
    if (el && normWs(el.textContent).length > 120) return el;
  }
  const sec = document.getElementById("dep-pr-comments");
  const up = sec ? sec.closest("article,main,section") : null;
  return up || document.body;
}

/* Visible text nodes under root, skipping our own UI + code/script/style. */
function collectTextNodes(root: Element): Text[] {
  const nodes: Text[] = [];
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n: Node): number {
      if (!n.nodeValue || !n.nodeValue.replace(/\s+/g, "")) return NodeFilter.FILTER_REJECT;
      let p: Node | null = n.parentNode;
      while (p && p !== root.parentNode) {
        const el = p as Element;
        const tag = (el.nodeName || "").toUpperCase();
        if (tag === "SCRIPT" || tag === "STYLE" || tag === "MARK") return NodeFilter.FILTER_REJECT;
        if (el.classList && el.classList.contains("dep-pr-ui")) return NodeFilter.FILTER_REJECT;
        if (el.id === "dep-pr-comments") return NodeFilter.FILTER_REJECT;
        p = p.parentNode;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  let node: Node | null;
  while ((node = walker.nextNode())) nodes.push(node as Text);
  return nodes;
}

/* Disambiguate repeated matches with a normalized preceding-text prefix. */
function pickByPrefix(raw: string, indices: number[], prefix: string): number {
  if (indices.length === 1 || !prefix) return indices[0];
  const np = normWs(prefix);
  if (!np) return indices[0];
  for (let i = 0; i < indices.length; i++) {
    const before = normWs(raw.slice(Math.max(0, indices[i] - np.length - 16), indices[i]));
    if (before.slice(-np.length) === np) return indices[i];
  }
  return indices[0];
}

/* Locate quote across text nodes: exact (prefix-disambiguated), then fuzzy. */
function locateQuote(root: Element, quote: string, prefix: string): Anchor | null {
  const nodes = collectTextNodes(root);
  let raw = "";
  const map: Array<{ node: Text; local: number }> = [];
  for (let i = 0; i < nodes.length; i++) {
    const v = nodes[i].nodeValue || "";
    for (let j = 0; j < v.length; j++) map.push({ node: nodes[i], local: j });
    raw += v;
  }
  const indices: number[] = [];
  let from = 0;
  let hit: number;
  while ((hit = raw.indexOf(quote, from)) !== -1) { indices.push(hit); from = hit + Math.max(1, quote.length); }
  if (indices.length) {
    const idx = pickByPrefix(raw, indices, prefix);
    return { map, start: idx, end: idx + quote.length };
  }
  let norm = "";
  const n2r: number[] = [];
  let prevWs = false;
  for (let k = 0; k < raw.length; k++) {
    const ch = raw.charAt(k);
    if (/\s/.test(ch)) { if (prevWs) continue; norm += " "; n2r.push(k); prevWs = true; }
    else { norm += ch; n2r.push(k); prevWs = false; }
  }
  const nq = normWs(quote);
  if (!nq) return null;
  const nidx = norm.indexOf(nq);
  if (nidx === -1) return null;
  const rawStart = n2r[nidx];
  const rawEnd = n2r[Math.min(nidx + nq.length - 1, n2r.length - 1)] + 1;
  return { map, start: rawStart, end: rawEnd };
}

/* Wrap [start,end) in <mark>, splitting across text-node boundaries as needed. */
function wrapRange(loc: Anchor, groupId: string): HTMLElement[] {
  const { map, start, end } = loc;
  const segs: Array<{ node: Text; lo: number; hi: number }> = [];
  let cur: { node: Text; lo: number; hi: number } | null = null;
  for (let i = start; i < end; i++) {
    const m = map[i];
    if (!cur || cur.node !== m.node) { cur = { node: m.node, lo: m.local, hi: m.local + 1 }; segs.push(cur); }
    else cur.hi = m.local + 1;
  }
  const marks: HTMLElement[] = [];
  for (let s = 0; s < segs.length; s++) {
    const seg = segs[s];
    let target: Text = seg.node;
    if (!target.parentNode) continue;
    if (seg.lo > 0) target = target.splitText(seg.lo);
    if (seg.hi - seg.lo < (target.nodeValue || "").length) target.splitText(seg.hi - seg.lo);
    const mk = document.createElement("mark");
    mk.className = "dep-pr-mark";
    mk.setAttribute("data-dep-pr-group", groupId);
    target.parentNode!.insertBefore(mk, target);
    mk.appendChild(target);
    marks.push(mk);
  }
  return marks;
}

function blockAncestor(node: Node, root: Element): Element {
  let el: Node | null = node.nodeType === 3 ? node.parentNode : node;
  while (el && el !== root) {
    if (BLOCK_TAGS[(el.nodeName || "").toUpperCase()]) return el as Element;
    el = el.parentNode;
  }
  return (node.nodeType === 3 ? node.parentNode : node) as Element;
}

function fmtDate(iso?: string): string {
  try { return new Date(iso || "").toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }); }
  catch { return iso || ""; }
}

function commentHtml(c: GhComment): string {
  const login = (c.user && c.user.login) || "unknown";
  const avatar = (c.user && c.user.avatar_url) || "";
  const body = String(c.body || "").split(/\n{2,}/)
    .map((p) => "<p>" + esc(p).replace(/\n/g, "<br>") + "</p>").join("");
  const av = avatar ? '<img class="dep-pr-avatar" src="' + esc(avatar) + '" alt="" width="20" height="20">' : "";
  return '<div class="dep-pr-comment">' +
    '<div class="dep-pr-comment-head">' + av +
    '<span class="dep-pr-author">' + esc(login) + '</span>' +
    '<span class="dep-pr-date">' + esc(fmtDate(c.created_at)) + '</span></div>' +
    '<div class="dep-pr-body">' + body + '</div>' +
    '<a class="dep-pr-link" href="' + esc(c.html_url) + '" target="_blank" rel="noopener">View / reply on GitHub &rarr;</a>' +
    '</div>';
}

function ensureStyles(): void {
  if (document.getElementById(STYLE_ID)) return;
  const css =
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
  const st = document.createElement("style");
  st.id = STYLE_ID;
  st.appendChild(document.createTextNode(css));
  document.head.appendChild(st);
}

/* Undo a previous render so effect re-runs / SPA nav don't stack. */
function cleanup(): void {
  const ui = document.querySelectorAll(".dep-pr-ui");
  for (let i = 0; i < ui.length; i++) if (ui[i].parentNode) ui[i].parentNode!.removeChild(ui[i]);
  const marks = document.querySelectorAll("mark.dep-pr-mark");
  for (let j = 0; j < marks.length; j++) {
    const mk = marks[j];
    const p = mk.parentNode;
    if (!p) continue;
    while (mk.firstChild) p.insertBefore(mk.firstChild, mk);
    p.removeChild(mk);
    if ((p as Element).normalize) (p as Element).normalize();
  }
}

function toggleCard(card: HTMLElement, badge: HTMLElement): void {
  const open = card.style.display !== "none";
  card.style.display = open ? "none" : "";
  badge.setAttribute("aria-expanded", open ? "false" : "true");
}

function run(cfg: Config, root: Element, allComments: GhComment[]): void {
  ensureStyles();
  cleanup();
  const section = document.getElementById("dep-pr-comments");
  const prUrl = "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + cfg.pr;
  const comments = (allComments || []).filter((c) => !cfg.path || c.path === cfg.path);

  const groups: Group[] = [];
  const byQuote: Record<string, Group> = {};
  comments.forEach((c) => {
    const sel = selectorFromComment(c);
    const key = sel.quote + "\u0000" + sel.prefix;
    if (!byQuote[key]) { byQuote[key] = { quote: sel.quote, prefix: sel.prefix, comments: [] }; groups.push(byQuote[key]); }
    byQuote[key].comments.push(c);
  });

  let anchored = 0;
  const unanchored: Group[] = [];
  groups.forEach((g, gi) => {
    const loc = g.quote ? locateQuote(root, g.quote, g.prefix) : null;
    if (!loc) { unanchored.push(g); return; }
    const marks = wrapRange(loc, "g" + gi);
    if (!marks.length) { unanchored.push(g); return; }
    anchored++;

    const badge = document.createElement("button");
    badge.className = "dep-pr-badge dep-pr-ui";
    badge.type = "button";
    badge.setAttribute("aria-expanded", "false");
    badge.innerHTML = "\uD83D\uDCAC " + g.comments.length;
    marks[marks.length - 1].insertAdjacentElement("afterend", badge);

    const card = document.createElement("div");
    card.className = "dep-pr-card dep-pr-ui";
    card.style.display = cfg.autoOpen ? "" : "none";
    card.innerHTML = '<div class="dep-pr-card-eyebrow">PR #' + esc(cfg.pr) + " review \u00B7 line comment</div>" +
      g.comments.map(commentHtml).join("");
    const block = blockAncestor(marks[0], root);
    block.insertAdjacentElement("afterend", card);

    badge.addEventListener("click", () => toggleCard(card, badge));
  });

  if (!section) return;
  const total = comments.length;
  const head =
    '<p class="dep-pr-heading">Inline review comments</p>' +
    '<p class="dep-pr-note">Read-only mirror of the ' + total + " review comment" + (total === 1 ? "" : "s") +
    ' on this file from <a href="' + esc(prUrl) + '" target="_blank" rel="noopener">PR #' + esc(cfg.pr) + "</a>" +
    " (" + anchored + " anchored to text above). Authoring stays on GitHub &mdash; " +
    '<a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener">discuss inline on the PR</a>.</p>';
  let un = "";
  if (unanchored.length) {
    un = '<div class="dep-pr-unanchored"><h4>Comments not anchored to current text (' + unanchored.length + ")</h4>" +
      '<p class="dep-pr-note">The commented line was edited or could not be located in the rendered page. Shown here so nothing is lost.</p><ul>' +
      unanchored.map((g) => g.comments.map((c) =>
        '<li><span class="dep-pr-quote">&ldquo;' + esc(g.quote || "(unknown line)") + "&rdquo;</span> &mdash; " +
        esc((c.user && c.user.login) || "unknown") + ": " + esc(normWs(c.body).slice(0, 140)) +
        ' <a class="dep-pr-link" href="' + esc(c.html_url) + '" target="_blank" rel="noopener">on GitHub &rarr;</a></li>'
      ).join("")).join("") + "</ul></div>";
  }
  section.innerHTML = head + un;
}

function renderError(cfg: Config, err: Error): void {
  ensureStyles();
  const section = document.getElementById("dep-pr-comments");
  if (!section) return;
  const prUrl = "https://github.com/" + cfg.owner + "/" + cfg.repo + "/pull/" + cfg.pr;
  const msg = err && /\b403\b/.test(String(err.message))
    ? "GitHub's unauthenticated API rate limit (60/hr per IP) was hit."
    : "Could not load PR comments.";
  section.innerHTML = '<p class="dep-pr-heading">Inline review comments</p>' +
    '<p class="dep-pr-note">' + esc(msg) + " View them directly on " +
    '<a href="' + esc(prUrl) + '/files" target="_blank" rel="noopener">PR #' + esc(cfg.pr) + "</a>.</p>";
}

function fetchComments(cfg: Config): Promise<GhComment[]> {
  const url = "https://api.github.com/repos/" + cfg.owner + "/" + cfg.repo + "/pulls/" + cfg.pr + "/comments?per_page=100";
  return fetch(url, { headers: { Accept: "application/vnd.github+json" } }).then((r) => {
    if (!r.ok) throw new Error("GitHub API " + r.status);
    return r.json();
  });
}

function boot(cfg: Config): void {
  if (typeof window === "undefined" || typeof document === "undefined") return;
  let tries = 0;
  const attempt = () => {
    const root = resolveRoot();
    if ((!root || normWs(root.textContent).length < 60) && tries++ < 40) { setTimeout(attempt, 250); return; }
    fetchComments(cfg).then((cs) => run(cfg, root, cs)).catch((e: Error) => renderError(cfg, e));
  };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", attempt);
  else attempt();
}

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

export function PrInlineComments({
  pr,
  owner = "ai-dynamo",
  repo = "dynamo",
  path = "docs/proposals/0000-example-dep.mdx",
}: PrInlineCommentsProps) {
  useEffect(() => {
    boot({ pr, owner, repo, path });
  }, [pr, owner, repo, path]);

  return <section id="dep-pr-comments" />;
}
