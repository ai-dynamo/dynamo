/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tests for the pure-string helpers in fern/js/dep-pr-comments.js.
 *
 * dep-pr-comments.js is a browser runtime and cannot be loaded directly by
 * node (references `window`, `document`). This test file extracts the
 * function bodies of the string-only helpers (`sanitizeHtml`) by regex and
 * evaluates them in isolation, so we can exercise them under `node
 * fern/js/test_dep_pr_comments.mjs` without pulling in jsdom.
 *
 * We test the sanitizer — which is the only new escape-hatch introduced
 * for GitHub's server-rendered `body_html`. Anchoring / badge / bot filter
 * behavior is unchanged and is covered by manual + preview verification.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(HERE, "dep-pr-comments.js"), "utf8");

/**
 * Extract the raw source of a top-level `function <name>(...) { ... }` block
 * from the runtime source. Matches functions declared at the IIFE indent
 * level. Returns just the source string; caller decides how to eval it.
 */
function extractFnSource(name) {
  const marker = `function ${name}(`;
  const start = SRC.indexOf(marker);
  if (start === -1) throw new Error(`function ${name} not found in source`);
  let depth = 0;
  let inFn = false;
  let i = start;
  while (i < SRC.length) {
    const ch = SRC[i];
    if (ch === "{") {
      if (!inFn) inFn = true;
      depth++;
    } else if (ch === "}") {
      depth--;
      if (inFn && depth === 0) return SRC.slice(start, i + 1);
    }
    i++;
  }
  throw new Error(`unterminated function body for ${name}`);
}

/**
 * Eval one runtime helper in isolation for unit-testing. Extracts the named
 * function body from the source and returns it, so tests can call it
 * directly without pulling in `window`/`document`.
 */
function extractFn(name) {
  const body = extractFnSource(name);
  // eslint-disable-next-line no-new-func
  return new Function(`${body}; return ${name};`)();
}

/**
 * Bundle-extract helpers: eval multiple named function declarations in one
 * Function scope and return the last one. Use this when a helper depends on
 * sibling helpers at module scope (e.g. findSourceLine reuses `normWs` +
 * `stripMarkdown`) — reproducing those dependencies inside the isolated
 * eval keeps the extracted helper self-contained.
 */
function extractBundle(names) {
  const bodies = names.map(extractFnSource).join("\n");
  const last = names[names.length - 1];
  // eslint-disable-next-line no-new-func
  return new Function(`${bodies}\nreturn ${last};`)();
}

const sanitizeHtml = extractFn("sanitizeHtml");
const buildBlockquote = extractFn("buildBlockquote");
const buildGitHubUrl = extractFn("buildGitHubUrl");
const buildLineAnchor = extractFn("buildLineAnchor");
const findSourceLine = extractBundle(["normWs", "stripMarkdown", "findSourceLine"]);
const sha256Hex = extractFn("sha256Hex");

let passes = 0;
let fails = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`  ok  ${name}`);
    passes++;
  } catch (err) {
    console.log(`  FAIL ${name}`);
    console.log(`       ${err.message}`);
    fails++;
  }
}

// Async twin: awaits the test body so a rejected Promise counts as a
// failure (the sync `test()` would silently pass because the rejection
// escapes the try/catch). Callers must `await testAsync(...)`.
async function testAsync(name, fn) {
  try {
    await fn();
    console.log(`  ok  ${name}`);
    passes++;
  } catch (err) {
    console.log(`  FAIL ${name}`);
    console.log(`       ${err.message}`);
    fails++;
  }
}

console.log("sanitizeHtml — strips dangerous constructs, preserves markdown output");

test("plain paragraph passes through unchanged", () => {
  const html = '<p dir="auto">hello <strong>world</strong></p>';
  assert.equal(sanitizeHtml(html), html);
});

test("strips <script> tags with body", () => {
  const html = '<p>keep</p><script>alert(1)</script><p>me</p>';
  const out = sanitizeHtml(html);
  assert.ok(!/script/i.test(out), `unexpected script tag: ${out}`);
  assert.ok(out.includes("<p>keep</p>") && out.includes("<p>me</p>"));
});

test("strips <iframe>, <object>, <embed>, <style>, <link>, <meta>, <base>", () => {
  const html =
    "<iframe src='x'></iframe><object></object><embed src='x'>" +
    "<style>body{}</style><link rel='x'><meta charset='utf-8'><base href='/'>";
  const out = sanitizeHtml(html);
  for (const bad of ["iframe", "object", "embed", "style", "link", "meta", "base"]) {
    assert.ok(!new RegExp(`<${bad}\\b`, "i").test(out), `${bad} survived: ${out}`);
  }
});

test("strips inline event-handler attributes (on*)", () => {
  const html = '<a href="/x" onclick="alert(1)" onmouseover=\'x\'>c</a>';
  const out = sanitizeHtml(html);
  assert.ok(!/\bon(click|mouseover|error|load)\s*=/i.test(out), `on* survived: ${out}`);
  assert.ok(out.includes('href="/x"'));
});

test("neutralizes javascript: URIs on href/src", () => {
  const html = '<a href="javascript:alert(1)">c</a><img src="JavaScript:x">';
  const out = sanitizeHtml(html);
  assert.ok(!/javascript:/i.test(out), `js scheme survived: ${out}`);
});

test("neutralizes data: URIs on href", () => {
  const html = '<a href="data:text/html,<script>alert(1)</script>">c</a>';
  const out = sanitizeHtml(html);
  assert.ok(!/href\s*=\s*["']?data:/i.test(out), `data URI survived: ${out}`);
});

test("keeps normal http/https/mailto/relative hrefs", () => {
  const html =
    '<a href="https://github.com/x">g</a>' +
    '<a href="/relative">r</a>' +
    '<a href="mailto:x@y.z">m</a>';
  const out = sanitizeHtml(html);
  assert.ok(out.includes("https://github.com/x"));
  assert.ok(out.includes('href="/relative"'));
  assert.ok(out.includes("mailto:x@y.z"));
});

test("preserves GitHub-style code blocks and inline code", () => {
  const html =
    '<p><code class="notranslate">Vec&lt;u8&gt;</code></p>' +
    '<div class="highlight highlight-source-rust">' +
    '<pre>fn main() {}</pre></div>';
  const out = sanitizeHtml(html);
  assert.ok(out.includes('<code class="notranslate">Vec&lt;u8&gt;</code>'));
  assert.ok(out.includes("<pre>fn main() {}</pre>"));
});

test("preserves GitHub blockquotes and lists", () => {
  const html =
    "<blockquote><p>quoted</p></blockquote>" +
    '<ol dir="auto"><li>one</li><li>two</li></ol>' +
    "<ul><li>a</li></ul>";
  const out = sanitizeHtml(html);
  assert.ok(out.includes("<blockquote>"));
  assert.ok(out.includes("<ol"));
  assert.ok(out.includes("<ul>"));
  assert.ok(out.includes("<li>one</li>"));
});

test("empty / null / undefined input returns empty string", () => {
  assert.equal(sanitizeHtml(""), "");
  assert.equal(sanitizeHtml(null), "");
  assert.equal(sanitizeHtml(undefined), "");
});

test("real ryanolson comment #7 with inline code survives", () => {
  const html =
    '<p dir="auto">note that it is also possible to request specific protocol to be used via ' +
    '<code class="notranslate">params-&gt;flags</code> in ' +
    '<code class="notranslate">ucp_am_send_nbx</code></p>';
  const out = sanitizeHtml(html);
  assert.ok(out.includes("params-&gt;flags"));
  assert.ok(out.includes("ucp_am_send_nbx"));
  assert.ok(out.includes("<code"));
});

test("real ryanolson comment #59 with blockquote + link survives", () => {
  const html =
    '<p dir="auto">That\'s not what the manual says.</p>' +
    '<p dir="auto"><a href="https://github.com/ai-dynamo/enhancements/blob/main/deps/0000-dep-process.md#proposal-process">' +
    "https://github.com/ai-dynamo/enhancements/blob/main/deps/0000-dep-process.md#proposal-process</a></p>" +
    "<blockquote><p dir=\"auto\">Copy the</p></blockquote>";
  const out = sanitizeHtml(html);
  assert.ok(out.includes("https://github.com/ai-dynamo/enhancements/blob/main/deps/0000-dep-process.md"));
  assert.ok(out.includes("<blockquote>"));
  assert.ok(!/on\w+\s*=/i.test(out));
});

test("SVG and MathML tags are stripped defensively", () => {
  const html = '<svg onload="alert(1)"><g></g></svg><math></math><p>ok</p>';
  const out = sanitizeHtml(html);
  assert.ok(!/<svg\b/i.test(out));
  assert.ok(!/<math\b/i.test(out));
  assert.ok(out.includes("<p>ok</p>"));
});

// ----- audit-driven regression cases (quality-gate follow-ups) ----- //

test("neutralizes UNQUOTED javascript:/data:/vbscript: URIs on href/src", () => {
  // HTML5 allows unquoted attribute values. Prior sanitizer only matched
  // quoted URIs; unquoted attack shapes slipped through the string pass.
  // hardenLinks catches this at DOM time for <a>, but the string pass is
  // the first defense and must be tight for other elements (img/track/source).
  const cases = [
    '<a href=javascript:alert(1)>c</a>',
    '<img src=javascript:alert(1)>',
    '<a href=data:text/html,x>c</a>',
    '<a href=vbscript:msgbox>c</a>',
  ];
  for (const html of cases) {
    const out = sanitizeHtml(html);
    assert.ok(
      !/href\s*=\s*[^"'\s>]*(?:javascript|data|vbscript)\s*:/i.test(out),
      `unquoted URI survived: ${html} -> ${out}`
    );
    assert.ok(
      !/src\s*=\s*[^"'\s>]*(?:javascript|data|vbscript)\s*:/i.test(out),
      `unquoted URI survived: ${html} -> ${out}`
    );
  }
});

test("neutralizes URIs with leading whitespace before scheme", () => {
  const html = '<a href="   javascript:alert(1)">c</a>';
  const out = sanitizeHtml(html);
  assert.ok(!/javascript:/i.test(out), `whitespace-prefixed js: survived: ${out}`);
});

console.log("");
console.log("buildBlockquote — turn selected text into a markdown quote");

test("prefixes single line with '> '", () => {
  assert.equal(buildBlockquote("hello world"), "> hello world");
});

test("prefixes each line of multi-line text", () => {
  assert.equal(
    buildBlockquote("line1\nline2\nline3"),
    "> line1\n> line2\n> line3"
  );
});

test("empty / whitespace-only input yields empty string", () => {
  assert.equal(buildBlockquote(""), "");
  assert.equal(buildBlockquote("   \n\n  "), "");
  assert.equal(buildBlockquote(null), "");
});

test("normalizes CRLF and stray CR to LF before prefixing", () => {
  // Selections that cross Windows-line-endings or copy in a stray \r must
  // not produce "> \r" runs.
  assert.equal(
    buildBlockquote("a\r\nb\rc"),
    "> a\n> b\n> c"
  );
});

test("collapses runs of blank lines into a single quoted blank", () => {
  // A selection that spans a paragraph break has multiple blank lines;
  // GitHub renders "> \n> \n> " as several quoted blanks. Collapse to one.
  const out = buildBlockquote("para1\n\n\n\npara2");
  assert.equal(out, "> para1\n>\n> para2");
});

test("truncates over the 1500-char sensible limit with a marker", () => {
  const big = "x".repeat(3000);
  const out = buildBlockquote(big);
  assert.ok(out.length < 1700, `over-cap: ${out.length}`);
  assert.ok(out.endsWith("…"), `no ellipsis: ${out.slice(-10)}`);
});

test("does not inject markdown or HTML into the quote content", () => {
  // Quote content is opaque plain text — HTML/markdown metacharacters
  // in the selection must not turn into structure once pasted.
  const out = buildBlockquote("<script>alert(1)</script>\n**bold**");
  // The `<` still leads its line but is inside a quote line — GitHub
  // renders it literally. What we DO NOT want is a stripped tag; assert
  // the original characters survive as text.
  assert.ok(out.includes("<script>"));
  assert.ok(out.includes("**bold**"));
});

console.log("");
console.log("buildGitHubUrl — deep-link target from mount config");

test("prefers PR /files view for line-level comments when pr is set", () => {
  const url = buildGitHubUrl({
    owner: "ai-dynamo",
    repo: "enhancements",
    pr: 61,
  });
  assert.equal(
    url,
    "https://github.com/ai-dynamo/enhancements/pull/61/files"
  );
});

test("prefers PR /files even when both pr and issue are set", () => {
  // Line-level comments require the diff view; the issue is the fallback,
  // not an alternative when the PR exists.
  const url = buildGitHubUrl({
    owner: "ai-dynamo",
    repo: "enhancements",
    pr: 61,
    issue: 999,
  });
  assert.ok(url.endsWith("/pull/61/files"), url);
});

test("falls back to the tracking issue when no pr is set", () => {
  const url = buildGitHubUrl({
    owner: "ai-dynamo",
    repo: "enhancements",
    issue: 42,
  });
  assert.equal(url, "https://github.com/ai-dynamo/enhancements/issues/42");
});

test("returns empty string when neither pr nor issue is set", () => {
  // The caller uses "" as a signal to hide the button — there's no
  // meaningful GitHub target if the mount div has no pr and no issue.
  assert.equal(buildGitHubUrl({ owner: "ai-dynamo", repo: "dynamo" }), "");
});

test("returns empty string when owner or repo is missing", () => {
  assert.equal(buildGitHubUrl({ repo: "dynamo", pr: 1 }), "");
  assert.equal(buildGitHubUrl({ owner: "ai-dynamo", pr: 1 }), "");
  assert.equal(buildGitHubUrl({}), "");
});

console.log("");
console.log("buildLineAnchor — GitHub diff-file line fragment");

// The DEP-adding PR file is `deps/0000-nova.md`. Its sha256 is used
// as the per-file diff container id on GitHub's Files-changed view.
// Verified empirically against the live PR #61 DOM.
const NOVA_HASH =
  "d7997b07b820f51651e7ee53b7f55c6f5319495344ff6c6f1cf9d75ad114744d";

test("formats a right-side line anchor with default side", () => {
  assert.equal(
    buildLineAnchor(NOVA_HASH, 42),
    `#diff-${NOVA_HASH}R42`
  );
});

test("formats a right-side line anchor with explicit side='R'", () => {
  assert.equal(
    buildLineAnchor(NOVA_HASH, 1, "R"),
    `#diff-${NOVA_HASH}R1`
  );
});

test("formats a left-side line anchor when side='L'", () => {
  // For a DEP-adding PR every line is an addition (R), but the helper
  // has to be symmetrical for future use on non-add-only patches.
  assert.equal(
    buildLineAnchor(NOVA_HASH, 12, "L"),
    `#diff-${NOVA_HASH}L12`
  );
});

test("rejects empty hash", () => {
  assert.equal(buildLineAnchor("", 5), "");
  assert.equal(buildLineAnchor(null, 5), "");
});

test("rejects non-positive line numbers", () => {
  assert.equal(buildLineAnchor(NOVA_HASH, 0), "");
  assert.equal(buildLineAnchor(NOVA_HASH, -1), "");
  assert.equal(buildLineAnchor(NOVA_HASH, null), "");
});

test("rejects unknown side markers", () => {
  // Guard so a caller passing side="foo" doesn't produce a broken URL
  // silently. Fall back to empty and let the outer caller use the
  // no-anchor fallback URL.
  assert.equal(buildLineAnchor(NOVA_HASH, 5, "X"), "");
});

console.log("");
console.log("findSourceLine — locate a rendered selection in the DEP source");

// Real Nova DEP source excerpt (verified by curl against the head ref of
// ai-dynamo/enhancements#61). Bold-key metadata + a section body.
const NOVA_SRC = [
  "# Nova: Active Messaging as a Foundational Network Primitive",
  "",
  "**Status**: Draft",
  "",
  "**Authors**: [@ryanolson](https://github.com/ryanolson)",
  "",
  "**Category**: Architecture",
  "",
  "# Summary",
  "",
  "Nova provides a transport-agnostic active messaging layer that serves as the foundational network primitive for dynamo.",
  "",
];

test("returns 1-based line number for an exact match", () => {
  assert.equal(
    findSourceLine(NOVA_SRC, "Category: Architecture"),
    7
  );
});

test("normalizes whitespace: extra spaces in the selection still match", () => {
  // Reader selected a line whose whitespace got mangled by the DOM (e.g.
  // rendered with two spaces where source had one). Normalization
  // collapses runs of whitespace so the contains-match still hits.
  assert.equal(
    findSourceLine(NOVA_SRC, "  Nova provides  a  transport-agnostic  active"),
    11
  );
});

test("strips bold markdown so `**Status**: Draft` matches selection `Status: Draft`", () => {
  // The selection is rendered prose (no asterisks); the source line has
  // bold-key syntax. findSourceLine must strip the markdown before
  // comparing, otherwise every metadata row would be un-anchorable.
  assert.equal(findSourceLine(NOVA_SRC, "Status: Draft"), 3);
});

test("matches on the first ~60 chars of the selection so long quotes still resolve", () => {
  // Selections often span more text than a single source line — e.g. a
  // whole paragraph. Anchor on the START of the selection so the URL
  // points at the paragraph's first line.
  const longSelection =
    "Nova provides a transport-agnostic active messaging layer that serves " +
    "as the foundational network primitive for dynamo. Second sentence lives in the rendered DOM only.";
  assert.equal(findSourceLine(NOVA_SRC, longSelection), 11);
});

test("returns null when the selection cannot be located in the source", () => {
  assert.equal(
    findSourceLine(NOVA_SRC, "This text does not appear anywhere in the source"),
    null
  );
});

test("returns null on empty / null / whitespace-only inputs", () => {
  assert.equal(findSourceLine([], "hello"), null);
  assert.equal(findSourceLine(null, "hello"), null);
  assert.equal(findSourceLine(NOVA_SRC, ""), null);
  assert.equal(findSourceLine(NOVA_SRC, "   \n  "), null);
});

test("skips blank source lines (does not return a line number for empty match)", () => {
  const src = ["", "", "actual content", ""];
  // The selection is a substring — should hit line 3 (1-based), not line
  // 1 (which is empty).
  assert.equal(findSourceLine(src, "actual"), 3);
});

test("returns null when the needle matches multiple non-blank source lines (ambiguous)", () => {
  // Two source lines share the same 60-char prefix. A confidently-wrong
  // anchor (matching the first hit) is a worse UX than the plain
  // top-of-file fallback — the caller expects null so it uses the
  // no-anchor URL instead of guessing.
  const dupSrc = [
    "Nova provides a transport-agnostic active messaging layer.",
    "",
    "Some intervening line that will not match.",
    "",
    "Nova provides a transport-agnostic active messaging layer.",
  ];
  assert.equal(
    findSourceLine(dupSrc, "Nova provides a transport-agnostic active messaging layer."),
    null
  );
});

test("still returns a line when only one non-blank source line matches", () => {
  // Guard against the ambiguity check false-positiving on blank lines
  // between the sole match and unrelated content.
  const src = [
    "",
    "Nova provides a transport-agnostic active messaging layer.",
    "",
    "Some other paragraph.",
  ];
  assert.equal(
    findSourceLine(src, "Nova provides a transport-agnostic active messaging layer."),
    2
  );
});

console.log("");
console.log("sha256Hex — hex-encoded SHA-256, used for the diff-file hash");

await testAsync("computes the expected hash for the Nova DEP path", async () => {
  // Verified against the live PR #61 DOM: the file container id is
  // `diff-<sha256hex('deps/0000-nova.md')>`.
  const hex = await sha256Hex("deps/0000-nova.md");
  assert.equal(hex, NOVA_HASH);
  assert.equal(hex.length, 64);
});

await testAsync("computes the empty-string SHA-256 vector", async () => {
  // NIST test vector: sha256("") == e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
  const hex = await sha256Hex("");
  assert.equal(
    hex,
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  );
});

await testAsync("returns null when crypto.subtle is unavailable", async () => {
  // Simulate the "web crypto missing" edge case (older browser, HTTP
  // context, insecure iframe). We can't unset globalThis.crypto without
  // breaking other tests, so eval an isolated copy with no crypto in
  // scope — mirrors what the runtime sees when it must fall back.
  const src = extractFnSource("sha256Hex");
  const noCrypto = new Function(
    "crypto",
    `${src}; return sha256Hex;`
  )(undefined);
  const result = await noCrypto("anything");
  assert.equal(result, null);
});

console.log("");
console.log("onQuoteBtnClick — popup-blocker discipline");

test("opens the tab synchronously before awaiting Promise.all (popup-blocker fix)", () => {
  // The transient user-gesture activation Chrome / Safari require for
  // `window.open` is CONSUMED at the first microtask boundary after a
  // fetched Promise resolves. That means any `window.open()` call
  // inside `Promise.all([...]).then(...)` after `resolveLineAnchoredUrl`
  // (which awaits two network fetches) gets popup-blocked.
  //
  // Structural check: `window.open(` must appear inside the
  // synchronous body of `onQuoteBtnClick` BEFORE the `Promise.all(`
  // call, so the fallback tab is opened while the gesture is still
  // live. The line-anchored URL is applied via `opened.location`
  // later; it never gates the initial open.
  const src = extractFnSource("onQuoteBtnClick");
  const openIdx = src.indexOf("window.open(");
  const promiseAllIdx = src.indexOf("Promise.all(");
  assert.notEqual(openIdx, -1, "onQuoteBtnClick must call window.open");
  assert.notEqual(promiseAllIdx, -1, "onQuoteBtnClick must await Promise.all");
  assert.ok(
    openIdx < promiseAllIdx,
    "window.open must be called BEFORE Promise.all to preserve the " +
    `user-gesture window (open at ${openIdx}, Promise.all at ${promiseAllIdx})`
  );
});

test("cold-path retained-handle open does not pass 'noopener' and pairs with `opened.opener = null`", () => {
  // Cold path: when the pre-warm memo hasn't resolved a line URL for
  // the current selection, the click handler falls back to opening
  // the fallback URL sync, retaining the tab handle so it can be
  // upgraded via `opened.location.replace(...)` once
  // `resolveLineAnchoredUrl` returns. That retention requires
  // OMITTING `noopener` from the feature string on THAT specific
  // open, and compensating with a `opened.opener = null` write to
  // sever the reverse-tabnabbing edge. The warm-path open is
  // separate and fully-hardened — see the "warm-path" tests below.
  const src = extractFnSource("onQuoteBtnClick");
  // Locate the cold-path open by the `opened =` assignment prefix —
  // it's the only `window.open` whose return value gets stored.
  const coldMatch = src.match(/opened\s*=\s*[^;]*window\.open\([^)]*\)/);
  assert.ok(
    coldMatch,
    "onQuoteBtnClick must have a cold-path `opened = window.open(...)` assignment"
  );
  assert.doesNotMatch(
    coldMatch[0],
    /noopener/,
    "the cold-path `opened = window.open(...)` must not pass 'noopener' — " +
    "the returned handle is required to navigate the tab to the line-anchored URL"
  );
  // Mitigation MUST be present: `opened.opener = null`.
  assert.match(
    src,
    /opened\.opener\s*=\s*null/,
    "onQuoteBtnClick must null `opened.opener` to compensate for the " +
    "missing 'noopener' feature on the cold-path open"
  );
  // Cold-path upgrade path is preserved.
  assert.match(
    src,
    /opened\.location\.replace\(/,
    "onQuoteBtnClick must still upgrade the retained tab via " +
    "`opened.location.replace(...)` when the resolver returns a line URL"
  );
});

console.log("");
console.log("onQuoteBtnClick / onDocumentMouseUp — pre-warm warm-path hardening");

test("warm-path uses fully-hardened `noopener,noreferrer` open when memo has resolved URL", () => {
  // When `onDocumentMouseUp` pre-warms `resolveLineAnchoredUrl` and
  // it settles to a truthy line-anchored URL BEFORE the click, the
  // click handler must take a fully-hardened synchronous open — no
  // retained handle, no opener exposure, no `location.replace`
  // needed. Structural check: `onQuoteBtnClick` must contain a
  // `window.open(..., "noopener,noreferrer")` call, distinct from
  // the cold-path retained-handle open asserted above.
  const src = extractFnSource("onQuoteBtnClick");
  assert.match(
    src,
    /window\.open\([^)]*"noopener,noreferrer"/,
    "onQuoteBtnClick must have a warm-path `window.open(..., " +
    '"noopener,noreferrer")` call — the strictly-preferred fully-hardened open'
  );
});

test("warm-path is keyed by normalized quote so a stale selection falls to cold path", () => {
  // The memo primed in `onDocumentMouseUp` is keyed by `normWs(text)`.
  // If the reader re-selects between mouseup and click, the memo's
  // quote no longer matches the click's `text`, and the click handler
  // must NOT open the stale URL — it must fall to the cold path.
  // Structural check: `onQuoteBtnClick` must reference `pendingAnchor`
  // AND compare via `normWs(` so the guard is in place.
  const src = extractFnSource("onQuoteBtnClick");
  assert.match(
    src,
    /pendingAnchor/,
    "onQuoteBtnClick must reference the `pendingAnchor` memo to " +
    "gate the warm path"
  );
  assert.match(
    src,
    /normWs\(/,
    "onQuoteBtnClick must key the warm-path check via `normWs(...)` " +
    "so a re-selection with different whitespace still binds correctly, " +
    "and a genuinely different selection falls to the cold path"
  );
});

test("onDocumentMouseUp pre-warms the pendingAnchor memo and calls resolveLineAnchoredUrl", () => {
  // The click's warm path is only useful if `onDocumentMouseUp`
  // actually primes the memo the moment the pill appears. Without
  // this pre-warm, every click takes the cold (opener-retained)
  // path and the fully-hardened open never fires. Structural check:
  // `onDocumentMouseUp` must set `pendingAnchor` AND fire
  // `resolveLineAnchoredUrl(` after the selection passes the
  // body-prose predicate.
  const src = extractFnSource("onDocumentMouseUp");
  assert.match(
    src,
    /pendingAnchor\s*=/,
    "onDocumentMouseUp must assign to `pendingAnchor` to seed the memo"
  );
  assert.match(
    src,
    /resolveLineAnchoredUrl\(/,
    "onDocumentMouseUp must call `resolveLineAnchoredUrl(...)` to " +
    "pre-warm the line URL while the tab handle isn't yet needed"
  );
});

console.log("");
console.log(`${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
