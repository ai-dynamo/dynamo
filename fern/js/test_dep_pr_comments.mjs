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
 * Extract a top-level `function <name>(...) { ... }` block from the runtime
 * source and eval it into an isolated scope, so tests can call it directly.
 * Matches functions that live at the IIFE indent level.
 */
function extractFn(name) {
  // Non-greedy body match up to the matching close-brace at column 0 of the
  // indented IIFE. We locate the function declaration and count braces.
  const marker = `function ${name}(`;
  const start = SRC.indexOf(marker);
  if (start === -1) throw new Error(`function ${name} not found in source`);
  let depth = 0;
  let inFn = false;
  let i = start;
  let braceOpen = -1;
  while (i < SRC.length) {
    const ch = SRC[i];
    if (ch === "{") {
      if (!inFn) {
        inFn = true;
        braceOpen = i;
      }
      depth++;
    } else if (ch === "}") {
      depth--;
      if (inFn && depth === 0) {
        const body = SRC.slice(start, i + 1);
        // eslint-disable-next-line no-new-func
        return new Function(`${body}; return ${name};`)();
      }
    }
    i++;
  }
  throw new Error(`unterminated function body for ${name}`);
}

const sanitizeHtml = extractFn("sanitizeHtml");
const buildBlockquote = extractFn("buildBlockquote");
const buildGitHubUrl = extractFn("buildGitHubUrl");

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
console.log(`${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
