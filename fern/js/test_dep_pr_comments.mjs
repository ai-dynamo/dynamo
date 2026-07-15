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

console.log("");
console.log(`${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
