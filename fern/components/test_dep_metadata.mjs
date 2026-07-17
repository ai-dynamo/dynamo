/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tests for the pure-JS parsing helper in fern/components/DepMetadata.tsx.
 *
 * DepMetadata.tsx is a React server component and cannot be loaded directly by
 * node (references JSX). This test file extracts the function body of the
 * string-only helper `parseLinkedItems` by regex and evaluates it in
 * isolation, so we can exercise it under
 * `node fern/components/test_dep_metadata.mjs` without a TSX toolchain.
 *
 * `parseLinkedItems` must be authored as plain JS syntax (no TypeScript
 * annotations inside its signature or body — use JSDoc for typing). That
 * matches the extraction pattern used in fern/js/test_dep_pr_comments.mjs.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(HERE, "DepMetadata.tsx"), "utf8");

/**
 * Extract the raw source of a top-level `function <name>(...) { ... }` block
 * from the .tsx source. Matches the plain-JS-shaped helper only — the helper
 * must not use TS type annotations in its signature or body.
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

function extractFn(name) {
  const body = extractFnSource(name);
  // eslint-disable-next-line no-new-func
  return new Function(`${body}; return ${name};`)();
}

const parseLinkedItems = extractFn("parseLinkedItems");
const statusVariant = extractFn("statusVariant");

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

console.log("parseLinkedItems — turns markdown-link author/reviewer strings into structured entries");

test("single markdown link becomes one linked entry", () => {
  const out = parseLinkedItems("[@ryanolson](https://github.com/ryanolson)");
  assert.deepEqual(out, [
    { label: "@ryanolson", href: "https://github.com/ryanolson" },
  ]);
});

test("two comma-separated markdown links split into two linked entries", () => {
  const out = parseLinkedItems(
    "[@grahamking](https://github.com/grahamking), [@biswapanda](https://github.com/biswapanda)",
  );
  assert.deepEqual(out, [
    { label: "@grahamking", href: "https://github.com/grahamking" },
    { label: "@biswapanda", href: "https://github.com/biswapanda" },
  ]);
});

test("plain text without markdown syntax falls back to text entry", () => {
  const out = parseLinkedItems("Dan Gil");
  assert.deepEqual(out, [{ label: "Dan Gil", href: null }]);
});

test("mixed markdown-link and plain-text entries are both preserved", () => {
  const out = parseLinkedItems(
    "[@ryanolson](https://github.com/ryanolson), TBD",
  );
  assert.deepEqual(out, [
    { label: "@ryanolson", href: "https://github.com/ryanolson" },
    { label: "TBD", href: null },
  ]);
});

test("empty string returns empty array", () => {
  assert.deepEqual(parseLinkedItems(""), []);
});

test("whitespace-only string returns empty array", () => {
  assert.deepEqual(parseLinkedItems("   "), []);
});

test("trailing/leading whitespace on entries is trimmed", () => {
  const out = parseLinkedItems(
    "  [@a](https://github.com/a) ,  [@b](https://github.com/b)  ",
  );
  assert.deepEqual(out, [
    { label: "@a", href: "https://github.com/a" },
    { label: "@b", href: "https://github.com/b" },
  ]);
});

test("http (not https) URLs are also accepted", () => {
  const out = parseLinkedItems("[@x](http://example.com/x)");
  assert.deepEqual(out, [{ label: "@x", href: "http://example.com/x" }]);
});

test("non-http URLs are rejected (fall back to plain text)", () => {
  const out = parseLinkedItems("[@x](javascript:alert(1))");
  assert.deepEqual(out, [{ label: "[@x](javascript:alert(1))", href: null }]);
});

test("empty commas do not produce empty entries", () => {
  const out = parseLinkedItems(
    "[@a](https://github.com/a),,[@b](https://github.com/b)",
  );
  assert.deepEqual(out, [
    { label: "@a", href: "https://github.com/a" },
    { label: "@b", href: "https://github.com/b" },
  ]);
});

test("null/undefined returns empty array (component-may-omit-value case)", () => {
  assert.deepEqual(parseLinkedItems(undefined), []);
  assert.deepEqual(parseLinkedItems(null), []);
});

console.log("");
console.log(
  "statusVariant() — maps a lifecycle status to a pill color; MUST mirror variant() in fern/js/dep-status-pills.js",
);

test("Draft → draft (amber, the default lifecycle state)", () => {
  assert.equal(statusVariant("Draft"), "draft");
});

test("Under Review → proposed (blue, via /review/)", () => {
  assert.equal(statusVariant("Under Review"), "proposed");
});

test("Accepted → accepted (green)", () => {
  assert.equal(statusVariant("Accepted"), "accepted");
});

test("Implemented → accepted (green, via /implement/)", () => {
  // An Implemented DEP is a ratified, shipped decision — it renders the
  // accepted (green) pill, not the draft (amber) fallback. This branch MUST
  // stay identical to variant() in fern/js/dep-status-pills.js.
  assert.equal(statusVariant("Implemented"), "accepted");
});

test("Rejected → rejected (red)", () => {
  assert.equal(statusVariant("Rejected"), "rejected");
});

test("Deferred → muted (gray)", () => {
  assert.equal(statusVariant("Deferred"), "muted");
});

test("Replaced → muted (gray)", () => {
  assert.equal(statusVariant("Replaced"), "muted");
});

test("legacy synonyms still color correctly (Proposed/Approved)", () => {
  // The canonical enum drops "Proposed"/"Approved", but the matcher stays
  // tolerant so any DEP still carrying the old label renders the right color.
  assert.equal(statusVariant("Proposed"), "proposed");
  assert.equal(statusVariant("Approved"), "accepted");
});

test("unknown / empty status → draft (default)", () => {
  assert.equal(statusVariant("Whatever"), "draft");
  assert.equal(statusVariant(""), "draft");
  assert.equal(statusVariant(null), "draft");
  assert.equal(statusVariant(undefined), "draft");
});

console.log(`\n${passes} passed, ${fails} failed`);
process.exit(fails === 0 ? 0 : 1);
