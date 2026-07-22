/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tests for the pure helpers in fern/js/dep-index.js.
 *
 * dep-index.js is a browser runtime and cannot be loaded directly by node
 * (references `window`, `document`). This file follows the extract-and-eval
 * pattern from test_dep_status_pills.mjs: the string-only helpers (`variant`,
 * `statusRank`) are extracted by regex and evaluated in isolation. DOM
 * behaviour (the MutationObserver + per-mount idempotency flag) is exercised
 * structurally — we assert the runtime contains the required constructs — plus
 * a live preview verification captured in the PR description.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(HERE, "dep-index.js"), "utf8");

/** Extract a top-level `function <name>(...) { ... }` block. Same helper as
 * test_dep_status_pills.mjs. */
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

function extractFn(name, preamble = "") {
  const body = extractFnSource(name);
  // eslint-disable-next-line no-new-func
  return new Function(`${preamble}\n${body}\nreturn ${name};`)();
}

const variant = extractFn("variant");
// statusRank closes over the module-level STATUS_RANK array — extract + inject.
const STATUS_RANK_SRC = SRC.match(/var STATUS_RANK = (\[[\s\S]*?\]);/)[1];
const statusRank = extractFn("statusRank", `var STATUS_RANK = ${STATUS_RANK_SRC};`);

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

console.log("variant() — must mirror DepMetadata.statusVariant() verbatim");

test("Draft → draft", () => assert.equal(variant("Draft"), "draft"));
test("Under Review → proposed", () => assert.equal(variant("Under Review"), "proposed"));
test("Accepted → accepted", () => assert.equal(variant("Accepted"), "accepted"));
test("Implemented → accepted", () => assert.equal(variant("Implemented"), "accepted"));
test("Rejected → rejected", () => assert.equal(variant("Rejected"), "rejected"));
test("Deferred → muted", () => assert.equal(variant("Deferred"), "muted"));
test("Replaced → muted", () => assert.equal(variant("Replaced"), "muted"));
test("unknown → draft", () => assert.equal(variant("Whatever"), "draft"));
test("null/undefined/empty → draft", () => {
  assert.equal(variant(null), "draft");
  assert.equal(variant(undefined), "draft");
  assert.equal(variant(""), "draft");
});

console.log("");
console.log("statusRank() — lifecycle sort order, unknown sorts last");

test("Under Review ranks before Draft", () => {
  assert.ok(statusRank("Under Review") < statusRank("Draft"));
});
test("Draft ranks before Accepted", () => {
  assert.ok(statusRank("Draft") < statusRank("Accepted"));
});
test("Accepted ranks before Implemented", () => {
  assert.ok(statusRank("Accepted") < statusRank("Implemented"));
});
test("Rejected ranks last among known states", () => {
  assert.ok(statusRank("Rejected") > statusRank("Replaced"));
});
test("unknown status sorts at/after the end", () => {
  assert.ok(statusRank("Mystery") >= statusRank("Rejected"));
});

console.log("");
console.log("Runtime — structural constraints");

test("guards double-load with window.__depIndexLoaded", () => {
  assert.match(SRC, /window\.__depIndexLoaded/);
});
test("reads dataset from window.__DEP_INDEX", () => {
  assert.match(SRC, /window\.__DEP_INDEX/);
});
test("mounts into #dep-index", () => {
  assert.match(SRC, /"dep-index"/);
});
test("uses data-dep-index-ready as the per-mount idempotency flag", () => {
  assert.match(SRC, /data-dep-index-ready/);
});
test("renders text via textContent, never innerHTML", () => {
  assert.match(SRC, /\.textContent\s*=/);
  assert.doesNotMatch(SRC, /\.innerHTML\s*=/);
});
test("encodes untrusted handle/slug into URLs", () => {
  assert.match(SRC, /encodeURIComponent/);
});
test("emits cards with the .dep-index-card class + a status pill modifier", () => {
  assert.match(SRC, /dep-index-card/);
  assert.match(SRC, /dep-index-pill--/);
});
test("registers a MutationObserver on document.body with childList + subtree", () => {
  assert.match(SRC, /new MutationObserver/);
  assert.match(SRC, /document\.body/);
  assert.match(SRC, /childList\s*:\s*true/);
  assert.match(SRC, /subtree\s*:\s*true/);
});
test("listens to popstate for SPA navigation", () => {
  assert.match(SRC, /popstate/);
});
test("wraps everything in an IIFE with 'use strict'", () => {
  assert.match(SRC, /\(function \(\) \{\s*"use strict";/);
});
test("derives the DEP link base from /proposals/", () => {
  assert.match(SRC, /\/proposals\//);
});

console.log("");
console.log(`${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
