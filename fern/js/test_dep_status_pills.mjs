/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tests for the pure helpers in fern/js/dep-status-pills.js.
 *
 * dep-status-pills.js is a browser runtime and cannot be loaded directly by
 * node (references `window`, `document`). This file follows the extract-
 * and-eval pattern from test_dep_pr_comments.mjs: the string-only helpers
 * (`variant`, `slugFromHref`) are extracted by regex and evaluated in
 * isolation. DOM behaviour (the MutationObserver + per-link idempotency
 * flag) is exercised structurally — we assert the runtime contains the
 * required constructs — plus a live preview verification captured in the
 * PR description.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import assert from "node:assert/strict";

const HERE = dirname(fileURLToPath(import.meta.url));
const SRC = readFileSync(join(HERE, "dep-status-pills.js"), "utf8");

/**
 * Extract the raw source of a top-level `function <name>(...) { ... }` block
 * from the runtime source. Same helper as test_dep_pr_comments.mjs.
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

const variant = extractFn("variant");
const slugFromHref = extractFn("slugFromHref");

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

test("Draft → draft", () => {
  assert.equal(variant("Draft"), "draft");
});

test("Proposed → proposed", () => {
  assert.equal(variant("Proposed"), "proposed");
});

test("Under Review → proposed (via /review/)", () => {
  assert.equal(variant("Under Review"), "proposed");
});

test("Accepted → accepted", () => {
  assert.equal(variant("Accepted"), "accepted");
});

test("Approved → accepted (via /approv/)", () => {
  assert.equal(variant("Approved"), "accepted");
});

test("Final → accepted", () => {
  assert.equal(variant("Final"), "accepted");
});

test("Active → accepted", () => {
  assert.equal(variant("Active"), "accepted");
});

test("Ratified → accepted (via /ratif/)", () => {
  assert.equal(variant("Ratified"), "accepted");
});

test("Rejected → rejected", () => {
  assert.equal(variant("Rejected"), "rejected");
});

test("Withdrawn → rejected", () => {
  assert.equal(variant("Withdrawn"), "rejected");
});

test("Replaced → muted", () => {
  assert.equal(variant("Replaced"), "muted");
});

test("Superseded → muted", () => {
  assert.equal(variant("Superseded"), "muted");
});

test("Deferred → muted", () => {
  assert.equal(variant("Deferred"), "muted");
});

test("Implemented → draft (documented parity with DepMetadata.statusVariant)", () => {
  // DepMetadata.statusVariant() has no /implement/ branch, so "Implemented"
  // falls through to the default `draft` (amber) bucket. Do NOT change one
  // side without changing the other — the sidebar pill and the on-page pill
  // MUST render identically or the two pills disagree for the same DEP.
  assert.equal(variant("Implemented"), "draft");
});

test("unknown status → draft (default)", () => {
  assert.equal(variant("Whatever"), "draft");
});

test("null / undefined / empty → draft", () => {
  assert.equal(variant(null), "draft");
  assert.equal(variant(undefined), "draft");
  assert.equal(variant(""), "draft");
});

test("mixed-case input still matches (lowercased inside)", () => {
  assert.equal(variant("PROPOSED"), "proposed");
  assert.equal(variant("accepted"), "accepted");
});

console.log("");
console.log("slugFromHref() — extract slug from /proposals/<slug>");

test("extracts final segment from a fully-qualified path", () => {
  assert.equal(
    slugFromHref("/dynamo/dev/proposals/0001-dep-process"),
    "0001-dep-process",
  );
});

test("tolerates a trailing slash", () => {
  assert.equal(
    slugFromHref("/dynamo/dev/proposals/0001-dep-process/"),
    "0001-dep-process",
  );
});

test("tolerates a query string", () => {
  assert.equal(
    slugFromHref("/dynamo/dev/proposals/0000-nova?ref=nav"),
    "0000-nova",
  );
});

test("tolerates a URL fragment", () => {
  assert.equal(
    slugFromHref("/dynamo/dev/proposals/0000-nova#heading"),
    "0000-nova",
  );
});

test("works with absolute (https://...) URLs", () => {
  assert.equal(
    slugFromHref("https://docs.nvidia.com/dynamo/dev/proposals/dep-nova-synced"),
    "dep-nova-synced",
  );
});

test("returns empty string for non-proposals URLs", () => {
  assert.equal(slugFromHref("/dynamo/dev/getting-started/quickstart"), "");
});

test("returns empty string for /proposals/ (no slug)", () => {
  assert.equal(slugFromHref("/dynamo/dev/proposals/"), "");
  assert.equal(slugFromHref("/dynamo/dev/proposals"), "");
});

test("returns empty string on falsy input", () => {
  assert.equal(slugFromHref(""), "");
  assert.equal(slugFromHref(null), "");
  assert.equal(slugFromHref(undefined), "");
});

test("does not match /proposals/ appearing in a NON-final segment", () => {
  // Only the last /proposals/<slug> position counts.
  assert.equal(
    slugFromHref("/dynamo/dev/proposals/nested/0000-nova"),
    "",
  );
});

console.log("");
console.log("Runtime — structural constraints");

test("guards double-load with window.__depStatusPillsLoaded", () => {
  assert.match(SRC, /window\.__depStatusPillsLoaded/);
});

test("scans #fern-sidebar a.fern-sidebar-link (confirmed Fern DOM hook)", () => {
  assert.match(SRC, /#fern-sidebar a\.fern-sidebar-link/);
});

test("reads status map from window.__DEP_STATUS", () => {
  assert.match(SRC, /window\.__DEP_STATUS/);
});

test("uses data-dep-pill as per-link idempotency flag", () => {
  assert.match(SRC, /data-dep-pill/);
});

test("sets pill label via textContent (never innerHTML with user data)", () => {
  // Any innerHTML= assignment is a red flag — status text comes from the
  // status map, which is emitted by sync_deps.py from DEP metadata; even
  // if that is trusted, textContent keeps the injection surface zero.
  assert.match(SRC, /\.textContent\s*=/);
  assert.doesNotMatch(SRC, /\.innerHTML\s*=/);
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

test("emits pills with the .dep-status-pill class and a --<variant> modifier", () => {
  // We rely on the pill class + modifier convention for CSS lookups
  // (fern/main.css maps `.dep-status-pill--<variant>` to color pairs
  // cloned from DepMetadata's DEP_META_CSS pill vocabulary).
  assert.match(SRC, /dep-status-pill/);
  assert.match(SRC, /dep-status-pill--/);
});

test("cleans stale pills before re-injecting (SPA-safe)", () => {
  // On SPA nav Fern swaps the sidebar subtree. If we don't strip pills
  // from stale anchors + the data-dep-pill flag, we get duplicates or
  // wrong labels. Structural check: source must contain a removal path
  // for existing pills.
  assert.match(SRC, /removeChild|remove\(\)/);
});

console.log("");
console.log(`${passes} passed, ${fails} failed`);
if (fails) process.exit(1);
