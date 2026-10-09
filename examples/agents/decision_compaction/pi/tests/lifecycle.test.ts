// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { SessionManager } from "@earendil-works/pi-coding-agent";
import { fixtureSession } from "../src/fixture-session.js";
import { snapshotDigest } from "../src/adapter.js";

test("real Pi adopts, persists, reopens and sends the accepted checkpoint", async () => {
  const f = await fixtureSession();
  try {
    const before = f.manager.getEntries().length;
    const result = await f.session.compact();
    assert.match(result.summary, /Verified fixture checkpoint/);
    assert.equal(f.acknowledgments.length, 1);
    assert.equal((f.acknowledgments[0] as { fromExtension: boolean }).fromExtension, true);
    assert.ok(f.manager.getEntries().length > before);
    const file = f.manager.getSessionFile();
    assert.ok(file);
    const reopened = SessionManager.open(file);
    assert.deepEqual(reopened.buildSessionContext().messages, f.manager.buildSessionContext().messages);
    const resumed = await f.reopen();
    await resumed.prompt("Continue the synthetic task", { expandPromptTemplates: false });
    assert.match(JSON.stringify(f.requests.at(-1)), /Verified fixture checkpoint/);
    assert.match(JSON.stringify(f.requests.at(-1)), /Constraint 0: do not push/);
    assert.equal(f.manager.getEntries().filter((e) => e.type === "message").length >= 8, true);
  } finally { await f.dispose(); }
});

test("model-qualified evidence is required independently of actor counter qualification", async () => {
  const f = await fixtureSession({ mode: "model", counterQualification: "model_qualified", helperCounterName: "test_profile_only", coordinator: async (r) => ({ version: 1, mode: "model", qualified_counter: true, counter_name: "test_profile_only", status: "accepted", snapshot_sha256: r.snapshot_sha256, summary: "Synthetic model response", retained_ids: r.units.filter((u) => u.protected).map((u) => u.id) }) });
  try { assert.match((await f.session.compact()).summary, /Synthetic model response/); }
  finally { await f.dispose(); }
});

test("native split turns and repeated compaction are explicitly unsupported", async () => {
  let calls = 0;
  const split = await fixtureSession({ coordinator: async () => { calls++; throw Error("must not run"); } }, 200);
  try { await assert.rejects(split.session.compact()); assert.equal(calls, 0); }
  finally { await split.dispose(); }
  const f = await fixtureSession();
  try {
    await f.session.compact();
    await f.session.prompt("Continue", { expandPromptTemplates: false });
    await assert.rejects(f.session.compact());
    assert.equal(f.manager.getEntries().filter((e) => e.type === "compaction").length, 1);
  } finally { await f.dispose(); }
});

test("telemetry cannot turn a rejection into native fallback", async () => {
  const f = await fixtureSession({ onOutcome: () => { throw Error("telemetry unavailable"); }, coordinator: async () => { throw Error("rejected"); } });
  try { await assert.rejects(f.session.compact()); assert.equal(f.requests.length, 0); }
  finally { await f.dispose(); }
});

test("unsupported custom compaction instructions are not silently discarded", async () => {
  let calls = 0;
  const f = await fixtureSession({ coordinator: async () => { calls++; throw Error("unexpected execution"); } });
  try { await assert.rejects(f.session.compact("Preserve the tool output verbatim")); assert.equal(calls, 0); }
  finally { await f.dispose(); }
});

test("selector receives frozen system and recent-tail context outside the candidate prefix", async () => {
  let observed = false;
  const f = await fixtureSession({ coordinator: async (r) => {
    assert.equal(r.context_units[0].role, "system");
    assert.match(r.context_units[0].text, /Preserve user constraints/);
    assert.ok(r.context_units.some((u) => u.text === "Constraint 3: do not push."));
    assert.ok(!r.units.some((u) => u.text === "Constraint 3: do not push."));
    assert.equal(r.session_id, f.manager.getSessionId());
    assert.equal(r.snapshot_sha256, snapshotDigest({ session_id: r.session_id, units: r.units, context_units: r.context_units }));
    observed = true;
    return { version: 1, mode: "fixture", status: "accepted", snapshot_sha256: r.snapshot_sha256, summary: "context-aware fixture", retained_ids: r.units.filter((u) => u.protected).map((u) => u.id) };
  } });
  try { await f.session.compact(); assert.equal(observed, true); }
  finally { await f.dispose(); }
});

for (const scenario of ["failed", "stale", "abort", "budget", "counter_hang", "coordinator_hang", "missing_protected", "wrong_hash", "fixture_opt_in", "unqualified_counter", "mode_mismatch", "unqualified_helper"] as const) {
  test(`real Pi does not persist rejected checkpoint: ${scenario}`, async () => {
    let f: Awaited<ReturnType<typeof fixtureSession>>;
    f = await fixtureSession({
      timeoutMs: 30,
      fixtureSession: scenario !== "fixture_opt_in",
      mode: ["unqualified_counter", "mode_mismatch", "unqualified_helper"].includes(scenario) ? "model" : "fixture",
      counterQualification: scenario === "unqualified_counter" ? "fixture" : "model_qualified",
      countRenderedTokens: () => scenario === "counter_hang" ? new Promise<number>(() => {}) : scenario === "budget" ? 999_999 : 100,
      coordinator: async (r) => {
        if (scenario === "failed") throw Error("private transcript must not leak");
        if (scenario === "coordinator_hang") return new Promise(() => {});
        if (scenario === "stale") f.manager.appendMessage({ role: "user", content: "new instruction", timestamp: 99 });
        if (scenario === "abort") f.session.abortCompaction();
        return { version: 1, mode: scenario === "unqualified_helper" ? "model" : "fixture", qualified_counter: false, status: "accepted", snapshot_sha256: scenario === "wrong_hash" ? "wrong" : r.snapshot_sha256, summary: "candidate", retained_ids: scenario === "missing_protected" ? [] : r.units.filter((u) => u.protected).map((u) => u.id) };
      },
    });
    try {
      await assert.rejects(f.session.compact());
      assert.equal(f.manager.getEntries().filter((e) => e.type === "compaction").length, 0);
      assert.equal(f.requests.length, 0);
    } finally { await f.dispose(); }
  });
}
