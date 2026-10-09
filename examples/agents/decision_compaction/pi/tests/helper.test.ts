// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { helperClient } from "../src/helper.js";
import type { Request } from "../src/adapter.js";

const request: Request = { version: 1, session_id: "fixture", snapshot_sha256: "fixture", units: [], context_units: [], budget: { max_input_bytes: 4096, max_output_bytes: 512, max_units: 256, max_summary_bytes: 256, max_model_calls: 3, max_output_tokens: 100 } };
test("helper has exact bounded JSON stdin/stdout and no shell interpolation", async () => {
  const invoke = helperClient(process.execPath, ["-e", 'process.stdin.on("data",x=>process.stdout.write(x))']);
  assert.deepEqual(await invoke(request, new AbortController().signal), request);
  assert.throws(() => helperClient("node", []), /absolute/);
});
for (const [name, source] of [
  ["malformed", 'process.stdout.write("not json")'],
  ["oversized", 'process.stdout.write("x".repeat(100000))'],
  ["nonzero", "process.exit(1)"],
] as const) {
  test(`helper rejects ${name}`, async () => {
    await assert.rejects(helperClient(process.execPath, ["-e", source])(request, new AbortController().signal));
  });
}
test("helper rejects oversize input, missing executable, preabort and in-flight abort", async () => {
  const c = new AbortController();
  const invoke = helperClient(process.execPath, ["-e", "setInterval(()=>{},1000)"]);
  await assert.rejects(invoke({ ...request, budget: { ...request.budget, max_input_bytes: 1 } }, c.signal));
  await assert.rejects(helperClient("/nonexistent/fixture-executable", [])(request, c.signal));
  const pending = invoke(request, c.signal);
  c.abort();
  await assert.rejects(pending);
  await assert.rejects(invoke(request, c.signal));
});
