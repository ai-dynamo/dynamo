// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { capturePayload, hasCheckpoint, liveConfig, modelHelperArgs, projectedContext, verifyCounterResult } from "../src/live-support.js";
import { snapshotDigest } from "../src/adapter.js";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { InMemoryCredentialStore } from "@earendil-works/pi-ai";
import { MODEL } from "../src/fixture-session.js";

const env = { DYNAMO_COMPACTION_PYTHON: "/venv/bin/python", DYNAMO_COMPACTION_TOKENIZER: "/models/pinned", DYNAMO_COMPACTION_BASE_URL: "http://127.0.0.1:8000/v1", DYNAMO_COMPACTION_DECISION_MODEL: "Qwen/Qwen3-0.6B", DYNAMO_COMPACTION_GENERATION_MODEL: "Qwen/Qwen3-0.6B", DYNAMO_COMPACTION_API_KEY: "test-runtime-only", DYNAMO_COMPACTION_CONTEXT_TOKENS: "8192" };

test("unsupported live profile pins reject before any actor request", () => {
  for (const patch of [{ DYNAMO_COMPACTION_DECISION_MODEL: "other" }, { DYNAMO_COMPACTION_GENERATION_MODEL: "other" }, { DYNAMO_COMPACTION_CONTEXT_TOKENS: "16384" }, { DYNAMO_COMPACTION_OUTPUT_TOKENS: "4097" }, { DYNAMO_COMPACTION_MIN_PROBABILITY: "0" }, { DYNAMO_COMPACTION_MIN_CONFIDENCE: "0" }]) assert.throws(() => liveConfig({ ...env, ...patch }));
});

test("live helper receives the same configured generation reserve as the actor", () => {
  const args = modelHelperArgs(liveConfig({ ...env, DYNAMO_COMPACTION_OUTPUT_TOKENS: "768" }));
  assert.equal(args[args.indexOf("--generation-max-tokens") + 1], "768");
});

test("checkpoint detection uses the actual native summary, not a fixture heading", () => {
  const summary = '{"kind":"untrusted_compaction_evidence","claims":[]}';
  const payload = { messages: [{ role: "user", content: `History was compacted:\n${summary}` }] };
  assert.equal(hasCheckpoint(payload, summary), true);
  assert.equal(hasCheckpoint({ messages: [{ content: [{ type: "text", text: summary }] }] }, summary), true);
  assert.equal(hasCheckpoint(payload, undefined), false);
  assert.equal(hasCheckpoint(payload, "Checkpoint observations"), false);
});
test("live configuration rejects missing pins, credentials and unsafe endpoint variants", () => {
  const config = liveConfig(env);
  assert.equal(config.contextTokens, 8192);
  for (const patch of [{ DYNAMO_COMPACTION_PYTHON: "python" }, { DYNAMO_COMPACTION_TOKENIZER: "remote/model" }, { DYNAMO_COMPACTION_API_KEY: "" }, { DYNAMO_COMPACTION_CONTEXT_TOKENS: "NaN" }, { DYNAMO_COMPACTION_BASE_URL: "https://example.com/v1" }, { DYNAMO_COMPACTION_BASE_URL: "http://user:pass@127.0.0.1:8000/v1" }]) assert.throws(() => liveConfig({ ...env, ...patch }));
});
test("projection uses Pi's actual compaction wrapper and retains recent native messages", () => {
  const context = projectedContext({ systemPrompt: "trusted", summary: "checkpoint", tail: [{ type: "message", id: "u", parentId: null, timestamp: "2026-01-01", message: { role: "user", content: "recent", timestamp: 1 } }] });
  assert.equal(context.messages[0].role, "system");
  assert.match(JSON.stringify(context.messages[1]), /conversation history before this point was compacted/);
  assert.match(JSON.stringify(context.messages[1]), /checkpoint/);
  assert.equal(context.messages[2].role, "user");
});
test("counter evidence must match exact payload and profile, never fixture evidence", () => {
  const payload = { messages: [{ role: "user", content: "test" }], max_output_tokens: 128 };
  assert.throws(() => verifyCounterResult({ input_tokens: 1, qualified_counter: false }, payload, 1024));
  assert.throws(() => verifyCounterResult({ input_tokens: -1, qualified_counter: true }, payload, 1024));
  const value = { version: 1, status: "counted", input_tokens: 12, qualified_counter: true, counter_name: "test_profile", request_sha256: snapshotDigest(payload), rendered_sha256: "a".repeat(64), output_reserve: 128, total_tokens: 140 };
  assert.deepEqual(verifyCounterResult(value, payload, 1024), { inputTokens: 12, counterName: "test_profile" });
  assert.throws(() => verifyCounterResult(value, payload, 128));
  assert.throws(() => verifyCounterResult(value, { ...payload, messages: [] }, 1024));
});

test("actual Pi provider serialization can be inspected with HTTP transport disabled", async () => {
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null, allowModelNetwork: false, refreshOnCreate: false });
  runtime.registerProvider(MODEL.provider, { baseUrl: MODEL.baseUrl, api: MODEL.api, apiKey: env.DYNAMO_COMPACTION_API_KEY, models: [MODEL] });
  const payload = await capturePayload(runtime, MODEL, projectedContext({ systemPrompt: "system", summary: "summary", tail: [] }));
  assert.equal(payload.model, MODEL.id);
  assert.match(JSON.stringify(payload.messages), /conversation history before this point/);
  assert.equal(payload.stream, true);
});
