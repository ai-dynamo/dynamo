// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { snapshotDigest, sourceUnits } from "../src/adapter.js";
import { assistant } from "../src/fixture-session.js";
import type { SessionEntry } from "@earendil-works/pi-coding-agent";

function entry(id: string, message: unknown): SessionEntry {
  return { type: "message", id, parentId: null, timestamp: "2026-01-01T00:00:00Z", message } as SessionEntry;
}

test("snapshot digest is stable and content-sensitive", () => {
  const units = [{ id: "u1", role: "user" as const, text: "Keep café", protected: true }];
  assert.equal(snapshotDigest(units), snapshotDigest(structuredClone(units)));
  assert.notEqual(snapshotDigest(units), snapshotDigest([{ ...units[0], text: "Changed" }]));
});

test("closed tool groups preserve the full untruncated source and remain protected", () => {
  const call = { ...assistant(""), content: [{ type: "toolCall", id: "call", name: "read", arguments: { path: "fixture" } }] };
  const output = "untrusted output ".repeat(10_000);
  const result = { role: "toolResult", toolCallId: "call", toolName: "read", content: [{ type: "text", text: output }], isError: false, timestamp: 1 };
  const units = sourceUnits([entry("user", { role: "user", content: [{ type: "text", text: "Keep me" }], timestamp: 1 }), entry("call", call), entry("result", result)]);
  assert.equal(units.length, 2);
  assert.equal(units[0].protected, true);
  assert.equal(units[1].role, "tool");
  assert.equal(units[1].protected, true);
  assert.equal(JSON.parse(units[1].text)[1].content[0].text, output);
});

test("unsupported content, metadata and broken tool groups fail closed", () => {
  const call = { ...assistant(""), content: [{ type: "toolCall", id: "call", name: "read", arguments: {} }] };
  const bad: SessionEntry[][] = [
    [entry("u", { role: "user", content: [{ type: "image", data: "unused", mimeType: "image/png" }] })],
    [entry("a", { ...assistant(""), content: [{ type: "thinking", thinking: "hidden" }] })],
    [entry("a", call)],
    [entry("a", { ...call, content: [...call.content, ...call.content] })],
    [entry("r", { role: "toolResult", toolCallId: "orphan", content: [] })],
    [{ type: "custom", id: "custom" } as SessionEntry],
  ];
  for (const entries of bad) assert.throws(() => sourceUnits(entries));
  assert.deepEqual(sourceUnits([{ type: "model_change" } as SessionEntry, { type: "thinking_level_change" } as SessionEntry]), []);
});

test("multiple tool calls are grouped once with out-of-order matching results", () => {
  const calls = ["one", "two"].map((id) => ({ type: "toolCall", id, name: "read", arguments: {} }));
  const result = (id: string) => entry(id, { role: "toolResult", toolCallId: id, toolName: "read", content: [{ type: "text", text: id }], timestamp: 1, isError: false });
  const units = sourceUnits([entry("calls", { ...assistant(""), content: calls }), result("two"), result("one")]);
  assert.equal(units.length, 1);
  assert.equal(JSON.parse(units[0].text).length, 3);
  assert.throws(() => sourceUnits([entry("calls", { ...assistant(""), content: calls }), result("one")]));
});
