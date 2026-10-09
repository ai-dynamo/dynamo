// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { randomBytes } from "node:crypto";
import { createAssistantMessageEventStream, InMemoryCredentialStore, type AssistantMessage, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, ModelRuntime, SessionManager, SettingsManager, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { decisionCompaction, type Options } from "./adapter.js";

export const MODEL: Model<"openai-completions"> = {
  id: "cpu-fixture", name: "CPU fixture", api: "openai-completions", provider: "fixture", baseUrl: "http://127.0.0.1:1",
  reasoning: false, input: ["text"], contextWindow: 200_000, maxTokens: 4096,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
export function assistant(value: string): AssistantMessage {
  return { role: "assistant", content: [{ type: "text", text: value }], api: MODEL.api, provider: MODEL.provider, model: MODEL.id, stopReason: "stop", timestamp: 1,
    usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0, totalTokens: 2, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } };
}

export async function fixtureSession(overrides: Partial<Options> = {}, keepRecentTokens = 170) {
  const dir = await mkdtemp(join(tmpdir(), "dynamo-pi-fixture-"));
  const manager = SessionManager.create(dir, dir);
  for (let i = 0; i < 4; i++) {
    manager.appendMessage({ role: "user", content: `Constraint ${i}: do not push.`, timestamp: i });
    manager.appendMessage(assistant(`Observation ${i}: ${"old evidence ".repeat(50)}`));
  }
  const acknowledgments: unknown[] = [];
  const options: Options = { enabled: true, mode: "fixture", fixtureSession: true, counterQualification: "fixture", countRenderedTokens: () => 100, maxInputTokens: 100_000, maxOutputTokens: 1024,
    coordinator: async (request) => ({ version: 1, mode: "fixture", status: "accepted", snapshot_sha256: request.snapshot_sha256, summary: "Verified fixture checkpoint", retained_ids: request.units.filter((u) => u.protected).map((u) => u.id) }), ...overrides };
  const extension: ExtensionFactory = (pi) => {
    decisionCompaction(options)(pi);
    pi.on("session_compact", (event) => { acknowledgments.push(event); });
  };
  const settings = SettingsManager.inMemory({ compaction: { enabled: false, reserveTokens: 4096, keepRecentTokens }, retry: { enabled: false } });
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null, allowModelNetwork: false, refreshOnCreate: false });
  runtime.registerProvider("fixture", { baseUrl: MODEL.baseUrl, api: MODEL.api, apiKey: randomBytes(24).toString("hex"), models: [MODEL] });
  const loader = new DefaultResourceLoader({ cwd: dir, agentDir: dir, settingsManager: settings, noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: "Synthetic fixture only. Preserve user constraints.", extensionFactories: [extension] });
  await loader.reload();
  const { session } = await createAgentSession({ cwd: dir, agentDir: dir, modelRuntime: runtime, model: MODEL, sessionManager: manager, settingsManager: settings, resourceLoader: loader, noTools: "all" });
  const requests: unknown[] = [];
  session.agent.streamFunction = async (_model, context) => {
    requests.push(structuredClone(context));
    const stream = createAssistantMessageEventStream();
    stream.push({ type: "done", reason: "stop", message: assistant("Fixture continuation") });
    return stream;
  };
  const sessions = [session];
  const reopen = async () => {
    const file = manager.getSessionFile();
    if (!file) throw Error("fixture_not_persisted");
    const resumed = await createAgentSession({ cwd: dir, agentDir: dir, modelRuntime: runtime, model: MODEL, sessionManager: SessionManager.open(file), settingsManager: settings, resourceLoader: loader, noTools: "all" });
    resumed.session.agent.streamFunction = session.agent.streamFunction;
    sessions.push(resumed.session);
    return resumed.session;
  };
  return { dir, manager, session, acknowledgments, requests, reopen, dispose: async () => { sessions.forEach((s) => s.dispose()); await rm(dir, { recursive: true, force: true }); } };
}
