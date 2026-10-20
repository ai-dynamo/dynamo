// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { InMemoryCredentialStore, type Model } from "@earendil-works/pi-ai";
import { createAgentSession, DefaultResourceLoader, estimateTokens, ModelRuntime, SessionManager, SettingsManager, type AgentSession, type ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { decisionCompaction, snapshotDigest } from "./adapter.js";
import { helperClient, jsonProcess } from "./helper.js";
import { capturePayload, hasCheckpoint, liveConfig, modelHelperArgs, projectedContext, verifyCounterResult } from "./live-support.js";

const config = liveConfig(process.env);
const directory = await mkdtemp(join(tmpdir(), "dynamo-pi-live-"));
const root = fileURLToPath(new URL("../../../../../", import.meta.url));
const env = { PYTHONPATH: resolve(root, "components/src"), PYTHONNOUSERSITE: "1", PYTHONUNBUFFERED: "1", DYNAMO_COMPACTION_API_KEY: config.apiKey };
const countProcess = jsonProcess(config.python, ["-m", "dynamo.compaction.helper", "--mode", "count", "--tokenizer-path", config.tokenizer], env, 1_048_576, 65_536);
const counts: { inputTokens: number; counterName: string; requestHash: string }[] = [];
let qualifiedName: string | undefined;
async function count(messages: unknown, signal: AbortSignal) {
  const request = { messages, max_output_tokens: config.outputTokens };
  const result = verifyCounterResult(await countProcess(request, signal), request, config.contextTokens);
  if (qualifiedName && qualifiedName !== result.counterName) throw Error("counter_artifacts_changed");
  counts.push({ ...result, requestHash: snapshotDigest(request) });
  return result;
}
const model: Model<"openai-completions"> = {
  id: config.generationModel, name: config.generationModel, api: "openai-completions", provider: "dynamo-live", baseUrl: config.baseUrl,
  reasoning: false, input: ["text"], contextWindow: config.contextTokens, maxTokens: config.outputTokens,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, compat: { supportsDeveloperRole: false, supportsStore: false, maxTokensField: "max_tokens" },
};
const sessions: AgentSession[] = [];
const actorEvidence: { inputTokens: number; measuredInputTokens: number; outputTokens: number; checkpointPresent: boolean }[] = [];
const acknowledgments: unknown[] = [];
let helperEvidence: unknown;
let checkpointSummary: string | undefined;
let stage = "tokenizer_preflight";
const outcomes: string[] = [];
try {
  qualifiedName = (await count([{ role: "user", content: "Tokenizer qualification probe." }], AbortSignal.timeout(30_000))).counterName;
  const manager = SessionManager.create(directory, directory);
  const runtime = await ModelRuntime.create({ credentials: new InMemoryCredentialStore(), modelsPath: null, allowModelNetwork: false, refreshOnCreate: false });
  runtime.registerProvider("dynamo-live", { baseUrl: config.baseUrl, api: model.api, apiKey: config.apiKey, models: [model], headers: { "x-dynamo-session-id": manager.getSessionId() } });
  const coordinator = helperClient(config.python, modelHelperArgs(config), env);
  const extension: ExtensionFactory = (pi) => {
    decisionCompaction({ enabled: true, mode: "model", counterQualification: "model_qualified", helperCounterName: qualifiedName, maxInputTokens: config.contextTokens - config.outputTokens, maxOutputTokens: config.outputTokens, timeoutMs: 30_000,
      onOutcome: (reason) => { outcomes.push(reason); },
      coordinator: async (request, signal) => { helperEvidence = await coordinator(request, signal); return helperEvidence; },
      countRenderedTokens: async (projection, signal) => { const payload = await capturePayload(runtime, model, projectedContext(projection)); return (await count(payload.messages, signal)).inputTokens; },
    })(pi);
    pi.on("session_compact", (event) => { acknowledgments.push({ fromExtension: event.fromExtension }); });
  };
  const settings = SettingsManager.inMemory({ compaction: { enabled: false, reserveTokens: config.outputTokens, keepRecentTokens: 128 }, retry: { enabled: false }, defaultThinkingLevel: "off" });
  const loader = new DefaultResourceLoader({ cwd: directory, agentDir: directory, settingsManager: settings, noExtensions: true, noSkills: true, noPromptTemplates: true, noThemes: true, noContextFiles: true, systemPrompt: "You are validating conversation continuity. Answer in plain text, without tools or reasoning traces. Preserve all user constraints. Do not execute commands, edit files, deploy, or push anything.", extensionFactories: [extension] });
  await loader.reload();
  async function open(sessionManager: SessionManager) {
    const { session } = await createAgentSession({ cwd: directory, agentDir: directory, modelRuntime: runtime, model, sessionManager, settingsManager: settings, resourceLoader: loader, noTools: "all", thinkingLevel: "off" });
    sessions.push(session);
    return session;
  }
  async function prompt(session: AgentSession, input: string) {
    let expected: number | undefined;
    let checkpointPresent = false;
    session.agent.onPayload = async (raw) => {
      const payload = raw as Record<string, unknown>;
      const result = await count(payload.messages, AbortSignal.timeout(30_000));
      expected = result.inputTokens;
      checkpointPresent = hasCheckpoint(payload, checkpointSummary);
      return { ...payload, chat_template_args: { enable_thinking: false }, max_tokens: config.outputTokens };
    };
    const timer = setTimeout(() => session.abort(), 120_000);
    try { await session.prompt(input, { expandPromptTemplates: false }); }
    finally { clearTimeout(timer); }
    const last = session.sessionManager.getEntries().at(-1);
    if (last?.type !== "message" || last.message.role !== "assistant" || last.message.stopReason !== "stop" || expected === undefined) throw Error("actor_incomplete");
    const usage = last.message.usage;
    const measured = usage.input + usage.cacheRead + usage.cacheWrite;
    actorEvidence.push({ inputTokens: expected, measuredInputTokens: measured, outputTokens: usage.output, checkpointPresent });
    if (measured !== expected) throw Error("actor_tokenizer_parity_failed");
    return last.message;
  }
  const session = await open(manager);
  stage = "actor_history";
  if (config.workload === "redundant-progress") {
    await prompt(session, "Remember these requirements: the project code word is compass; never deploy or push. For disposable progress logging only, write the line 'Old progress check complete; no new facts.' exactly ten times. These repeated logs will be obsolete after this step.");
    await prompt(session, "The previous repetitive assistant progress log is obsolete and contains no task facts. For another disposable log, write 'Redundant progress report finished.' ten times. All such progress logs may be summarized later; preserve only the code word and no-deploy/no-push restrictions.");
  } else {
    await prompt(session, "Remember these requirements: the project code word is compass; never deploy or push. Explain a read-only test plan in about 120 words.");
    await prompt(session, "List three checks for a documentation-only change, retaining the earlier restrictions. About 120 words.");
  }
  const recent = await prompt(session, "Confirm that this session remains read-only. State the code word and restrictions, without adding new facts.");
  // Select a native whole-turn cut using Pi's own estimate, not a model-token claim.
  settings.applyOverrides({ compaction: { enabled: false, reserveTokens: config.outputTokens, keepRecentTokens: estimateTokens(recent) + 1 } });
  stage = "compaction";
  await session.compact();
  assert.equal(acknowledgments.length, 1);
  assert.equal((acknowledgments[0] as { fromExtension: boolean }).fromExtension, true);
  const checkpoint = manager.getEntries().findLast((entry) => entry.type === "compaction");
  assert.ok(checkpoint && checkpoint.type === "compaction");
  checkpointSummary = checkpoint.summary;
  const file = manager.getSessionFile();
  assert.ok(file);
  const reopened = await open(SessionManager.open(file));
  stage = "resumed_actor";
  const continuation = await prompt(reopened, "What is the project code word, and which actions must not be performed?");
  const answer = continuation.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
  assert.match(answer.toLowerCase(), /compass/);
  assert.match(answer.toLowerCase(), /(?:never|must not|do not|no|avoid|prohibited)[^.;\n]{0,100}\bdeploy\b/);
  assert.match(answer.toLowerCase(), /(?:never|must not|do not|no|avoid|prohibited)[^.;\n]{0,100}\bpush\b/);
  assert.equal(actorEvidence.at(-1)?.checkpointPresent, true);
  const evidence = { mode: "model", workload: config.workload, thresholds: { probability: config.minProbability, confidence: config.minConfidence }, pi: "1.1.0", decisionModel: config.decisionModel, generationModel: config.generationModel, counterName: qualifiedName, actorEvidence, counts, helperEvidence, persisted: true, resumed: true, continuationCheck: "lexical_codeword_and_restrictions", modelQualityQualified: false };
  await writeFile(join(directory, "evidence.json"), JSON.stringify(evidence, null, 2), { mode: 0o600 });
  console.log(JSON.stringify({ status: "passed", evidenceDirectory: directory, decisionModel: config.decisionModel, actorRequests: actorEvidence.length, persisted: true, resumed: true, tokenizerParity: true }));
} catch (error) {
  const safeCodes = ["actor_incomplete", "actor_tokenizer_parity_failed", "counter_artifacts_changed", "unqualified_or_oversized_actor_payload", "Compaction cancelled"];
  const reason = error instanceof Error && safeCodes.includes(error.message) ? error.message : "validation_failed";
  await writeFile(join(directory, "evidence.json"), JSON.stringify({ status: "failed", workload: config.workload, thresholds: { probability: config.minProbability, confidence: config.minConfidence }, stage, reason, counts, actorEvidence, helperEvidence, acknowledgments, outcomes }, null, 2), { mode: 0o600 });
  console.error(JSON.stringify({ status: "failed", stage, reason, evidenceDirectory: directory }));
  process.exitCode = 1;
} finally { sessions.forEach((session) => session.dispose()); }
