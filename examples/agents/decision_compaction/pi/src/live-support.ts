// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { isAbsolute } from "node:path";
import { convertToLlm, type ModelRuntime } from "@earendil-works/pi-coding-agent";
import type { Context, Model } from "@earendil-works/pi-ai";
import { snapshotDigest, type Projection } from "./adapter.js";

export function liveConfig(env: NodeJS.ProcessEnv) {
  const required = (name: string) => { const value = env[`DYNAMO_COMPACTION_${name}`]; if (!value) throw Error(`missing_${name.toLowerCase()}`); return value; };
  const python = required("PYTHON"), tokenizer = required("TOKENIZER");
  if (!isAbsolute(python) || !isAbsolute(tokenizer)) throw Error("absolute_local_paths_required");
  const url = new URL(required("BASE_URL"));
  if (url.protocol !== "http:" || !["127.0.0.1", "localhost", "[::1]"].includes(url.hostname) || url.username || url.password || url.search || url.hash || url.pathname.replace(/\/$/, "") !== "/v1") throw Error("loopback_api_root_required");
  const contextTokens = Number(required("CONTEXT_TOKENS"));
  const outputTokens = Number(env.DYNAMO_COMPACTION_OUTPUT_TOKENS ?? "1024");
  const minProbability = Number(env.DYNAMO_COMPACTION_MIN_PROBABILITY ?? "0.9");
  const minConfidence = Number(env.DYNAMO_COMPACTION_MIN_CONFIDENCE ?? "0.9");
  const workload = env.DYNAMO_COMPACTION_WORKLOAD ?? "ordinary";
  if (!["ordinary", "redundant-progress"].includes(workload)) throw Error("unsupported_live_workload");
  if (![minProbability, minConfidence].every((n) => Number.isFinite(n) && n > 0 && n <= 1)) throw Error("invalid_selection_threshold");
  if (![contextTokens, outputTokens].every((n) => Number.isSafeInteger(n) && n > 0) || outputTokens >= contextTokens || contextTokens > 1_048_576) throw Error("invalid_context_budget");
  if (required("DECISION_MODEL") !== "Qwen/Qwen3-0.6B" || required("GENERATION_MODEL") !== "Qwen/Qwen3-0.6B" || contextTokens !== 8192 || outputTokens > 4096) throw Error("unqualified_model_profile");
  return { python, tokenizer, baseUrl: url.toString().replace(/\/$/, ""), contextTokens, outputTokens, minProbability, minConfidence, workload, decisionModel: required("DECISION_MODEL"), generationModel: required("GENERATION_MODEL"), apiKey: required("API_KEY") };
}

export function projectedContext(projection: Projection): Context {
  return { messages: convertToLlm([
    { role: "system", content: projection.systemPrompt, timestamp: 0 },
    { role: "compactionSummary", summary: projection.summary, tokensBefore: 0, timestamp: 0 },
    ...projection.tail.flatMap((entry) => entry.type === "message" ? [entry.message] : []),
  ]) };
}

export function modelHelperArgs(config: ReturnType<typeof liveConfig>): string[] {
  return ["-m", "dynamo.compaction.helper", "--mode", "model", "--base-url", config.baseUrl, "--decision-model", config.decisionModel, "--generation-model", config.generationModel, "--tokenizer-path", config.tokenizer, "--max-context-tokens", String(config.contextTokens), "--generation-max-tokens", String(config.outputTokens), "--min-probability", String(config.minProbability), "--min-confidence", String(config.minConfidence)];
}

export function hasCheckpoint(payload: Record<string, unknown>, summary: string | undefined): boolean {
  if (!summary || !Array.isArray(payload.messages)) return false;
  return payload.messages.some((message) => {
    const content = message.content;
    return typeof content === "string" ? content.includes(summary) : Array.isArray(content) && content.some((part) => part.type === "text" && typeof part.text === "string" && part.text.includes(summary));
  });
}

export async function capturePayload(runtime: ModelRuntime, model: Model<"openai-completions">, context: Context): Promise<Record<string, unknown>> {
  let payload: Record<string, unknown> | undefined;
  await runtime.streamSimple(model, context, { maxRetries: 0, fetch: async () => { throw Error("capture_transport_disabled"); }, onPayload: (value) => { payload = structuredClone(value) as Record<string, unknown>; throw Error("local_payload_capture"); } }).result();
  if (!payload) throw Error("provider_payload_unavailable");
  return payload;
}

export function verifyCounterResult(value: unknown, request: { messages: unknown; max_output_tokens: number }, contextTokens: number) {
  const result = value as Record<string, unknown>;
  if (!result || result.version !== 1 || result.status !== "counted" || result.qualified_counter !== true || typeof result.counter_name !== "string" || !result.counter_name || result.request_sha256 !== snapshotDigest(request) || typeof result.rendered_sha256 !== "string" || !/^[a-f0-9]{64}$/.test(result.rendered_sha256) || result.output_reserve !== request.max_output_tokens || result.total_tokens !== Number(result.input_tokens) + request.max_output_tokens || !Number.isSafeInteger(result.input_tokens) || Number(result.input_tokens) < 0 || Number(result.input_tokens) + request.max_output_tokens > contextTokens) throw Error("unqualified_or_oversized_actor_payload");
  return { inputTokens: Number(result.input_tokens), counterName: result.counter_name };
}
