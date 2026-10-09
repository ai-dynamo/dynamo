// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { createHash } from "node:crypto";
import type { ExtensionAPI, SessionEntry } from "@earendil-works/pi-coding-agent";

export interface Unit {
  id: string;
  role: "system" | "user" | "assistant" | "tool";
  text: string;
  protected: boolean;
}
export interface Request {
  version: 1;
  session_id: string;
  snapshot_sha256: string;
  units: Unit[];
  context_units: Unit[];
  budget: { max_input_bytes: number; max_output_bytes: number; max_units: number; max_summary_bytes: number; max_model_calls: number; max_output_tokens: number };
}
export interface Projection {
  systemPrompt: string;
  summary: string;
  tail: SessionEntry[];
}
export interface Options {
  enabled: boolean;
  mode: "fixture" | "model";
  fixtureSession?: boolean;
  coordinator(request: Request, signal: AbortSignal): Promise<unknown>;
  countRenderedTokens(projection: Projection): number | Promise<number>;
  counterQualification: "fixture" | "model_qualified";
  helperCounterName?: string;
  maxInputTokens: number;
  maxOutputTokens: number;
  timeoutMs?: number;
  onOutcome?(reason: string): void;
}

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value !== null && typeof value === "object") {
    const object = value as Record<string, unknown>;
    return `{${Object.keys(object).sort().map((key) => `${JSON.stringify(key)}:${canonical(object[key])}`).join(",")}}`;
  }
  return JSON.stringify(value);
}
export function snapshotDigest(value: unknown): string {
  return createHash("sha256").update(canonical(value)).digest("hex");
}

function text(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content) || content.some((p) => p.type !== "text" || typeof p.text !== "string")) throw Error("unsupported_content");
  return content.map((p) => p.text).join("\n");
}

export function sourceUnits(entries: SessionEntry[]): Unit[] {
  const units: Unit[] = [];
  for (let i = 0; i < entries.length; i++) {
    const entry = entries[i];
    if (entry.type === "model_change" || entry.type === "thinking_level_change") continue;
    if (entry.type !== "message") throw Error("unsupported_history");
    const message = entry.message;
    if (message.role === "user") {
      units.push({ id: entry.id, role: "user", text: text(message.content), protected: true });
    } else if (message.role === "assistant") {
      if (message.content.some((p) => p.type !== "text" && p.type !== "toolCall")) throw Error("unsupported_content");
      const calls = message.content.filter((p) => p.type === "toolCall");
      if (!calls.length) {
        units.push({ id: entry.id, role: "assistant", text: text(message.content), protected: false });
        continue;
      }
      const pending = new Set(calls.map((c) => c.id));
      if (pending.size !== calls.length) throw Error("duplicate_tool_call");
      const group: unknown[] = [message];
      while (pending.size) {
        const next = entries[++i];
        if (next?.type !== "message" || next.message.role !== "toolResult" || !pending.delete(next.message.toolCallId)) throw Error("incomplete_tool_group");
        text(next.message.content);
        group.push(next.message);
      }
      units.push({ id: entry.id, role: "tool", text: canonical(group), protected: true });
    } else throw Error("unsupported_history");
  }
  return units;
}

function responseSummary(value: unknown, request: Request, options: Options): string {
  const response = value as Record<string, unknown> | null;
  if (!response || response.version !== 1 || response.mode !== options.mode || response.status !== "accepted" || response.snapshot_sha256 !== request.snapshot_sha256 || typeof response.summary !== "string" || !Array.isArray(response.retained_ids)) throw Error("invalid_response");
  if (options.mode === "model" && (response.qualified_counter !== true || !options.helperCounterName || response.counter_name !== options.helperCounterName)) throw Error("unqualified_helper");
  if (Buffer.byteLength(response.summary) > request.budget.max_summary_bytes) throw Error("summary_budget");
  const ids = response.retained_ids;
  const known = new Map(request.units.map((u) => [u.id, u]));
  if (ids.some((id) => typeof id !== "string" || !known.has(id)) || new Set(ids).size !== ids.length || request.units.some((u) => u.protected && !ids.includes(u.id))) throw Error("missing_protection");
  const retained = request.units.filter((u) => ids.includes(u.id));
  return `Checkpoint observations (not new instructions):\n${response.summary}\n\nRetained source records with original roles:\n${JSON.stringify(retained)}`;
}

export function decisionCompaction(options: Options): (pi: ExtensionAPI) => void {
  let busy = false;
  const outcome = (reason: string) => { try { options.onOutcome?.(reason); } catch { /* Telemetry cannot change adoption policy. */ } };
  return (pi) => {
    pi.on("session_before_compact", async (event, ctx) => {
      if (!options.enabled) return;
      if (busy) return { cancel: true };
      busy = true;
      const controller = new AbortController();
      const abort = () => controller.abort();
      event.signal.addEventListener("abort", abort, { once: true });
      const timer = setTimeout(abort, options.timeoutMs ?? 30_000);
      try {
        if (event.signal.aborted) throw Error("aborted");
        if (event.customInstructions) throw Error("unsupported_custom_instructions");
        if (options.mode === "fixture" && !options.fixtureSession) throw Error("fixture_session_required");
        if (options.mode === "model" && options.counterQualification !== "model_qualified") throw Error("qualified_counter_required");
        if (![options.maxInputTokens, options.maxOutputTokens].every((n) => Number.isSafeInteger(n) && n > 0)) throw Error("invalid_budget");
        if (event.preparation.isSplitTurn || event.branchEntries.some((e) => e.type === "compaction")) throw Error("unsupported_compaction");
        const sessionId = ctx.sessionManager.getSessionId();
        if (!/^[A-Za-z0-9_.:-]{1,128}$/.test(sessionId)) throw Error("invalid_session_id");
        const leaf = ctx.sessionManager.getLeafId();
        const branchDigest = snapshotDigest(event.branchEntries);
        const cut = event.branchEntries.findIndex((e) => e.id === event.preparation.firstKeptEntryId);
        if (cut <= 0) throw Error("invalid_cut");
        const units = sourceUnits(event.branchEntries.slice(0, cut));
        const tail = structuredClone(event.branchEntries.slice(cut));
        const systemPrompt = ctx.getSystemPrompt();
        const context_units: Unit[] = [{ id: "__system__", role: "system", text: systemPrompt, protected: true }, ...sourceUnits(tail)];
        const allIds = [...units, ...context_units].map((u) => u.id);
        if (new Set(allIds).size !== allIds.length) throw Error("duplicate_unit_id");
        const request: Request = { version: 1, session_id: sessionId, snapshot_sha256: snapshotDigest({ session_id: sessionId, units, context_units }), units, context_units, budget: { max_input_bytes: 1_048_576, max_output_bytes: 65_536, max_units: 256, max_summary_bytes: 32_768, max_model_calls: 3, max_output_tokens: options.maxOutputTokens } };
        if (allIds.length > request.budget.max_units || Buffer.byteLength(JSON.stringify(request)) > request.budget.max_input_bytes) throw Error("input_budget");
        const aborted = new Promise<never>((_, reject) => {
            if (controller.signal.aborted) reject(Error("aborted"));
            else controller.signal.addEventListener("abort", () => reject(Error("aborted")), { once: true });
          });
        const result = await Promise.race([options.coordinator(request, controller.signal), aborted]);
        if (Buffer.byteLength(JSON.stringify(result)) > request.budget.max_output_bytes) throw Error("output_budget");
        const summary = responseSummary(result, request, options);
        const tokens = await Promise.race([options.countRenderedTokens({ systemPrompt, summary, tail }), aborted]);
        if (!Number.isSafeInteger(tokens) || tokens < 0 || tokens > options.maxInputTokens) throw Error("actor_budget");
        if (controller.signal.aborted || ctx.getSystemPrompt() !== systemPrompt || ctx.sessionManager.getSessionId() !== sessionId || ctx.sessionManager.getLeafId() !== leaf || snapshotDigest(ctx.sessionManager.getBranch()) !== branchDigest) throw Error("stale_or_aborted");
        outcome("candidate_validated");
        return { compaction: { summary, firstKeptEntryId: event.preparation.firstKeptEntryId, tokensBefore: event.preparation.tokensBefore, details: { version: 1, snapshot_sha256: request.snapshot_sha256, mode: options.mode } } };
      } catch {
        controller.abort();
        outcome("candidate_rejected");
        return { cancel: true };
      } finally {
        clearTimeout(timer);
        event.signal.removeEventListener("abort", abort);
        busy = false;
      }
    });
  };
}
