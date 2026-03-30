# Next Experiment Fixes

The initial round of experiments (2026-03-19) produced structural artifacts and one clean timing measurement, but the three primary sections each have a specific weakness that a redesigned experiment can fix.

| Section | Current weakness | Root cause | Fix |
|---------|-----------------|------------|-----|
| 1. Prompt stripping | No measurable TTFT/cache difference on Dynamo | Aggregated single-GPU deployment, ~400 token prompts, prefix caching may not be active | Disaggregated deployment, large realistic prompt, confirmed prefix caching, TTFT measurement |
| 2. Reasoning order | No measurable TTFT/cache difference on Dynamo | Same as above + prompt too short for cache delta to exceed noise | Same fix + multi-turn trace with thousands of tokens of reasoning |
| 3. Streaming dispatch | Measured stream structure, not harness behavior | Script measures when tool call is *available*, not when the harness *starts executing* | Harness-in-the-loop experiment that measures actual tool execution overlap |

---

## Revised Experiment 1: Prompt Stripping with Measurable Cache Delta

### Goal

Show a TTFT difference between "preamble present" and "preamble stripped" on Dynamo, backed by per-request cache metrics if available.

### Why the original failed

1. **Prompt too small.** The test system prompt was ~400 tokens. At this scale, full prefix recomputation takes <10ms — invisible in the ~100ms SSH tunnel noise.
2. **Aggregated serving.** On a single GPU, cache "misses" just recompute locally. There is no cross-worker transfer cost. The benefit of correct prefix matching is small.
3. **Prefix caching may not have been active.** SGLang's prefix caching requires `--enable-cache-report` or similar flags to confirm it's working, and has minimum token thresholds.

### Revised design

**Deployment:** Disaggregated serving (separate prefill and decode workers). Use the Qwen3-32B recipe (`recipes/qwen3-32b/`) which has existing aggregated vs disaggregated configurations. Alternatively, use any model on 2+ GPUs with `--disaggregated-prefill-decode`.

**Prompt scale:** Use a realistic Claude Code system prompt. The Anthropic baseline showed Claude Code sends ~54,000 tokens of system context. Reconstruct a prompt of similar size:
- The real CLAUDE.md from this repo (~5K tokens)
- The tool definitions Claude Code sends (~10K tokens) — extract from a cc-proxy trace
- Pad with realistic instruction content to reach ~50K tokens total

**Prefix caching:** Confirm active before starting:
- Check SGLang logs for prefix cache hit/miss lines
- If available, use `--enable-cache-report` or equivalent flag
- If SGLang exposes `/metrics`, check for `sglang_cache_hit_rate` or equivalent Prometheus metric

**Procedure:**

1. Deploy disaggregated Qwen3-32B (prefill worker + decode worker)
2. Confirm prefix caching is active (check logs/metrics after a warmup request)
3. **Condition A — Stable prefix (5 sequential requests):**
   - Same 50K-token system prompt every request
   - Different short user message each time
   - Record TTFT per request
4. **Condition B — Varying prefix (5 sequential requests):**
   - Same 50K-token system prompt but with a different `x-anthropic-billing-header` prepended each time
   - Same user messages as Condition A
   - Record TTFT per request
5. **Condition C — Stripped (5 sequential requests):**
   - Same as B (billing header present) but with `DYN_STRIP_ANTHROPIC_PREAMBLE=1` enabled
   - Record TTFT per request
6. Repeat conditions A-C for 3 rounds

**Expected outcome:** Condition A and C should show TTFT dropping after the first request (cache hit). Condition B should show flat TTFT (no cache reuse — each request has a unique prefix). The delta between B and A/C is the measurable cache benefit.

**What to record:**
- TTFT per request (client-side, ms)
- SGLang cache hit/miss metrics per request if available
- Total prompt tokens per request
- Exact deployment command line (model, TP, disaggregation flags)
- Exact system prompt used (save as artifact)

**Script:** Modify `scripts/measure_prompt_instability.py` to accept `--system-prompt-file` for the large prompt and add a warmup phase.

### Acceptance criteria

- TTFT for Condition A request 2+ is measurably lower than request 1 (cache warming)
- TTFT for Condition B shows no improvement across requests (prefix never matches)
- TTFT for Condition C matches Condition A (stripping restores prefix stability)
- The delta is statistically significant (outside the error bars of the 3-round repetition)

---

## Revised Experiment 2: Reasoning Order with Measurable Cache Delta

### Goal

Show a TTFT difference between "correct reconstruction (Segments)" and "incorrect reconstruction (flattened Text)" on the next turn of a multi-turn conversation with substantial reasoning content.

### Why the original failed

1. **Prompt too small.** The test conversation was ~387 tokens. The divergence point between correct and incorrect reconstruction was at most ~50 tokens into the assistant turn. Cache savings of 50 tokens are unmeasurable.
2. **Reasoning too short.** The test model generated 1-2 sentences of reasoning per tool call. Need longer reasoning to push the divergence point deeper into the sequence.
3. **Same aggregated/caching issues as Section 1.**

### Revised design

**Deployment:** Same disaggregated deployment as Section 1 (share the setup cost).

**Prompt scale:** Construct a multi-turn conversation where:
- System prompt: ~5K tokens (realistic but not huge — the focus is the assistant turn)
- Turn 1 user: a complex coding question requiring multiple steps
- Turn 1 assistant: interleaved reasoning + 3 tool calls, with **100+ reasoning tokens per segment**
  - Reasoning segment 0: 100+ tokens analyzing the problem
  - Tool call 0: file read
  - Reasoning segment 1: 100+ tokens analyzing the file
  - Tool call 1: code edit
  - Reasoning segment 2: 100+ tokens verifying the edit
  - Tool call 2: test run
  - Trailing reasoning: 50+ tokens summarizing
- Turn 1 tool results: realistic file content, edit confirmation, test output (~2K tokens total)
- Turn 2 user: "Now also fix the related test file"

**Total assistant turn tokens to diverge on:** ~500+ tokens (3 reasoning segments × 100+ tokens each + 3 tool calls). The correct reconstruction interleaves these; the flattened form puts all 300+ reasoning tokens before all 3 tool calls. The divergence starts ~100 tokens into the assistant turn — everything after that is a cache miss.

**Procedure:**

1. Use the same disaggregated deployment from Section 1
2. **Condition A — Correct (Segments), 10 requests:**
   - Send the multi-turn conversation with `reasoning_content: ["seg0", "seg1", "seg2", "trailing"]`
   - Record TTFT for the Turn 2 response
3. **Condition B — Incorrect (Flattened), 10 requests:**
   - Same conversation but with `reasoning_content: "seg0\nseg1\nseg2\ntrailing"`
   - Record TTFT for the Turn 2 response
4. Between conditions, send a cache-busting warmup to clear the prefix cache

**Expected outcome:** Condition A should benefit from prefix cache reuse of the correctly-reconstructed assistant turn. Condition B should show higher TTFT because the flattened form diverges from what the model would have generated, forcing recomputation of the assistant turn tokens.

**Key subtlety:** This experiment requires that the backend caches the prefix including the assistant turn from a *previous* request with the same prefix. If the cache only holds the current request's KV, both conditions would show the same TTFT. Verify by running Condition A twice in sequence — the second run should have lower TTFT than the first.

**What to record:**
- TTFT per request
- Total prompt tokens
- Number of tokens in the assistant turn (where the divergence occurs)
- Whether the backend reports cache hits
- Exact conversation payload (save as artifact)

**Script:** Modify `scripts/measure_reasoning_order.py` to accept `--conversation-file` for the large multi-turn conversation.

### Acceptance criteria

- Condition A request 2+ shows lower TTFT than request 1 (prefix cached)
- Condition B shows flat or higher TTFT (prefix mismatch forces recomputation)
- The difference is attributable to the assistant turn size, not prompt noise

---

## Revised Experiment 3: Harness-in-the-Loop Dispatch Timing

### Goal

Measure the end-to-end time from "model starts generating" to "tool result is available" — comparing a harness that waits for `finish_reason` vs one that acts on `event: tool_call_dispatch`.

### Why the original failed

The original experiment measured **stream structure** (when each SSE event arrives) but not **harness behavior** (when the tool actually runs). The ~31ms gap between "tool call complete" and "stream done" is real, but the experiment didn't show that the harness can actually exploit it. The dispatch feature turned out to be structural (typed notification) rather than temporal (the gap doesn't shrink with dispatch enabled).

### Revised design

**Build a minimal test harness** that:
1. Sends a streaming chat completion request with tool definitions
2. Watches the SSE stream for tool calls
3. **Executes the tool** as soon as it's detected (either at `tool_call_dispatch` event or at `finish_reason`)
4. Records timestamps: request sent, first token, tool call detected, tool execution started, tool execution finished, response to user ready

**Two harness variants:**

**Variant A — Buffered (waits for finish_reason):**
```
stream starts → accumulate deltas → finish_reason: tool_calls → parse tool call → execute tool → done
```

**Variant B — Dispatch-aware (acts on tool_call_dispatch):**
```
stream starts → event: tool_call_dispatch → execute tool immediately → stream continues → finish_reason → done
```

**Tool:** Use a tool with measurable execution time — e.g., a simulated file read with a 50ms `time.sleep()`. This makes the overlap visible: in Variant B, the 50ms tool execution overlaps with the remaining stream, so total time is shorter.

**Workload:** Same calculator prompt as the original, but with multiple tool calls (`"What is 42*17 and what is sqrt(144)?"`). This produces 2 sequential tool calls — Variant B can start executing the first while the second is still generating.

**Procedure:**

1. Run Variant A (buffered) × 10 requests
2. Run Variant B (dispatch-aware) × 10 requests
3. Record per-request: total wall time from request-sent to all-tools-executed

**Expected outcome:**
- With a single fast tool call: Variant B saves ~31ms (the gap) minus overhead
- With a simulated 50ms tool: Variant B saves ~50ms because tool execution overlaps with trailing stream
- With 2 tool calls: Variant B may start the first tool earlier, saving more

**What to record:**
- Total wall time per request (request sent → all tool results ready)
- Per-tool timestamps: detected, started, finished
- Whether tools overlapped with stream in Variant B
- Exact deployment flags (dispatch ON for Variant B)

**Script:** Write `scripts/harness_dispatch_experiment.py` with both variants implemented as async functions. The script alternates between variants to control for transient load.

### Acceptance criteria

- Variant B total wall time is measurably less than Variant A for the simulated-slow-tool workload
- The timestamps show tool execution overlapping with ongoing stream in Variant B
- The improvement scales with tool execution time (faster tools → smaller absolute saving, larger relative saving)

---

## Infrastructure Requirements

All three revised experiments need:

| Requirement | Which sections | Notes |
|-------------|---------------|-------|
| Disaggregated deployment (prefill + decode) | Sections 1, 2 | Use `recipes/qwen3-32b/` disaggregated config or allocate 2+ GPUs with `--disaggregated-prefill-decode` |
| Confirmed prefix caching | Sections 1, 2 | Check SGLang logs/metrics before running |
| Large realistic prompt (~50K tokens) | Section 1 | Extract from cc-proxy Claude Code trace or reconstruct from CLAUDE.md + tools |
| Multi-turn conversation with long reasoning | Section 2 | Construct manually — ~500 tokens of interleaved reasoning + 3 tool calls |
| `DYN_ENABLE_STREAMING_TOOL_DISPATCH=1` | Section 3 | Already verified working on current deployment |
| Python async harness with SSE parsing | Section 3 | New script needed — `harness_dispatch_experiment.py` |

**Estimated time:** 1 day to set up disaggregated deployment + confirm caching. 1 day to run all three revised experiments + produce plots. Can be parallelized if deployment is shared.

---

## Relationship to Existing Artifacts

The revised experiments **replace** the noisy TTFT data in Sections 1 and 2, and **extend** the stream timing data in Section 3. The existing structural artifacts (diagrams, code snippets, curl captures, Anthropic baseline stats) remain valid and should be kept. The new data slots into the same directory structure:

```
prompt-instability/
  raw/anthropic-baseline-stats.json       ← KEEP (Anthropic baseline)
  raw/ttft-preamble-comparison.jsonl      ← REPLACE with disaggregated data
  derived/ttft-preamble-comparison.csv    ← REPLACE with disaggregated data
  plots/                                  ← NEW: TTFT chart that actually shows the effect

reasoning-order/
  raw/trace-example.json                  ← KEEP (structural artifact)
  raw/ttft-comparison.jsonl               ← REPLACE with long-conversation data
  derived/ttft-comparison.csv             ← REPLACE with long-conversation data
  plots/                                  ← NEW: TTFT chart that actually shows the effect

streaming-actionable-state/
  raw/timing-*.jsonl                      ← KEEP (stream structure data)
  plots/timeline-no-dispatch.png          ← KEEP (hero figure)
  raw/harness-dispatch.jsonl              ← NEW: end-to-end harness timing
  plots/harness-overlap.png               ← NEW: showing tool execution overlap
```
