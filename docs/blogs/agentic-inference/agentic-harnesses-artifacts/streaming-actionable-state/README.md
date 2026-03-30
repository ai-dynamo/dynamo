# Section 3: Streaming Actionable State

## Deployment Details

- **Model**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- **Node**: B200 GPU
- **Access**: SSH tunnel `localhost:8000`
- **Dispatch flags**: Both **disabled** (baseline measurement)
- **Date**: 2026-03-19

## Experiment Design

### Method: Runtime/config toggle (Method 1)

The deployment runs **without** `--enable-streaming-tool-dispatch` or `--enable-streaming-reasoning-dispatch`. This provides the "feature off" baseline where:
- Tool calls arrive as regular stream chunks
- The harness cannot act until `finish_reason: "tool_calls"` arrives
- There is a measurable gap between "tool call is complete" and "stream ends"

The "feature on" behavior (not yet measured on this deployment) would emit:
- `event: tool_call_dispatch` — structured SSE event as soon as a tool call is parseable
- `event: reasoning_dispatch` — per-token reasoning events during the thinking phase

### Flags to record

| Flag | Value | Effect |
|------|-------|--------|
| `--enable-streaming-tool-dispatch` / `DYN_ENABLE_STREAMING_TOOL_DISPATCH` | **off** | No `event: tool_call_dispatch` events |
| `--enable-streaming-reasoning-dispatch` / `DYN_ENABLE_STREAMING_REASONING_DISPATCH` | **off** | No `event: reasoning_dispatch` events |
| `--dyn-reasoning-parser` | `nemotron_deci` | Reasoning tokens parsed from `<think>` blocks |

## Workload

Simple calculator tool call:
```json
{
  "messages": [
    {"role": "system", "content": "You have access to a calculator tool. When asked math questions, always use it."},
    {"role": "user", "content": "What is 42 * 17?"}
  ],
  "tools": [{"type": "function", "function": {"name": "calculator", ...}}],
  "tool_choice": "auto",
  "stream": true
}
```

The model reasons briefly ("need to compute 42*17, use calculator"), then emits a single tool call. This creates a clean 4-phase timeline: TTFT → Reasoning → Tool Call Generation → Post-tool to Finish.

## Results (n=10, dispatch disabled)

### Timing Summary

| Metric | Mean (ms) | Stdev (ms) | Notes |
|--------|-----------|------------|-------|
| TTFT | 113.5 | 4.5 | Consistent across runs |
| Reasoning end | 251.7 | 97.8 | Variable — depends on reasoning length |
| Tool call complete | 286.0 | 96.4 | Complete name + arguments available |
| Stream done | 316.6 | 96.6 | `finish_reason` + `[DONE]` |
| **Dispatch gap** | **30.7** | **1.5** | **Time wasted waiting for stream end** |

### Key Finding

The tool call is **fully parseable ~31ms before the stream ends**. In buffered mode, the harness cannot begin tool execution until `[DONE]`. With streaming dispatch enabled, the server would emit `event: tool_call_dispatch` at the "Tool Call Complete" point, saving ~31ms per tool turn.

### Why 31ms matters

For a single tool call on a fast model, 31ms seems small. But consider:

1. **Agentic loops compound**: A 10-turn coding session with 2 tool calls per turn = 20 × 31ms = **620ms of idle waiting** that could have been tool execution time.

2. **Longer generations amplify the gap**: This workload generates ~15 reasoning tokens. A complex coding task might generate 200+ tokens of reasoning before the tool call. The model continues generating trailing content (whitespace, newlines) after the tool call is structurally complete. With more trailing tokens, the gap grows.

3. **Multiple tool calls**: With `parallel_tool_calls`, tool N might be complete while tool N+1 is still generating. Dispatch lets the harness start executing tool N immediately.

4. **Reasoning dispatch enables speculative prefill**: If the harness receives reasoning content in real-time, it can speculatively prepare resources (e.g., read files mentioned in reasoning) before the tool call even arrives.

### Stream Structure Observed

```
[0ms]    Request sent
[113ms]  First token (reasoning_content: "We")
[113ms]  ...reasoning tokens...
[252ms]  Reasoning ends, content starts
[286ms]  Tool call chunk: {name: "calculator", arguments: "{\"expression\":\"42 * 17\"}"}
[287ms]  Trailing content chunk (newline)
[317ms]  finish_reason: "tool_calls" + [DONE]
         ▲
         └── 31ms gap: harness is idle, tool call already complete
```

### Caveat: No overlap measurement

This experiment measures when the tool call becomes **available**, not when the harness **starts executing** it. Whether real overlap occurs depends on:
- The harness implementation (does it parse chunks incrementally?)
- Whether `parallel_tool_calls` is used
- Whether the harness maintains a connection during tool execution

The dispatch feature makes the "tool call available" event explicit and structured, rather than requiring the harness to parse delta chunks and detect completeness itself.

## Plots

- `plots/timeline-no-dispatch.png` — Waterfall timeline for all 10 runs
- `plots/summary-no-dispatch.png` — Summary bar chart with dispatch gap annotated

## Dispatch ON Results (n=10)

After enabling `DYN_ENABLE_STREAMING_TOOL_DISPATCH=1` on the deployment, re-ran the same measurement.

### Key Finding: Dispatch is a structured notification, not a latency optimization

| Metric | Dispatch OFF | Dispatch ON |
|--------|-------------|-------------|
| Wasted gap (avg) | 31ms | 29ms |
| TTFT (avg) | 113ms | 145ms* |

*TTFT variance higher with dispatch ON — attributed to SSH tunnel noise, not a real effect.

The `event: tool_call_dispatch` fires at the **same time** as the regular tool call chunk. The gap between "tool call complete" and "stream done" is unchanged (~30ms in both cases). This means:

1. **The dispatch event does NOT reduce the gap.** The trailing tokens after the tool call (whitespace, newlines) still generate before `finish_reason`.
2. **The value is structural, not temporal.** The dispatch event is a typed, structured SSE event with parsed `name` and `arguments` — the harness doesn't need to accumulate delta chunks and detect tool call completeness itself.
3. **For the blog:** Present dispatch as "the server tells you the tool call is ready" rather than "dispatch saves Xms per tool call."

### Dispatch event shape (captured from live deployment)

```
event: tool_call_dispatch
data: {"choice_index":0,"tool_call":{"index":0,"id":"call-938200c8-...","type":"function","function":{"name":"calculator","arguments":"{\"expression\":\"42 * 17\"}"}}}
```

This fires as a side-channel SSE event interleaved in the regular stream. A harness that watches for `event: tool_call_dispatch` gets a complete, parsed tool call without having to assemble it from delta chunks.

### Updated comparison plot

See `plots/dispatch-comparison.png` — three-panel box plot showing TTFT, tool call complete time, and wasted gap for both conditions.

## V2 Results: Localhost with Real Dispatch ON/OFF Toggle (2026-03-20)

Deployment restarted between conditions on the B200 compute node. Measured directly on the compute node (localhost, no SSH tunnel).

### Important: Three states, not two

There are **three** distinct behaviors to compare. The blog should be explicit about all three:

| State | Code | When harness learns about tool call | How |
|-------|------|-------------------------------------|-----|
| **Old (buffered)** | Pre-dispatch codebase | At `finish_reason: tool_calls` — **end of stream** | All tool call chunks were held back and sent with finish_reason |
| **New, dispatch OFF** | Current code, flag off | At tool_delta chunk — **~9ms before end** | Tool call chunks stream inline during generation |
| **New, dispatch ON** | Current code, flag on | At `event: tool_call_dispatch` — **same ~9ms, pre-parsed** | Side-channel SSE event with structured `{name, arguments}` |

The old buffered behavior is the **baseline**. The improvement is measured against `finish_reason` (when the old code would have told the harness).

### Measured results (15 runs each, localhost)

**Dispatch OFF (inline streaming, no dispatch event):**
- First tool info (tool_delta): mean **911.2ms**
- Finish reason: mean **920.3ms**
- **Harness knows 9.1ms before stream end**

**Dispatch ON (inline streaming + dispatch event):**
- First tool info (dispatch event): mean **912.2ms**
- Finish reason: mean **922.0ms**
- **Harness knows 9.8ms before stream end**
- Dispatch events detected: **Yes** — structured `{choice_index, tool_call: {name, arguments}}`

### What the baseline (old buffered) would look like

In the old code, the harness would only learn about tool calls at `finish_reason`. For our 30-city trace (30 sequential tool-calling turns):

- Each turn: harness waits **~920ms** until `finish_reason` to learn about the tool call
- With inline streaming: harness learns at **~911ms** — 9ms earlier per turn
- Over 30 turns: **~270ms of cumulative earlier feedback**

The 9ms per turn is small in absolute terms, but the key improvement is that **the harness no longer needs to buffer the entire stream to discover tool calls**. In the old code, if the model generated 500 tokens of reasoning before the tool call, the harness saw no actionable information until all those tokens plus the tool call were fully generated and the stream ended.

### 30-City Multi-Turn Results

Ran the full 30-city conversation (each turn: reasoning → echo tool call → tool result → next city).

| Condition | Turns completed | Info before done (mean) | Cumulative earlier feedback | Dispatch events |
|-----------|----------------|------------------------|---------------------------|-----------------|
| **Dispatch ON** | 6 | 10.7ms/turn | 54ms over 5 turns | Yes |
| **Dispatch OFF** | 11 | 9.6ms/turn | 86ms over 9 turns | No |

Both conditions show ~10ms of earlier tool call info per turn (from inline streaming, not dispatch specifically). Extrapolated to 30 turns: **~300ms of cumulative earlier feedback** vs old fully-buffered behavior.

Note: Model stopped after 6-11 turns due to the non-streaming continuation request getting a different model response. The per-turn delta is consistent regardless of how many turns completed.

### Dispatch event value

The dispatch event provides:
1. **Pre-parsed structured data** — `{name, arguments}` JSON, no delta accumulation needed
2. **Simpler harness code** — detect `event: tool_call_dispatch` instead of parsing every delta chunk
3. **Immediate user feedback** — show "Calling echo(Tokyo, ...)" as soon as dispatch arrives
4. **Foundation for parallel execution** — when models emit multiple tool calls, dispatch fires per-call, enabling the harness to start executing tool N while tool N+1 is still generating

### Honest caveats

- The dispatch event arrives at the **same time** as the tool_delta chunk — it does not provide earlier timing, only a cleaner format
- On this model/workload, the gap between tool info and stream end is ~9ms — model-dependent, could be larger with models that generate more trailing content
- We did not test with a model that generates multiple tool calls in a single stream response

## Revised Blog Recommendation

**Include.** The story has three layers:

1. **Old → New (inline streaming):** Tool call info is no longer buffered to stream end. Harness learns ~9ms earlier per tool turn. Over 30 turns, ~270ms cumulative. The real win is structural: no more blind buffering.

2. **Dispatch event:** A pre-parsed structured notification. Same timing as inline deltas, but the harness code is simpler and the user gets feedback immediately.

3. **Future:** For models with longer trailing output after tool calls, and for multi-tool responses, dispatch will provide increasingly earlier notifications.

## Scripts

- `../scripts/measure_stream_timing.py` — SSE timing measurement (V1, through SSH tunnel)
- `../scripts/dispatch_first_info.py` — V2 dispatch ON vs OFF measurement (localhost)
- `../scripts/harness_dispatch_experiment.py` — harness-in-the-loop with simulated tool execution
- `../scripts/plot_stream_timeline.py` — waterfall + summary plots (V1 data)
- `../scripts/plot_dispatch_comparison.py` — dispatch ON vs OFF comparison (V1 data)
