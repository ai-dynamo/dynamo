# Agent Handoff: Agentic Harnesses Experiment Plan

This file is for an agent that will collect data and produce artifacts for `agentic-harnesses.md`.

Your job is not to rewrite the blog. Your job is to produce clean data, reproducible experiment notes, and candidate artifacts that the blog can cite.

Work section by section. Do not invent numbers. If a measurement is noisy, incomplete, or blocked by infrastructure, write that down plainly and move on.

## Primary Goal

Produce data, plots, snippets, and traces for the following blog sections:

1. Prompt instability
2. Reasoning fidelity and KV correctness
3. Streaming actionable state
4. Anthropic / Claude Code API fidelity
5. Responses / Codex fidelity

Treat these as:

- 3 primary experiment tracks: sections 1-3
- 2 supporting evidence packs: sections 4-5

The main draft is:

- `docs/blogs/agentic-inference/agentic-harnesses.md`

The companion architecture post is:

- `docs/blogs/agentic-inference/agentic-inference.md`

## Output Requirements

For sections 1-3, produce:

- one short `results.md` style summary
- raw measurements in machine-readable form if possible (`.json`, `.csv`, `.jsonl`)
- one or more candidate figures
- a short note on caveats, open questions, and whether the result is strong enough to include in the blog

For sections 4-5, produce:

- one short `results.md` style summary
- curl examples, snippets, or diagrams as appropriate
- raw responses or request/response captures where useful
- a short note on whether the material is worth including in the blog

Save outputs in a new subdirectory:

- `docs/blogs/agentic-inference/agentic-harnesses-artifacts/`

Create one subdirectory per section:

- `prompt-instability/`
- `reasoning-order/`
- `streaming-actionable-state/`
- `anthropic-fidelity/`
- `responses-fidelity/`

Inside each subdirectory, prefer this structure:

- `README.md`
- `raw/`
- `derived/`
- `plots/`

If you need scripts, keep them with the artifacts they generate, or put them in:

- `docs/blogs/agentic-inference/agentic-harnesses-artifacts/scripts/`

## Global Rules

- Do not fabricate data.
- Do not silently drop failed runs.
- Record exact command lines, model names, flags, and deployment mode.
- Keep aggregated and disaggregated serving results separate.
- If a comparison is not apples-to-apples, say so explicitly.
- When using estimates, label them as estimates.
- When a client is Claude Code, note whether it is direct-to-Dynamo or going through another proxy.
- Save enough raw data that another person could regenerate the plot later.

## Experiment Controls and Before/After Strategy

For every section, prefer a real before/after comparison over a single absolute number.

That means the agent should always ask:

- what is the "feature on" condition?
- what is the "feature off" condition?
- if there is no clean runtime switch, can the old behavior be reproduced by replaying or mutating traces?

Use one of these methods for each experiment and document which one you used:

1. Runtime/config toggle
   - Use a real flag, env var, or codepath switch if one exists.
   - Record the exact flag or env var in the section README.

2. Endpoint/path comparison
   - Compare an old vs corrected endpoint shape or event shape.
   - Good for model retrieval, Anthropic-format responses, or SSE event ordering.

3. Trace mutation / replay simulation
   - If the behavior is already fixed in the current codebase and no clean switch exists, create a controlled "broken" replay artifact by mutating the trace.
   - This is especially appropriate for reasoning-order experiments.
   - The mutation must be explained plainly and saved alongside the raw trace.

4. Branch or commit comparison
   - If needed, compare two code revisions.
   - Only use this when the simpler options above are not practical.
   - Record the exact commit hashes or PR branch names used.

When a section uses a simulated "feature off" condition, label it clearly as:

- `simulated broken replay`
- `mutated trace`
- `pre-fix branch`

Do not present a simulated condition as if it were a runtime flag if it is not.

## Available Test Resource

The blog author has a Nemotron 3 Super deployment ready on Computelab that may be used for these experiments.

Treat this as an available starting point, not a requirement:

- if it is the fastest way to get a clean result, use it
- if another model is better suited to a specific section, note why and use that instead
- always record exactly which deployment and model were used for each result
- do not assume every section must use the same model

## Recommended Models and In-Repo Guides

Do not assume one model is best for every section. Prefer the model that makes the behavior easiest to observe and easiest to explain.

Recommended starting points:

1. **Nemotron 3 Super on Computelab**
   - Good default starting point when the goal is to get a result quickly on an existing deployment.
   - Especially useful for direct harness compatibility checks if the deployment is already stable and reachable.
   - Since this is an existing deployment rather than a repo recipe, document the deployment details carefully in the artifact README.

2. **Qwen3-32B**
   - Best fit for aggregated vs disaggregated comparisons using an existing trace-driven recipe.
   - Useful for prompt-instability or replay-style experiments if you want a repeatable benchmark structure.
   - Start here:
     - `recipes/qwen3-32b/README.md`
     - `recipes/README.md`

3. **DeepSeek V3.2 NVFP4**
   - Useful when you want a large-scale trace benchmark with clear aggregated vs disaggregated comparison structure.
   - This is heavier-weight, so use it when the experiment benefits from the existing recipe rather than for quick iteration.
   - Start here:
     - `recipes/deepseek-v32-fp4/README.md`
     - `recipes/README.md`

General reference when parser settings matter:

- `docs/agents/tool-calling.md`

### Model-to-Section Suggestions

- **Prompt instability**
  - First choice: Nemotron 3 Super if it is already deployed and easy to trace
  - Second choice: Qwen3-32B recipe if you want a more repeatable trace-driven comparison

- **Reasoning fidelity and KV correctness**
  - First choice: use whichever existing deployment already gives you clean interleaved reasoning/tool traces
  - If you need a recipe-backed option, prefer a deployment where parser settings are easy to inspect and record
  - Use `docs/agents/tool-calling.md` to document parser choices when relevant

- **Streaming actionable state**
  - First choice: use whichever existing deployment already shows clear tool/reasoning patterns in your setup
  - Second choice: Nemotron 3 Super if it already exposes the relevant tool-use patterns in your test setup

- **Anthropic / Claude Code API fidelity**
  - Use whichever deployment is easiest to connect Claude Code to directly
  - Nemotron 3 Super is a reasonable first choice if already running

- **Responses / Codex fidelity**
  - Use the deployment that makes `v1/responses` testing easiest; model choice matters less here than endpoint behavior

When you cite these in the artifact README or later in the blog draft, use repo-local paths so another contributor can find the same guides without external search.

## Known Flags and Controls to Record

When a result depends on a feature switch, record the exact flag, env var, header, route, or parser setting used.

At minimum, check whether the experiment uses any of these:

- `--strip-anthropic-preamble`
- `DYN_STRIP_ANTHROPIC_PREAMBLE`
- `--enable-streaming-tool-dispatch`
- `DYN_ENABLE_STREAMING_TOOL_DISPATCH`
- `--enable-streaming-reasoning-dispatch`
- `DYN_ENABLE_STREAMING_REASONING_DISPATCH`
- `--dyn-reasoning-parser <parser_name>`
- `anthropic-version` request header
- `DYN_CONTEXT_WINDOW`
- `DYN_MAX_OUTPUT_TOKENS`

Important:

- for tool streaming experiments, the frontend behavior is controlled by `DYN_ENABLE_STREAMING_TOOL_DISPATCH=1`
- treat this as a frontend startup setting, not just a per-request knob
- when you switch it on or off, restart the frontend and record that restart in the experiment notes

Also record any runtime or deployment settings that materially affect the result, for example:

- aggregated vs disaggregated serving mode
- model name
- parser choice
- tool-call parser choice if relevant
- whether the client is direct-to-Dynamo or behind another proxy

If a switch does not exist and you are simulating the "off" state, say that explicitly and document the simulation method.

## Section 1: Prompt Instability

### Question

How much prompt-cache reuse do we lose when an unstable Anthropic/Claude preamble appears near the front of the prompt, and how much do we recover after stripping it?

### Why this matters

This section supports the blog's claim that small prompt-shape changes at the front of the prefix reduce reuse of the stable prompt behind them.

### What to collect

- one or more representative Claude Code traces
- one or more replayable Anthropic-style request traces
- cache-related metrics before and after preamble stripping
- TTFT before and after preamble stripping
- prompt token counts and any provider-reported cache read metrics if you are comparing against Anthropic
- one required baseline run based on vanilla Claude Code using Anthropic as the backend, so Dynamo results can be compared against the behavior users see on the managed API

### Required comparisons

Run the same trace or replay-derived request sequence in these modes:

1. Anthropic baseline derived from a vanilla Claude Code workflow
2. Dynamo aggregated serving, preamble present
3. Dynamo aggregated serving, preamble stripped
4. Dynamo disaggregated serving, preamble present
5. Dynamo disaggregated serving, preamble stripped

### Required Anthropic baseline procedure

Do this exactly once for each workflow you intend to compare against Dynamo.

1. Choose a fixed workflow.
   - Use a short but realistic coding workflow with multiple turns.
   - Prefer a workflow that produces repeated prompt reuse and at least one tool-use turn.
   - Save the workflow description in `prompt-instability/README.md`.

2. Run the workflow with vanilla Claude Code against Anthropic.
   - Do not point Claude Code at Dynamo for this run.
   - Do not use a proxy that changes request bodies.
   - Keep the workflow stable enough that it can be replayed later.

3. Capture the raw Anthropic-style request sequence generated by that workflow.
   - Save the sequence as sanitized replay artifacts if possible.
   - The goal is to obtain a replayable series of `v1/messages` style requests that preserves:
     - system content
     - messages
     - tools
     - `cache_control` placement
     - model name
   - Save the sanitized sequence under `prompt-instability/raw/anthropic-baseline/`.

4. Replay that exact request sequence against Anthropic's API.
   - This replay is the baseline used for measurement.
   - Record provider-side cache fields such as `cache_read_input_tokens` and related usage metadata when available.
   - Record TTFT if you can measure it externally from the replay client.

5. Create Dynamo replay variants from the same request sequence.
   - Variant A: original payloads with the unstable preamble intact
   - Variant B: normalized payloads with the unstable preamble removed
   - Use those same payload variants for aggregated and disaggregated Dynamo runs.

6. Compare Anthropic baseline metrics against Dynamo replay metrics.
   - Keep provider-side metrics and Dynamo-side metrics in separate columns.
   - Do not collapse them into one synthetic "cache hit" number unless the definition is truly the same.

What to capture for the Anthropic baseline:

- the exact workflow description
- the replayable sanitized request sequence
- whether prompt caching is enabled and where `cache_control` is applied
- provider-reported cache metrics such as `cache_read_input_tokens`
- TTFT if measured
- any mismatch between the managed Anthropic semantics and the Dynamo replay semantics

Important caveat:

- Anthropic-managed prompt caching and Dynamo KV-cache behavior are not the same system
- present this as a directional baseline, not an apples-to-apples identity comparison

### How to produce the comparison

Preferred order:

1. Use a real preamble-stripping on/off switch if one is available in the deployment you are testing.
2. If a clean switch is not available, replay the same trace through:
   - a path that keeps the preamble intact
   - a path that strips the preamble before inference
3. If neither is easy, create two replay payload sets from the same trace:
   - original payloads with the unstable preamble
   - normalized payloads with the unstable preamble removed
4. Use the required Anthropic baseline procedure above and save provider-side cache metrics separately from Dynamo metrics

Save both payload variants so the difference is inspectable.

### Good outputs

- before/after cache hit rate chart
- before/after TTFT chart
- one prompt diff showing the unstable line at the front
- one short trace walkthrough that explains why the prefix no longer matches

### Save as

- `agentic-harnesses-artifacts/prompt-instability/README.md`
- `agentic-harnesses-artifacts/prompt-instability/raw/*.jsonl`
- `agentic-harnesses-artifacts/prompt-instability/derived/*.csv`
- `agentic-harnesses-artifacts/prompt-instability/plots/*`

### Acceptance criteria

- The result clearly shows whether stripping changes reuse or latency.
- Aggregated vs disaggregated behavior is separated.
- The artifact includes at least one concrete prompt-level example, not just aggregate metrics.

## Section 2: Reasoning Fidelity and KV Correctness

### Question

What happens to cache reuse when interleaved reasoning and tool calls are replayed in the wrong order, and what changes after preserving the original order?

### Why this matters

This section supports the claim that reasoning fidelity is part of prompt reconstruction correctness, not just output formatting.

### What to collect

- one or more traces with interleaved reasoning and tool calls
- the original generated assistant content
- the old reconstructed replay form
- the corrected reconstructed replay form
- cache outcome or latency impact on the next turn

### Preferred comparisons

1. old reconstruction path
2. corrected reconstruction path

If possible, show this separately for:

- aggregated serving
- disaggregated serving

### How to produce the comparison

This section may require simulation if the current deployment only has the corrected behavior.

Acceptable approaches:

1. Replay against an older code path that still reconstructs reasoning incorrectly.
2. Create a controlled mutated replay from a correct trace:
   - keep the original model output
   - construct an "incorrect replay" form that flattens or reorders reasoning segments relative to tool calls
   - construct the corrected replay form that preserves the original order
3. Feed both replay forms into the same next-turn continuation and compare cache/latency outcomes.

When using a mutated replay, save all three artifacts:

- original generated assistant content
- mutated incorrect replay
- corrected replay

The blog should be able to show exactly what changed.

### Good outputs

- side-by-side snippet of original output, old replay, corrected replay
- one diagram or trace showing why the next-turn prefix no longer matched before the fix
- one table with cache hit or TTFT on the next turn before/after

### Important caveat

If decode KV is not returned to prefill workers in disaggregated serving, say clearly that this limits the visible cache benefit in that mode.

### Acceptance criteria

- The artifact contains one concrete example that a reader can follow visually.
- The before/after difference is tied to prompt shape, not described abstractly.

## Section 3: Streaming Actionable State

### Question

What latency do we save by emitting actionable tool and reasoning events during the stream instead of waiting until end-of-stream, and how should we present that result?

### Why this matters

This section supports the claim that agent loops need actionable state, not only token deltas.

### What to collect

- one or more tool-using traces
- timestamps for:
  - first token
  - complete tool call available
  - tool execution start
  - tool execution finish
  - model stream end
- the same run with inline dispatch disabled if possible
- `message_start` / `message_delta` usage behavior for Anthropic streaming

### Preferred comparisons

1. buffered tool completion at end of stream
2. inline tool dispatch as soon as the tool call is complete

If possible, also collect:

- reasoning block dispatch timing
- behavior with multiple tool calls

### How to produce the comparison

Preferred order:

1. Use the real inline dispatch feature flag or runtime toggle if one exists in your test setup.
   - For tool dispatch, use `DYN_ENABLE_STREAMING_TOOL_DISPATCH=1` and restart the frontend.
2. If the deployment does not expose a simple toggle, compare:
   - a path that emits only the standard buffered stream behavior
   - a path that emits the inline actionable events
3. If needed, capture raw SSE streams and derive timing from the event order and timestamps even if the harness itself does not expose a direct toggle.

For Anthropic usage behavior, explicitly capture:

- `message_start`
- any later usage-bearing event
- whether `input_tokens` remains stable or changes across the stream

### Good outputs

- waterfall/timeline figure for one or two representative turns
- aggregate latency savings across several tool turns
- a note on `parallel_tool_calls` and whether the harness overlaps tool execution with ongoing decoding
- a simple time*memory pressure approximation if you can produce one honestly

### Important note

Do not over-claim. If the harness or runtime does not actually overlap execution in a given setup, say that directly. Distinguish:

- actionable event emitted
- harness begins tool execution
- real overlap achieved

### Acceptance criteria

- At least one timeline plot exists.
- The result distinguishes end-of-stream buffering from actual dispatch timing.
- Input token behavior in `message_start` is documented for the Anthropic path.

## Section 4: Anthropic / Claude Code API Fidelity

### Question

Which API details were required for Claude Code to behave correctly when connected directly to Dynamo?

### Why this matters

This section supports the claim that compatibility depends on exact endpoint behavior and metadata shape, not just basic text generation.

This is a supporting evidence pack, not a primary benchmark section. Do not spend time trying to force a quantitative plot here unless one falls out naturally from the work.

### What to collect

- example `GET /v1/models` response in Anthropic format
- example `GET /v1/models/{model_id}` response in Anthropic format
- one example of a slashed model id lookup
- one example of `message_start` usage with non-zero `input_tokens`
- one example request using `cache_control`
- one example request using image content

### Preferred checks

Verify and save evidence for:

- `anthropic-version` content negotiation on model endpoints
- `context_window` and `max_output_tokens` showing up correctly
- individual model retrieval path working
- wildcard route correctly handling slashed model ids
- leading slash normalization if present in the captured route
- `created_at` format matching the intended Anthropic response shape

### How to produce the comparison

This section usually does not need a feature toggle. It is mostly endpoint-behavior verification.

Useful comparison styles:

1. broken vs fixed endpoint behavior if an older deployment is available
2. expected client request shape vs returned Dynamo response shape
3. list endpoint vs retrieve endpoint behavior for the same model

### Good outputs

- curl examples and matching responses
- one short table of "expected by Claude Code" vs "returned by Dynamo"
- a short README that says which details are worth mentioning in the blog and which are too implementation-specific

### Acceptance criteria

- The output demonstrates both list and retrieve endpoints.
- The output includes one slashed model-id example.
- The output clearly distinguishes model metadata issues from streaming issues.

## Section 5: Responses / Codex Fidelity

### Question

Which Responses-side details matter for Codex-style harnesses, and what evidence do we have that preserving request context changes behavior?

### Why this matters

This section keeps the post from being only about Anthropic/Claude Code.

This is also a supporting evidence pack rather than a primary benchmark section.

### What to collect

- one realistic `v1/responses` replay example
- one example where assistant `output_text` history omits output-only fields like `id` and `status`
- one before/after view of fields dropped by lossy conversion vs preserved by a unified request representation

### Good outputs

- a side-by-side request/response reconstruction example
- one diagram showing fields lost in the old hourglass conversion
- a short explanation of which fields matter most for realistic harness behavior

### How to produce the comparison

Preferred order:

1. Use a branch or commit comparison if the old lossy conversion path is easy to run.
2. Otherwise, create a field-level comparison using:
   - the original request context
   - the fields preserved in the corrected path
   - the fields that would have been dropped by the older conversion

This section can be partially static if needed, but any behavior claims should still be backed by a real request example.

### Acceptance criteria

- The material is specific enough to justify a short section in the blog.
- The section does not turn into a second blog post by itself.

## Deliverables Summary

At the end, produce:

1. `agentic-harnesses-artifacts/index.md`
   - one paragraph per section
   - strongest artifact to use
   - weakest or blocked area
   - recommendation: include, include with caveat, or omit

2. one final note for the blog author
   - which figures are strong enough for publication
   - which experiments need another pass
   - which claims should be softened or removed

## Suggested Working Style

- Start with the sections that are most likely to yield clear visual artifacts:
  - prompt instability
  - streaming actionable state
  - reasoning order
- Keep notes concise but complete.
- If you create scripts, make them rerunnable.
- Prefer one honest and readable plot over five noisy ones.
