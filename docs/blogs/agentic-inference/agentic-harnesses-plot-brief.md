# Agent Handoff: Plot and Asset Brief

This file is for an agent producing figures and supporting visual assets for `agentic-harnesses.md`.

Use this together with:

- `agentic-harnesses-experiment-plan.md`

Your job is to turn experiment outputs into figures that are simple, accurate, and easy to drop into a technical blog post.

## Output Directory

Save all candidate plots and visual artifacts under:

- `docs/blogs/agentic-inference/agentic-harnesses-artifacts/`

Prefer one subdirectory per topic:

- `prompt-instability/plots/`
- `reasoning-order/plots/`
- `streaming-actionable-state/plots/`
- `anthropic-fidelity/plots/`
- `responses-fidelity/plots/`

For each plot, also save:

- the source data file used to generate it
- the plotting script or notebook
- a one-paragraph note saying what the figure shows and what caveats apply

## General Rules

- Do not use decorative chart choices that obscure the result.
- Label aggregated and disaggregated serving clearly.
- If a value is estimated, label it as estimated.
- If the sample size is small, say so on the plot or in the sidecar note.
- Keep colors consistent across sections where possible.
- Use titles that describe the measurement, not the conclusion.

## Plot 1: Prompt Instability Before/After

### Goal

Show how an unstable preamble affects reuse and/or latency.

### Good candidate formats

- grouped bar chart for cache hit rate before/after
- grouped bar chart for TTFT before/after
- one annotated prompt diff or schematic showing the unstable prefix line

### Suggested filenames

- `prompt-instability-cache-hit-before-after.png`
- `prompt-instability-ttft-before-after.png`
- `prompt-instability-prefix-diff.png`

### Notes

- If you have both Anthropic and Dynamo results, do not over-combine them into one confusing chart.
- Prefer separate panels over overloaded legends.

## Plot 2: Reasoning Order Reconstruction

### Goal

Show that replay order changed the prefix and therefore changed the next-turn cache behavior.

### Good candidate formats

- sequence diagram or aligned text diff
- small side-by-side schematic:
  - original model output
  - old replay
  - corrected replay
- compact table with next-turn cache hit or TTFT before/after

### Suggested filenames

- `reasoning-order-replay-comparison.png`
- `reasoning-order-next-turn-impact.png`

### Notes

- This section may work better as a visual code/text artifact than a standard chart.
- If the main value is qualitative, do not force it into a quantitative figure.

## Plot 3: Streaming Actionable State Timeline

### Goal

Show when the tool call became actionable relative to stream end, and how much overlap was possible.

### Good candidate formats

- waterfall chart
- Gantt-style timeline
- two-panel comparison:
  - buffered completion
  - inline dispatch

### Minimum timestamps to show

- first token
- tool call complete
- tool execution start
- stream end
- tool execution end

### Suggested filenames

- `streaming-tool-dispatch-timeline.png`
- `streaming-tool-overlap-savings.png`

### Notes

- Keep one representative turn as the main visual.
- If you also have aggregate latency savings, add a second smaller chart.

## Supporting Artifact 4: Anthropic / Claude Code API Fidelity

### Goal

Show that the endpoint and metadata shape match what the client expects.

### Good candidate formats

- request/response table
- endpoint matrix
- one short route diagram for:
  - `GET /v1/models`
  - `GET /v1/models/{model_id}`
- one screenshot or clipped JSON response for Anthropic-format model metadata

### Suggested filenames

- `anthropic-model-endpoint-matrix.png`
- `anthropic-model-retrieve-slashed-id.png`
- `anthropic-streaming-usage-example.png`

### Notes

- This section should usually be handled with snippets, small tables, and route diagrams rather than a standalone plot.
- If the image input path is visually interesting, keep it as a short request-path diagram.

## Supporting Artifact 5: Responses / Codex Conversion Fidelity

### Goal

Show what was lost in the older conversion path and what is preserved now.

### Good candidate formats

- hourglass diagram showing dropped fields
- side-by-side field preservation table
- simple request-context flow diagram

### Suggested filenames

- `responses-lossy-vs-unified-conversion.png`
- `responses-field-preservation-table.png`

### Notes

- This section is architectural. A clean diagram is more useful than many small charts.

## Figure Quality Checklist

Before handing off a figure, confirm:

1. The title says what is being measured.
2. Units are shown where relevant.
3. Aggregated vs disaggregated modes are labeled clearly.
4. Estimated values are marked.
5. Source data is saved nearby.
6. The filename is descriptive.
7. A short sidecar note explains whether the figure is strong enough for the blog.

## Final Deliverable

Create:

- `docs/blogs/agentic-inference/agentic-harnesses-artifacts/plot-index.md`

For each candidate figure, include:

- filename
- section it supports
- one-sentence description
- whether it should be the primary figure, backup figure, or omitted
