---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Tool Calling Probe Snapshot for Dynamo 1.2"
subtitle: Static release snapshot of tool-calling probe results across supported model families
---

This page captures a one-time Dynamo 1.2.0 release snapshot from the
tool-calling probe harness generated on 2026-06-05 at 07:24 UTC. It is not a
live dashboard.

Failures are non-passing probe requests, and lower is better. The same scenario
can contribute separate failures for streaming and non-streaming request modes.
`Dynamo errors` counts Dynamo/parser/API-contract failures, including boundary
cases. It also counts Dynamo runtime or endpoint/deployment failures where the
request timed out before a usable OpenAI response was returned. `Other errors`
counts engine/model behavior and mixed/needs inspection failures. Issue notes
use the probe classifier:

- **Dynamo/parser likely**: raw model-native tool-call syntax leaked into the
  OpenAI response instead of structured `tool_calls`, final assistant text was
  routed into reasoning output, delimiter-like literal text was not preserved in
  a structured argument, or the parser/API contract was otherwise not satisfied.
- **Engine/model behavior likely**: the endpoint returned a response, but the
  model behavior did not satisfy the requested tool workflow.
- **Endpoint/deployment**: the request timed out before a usable response.
  These are counted as Dynamo runtime failures in this static release table.
- **Mixed/needs inspection**: raw request/response details need follow-up before
  assigning ownership.

Some current-main rows were run with a different number of probes than the
Dynamo 1.2.0 snapshot. Compare each `failures / total` count directly instead
of treating every row as an exact A/B pass-rate comparison.

The release-note cells below are based on the failed request and response
artifacts for both Dynamo 1.2.0 and current main.

With this classification, Dynamo runtime/parser/API failures improve on Kimi
K2.6, GLM 5.1, and Qwen3.6-35B-A3B. MiniMax 2.7 improves in total failures, but
its remaining parser-boundary failure count is unchanged.

<table>
  <thead>
    <tr>
      <th rowspan="2">Model</th>
      <th rowspan="2">Tool-call format</th>
      <th colspan="3">Dynamo 1.2.0 release</th>
      <th colspan="3">Current main</th>
      <th colspan="2">Release notes</th>
    </tr>
    <tr>
      <th>Total</th>
      <th>Dynamo errors</th>
      <th>Other errors</th>
      <th>Total</th>
      <th>Dynamo errors</th>
      <th>Other errors</th>
      <th>Current failures</th>
      <th>Improvement from 1.2 to main</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Kimi K2.6</td>
      <td>Kimi tool-call and reasoning format</td>
      <td>22 / 36</td>
      <td>21</td>
      <td>1</td>
      <td>2 / 36</td>
      <td>0</td>
      <td>2</td>
      <td>Current main only fails a multi-step search-and-crawl workflow in streaming and non-streaming modes. The model returns no structured tool calls and asks for endpoint clarification instead of executing the workflow. No raw marker leakage was observed in current main.</td>
      <td>Dynamo 1.2.0 had 18 parser/API-boundary failures and three endpoint timeouts. Model-native tool-call syntax appeared in reasoning instead of structured <code>tool_calls</code>, and some final assistant text was routed away from assistant content. Current main removes those Dynamo failures and leaves two model-workflow failures.</td>
    </tr>
    <tr>
      <td>DeepSeek V4 Pro</td>
      <td>DeepSeek tool-call and reasoning format</td>
      <td>0 / 46</td>
      <td>0</td>
      <td>0</td>
      <td>0 / 46</td>
      <td>0</td>
      <td>0</td>
      <td>No failures in the captured current-main run.</td>
      <td>No change needed. Dynamo 1.2.0 and current main are both clean.</td>
    </tr>
    <tr>
      <td>GLM 5.1</td>
      <td>GLM tool-call format</td>
      <td>4 / 48</td>
      <td>4</td>
      <td>0</td>
      <td>3 / 48</td>
      <td>3</td>
      <td>0</td>
      <td>Current main still fails delimiter-literal preservation in streaming and non-streaming modes because delimiter-looking text is not preserved in the structured argument. One non-streaming no-tools request also timed out.</td>
      <td>Current main improves from 4 to 3 Dynamo/runtime failures by removing a Dynamo 1.2.0 timeout in the multi-step search-and-crawl workflow. The delimiter-string preservation issue remains.</td>
    </tr>
    <tr>
      <td>MiniMax 2.7</td>
      <td>MiniMax tool-call format</td>
      <td>8 / 46</td>
      <td>2</td>
      <td>6</td>
      <td>4 / 46</td>
      <td>2</td>
      <td>2</td>
      <td>Current main has four failures. A simple arithmetic auto-tool prompt answers in text instead of producing the requested structured tool call in streaming and non-streaming modes. A delimiter-like literal string prompt returns a structured tool call in both modes, but the marker-looking text inside the argument is not preserved exactly; this is counted as a parser/API-boundary failure.</td>
      <td>Current main now uses the full 46-probe coverage and improves from 8 failures to 4. The multi-step tool-loop workflow and context echo auto-tool prompt that failed in Dynamo 1.2.0 now pass. Dynamo/parser-boundary failures remain at 2, while other failures drop from 6 to 2.</td>
    </tr>
    <tr>
      <td>Gemma 4 31B IT</td>
      <td>Gemma tool-call and reasoning format</td>
      <td>2 / 48</td>
      <td>2</td>
      <td>0</td>
      <td>2 / 46</td>
      <td>2</td>
      <td>0</td>
      <td>Current main still fails delimiter-literal preservation in streaming and non-streaming modes. The response produces a structured tool call, but the SQL string is truncated before the expected literal marker text.</td>
      <td>No observed failure-count improvement. Dynamo 1.2.0 and current main have the same failure class, with fewer probes in the current-main run.</td>
    </tr>
    <tr>
      <td>Qwen3.6-35B-A3B</td>
      <td>Qwen tool-call format</td>
      <td>1 / 48</td>
      <td>1</td>
      <td>0</td>
      <td>0 / 46</td>
      <td>0</td>
      <td>0</td>
      <td>No failures in the captured current-main run.</td>
      <td>Current main is clean. The Dynamo 1.2.0 non-streaming timeout in the multi-step search-and-crawl workflow is gone.</td>
    </tr>
    <tr>
      <td>GPT-OSS 120B</td>
      <td>GPT-OSS tool-call format</td>
      <td>14 / 48</td>
      <td>2</td>
      <td>12</td>
      <td>14 / 48</td>
      <td>2</td>
      <td>12</td>
      <td>Current main still has 14 failures. Multi-tool and parallel-tool prompts produce only one structured tool call, a simple calculation prompt answers in text instead of calling the tool, a marker-literal string argument omits the requested marker-like text, and the search/crawl final answer still misses the expected evidence. No raw model-native marker leakage was observed.</td>
      <td>The refreshed GPT-OSS current-main run is no longer worse than Dynamo 1.2.0 by count; both are 14 / 48. The prior main-only required-tool regression is gone, and the streaming multi-step workflow now returns final content instead of an empty assistant message, but the core multi-tool, parallel-tool, literal-marker, and final-answer gaps remain.</td>
    </tr>
  </tbody>
</table>
