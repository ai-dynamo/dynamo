# Scenario Charter

Every scenario has a harness-owned reach signal, a wire-level expectation, and a Dynamo assertion. Run in order; later scenarios are not meaningful without the earlier reach signal.

| ID | Harness behavior | Reach signal | Primary Dynamo observation | Initial priority |
| --- | --- | --- | --- | --- |
| B0 | Ordinary multi-turn coding task with shell/file edits | file change plus second tool turn | ordinary Responses and Messages streaming complete | P0 |
| C1 | Codex root tool loop | app-server tool item completed | Responses item/event set and headers | P0 |
| C1c | Codex plan collaboration mode | native `thread/settings/update` then plan-mode shell inspection | plan-mode Responses request remains accepted | P2 |
| C1d | Codex client-owned dynamic tool | app-server dynamic-tool callback plus a file derived from its result | dynamic Responses tool call/result continuation remains accepted | P2 |
| C1e | Codex dynamic-tool namespace | namespaced app-server callback plus a file derived from its result | namespace tool declaration and result continuation remain accepted | P2 |
| C1f | Codex dynamic-tool failure | failed app-server callback followed by a shell recovery effect | dynamic tool-error continuation remains accepted | P2 |
| C1f1 | Codex typed dynamic tool | native callback receives an exact two-integer argument object, then a file derives from its result | dynamic-tool JSON-schema declaration, argument delivery, and result continuation remain accepted | P2 |
| C1g | Codex automatic approval reviewer | reviewer-agent lifecycle before a read-only write escalation | `codex-auto-review` must be registered as a Dynamo model alias; reviewer subagent traffic and resumed Responses tool loop remain accepted | P2 |
| C1h | Codex stdio MCP tool | one native `mcpToolCall` start/completion pair plus a file derived from its fixed result | MCP tool declaration and function-call output continuation remain accepted | P1 |
| C1i | Codex stdio MCP error then recovery | one native MCP error result plus a shell recovery effect | MCP error continuation remains accepted and the same turn can recover | P1 |
| C1j | Codex native user-input request | experimental `request_user_input` request followed by a file derived from the supplied option | experimental user-input tool advertisement, server request, response, and continuation remain accepted | P2 |
| C1k | Codex MCP elicitation | stdio fixture requests an MCP form response during a native tool call | record the headless client action and any app-server elicitation callback without treating a local decline as a Dynamo failure | P3 |
| C1l | Codex MCP progress | stdio fixture sends one standards-compliant progress notification using the client token | record whether app-server exposes `mcpToolCall` progress without affecting the completed tool/result loop | P3 |
| C1m | Codex goal lifecycle | model creates, reads, and completes one goal around a shell verification | Responses goal-tool continuations and `thread/goal/updated` lifecycle remain accepted | P1 |
| C1a | Codex JSON-schema final output | schema-constrained completed turn | Responses `text.format` discriminator remains accepted | P1 |
| C1b | Codex failed shell tool then recovery | native command exit code 1 plus recovery file | tool-result continuation remains a valid Responses turn | P1 |
| C2 | Codex child agent | `spawn_agent` then child completion | child request has `thread-id` and parent mapping | P0 |
| C3 | Codex nested child agent | child invokes its own child and root joins | all parent-child request edges preserve session lineage | P1 |
| C4 | Codex parallel children | two children overlap then root joins | independent child contexts; no cross-request contamination | P1 |
| C5 | Codex manual compaction | `contextCompaction` notification | post-compact request is accepted and continues the tool loop | P0 |
| C5c | Codex thread rollback | rollback followed by a tool-using turn | model-visible history mutation leaves a valid Responses turn | P2 |
| C5a | Codex forked coding thread | fork and original each complete a later tool turn | forked history/session lineage remains isolated and accepted | P1 |
| C5a1 | Codex persisted-thread resume | fresh app-server process completes a later tool turn | stored Responses history remains accepted after process restart | P1 |
| C5a2 | Codex archive/unarchive | archive and restore a completed thread, then resume it for a state-dependent tool turn | archival lifecycle notifications, stored Responses history, and subsequent rehydration remain accepted | P2 |
| C5a3 | Codex injected Responses history | inject an assistant `output_text` item, then have a coding turn derive a file from it | app-server’s raw Responses history injection remains accepted by Dynamo and preserves assistant context | P2 |
| C5a4 | Codex injected agent handoff | inject Codex `agent_message` with `encrypted_content`, then have a coding turn derive a file from it | agent handoff normalization is exercised through native app-server persistence and the Responses tool loop | P1 |
| C5b | Codex detached review | detached review thread emits entry/exit review items | reviewer-specific Responses stream and lifecycle remain accepted | P1 |
| C5b1 | Codex inline review | inline review emits entry/exit items, then the same thread completes a tool turn | reviewer-specific Responses stream leaves root-thread history valid for later coding | P1 weekly |
| C6 | Codex active-turn steering | `turn/steer` accepted while turn is active | cancellation/final SSE semantics leave next user item valid | P0 |
| C7 | Codex interruption | `turn/interrupt` accepted before completion | canceled stream terminates without corrupting follow-up turn | P0 |
| C8 | Codex invalid lifecycle preconditions | deliberately stale turn ID rejected by app-server | no orphaned in-flight request or malformed retry | P2 |
| A1 | Claude ordinary tool loop | Bash/file tool result and follow-up model call | Anthropic Messages blocks, deltas, usage, and stop reason | P0 |
| A1f | Claude stdio MCP tool | one-or-more native MCP tool uses plus a file derived from the fixed result | Messages tool-use/tool-result continuation from `--mcp-config` remains accepted | P1 |
| A1g | Claude stdio MCP error then recovery | MCP error result reaches a following Messages request plus a shell recovery file | errored MCP `tool_result` continuation remains accepted | P1 |
| A1h | Claude MCP elicitation control boundary | stdio fixture requests an MCP form response during a native tool call | stream-json `control_request` is retained; noninteractive input cannot answer it | P3 |
| A1i | Claude MCP progress | stdio fixture sends one standards-compliant progress notification during a native tool call | progress-token negotiation and the subsequent Messages tool/result loop remain accepted | P2 |
| A1j | Claude MCP roots | stdio fixture requests `roots/list` during a native tool call | advertised roots capability, client response, and subsequent Messages tool/result loop remain accepted | P2 |
| A1a | Claude JSON-schema final output | tool-created file plus terminal result | Messages output-config discriminator remains accepted | P1 |
| A1e | Claude prompt suggestions | native `prompt_suggestion` event after a successful tool turn | any prompt-suggestions Messages request/header remains accepted | P1 |
| A1d | Claude failed shell tool then recovery | `tool_result.is_error` reaches next Messages request plus recovery file | error tool-result block remains accepted and the agent can continue | P1 |
| A1b | Claude persisted-session resume | second native process completes a tool turn | stored Messages history remains accepted after process restart | P1 |
| A1c | Claude forked session | resumed process creates a new session and completes a tool turn | forked Messages history remains accepted after process restart | P1 |
| A2 | Claude Agent subagent | Agent tool child transcript exists | `x-claude-code-agent-id` and parent/session lineage | P0 |
| A2a | Claude Agent with forwarded child text | Agent lifecycle plus parent-linked child blocks | forwarded Messages items retain `parent_tool_use_id` lineage | P1 |
| A3 | Claude nested-agent attempt | client response documents support or refusal | actual emitted wire shape, including any nested headers | P1 |
| A4 | Claude compaction | compact boundary in native transcript | continued Messages turn after compact summary | P0 |
| A4a | Claude automatic compaction | native boundary after an oversized turn | continued Messages request after threshold-driven summary | P1 |
| A5 | Claude cancellation plus user steering | canceled turn and immediately accepted next user prompt | error/SSE termination behavior and next turn integrity | P0 |
| A6 | Claude background-agent workflow | background-session terminal state plus isolated file effect | background service preserves Messages transport and tool-result continuation | P2 |
| X1 | Mid-stream injected 4xx/5xx/429/overload | controller observes client retry/failure | endpoint-specific error decoding and header preservation | P1 |
| X2 | Upstream SSE truncation/disconnect | controller observes recover/retry/failure | no invalid terminal event or stuck stream | P1 |
| X3 | malformed but validly-framed item/event | capture fixture reaches frontend parser | structured Dynamo error, never crash/hang | P2 |
| R1 | same scenarios after a harness upgrade | drift fingerprint changes | diff new headers, endpoint use, item/event discriminators | P1 |

## Initial run order

`B0 → C1 → A1 → C2 → A2 → C5 → A4 → C6 → C7 → A5 → C3/C4/A3 → X1/X2/X3 → R1`

`P0` is the initial live campaign. `P1` begins once the base lifecycle is demonstrated. `P2` is deliberately exploratory and should generate findings/fixtures instead of gatekeeping nightly immediately.

## First nightly distillation

The initial nightlies should run `C1a`, `C1b`, `C5`, `C5a4`, `C6` (with observed tool use), `A1a`, `A1b`, `A1d`, and `A4`, plus a rotation of `X1` and `X2`. The confirmed `C2` parser regression belongs in the Rust protocol unit test; `C5a4` adds a bounded native regression for its exact wire shape. Keep the model-directed child-agent workflow weekly because the model may choose unbounded additional delegation despite the one-child prompt. `A2`–`A5` remain discovery-only until their native reach signals become deterministic.
