# Live Findings

This is an evidence index for the exploratory campaign. It is not a nightly contract; promote only deterministic, minimized cases after another clean rerun.

## Confirmed Dynamo regression and fix

- A native Codex child request included an `agent_message` content part with `type: encrypted_content`. Dynamo rejected it before the fix even though current Codex supplies the child assignment as that part's string value.
- The smallest fix preserves that string while flattening Codex `agent_message` content. Unit coverage is in `lib/llm/src/protocols/openai/responses/mod.rs`.
- The fixed live rerun completed a deep native subagent graph with no child-agent errors: `/ephemeral/harness-compat-artifacts/20260801T001043Z-codex-subagent-binding-fix`.

## Confirmed native lifecycle coverage

- Codex manual compaction followed by a tool-using turn: `/ephemeral/harness-compat-artifacts/20260801T002650Z-codex-compact-threadscope`.
- Codex active-turn steering and a valid later turn: `/ephemeral/harness-compat-artifacts/20260801T004340Z-codex-steer-active`.
- Codex steering after an observed native `exec_command` call, followed by the steered file edit and a valid later turn: `/ephemeral/harness-compat-artifacts/20260801T014706Z-codex-steer-after-tool-rerun`.
- Codex interrupt produced `turn_status: interrupted`, then a valid later turn: `/ephemeral/harness-compat-artifacts/20260801T004656Z-codex-interrupt-active`.
- Codex parallel subagents expanded to 17 spawns, 43 Responses requests, five sessions, depth three, and zero agent errors: `/ephemeral/harness-compat-artifacts/20260801T011121Z-codex-parallel-subagents`.
- Codex nested subagents completed 21 Responses requests, four sessions, depth three, and zero agent errors: `/ephemeral/harness-compat-artifacts/20260801T011502Z-codex-nested-subagents`.
- Codex completed a child-agent turn with zero agent errors, compacted the parent thread, and completed the parent follow-up through Dynamo. MiniMax expanded a one-child request into four spawns and skipped the final requested file action, so C5d is transport evidence but remains P2 discovery: `/ephemeral/harness-compat-artifacts/20260801T1900Z-codex-subagent-compact`.
- Codex forked a completed coding thread, completed a tool turn in the fork, then completed a later tool turn in the original thread: eight Responses streams all reached `response.completed`. The fork's file existed before resuming the root, whose workspace view then restored independently; Codex also emitted one native command-approval callback for the fork: `/ephemeral/harness-compat-artifacts/20260801T031202Z-codex-thread-fork-branch-isolation`.
- Codex detached review created a separate review thread, completed its Responses streams, and emitted native `enteredReviewMode` and `exitedReviewMode` items in two clean runs: `/ephemeral/harness-compat-artifacts/20260801T032503Z-codex-detached-review`, `/ephemeral/harness-compat-artifacts/20260801T2200Z-codex-detached-review-rerun`. Retain C5b as a P1 weekly sentinel.
- Codex inline review kept reviewer lifecycle items on the root thread, then that same thread completed a later shell-tool turn with a file effect in two clean runs: `/ephemeral/harness-compat-artifacts/20260801T1735Z-codex-inline-review`, `/ephemeral/harness-compat-artifacts/20260801T1745Z-codex-inline-review-rerun`.
- Codex persisted a completed coding thread, then a fresh app-server process resumed the exact thread ID and completed a state-dependent tool turn: ten Responses streams all reached `response.completed`: `/ephemeral/harness-compat-artifacts/20260801T032959Z-codex-thread-resume`.
- Codex archived a completed coding thread, restored its rollout, rehydrated it with `thread/resume`, and completed a later state-dependent tool turn. The archive and unarchive notifications both arrived, and every Responses stream reached `response.completed`: `/ephemeral/harness-compat-artifacts/20260801T1800Z-codex-thread-archive-rerun`.
- Codex injected a raw assistant `output_text` Responses item, then completed a tool turn that wrote the exact context-derived marker. The first Dynamo request carried the injected assistant message before the user turn, and the tool continuation reached `response.completed`: `/ephemeral/harness-compat-artifacts/20260801T1820Z-codex-inject-items`.
- Codex injected its `agent_message` extension with both `encrypted_content` and `input_text`, then completed a tool turn that wrote the exact handoff-derived marker. Two clean runs carried that extension into Dynamo and reached `response.completed`: `/ephemeral/harness-compat-artifacts/20260801T1840Z-codex-inject-agent-message`, `/ephemeral/harness-compat-artifacts/20260801T1850Z-codex-inject-agent-message-rerun`.
- Codex rolled back its last completed turn, then completed a later tool turn with a valid file effect: `/ephemeral/harness-compat-artifacts/20260801T033341Z-codex-thread-rollback`.
- Codex rejected a deliberately stale `turn/interrupt` request after a completed turn, then completed the next normal tool/file turn through Dynamo in two clean runs: `/ephemeral/harness-compat-artifacts/20260801T1940Z-codex-invalid-lifecycle`, `/ephemeral/harness-compat-artifacts/20260801T1950Z-codex-invalid-lifecycle-rerun`. Promote C8 to P1 weekly; the malformed lifecycle call is handled locally, while the subsequent coding request verifies that the session remains usable.
- Codex JSON-schema final output completed with the native app-server's per-turn `outputSchema`; the resulting Responses request carried `text.format` and streamed to `response.completed`: `/ephemeral/harness-compat-artifacts/20260801T025814Z-codex-structured-output-rerun`.
- Claude Code manual compaction produced a native compact boundary and a successful post-compact turn: `/ephemeral/harness-compat-artifacts/20260801T010702Z-claude-compact`.
- Claude Code JSON-schema final output completed a tool-edit task. The first Messages request carried `output_config.format`; tool follow-ups remained accepted without it: `/ephemeral/harness-compat-artifacts/20260801T023222Z-claude-structured-output`.
- Claude Code persisted a completed tool session, then a second native CLI process resumed it and completed another tool loop that read prior state and wrote `resumed_session.txt`: eight Messages streams all ended in `message_stop`: `/ephemeral/harness-compat-artifacts/20260801T031556Z-claude-session-resume`.
- Claude Code resumed the same saved history with `--fork-session`, then completed a tool loop in the new session and wrote `resumed_session.txt`: nine Messages streams all ended in `message_stop`: `/ephemeral/harness-compat-artifacts/20260801T031915Z-claude-fork-session`.
- Claude Code completed a Bash/file task, then exited zero after the stream-json driver closed stdin. The result and terminal stream were present before close in two clean runs: `/ephemeral/harness-compat-artifacts/20260801T2000Z-claude-baseline-eof`, `/ephemeral/harness-compat-artifacts/20260801T2010Z-claude-baseline-eof-rerun`. Promote A1k to P1 weekly.
- Codex ran a native shell command with exit code `1`, then recovered in the same turn by creating and reading `tool_failure_recovered.txt`: `/ephemeral/harness-compat-artifacts/20260801T044528Z-codex-tool-failure-recovery`.
- Claude Code emitted a failed `tool_result` block into a following Messages request, then recovered by creating and reading `tool_failure_recovered.txt`: `/ephemeral/harness-compat-artifacts/20260801T044255Z-claude-tool-failure-recovery`.
- Codex experimental plan collaboration mode accepted the native `thread/settings/update` capability negotiation, completed a plan-mode shell-inspection turn, and made four Responses requests: `/ephemeral/harness-compat-artifacts/20260801T0542Z-codex-collaboration-plan-rerun`.
- Codex goal mode completed one real `create_goal`, `get_goal`, and `update_goal` call around a shell-created and read-back verification file on four clean runs. Dynamo accepted the complete Responses continuation sequence and Codex emitted native `thread/goal/updated` lifecycle notifications: `/ephemeral/harness-compat-artifacts/20260801T0940Z-codex-goal-lifecycle`, `/tmp/dynamo-harness-compat/goal-lifecycle-rerun-20260801T1605Z`, `/ephemeral/harness-compat-artifacts/20260801T1620Z-codex-goal-error-fingerprint`, and `/ephemeral/harness-compat-artifacts/20260801T2315Z-codex-goal-lifecycle-rerun`.
- The first serialized goal candidate reached the same four native calls but its fourth Responses stream ended after HTTP 200 output deltas without `response.completed`; Codex surfaced native errors and failed the turn. The artifact was also interrupted before its driver wrote a result, which the parent runner now classifies reliably. Keep C1m as a weekly discovery case until that missing-terminal boundary is reproduced and traced: `/ephemeral/harness-compat-artifacts/nightly-goal-expanded-canary/20260801T160218Z-05-codex-goal_lifecycle`.
- Codex accepted an explicit native `skill` input, expanded it into ordinary developer/user context, and completed the required skill-directed shell/file loop through four terminal Responses streams: `/ephemeral/harness-compat-artifacts/20260801T1810Z-codex-skill-input-reader-limit`. The initial attempt exposed only the lab's default 64 KiB app-server stdout line limit; raising that reader limit to 1 MiB made the identical native workflow pass. A follow-up completed its Responses streams but MiniMax skipped the required command, so C1n remains P2 discovery rather than a weekly gate: `/ephemeral/harness-compat-artifacts/20260801T1830Z-codex-skill-input-rerun`.
- Codex client-owned dynamic tool execution completed one native `item/tool/call` callback, continued through three successful Responses requests, and created `dynamic_tool.txt` from the client-supplied result: `/ephemeral/harness-compat-artifacts/20260801T0617Z-codex-dynamic-tool-rerun`.
- Codex also completed a typed client-owned dynamic tool: its native callback carried the exact required `{left: 20, right: 22}` integer object, then the Responses continuation created and read a file from the returned value: `/ephemeral/harness-compat-artifacts/20260801T1640Z-codex-dynamic-tool-schema`.
- Codex completed both a namespaced dynamic-tool callback and a failed dynamic-tool callback followed by a shell recovery effect: `/ephemeral/harness-compat-artifacts/20260801T0620Z-codex-dynamic-namespace-tool` and `/ephemeral/harness-compat-artifacts/20260801T0623Z-codex-dynamic-tool-failure`.

## Codex automatic approval reviewer

- The installed Codex 0.144.0 app-server starts a native automatic-review subagent with the fixed model name `codex-auto-review`, distinct from the root model. A single-name MiniMax deployment therefore completed the reviewer lifecycle fail-closed after two reviewer-model 404s: `/ephemeral/harness-compat-artifacts/20260801T0631Z-codex-approval-auto-review-model-id`.
- Setting `review_model` in the isolated client configuration did not alter that 0.144.0 reviewer selection: `/ephemeral/harness-compat-artifacts/20260801T0640Z-codex-approval-auto-review-review-model`.
- Dynamo already supports this cleanly through SGLang served-model aliases. Launching the same TP4 worker with primary `MiniMaxAI/MiniMax-M2` and alias `codex-auto-review` registered both `/v1/models` entries and made the native reviewer lifecycle complete with three reviewer and four root Responses requests, all HTTP 200 and all seven SSE streams terminal: `/ephemeral/harness-compat-artifacts/20260801T0629Z-codex-approval-auto-review-alias-pass`.
- MiniMax did not perform the requested post-review write in that rerun, so the scenario remains `not_reached` as a semantic coding workload. The reviewer transport, model routing, and lifecycle were nevertheless exercised successfully; retain it as a bounded P2 discovery case rather than a nightly file-effect gate.

## Native MCP tool coverage

- A content-free local stdio MCP fixture exposing one fixed answer reached a complete Codex `mcpToolCall` start/completion pair, a shell-derived file effect, and three terminal Responses streams: `/ephemeral/harness-compat-artifacts/20260801T0650Z-codex-mcp-tool-rerun`.
- The same fixture through Claude Code's `--mcp-config --strict-mcp-config` path completed a shell-derived file effect after two native MCP tool uses and five terminal Messages streams: `/ephemeral/harness-compat-artifacts/20260801T0710Z-claude-mcp-tool-rerun`.
- A failure-only variant of the fixture completed Codex recovery after one native MCP start/completion pair and three terminal Responses streams: `/ephemeral/harness-compat-artifacts/20260801T0720Z-codex-mcp-tool-failure`.
- The same failure variant made Claude Code emit an errored `tool_result` in a following Messages request, then complete a recovery file effect through five terminal Messages streams: `/ephemeral/harness-compat-artifacts/20260801T0730Z-claude-mcp-tool-failure`.
- None of the four MCP runs introduced a content-free protocol discriminator beyond the current accepted baseline. Keep the ordinary and error variants as P1 discovery/weekly candidates until one additional clean rerun establishes their model reach signals.
- Codex 0.144.0 negotiates MCP `2025-06-18` and advertises form/url elicitation, but its headless MCP client returns `decline` for both standard `form` and the app-server's `openai/form` extension before an app-server `mcpServer/elicitation/request` callback. The tool loop recovers cleanly, so retain this as C1k discovery, not a Dynamo defect: `/ephemeral/harness-compat-artifacts/20260801T0800Z-codex-mcp-elicitation-negotiated` and `/ephemeral/harness-compat-artifacts/20260801T0810Z-codex-mcp-openai-form-elicitation`.
- Claude Code 2.1.220 negotiates MCP `2025-11-25` and emits a native stream-json `control_request` with subtype `elicitation`. Its stream-json stdin accepts text user events only, so the probe records that boundary and ends `not_reached` without pretending to supply an unsupported response: `/ephemeral/harness-compat-artifacts/20260801T0830Z-claude-mcp-elicitation-control-boundary`.
- Codex supplied a standard MCP progress token and the fixture emitted one valid `notifications/progress` event, but app-server 0.144.0 produced no `item/mcpToolCall/progress` event. The same result held with a 50 ms delay before the terminal result, ruling out back-to-back fixture framing. The MCP call and subsequent shell/file loop completed normally through Dynamo, so retain C1l as a P3 harness-discovery signal: `/ephemeral/harness-compat-artifacts/20260801T0850Z-codex-mcp-progress-traced` and `/ephemeral/harness-compat-artifacts/20260801T0920Z-codex-mcp-progress-delayed`.
- Claude Code supplied the same standard progress token, accepted the notification, and completed two clean MCP-progress runs with the expected Messages tool/result continuation and file effect: `/ephemeral/harness-compat-artifacts/20260801T0900Z-claude-mcp-progress` and `/ephemeral/harness-compat-artifacts/20260801T0910Z-claude-mcp-progress-rerun`. Promote A1i to P2 discovery/weekly candidate; it does not yet belong in the small nightly core.
- Claude Code advertised `roots`, returned a roots list to the fixture's `roots/list` request, then completed the native MCP tool/result continuation and file effect on two clean runs. The fixture retained only the root count and result keys: `/ephemeral/harness-compat-artifacts/20260801T1650Z-claude-mcp-roots` and `/ephemeral/harness-compat-artifacts/20260801T1700Z-claude-mcp-roots-rerun`. Promote A1j to a P2 weekly candidate; it remains outside the small nightly core.

## Codex native user-input capability

- With `experimental_request_user_input = true`, Codex 0.144.0 advertised `request_user_input` in each captured Responses `tools` array. Dynamo accepted two complete Responses streams without a protocol or routing error: `/ephemeral/harness-compat-artifacts/20260801T0755Z-codex-request-user-input-tool-advertisement`.
- A second strict native probe completed the requested file effect without emitting `item/tool/requestUserInput`; MiniMax again bypassed the advertised tool rather than exercising the controller response: `/ephemeral/harness-compat-artifacts/20260801T1710Z-codex-request-user-input-rerun`.
- MiniMax did not elect the native tool in either probe, so C1j remains `not_reached`, not a compatibility pass. The controller now recognizes `item/tool/requestUserInput` and replies with the documented option-id mapping when a model does reach it. Retain C1j as P2 discovery until a model produces the server request.

## First nightly distillation

- Keep the encrypted Codex child-handoff regression as a focused Rust protocol unit. The bounded native `inject_agent_message` probe now covers the same `agent_message` and `encrypted_content` wire shape without relying on model-directed subagents.
- Use Codex `steer_after_tool`, `compact`, `structured_output`, `inject_agent_message`, and `tool_failure`; and Claude Code `compact`, `structured_output`, `resume`, and `tool_failure` as the preliminary native nightly set. Their reach signals are deterministic in this campaign and each exercises a distinct current harness discriminator or lifecycle.
- Rotate one endpoint-native injected error and one mid-loop SSE truncation case across both harnesses. Preserve observed dispositions; do not require their retry policies to match.
- Keep Codex child-agent execution as a bounded weekly discovery run. A candidate rerun expanded a prompt requesting one child into four agent sessions, 35 successful Responses requests, and 33 completed streams before the six-minute observation cap, with no protocol error: `/ephemeral/harness-compat-artifacts/20260801T034255Z-codex-subagent-nightly-candidate`. This is useful drift evidence but is not a stable nightly completion oracle.
- The first serialized canary of the preliminary live set passed all eight cases against TRY-67676: `/ephemeral/harness-compat-artifacts/nightly-canary/`. It covered the six core cases plus HTTP 409 for each endpoint family; every artifact has a terminal pass result.
- Codex and Claude Code tool-failure recovery both passed a second clean live run. Promote `C1b` and `A1d` into the next nightly-core expansion; retain the original canary as the six-core baseline until the expanded set has its own full canary.
- The expanded nightly core completed its first full serial canary, 10/10 passes: `/ephemeral/harness-compat-artifacts/nightly-expanded-canary/`. This was the prior nightly baseline: eight core workflows plus the Codex and Claude Code 409 paths.
- The same 10 cases completed again with the accepted content-free protocol baseline enforced: `/ephemeral/harness-compat-artifacts/nightly-drift-gated-canary/`. All passed and no header, request field, input/content type, tool, or output-format discriminator was added.
- The fully expanded runner canary passed 12/12 against TRY-67676: `/ephemeral/harness-compat-artifacts/nightly-expanded-drift-sse-canary/`. It covered the eight core workflows, both HTTP 409 paths, and a three-event mid-loop SSE truncation for each harness; no protocol drift artifact was written.
- The agent-handoff expansion passed a clean 13/13 gated canary against TRY-67676: `/ephemeral/harness-compat-artifacts/nightly-agent-handoff-gated-canary/`. It covered nine core workflows, including injected Codex `agent_message.encrypted_content`, plus both HTTP 409 and three-event SSE-truncation paths; no protocol drift artifact was written.
- The first serialized `--include-weekly` canary completed all 17 behavioral cases against TRY-67676. Its drift gate correctly stopped promotion for two previously unseen review-only headers, `x-codex-parent-thread-id` and `x-openai-subagent`. Both have identical evidence in independent inline and detached review artifacts, so the Codex baseline now admits exactly those two names; rerun the full gate before treating the weekly suite as green: `/ephemeral/harness-compat-artifacts/nightly-weekly-sentinel-canary/`.
- The baseline-updated weekly rerun passed 17/17 against TRY-67676 with zero protocol-drift artifacts: `/ephemeral/harness-compat-artifacts/nightly-weekly-gated-rerun/`. It covered nine daily core workflows, four stable weekly lifecycle sentinels, both endpoint-native 503 paths, and both mid-loop SSE truncations.
- A second 8/8 stable-core calibration captured the advertised tool-name sets before enforcing them: Codex has `collaboration`, `create_goal`, `exec_command`, `get_goal`, `request_user_input`, `update_goal`, `update_plan`, `view_image`, and `write_stdin`; Claude Code has `Bash`, `Edit`, `Read`, and `StructuredOutput`: `/ephemeral/harness-compat-artifacts/nightly-tool-name-calibration/`.
- Claude's configured result budget is now one whole native workflow deadline rather than one deadline per terminal event. A one-second live cancellation rerun exited cleanly as `inconclusive`, not a false `harness_failure`: `/ephemeral/harness-compat-artifacts/20260801T0540Z-claude-total-timeout-cleanup-rerun`.

## Injected endpoint error matrix

Each one-shot fault used the endpoint-native error envelope. Every cell reached a valid harness terminal state; differing retry policies are recorded rather than normalized.

| Status | Codex disposition | Claude Code disposition |
| --- | --- | --- |
| 400 | terminal failed turn | recovered, 4 later 200 streams |
| 401 | recovered, 1 later 200 stream | recovered, 6 later 200 streams |
| 403 | recovered, 1 later 200 stream | recovered, 3 later 200 streams |
| 404 | recovered, 1 later 200 stream | recovered, 5 later 200 streams |
| 409 | recovered, 1 later 200 stream | recovered, 9 later 200 streams |
| 429 | terminal failed turn | recovered, 4 later 200 streams |
| 500 | transparent retry, 1 later 200 stream | recovered, 5 later 200 streams |
| 502 | transparent retry, 1 later 200 stream | recovered, 5 later 200 streams |
| 503 | transparent retry, 1 later 200 stream | recovered, 5 later 200 streams |
| 529 | transparent retry, 1 later 200 stream | recovered, 3 later 200 streams |

The fresh 400–503 artifacts are under `/ephemeral/harness-compat-artifacts/20260801T2030Z-codex-injected-400` through `/ephemeral/harness-compat-artifacts/20260801T2141Z-claude-injected-503`; the 409 and 529 cells are in the gated canaries cited above. The nightly runner rotates all ten statuses and accepts either a valid surfaced failure or a valid retry/recovery; a hang, malformed terminal, or polluted follow-up is a failure.

## Error and stream behavior (observed, not normative)

The one-shot proxy fault always uses the endpoint's native error envelope and never contacts Dynamo for that selected request.

| Harness | Fault | Observed disposition |
| --- | --- | --- |
| Codex | 429 | One native error notification; terminal failed turn; no retry. |
| Codex | 400 | Surfaced one native app-server error; terminal failed turn. |
| Codex | 404 | Surfaced one native app-server error, then recovered successfully. |
| Codex | 401 | Native error notification followed by one successful retry. |
| Codex | 500 | Transparent successful retry; no app-server error notification. |
| Claude Code | 401 / 429 / 500 | Successful retry and completed coding task in each run. |
| Claude Code | 400 | Retried and completed the coding task. |
| Codex | 403 | Surfaced one native app-server error, then recovered successfully. |
| Codex | 409 | Surfaced one native app-server error, then recovered successfully. |
| Codex | 503 | Transparent retry; no native app-server error. |
| Codex | 502 | Transparent retry; no native app-server error. |
| Claude Code | 403 | Retried and completed the coding task. |
| Claude Code | 409 | Retried and completed the coding task. |
| Claude Code | 404 | Retried and completed the coding task. |
| Claude Code | 503 | Retried and completed the coding task. |
| Claude Code | 502 | Retried and completed the coding task. |
| Codex | 529 | Transparent retry; no native app-server error. |
| Claude Code | 529 | Retried and completed the coding task. |
| Codex | 429 during an active tool loop | The current turn failed after partial tool side effects; a fresh follow-up turn completed and created `error_recovery.txt`. |
| Claude Code | 429 during an active tool loop | Retried and completed the original coding task. |
| Codex | SSE truncated after three event lines | Recovered and completed the task. |
| Claude Code | SSE truncated after three event lines | Recovered and completed the task. |
| Codex | SSE truncated on the second request of a tool loop | Recovered, completed its fourth Responses request, and created the target file. |
| Claude Code | SSE truncated on the second request of a tool loop | Recovered, completed six Messages requests, and created the target file. |

The exact artifact directories are named `codex-expected-*`, `claude-*-once`, and `*-sse-truncate` under `/ephemeral/harness-compat-artifacts/`. The mid-loop stream cases are `/ephemeral/harness-compat-artifacts/20260801T014918Z-codex-midloop-sse-truncate` and `/ephemeral/harness-compat-artifacts/20260801T015039Z-claude-midloop-sse-truncate`.

## Open questions

- Claude Code's Agent use is model-sensitive on MiniMax. One long run emitted native `task_started` and completed task lifecycle events, while bounded retries ended without an Agent task. Keep this as a discovery workload, not a nightly pass criterion, until a stronger reach signal is available.
- A further native Agent probe completed successfully with two Bash uses and no Agent task lifecycle, confirming the current MiniMax limitation without any Messages transport error: `/ephemeral/harness-compat-artifacts/20260801T1630Z-claude-agent-repeat`.
- Closing stream-json input immediately after the native Agent instruction likewise emitted zero Agent task events and exhausted the bounded workflow timeout without a Dynamo transport error: `/ephemeral/harness-compat-artifacts/20260801T2300Z-claude-agent-eof`.
- Enabling Claude Code's native `--forward-subagent-text` did not change that MiniMax adherence limitation: the terminal result was successful, but no Agent lifecycle or parent-linked forwarded child block was emitted: `/ephemeral/harness-compat-artifacts/20260801T032243Z-claude-agent-forwarded`.
- Sending a second Claude stream-json user message while the first turn is requesting is accepted and opens a second requesting turn, but the current steering prompt has not deterministically yielded the requested steered file. Treat it as an inconclusive harness-behavior finding and cover the interactive cancellation surface separately.
- The native Claude terminal cancellation controller now deterministically reaches real Messages request → `message_start` → Escape → post-Escape `message_stop` → ready terminal input → committed steering turn. On MiniMax, that steering turn made seven successful Messages requests/six terminals but did not create the requested file: `/ephemeral/harness-compat-artifacts/20260801T042117Z-claude-interactive-cancel-followup`. This is model adherence after a reached native lifecycle, not a Dynamo transport failure; retain it as a bounded discovery workload until a model yields a stable file reach signal.
- Claude Code accepted a 25k-token synthetic context plus a successful follow-up against Dynamo, but emitted no automatic `compact_boundary` with `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=1`: `/ephemeral/harness-compat-artifacts/20260801T025222Z-claude-auto-compact-25k-retry`. This is a client threshold/configuration discovery result, not a Dynamo failure; retain manual compaction as the deterministic coverage and do not promote automatic compaction until the native client exposes a reproducible trigger.
- A native Codex `localImage` input was accepted and stored in the isolated session journal, but the configured `wire_api = "responses"` local provider omitted it from the outgoing `input` (only `input_text` reached Dynamo): `/ephemeral/harness-compat-artifacts/20260801T025656Z-codex-image-input`. This does not test Dynamo image support; do not retain it as a compatibility scenario until Codex emits a real `input_image` request for this provider.
- Claude Code completed two prompt-suggestions-flagged coding tasks but emitted no native `prompt_suggestion` event after either terminal result. The corrected probe classifies this as `not_reached`, not transport failure: `/ephemeral/harness-compat-artifacts/20260801T0554Z-claude-prompt-suggestions-rerun`.
- Claude Code's separate background-agent service rejected the current custom-endpoint session before sending any Messages request. Retain the native background probe so it will automatically exercise Dynamo if a client/provider version enables the feature; today it is a client preflight `not_reached`, not a Dynamo incompatibility: `/ephemeral/harness-compat-artifacts/20260801T0612Z-claude-background-agent-preflight`.
