# Agent API replay fixtures

These fixtures are minimized, hand-authored chat-completions SSE traces. They
preserve the response shapes used while investigating the fragmented agent
tool-call regression in [PR #8284](https://github.com/ai-dynamo/dynamo/pull/8284)
(internally tracked as DYN-2764); they are not unmodified captures from a hosted
model backend.

The fragmented-tool sequence is a minimized reconstruction of the ordering
recorded in PR #8284: its first tool delta carried the call ID, function name,
and empty arguments, while later deltas supplied the JSON arguments. That
snapshot did not record a backend name or version, so these fixtures make no
backend-specific compatibility claim.

Each fixture terminates with exactly one `[DONE]` event. The scenarios cover a
plain text response, a tool call whose initially empty arguments arrive in
later fragments, parallel tool calls, and a reasoning-plus-tool-call response.

The parallel-tools fixture emits its two calls sequentially — all of call 0's
deltas precede call 1's, with a monotonically increasing index — which matches
how the in-tree `dynamo-parsers-v2` parsers emit parallel calls. It exercises
preserved identity and contiguous indices across sequential calls, not the
reassembly of interleaved call fragments (no current backend produces those).
