# Agent API replay fixtures

These fixtures are minimized, hand-authored chat-completions SSE traces. They
preserve the response shapes used while investigating DYN-2764 and the earlier
PR #8284; they are not unmodified captures from a hosted model backend.

The fragmented-tool sequence is a minimized reconstruction of the ordering
recorded in PR #8284: its first tool delta carried the call ID, function name,
and empty arguments, while later deltas supplied the JSON arguments. That
snapshot did not record a backend name or version, so these fixtures make no
backend-specific compatibility claim.

Each fixture terminates with exactly one `[DONE]` event. The scenarios cover a
plain text response, a tool call whose initially empty arguments arrive in
later fragments, parallel tool calls, and a reasoning-plus-tool-call response.
