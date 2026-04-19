# Harness replay fixtures

One line per pre-recorded `chat.completion.chunk`. Loaded by
`tests/common/replay_engine.rs` and replayed into Dynamo's `/v1/messages`
(Anthropic) and `/v1/responses` (Codex) endpoints by the integration tests
in `anthropic_replay.rs` and `responses_replay.rs`.

Fixtures are handwritten — no model weights, tokenizer, or live engine is
involved. Add new fixtures here for additional scenarios rather than
modifying existing ones (each fixture is paired with a committed `insta`
snapshot).

Lines beginning with `//` (after optional whitespace) and blank lines are
ignored by the loader so JSONL files can include short comments for
readability.

Naming convention: `<series>__<scenario>.chunks.jsonl` where `<series>` is
`claude_code` (A-series / Anthropic) or `codex` (R-series / Responses).
