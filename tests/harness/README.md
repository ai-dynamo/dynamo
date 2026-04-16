# Real-CLI e2e harness (`e2e_agent`)

Spins up a real `dynamo.frontend` + `dynamo.mocker` pair on free ports and
drives them with the **actual** `codex` and `claude` CLIs in non-interactive
mode. The mocker emits deterministic fake tokens so tests run in seconds
and need no GPU, but every request still flows through the real HTTP
surface, router, and protocol translators — the same code paths an
end-user's CLI would hit against a real backend.

Complementary to the upstream compliance CI (which pins OpenResponses and
Codex smoke tests against a real sglang backend): this tier catches the
wire-level translation layer and agent-loop plumbing on every PR, without
needing a GPU runner.

## Layout

```
tests/harness/
├── conftest.py               # harness_service fixture (frontend + mocker)
├── test_codex_mocker.py      # hello / tool_call / multiturn via `codex exec`
├── test_claude_mocker.py     # hello / tool_call / multiturn via `claude --print`
└── README.md                 # this file
```

## Running

```bash
pytest tests/harness/ -m e2e_agent -v
```

Tests auto-skip when the corresponding CLI is not on `PATH`, so a partial
install (only one of codex/claude present) still exercises what it can.

### Prerequisites

1. **Dynamo Python package** — `uv pip install -e ".[dev]"` from the repo
   root. The harness imports `dynamo.frontend` and `dynamo.mocker` as
   subprocesses via `python -m`.

2. **`claude` CLI** — [Claude Code](https://docs.claude.com/en/docs/claude-code).
   Install via `npm install -g @anthropic-ai/claude-code` or your package
   manager. Verify with `claude --version`.

3. **`codex` CLI** — [OpenAI Codex CLI](https://github.com/openai/codex).
   Install via `npm install -g @openai/codex` or download a release
   binary. Verify with `codex --version`.

No real API keys are needed: each test sets `ANTHROPIC_API_KEY=harness-dummy`
or `OPENAI_API_KEY=harness-dummy` and redirects `ANTHROPIC_BASE_URL` /
`OPENAI_BASE_URL` at the mocker.

## Test matrix

Each CLI exercises three scenarios against the same mocker backend:

| Scenario     | Codex (`/v1/responses`)                  | Claude (`/v1/messages`)                  |
|--------------|------------------------------------------|------------------------------------------|
| `hello`      | one-shot text prompt                      | one-shot text prompt                      |
| `tool_call`  | shell-tool definition on request          | `--allowedTools Bash` on request          |
| `multiturn`  | two sequential `codex exec` invocations   | two `claude --print` calls w/ `--resume`  |

`tool_call` asserts only that a tools-bearing request survives translation
and returns a valid reply; the mocker will not emit a `tool_use` /
`function_call` block because it only produces plain tokens.

## Why the mocker (not a real backend)

The mocker simulates KV cache, continuous batching, and realistic timing
but no ML — so every CLI run completes in under a second per turn. The
test focus is deliberately on *conformance of the protocol bridge and the
HTTP surface*, not on model quality.

Real-backend conformance is covered separately by the upstream compliance
CI against an sglang container.
