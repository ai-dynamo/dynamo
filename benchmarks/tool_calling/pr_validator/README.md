# Guided Tool-Calling PR Validator

This harness validates the OpenAI-compatible response contract for reasoning
models used with guided tool calling. It saves each raw request and response,
then fails if reasoning, tool calls, arguments, finish reasons, or parser-marker
separation are incorrect.

## Standard Deployment

Run cases 1 through 11 against an existing aggregated deployment:

```bash
python3 benchmarks/tool_calling/pr_validator/validator.py \
  --base-url http://127.0.0.1:8000 \
  --model nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 \
  --omitted-thinking present \
  --output-dir /tmp/pr-validator/nemotron-vllm
```

Kimi uses a different request-level thinking key:

```bash
python3 benchmarks/tool_calling/pr_validator/validator.py \
  --base-url http://127.0.0.1:8000 \
  --model moonshotai/Kimi-K2.6 \
  --thinking-enabled-json '{"thinking": true}' \
  --thinking-disabled-json '{"thinking": false}' \
  --omitted-thinking present \
  --output-dir /tmp/pr-validator/kimi-vllm
```

Use `--omitted-thinking absent` for models whose default disables reasoning,
or `either` when the model default is intentionally not part of the assertion.

## Structural-Tag Deployment

Case 12 must use a separate worker launched with
`--dyn-enable-structural-tag`. Point the validator at that deployment and add:

```bash
python3 benchmarks/tool_calling/pr_validator/validator.py \
  --base-url http://127.0.0.1:8000 \
  --model MODEL \
  --structural-tag-deployment \
  --case 12 \
  --output-dir /tmp/pr-validator/MODEL-structural-tag
```

The flag records the deployment mode and enables case 12; it cannot change a
running worker's structural-tag configuration.

## Matrix

| ID | Request |
|---|---|
| 01 | `tool_choice=none` |
| 02 | `tool_choice=auto`, tool expected |
| 03 | `tool_choice=auto`, direct answer expected |
| 04 | Required, nonstreaming, thinking enabled |
| 05 | Named, nonstreaming, thinking enabled |
| 06 | Required, streaming, thinking enabled |
| 07 | Named, streaming, thinking enabled |
| 08 | Plain reasoning, thinking enabled |
| 09 | Required, thinking disabled |
| 10 | Required, thinking omitted |
| 11 | Consume a prior tool result |
| 12 | Required with `parallel_tool_calls=false` on a structural-tag deployment |

For tool-call cases, the validator requires exactly one `get_weather` call
with `{"city":"Paris","unit":"celsius"}` and
`finish_reason="tool_calls"`. It also rejects raw JSON and parser markers in
`content` or `reasoning_content`. Thinking-enabled cases require separated
reasoning; thinking-disabled cases require no reasoning.

Every run writes `summary.json`, `summary.md`, and per-case raw artifacts. A
nonzero exit code means at least one semantic check failed. This focused matrix
is intended for quick local parser checks. External deployment harnesses should
use `benchmarks/tool_calling/e2e_verifier`; its `custom` section runs the full
25-case shared qualification matrix in streaming and nonstreaming modes and
keeps behavioral verdicts separate from evaluator execution failures.
