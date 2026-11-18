<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SGLang chat processor replay against parser conformance

This replays the frontend-crates Unified conformance corpus through Dynamo's SGLang chat processor (`--dyn-chat-processor sglang`) and compares the result with native SGLang on identical inputs. It implements the output-replay part of [DEP #18601](https://github.com/ai-dynamo/dynamo/issues/18601) for the Python SGLang processor.

The processor uses SGLang's own parser classes, but the frontend-crates report measures those classes directly. It does not measure the Dynamo code around them, which decides when tool calls are emitted, which calls are kept, and how the final output is rebuilt.

## What runs

| Side | Component under test | Source |
|---|---|---|
| Native | SGLang `ReasoningParser` and `FunctionCallParser` stream calls | frontend-crates `conformance/utils/src/capture_sglang_unified.py`, unmodified |
| Dynamo | `create_parsers` and `SglangStreamingPostProcessor` from `dynamo.frontend.sglang_prepost` | [`capture_dynamo_sglang_unified.py`](capture_dynamo_sglang_unified.py), built on [`dynamo_sglang_rows.py`](dynamo_sglang_rows.py) |

Both sides use the same frozen jobs (frontend-crates `build_job`), the same support gate, and the same assembly (`assemble_stream`). Nothing in frontend-crates is modified.

**Boundary:** each text chunk enters `SglangStreamingPostProcessor._process_text`, the step after detokenization. The final chunk carries the finish reason, as the last token batch does in production. This bypasses incremental detokenization, stop-token and stop-string filtering, token-id delimiter gating, logprobs, and the Rust HTTP/SSE projection. Differences found here are a lower bound for the full path.

**Integrity check:** the native run must reproduce the committed `sglang_python` captures when the installed SGLang version matches the published one. `--require-published-match` fails the run otherwise.

## Run it

Use an environment that has `sglang` and the Dynamo frontend importable, such as the SGLang dev container. Then point it at a frontend-crates checkout:

```bash
git clone https://github.com/ai-dynamo/frontend-crates /tmp/frontend-crates
FRONTEND_CRATES_DIR=/tmp/frontend-crates \
  python3 tests/parity/sglang_processor/run.py \
  --output /tmp/sglang-replay --require-published-match
```

Use `--families glm47 qwen3` to limit the run, and `--strict` to exit 1 when any final output differs from native SGLang. The output directory contains:

- `captures/<family>.native.yaml` and `captures/<family>.dynamo.yaml`: per-chunk rows in the harness format
- `comparison.json`: per-case states, assembled outputs, difference category, and provenance (frontend-crates and Dynamo revisions, SGLang version)
- `summary.md` and `report.html`: totals per family, and every case whose final output differs, with native and Dynamo output side by side

The adapter contract tests do not need frontend-crates:

```bash
python3 -m pytest tests/parity/sglang_processor/test_dynamo_sglang_rows.py -q
```

## Current result

On frontend-crates `4608c17`, Dynamo `807aaaa0ac` and SGLang 0.5.23, the native run reproduced 2,015 of 2,015 committed captures. Of 2,980 cases that ran on both sides:

| | Cases |
|---|---|
| Same final output | 2,776 |
| Same output chunk by chunk | 309 |
| Different final output | 204 |

The differences come from the processor's wrapper: it holds tool calls until the finish chunk, drops calls with unknown names or invalid JSON arguments, and re-parses the whole text at finish.

| Difference | Cases |
|---|---|
| Same content, tool calls moved after text | 125 |
| Dynamo returns fewer tool calls | 34 |
| Same call count, different names or arguments | 28 |
| Different text or reasoning | 9 |
| Dynamo returns more tool calls | 8 |

`gpt_oss` and `hermes` have no registered SGLang stream pipeline in frontend-crates, so neither side runs them.

## Not covered

- Token-id replay through detokenization and delimiter gating. This needs per-family tokenizers.
- Input parity (prompt tokens and effective controls), which is a separate DEP #18601 item.
- A column in the frontend-crates report. The captures use the harness format so that they can be imported later.
- CI. The run needs a frontend-crates checkout and has known differences, so it is not a merge gate.
