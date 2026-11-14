---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Parser Compatibility Checks
subtitle: Compare observed prompt tokens and generate a scoped compatibility matrix
---

> [!NOTE]
> **Experimental.** This first delivery checks non-streaming, text-only Chat Completions prompt token IDs. It does not certify output parsing, streaming, generation constraints, media, or complete API compatibility. The broader design is [DEP #18601](https://github.com/ai-dynamo/dynamo/issues/18601).

## Scope

`python3 -m dynamo.compatibility` attaches to an existing NVIDIA Dynamo endpoint and a separate native SGLang endpoint. It sends the same semantic request to both and compares the actual returned prompt token arrays element by element. The checker does not provision servers or load model weights. Requests use `max_tokens=1`, `temperature=0`, `n=1`, and `stream=false`; the existing engines may execute that bounded generation.

Use the same checkpoint, tokenizer, template, thinking defaults and parser configuration on both endpoints. `/v1/models` verifies the requested model name is advertised; it does not verify artifact revisions or the active processor. Build, revision and processor values in the manifest are explicitly **caller-declared**, not server-verified provenance.

## Run a Check

Start from the [Qwen text-input smoke profile](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/compatibility/profiles/qwen3-text.json). Copy it outside the checkout and replace every `SET_TO_...` provenance placeholder with the deployed revision, builds and processor. The shipped example cannot pass with placeholders unchanged.

The profile contains three required cases: single-turn text, a system message with Unicode/whitespace, and historical tool calls/results. This is a small smoke scope, not a complete tool or thinking inventory. Add cases and their IDs to `required_cases` to expand the declared scope. Missing required requests block execution before any inference requests are sent.

```bash
python3 -m dynamo.compatibility \
  --manifest /path/to/compatibility.json \
  --dynamo-url http://127.0.0.1:8000 \
  --sglang-url http://127.0.0.1:30000 \
  --timeout 30 \
  --output /path/to/compat-results
```

URL arguments are HTTP(S) origins, without a `/v1` suffix, user information, query or fragment. Redirects are not followed. Use `DYNAMO_COMPAT_DYNAMO_API_KEY` and `DYNAMO_COMPAT_SGLANG_API_KEY` environment variables when endpoints need bearer authentication. Keys are not included in reports. The timeout is per socket operation, not a total execution deadline. Each response and manifest is limited to 4 MiB, and each run declares at most 128 cases.

In a source checkout without an editable installation, prefix the command with `PYTHONPATH=components/src`. The module uses the Python standard library; backend dependencies remain with the running servers.

## Manifest Contract

The JSON object has exactly these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer `1` |
| `profile` | Nonempty scope identifier; change it when changing the support claim |
| `model` | Exact served model ID expected from both endpoints |
| `provenance` | Nonempty declarations for `model_revision`, `dynamo_build`, `sglang_build` and `processor` |
| `required_cases` | Nonempty list of unique required case IDs, frozen before execution |
| `cases` | Objects containing only `id` and `request`; every ID must be declared and unique |

Requests accept text messages, tools/tool choices, parallel-tool policy, template arguments, thinking/effort controls and assistant-continuation settings. The checker rejects direct token inputs, media content blocks, unknown top-level request fields and client-supplied transport/diagnostic overrides. It preserves message/tool order and does not normalize arguments or whitespace.

Dynamo receives `nvext.extra_fields=["prompt_token_ids"]`; the checker reads response-level `nvext.prompt_token_ids`. Native SGLang receives `return_prompt_token_ids=true`; the checker reads `choices[0].prompt_token_ids`. There is no fallback to prompt-token counts, re-rendered text or guessed token IDs. A deployment without these diagnostics is blocked, not certified.

## Results and Exit Status

| Status | Meaning |
| --- | --- |
| `PASS` | Every required case has equal nonempty observed token arrays |
| `FAIL` | Token arrays differ; the first mismatch and bounded token neighborhoods are retained |
| `MISSING` | A declared required case has no request |
| `BLOCKED` | Invalid scope, endpoint failure, missing diagnostics or malformed evidence prevents checking |

Exit status is `0` only for a nonempty complete passing scope, `1` for a failed/blocked check, and `2` when reports cannot be written. Reports are written for normal validation and endpoint failures, including failed comparisons; abrupt process termination or an unwritable output location cannot guarantee reports.

## Generated Matrix

Each run writes:

- `results.json`: manifest and SHA-256, declared provenance, observed model lists, timestamps, token arrays and per-case results.
- `compatibility-matrix.md`: documentation-ready scoped case table.
- `compatibility-matrix.html`: escaped table and run metadata for inspection.

Example layout only; the following is not measured model qualification:

| Model | Profile | Case | Input Token Parity | Details |
| --- | --- | --- | --- | --- |
| Qwen/Qwen3-0.6B | qwen3-text-input-smoke-v1 | single-turn | PASS | Equal observed tokens |
| Qwen/Qwen3-0.6B | qwen3-text-input-smoke-v1 | tool-history | FAIL | First mismatch at token 42 |

Reports retain the untested-layer disclaimer and declared-provenance limitations. Copy a matrix into documentation only with its scope, manifest hash, versions and evidence reference. A successful check of these cases does not establish compatibility of untested requests or newer model/software revisions. Changing the manifest changes the claim; a smaller local profile does not satisfy a larger release inventory.

Use synthetic or reviewed sanitized requests. Results include the full manifest and token arrays and should be handled as test evidence, not as a place to store production prompts or credentials.

## CI Boundary

The [runner-contract workflow](https://github.com/ai-dynamo/dynamo/blob/main/.github/workflows/parser-compatibility.yml) uses deterministic HTTP fixtures to test request preservation, strict exits, missing coverage and report generation. It does not run either parser and is labeled **no model certification**. Actual model checks require the same Python command against real endpoints; missing endpoints must fail that invocation. Broader model-profile and release CI gates follow the DEP delivery plan.
