<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multiple choices: schema defaults versus observed behavior

A missing OpenAPI default annotation does not prove a different server default.
In the pinned paired-server assessment below, both Dynamo and direct vLLM
returned one choice when `n` was omitted, although only vLLM declared
`default: 1`. The omitted-default concern was a false positive; the entire
`n` field was not compatible in every tested case.

Use this case when reviewing schema differences: check the requested effect,
then classify omission, bounds, coercion and null handling separately. This is
historical behavioral evidence, not a current-release compatibility guarantee.
It changes no runtime behavior, schema, test or policy. For new schema
assessments, use the [current response assessment](dynamo-vllm-response-assessment.md).

## Historical scope

The probe compared POST `/v1/chat/completions` and `/v1/completions`, each with
unary JSON and streaming server-sent events (SSE). It followed the original
`n` candidates `REQUEST-f3ad9b33807f` (chat) and `REQUEST-34ff6995732a`
(completion).

| Input | Pinned identity |
| --- | --- |
| Dynamo source | `7d31af16e147e48f88914550da31cb7cc59ffbfa` |
| Startup prerequisite | [#16073](https://github.com/ai-dynamo/dynamo/pull/16073), revision `1f5429de07327ca36427bca90a64920d7d513a78` |
| Direct vLLM and Dynamo's vLLM backend | `b04e80127bea278758743ca3277d40a234f73b09` |
| Model and tokenizer | `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Parsers | Qwen3 reasoning; Hermes tool calls |

The historical [source and launch profile](https://github.com/ai-dynamo/dynamo/blob/db36fbfee491cf1cdc4dfb01a62d616c69e5fd99/scripts/protocol_compatibility/frameworks/vllm-openapi32.json)
records the environment and shared engine settings. Its initial Dynamo revision
predates this follow-up; the Dynamo revision in the table is the follow-up source.
The probe reused a runtime wheel built from the production-equivalent
pre-commit tree, rather than performing a fresh committed-source build.
These pins describe the historical execution, not dependencies that must merge
before this documentation.

## Repeat the controlled requests

Use the same source, model/tokenizer and parser configuration on both servers.
Supply the served model name explicitly.

1. For chat, ask for exactly the word `hello` and set
   `chat_template_kwargs: {"enable_thinking": false}`.
2. For completions, use this prompt so the configured reasoning phase has ended:

   ```text
   <|im_start|>user
   Reply with exactly the word hello.<|im_end|>
   <|im_start|>assistant
   <think>

   </think>

   ```

3. Set `temperature: 1`, `seed: 0`, `max_tokens: 4`, and
   `allowed_token_ids: [22]`. Token `22` means `7` for the pinned tokenizer;
   verify the token identity before using another tokenizer.
4. Send each applicable row below to both servers with `stream: false` and
   `stream: true`. Change temperature to zero only for the greedy control.

The singleton allowed-token set makes output deterministic without greedy
sampling. Both servers reject `n > 1` at temperature zero, so that combination
cannot establish whether valid multi-choice generation works.

## Observed results

| `n` input | Direct vLLM | Dynamo | Interpretation |
| --- | --- | --- | --- |
| Omitted or integer `1` | One choice | One choice | Effective default agrees |
| Integer `2` or `128` | Requested choice count | Requested choice count | Valid multi-choice effect agrees |
| `null`, chat | One choice | One choice | Nullable input agrees |
| `null`, completion | HTTP 400 | One choice | Dynamo-only/additive admission |
| Integer `129` | 129 choices | HTTP 400 | Limit divergence; separate policy question |
| String `"2"` | Two choices | HTTP 400 | Reference coercion outside its integer schema |
| Integer `0` or `-1` | HTTP 400 | HTTP 400 | Shared rejection, not equal error envelopes |
| Integer `2`, temperature `0` | HTTP 400 | HTTP 400 | Invalid greedy multi-choice control |

All successful choices had indices `0..n-1`, text `7777`, and exactly one
`length` finish per choice. Unary responses contained each index once. Streaming
responses combined deltas within each index and ended with exactly one `[DONE]`,
with no error events. These checks established the per-choice effect, not just
HTTP acceptance.

The matrix contained 40 paired cases, or 80 HTTP exchanges: 18 matching valid
pairs, four limit divergences, four coercion differences, two additive-null
pairs, and 12 shared rejections. The raw audit inspected requests, statuses,
headers and bodies. Its initial assumption of a direct `properties` request root
failed on Dynamo's `allOf`; the corrected audit followed that composition.
No server was changed or rerun to repair the checker. A successful audit confirms
the observations, not compatibility: the matrix still contains real differences.

## Disposition and limits

Close only the omitted-default concern. Do not change runtime behavior or
suppress all absent-default findings on the strength of this example.

Dynamo's historical `n <= 128` limit was introduced in
[#1674](https://github.com/ai-dynamo/dynamo/pull/1674); the `batch_size × n <= 128`
limit followed in [#3953](https://github.com/ai-dynamo/dynamo/pull/3953).
This single-prompt probe does not qualify batch interactions or justify changing
either limit. Likewise, accepting a numeric string does not make it part of
vLLM's exported integer contract.

The evidence does not establish error-envelope equality, arbitrary models or
parsers, mixed-version deployments, cancellation behavior, or compatibility
of current releases. No new inference was run for this documentation cleanup.

For comparison tooling, preserve absent-default differences as investigation
candidates. A future regression fixture could pair a missing annotation with
verified identical omission behavior, while retaining separately demonstrated
effective-default differences. Such a fixture is a follow-up, not part of this
documentation change.
