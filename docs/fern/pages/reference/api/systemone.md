---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: System One API
subtitle: Typed decisions from native SGLang candidate-token scoring
---

**Experimental.** NVIDIA Dynamo exposes `POST /v1/systemone` when the frontend starts with `--enable-systemone-api`. The endpoint evaluates `noul`, `choice`, and `score` questions about shared state and returns one JSON response. It is disabled by default and returns 404 while disabled. See [Frontend Configuration](../components/frontend-configuration.mdx#system-one-experimental) for enablement, limits, and path customization.

## Compatibility

Only aggregate SGLang workers advertising native Generate support are eligible. Workers must have no dependent worker roles, LoRA adapter, speculative decoding, or request migration (`--migration-limit 0`). The frontend must load the model tokenizer and chat template. Each candidate label must append exactly one distinct token at the answer position, and reasoning must be disabled before that position. Known always-on reasoning modes and templates that leave reasoning open are rejected. A tokenizer that splits a label into multiple tokens is rejected rather than approximated.

Models configured with always-on reasoning parsers (`deepseek_r1`, `step3`, `gpt_oss`, or `kimi`) are rejected even when template arguments request disabled reasoning. The endpoint requires the model's first answer-position distribution, not a reasoning-channel distribution.

This endpoint sends SGLang native candidate-token scoring requests with `max_new_tokens: 0` and neutral sampling settings. It does not generate an answer token or reasoning text. vLLM and TensorRT-LLM are not supported by this endpoint. Support for a model's ordinary chat/completion endpoint does not imply System One compatibility.

The weight-backed parity test targets SGLang 0.5.19 and compares this endpoint with native `/generate` candidate scoring. This comparison does not establish compatibility with every upstream System One implementation or model.

## Request

Send `Content-Type: application/json`. Field names and question types are case-sensitive.

| Field | Type | Required | Contract |
| --- | --- | --- | --- |
| `model` | string | Yes | Nonblank registered model name or registered alias. LoRA names containing `:` are rejected. |
| `state` | string, object, array, or null | Yes | Shared context rendered once per question. Objects and arrays are serialized as JSON text; strings are used directly; null renders as empty text. Numbers and booleans are rejected. |
| `questions` | object | Yes | Between 1 and 128 named questions. Object insertion order defines answer order. |
| `chat_template_kwargs` | object or null | No | Arguments passed to the model chat template. Keys containing `think` or `reason`, case-insensitively, must have value `false`, `"disabled"`, or `"none"` to disable reasoning. |

`jev-latest` resolves to the only eligible registered model when exactly one exists, unless it is already a registered name or alias. Other unknown names return 404. With multiple eligible models, name one explicitly or register an alias. The response reports the canonical model name.

`temperature`, `prompt_format_version`, and `return_prompt_token_ids` are unsupported and return 422 when non-null. Prompt format version 1 is fixed. Unknown top-level fields are ignored; unknown fields inside a question or `noul.criteria` are rejected. Generation controls such as tools, streaming, token budgets, and client-selected routing are not part of this API.

### Question Fields

Every question requires `type`. Optional `instructions` and criteria descriptions accept a string, object, array, or null. Numbers and booleans are rejected. Objects and arrays render as JSON text.

| Type | `criteria` | Validation |
| --- | --- | --- |
| `noul` | Optional object with optional `"true"` and `"false"` descriptions | Requires nonblank instructions or at least one nonblank, non-null true/false description. |
| `choice` | Required object mapping option names to descriptions | Between 1 and 255 options. Names are nonblank, at most 128 characters, and contain no control or line-break characters. Names must be unique after trimming whitespace and converting to lowercase. Null or blank descriptions are allowed. |
| `score` | Required array of ordered level descriptions | Between 1 and 10 levels. Each level must be non-null and nonblank. Array indices define score values from `0` through `N - 1`. |

The prompt uses `yes`/`no` labels for `noul`, `A` through `Z` for choices with at most 26 options, two-letter labels beginning at `AA` for larger choice sets, and digit labels for score levels. The tokenizer constraint can reduce the usable choice count below the schema maximum.

### Example Request

```json
{
  "model": "jev-latest",
  "state": {"review": "The delivery arrived on time and the package was intact."},
  "questions": {
    "positive": {"type": "noul", "instructions": "The review is positive."},
    "sentiment": {
      "type": "choice",
      "instructions": "Classify the review sentiment.",
      "criteria": {"positive": "Favorable experience", "negative": "Unfavorable experience"}
    },
    "satisfaction": {
      "type": "score",
      "instructions": "Rate customer satisfaction.",
      "criteria": ["Dissatisfied", "Neutral", "Satisfied"]
    }
  }
}
```

## Response

A successful response contains `model`, an `answers` object keyed by the request's question names in the same order, and `usage`.

| Field | Type | Contract |
| --- | --- | --- |
| `model` | string | Canonical served model name. |
| `answers` | object | Exactly one typed answer per question. |
| `usage.input_tokens` | integer | Sum of prompt tokens across all question branches, including repeated shared state and chat-template tokens. This is not a count of unique cached tokens. |
| `usage.output_tokens` | integer | Always `0`; scoring generates no output token. |

Success headers include `x-request-id` and `x-dynamo-systemone-version: 1`. A request's `x-request-id` is reused when supplied; otherwise Dynamo creates one.

| Answer Type | Fields |
| --- | --- |
| `noul` | `type: "noul"`, `noul` (probability of the `yes` label), `x_label_mass` |
| `choice` | `type: "choice"`, `choice` (selected option name), `probabilities` (option-name-to-probability object), `confidence`, `x_label_mass` |
| `score` | `type: "score"`, `score` (expected level index), `probabilities` (string-index-to-probability object), `confidence`, `legend` (string-index-to-original-description object), `x_label_mass` |

### Probability and Confidence Semantics

Let `l_i` be the model's vocabulary-normalized log probability of candidate label `i`. The API reports candidate-conditioned probabilities `p_i = exp(l_i) / sum_j exp(l_j)` using a stable softmax. They sum to 1 over the requested labels, not over the model vocabulary. `x_label_mass = sum_i exp(l_i)` reports their total vocabulary probability mass and can reveal that the model assigns little mass to the allowed labels.

Nonfinite or positive log probabilities and total label mass above `1.001` fail backend validation. Tiny mass overshoot within that rounding tolerance is clamped to `1` in the response.

`noul` returns `p_yes`. `choice` returns the option with largest `p_i`; ties select the first option in request order. `score` returns `sum_i i * p_i`, which can be fractional and lies between `0` and `N - 1`.

For `choice` with `N > 1`, `confidence = (max_i p_i - 1/N) / (1 - 1/N)`. For `score` with `N > 1`, let `m` be the first modal index, `c = (N - 1)/2`, `D = sum_i p_i * abs(i - m)`, and `U = (sum_i abs(i - c))/N`; then `confidence = max(0, 1 - D/U)`. Either confidence is 1 for a single candidate. `noul` has no separate confidence field.

> [!WARNING]
> These are model score summaries, not calibrated probabilities of real-world correctness. A high confidence value does not establish accuracy. Evaluate the chosen model and criteria against labeled examples before using a decision in a consequential workflow.

## Limits and Execution

| Limit | Value |
| --- | --- |
| HTTP request body | 4 MiB for this endpoint, independently of the larger general frontend body cap |
| Questions per request | 128 |
| Choice options per question | 255, subject to single-token label compatibility |
| Score levels per question | 10 |
| Concurrent admitted branches per frontend | 256 by default; `DYN_SYSTEMONE_MAX_INFLIGHT_BRANCHES` can reduce the limit |
| Cumulative expanded prompt tokens per request | 65,536 by default; `DYN_SYSTEMONE_MAX_INPUT_TOKENS` can reduce the limit |
| Prompt length per question | Strictly less than the model's effective context length when that length is known |

Admission reserves one permit per question before rendering/tokenization and holds the permits through dispatch. All prompts pass validation before backend dispatch. The first question establishes a worker and data-parallel rank; the remaining questions run concurrently pinned to that placement. Dynamo returns the complete answer set or one error, never partial answers. A failed branch cancels its siblings. Disconnecting the HTTP client cancels dispatched branches; blocking preflight may finish first and retains admission permits while it runs.

Preflight checks a question's prompt size immediately after its initial tokenization, before tokenizing that prompt again for candidate-label checks. The effective limit is the smaller of the remaining request token budget and the model context limit minus one token.

Each parent request receives a fresh random 128-bit cache salt shared only by its question branches. This permits within-request prefix reuse while isolating prefix-cache reuse between requests. Clients cannot set the salt or pin a worker through this endpoint. Prefix reuse is an engine optimization, not a guarantee that state tokens are processed only once.

## Errors

System One errors use `{"error":{"message":"..."}}`. The HTTP status carries the error category; responses do not contain a partial `answers` object.

| Status | Meaning |
| --- | --- |
| `400` | Malformed JSON or permanent capability/topology mismatch for a registered model. |
| `404` | Endpoint disabled, model name not registered/resolvable, or `jev-latest` fallback is ambiguous. |
| `413` | Request body exceeds the endpoint's 4 MiB cap. |
| `415` | Unsupported or missing JSON content type. |
| `422` | Invalid question/state fields, unsupported controls, incompatible tokenizer/chat template, or prompt/token limits exceeded. |
| `499` | Parent request cancelled; a disconnected client normally cannot receive this response. |
| `500` | Internal failure or malformed native scoring response, including missing, reordered, duplicate, nonfinite, or aborted candidate scores. |
| `503` | Frontend not ready, no live eligible worker, worker unavailable, or inability to preserve worker/rank placement. |
| `529` by default | Branch admission capacity exhausted. `DYN_HTTP_OVERLOAD_STATUS_CODE` can change the status; admission rejection includes `Retry-After: 1`. |

## Related Resources

- [System One Metrics](../observability/metrics-catalog.mdx#system-one-experimental)
- [SGLang Local Deployment Examples](../../recipes/cli-templates/sglang.mdx#system-one-typed-decisions-experimental)
- [System One Shell Example](https://github.com/ai-dynamo/dynamo/blob/main/examples/systemone/README.md)
- [System One Protocol Source](https://github.com/ai-dynamo/dynamo/tree/main/lib/llm/src/protocols/systemone)
