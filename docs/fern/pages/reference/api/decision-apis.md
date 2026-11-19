---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Decision API Reference
sidebar-title: Decision APIs
subtitle: Text evaluation contracts, accounting, and errors for Decisions and System One
---

> [!IMPORTANT]
> **Experimental.** These contracts require a worker with decision-scoring support.

NVIDIA Dynamo exposes two non-streaming text-evaluation contracts. For runnable curl and SDK examples, see the [Decision Model API guide](../../use-cases/decision-model/apis.md).

## Contracts

| Property | OpenAI Decisions | Jev System One |
| --- | --- | --- |
| Route | `POST /v1/decisions` | `POST /v1/systemone` |
| Client | Official OpenAI Python SDK | Official Jev/TypeSafe Python SDK |
| Shared input | `input` | `state` |
| Questions | Ordered array with optional `name` | Object keyed by question name |
| Classification | `type: "choice"`, `choices` with typed string/boolean values | `type: "choice"`, string-valued `criteria` keys |
| Binary judgment | `type: "predicate"`; answer has `probability` | `type: "noul"`; answer has `noul` |
| Ordered score | `levels` containing labels; answer has a zero-based expected `score` | Ordered `criteria`; answer has expected `score` and an index-to-label `legend` |
| Choice count | 2–255 before backend/tokenizer limits | 1–255 before backend/tokenizer limits |
| Score level count | 2–10 | 1–10 |
| Answers | Array in question order; preserves omitted, empty, and duplicate names; can include `type: "refusal"` | Object keyed by question name; a refusal produces an error instead |
| Distributions | Ordered arrays of value/probability entries | Probability maps keyed by choice or score index |
| SDK `base_url` | API root, including `/v1` | Server origin, without `/v1` |

The public contracts are the supported text-evaluation subset of [OpenAI Decisions](https://developers.openai.com/api/reference/resources/decisions/methods/create) and [Jev/TypeSafe System One](https://docs.typesafe.ai/sdk/python). Do not send SGLang-native `/v1/decisions` request fields to Dynamo. Neither route accepts `nvext`, including `{}` or `{"format":"oai"}`.

Use the exact model alias returned by `GET /v1/models`. Supported inputs, candidate counts, and accounting depend on the model/executor profile. Dynamo currently admits at most 128 questions per request, subject to independent aggregate resource limits. These APIs accept text inputs; image, audio, and video inputs are not supported by the documented profiles.

## Usage

Input usage counts rendered prompts across question executions, not only the shared input text. SGLang scores without decoding output tokens. OpenAI `total_tokens` is the sum of `input_tokens` and `output_tokens`. Cache-read values come from backend counters or a profile verified to perform no cache reads; Dynamo returns 500 rather than reporting an unmeasured read count as zero. Questions in one request can reuse cached prefixes, but separate requests do not share them.

> [!NOTE]
> Dynamo currently always returns `usage.input_tokens_details.cache_write_tokens: 0` because cache-write usage is not measured. This value does not mean that caching is disabled or that no KV blocks were written. The current Decisions response schema requires an integer, so omission, `null`, and `"N/A"` are not supported. Use `cached_tokens` for measured cache reads; it is separate from the unmeasured write count.

## Answer Statistics

`x_label_mass` is a Dynamo extension on Jev answers reporting candidate vocabulary mass, not another conditional choice probability. It is not added to OpenAI answer objects. `confidence` is an executor-derived statistic, not a calibrated probability of correctness.

## Errors and Headers

| Outcome | OpenAI Decisions | Jev System One |
| --- | --- | --- |
| Invalid or unsupported request shape | 400 | 422 |
| Executor refused a question | 200 with a `type: "refusal"` answer | 422, code `decision_refused` |
| Unknown model | 404 | 404 |
| Registered model lacks decision capability | 400 | 400 |
| Eligible worker unavailable | 503 | 503 |
| Invalid backend scoring result or usage | 500 | 500 |
| Admission capacity exhausted | Configured overload status, default 529 | Configured overload status, default 529 |
| Request deadline exceeded | 504 | 504 |

An OpenAI refusal is an answer variant, not an HTTP error. If an executor refuses any Jev question, Dynamo returns `decision_refused` for the request rather than a partial answer object.

Capacity responses have code `rate_limit_exceeded` and `Retry-After: 1`; status alone does not identify overload. OpenAI errors use a nested `error` object; Jev uses top-level error fields. Responses expose `x-request-id` and `x-typesafe-request-id`.

For enablement, admission limits, and cache isolation, see [Decision API Configuration](../components/frontend-configuration.mdx#decision-apis). For backend-specific requirements, see the [SGLang guide](../../use-cases/decision-model/sglang.md).
