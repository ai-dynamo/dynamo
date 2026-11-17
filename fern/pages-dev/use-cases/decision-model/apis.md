---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Decision APIs
sidebar-title: APIs
subtitle: Classify and score text with the OpenAI Decisions and Jev System One contracts
---

<Info>
**Experimental.** Start an [SGLang decision worker](sglang.md) with the required scoring configuration before running these examples.
</Info>

NVIDIA Dynamo exposes two non-streaming decision APIs over one scoring pipeline. The HTTP route chooses the public contract; the registered model chooses the executor. Neither route is a chat/completions alias, and neither returns generated prose.

## Choose a Contract

| Property | OpenAI Decisions | Jev System One |
| --- | --- | --- |
| Route | `POST /v1/decisions` | `POST /v1/systemone` |
| Client | Official OpenAI Python SDK | Official Jev/TypeSafe Python SDK |
| SDK `base_url` | API root, including `/v1` | Server origin, without `/v1` |

Use the exact model alias returned by `GET /v1/models`. See the [Decision API Reference](../../reference/api/decision-apis.md) for supported fields, limits, and wire-format restrictions.

## Call OpenAI Decisions with curl

The following examples use the `Qwen/Qwen3-0.6B` alias from the [SGLang guide](sglang.md) and a trusted local server without gateway authentication. For an authenticated deployment, configure the gateway's required credentials and HTTPS endpoint for curl and the SDKs.

```bash
curl --fail-with-body http://localhost:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "input": "A customer was charged twice and requests a refund today.",
    "questions": [
      {
        "type": "choice", "name": "route",
        "instructions": "Choose the responsible team.",
        "choices": [
          {"value": "billing", "description": "Payments"},
          {"value": "technical", "description": "Software"}
        ]
      },
      {"type": "predicate", "name": "urgent", "instructions": "Action is needed today."},
      {
        "type": "score", "name": "severity", "instructions": "Rate the impact.",
        "levels": [{"label": "Low"}, {"label": "Moderate"}, {"label": "High"}]
      }
    ]
  }'
```

An illustrative SGLang response follows. These probabilities, confidence values, and input-token counts demonstrate the shape; they are not measured inference results or an accuracy claim.

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "answers": [
    {
      "type": "choice", "name": "route", "choice": "billing",
      "probabilities": [{"value": "billing", "probability": 0.8}, {"value": "technical", "probability": 0.2}],
      "confidence": 0.6
    },
    {"type": "predicate", "name": "urgent", "probability": 0.9},
    {
      "type": "score", "name": "severity", "score": 1.5,
      "probabilities": [
        {"value": 0, "label": "Low", "probability": 0.1},
        {"value": 1, "label": "Moderate", "probability": 0.3},
        {"value": 2, "label": "High", "probability": 0.6}
      ],
      "confidence": 0.25
    }
  ],
  "usage": {
    "input_tokens": 141, "output_tokens": 0, "total_tokens": 141,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

Score levels are numbered from zero: Low is 0, Moderate is 1, and High is 2. The returned `score` is their probability-weighted average: `0 × 0.1 + 1 × 0.3 + 2 × 0.6 = 1.5`. High has the largest probability (0.6), but the API returns the average (1.5), not High's index (2). Use the distribution to select the most likely level when you need a single label.

Choice values remain typed: `true` and `"true"` are different options.

## Call Jev System One with curl

Express the same questions using the System One wire contract:

```bash
curl --fail-with-body http://localhost:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "state": "A customer was charged twice and requests a refund today.",
    "questions": {
      "route": {
        "type": "choice", "instructions": "Choose the responsible team.",
        "criteria": {"billing": "Payments", "technical": "Software"}
      },
      "urgent": {"type": "noul", "instructions": "Action is needed today."},
      "severity": {"type": "score", "instructions": "Rate the impact.", "criteria": ["Low", "Moderate", "High"]}
    }
  }'
```

The equivalent illustrative SGLang response uses named answers and probability maps:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "answers": {
    "route": {
      "type": "choice", "choice": "billing",
      "probabilities": {"billing": 0.8, "technical": 0.2},
      "confidence": 0.6, "x_label_mass": 0.9
    },
    "urgent": {"type": "noul", "noul": 0.9, "x_label_mass": 0.8},
    "severity": {
      "type": "score", "score": 1.5,
      "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
      "legend": {"0": "Low", "1": "Moderate", "2": "High"},
      "confidence": 0.25, "x_label_mass": 0.9
    }
  },
  "usage": {"input_tokens": 141, "output_tokens": 0}
}
```

See the [answer-statistic definitions](../../reference/api/decision-apis.md#answer-statistics) before using `x_label_mass` or `confidence` in application logic.

## Use the Official OpenAI SDK

Install the recorded client versions in a separate client environment, not by replacing the inference runtime's dependencies:

```bash
uv venv decision-client
uv pip install --python decision-client/bin/python 'openai==3.26.0' 'typesafe-sdk==0.7.2'
source decision-client/bin/activate
```

The OpenAI SDK base URL includes `/v1`. Set `DYNAMO_DECISION_API_KEY` to the gateway credential when authentication is required. For this unauthenticated local example, the client falls back to a placeholder because the SDK requires a key; this does not configure server authentication.

```python
import os

from openai import OpenAI

with OpenAI(base_url="http://localhost:8000/v1", api_key=os.environ.get("DYNAMO_DECISION_API_KEY", "local-test-key"), max_retries=0, timeout=60) as client:
    result = client.decisions.create(
        model="Qwen/Qwen3-0.6B",
        input="A customer was charged twice and requests a refund today.",
        questions=[
            {
                "type": "choice", "name": "route", "instructions": "Choose the responsible team.",
                "choices": [{"value": "billing", "description": "Payments"}, {"value": "technical", "description": "Software"}],
            },
            {"type": "predicate", "name": "urgent", "instructions": "Action is needed today."},
            {
                "type": "score", "name": "severity", "instructions": "Rate the impact.",
                "levels": [{"label": "Low"}, {"label": "Moderate"}, {"label": "High"}],
            },
        ],
    )
    for answer in result.answers:
        print(answer.type, answer.model_dump())
    print(result.usage, result._request_id)
```

Answers are a union of types. Check `answer.type` before accessing type-specific fields. These examples return scored answers; do not interpret a refusal answer as a classification.

## Use the Official Jev/TypeSafe SDK

Use the TypeSafe SDK (`typesafe-sdk`) to call Jev's System One API, in the same client environment. Set the System One model explicitly to the registered alias. The TypeSafe SDK base URL is the server origin **without `/v1`**; the client supplies its own `/v1/systemone` path.

```python
import os

from typesafe_sdk import Choice, Noul, RetryPolicy, Score, TypeSafeClient

with TypeSafeClient(
    base_url="http://localhost:8000",
    model="Qwen/Qwen3-0.6B",
    api_key=os.environ.get("DYNAMO_DECISION_API_KEY", "local-test-key"),
    retry=RetryPolicy(max_retries=0),
    timeout=60,
) as client:
    result = client.system_one(
        state="A customer was charged twice and requests a refund today.",
        questions={
            "route": Choice(instructions="Choose the responsible team.", criteria={"billing": "Payments", "technical": "Software"}),
            "urgent": Noul(instructions="Action is needed today."),
            "severity": Score(instructions="Rate the impact.", criteria=["Low", "Moderate", "High"]),
        },
    )
    print(result.choices["route"].choice)
    print(result.nouls["urgent"].noul)
    print(result.scores["severity"].score)
    print(result.request_id)
```

## Interpret Usage and Failures

Use the [usage-accounting reference](../../reference/api/decision-apis.md#usage) to interpret token counts, including the unmeasured cache-write count. For a failed request, preserve its request ID and consult the [errors and headers reference](../../reference/api/decision-apis.md#errors-and-headers). This guide disables SDK retries so rejections remain visible; choose any production retry policy explicitly.

For admission limits and cache isolation, see [Decision API Configuration](../../reference/components/frontend-configuration.mdx#decision-apis). For compatibility tests against an existing deployment, see the [official-client qualification suite](https://github.com/ai-dynamo/dynamo/blob/main/tests/serve/test_decisions_external.py). API/SDK conformance, native numerical parity, model-quality evaluation, and distributed lifecycle correctness are separate qualification gates.
