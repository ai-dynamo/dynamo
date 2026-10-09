<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Response schemas in the frontend OpenAPI document

The chat-completions and legacy completions POST operations describe two HTTP
200 media types: `application/json` for unary responses and `text/event-stream`
for successful streamed JSON payloads. Chat uses separate response and chunk
types. Legacy completions use the same response type in both modes.

```yaml
/v1/chat/completions:
  post:
    responses:
      "200":
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/NvCreateChatCompletionResponse"
          text/event-stream:
            schema:
              $ref: "#/components/schemas/NvCreateChatCompletionStreamResponse"
```

The stream schema describes the JSON following `data:`, not the entire SSE
stream. It does not include the literal `[DONE]` marker, error/annotation events,
ordering constraints, or rules about when usage and finish reasons appear.
Those require separate protocol/behavioral checks. Other endpoint families and
error response bodies are outside this change.

## Complete dependency imports before comparing

Dynamo derives its owned response schemas from the runtime Rust types. The
version-pinned `dynamo-protocols` schema feature supplies chat response components.
Unannotated async-openai types remain explicit `x-dynamo-schema-import` slots in
the raw HTTP export. A placeholder is a coverage gap, not a complete contract.

Use the **shared request/response manifest and composition engine** to fill
these slots from the pinned async-openai OpenAPI document:

```bash
python -m scripts.protocol_compatibility.composition.responses \
  --raw /absolute/path/to/openapi.raw.json \
  --openai /absolute/path/to/pinned-openai.yaml \
  --output-dir /absolute/path/to/new-response-export
```

The output directory must not exist. The command writes `openapi.composed.json`
and the exact `composition.yaml` used. Preserve the raw capture and its existing
acquisition metadata separately. The composer checks the OpenAI document hash,
applies guarded JSON Patch corrections, resolves reference closure, and records
composition provenance. It does not compare providers and has no dependency
on comparison tooling.

See [the composition guide](protocol-openapi-composition.md) for dependency pins,
provenance requirements, and the standalone export workflow.
Do not update a source checksum merely to bypass a failed correction guard.

## Serialization details and limits

- The HTTP exporter marks always-emitted nullable chat fields required, including
  choice `finish_reason`/`logprobs` and message `content`/`refusal`.
- The served schema uses the configured reasoning field name (`reasoning_content`
  by default, or `reasoning`). The default offline generator describes default
  configuration; capture the actual server for non-default deployment settings.
- Internal `llm_metrics`, `tool_call_completion`, and streaming
  `prompt_logprobs` are absent from client-facing chunk schemas. Unary
  `prompt_logprobs` remains part of the response schema.
- Chat's usage serializer omits absent detail members. Legacy completion uses
  upstream serialization, including null detail members. Separate import paths
  keep these different serializers from sharing an incorrect usage definition.
- Legacy completion always emits nullable `usage` and `system_fingerprint`;
  absent choice `logprobs` and `finish_reason` are omitted. Guarded baseline
  corrections describe these differences rather than assuming OpenAI's document
  exactly matches the Rust crate.
- Derived optional types can remain nullable even where a serializer omits
  absent fields. Numeric representability, conditional values and complete
  coverage of every response variant are not certified by this export.
  Keep those fidelity limits explicit when interpreting comparison results.

Shared fixtures are checked by Rust serialization and JSON Schema validation:

```bash
cargo test --locked -p dynamo-llm --no-default-features --test openapi_response_schema
python -m scripts.protocol_compatibility.tests.composition.response_fidelity \
  --spec /absolute/path/to/new-response-export/openapi.composed.json
```

These tests establish representative schema/serializer agreement, not inference
correctness or end-to-end compatibility with another provider.
