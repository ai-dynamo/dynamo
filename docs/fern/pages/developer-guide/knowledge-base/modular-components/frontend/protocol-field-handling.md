---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Protocol Field Handling
subtitle: How contributors assign ownership and preserve request and response semantics across Dynamo
---

Use this design contract when adding or changing a request or response field in the Dynamo
frontend. It separates the field's public location, semantic ownership, and internal transport so
that implementation details do not become accidental compatibility claims.

This page defines contributor and review practice. It is not a field-by-field support matrix. For
the currently documented public surface, see the
[frontend configuration reference](../../../../reference/components/frontend-configuration.mdx)
and [NVIDIA request extensions](../../../additional-resources/nvidia-request-extensions-nvext.md).

## Scope

The examples focus on OpenAI-compatible chat completion and completion requests that cross the
Rust frontend, Python processors, and backend adapters. Apply the same ownership questions to
other endpoints when they cross those layers.

Support is specific to an endpoint, backend, version, transport, and deployment mode. Acceptance
on a vLLM path does not establish support on SGLang or TensorRT-LLM, and support on one endpoint
does not establish support on another. Verify each claimed path in code and with behavioral
evidence.

## Classify Fields on Three Dimensions

Record three independent decisions for every field:

| Dimension | Question | Examples |
| --- | --- | --- |
| Public ownership and location | Which public contract owns the field, and where does it appear on the wire? | OpenAI field at the request root, backend-compatible field at the root, Dynamo field under `nvext`, or unsupported |
| Semantic ownership | Which components must behave differently because the field is present? | Dynamo-handled, backend-handled, or jointly handled |
| Internal representation and transport | How does the value cross each internal boundary? | Canonical typed field, backend-specific typed field, namespaced opaque value, translated backend parameter, or frontend-only state |

Do not collapse these dimensions. A top-level field is not necessarily handled by Dynamo. A typed
field can still have backend-owned semantics. An opaque internal value can retain the public
spelling defined by a backend-compatible API.

## Assign Semantic Ownership

| Category | Responsibility |
| --- | --- |
| Dynamo-handled | Dynamo changes validation, preprocessing, routing, aggregation, streaming, or response serialization. The original field does not need to reach the engine when Dynamo completes its semantics. |
| Backend-handled | Dynamo preserves and delivers the value in the required representation. The backend implements the functional behavior. |
| Jointly handled | Dynamo and the backend implement different parts of the end-to-end behavior. Specify each responsibility and ensure that neither component repeats the other's transformation. |

The jointly handled category prevents a false two-way split between fields that Dynamo
"interprets" and fields that it "forwards." A field can require backend computation and Dynamo
response handling.

These examples illustrate the ownership model; they do not declare support on every backend or
endpoint:

| Example | Category | Ownership lesson |
| --- | --- | --- |
| `continue_final_message` | Dynamo-handled | Dynamo uses the value while preparing the chat template. The original field does not need to reach the engine after preprocessing. |
| `temperature` | Backend-handled | Backend sampling behavior can use a typed frontend representation. Typed storage does not make the sampling semantics frontend-owned. |
| `prompt_logprobs` | Jointly handled | The backend computes the values while Dynamo carries, aggregates, and exposes the response payload. Document the streaming and non-streaming projections separately. |
| `bad_words_token_ids` | Backend-handled | On supported vLLM paths, Dynamo validates and transports the token sequences while vLLM applies the sampling constraint. |
| `nvext.extra_fields` | Dynamo-handled | This Dynamo-owned option selects response metadata. It is not a catch-all request map. |

## Trace the Complete Field Lifecycle

Follow the stages in request-processing order. Each stage establishes the input or decision needed
by the next stage:

1. **Parse and capture:** Deserialize the value from its public wire location without losing
   distinctions such as omitted, `null`, `false`, `0`, and an empty collection. Parsing comes first
   because no later policy or behavior can inspect a value that was discarded or normalized
   incorrectly at the boundary. Capture alone does not mean that Dynamo supports the field.
2. **Admit:** Decide whether the selected endpoint, backend profile, version, transport, and
   deployment mode accept the captured field. Admission follows parsing because the frontend must
   know which field the client supplied, but it precedes semantic validation so unsupported fields
   receive the configured reject-or-ignore outcome instead of accidentally reaching an
   implementation path.
3. **Validate:** Check the admitted value's type, range, shape, conflicts, and supported
   combinations. Validation depends on the selected compatibility path and must finish before
   preprocessing or generation creates side effects or consumes expensive resources.
4. **Interpret:** Apply every Dynamo-owned semantic effect, such as changing chat-template
   rendering, routing, aggregation, or response selection. Interpretation follows validation so
   components operate on a valid value, and it precedes transport because it determines whether the
   backend needs the original value, a translated value, or no value at all.
5. **Transport:** Encode the backend-relevant representation across each applicable Rust, Python,
   serialization, and process boundary. Transport follows interpretation because the semantic owner
   determines the required representation, and it precedes execution because a backend cannot act
   on a value that disappears at an intermediate conversion.
6. **Execute:** Identify the backend operation that consumes the transported value and verify its
   behavioral effect. Execution follows successful transport. Reaching a backend data structure is
   not enough; the engine must use the value as the public contract specifies.
7. **Project the response:** Map backend results and failures into the public response contract.
   Check streaming and non-streaming behavior separately when they differ. Projection comes last
   because it depends on the execution result, but request-only fields still require intentional
   error behavior even when they add no response field.

The order narrows uncertainty at each boundary: preserve what the client sent, select the supported
contract, reject invalid input, apply frontend semantics, deliver backend semantics, execute them,
and expose the outcome.

### Example: `bad_words_token_ids` on a vLLM Path

Consider this chat completion request fragment:

```json
{
  "bad_words_token_ids": [[12, 13]]
}
```

The field traces through the lifecycle as follows:

1. **Parse and capture:** The chat request parser captures the top-level field in its extra-field
   map. An omitted field remains different from an empty list. A `null` value for this named
   passthrough field is normalized to omission.
2. **Admit:** The frontend recognizes `bad_words_token_ids` as one of a small set of fields accepted
   for backend-specific handling. An arbitrary extra field remains unsupported; this allowlist does
   not create general passthrough.
3. **Validate:** Dynamo requires an array of token-ID arrays. A scalar, string token ID, negative
   token ID, or incorrectly nested array fails before generation.
4. **Interpret:** Dynamo does not implement the token-blocking semantics. Its responsibility is to
   normalize and validate the input, so the field remains backend-handled.
5. **Transport:** The preprocessor copies the value to
   `extra_args.sampling_options.bad_words_token_ids`, preserving the nested token sequences across
   the Rust-to-Python boundary.
6. **Execute:** The vLLM handler installs the sequences in vLLM sampling parameters. It checks that
   the expected vLLM field exists so a backend upgrade fails visibly instead of silently dropping
   the constraint. vLLM then applies the constraint during token sampling.
7. **Project the response:** The option adds no dedicated response field. The generated text is the
   client-visible result, while validation or backend compatibility failures must surface as errors.
   Transport-level tests can prove that the sampling parameters contain the value; end-to-end tests
   are still needed to prove that generated tokens obey the constraint.

Dynamo's token pipeline does not generally proxy the original client JSON to a native backend
server. A field that parses successfully can still disappear at a conversion boundary, and a
backend feature can still be unavailable through Dynamo.

## Distinguish Admission from Support

A field is recognized only when Dynamo makes an explicit compatibility commitment. That
commitment identifies:

- its public location and value semantics;
- the supported endpoints, backends, versions, transports, and deployment modes;
- every semantic owner and the representation delivered to that owner;
- its streaming and non-streaming response behavior; and
- the error returned on unsupported paths.

Parsing, schema presence, typed storage, or catch-all capture alone does not establish that
commitment.

Chat completion and completion requests currently capture extra top-level fields in an internal
map named `unsupported_fields`. The name describes the capture mechanism, not the final admission
decision. A small named set is accepted and validated for downstream handling. Other fields are
rejected by default. When `DYN_IGNORE_OPENAI_FE_UNSUPPORTED_FIELDS` is truthy, those other fields
are ignored and dropped; the switch does not enable general forwarding. See the
[frontend configuration reference](../../../../reference/components/frontend-configuration.mdx)
for the exact switch behavior.

A top-level catch-all does not preserve unknown members nested inside typed objects. Trace nested
objects independently.

## Preserve Value Semantics

Preserve meaningful distinctions among omission, `false`, `0`, `null`, empty strings, and empty
collections. Do not use truthiness checks when the field contract assigns different meanings to
those values.

A field can define normalization explicitly. For example, the current extra-field decoder treats
`null` for named passthrough fields as omission. Document and test such normalization instead of
claiming that every JSON value survives unchanged.

Ignoring an unknown field is different from supporting a recognized field. Ignore mode can drop an
unknown value by design. A recognized field must not silently lose semantics on a path that claims
to support it.

## Separate Contract and Behavioral Evidence

Use two acceptance layers:

1. **Declared contract:** Compare public names, locations, types, requiredness, nullability,
   unions, and declared omission behavior.
2. **Behavioral conformance:** Exercise client-visible validation, transformations, backend
   effects, returned values, error mapping, streaming order, and termination.

Schema generation and permissive parsing belong to the declared-contract layer. They do not prove
behavioral compatibility. Source inspection can explain a result or identify a likely gap, but it
does not replace either acceptance layer.

## Development Checklist

Before merging a protocol-field change:

1. Name the public owner and wire location.
2. Assign Dynamo, backend, or joint semantic ownership for every supported path.
3. Trace the field through each applicable lifecycle stage.
4. State endpoint, backend, version, transport, and deployment-mode support.
5. Define omission, `false`, `0`, `null`, and empty-value behavior where applicable.
6. Specify streaming and non-streaming response behavior separately.
7. Update the public schema and reference documentation for the promised surface.
8. Add the smallest end-to-end regression that demonstrates the promised behavior.
9. Record unsupported paths and staged gaps explicitly instead of relying on silent degradation.

Keep temporary exceptions narrow and assign them a tracked follow-up. Do not present a planned
behavior as an existing guarantee.
