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

| Example | Ownership lesson |
| --- | --- |
| `continue_final_message` | Dynamo uses the value while preparing the chat template. The original field does not need to reach the engine after preprocessing. |
| `temperature` | Backend sampling behavior can use a typed frontend representation. Typed storage does not make the sampling semantics frontend-owned. |
| `prompt_logprobs` | The backend computes the values while Dynamo carries, aggregates, and exposes the response payload. Document the streaming and non-streaming projections separately. |
| Backend processor option | Backend-owned semantics can use opaque transport, but only on explicitly supported paths. Opaque transport is not unrestricted passthrough. |
| `nvext.extra_fields` | This Dynamo-owned option selects response metadata. It is not a catch-all request map. |

## Trace the Complete Field Lifecycle

Follow the field through every applicable stage:

1. **Parse and capture:** Identify the wire location and distinguish omission from an explicit
   value.
2. **Admit:** Decide which endpoint, backend profile, version, and mode accept the field.
3. **Validate:** Check types, ranges, conflicts, and unsupported combinations before generation
   starts.
4. **Interpret:** Name each Dynamo component that changes behavior because of the value.
5. **Transport:** Preserve or translate the value across Rust, Python, and backend boundaries.
6. **Execute:** Identify the backend behavior that consumes the value.
7. **Project the response:** Preserve errors and client-visible results, including distinct
   streaming and non-streaming contracts.

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
