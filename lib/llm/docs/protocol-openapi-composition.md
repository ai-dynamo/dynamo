<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OpenAPI composition implementation notes

Native schema export is independent of this offline composition tooling.
The composer resolves explicit dependency imports in exported request and response
schemas and can run without invoking the framework comparator.
The native `/openapi.json` endpoint remains uncomposed; use the separate composer
for a self-contained document.

## Dependency boundary

Dynamo uses released dynamo-protocols 8.1.0 with its optional schema feature,
async-openai 0.42.1, and aligned renderer 8.0.0 / parsers 10.0.0 / tokenizers 3.1.0.
The schema-only fork is no longer needed. The schema exports originate in
[frontend-crates #338](https://github.com/ai-dynamo/frontend-crates/pull/338).

The dependency exports owned schemas and explicit `x-dynamo-schema-import`
slots for unannotated async-openai types. A slot is not a complete contract.
The composition stage must resolve every in-scope slot or fail visibly.

Use the OpenAI YAML from async-openai's published 0.42.1 source revision
`d453592328fdbb3f59c1ea74f8e6cdf7f9140efb`:

- File: `openapi.yaml`.
- SHA-256: `6bfa47154802813b1241490dd89d909e0295f20b16efb2b9e620c75fea894f83`.
- This establishes provenance, not automatic equivalence with Rust types.

A dependency upgrade must revalidate both runtime consumers and these schema
corrections; successful composition alone does not prove runtime compatibility.

## Composition invariants

1. Preserve raw inputs; produce a separate composed Dynamo document.
2. Import only explicitly selected definitions and their reference closure.
3. Preserve Dynamo endpoints and owned fields. Never fill gaps from a framework.
4. Namespace imported components; reject collisions and unmapped slots.
5. Express corrections with standard JSON Patch. Guard each change with a
   preceding `test` of the affected value (or parent for additions), and record
   the Rust-code rationale. A failed guard requires review, not silent repair.
6. Redirect references to corrected definitions so nested uses cannot bypass
   corrections. Preserve recursion as references, not infinite expansion.
7. Do not fetch external references. Reject unsupported resolution constructs.
8. Walk schema positions only; literal example/default payloads must remain data.

An explicit `serde_json::Value` field is intentionally untyped, not necessarily
an extraction defect. An unresolved dependency marker is a gap. The contents of
`nvext` may be excluded because they are Dynamo-specific; record the exclusion,
never label it compatible, and do not generalize it without evidence.

## Fidelity Checks

Rust and composed-schema tests share cases under
`lib/llm/tests/fixtures/openapi/`. Keep corrections tied to the
Rust-code rationale and add a shared case when closing a schema/Serde gap.
A passing JSON Patch guard checks document assumptions; it does not attest the
source version or runtime behavior.

See the [tooling test commands](../../../scripts/protocol_compatibility/README.md#validation).
