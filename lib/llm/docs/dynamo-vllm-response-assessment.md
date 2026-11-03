<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pinned upstream vLLM response assessment

Use [vllm-response-openapi.json](../../../scripts/protocol_compatibility/frameworks/vllm-response-openapi.json)
for new request/response assessments. It pins `338dab077c187e1bd421c4624af1169f95f60e06`,
the source of the latest cu130 nightly wheel observed on 2026-10-15, rather than
the moving `nightly` alias. This revision includes the merged response-schema change
[#60926](https://github.com/vllm-project/vllm/pull/60926).

Use this nightly pin temporarily. Once a stable vLLM release includes the
response-schema change, update the profile to that release's exact source revision
and wheel. Refresh the wheel checksum and schema capture;
verify the SSE selectors against that release before using it for assessments.

## How SSE schemas become comparable

For `/v1/chat/completions` and `/v1/completions`, vLLM declares parsed SSE events
under OpenAPI 3.1 `schema`; Dynamo uses OpenAPI 3.2 `itemSchema`. The profile's
explicit `event-schema` adapter tells the comparator how to interpret vLLM's
declaration. The comparator projects both into the same decoded-event
representation, preserving envelope fields, JSON chunks, errors and literal
`[DONE]`. It does not change either raw export or either server's behavior.

See [source-qualified SSE adapters](../../../scripts/protocol_compatibility/README.md#source-qualified-sse-adapters)
for selection rules and the [streaming comparison model](../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/frontend/openapi-request-contract-comparison.md#streaming-responses)
for projection details and coverage limits.

## Qualification boundary

The pinned wheel supports production-app schema export without starting an engine
or loading a model. The profile records source identity, wheel checksum and capture
details. This establishes schema-export availability, not inference compatibility
or stream lifecycle behavior. Upstream runtime tests, paired inference and a new
full assessment have **not been run** for this pin.

## Prepare inputs

1. On Linux x86_64 with Python 3.12, install the exact wheel from the profile using
   the commands below. Dependencies are not pinned, so their versions can change
   the exported schema. Retain the package freeze with each capture's evidence.
   Do not let an installer silently substitute another commit's wheel. Retain
   source identity, install commands, wheel hashes and the resulting package freeze.
2. Run the capture helper below from the Dynamo checkout. It produces only
   `vllm-openapi.json` and `response-adapters.json` in a new directory. A successful
   exit is required before using the pair; do not use partial outputs after an
   error. Unexpected installation identities or declarations require review.
3. Generate the Dynamo full-surface input from the selected Dynamo checkout using
   the [generic instructions](../../../scripts/protocol_compatibility/README.md).
   Retain its exact source independently. Schema-only comparison does not establish
   that Dynamo can serve with this vLLM revision.

```sh
PROFILE=scripts/protocol_compatibility/frameworks/vllm-response-openapi.json
uv venv --python 3.12 .venv-vllm-schema
WHEEL_URL=$(jq -r '.wheel.url' "$PROFILE")
WHEEL_SHA256=$(jq -r '.wheel.sha256' "$PROFILE")
uv pip install --python .venv-vllm-schema/bin/python --torch-backend=cu130 \
  "${WHEEL_URL}#sha256=${WHEEL_SHA256}"
uv pip check --python .venv-vllm-schema/bin/python
```

The [capture helper](../../../scripts/protocol_compatibility/frameworks/capture_vllm_openapi.py)
uses the pinned upstream [production-app schema test](https://github.com/vllm-project/vllm/blob/338dab077c187e1bd421c4624af1169f95f60e06/tests/entrypoints/openai/test_openai_schema.py#L53-L69)
configuration. It requests `/openapi.json` in-process without entering lifespan,
clears inherited `VLLM_*` settings before importing vLLM, disables GPU visibility
and plugins, and enables offline model access. This is a schema-only capture, not
a running inference-server capture.

```sh
.venv-vllm-schema/bin/python \
  scripts/protocol_compatibility/frameworks/capture_vllm_openapi.py \
  --profile "$PROFILE" --output-dir inputs/vllm
```

The helper checks the installed version and direct wheel URL, rejects a shadowing
source checkout, and checks imported vLLM Python files against the installation's
`RECORD`. These are consistency checks, not independent source attestation; the
install command above enforces the archive checksum. It also checks both unary
and SSE declarations and local reference closure, then preserves the HTTP bytes
unchanged. Dependencies and enabled routes can affect those bytes.

The reviewed profile owns SSE interpretation and selectors. The helper copies
them into `response-adapters.json` with the actual capture hash and profile source
revision. Agents need not infer SSE semantics or hand-write adapter JSON. The
generic comparator remains responsible for projection and coverage analysis.

## Run the comparison

Use the comparison environment from the generic instructions, not necessarily
the vLLM capture environment:

```sh
python -m scripts.protocol_compatibility assess \
  --dynamo docs/frontends/openapi.json --framework inputs/vllm/vllm-openapi.json \
  --framework-name vllm --response-adapters inputs/vllm/response-adapters.json \
  --openai openai-d4535923.yaml --oasdiff /absolute/path/to/oasdiff \
  --output-dir assessment-vllm-current
```

Use a new output directory. The tool verifies the input hash and selector syntax,
not the supplied source/revision labels. Preserve native errors, terminators and
optional event fields; do not replace missing
declarations with an assumed universal error model.

Read `assessment-vllm-current/assessment/report.md` together with its retained
inputs, source-selection audit and coverage gaps. Exit 1 means differences or
gaps need review; it does not mean every difference is a defect. No exit code
establishes stream ordering, termination, cancellation or inference compatibility.
