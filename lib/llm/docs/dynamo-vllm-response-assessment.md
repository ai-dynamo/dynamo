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

The selected source registers unary JSON and SSE contracts for
`/v1/chat/completions` and `/v1/completions`. Its OpenAPI 3.1 SSE `schema` describes
a parsed event envelope: string `data` contains a JSON chunk, JSON error or literal
`[DONE]`; optional `event`, `id` and `retry` fields remain part of the declaration.
Select `event-schema` explicitly. Do not treat this as the OpenAPI 3.2
`itemSchema` representation or as a decoded-chunk-only contract.

## Qualification boundary

The upstream nightly wheel produced a fresh HTTP 200 capture through the production
app's in-process `/openapi.json` route, without entering lifespan. The document has
400 component schemas, 23 paths and 642 locally resolved reference occurrences.
Both unary responses and SSE event envelopes were checked. CUDA remained
uninitialized; no engine or model was started. The profile records the raw-byte
checksum and immutable wheel URL/hash. Loaded Python source was checked against
the pinned checkout, excluding generated `_version.py` (438 matching modules).

Upstream runtime tests, paired inference and a new full assessment remain **not
run**. Importing native libraries does not establish kernel execution coverage.

The recipe's synthetic integration tests exercise these selectors through the
generic adapter and comparator. They do not validate the current vLLM app or prove
that a particular installation produces the expected schema.

## Prepare inputs

1. On Linux x86_64 with Python 3.12, install the exact wheel from the profile using
   the commands below. The [dependency snapshot](../../../scripts/protocol_compatibility/frameworks/vllm-response-constraints.txt)
   records the capture environment; it is not a serving compatibility guarantee.
   Do not let an installer silently substitute another commit's wheel. Retain
   source identity, install commands, wheel hashes and the resulting package freeze.
2. Obtain its complete OpenAPI document. A production-app ASGI capture without
   lifespan can collect declarations without loading a model; label it as such,
   not as a running inference-server capture. An HTTP capture from a deployment
   must retain that deployment's configuration and scope. Record source identity
   and capture method separately; a byte hash does not attest their truth.
3. Save the exact bytes as `vllm-openapi.json`. Check both unary schemas, both SSE
   pointers from the profile, and local reference closure before comparison.
   Unexpected dialects or shapes require review, not a rewrite to fit this recipe.
4. Generate the Dynamo full-surface input from the selected Dynamo checkout using
   the [generic instructions](../../../scripts/protocol_compatibility/README.md).
   Retain its exact source independently. Schema-only comparison does not establish
   that Dynamo can serve with this vLLM revision.

```sh
PROFILE=scripts/protocol_compatibility/frameworks/vllm-response-openapi.json
CONSTRAINTS=scripts/protocol_compatibility/frameworks/vllm-response-constraints.txt
uv venv --python 3.12 .venv-vllm-schema
WHEEL_URL=$(jq -r '.wheel.url' "$PROFILE")
WHEEL_SHA256=$(jq -r '.wheel.sha256' "$PROFILE")
uv pip install --python .venv-vllm-schema/bin/python --torch-backend=cu130 \
  -c "$CONSTRAINTS" "${WHEEL_URL}#sha256=${WHEEL_SHA256}"
uv pip check --python .venv-vllm-schema/bin/python
```

For schema-only capture, use the pinned upstream
[production-app schema test](https://github.com/vllm-project/vllm/blob/338dab077c187e1bd421c4624af1169f95f60e06/tests/entrypoints/openai/test_openai_schema.py#L32-L63)
as the app configuration: `build_app(args, supported_tasks=("generate",))`, then
`TestClient(app).get("/openapi.json")`. Do not enter the client's context manager;
that starts lifespan. Close the client afterward. Keep `CUDA_VISIBLE_DEVICES` and
`VLLM_PLUGINS` empty and set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
Import from the installed wheel, not an uninstalled source checkout. Dependencies
and enabled routes affect the document, so verify each new capture independently.

## Select the source-qualified SSE interpretation

On the Linux validation host, create an adapter from the actual raw input bytes
and the profile's selectors. JSON is accepted by the YAML adapter loader:

```sh
PROFILE=scripts/protocol_compatibility/frameworks/vllm-response-openapi.json
RAW_SHA256=$(sha256sum vllm-openapi.json | cut -d ' ' -f 1)
jq -n --slurpfile profile "$PROFILE" --arg sha256 "$RAW_SHA256" '
  {version: 1, sources: {framework: {
    sha256: $sha256,
    source: "vllm-project/vllm",
    revision: $profile[0].source.revision,
    streams: $profile[0].response_contract.streams
  }}}' > vllm-response-adapters.json
python -m scripts.protocol_compatibility assess \
  --dynamo docs/frontends/openapi.json --framework vllm-openapi.json \
  --framework-name vllm --response-adapters vllm-response-adapters.json \
  --openai openai-d4535923.yaml --oasdiff /absolute/path/to/oasdiff \
  --output-dir assessment-vllm-current
```

Use a new output directory. The tool verifies the input hash and selector syntax,
not the supplied source/revision labels. Supply independently reviewed provenance
through `--framework-metadata` and `--dynamo-metadata` when available. Preserve
native errors, terminators and optional event fields; do not replace missing
declarations with an assumed universal error model.

Read `assessment-vllm-current/assessment/report.md` together with its retained
inputs, source-selection audit and coverage gaps. Exit 1 means differences or
gaps need review; it does not mean every difference is a defect. No exit code
establishes stream ordering, termination, cancellation or inference compatibility.
