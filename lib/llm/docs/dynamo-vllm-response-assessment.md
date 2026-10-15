<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pinned upstream vLLM response assessment

Use [vllm-response-openapi.json](../../../scripts/protocol_compatibility/frameworks/vllm-response-openapi.json)
for new request/response assessments. It pins `7cd7982dfda0d29c68f73c54cd343b674b290a47`,
the latest vLLM `main` observed on 2026-10-15, rather than a moving branch or release
tag. This revision includes the merged response-schema change
[#60926](https://github.com/vllm-project/vllm/pull/60926).

The selected source registers unary JSON and SSE contracts for
`/v1/chat/completions` and `/v1/completions`. Its OpenAPI 3.1 SSE `schema` describes
a parsed event envelope: string `data` contains a JSON chunk, JSON error or literal
`[DONE]`; optional `event`, `id` and `retry` fields remain part of the declaration.
Select `event-schema` explicitly. Do not treat this as the historical OpenAPI 3.2
`itemSchema` representation or as a decoded-chunk-only contract.

## Qualification boundary

The pin and response registration were checked against upstream source. A fresh
production-app capture, upstream runtime tests, paired inference and a new full
assessment have **not** been run for this revision. Matching precompiled wheel
indexes were unavailable when this pin was selected. No kernel build, installed
environment, model configuration, capture checksum or finding count is carried
forward from an older revision.

The recipe's synthetic integration tests exercise these selectors through the
generic adapter and comparator. They do not validate the current vLLM app or prove
that a particular installation produces the expected schema.

## Prepare inputs

1. Build/install the exact profile revision, using its own dependency requirements
   and either matching native artifacts or a recorded source build. Do not let an
   installer silently substitute another commit's precompiled wheel. Retain source
   status, build commands, wheel hashes and the installed dependency freeze.
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

## Historical evidence

The [OpenAPI 3.2 experiment](dynamo-vllm-openapi32-assessment.md) remains pinned to
the earlier `b04e80127` development revision. Its environment, captures, counts and
paired-server observations are historical and do not qualify this newer source.
The [vLLM 0.30 recipe](dynamo-vllm-protocol-assessment.md) and five-field fixtures
also remain unchanged as request-only regression evidence.
