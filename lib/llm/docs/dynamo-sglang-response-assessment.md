<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo versus SGLang schema assessment

Compare the declared chat/completion requests, unary responses and parsed SSE
responses using the [maintained comparator](../../../scripts/protocol_compatibility/README.md).
Differences are investigation candidates, not runtime defects or a parity score.

The [profile](../../../scripts/protocol_compatibility/frameworks/sglang-response-openapi.json)
pins SGLang PR [#44146](https://github.com/sgl-project/sglang/pull/44146) at
`43fbd836ab207917fd8cd147ec14f13c6dc17da3`. It was open when frozen. This
assessment describes that patch, not a released SGLang contract. SGLang remains
OpenAPI 3.1: `schema` describes one parsed event; Dynamo's OpenAPI 3.2 uses
`itemSchema`. Both need explicit JSON payload decoding. SGLang's streaming error
is an inline `error` wrapper around `ErrorResponse`, not the vLLM error reference.

## Acquire the patched SGLang input

Use a clean checkout of the exact profile revision and a separate Python 3.12
SGLang environment. Follow that revision's dependency installation instructions.
Retain the exact dependency freeze with each capture: changes to FastAPI,
Pydantic, OpenAI or other dependencies can change schema bytes. The helper records
all installed distribution versions and hashes each imported SGLang Python file.
A version list is not a dependency lock or native-library attestation.

```sh
git clone https://github.com/sgl-project/sglang.git sglang-schema
git -C sglang-schema fetch origin pull/44146/head
git -C sglang-schema checkout 43fbd836ab207917fd8cd147ec14f13c6dc17da3
PROFILE=scripts/protocol_compatibility/frameworks/sglang-response-openapi.json
PYTHONHASHSEED=0 /path/to/sglang-environment/bin/python \
  scripts/protocol_compatibility/frameworks/capture_sglang_openapi.py \
  --source sglang-schema --profile "$PROFILE" --output-dir inputs/sglang
/path/to/sglang-environment/bin/python -m pip freeze > inputs/sglang/dependencies.txt
```

The helper requests `/openapi.json` through the production ASGI app without
lifespan. It disables GPU visibility and model downloads, rejects CUDA
initialization and source drift, and preserves unmodified HTTP bytes. This is
not an inference-server capture. Outputs are `sglang-openapi.json`,
`response-adapters.json`, and `acquisition.json`. Use a new output directory;
partial output from a failed command is not qualified input.

The profile owns the two `event-schema` selectors. The helper checks native
unary/chunk/error references and local reference closure, then binds selectors to
the actual raw SHA256. The comparator checks that hash, not the truth of the
source labels. It preserves native errors, optional envelope fields and `[DONE]`.

## Freeze and compare

Follow the generic instructions for the pinned oasdiff executable, Python
requirements, Dynamo export and OpenAI composition input. Keep capture and
comparison environments separate. Record their exact source and dependency pins.

```sh
python -m scripts.protocol_compatibility assess \
  --dynamo inputs/dynamo/openapi.json \
  --framework inputs/sglang/sglang-openapi.json --framework-name sglang \
  --response-adapters inputs/sglang/response-adapters.json \
  --openai inputs/openai.yaml --oasdiff /absolute/path/to/oasdiff \
  --output-dir assessment-sglang
```

Read `assessment-sglang/report.md` and its canonical `report.json`
`finding_inventory`. Retain the whole output, source/configuration records,
raw inputs and checksums. Exit 1 means differences or coverage gaps need review;
exit 2 means invalid input or tool failure. To check reproducibility, repeat in
a new directory and compare raw inputs, projections, deltas and inventory;
acquisition timestamps and absolute command paths can differ.

## Scope and interpretation

Only POST `/v1/chat/completions` and `/v1/completions` are compared. The same
request schemas cover unary and streaming admission. JSON success/error bodies
and supported parsed SSE declarations are compared separately. The retained
Dynamo full-surface export enables both dispatch extensions; it is not the
ordinary default-off deployment configuration. The pinned OpenAI input supplies
reviewed dependency definitions, not the latest OpenAI service contract.

Missing declarations, opaque values, custom validation and unsupported constructs
remain coverage limits. Response unions and `allOf` changes must be inspected
before interpreting nested fields as missing. Required-model policy and deferred
compatibility decisions remain unchanged. `nvext`, other endpoints, model and
parser effects, framing, ordering, termination, cancellation and client-library
behavior are outside this assessment. No inference or hosted CI is required to
capture and compare these declarations.

## Frozen assessment result (2026-10-16)

The complete comparison ran twice with oasdiff 1.33.0 and comparator source
`e2282fda3e2ca23d4d0af1bb6a62b060e4a41d35`. Dynamo input came from the retained
static export at `2a4fa931622d1b48099ccf0f81344db7a5cae8d2`, with
async-openai 0.42.1 and dynamo-protocols 8.1.0. It was not freshly rebuilt here.
The composition manifest SHA256 is
`4091e027d9f5b058bdb138e2c9c025ab8ab48eae4ff969f418b67fe9472d48c6`.

| Frozen artifact | SHA256 |
| --- | --- |
| Dynamo raw input | `b8b4f634c0b1b7f71d7742760c36394756e07d9b0c8df0b2ca6f2d30c9d0dc60` |
| SGLang raw input | `7d23e49ca2bcb1a659b0fc42ec0d7d63d4b189ff24efe98e6ab6b74c3d579df6` |
| First complete report.json | `7ca8a20bbd3d67408b88d879250c1c803e7ed0b0739f6b2565b0a6c991507f74` |

The 44 inventory rows account for all 372 reported delta terminals: 8 target-only,
6 Dynamo-only, 1 both-declared mismatch, 1 alias, 6 representation ambiguities,
21 coverage gaps and 1 intentional exclusion. These are report units, not 44
runtime defects or complete field coverage. All eight target-only rows concern
responses; request coverage is limited as described below.

Four request gaps arise from SGLang `DispatchFormat` and `TokenDispatchFormat`
`prefixItems`, once per endpoint. The comparator disables flattening for the
whole request projection when such a gap exists. Both request roots therefore
remain coarse `allOf` representation-ambiguity findings. **This does not show
that migrating requests have no incompatibilities.** Investigate their retained
source declarations before selecting finer request candidates; do not remove
`prefixItems` to obtain a more detailed-looking report. The remaining gaps are
seven Dynamo custom request/alias/validation entries, eight Dynamo non-200 body
declarations and two unspecified SGLang default response bodies.

Two independent SGLang captures and ten projection/normalization files matched
byte-for-byte. Repeated deltas and all 44 finding identities matched after a
bounded audit normalized oasdiff's unordered added/deleted property, required,
component and status name lists, plus invocation output paths. Each original
report pointer was checked against its own report. Raw reports were preserved;
schema arrays were not broadly reordered. Failed byte-equality and incomplete
normalization attempts remain in the validation record.

All 98 protocol tests passed with real oasdiff, including six capture/projection
tests. Scoped hooks passed after formatting. Capture used the existing recorded
schema environment: Python 3.12.3, FastAPI 0.136.3, Pydantic 2.13.5, OpenAI 2.6.1
and Torch 2.14.0 (the SGLang source requests 2.14.1). This is not a clean full
installation qualification. Native-library compatibility and serving behavior
were not tested. Duplicate operation-ID warnings on other SGLang routes are
retained acquisition warnings; those routes are outside comparison scope.
