<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Pinned vLLM OpenAPI 3.2 assessment

Use the [response-capable profile](../../../scripts/protocol_compatibility/frameworks/vllm-openapi32.json)
for the ongoing request **and response** assessment. The
[vLLM 0.30 recipe](dynamo-vllm-protocol-assessment.md) and its five-field generated
fixture are historical request-comparison evidence, not evidence for this build.

The compatibility direction is a client written for the pinned vLLM server using
Dynamo backed by that same vLLM revision. Additive and reverse-direction differences
are recorded separately. This assessment is incomplete: neither successful smoke
tests nor schema agreement establishes full behavioral compatibility.

## Reproduce the source and configuration

The profile owns exact source, kernel, dependency, model/tokenizer and launch pins.
Its [constraints](../../../scripts/protocol_compatibility/frameworks/vllm-openapi32-constraints.txt)
record the observed Linux Python environment, not a hash-locked installer. Build
vLLM from its exact source using the specified unchanged parent kernels, and build
the Dynamo runtime from the exact Cargo lockfile. Build/install the Dynamo and
vLLM source distributions separately; do not replace their source identities with
the shallow editable version string. Retain build commands, wheel hashes and the
final dependency freeze, then run the package consistency check.

Dynamo needs the separately pinned startup prerequisite
[#16073](https://github.com/ai-dynamo/dynamo/pull/16073). Apply that exact revision
to the profile's Dynamo parent in an isolated validation checkout and verify the
recorded full-index patch checksum. This prerequisite is not part of the three
schema PRs. Do not modify vLLM or silently substitute a newer prerequisite head.

Launch both servers with the profile's common engine settings and cached model
revision. Direct vLLM uses `--reasoning-parser qwen3 --enable-auto-tool-choice
--tool-call-parser hermes`; Dynamo's worker uses `--dyn-reasoning-parser qwen3
--dyn-tool-call-parser hermes`. Use unique ports and file-discovery storage, and
stop only owned services. These single-GPU runs are functional probes, not
benchmarks or qualification of every engine/model capability.

Capture `/openapi.json` from each exact running build using the
[generic acquisition CLI](../../../scripts/protocol_compatibility/README.md).
Retain the raw captures, version/source observations and deployment arguments.
Run `assess` with the manifest-pinned OpenAI composition input and oasdiff 1.33.0.
Do not interpret native Dynamo's unresolved dependency slots as absent contracts.

## Current evidence and limits

- Clean Dynamo parent cannot import the pinned vLLM backend; with the prerequisite,
  all 19 embedding-worker unit tests pass. This does not test real embedding serving.
- Both real servers pass chat/completions unary and SSE inference smoke cases.
  Their live exports report OpenAPI 3.2 with streaming `itemSchema`.
- The full comparison reports two endpoint-level request groups and two
  endpoint-level response groups. The retained location inventory subdivides them
  into 147 request groups and 15 response status groups, plus 15 coverage gaps.
  These are **not** counts of defects: nested deltas and union representations
  still need interpretation and root-cause deduplication.
- A frozen first request batch ran 22 cases on each real server. Omitted/null model
  selects the served model and infers in vLLM, but Dynamo returns 400 during
  deserialization; tracked in [#16192](https://github.com/ai-dynamo/dynamo/issues/16192).
  Chat `structured_outputs` produces the requested JSON in vLLM but is rejected
  by Dynamo. Neither finding has a passing Dynamo fix yet.
- The same batch deliberately retains inconclusive probes. Both servers reject
  `n=2` with greedy sampling. A lowercase stop string did not match the generated
  uppercase text, so acceptance does not prove stop semantics. Completion
  `structured_outputs` returned 200 from vLLM without satisfying the JSON constraint;
  that is not proof of a Dynamo-only semantic failure.

Other endpoints and model-specific capabilities remain outside this text-model
execution evidence. Missing response-body contracts, unsupported schema features,
stream ordering/termination/cancellation, and unresolved candidate groups are
coverage gaps, never compatibility passes. Investigate request candidates before
responses. For each confirmed root cause, retain a parent-failing and fix-passing
real-server regression with the unchanged reference; record only demonstrated
false positives as such. The complete final matrix and candidate ledger are still
required before claiming the assessment complete.
