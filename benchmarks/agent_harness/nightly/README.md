<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nightly coding-agent workloads

This directory is the first stage of a nightly compatibility test for native
Claude Code and Codex sessions against a Dynamo endpoint. It uses Harbor for
SWE-bench container lifecycle and agent execution, then gates the run on
Dynamo's own request trace.

The initial test is intentionally a protocol soak rather than a model-quality
benchmark. A task may fail its SWE-bench verifier without failing this gate. The
gate requires every expected root agent session to make several model requests
and to exercise both a user-initiated turn and a tool-result continuation. This
catches long-session wire incompatibilities while keeping model quality drift
out of the compatibility signal. Harbor's job directory remains the source for
the native Claude/Codex logs, trajectories, patches, and verifier results.

## Contract

- Pin Harbor to `0.21.0`. Let Harbor install the current Claude Code
  and Codex CLIs so nightly detects upstream harness changes.
- Serve one model at a time on two H100s under a slash-free alias such as
  `agent-nightly`. Codex removes provider prefixes from model names, so the
  shared alias keeps Claude and Codex on the same advertised model.
- Keep a reviewed five-task SWE-bench Pro set in a task-ID file. Cache those
  exact task images in the Harbor Docker service before the measured run. The
  committed `task_ids.txt` is the initial cross-project set covering NodeBB,
  qutebrowser, Flipt, Tutanota, and Open Library; the runner uses Harbor's
  `--no-delete` mode so those images stay warm.
- Enable Dynamo request tracing with a JSONL sink and preserve both the trace
  and Harbor job directory as CI artifacts.
- Run the two harnesses sequentially against the warm model. Run task
  containers with low concurrency initially; increase it only after the signal
  is stable.

The job runs in the approved SGLang runtime job container on
`aws-dev-02-tester-amd-gpu-v2`. Dynamo owns the job container's two H100s,
while a CPU-only Docker-in-Docker service creates Harbor's sibling task
containers. The runner's Kubernetes job adapter gives those containers one
pod network, so Harbor reaches the Docker sidecar on `127.0.0.1:2375`. The
local frontend and worker share Dynamo's file discovery backend, so this
aggregated deployment does not require etcd or NATS sidecars. Manual canaries
set the nightly workflow's `agent_harness_only` input to skip the standard
vLLM, SGLang, and TensorRT-LLM H100 jobs.

## Model matrix

Keep model weights, serving parsers, and the SGLang runtime pinned so a nightly
failure is attributable to a harness or API compatibility change. Both models
fit the `aws-dev-02-tester-amd-gpu-v2` two-H100 runner with tensor parallelism
of two.

| Model | Hugging Face revision | Tool parser | Reasoning parser | Thinking |
| --- | --- | --- | --- | --- |
| `zai-org/GLM-4.7-Flash` | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | `glm47` | `glm45` | enabled |
| `Qwen/Qwen3-Coder-30B-A3B-Instruct` | `b2cff646eb4bb1d68355c01b18ae02e7cf42d120` | `qwen3_coder` | none | unsupported by this checkpoint |

Start with ordinary autoregressive decoding. Do not enable GLM's EAGLE
speculative decoding until the base compatibility signal is stable; otherwise
a speculative-decoding regression can look like a harness or parser failure.
Use the Dynamo SGLang runtime pinned by this repository rather than an
unversioned upstream image.

For GLM, launch the existing agent-serving example with the pinned model
revision and thinking enabled:

```bash
./examples/backends/sglang/launch/agg_agent.sh \
  --model-path zai-org/GLM-4.7-Flash \
  --served-model-name agent-nightly \
  --tp 2 \
  --revision 7dd20894a642a0aa287e9827cb1a1f7f91386b67 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --default-thinking-mode enabled
```

The example translates those parser options to Dynamo SGLang arguments.
Preserve the launch command and resolved model revision in the nightly
artifacts. GLM's model card recommends preserved
thinking for multi-turn agent workloads, but the native harness requests do
not currently set its `clear_thinking=false` chat-template option. Add that as
a separately validated follow-up instead of silently rewriting harness
requests in the base lane.

## Run

Start a Dynamo endpoint with both `/v1/responses` and `/v1/messages` enabled,
request tracing pointed at a JSONL file, and the real model served under the
same alias passed below. Install a pinned Harbor release on the host, prepare
the task images, then run:

```bash
export DYNAMO_BASE_URL=http://HOST_REACHABLE_FROM_DOCKER:8000
export DYNAMO_MODEL_ALIAS=agent-nightly
export DYNAMO_API_KEY=dummy
export DYN_REQUEST_TRACE_OUTPUT_PATH=/var/tmp/dynamo-agent-nightly.jsonl
export TASK_IDS_FILE=benchmarks/agent_harness/nightly/task_ids.txt
export RESULTS_DIR=/var/tmp/dynamo-agent-nightly-results

./benchmarks/agent_harness/nightly/run_harbor.sh
```

Optional inputs are `HARBOR_COMMAND`, `HARBOR_DATASET` (default
`swebenchpro@1.0`), `HARBOR_VERSION` (default `0.21.0`),
`HARBOR_CONCURRENCY` (default `1`), `EXPECTED_TASK_COUNT` (default `5`), and
`MINIMUM_REQUESTS_PER_SESSION` (default `4`). The runner waits up to 30 seconds
for the trace sink to flush; override that with
`TRACE_VALIDATION_TIMEOUT_SECONDS`. `RUN_NAME_SUFFIX` defaults to a UTC
timestamp and can be set to the GitHub run ID and attempt for stable artifact
correlation.

## Follow-up scenarios

Add these as separate, diagnosable lanes after the base matrix is stable:

1. Native nested subagents, requiring parent and child session IDs in the trace.
2. Context compaction, requiring a task whose trajectory proves the threshold
   was crossed and the session continued afterward.
3. Cancellation followed by a user steering message in the same session.
4. A small per-harness HTTP error matrix for retryable, terminal, overloaded,
   authentication, and malformed-response behavior.
5. OpenCode, after the Claude/Codex workload signal is reliable.
