<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qualify Agent-Shaped Load Against a Dynamo Recipe

This path sends a small deterministic causal workload to the OpenAI-compatible endpoint produced by a Dynamo Kubernetes recipe. It is cluster-agnostic: deploy any supported recipe first, then provide the recipe's root service URL and served model. The included profile proves request transport, Codex-shaped session headers, causal ordering, and artifact generation; it is not a capacity benchmark or a simulation of the Codex Responses API.

## Scope and Boundaries

`agent-loadgen` currently sends Chat Completions requests with synthetic token data. Its `codex` renderer adds Codex-shaped lineage headers to that protocol, but it does not reproduce native Codex Responses requests, reasoning items, tool execution, or task quality. Use a real harness path to evaluate those behaviors.

The included profile is deliberately small: two top-level sessions, two sequential turns per session, two concurrent sessions, four total requests, no tools, no subagents, no early completion, and no compaction. Its safety limits cap the graph at four nodes, two sessions, and 4,096 aggregate input tokens.

## Prerequisites

- A supported Dynamo Kubernetes recipe deployed through the [Kubernetes deployment recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes), with a reachable root OpenAI-compatible service URL and exact served model name.
- Rust 1.85 or later to build the pinned load generator.
- Python 3.10 or later to run the campaign wrapper.
- A local tokenizer path or Hugging Face tokenizer identifier that matches the served model.

## Build the Pinned Load Generator

The generic path is pinned to [`NVIDIA-dev/agent-loadgen` commit `9057201e23663baaaf076820f3772d55468dec25`](https://github.com/NVIDIA-dev/agent-loadgen/commit/9057201e23663baaaf076820f3772d55468dec25). The wrapper verifies that the source checkout is clean and exactly at this commit, then records both the source revision and binary SHA-256 digest.

```bash
git clone https://github.com/NVIDIA-dev/agent-loadgen.git
git -C agent-loadgen checkout --detach 9057201e23663baaaf076820f3772d55468dec25
cargo build --release --manifest-path agent-loadgen/Cargo.toml
```

## Export the Recipe Endpoint

Use the root service URL. Do not append `/v1` or `/v1/chat/completions`; `agent-loadgen` constructs the Chat Completions path itself.

```bash
export DYNAMO_BASE_URL="https://your-recipe-endpoint.example.com"
export DYNAMO_MODEL="the-exact-served-model-name"
export LOADGEN_TOKENIZER="the-matching-tokenizer-path-or-id"
```

If the endpoint requires authentication, pass each static header to the wrapper. Header values are sent to `agent-loadgen` but replaced with `<redacted>` in wrapper command metadata and captured stdout/stderr logs.

## Run the Bounded Smoke

Run from the Dynamo checkout:

```bash
python3 examples/agent_harnesses/benchmark/run_campaign.py \
  --loadgen ./agent-loadgen/target/release/agent-loadgen \
  --loadgen-source ./agent-loadgen \
  --output-root ./agent-loadgen-artifacts \
  --header "Authorization=Bearer your-token"
```

Omit `--header` for an unauthenticated endpoint. The wrapper validates the URL, model, tokenizer, profile, dependency pin, clean source state, bounded request count, and non-overwriting output path. It always runs `agent-loadgen plan` before `agent-loadgen generate`; a failed or oversized plan prevents traffic.

Each invocation creates a new timestamped campaign directory. `campaign.json` records software, source revision, binary digest, profile file and semantic digests, redacted commands, plan digests, target identity, run status, and evidence classification. The wrapper verifies the semantic digest across the plan output, planned scenario, and trace manifest; it records the TOML byte hash separately. The `plan/` directory contains the generated causal graph. The `run/` directory contains `run.json`, `requests.jsonl`, and the generated scenario. Stage stdout and stderr logs live beside `campaign.json` with supplied header values redacted.

The smoke passes when `campaign.json` has `status: completed`, `classification.transport_passed: true`, and `run.protocol_surface: chat_completions`. It always records `classification.performance_qualified: false`.

At the pinned loadgen commit, the checked-in profile resolves to four requests, 1,120 aggregate input tokens, 64 aggregate output tokens, semantic profile digest `f1ac5afe584bc913c6c29ad7b142640b4db61173dedd1b215d1f88bc8fcb546f`, and scenario digest `298da3dd28179d89d3ca03a7f0189f0abe1c63f625b50821c6104a5bff2ad388`. The wrapper rejects inconsistent plan/scenario digests or a request count above the declared guard.

## Run the Independent AIPerf Endpoint Smoke

Use the AIPerf revision pinned by Router Zoo's AgentX benchmark to verify endpoint readiness, streaming Chat Completions, tokenizer resolution, request accounting, and artifact export independently of agent-loadgen. This command is intentionally c1 with eight requests. It is an endpoint smoke, not a performance result.

```bash
export AIPERF_ARTIFACT_DIR="$PWD/aiperf-artifacts/endpoint-smoke"

uvx \
  --isolated \
  --no-config \
  --refresh-package aiperf \
  --from git+https://github.com/ai-dynamo/aiperf.git@0883bd1aee552472124aa710e4cf067b7b77cddb \
  aiperf profile \
  --url "$DYNAMO_BASE_URL" \
  --model "$DYNAMO_MODEL" \
  --endpoint-type chat \
  --streaming \
  --use-server-token-count \
  --wait-for-model-timeout 300 \
  --wait-for-model-mode both \
  --concurrency 1 \
  --request-count 8 \
  --isl 128 \
  --osl 16 \
  --random-seed 20260821 \
  --artifact-dir "$AIPERF_ARTIFACT_DIR"
```

Add AIPerf's `--api-key` or repeated `--header` options when the endpoint requires authentication. The smoke passes when the command exits zero, `profile_export_aiperf.json` reports eight requests, `error_summary` is empty, and `was_cancelled` is false. Keep the artifact directory immutable after the run. Do not use its latency or throughput values as capacity evidence: c1 is too small, and endpoint readiness does not establish matched treatments, telemetry coverage, or workload calibration.

## Performance Measurement Gate

Use `--intent performance-measurement` only when the recipe's token path and cache settings have been independently verified. The wrapper requires both declarations:

```bash
python3 examples/agent_harnesses/benchmark/run_campaign.py \
  --loadgen ./agent-loadgen/target/release/agent-loadgen \
  --loadgen-source ./agent-loadgen \
  --intent performance-measurement \
  --token-path-verified \
  --engine-cache-mode ownership=session \
  --profile /path/to/a-calibrated-bounded-profile.toml \
  --max-planned-requests 10000 \
  --output-root ./agent-loadgen-artifacts
```

This intent can set `classification.agent_loadgen_performance_eligible: true` when the loadgen run also passes its fidelity gates. It still does not set `performance_qualified: true`: a calibrated workload, repetitions, treatment controls, and correlated telemetry are separate requirements.

## Qualification Phases

| Phase | Tooling and action | Exit boundary |
|---|---|---|
| 0. Recipe deployment | Deploy any supported Dynamo Kubernetes recipe and record its immutable recipe revision, resolved configuration, root service URL, served model, topology, and engine/cache settings. | The endpoint passes the recipe's own readiness and basic model smoke checks. |
| 1. Endpoint and agent smoke | Run the pinned AIPerf c1 smoke and this four-request agent-loadgen profile with `--intent transport-smoke`. | Readiness, streaming Chat Completions, Codex-shaped headers, causal release, output accounting, and artifact creation pass. No performance claim is allowed. |
| 2. Direct baseline campaigns | Execute a frozen, calibrated agent-loadgen profile at declared concurrency points and repetitions. Keep the profile, seed, target model, deployment capacity, and client placement fixed across treatments. | Every campaign has immutable configuration, request artifacts, treatment identity, warmup policy, and successful repetitions. |
| 3. Telemetry collection | Correlate agent-loadgen request records with Dynamo request traces, router decisions and metrics, engine metrics, and GPU telemetry such as DCGM. Use Tachometer or equivalent immutable report packaging when explicit Prometheus endpoints are available. | Request IDs and timestamps join across client, router, engine, and GPU evidence; missing or ambiguous joins invalidate performance conclusions. |
| 4. Router Zoo integration | Add and validate an external-endpoint execution mode before using Router Zoo with a recipe-deployed service. Current Router Zoo runners create their own frontend and mock workers and cannot attach directly to an arbitrary existing endpoint. | The external mode skips builds and mock workers, records endpoint identity and credentials safely, and captures telemetry from the deployed recipe. |
| 5. Routing A/B | Run matched no-affinity, stock session-affinity, and ThunderAgent treatments. Change only the routing treatment and use the same campaign order or a documented randomization to control drift. | Report latency, throughput, dispatch lag, route stickiness, KV reuse/cache behavior, errors, and resource utilization with repetitions and uncertainty. Sequential smoke runs are not an A/B. |
| 6. Qualification and optimization | Check calibration, token-path verification, cache declarations, reproducibility, telemetry, treatment isolation, and statistical sufficiency. Only after the baseline passes, use a new Router Autoresearch campaign and Router Forge policy changes. | Evidence failing any gate remains transport-only or exploratory. Forge policy candidates require a controlled Dynamo rebuild and must repeat the matched campaign; they cannot be loaded into an arbitrary live endpoint. |

Tachometer cannot infer telemetry sources from the OpenAI-compatible URL; provide explicit Prometheus endpoints for Dynamo, DCGM Exporter, and node-exporter as applicable. It packages infrastructure metrics, not request-to-worker routing decisions, so retain router traces or metrics for causal joins. Router Autoresearch and Router Forge are optimization tools rather than first-smoke dependencies.

## DeepSeek Harness Dependency

The pinned generic loadgen commit does not contain a DeepSeek Harness renderer. DSH qualification therefore remains blocked on a separate draft change in `NVIDIA-dev/agent-loadgen`, based on commit `9057201e23663baaaf076820f3772d55468dec25`, that adds native DSH session and compaction headers without changing the causal graph. That draft must stay separate from this generic path and must be recorded by immutable full commit SHA and draft PR URL before a DSH campaign runs.

The DSH renderer must emit `x-deepseek-harness-session-id` for the current session, emit `x-deepseek-harness-compact: 1` only for compaction, omit a parent header because the published DSH package does not provide one, and omit canonical `x-dynamo-session-id` while testing native normalization so Dynamo header precedence does not mask the DSH path. A static `x-deepseek-harness-user-id` campaign header is optional. Do not patch DSH source for this integration.

Until that pinned draft exists and passes deterministic header/profile tests, use this generic path only for supported renderers. Do not approximate DSH with repeated static `--header` values because session and compaction values change within a causal campaign.
