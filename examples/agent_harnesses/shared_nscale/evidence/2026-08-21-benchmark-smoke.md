<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Controlled-load transport qualification — 2026-08-21

## Result

The recipe-endpoint, AIPerf, Codex-shaped agent-loadgen, and native DeepSeek Harness agent-loadgen transport paths passed against a CPU-only Dynamo Mocker deployment. The deployment used zero GPUs and placed its frontend and two advertised mock workers on one unprotected CPU node. This is transport-only evidence; Mocker latency and throughput are not performance results, and the no-affinity/session-affinity/ThunderAgent GPU A/B remains blocked on safe shared-cluster capacity.

## Frozen inputs

| Artifact | Revision or identity |
|---|---|
| Dynamo source baseline | `a6261680a974ca7c74dcf49592a7376d7de99380` |
| Runtime image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:effd250754b8a70517c27eab8f18463b395a7b2a8e868fd919226c3180636939` |
| Model/tokenizer identity | `Qwen/Qwen3-0.6B` |
| Generic agent-loadgen | `NVIDIA-dev/agent-loadgen@9057201e23663baaaf076820f3772d55468dec25` |
| DSH renderer patch | Local signed draft commit `2b246254cc91a6b9b7951116444fdaed4f90e9df`; upstream repository policy denied both a fork and direct branch push |
| AIPerf | `ai-dynamo/aiperf@0883bd1aee552472124aa710e4cf067b7b77cddb` |

## Kubernetes deployment

`benchmark-mocker-dgd.yaml` created one CPU frontend and one CPU Mocker process with `--num-workers 2`, no GPU resource requests, and the pinned runtime image. Both pods became Ready on `cluster-0967a26d-pool-1f83edbe-mj5s4-lhwhj`; the DGD reported `Ready=True` with `All resources are ready`. `/v1/models` returned `Qwen/Qwen3-0.6B`, and a unary Chat Completions request returned HTTP 200 with eight output tokens.

The first Mocker rollout failed before readiness because the frozen source tree documents `--max-model-len` but the pinned runtime image does not expose that flag. Removing the optional flag and reapplying the same image digest produced a Ready deployment. The checked-in manifest reflects the tested runtime contract.

## Load tests

### AIPerf endpoint smoke

The pinned AIPerf command ran at concurrency 1 with eight requests, ISL 128, OSL 16, and seed `20260821`. It completed with exit 0, `request_count.avg=8`, an empty `error_summary`, and `was_cancelled=false`. No GPU telemetry was collected. The reported Mocker latency and throughput are intentionally excluded from qualification claims.

### Codex-shaped causal smoke

The generic campaign wrapper planned before traffic, verified the clean pinned source and binary digest, and then completed four of four requests over Chat Completions. The causal graph contained two concurrent top-level sessions and two sequential turns per session, with 1,120 aggregate input tokens and 64 output tokens. The semantic profile digest was `f1ac5afe584bc913c6c29ad7b142640b4db61173dedd1b215d1f88bc8fcb546f`; the scenario digest was `298da3dd28179d89d3ca03a7f0189f0abe1c63f625b50821c6104a5bff2ad388`. The run recorded `transport_passed=true`, `performance_qualified=false`, `token_path_unverified`, and `engine_cache_mode_undeclared`.

The first wrapper invocation correctly stopped before generate but exposed a validation bug: agent-loadgen's semantic profile digest had been compared to the TOML byte hash. The wrapper now records both values separately and verifies the semantic digest across plan output, `scenario.json`, and the trace manifest. Nine hermetic wrapper tests and the repeated live campaign passed after the fix.

### DeepSeek Harness persistent-session smoke

The DSH renderer planned and completed four of four requests with HTTP 200. The graph contained two concurrent sessions with two causally ordered requests per session, proving this load path is persistent rather than one-shot. It emitted native `x-deepseek-harness-session-id`, omits canonical Dynamo and unsupported parent headers, and reserves `x-deepseek-harness-compact: 1` for planned compaction attempts. The bounded smoke disables compaction and therefore did not send that marker in this run. Its semantic profile digest was `5c7e1cdd26d971b080439e683ad13b082f488dfa650c50a7c37c63f1f7fb5df9`; its scenario digest was `affa5c2aaf4deff02fba7a28b1fad4287c25dbb6238c6cb02504918a1495f04e`. All 54 Rust workspace tests and clippy passed before the live run.

## GPU routing campaign status

The no-affinity and session-affinity stock manifests plus the ThunderAgent manifest passed client and server dry-run validation. A guarded attempt to schedule the two-GPU stock graph was rejected immediately: DRA-only nodes could not satisfy classic device-plugin requests, the remaining classic node had no safe capacity, and reservation-tainted nodes were intentionally ineligible. No GPU was allocated. The pending graph and ConfigMap were deleted immediately, and no taint, label, node, cordon, drain, reservation, other namespace, or other user's workload or storage was changed.

Router Zoo cannot currently target an arbitrary recipe-deployed endpoint because its runners create their own frontend and mock workers. Tachometer needs explicit Prometheus URLs and does not supply request-to-worker routing causality by itself. Router Autoresearch and Router Forge remain post-baseline work; Forge policy changes require a controlled Dynamo rebuild. The cluster-agnostic benchmark guide states these boundaries and provides the direct AIPerf plus agent-loadgen path that works today.

## Cleanup

The port-forward, Mocker DGD, pods, project cache, ResourceClaimTemplate, NetworkPolicy, and namespace were removed after evidence capture. Kubernetes reported all namespace content removed; deletion then encountered the cluster's pre-existing stale `cluster.loft.sh/v1` discovery entry. Only the empty project namespace finalizer was cleared. A final lookup found no project namespace, owner-labeled pod, or owner-labeled DGD.
