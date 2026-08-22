<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared NScale well-lit-path qualification — 2026-08-21

## Frozen deployment

- Cluster: `dynamo-nscale-dev-cluster`
- Namespace: `anish-agent-well-lit-path`
- Runtime: `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:effd250754b8a70517c27eab8f18463b395a7b2a8e868fd919226c3180636939`
- Model: `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Project allocation: one exact-count DRA GPU, one project-owned 20 GiB RWX model-cache PVC, two nodes maximum

## Stock arm

The stock round-robin graph became Ready with one CPU frontend and one vLLM worker on two nodes and one GPU. Codex proved persistent Responses transport and permission enforcement; DeepSeek Harness proved streamed Chat Completions transport under one native session; Omnigent returned exactly `OMNIGENT_STOCK_OK` through its pinned Codex harness with invocation-scoped cleanup. The stock graph, both pods, and its DRA claim were deleted and observed absent before ThunderAgent was created.

## ThunderAgent arm

The authoritative graph became Ready with frontend `agent-well-lit-thunderagent-0-frontend-bcfp4` and router `agent-well-lit-thunderagent-0-thunderagentrouter-cqzrc` co-located on CPU node `cluster-0967a26d-pool-1f83edbe-mj5s4-gnsf4`. Worker `agent-well-lit-thunderagent-0-vllmdecodeworker-fsg4j` held only `gpu-6` on `cluster-0967a26d-pool-14bee067-prctr-d6dn5`. This was exactly two nodes and one GPU.

Codex thread `01a02330-79ae-7e11-a9be-c9d27a37456e` returned exactly `CODEX_THUNDERAGENT_OK`, emitted one successful terminal record, and was released by the router with zero programs remaining. DSH session `session-aa5d0507-f889-45a5-bdaf-be1775584392` received two HTTP 200 model responses, emitted an HTTP 200 terminal record, exited 0, and was independently released by the router with zero programs remaining. Omnigent is intentionally stock-only because the audited harness does not emit Dynamo session finalization.

```text
2026-08-21T07:20:23.592650Z Released program 01a02330-79ae-7e11-a9be-c9d27a37456e (0 remaining)
2026-08-21T07:20:47.252292Z Released program session-aa5d0507-f889-45a5-bdaf-be1775584392 (0 remaining)
```

## Compatibility findings resolved

- The pinned Dynamo 1.3 worker rejected the newer `--endpoint-types none` value. The final manifest gives the backend a private served-model alias and leaves the public model name to ThunderAgent.
- ThunderAgent's model card references the pinned local snapshot. The frontend now mounts the project cache read-only so it resolves that card locally instead of treating the path as a Hugging Face model ID.
- Router and frontend have required same-node affinity, all images are immutable digests, the model revision is pinned, and same-namespace-only ingress is enforced without modifying shared taints or labels.
- The pinned image does not emit the newer `thunderagent.route path=program|session_final` labels. Lifecycle proof therefore combines each client's successful terminal trace with the router's same-ID `Released program ... (0 remaining)` record.

## Cleanup

The ThunderAgent graph, all three pods, its DRA claim, the project claim template, NetworkPolicy, and 20 GiB project cache were deleted and observed absent. The remaining inventory contained only the documented namespace/platform objects: root CA, default ServiceAccount, admission registry secret and namespace cache, plus the Dynamo-operator-managed planner ServiceAccount and binding.

Namespace deletion removed that content but the namespace controller remained blocked by stale cluster-wide `cluster.loft.sh/v1` discovery. Its conditions reported `All content successfully removed` and `All content-preserving finalizers finished`; after several retries, only the exact project namespace's Kubernetes finalizer was cleared. No CRD, API service, node, taint, label, other namespace, or other user's object was changed. The namespace and all cluster resources labeled with project owner `anish-maddipoti` were then verified absent.

The deleted project cache and admission-created namespace cache are not recoverable.
