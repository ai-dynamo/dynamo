<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Shared NScale Tier 0 backend

This branch owns the common NScale backend used to qualify the Codex, DeepSeek Harness, and Omnigent well-lit paths. Harness-specific clients and patches stay on their three isolated branches; this shared branch contains only common cluster topology and evidence.

## Safety contract

- Cluster: `dynamo-nscale-dev-cluster`.
- Namespace: `anish-agent-well-lit-path`, created only for this project.
- Initial capacity: one NVIDIA DRA GPU on one node. The project hard ceiling remains two nodes and 16 GPUs across every overlapping well-lit-path workload.
- Never select a node or device UUID, copy another namespace's claim/PVC/secret, tolerate a reservation-specific taint, or mutate labels, taints, cordons, drains, nodes, or another user's resources.
- Recalculate classic and DRA allocation immediately before every apply. Stop if the only capacity is protected by another user's taint or reservation.

## Frozen Tier 0 tuple

| Artifact | Value |
| --- | --- |
| Dynamo source baseline | `a6261680a974ca7c74dcf49592a7376d7de99380` |
| DGD API | `nvidia.com/v1beta1` |
| Runtime image | `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:effd250754b8a70517c27eab8f18463b395a7b2a8e868fd919226c3180636939` |
| Model | `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca` |
| Frontend | CPU, round-robin stock Dynamo, port 8000 |
| Worker | vLLM, one DRA GPU, `--gpu-memory-utilization 0.70`, `--max-model-len 32768`, `hermes` tool parser, `qwen3` reasoning parser |
| Model storage | Project-owned 20 GiB RWX VAST PVC |

The runtime image is a cluster-proven baseline, not the frozen source branch image. It predates the branch's native DSH header mapping. DSH cluster tests against this image must either send canonical Dynamo headers through the documented compatibility relay or limit claims to model/tool behavior; native DSH normalization remains a branch-local Rust proof until a branch image is published.

## Apply in guarded phases

First create the namespace and its ingress boundary, then inspect both:

```bash
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/namespace.yaml
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/network-policy.yaml
kubectl get namespace anish-agent-well-lit-path --show-labels
kubectl get all,configmap,secret,pvc,serviceaccount,role,rolebinding,networkpolicy,resourceclaimtemplate,resourceclaim,dynamographdeployment -n anish-agent-well-lit-path
```

The ingress policy allows traffic only from pods in this namespace. It preserves the operator-created ClusterIP services and `kubectl port-forward`, but it is a namespace trust boundary rather than request authentication: every pod admitted to this namespace can reach the frontends. Keep the namespace project-exclusive. Validate the policy object with `kubectl describe networkpolicy agent-well-lit-same-namespace-ingress -n anish-agent-well-lit-path`, then bind the port-forward to loopback and validate only through `http://127.0.0.1:8000` as shown below. A live cross-namespace denial test requires creating a probe pod and is intentionally outside this read-only validation.

Then apply the project-owned PVC and one-GPU claim template, followed by the stock graph:

```bash
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/storage-and-gpu.yaml
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/stock-dgd.yaml
kubectl wait -n anish-agent-well-lit-path --for=condition=Ready dynamographdeployment/agent-well-lit-stock --timeout=20m
```

Inspect the operator-created resources and allocated device before running a client:

```bash
kubectl get pods,pvc,resourceclaim -n anish-agent-well-lit-path -o wide
kubectl get resourceclaim -n anish-agent-well-lit-path -o yaml
kubectl get dynamographdeployment -n anish-agent-well-lit-path agent-well-lit-stock -o yaml
```

Port-forward the project-owned frontend only:

```bash
kubectl -n anish-agent-well-lit-path port-forward --address 127.0.0.1 svc/agent-well-lit-stock-frontend 8000:8000
curl -fsS http://127.0.0.1:8000/v1/models
```

## ThunderAgent arm

Run the stock and ThunderAgent graphs sequentially. Never create both graphs at once: each declares a one-GPU worker, and an overlapping rollout could transiently exceed the intended Tier 0 allocation or deadlock on the final free device.

After preserving the stock-arm evidence, remove only its graph and wait for the DGD, every project pod, and every ResourceClaim to disappear. Keep the project PVC, claim template, and NetworkPolicy:

```bash
kubectl delete -f examples/agent_harnesses/shared_nscale/kubernetes/stock-dgd.yaml
kubectl wait -n anish-agent-well-lit-path --for=delete dynamographdeployment/agent-well-lit-stock --timeout=5m
kubectl wait -n anish-agent-well-lit-path --for=delete pod -l app.kubernetes.io/part-of=agent-well-lit-path --timeout=5m
kubectl wait -n anish-agent-well-lit-path --for=delete resourceclaim --all --timeout=5m
test -z "$(kubectl get pods -n anish-agent-well-lit-path -l app.kubernetes.io/part-of=agent-well-lit-path -o name)"
kubectl get pods -A -o wide
kubectl get resourceclaim -A -o wide
kubectl get resourceslice -o wide
kubectl get nodes -o custom-columns='NAME:.metadata.name,UNSCHEDULABLE:.spec.unschedulable,TAINTS:.spec.taints'
kubectl apply --dry-run=server -f examples/agent_harnesses/shared_nscale/kubernetes/thunderagent-dgd.yaml
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/thunderagent-dgd.yaml
kubectl wait -n anish-agent-well-lit-path --for=condition=Ready dynamographdeployment/agent-well-lit-thunderagent --timeout=20m
```

Stop before the ThunderAgent apply until the read-only capacity inventory has been recalculated: the Frontend and ThunderAgentRouter must fit together on one unprotected CPU node, the single DRA claim must fit on one unprotected GPU node, and those two nodes must keep the project at or below its two-node/16-GPU ceiling. Do not use a node protected by another user's reservation or modify any taint, label, cordon, or drain state to make the graph fit.

The ThunderAgentRouter has required pod affinity to the same graph's Frontend, so both CPU components occupy one node while the one-GPU vLLM worker may occupy a second node. The router and frontend mount the project model cache read-only; the router waits for the pinned snapshot's `config.json` for at most ten minutes before starting, and the frontend can resolve the router's local-path model card without downloading or mistaking that path for a Hugging Face model ID. The pinned Dynamo 1.3 runtime predates the newer `--endpoint-types none` worker mode, so the vLLM worker advertises only the private `agent-well-lit-thunderagent-backend` alias while `dynamo.thunderagent_router` alone registers the public `Qwen/Qwen3-0.6B` surface. This prevents requests for the documented public model from bypassing lifecycle handling. The runtime image contains the experimental ThunderAgent entry point; its API and lifecycle contract are not production-stable.

Port-forward `svc/agent-well-lit-thunderagent-frontend`, run lifecycle-qualified Codex and DeepSeek clients with their explicit `--session-final` options, then match each client's normalized session ID to the router lifecycle record. Newer source builds emit both `path=program` and `path=session_final`; the pinned Dynamo 1.3 image instead proves terminal handling with `Released program <id> (0 remaining)`, paired with the client's successful model response and HTTP 200 terminal trace for that same ID. Omnigent does not emit a final signal and is stock-only in this qualification.

Do not delete the namespace until required traces, logs, image/model identities, and route evidence have been copied. Delete the resources authored by this branch directly, then inventory the namespace before the broader namespace deletion:

```bash
kubectl delete -f examples/agent_harnesses/shared_nscale/kubernetes/stock-dgd.yaml --ignore-not-found
kubectl delete -f examples/agent_harnesses/shared_nscale/kubernetes/thunderagent-dgd.yaml --ignore-not-found
kubectl delete -f examples/agent_harnesses/shared_nscale/kubernetes/benchmark-mocker-dgd.yaml --ignore-not-found
kubectl delete -f examples/agent_harnesses/shared_nscale/kubernetes/storage-and-gpu.yaml
kubectl delete -f examples/agent_harnesses/shared_nscale/kubernetes/network-policy.yaml
kubectl get all,configmap,secret,pvc,serviceaccount,role,rolebinding,networkpolicy,resourceclaimtemplate,resourceclaim,dynamographdeployment -n anish-agent-well-lit-path -o wide
kubectl delete -f examples/agent_harnesses/shared_nscale/kubernetes/namespace.yaml
```

The platform admission layer creates `acr-token-secret` and the `shared-model-cache` PVC in this namespace; Kubernetes creates the default ServiceAccount and root CA ConfigMap, and the Dynamo operator creates the `planner-serviceaccount` plus `planner-serviceaccount-binding` for the cluster planner role. They are not authored or directly deleted by this branch. The final inventory must contain no project pods, graphs, claims, project RBAC, or unexpected user objects. Stop and investigate rather than deleting the namespace if anything beyond these documented platform and namespace-baseline objects remains. Namespace deletion is intentionally broader than the direct cleanup above and removes them along with every other namespaced object.

Deleting `agent-well-lit-model-cache` removes the project-owned model cache, and deleting the namespace removes the admission-created `shared-model-cache` too. Preserve the project cache between stock and ThunderAgent arms when reproducibility and startup cost matter; do not point either arm at another namespace's cache.

## Internal controlled-load extension

The files prefixed `benchmark-` are an internal qualification extension for the cluster named above. They are not part of the cluster-agnostic operator guides. They keep the public contract recipe-first while providing a reproducible two-worker endpoint for the shared `agent-loadgen` campaign.

`benchmark-mocker-dgd.yaml` is the zero-GPU transport fixture. Use it first to validate endpoint discovery, the campaign runner, AIPerf, and harness-specific request shapes when shared GPU capacity is unavailable. It advertises two mock workers from one CPU process and is explicitly unsuitable for latency, throughput, cache-benefit, routing-benefit, or capacity claims. The tested result and cleanup record are in `evidence/2026-08-21-benchmark-smoke.md`.

```bash
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/storage-and-gpu.yaml
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/benchmark-mocker-dgd.yaml
kubectl wait -n anish-agent-well-lit-path --for=condition=Ready dynamographdeployment/agent-well-lit-benchmark-mocker --timeout=10m
kubectl -n anish-agent-well-lit-path port-forward --address 127.0.0.1 svc/agent-well-lit-benchmark-mocker-frontend 8000:8000
```

Delete the Mocker DGD and wait for both CPU pods to disappear before starting any GPU arm.

The stock graph is paired with exactly one router ConfigMap at a time. `benchmark-no-affinity-config.yaml` and `benchmark-session-affinity-config.yaml` both use the KV router and differ only by the 300-second session-affinity TTL, making the first arm a controlled no-affinity baseline. Delete the graph and ConfigMap between arms so the frontend starts with a clean router state.

Both stock and ThunderAgent graphs request two classic GPUs. Required self-affinity places both worker replicas on one unprotected GPU node; the CPU frontend, and the co-located ThunderAgent router when present, occupy at most one additional CPU node. The pods tolerate only the standard GPU taint, so nodes protected by another team's reservation taint remain ineligible. Recalculate allocation immediately before every apply and stop if one unprotected GPU node no longer has two free devices.

Apply the model cache from `storage-and-gpu.yaml`, but do not apply either older one-GPU graph. Run the three benchmark arms sequentially:

```bash
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/storage-and-gpu.yaml
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/benchmark-no-affinity-config.yaml
kubectl apply -f examples/agent_harnesses/shared_nscale/kubernetes/benchmark-stock-dgd.yaml
```

After preserving the first result, delete `agent-well-lit-benchmark-stock` and `agent-well-lit-benchmark-router`, verify that all project GPU pods are gone, then repeat with `benchmark-session-affinity-config.yaml` and the same stock graph. After the affinity result, clean the graph and ConfigMap again before applying `benchmark-thunderagent-dgd.yaml`. Never overlap two arms.

These four-request smoke campaigns validate endpoint compatibility, causal scheduling, headers, result capture, and bounded concurrency. They are not enough for a routing-performance claim. The larger concurrency and saturation series, repetitions, telemetry bundle, and failure campaign remain separate gates.
