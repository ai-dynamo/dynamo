---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Operate Shared GPUs
subtitle: Plan for faults, resize models, guard against silent limit failures, and fix common problems with GPU fractions
---

Models on a shared GPU affect each other in ways that models on dedicated GPUs don't. This page covers what to plan for once GPU fractions runs in production: where a fault can spread, how to resize a model, the MPS v3 failures that are silent, and what to check when something goes wrong.

## Plan for Faults

Under GPU fractions, the models on a GPU share one MPS server, and some faults in one model stop every model on that GPU:

| Fault in one model | Typical cause | What it stops |
|---|---|---|
| Memory fault, such as `an illegal memory access was encountered` | A kernel bug, such as an out-of-bounds access on an input shape the kernel didn't expect | Every model on the GPU |
| Trap or breakpoint, such as `unspecified launch failure` | A library that stops on a fatal condition, such as a DeepEP communication timeout | Every model on the GPU |
| Device-side assert, such as `device-side assert triggered` | An index check, such as a token ID outside the vocabulary | Only the model that asserted |

Every model that a fault stops must restart. Models on other GPUs keep serving. Place models with that in mind:

- **Spread the replicas of a critical model across GPUs or nodes.** Two replicas on one GPU can fail together.
- **Keep the most critical models on dedicated GPUs.** A neighbor can't take them down there.
- **Use MIG or separate GPUs for tenants that don't trust each other.** Each MIG instance has its own fault domain.

[MPS Fault Isolation](../../developer-guide/knowledge-base/concepts/gpu-sharing/mps-fault-isolation.md) shows how each fault behaved in a test, and how static SM partitions contain memory faults.

## Resize or Update a Model

A model's share is set when its worker pod starts. To change it, change the worker's annotations and restart that model. Its neighbors keep running.

Don't edit a running DGD's worker in place. The operator starts the new worker before it removes the old one, and when every share on the GPU is allocated, the new worker stays `Pending` indefinitely. Deleting the old pod by hand doesn't help, because the operator recreates it. Delete the DGD and apply it again:

```bash
kubectl delete dgd <name> -n <namespace>
kubectl apply -f <dgd-file>.yaml
```

## Avoid the MPS v3 Sharp Edges

MPS v3 enforces limits only for clients that actually connect to their namespace, and some failures are silent. The GPU fractioning agents guard against most of them, but verify the limits on every new deployment, as [Enable GPU Sharing](enable-gpu-sharing.mdx#run-the-smoke-test) shows.

- **A client that misses its namespace runs unlimited.** In the default compute mode, a client whose pipe directory is missing or misspelled runs as an ordinary CUDA process with the whole GPU, and nothing reports an error.
- **A configuration error can stop MPS without an error.** A namespace name the daemon can't parse, such as one with a hyphen, or a server whose allowed devices name a GPU that isn't on the node, leaves no server running, while the start command still succeeds. Check `server list` and the MPS logs after every start.
- **One bad per-GPU memory entry removes every memory cap.** If any entry in a client's `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` names a GPU the client can't see, the client drops every pinned memory limit, including its namespace's. Its compute limit still applies.
- **A root container can raise its own limits.** Run every fractional worker as non-root, and enforce it with Pod Security Admission or a policy engine.
- **The reported total memory is the whole GPU.** Under a pinned memory limit, `cudaMemGetInfo` reports the whole GPU as total memory. Set the engine's memory fraction from the cap explicitly, such as vLLM's `--gpu-memory-utilization`, as [Size Each Model's Share](size-each-models-share.mdx#work-out-each-engines-memory-setting) shows.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| A fault in one model affects every model on its GPU | Spread replicas of critical models across GPUs or nodes; keep the most critical models on dedicated GPUs |
| A busy neighbor slows a model through shared memory bandwidth, L2 cache, or power | Measure each share with the rest of the GPU loaded; don't pair two latency-critical models; leave headroom |
| A pod that misses its MPS namespace runs with the whole GPU and no error | Run the smoke test after every install and upgrade, check the limits on new deployments, and run workers as non-root |
| An engine sizes its memory from the GPU's total and fails or overruns its cap | Set the engine's memory fraction from the cap explicitly |
| Updating a DGD in place hangs when every share on a node is allocated | Delete and re-apply the DGD instead of editing it |
| The scheduler components are pre-release | Pin the commits, validate in staging, and keep a dedicated-GPU or MIG fallback |

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| A fractional pod stays `Pending`, and its events say nothing useful | KAI's reservation pod has no `nvidia` RuntimeClass and can't load NVML. Check `kubectl -n kai-resource-reservation logs <gpu-reservation-pod>` | Reinstall KAI-Scheduler with `--set binder.runtimeClassName=nvidia` |
| A fractional pod stays `Pending` with a `BindingError` about the `nvidia.com/gpu.memory` label | The node has no GPU memory label | Run GPU Feature Discovery, part of the GPU Operator, and confirm with `kubectl get nodes -L nvidia.com/gpu.memory` |
| A fractional pod stays `Pending` with a queue-quota reason | The KAI queues have no quota for the workload | Open the queue quotas, as the [GPU fractions deploy guide](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/docs/deploy-kai-gpu-fractions.md#queues) shows |
| A pod is rejected at admission | It asks for both a GPU fraction and `nvidia.com/gpu` | Remove `nvidia.com/gpu` from fractional workers |
| The container gets `NVIDIA_VISIBLE_DEVICES=void`, and no `gpus.devices` annotation | The scheduler and the node agent come from different builds | Rebuild and reinstall both from the pinned commits |
| The container has no `CUDA_MPS_*` variables | `fractiond` didn't register as an NRI plugin | Check `kubectl -n gpu-fractioning logs ds/gpu-fractioning-fractiond`, and that NRI is enabled in the container runtime |
| An `sm-sharing` pod won't start, and `mpsd` logs a namespace error | Namespace setup failed, and the node agent refuses to start the container uncapped | Check `kubectl -n gpu-fractioning logs ds/gpu-fractioning-mpsd` |
| `mpsd` exits with `CUDA_ERROR_OUT_OF_MEMORY` while the GPUs are idle | The `mpsd` pod's host-memory limit is too small for the number of GPUs | Raise `mpsDaemon.resources.limits.memory`, about 50 MiB per GPU |
| The frontend lists no models, and chat requests return 404 | The platform runs without etcd and NATS, and the DGDs use etcd discovery | Install the platform with etcd and NATS, as the deploy guide's `install-dynamo-platform.sh` does |
| Pods won't start after repeated teardown | MPS state outlives the deleted pods | Delete the `gpu-fractioning-mpsd` pods to restart the daemons |

The [full build guide](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/docs/build-gpu-fractioning-forks.md#troubleshooting) covers build and install problems in more detail.
