---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy the Baseline Experiment
subtitle: Run 8 Qwen3-4B models on 8 A100 GPUs, one model per GPU, as the reference for the GPU sharing experiments
---

This guide deploys the baseline for the many-model GPU sharing benchmark: eight Qwen3-4B models, each in its own `DynamoGraphDeployment` (DGD), each on a dedicated A100 40GB GPU. You then sweep concurrency against all eight models at once. The numbers you collect are the denominator for every percentage in [Interpreting GPU Sharing Results](gpu-sharing-results.mdx).

The scripts, manifests, and results for this experiment are co-authored by [@marckarp](https://github.com/marckarp) and [@scheckerNV](https://github.com/scheckerNV).

The baseline uses the stock Kubernetes scheduler and whole-GPU requests, so it needs no scheduler or isolation add-ons. The other two experiments pack two models per GPU and differ only in how they isolate the models:

| Experiment | DGDs | Placement | Isolation | Guide |
|---|---:|---|---|---|
| Baseline (this page) | 8 | One model per GPU (`nvidia.com/gpu: "1"`) | Dedicated GPU | This page |
| KAI + HAMi | 16 | Two models per GPU | Memory cap only; compute is time-sliced | [Deploy the KAI + HAMi Experiment](deploy-kai-hami.md) |
| KAI + GPU fractions | 16 | Two models per GPU | Memory cap and 50% of the SMs | [Deploy the KAI + GPU Fractions Experiment](deploy-kai-gpu-fractions.md) |

All commands run from `examples/gpu-sharing` in a checkout of the [Dynamo repository](https://github.com/ai-dynamo/dynamo). The files this guide uses are in [`examples/gpu-sharing/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing).

> [!NOTE]
> Any performance results on this page are purely illustrative and are not indicative of optimal performance. Your deployment or configuration may vary.

## Requirements

This guide assumes a multi-node Kubernetes cluster that you administer from a workstation with `kubectl` and `helm`. All eight models run on one node with eight A100 40GB GPUs. The model cache is a `hostPath` directory on that node, so every frontend and worker is pinned to it. The load generator runs in its own pod on a different node and reaches each frontend through its Service DNS name.

| Requirement | Published run | Check | References |
|---|---|---|---|
| One node with 8 NVIDIA A100 40GB GPUs, with no other GPU workloads on it | 8x A100-SXM4 40GB | `kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count` | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/), [GPU Feature Discovery labels](https://github.com/NVIDIA/k8s-device-plugin/blob/main/docs/gpu-feature-discovery/README.md) |
| NVIDIA driver on that node | 615.71.09 | `kubectl get node "$GPU_NODE" -o jsonpath='{.metadata.labels.nvidia\.com/cuda\.driver-version\.full}'` | [Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) |
| NVIDIA GPU Operator | | `kubectl get pods -n gpu-operator` | [Installing the GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) |
| `nvidia` RuntimeClass | Created by the GPU Operator | `kubectl get runtimeclass nvidia` | [Kubernetes RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/) |
| `kubectl` and `helm` on your workstation, with cluster-admin | | `kubectl auth can-i '*' '*' --all-namespaces` | [Install kubectl](https://kubernetes.io/docs/tasks/tools/), [Installing Helm](https://helm.sh/docs/intro/install/) |
| A second node with 16 free CPUs and 32 GiB of memory, for the load generator | AIPerf 0.11.0 | `kubectl describe nodes` (allocatable minus allocated) | [AIPerf](https://github.com/ai-dynamo/aiperf) |
| Pods in the `default` namespace may run as root and mount `hostPath` volumes | | `kubectl get ns default --show-labels` shows no `pod-security.kubernetes.io/enforce` label stricter than `privileged` | [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/), [Kubernetes `hostPath` volumes](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath) |
| Egress to Hugging Face, `nvcr.io`, and PyPI | | | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Dynamo vLLM runtime on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime), [aiperf 0.11.0 on PyPI](https://pypi.org/project/aiperf/0.11.0/) |

The download Job's init container runs as root to create the cache directory, and the DGDs mount it as `hostPath`. If your cluster enforces the `baseline` or `restricted` Pod Security Standard, run the experiment in a namespace labeled `pod-security.kubernetes.io/enforce=privileged`, and set `NAMESPACE` for every script and command below.

Pick the GPU node and export its name; every later step uses it:

```bash
kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count
export GPU_NODE=<node-name>
kubectl get node "$GPU_NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}{"\n"}'   # 8
```

## Step 1: Install the Dynamo Platform

```bash
cd examples/gpu-sharing
common/install-dynamo-platform.sh
```

The script installs the `dynamo-platform` Helm chart, version 1.4.2, into the `dynamo-system` namespace and waits for its pods to be Ready. It sets `global.etcd.install=true` and `global.nats.install=true`, which is deliberate. Platform 1.4.x installs without etcd and NATS by default and configures Kubernetes-based discovery, and with that default the `vllm-runtime:1.3.0` frontend never lists the model: `/v1/models` is empty and chat requests return 404. The generated DGDs set `DYN_DISCOVERY_BACKEND=etcd`, which needs the etcd and NATS the script installs.

Override `PLATFORM_NAMESPACE` or `PLATFORM_VERSION` to change the namespace or chart version.

## Step 2: Download the Model to the GPU Node

```bash
NODE_NAME="$GPU_NODE" common/download-model.sh
```

The script runs a Kubernetes Job on `$GPU_NODE` that downloads `Qwen/Qwen3-4B` at the pinned revision `1cfa9a7208912126459214e8b04321603b3df60c` into a `hostPath` cache on that node, and prints the snapshot path. An init container running as root creates the directory and opens its permissions, because the download container, like the workers, runs as a non-root user. The workers then run fully offline from that cache.

| Variable | Default | Purpose |
|---|---|---|
| `NODE_NAME` | unset | Node to place the cache on. Set it to `$GPU_NODE` |
| `HF_CACHE_DIR` | `/opt/hf-cache` | `hostPath` cache directory on the node |
| `NAMESPACE` | `default` | Namespace for the download Job |
| `HF_TOKEN` | empty | Hugging Face token, if your network requires one |
| `RUNTIME_IMAGE` | `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0` | Image that runs the download |

## Step 3: Deploy the Eight DGDs

[`baseline/gen-dgds.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/baseline/gen-dgds.sh) renders the eight DGDs. With `NODE_NAME` set, it pins every frontend and worker to that node, where the model cache is. Render and apply in one step:

```bash
NODE_NAME="$GPU_NODE" baseline/gen-dgds.sh | kubectl apply -f -
kubectl get dgd -n default      # wait until all eight DGDs are Ready
```

| Variable | Default | Purpose |
|---|---|---|
| `NODE_NAME` | unset | Node to pin the frontends and workers to. Set it to `$GPU_NODE` |
| `NAMESPACE` | `default` | Namespace for the DGDs |
| `HF_CACHE_DIR` | `/opt/hf-cache` | Model cache the pods mount read-only. Use the value from Step 2 |
| `NUM_MODELS` | `8` | Number of DGDs (`qwen3-4b-01` and up) |

The generator also takes an optional model snapshot path as its first argument. The checked-in [`baseline/dgds-8x.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/baseline/dgds-8x.yaml) is the same output without a node pin, for reference.

Each DGD has one frontend and one aggregated vLLM worker named `qwen3-4b-NN`. The worker requests and is limited to `nvidia.com/gpu: "1"`, so the default scheduler gives each model its own GPU and the pod sees the full 40,960 MiB.

### Verify Placement

All 16 pods should run on `$GPU_NODE`:

```bash
kubectl get pods -n default -l experiment.nvidia.com/dgd -o custom-columns=NODE:.spec.nodeName --no-headers | sort | uniq -c
```

Expect one line, `16 <GPU_NODE>`. Each of the eight GPU UUIDs should host exactly one worker:

```bash
for p in $(kubectl get pods -n default -l experiment.nvidia.com/role=worker -o name); do
  kubectl exec -n default "${p#pod/}" -- env | grep NVIDIA_VISIBLE
done | sort | uniq -c
```

Expect eight lines, each with a count of `1`. Then confirm that all eight frontend Services exist:

```bash
kubectl get svc -n default | grep frontend      # 8 services, qwen3-4b-NN-frontend
```

## Step 4: Run the Concurrency Sweep

The frontends' ClusterIPs are reachable only from inside the cluster, so the sweep runs in a load-generator pod, [`common/aiperf-client-pod.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/common/aiperf-client-pod.yaml). The pod requests 16 CPUs, and its node affinity keeps it off `$GPU_NODE`, so load generation does not compete with the workers for host CPU.

Start the pod and copy the scripts into it:

```bash
sed "s/<gpu-node>/$GPU_NODE/" common/aiperf-client-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/aiperf-client --timeout=300s
kubectl cp common aiperf-client:/work/common
kubectl exec aiperf-client -- bash /work/common/setup-aiperf.sh
```

Start the sweep as a detached process inside the pod, so that it keeps running if your `kubectl exec` session drops, then follow its log. Stopping `tail` with Ctrl+C does not stop the sweep:

```bash
kubectl exec aiperf-client -- bash -c \
  'cd /work && setsid bash -c "ENDPOINT_MODE=dns NUM_MODELS=8 RESULTS_DIR=/work/results \
     bash common/run-sweep.sh baseline-8x-sweep > sweep.log 2>&1; echo \$? > sweep.exit" \
   < /dev/null > /dev/null 2>&1 &'
kubectl exec aiperf-client -- tail -f /work/sweep.log
```

`NUM_MODELS` defaults to 16, so you must set it to 8 here. The sweep is finished when `/work/sweep.exit` exists; `0` means every AIPerf run succeeded. Copy the results to your workstation:

```bash
kubectl exec aiperf-client -- cat /work/sweep.exit
kubectl cp aiperf-client:/work/results ./results
```

The published run drove the frontends from the GPU node itself. Running the load generator on another node adds one network hop per request, which is small next to the TTFT values below.

For each concurrency in `1 2 4 8 16 32`, the script sends `max(40, 10 * c)` requests per model after `min(16, 2 * c)` warmup requests. The payload is streaming chat with 2048 input and 256 output tokens, fixed, with seed 42. With `ENDPOINT_MODE=dns`, each AIPerf process targets `http://<NAME_PREFIX>-<NN>-frontend.<NAMESPACE>.svc.cluster.local:8000`.

| Variable | Default | Purpose |
|---|---|---|
| `ENDPOINT_MODE` | `clusterip` | `dns` targets each frontend by its Service DNS name and needs no `kubectl`; use it inside the cluster. `clusterip` looks up ClusterIPs with `kubectl` |
| `NUM_MODELS` | `16` | Number of frontends to drive. Use `8` for the baseline |
| `NAME_PREFIX` | `qwen3-4b` | DGD name prefix. The script targets Service `<NAME_PREFIX>-<NN>-frontend` |
| `NAMESPACE` | `default` | Namespace of the frontend Services |
| `CLUSTER_DOMAIN` | `cluster.local` | Cluster DNS domain, for `dns` mode |
| `RESULTS_DIR` | `./results` | Output root |
| `CONCURRENCIES` | `1 2 4 8 16 32` | Space-separated per-model concurrency points |
| `AIPERF_VENV` | `$HOME/.aiperf-venv` | Virtual environment that `setup-aiperf.sh` creates and the sweep uses. In the pod, `HOME` is `/work` |
| `AIPERF` | `$AIPERF_VENV/bin/aiperf` | Path to the `aiperf` binary |
| `HF_HOME` | `/work/hf-cache` in the pod | Where AIPerf caches the Qwen3-4B tokenizer, which it downloads from Hugging Face |

`setup-aiperf.sh` also reads `AIPERF_VERSION` (default `0.11.0`). Results land in `results/baseline-8x-sweep/c<N>/worker-NN/`, with one log per worker next to each directory. If any AIPerf run fails, the script prints the path of its `worker-NN.log` to `sweep.log` and exits non-zero after the last point.

## Expected Results

The published run on 8x A100 40GB, driver 615.71.09, completed all 5,440 measured requests with zero errors. Aggregate tok/s is the sum of the eight per-model output tokens per second. The latency columns combine the eight per-model p50 values, which agreed to within 3% at every point except TTFT at `c=8` (297 to 333 ms across models).

| Concurrency per model | TTFT p50 (ms) | ITL p50 (ms) | Aggregate tok/s | tok/s per GPU |
|---:|---:|---:|---:|---:|
| 1 | 103.6 | 8.58 | 891 | 111.4 |
| 2 | 48.6 | 8.61 | 1,820 | 227.5 |
| 4 | 49.0 | 9.13 | 3,433 | 429.1 |
| 8 | 331.5 | 11.00 | 5,370 | 671.2 |
| 16 | 431.0 | 15.56 | 7,434 | 929.2 |
| 32 | 481.8 | 26.20 | 9,134 | 1,141.8 |

The `tok/s per GPU` column is the baseline column in [Interpreting GPU Sharing Results](gpu-sharing-results.mdx); each GPU hosts one model, so it equals per-model throughput. Per-worker throughput varied by less than 0.2% at every point. Your absolute numbers will differ; compare ratios against the other two experiments, not raw values.

## Key Configuration

| Setting | Value | Why |
|---|---|---|
| `nvidia.com/gpu` | `"1"` request and limit | One exclusive GPU per worker. This is the only experiment that uses it: KAI rejects pods that carry both a `gpu-fraction` annotation and an `nvidia.com/gpu` resource |
| `--gpu-memory-utilization` | `0.85` | The pod sees the full 40,960 MiB, so vLLM claims about 33.6 GiB (0.85 of 39.5 GiB). Fractional experiments size to about 16 to 17 GB per worker instead |
| `--max-model-len` / `--max-num-seqs` | `4096` / `32` | Same as the other experiments. A cap of 32 sequences keeps the high-concurrency points measuring GPU contention rather than vLLM queueing |
| `DYN_DISCOVERY_BACKEND` | `etcd` on both components | Required with operator 1.4.2 and `vllm-runtime:1.3.0` (see Step 1) |
| `DYN_NAMESPACE_WORKER_SUFFIX` | empty, on the worker | The operator otherwise appends a random suffix to the worker namespace. Pinning it empty makes the worker and frontend namespaces match exactly |
| `runtimeClassName` | `nvidia` | Gives the worker the driver libraries |

## Clean Up

```bash
kubectl delete pod aiperf-client -n default --ignore-not-found
NODE_NAME="$GPU_NODE" baseline/gen-dgds.sh | kubectl delete -f -
kubectl delete job download-qwen3-4b -n default --ignore-not-found
kubectl delete secret download-qwen3-4b-hf-token -n default --ignore-not-found
```

The model cache stays in `/opt/hf-cache` on `$GPU_NODE`; keep it if you plan to run the other experiments on that node, since they use the same cache. To remove the platform, run `helm uninstall dynamo-platform -n dynamo-system`.

## Next Steps

- [Deploy the KAI + HAMi Experiment](deploy-kai-hami.md) packs two models per GPU with memory caps and time-sliced compute.
- [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md), then [deploy the KAI + GPU Fractions experiment](deploy-kai-gpu-fractions.md), which adds a hard 50% compute share per model.
- [Interpreting GPU Sharing Results](gpu-sharing-results.mdx) explains how to compare your sweep with the other two.
