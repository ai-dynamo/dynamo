---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy the KAI + HAMi Experiment
subtitle: Run 16 Qwen3-4B models on 8 A100 GPUs, two per GPU, with KAI-Scheduler and HAMi memory caps
---

This guide deploys 16 Qwen3-4B models, each in its own `DynamoGraphDeployment` (DGD), packed two per A100 40GB GPU. The upstream [KAI-Scheduler](https://github.com/kai-scheduler/KAI-Scheduler) places each worker on a half-GPU fraction, and the HAMi resource isolator enforces a per-pod memory cap. HAMi limits memory only: the two models on a GPU share all of its streaming multiprocessors (SMs), and the CUDA driver time-slices between them.

The scripts, manifests, and results for this experiment are co-authored by [@marckarp](https://github.com/marckarp) and [@scheckerNV](https://github.com/scheckerNV).

This is the middle configuration in [Interpreting GPU Sharing Results](gpu-sharing-results.mdx). It uses stock upstream components, so you can run it without building anything:

| Experiment | DGDs | Placement | Isolation | Guide |
|---|---:|---|---|---|
| Baseline | 8 | One model per GPU | Dedicated GPU | [Deploy the Baseline Experiment](deploy-baseline.md) |
| KAI + HAMi (this page) | 16 | Two models per GPU (`gpu-fraction: "0.5"`) | Memory cap only; compute is time-sliced | This page |
| KAI + GPU fractions | 16 | Two models per GPU | Memory cap and 50% of the SMs | [Deploy the KAI + GPU Fractions Experiment](deploy-kai-gpu-fractions.md) |

All commands run from `examples/gpu-sharing` in a checkout of the [Dynamo repository](https://github.com/ai-dynamo/dynamo). The files this guide uses are in [`examples/gpu-sharing/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing).

> [!NOTE]
> Any performance results on this page are purely illustrative and are not indicative of optimal performance. Your deployment or configuration may vary.

## Requirements

This guide assumes a multi-node Kubernetes cluster that you administer from a workstation with `kubectl` and `helm`. All 16 models run on one node with eight A100 40GB GPUs. The model cache is a `hostPath` directory on that node, so every frontend and worker is pinned to it. The load generator runs in its own pod on a different node and reaches each frontend through its Service DNS name.

| Requirement | Published run | Check | References |
|---|---|---|---|
| One node with 8 NVIDIA A100 40GB GPUs, with no other GPU workloads on it | 8x A100-SXM4 40GB | `kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count` | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/), [GPU Feature Discovery labels](https://github.com/NVIDIA/k8s-device-plugin/blob/main/docs/gpu-feature-discovery/README.md) |
| NVIDIA driver on that node | 595.91.07 | `kubectl get node "$GPU_NODE" -o jsonpath='{.metadata.labels.nvidia\.com/cuda\.driver-version\.full}'` | [Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) |
| NVIDIA GPU Operator | v25.10.1 | `kubectl get pods -n gpu-operator` | [Installing the GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html) |
| `nvidia` RuntimeClass | Created by the GPU Operator | `kubectl get runtimeclass nvidia` | [Kubernetes RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/) |
| `kubectl` and `helm` on your workstation, with cluster-admin (the KAI and HAMi charts install from OCI registries) | | `kubectl auth can-i '*' '*' --all-namespaces` | [Install kubectl](https://kubernetes.io/docs/tasks/tools/), [Installing Helm](https://helm.sh/docs/intro/install/) |
| A second node with 16 free CPUs and 32 GiB of memory, for the load generator | AIPerf 0.11.0 | `kubectl describe nodes` (allocatable minus allocated) | [AIPerf](https://github.com/ai-dynamo/aiperf) |
| Pods in the `default` namespace may run as root and mount `hostPath` volumes | | `kubectl get ns default --show-labels` shows no `pod-security.kubernetes.io/enforce` label stricter than `privileged` | [Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/), [Kubernetes `hostPath` volumes](https://kubernetes.io/docs/concepts/storage/volumes/#hostpath) |
| Egress to Hugging Face, `nvcr.io`, `ghcr.io`, Docker Hub, and PyPI | | | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B), [Dynamo vLLM runtime on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/containers/vllm-runtime), [KAI-Scheduler](https://github.com/kai-scheduler/KAI-Scheduler), [HAMi `kai-resource-isolator`](https://github.com/Project-HAMi/kai-resource-isolator), [aiperf 0.11.0 on PyPI](https://pypi.org/project/aiperf/0.11.0/) |

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

## Step 2: Download the Model to the GPU Node

```bash
NODE_NAME="$GPU_NODE" common/download-model.sh
```

The script runs a Kubernetes Job on `$GPU_NODE` that downloads `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` into the `hostPath` cache `HF_CACHE_DIR` (default `/opt/hf-cache`) on that node, and prints the snapshot path. An init container running as root creates the directory and opens its permissions, because the download container, like the workers, runs as a non-root user. The script also reads `NAMESPACE` (default `default`), `HF_TOKEN`, and `RUNTIME_IMAGE`.

## Step 3: Install KAI-Scheduler and HAMi

```bash
kai-hami/install-kai-hami.sh
```

[`install-kai-hami.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-hami/install-kai-hami.sh) runs three stages and waits for the pods of each to be Ready:

| Stage | What it installs | Notes |
|---|---|---|
| KAI-Scheduler v0.17.0 | Helm release `kai-scheduler` in namespace `kai-scheduler` | `global.gpuSharing=true` turns on fractional GPU requests; `binder.plugins.hamicore.enabled=true` hands memory enforcement to HAMi |
| HAMi resource isolator 1.1.0-chart | Helm release `kai-resource-isolator` in namespace `kai-resource-isolator` | `monitor.enabled=true` and `monitor.runtimeClassName=nvidia` run the monitor with the `nvidia` RuntimeClass |
| Default queue quotas | Patches `default-parent-queue` and `default-queue` | Sets quota and limit to `-1` (unlimited) for GPU, CPU, and memory, so the 16 workers are not held back by queue accounting |

The script waits up to 30 attempts at 2 seconds for each queue to exist before it patches it. It ends with `KAI + HAMi stack ready`.

## Step 4: Deploy the 16 DGDs

[`kai-hami/gen-dgds.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-hami/gen-dgds.sh) renders the 16 DGDs. With `NODE_NAME` set, it pins every frontend and worker to that node, where the model cache is. Render and apply in one step:

```bash
NODE_NAME="$GPU_NODE" kai-hami/gen-dgds.sh | kubectl apply -f -
kubectl get dgd -n default      # wait until all 16 DGDs are Ready
```

The generator also reads `NAMESPACE` (default `default`) and `HF_CACHE_DIR` (default `/opt/hf-cache`); set them to the values you used in Step 2. It takes an optional model snapshot path as its first argument. The checked-in [`kai-hami/dgds-16x.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-hami/dgds-16x.yaml) is the same output without a node pin, for reference.

`NUM_MODELS` (default `16`) sets the number of DGDs. Each worker carries the label `kai.scheduler/queue: default-queue`, the annotation `gpu-fraction: "0.5"`, and `schedulerName: kai-scheduler`. It has no `nvidia.com/gpu` resource.

> [!WARNING]
> Do not edit a running DGD to change the worker spec. The operator starts the new worker pods before it removes the old ones, and with every GPU fraction allocated the new pods stay `Pending` indefinitely; deleting the old pods by hand does not help because the operator recreates them. Run `kubectl delete dgd --all -n default` and apply again.

### Verify Placement and Caps

All 32 pods should run on `$GPU_NODE`:

```bash
kubectl get pods -n default -l experiment.nvidia.com/dgd -o custom-columns=NODE:.spec.nodeName --no-headers | sort | uniq -c
```

Expect one line, `32 <GPU_NODE>`. Each GPU UUID should host exactly two workers:

```bash
for p in $(kubectl get pods -n default -l experiment.nvidia.com/role=worker -o name); do
  kubectl exec -n default "${p#pod/}" -- env | grep NVIDIA_VISIBLE
done | sort | uniq -c
```

Expect eight lines, each with a count of `2`. Then confirm that HAMi applied the memory cap to a worker:

```bash
POD=$(kubectl get pods -n default -l experiment.nvidia.com/role=worker -o name | head -1)
kubectl exec -n default "${POD#pod/}" -- nvidia-smi --query-gpu=memory.total --format=csv,noheader
kubectl exec -n default "${POD#pod/}" -- env | grep CUDA_DEVICE_MEMORY_LIMIT
```

`nvidia-smi` reports about 20,070 MiB, not 40,960, and the environment has `CUDA_DEVICE_MEMORY_LIMIT=20070m`. A `Fail to create synthesized container dir` warning from HAMi is benign.

## Step 5: Run the Concurrency Sweep

The frontends' ClusterIPs are reachable only from inside the cluster, so the sweep runs in a load-generator pod, [`common/aiperf-client-pod.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/common/aiperf-client-pod.yaml). The pod requests 16 CPUs, one per concurrent AIPerf process, and its node affinity keeps it off `$GPU_NODE`, so load generation does not compete with the workers for host CPU.

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
  'cd /work && setsid bash -c "ENDPOINT_MODE=dns NUM_MODELS=16 RESULTS_DIR=/work/results \
     bash common/run-sweep.sh kai-hami-16x-sweep > sweep.log 2>&1; echo \$? > sweep.exit" \
   < /dev/null > /dev/null 2>&1 &'
kubectl exec aiperf-client -- tail -f /work/sweep.log
```

The sweep is finished when `/work/sweep.exit` exists; `0` means every AIPerf run succeeded. Copy the results to your workstation:

```bash
kubectl exec aiperf-client -- cat /work/sweep.exit
kubectl cp aiperf-client:/work/results ./results
```

The published run drove the frontends from the GPU node itself. Running the load generator on another node adds one network hop per request, which is small next to the TTFT values below.

For each concurrency in `1 2 4 8 16 32`, the script sends `max(40, 10 * c)` requests per model after `min(16, 2 * c)` warmup requests. It uses streaming chat with 2048 input and 256 output tokens and seed 42. With `ENDPOINT_MODE=dns`, each AIPerf process targets `http://<NAME_PREFIX>-<NN>-frontend.<NAMESPACE>.svc.cluster.local:8000`.

| Variable | Default | Purpose |
|---|---|---|
| `ENDPOINT_MODE` | `clusterip` | `dns` targets each frontend by its Service DNS name and needs no `kubectl`; use it inside the cluster. `clusterip` looks up ClusterIPs with `kubectl` |
| `NUM_MODELS` | `16` | Number of frontends to drive |
| `NAME_PREFIX` | `qwen3-4b` | DGD name prefix. The script targets Service `<NAME_PREFIX>-<NN>-frontend` |
| `NAMESPACE` | `default` | Namespace of the frontend Services |
| `CLUSTER_DOMAIN` | `cluster.local` | Cluster DNS domain, for `dns` mode |
| `RESULTS_DIR` | `./results` | Output root |
| `CONCURRENCIES` | `1 2 4 8 16 32` | Space-separated per-model concurrency points |
| `AIPERF_VENV` | `$HOME/.aiperf-venv` | Virtual environment that `setup-aiperf.sh` creates and the sweep uses. In the pod, `HOME` is `/work` |
| `AIPERF` | `$AIPERF_VENV/bin/aiperf` | Path to the `aiperf` binary |
| `HF_HOME` | `/work/hf-cache` in the pod | Where AIPerf caches the Qwen3-4B tokenizer, which it downloads from Hugging Face |

Results land in `results/kai-hami-16x-sweep/c<N>/worker-NN/`, with a `worker-NN.log` next to each directory. If any AIPerf run fails, the script prints the path of its `worker-NN.log` to `sweep.log` and exits non-zero after the last point.

## Expected Results

The published run (8x A100 40GB, driver 595.91.07) used the same payload and request counts as above. Aggregate tok/s is the sum of the 16 per-model output tokens per second, and the latency columns are the mean of the 16 per-model p50 values.

| Concurrency per model | TTFT p50 (ms) | ITL p50 (ms) | Aggregate tok/s | tok/s per GPU |
|---:|---:|---:|---:|---:|
| 1 | 216 | 19.0 | 809 | 101.1 |
| 2 | 374 | 19.3 | 1,544 | 193.0 |
| 4 | 571 | 21.1 | 2,751 | 343.9 |
| 8 | 731 | 25.2 | 4,576 | 572.0 |
| 16 | 777 | 35.6 | 6,651 | 831.4 |
| 32 | 1,702 | 55.9 | 7,544 | 943.0 |

The first five rows of `tok/s per GPU` are the HAMi column in [Interpreting GPU Sharing Results](gpu-sharing-results.mdx), which matches rows on requests resident per GPU (this configuration at `c` equals the baseline at `2c`). The `c=32` row has no counterpart there, because the baseline would need `c=64`.

Throughput scales about 9.3x from `c=1` to `c=32`, but the gain from `c=16` to `c=32` is only 13%: the shared GPUs saturate near `c=16`. At `c=32` the TTFT p90 reaches about 5.5 s as prefills queue behind full batches. Per-model results were uniform to within a few percent at every point, because the two models on a GPU are identical.

## Key Configuration

| Setting | Value | Why |
|---|---|---|
| `gpu-fraction` annotation | `"0.5"`, with no `nvidia.com/gpu` | KAI's admission webhook rejects a pod that carries both a fraction and an `nvidia.com/gpu` resource |
| `--gpu-memory-utilization` | `0.85` | HAMi shows the pod a virtual GPU of about 20 GB. 0.85 of 20 GB is about 17 GB. The 0.40 that suits a full 40 GB card would give 8 GB, less than the model weights, and the worker would crash |
| `--max-model-len` / `--max-num-seqs` | `4096` / `32` | A cap of 32 sequences keeps the high-concurrency points measuring GPU contention rather than vLLM queueing |
| `DYN_DISCOVERY_BACKEND` | `etcd` on both components | Required with operator 1.4.2 and `vllm-runtime:1.3.0` (see Step 1) |
| `DYN_NAMESPACE_WORKER_SUFFIX` | empty, on the worker | The operator otherwise appends a random suffix to the worker namespace. Pinning it empty makes the worker and frontend namespaces match exactly |
| `nvidia.com/container.main.gpu-compute.mode: sm-sharing` | Present but inert | It belongs to the GPU fractions stack. HAMi ignores it. It stays so the manifest matches the one that produced the published results |
| Compute isolation | None | The CUDA driver time-slices the two workers on each GPU, so one model's kernels can wait behind the other's. This is the effect the GPU fractions experiment removes |

## Clean Up

```bash
kubectl delete pod aiperf-client -n default --ignore-not-found
NODE_NAME="$GPU_NODE" kai-hami/gen-dgds.sh | kubectl delete -f -
kubectl delete job download-qwen3-4b -n default --ignore-not-found
kubectl delete secret download-qwen3-4b-hf-token -n default --ignore-not-found
helm uninstall kai-resource-isolator -n kai-resource-isolator
helm uninstall kai-scheduler -n kai-scheduler
```

The model cache stays in `/opt/hf-cache` on `$GPU_NODE`; keep it if you plan to run the other experiments on that node, since they use the same cache. Uninstall the KAI-Scheduler before you install the GPU fractions forks, because both use the release name `kai-scheduler` in the `kai-scheduler` namespace and the two builds do not mix.

## Next Steps

- [Deploy the Baseline Experiment](deploy-baseline.md) produces the dedicated-GPU numbers that the percentages are measured against.
- [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md), then [deploy the KAI + GPU Fractions experiment](deploy-kai-gpu-fractions.md), to give each model a hard 50% compute share.
- [Interpreting GPU Sharing Results](gpu-sharing-results.mdx) explains how to compare the three sweeps.
