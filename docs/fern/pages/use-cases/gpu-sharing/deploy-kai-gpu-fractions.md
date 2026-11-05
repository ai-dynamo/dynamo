---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Deploy the KAI + GPU Fractions Experiment
subtitle: Run 16 Qwen3-4B models on 8 A100 GPUs, two per GPU, each capped at 50% of the SMs and 50% of the memory
---

This guide deploys 16 Qwen3-4B models, each in its own `DynamoGraphDeployment` (DGD), packed two per A100 40GB GPU. Unlike the [KAI + HAMi experiment](deploy-kai-hami.md), each model gets a hard share of the GPU's compute as well as its memory. The fork-built KAI-Scheduler and kai-gpu-fractioning place each worker in its own CUDA Multi-Process Service (MPS) namespace capped at 50% of the active threads, with a 19,968 MiB memory cap. The two models on a GPU run concurrently instead of taking turns.

The scripts, manifests, and results for this experiment are co-authored by [@marckarp](https://github.com/marckarp) and [@scheckerNV](https://github.com/scheckerNV).

This is the third configuration in [Interpreting GPU Sharing Results](gpu-sharing-results.mdx):

| Experiment | DGDs | Placement | Isolation | Guide |
|---|---:|---|---|---|
| Baseline | 8 | One model per GPU | Dedicated GPU | [Deploy the Baseline Experiment](deploy-baseline.md) |
| KAI + HAMi | 16 | Two models per GPU | Memory cap only; compute is time-sliced | [Deploy the KAI + HAMi Experiment](deploy-kai-hami.md) |
| KAI + GPU fractions (this page) | 16 | Two models per GPU (`gpu-fraction: "0.5"` plus `sm-sharing`) | Memory cap (19,968 MiB) and 50% of the SMs | This page |

All commands run from `examples/gpu-sharing` in a checkout of the [Dynamo repository](https://github.com/ai-dynamo/dynamo). The files this guide uses are in [`examples/gpu-sharing/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing).

> [!NOTE]
> Any performance results on this page are purely illustrative and are not indicative of optimal performance. Your deployment or configuration may vary.

## Requirements

You must build and install the two forks before you deploy any models. That procedure, including the full node requirements, is in [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md) and is not repeated here.

This guide assumes a multi-node Kubernetes cluster that you administer from a workstation with `kubectl` and `helm`. All 16 models run on one node with eight A100 40GB GPUs. The model cache is a `hostPath` directory on that node, so every frontend and worker is pinned to it. The load generator runs in its own pod on a different node and reaches each frontend through its Service DNS name.

| Requirement | Published run | Check | References |
|---|---|---|---|
| One node with 8 NVIDIA A100 40GB GPUs, with no other GPU workloads on it | 8x A100-SXM4 40GB | `kubectl get nodes -L nvidia.com/gpu.product,nvidia.com/gpu.count` | [NVIDIA A100](https://www.nvidia.com/en-us/data-center/a100/), [GPU Feature Discovery labels](https://github.com/NVIDIA/k8s-device-plugin/blob/main/docs/gpu-feature-discovery/README.md) |
| NVIDIA driver r615 or newer on that node, for per-namespace MPS limits | 615.71.09 | `kubectl get node "$GPU_NODE" -o jsonpath='{.metadata.labels.nvidia\.com/cuda\.driver-version\.full}'` prints `615.x` or newer | [Driver Installation Guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) |
| NVIDIA GPU Operator, and NRI enabled in the container runtime | | `kubectl get pods -n gpu-operator` | [Installing the GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html), [NRI Support in containerd](https://github.com/containerd/containerd/blob/main/docs/NRI.md) |
| `nvidia` RuntimeClass | Created by the GPU Operator | `kubectl get runtimeclass nvidia` | [Kubernetes RuntimeClass](https://kubernetes.io/docs/concepts/containers/runtime-class/) |
| KAI-Scheduler and kai-gpu-fractioning, built from the pinned fork commits | [KAI-Scheduler#2368](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) at `44d1d0a`, [gpu-fractioning#147](https://github.com/kai-scheduler/gpu-fractioning/pull/147) at `af0544e` | [Steps 3 and 4 of the build guide](build-gpu-fractioning-forks.md#step-3-install-kai-scheduler) | [KAI-Scheduler pull request](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368), [gpu-fractioning pull request](https://github.com/kai-scheduler/gpu-fractioning/pull/147) |
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

## Step 2: Download the Model to the GPU Node

```bash
NODE_NAME="$GPU_NODE" common/download-model.sh
```

The script runs a Kubernetes Job on `$GPU_NODE` that downloads `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c` into the `hostPath` cache `HF_CACHE_DIR` (default `/opt/hf-cache`) on that node, and prints the snapshot path. An init container running as root creates the directory and opens its permissions, because the download container, like the workers, runs as a non-root user. The script also reads `NAMESPACE` (default `default`), `HF_TOKEN`, and `RUNTIME_IMAGE`.

## Step 3: Install the Forks

Follow [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md) through Step 4. Do not install the upstream KAI-Scheduler or the HAMi isolator on the same cluster: the forks are a matched pair, and the KAI + HAMi stack uses the same `kai-scheduler` release and namespace. If you ran the [KAI + HAMi experiment](deploy-kai-hami.md) on this cluster, uninstall it first.

### Queues

The experiment's workers use the `default-queue` queue, which the KAI-Scheduler chart creates. The KAI + HAMi install script patches both default queues to unlimited quota, but the fork chart creates them with quota `0` and limit `-1`, which admits over-quota work, and the published run's records show no queue patch. If workers stay `Pending` with a queue-quota reason, apply the same patch that [`install-kai-hami.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-hami/install-kai-hami.sh) uses:

```bash
for q in default-parent-queue default-queue; do
  kubectl patch queue "$q" --type=merge -p '{"spec":{"resources":{
    "gpu":{"quota":-1,"limit":-1,"overQuotaWeight":1},
    "cpu":{"quota":-1,"limit":-1,"overQuotaWeight":1},
    "memory":{"quota":-1,"limit":-1,"overQuotaWeight":1}}}}'
done
```

### Verify the Caps With the Smoke Test

[`smoke-test-half-gpu.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-gpu-fractions/smoke-test-half-gpu.yaml) starts two idle pods, `fraction-smoke-a` and `fraction-smoke-b`, each requesting `gpu-fraction: "0.5"`, the same pods as the [build guide smoke test](build-gpu-fractioning-forks.md#step-5-verify-with-a-two-pod-smoke-test). Run it before you commit eight GPUs to the full experiment:

```bash
kubectl apply -f kai-gpu-fractions/smoke-test-half-gpu.yaml
kubectl wait --for=condition=Ready pod/fraction-smoke-a pod/fraction-smoke-b --timeout=180s
for p in fraction-smoke-a fraction-smoke-b; do
  kubectl exec "$p" -- env | grep -E 'CUDA_MPS|NVIDIA_GPU_MEMORY|NVIDIA_VISIBLE_DEVICES'
done
```

Both pods must report the same `NVIDIA_VISIBLE_DEVICES` GPU UUID, `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`, and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=19968M`. If any of the three differs, fix the install before you continue; the [build guide](build-gpu-fractioning-forks.md#troubleshooting) lists the common causes. Delete the pods before you deploy the DGDs:

```bash
kubectl delete -f kai-gpu-fractions/smoke-test-half-gpu.yaml
```

## Step 4: Deploy the 16 DGDs

[`kai-gpu-fractions/gen-dgds.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-gpu-fractions/gen-dgds.sh) renders the 16 DGDs. With `NODE_NAME` set, it pins every frontend and worker to that node, where the model cache is. Render and apply in one step:

```bash
NODE_NAME="$GPU_NODE" kai-gpu-fractions/gen-dgds.sh | kubectl apply -f -
kubectl get dgd -n default      # wait until all 16 DGDs are Ready
```

The generator also reads `NAMESPACE` (default `default`) and `HF_CACHE_DIR` (default `/opt/hf-cache`); set them to the values you used in Step 2. It takes an optional model snapshot path as its first argument. The checked-in [`kai-gpu-fractions/dgds-16x.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/gpu-sharing/kai-gpu-fractions/dgds-16x.yaml) is the same output without a node pin, for reference.

`NUM_MODELS` (default `16`) sets the number of DGDs. Each worker carries the label `kai.scheduler/queue: default-queue`, the annotations `gpu-fraction: "0.5"` and `nvidia.com/container.main.gpu-compute.mode: sm-sharing`, `schedulerName: kai-scheduler`, `runtimeClassName: nvidia`, and a non-root security context. It has no `nvidia.com/gpu` resource.

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

Expect eight lines, each with a count of `2`. Then confirm the caps on a worker, and the MPS server's own view, which is the ground truth for enforcement:

```bash
POD=$(kubectl get pods -n default -l experiment.nvidia.com/role=worker -o name | head -1)
kubectl exec -n default "${POD#pod/}" -- env | grep -E 'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE|CUDA_MPS_PINNED_DEVICE_MEM_LIMIT'
kubectl -n gpu-fractioning exec ds/gpu-fractioning-mpsd -- \
  nvidia-cuda-mps-control -p 3 namespace list --server=shared
```

The worker reports `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=19968M`. The namespace list shows 16 `kai_*` namespaces at `50.00` active thread percentage, plus the `default` namespace at `10.00`.

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
     bash common/run-sweep.sh kai-fractions-16x-sweep > sweep.log 2>&1; echo \$? > sweep.exit" \
   < /dev/null > /dev/null 2>&1 &'
kubectl exec aiperf-client -- tail -f /work/sweep.log
```

The sweep takes several hours. It is finished when `/work/sweep.exit` exists; `0` means every AIPerf run succeeded. Copy the results to your workstation:

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

Results land in `results/kai-fractions-16x-sweep/c<N>/worker-NN/`, with a `worker-NN.log` next to each directory. If any AIPerf run fails, the script prints the path of its `worker-NN.log` to `sweep.log` and exits non-zero after the last point. Re-run the placement and MPS namespace checks after the sweep; the published run still showed two workers per GPU and 16 namespaces at 50% afterward.

## Expected Results

The published run (8x A100 40GB, driver 615.71.09) completed all 10,880 measured requests with zero errors and no restarts. Aggregate tok/s is the sum of the 16 per-model output tokens per second, and the latency columns are the mean of the 16 per-model p50 values.

| Concurrency per model | TTFT p50 (ms) | ITL p50 (ms) | Aggregate tok/s | tok/s per GPU | Share of baseline per GPU |
|---:|---:|---:|---:|---:|---:|
| 1 | 185 | 13.2 | 1,155 | 144.4 | 63.5% |
| 2 | 347 | 14.1 | 2,074 | 259.2 | 60.4% |
| 4 | 549 | 15.4 | 3,649 | 456.1 | 68.0% |
| 8 | 739 | 19.3 | 5,782 | 722.8 | 77.8% |
| 16 | 786 | 28.9 | 8,027 | 1,003.4 | 87.9% |
| 32 | 2,159 | 39.2 | 9,059 | 1,132.4 | not compared |

The last column compares each row with the baseline at the same requests per GPU, as in [Interpreting GPU Sharing Results](gpu-sharing-results.mdx), where this configuration at `c` is matched with the baseline at `2c`. The `c=32` row has no counterpart there, because the baseline would need `c=64`. The same table gives the HAMi comparison: at equal per-GPU load, fractions keeps 15 to 20 points more throughput than time-slicing.

Throughput grows about 7.8x from `c=1` to `c=32`, with only a 13% gain from `c=16` to `c=32`, so the knee is near `c=16`. TTFT is the main cost at saturation: the mean TTFT p90 at `c=32` is about 8.5 s. Throughput varied by less than 0.6% across the 16 models at every point.

## Key Configuration

| Setting | Value | Why |
|---|---|---|
| `gpu-fraction` annotation | `"0.5"`, with no `nvidia.com/gpu` | KAI's admission webhook rejects a pod that carries both a fraction and an `nvidia.com/gpu` resource |
| `gpu-compute.mode: sm-sharing` | On container `main` | Puts the container in its own MPS namespace capped at 50% active threads. The container must be named `main` because the annotation names the container it applies to |
| Memory cap | 19,968 MiB | The binder computes (40,960 - 1,024) MiB x 0.5 on an A100 40GB; the 1 GiB reserve is the MPS server's own context |
| `--gpu-memory-utilization` | `0.40` | MPS does not virtualize the memory total the way HAMi does, so vLLM sizes against the physical 39.5 GiB card. 0.40 is about 15.8 GiB, comparable to the HAMi experiment's 0.85 of about 20 GB. Recompute it for other GPUs |
| `runtimeClassName` and non-root security context | `nvidia`, `runAsNonRoot` | A root container is equivalent to the MPS owner and could raise its own cap. Keep the non-root context |
| `--max-model-len` / `--max-num-seqs` | `4096` / `32` | A cap of 32 sequences keeps the high-concurrency points measuring GPU contention rather than vLLM queueing |
| `DYN_DISCOVERY_BACKEND` | `etcd` on both components | Required with operator 1.4.2 and `vllm-runtime:1.3.0` (see Step 1) |
| `DYN_NAMESPACE_WORKER_SUFFIX` | empty, on the worker | The operator otherwise appends a random suffix to the worker namespace. Pinning it empty makes the worker and frontend namespaces match exactly |

## Clean Up

```bash
kubectl delete pod aiperf-client -n default --ignore-not-found
NODE_NAME="$GPU_NODE" kai-gpu-fractions/gen-dgds.sh | kubectl delete -f -
kubectl delete job download-qwen3-4b -n default --ignore-not-found
kubectl delete secret download-qwen3-4b-hf-token -n default --ignore-not-found
```

To remove the forks, follow the [uninstall steps](build-gpu-fractioning-forks.md#uninstall). If MPS state is left over after repeated teardown, delete the `gpu-fractioning-mpsd` pods to restart the daemons. The model cache stays in `/opt/hf-cache` on `$GPU_NODE`; keep it if you plan to run the other experiments on that node.

## Next Steps

- [Deploy the Baseline Experiment](deploy-baseline.md) and [Deploy the KAI + HAMi Experiment](deploy-kai-hami.md) produce the other two sweeps that the percentages compare against.
- [Per-GPU Compute and Memory Limits with MPS](mps-per-gpu-limits.md) explains the MPS v3 mechanism that the forks drive.
- [Interpreting GPU Sharing Results](gpu-sharing-results.mdx) explains how to read the three sweeps together.
