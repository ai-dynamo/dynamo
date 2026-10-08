<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Many Models per GPU: Reference Manifests and Scripts

This directory holds the manifests and scripts behind the
[GPU sharing results](../../docs/fern/pages/use-cases/gpu-sharing/gpu-sharing-results.mdx)
for many-model serving. It reproduces three experiments on one node with 8x A100 40GB GPUs:

| Experiment | Directory | DGDs | Placement | Isolation |
|---|---|---:|---|---|
| Baseline | [`baseline/`](baseline/) | 8 | One model per GPU (`nvidia.com/gpu: "1"`) | Dedicated GPU |
| KAI + HAMi | [`kai-hami/`](kai-hami/) | 16 | Two models per GPU (`gpu-fraction: "0.5"`) | Memory cap only (HAMi); compute is time-sliced |
| KAI + GPU fractions | [`kai-gpu-fractions/`](kai-gpu-fractions/) | 16 | Two models per GPU (`gpu-fraction: "0.5"` + `sm-sharing`) | Memory cap (19,968 MiB) and 50% MPS active threads |

Scripts, manifests, and results by [@marckarp](https://github.com/marckarp) and
[@scheckerNV](https://github.com/scheckerNV).

## Common Setup

All three experiments share the same workload:

- **Model:** `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c`
- **Runtime:** `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.3.0`
- **Dynamo platform:** Helm chart `dynamo-platform` 1.4.2, with etcd and NATS installed
- **DGD shape:** one frontend and one aggregated vLLM worker per DGD, named
  `qwen3-4b-01`, `qwen3-4b-02`, and so on. Each worker uses `--max-model-len 4096` and
  `--max-num-seqs 32`, with CUDA graphs enabled.
- **Load:** AIPerf 0.11.0, streaming chat, ISL 2048 / OSL 256 (standard deviation 0),
  seed 42. Each model is swept at concurrency 1, 2, 4, 8, 16, and 32, with every model
  driven at the same time. Each point sends `max(40, 10 * c)` requests per model after
  `min(16, 2 * c)` warmup requests.

The published runs used one 8x A100 40GB node with the NVIDIA GPU Operator, with the
load generator running on that node. The baseline and GPU fractions runs used driver
615.71.09. The KAI + HAMi run used driver 595.91.07.

The steps below target a multi-node Kubernetes cluster administered from a workstation.
All models run on one 8-GPU node, which also holds the `hostPath` model cache, and the
load generator runs in a pod on another node.

## Files

| Path | Purpose |
|---|---|
| [`common/download-model.sh`](common/download-model.sh) | Downloads Qwen3-4B at the pinned revision into a hostPath cache (`HF_CACHE_DIR`, default `/opt/hf-cache`) with a Kubernetes Job |
| [`common/install-dynamo-platform.sh`](common/install-dynamo-platform.sh) | Installs `dynamo-platform` 1.4.2 with `global.etcd.install=true` and `global.nats.install=true` |
| [`common/setup-aiperf.sh`](common/setup-aiperf.sh) | Creates a venv with AIPerf 0.11.0 |
| [`common/run-sweep.sh`](common/run-sweep.sh) | Runs the concurrency sweep against `NUM_MODELS` DGD frontends at once |
| [`common/aiperf-client-pod.yaml`](common/aiperf-client-pod.yaml) | In-cluster load-generator pod that runs `run-sweep.sh` with `ENDPOINT_MODE=dns` |
| [`baseline/gen-dgds.sh`](baseline/gen-dgds.sh), [`baseline/dgds-8x.yaml`](baseline/dgds-8x.yaml) | Generator and rendered manifests for the 8-DGD baseline |
| [`kai-hami/install-kai-hami.sh`](kai-hami/install-kai-hami.sh) | Installs KAI-Scheduler v0.17.0 with the hamicore plugin, the HAMi resource isolator 1.1.0-chart, and opens the default queue quotas |
| [`kai-hami/gen-dgds.sh`](kai-hami/gen-dgds.sh), [`kai-hami/dgds-16x.yaml`](kai-hami/dgds-16x.yaml) | Generator and rendered manifests for the 16-DGD KAI + HAMi run |
| [`kai-gpu-fractions/gen-dgds.sh`](kai-gpu-fractions/gen-dgds.sh), [`kai-gpu-fractions/dgds-16x.yaml`](kai-gpu-fractions/dgds-16x.yaml) | Generator and rendered manifests for the 16-DGD GPU fractions run |
| [`kai-gpu-fractions/smoke-test-half-gpu.yaml`](kai-gpu-fractions/smoke-test-half-gpu.yaml) | Two idle pods sharing one GPU at 50% each, to verify the caps before deploying models |
| [`kai-gpu-fractions/microk8s-gpu-fractioning-config.yaml`](kai-gpu-fractions/microk8s-gpu-fractioning-config.yaml) | Reference copy of the MicroK8s `GpuFractioningConfig` that the chart's socket-path flags produce; not meant to be applied |

The checked-in `dgds-*.yaml` files are the generators' default output: namespace
`default`, the model at `/opt/hf-cache/hub/models--Qwen--Qwen3-4B/snapshots/<revision>`,
and no node pin. To deploy, run the generator with `NODE_NAME` set, as below, so that
every pod lands on the node that holds the model cache.

## Run an Experiment

Install the scheduler stack for the experiment first:

- **Baseline:** no extra components.
- **KAI + HAMi:** run `kai-hami/install-kai-hami.sh`.
- **KAI + GPU fractions:** build and install KAI-Scheduler and kai-gpu-fractioning from
  pull requests
  [kai-scheduler/KAI-Scheduler#2368](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) and
  [kai-scheduler/gpu-fractioning#147](https://github.com/kai-scheduler/gpu-fractioning/pull/147),
  as described in
  [Build the KAI-Scheduler and GPU Fractioning Forks](../../docs/fern/pages/use-cases/gpu-sharing/build-gpu-fractioning-forks.md).
  This requires driver r615 or later for per-namespace MPS limits. Then verify the caps
  with `kai-gpu-fractions/smoke-test-half-gpu.yaml`.

Then run the shared steps. The example below is for KAI + HAMi:

```bash
cd examples/gpu-sharing
export GPU_NODE=<node-name>          # the 8-GPU node
common/install-dynamo-platform.sh
NODE_NAME="$GPU_NODE" common/download-model.sh

NODE_NAME="$GPU_NODE" kai-hami/gen-dgds.sh | kubectl apply -f -
kubectl get dgd -n default           # wait until all DGDs are Ready

# Load generator, on another node
sed "s/<gpu-node>/$GPU_NODE/" common/aiperf-client-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/aiperf-client --timeout=300s
kubectl cp common aiperf-client:/work/common
kubectl exec aiperf-client -- bash /work/common/setup-aiperf.sh
kubectl exec aiperf-client -- bash -c \
  'cd /work && setsid bash -c "ENDPOINT_MODE=dns NUM_MODELS=16 RESULTS_DIR=/work/results \
     bash common/run-sweep.sh kai-hami-16x-sweep > sweep.log 2>&1; echo \$? > sweep.exit" \
   < /dev/null > /dev/null 2>&1 &'
kubectl exec aiperf-client -- tail -f /work/sweep.log
kubectl cp aiperf-client:/work/results ./results
```

For the baseline, use `baseline/gen-dgds.sh` and run the sweep with `NUM_MODELS=8`.
For GPU fractions, use `kai-gpu-fractions/gen-dgds.sh` with `NUM_MODELS=16`.
The deployment guides walk through each experiment step by step:
[baseline](../../docs/fern/pages/use-cases/gpu-sharing/deploy-baseline.md),
[KAI + HAMi](../../docs/fern/pages/use-cases/gpu-sharing/deploy-kai-hami.md), and
[KAI + GPU fractions](../../docs/fern/pages/use-cases/gpu-sharing/deploy-kai-gpu-fractions.md).

The sweep is done when `/work/sweep.exit` exists in the pod; `0` means every AIPerf run
succeeded. Results land in `results/<run-label>/c<N>/worker-NN/`, with a
`worker-NN.log` next to each directory, and failing runs are listed in `sweep.log`.

`run-sweep.sh` defaults to `ENDPOINT_MODE=clusterip`, which looks up each Service's
ClusterIP with `kubectl`; use that only from a host that can route to ClusterIPs. The
scripts that call `kubectl` or `helm` read the `KUBECTL` and `HELM` variables if your
cluster needs a different command.

### Verify Placement Before Benchmarking

For the 16-DGD experiments, confirm that each GPU UUID hosts exactly two workers:

```bash
for p in $(kubectl get pods -n default -l experiment.nvidia.com/role=worker -o name); do
  kubectl exec -n default "${p#pod/}" -- env | grep NVIDIA_VISIBLE
done | sort | uniq -c
```

Also confirm the per-worker caps:

- **KAI + HAMi:** `nvidia-smi` inside a worker reports about 20,070 MiB total, and the
  environment has `CUDA_DEVICE_MEMORY_LIMIT=20070m`.
- **KAI + GPU fractions:** the environment has `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`
  and `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=19968M`.

## Known Issues

- **Models never appear on the frontend.** With platform 1.4.x defaults (no etcd or
  NATS, `DYN_DISCOVERY_BACKEND=kubernetes`), the `vllm-runtime:1.3.0` frontend never lists
  the model, so `/v1/models` is empty and chat requests return 404. Install etcd and NATS
  with `common/install-dynamo-platform.sh`. The generators set
  `DYN_DISCOVERY_BACKEND=etcd` on both components and an empty
  `DYN_NAMESPACE_WORKER_SUFFIX` on the worker.
- **Model download fails with `BackoffLimitExceeded`.** The runtime container runs as a
  non-root user, so the hostPath cache must be writable. With `NODE_NAME` set,
  `download-model.sh` opens it with a root init container on that node, so the
  namespace's Pod Security level must allow root and `hostPath` pods.
- **Pods rejected at admission.** KAI rejects pods that carry both a `gpu-fraction`
  annotation and an `nvidia.com/gpu` resource. Fractional workers must not request
  `nvidia.com/gpu`.
- **DGD updates hang on a fully packed node.** The operator starts new worker pods
  before it removes the old ones. When every GPU fraction is allocated, the new pods
  stay Pending. Run `kubectl delete dgd --all` and re-apply instead of editing in place.
- **`--gpu-memory-utilization` differs between experiments on purpose.** HAMi shows the
  pod a virtual ~20 GB GPU, so the KAI + HAMi workers use 0.85. GPU fractions shows the
  physical ~39.5 GiB GPU while enforcing a 19,968 MiB cap, so those workers use 0.40.
  Both settings give each worker about 16 to 17 GB. Recompute both values for other GPUs.

## Related Documentation

- [Interpreting GPU Sharing Results](../../docs/fern/pages/use-cases/gpu-sharing/gpu-sharing-results.mdx)
- [Deploy the Baseline Experiment](../../docs/fern/pages/use-cases/gpu-sharing/deploy-baseline.md)
- [Deploy the KAI + HAMi Experiment](../../docs/fern/pages/use-cases/gpu-sharing/deploy-kai-hami.md)
- [Build the KAI-Scheduler and GPU Fractioning Forks](../../docs/fern/pages/use-cases/gpu-sharing/build-gpu-fractioning-forks.md)
- [Deploy the KAI + GPU Fractions Experiment](../../docs/fern/pages/use-cases/gpu-sharing/deploy-kai-gpu-fractions.md)
