---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reproduce Multi-GPU GPU Fractions
subtitle: Verify TP2 inference, per-GPU limits, and automatic MPS partition assignment on two A100 GPUs
---

Run one NVIDIA Dynamo worker with tensor parallelism (TP) across two GPUs, verify its limits, and then add a second worker. An optional CUDA Multi-Process Service (MPS) extension gives the two workers unequal compute shares: **36/72 streaming multiprocessors (SMs)** for worker A and **72/36 SMs** for worker B. Each GPU has 108 SMs, so 36 + 72 = 108. The two workers each receive 19 GiB per GPU, totaling 38 GiB on each 40 GiB GPU.

This is a bounded functionality and enforcement test. It sends one inference request per frontend; it does not run a benchmark sweep.

> [!NOTE]
> Any performance results on this page are purely illustrative and are not indicative of optimal performance. Your deployment or configuration may vary.

> [!WARNING]
> **Experimental.** The scheduler and GPU Fractioning namespace implementation require the forks below. Unequal compute additionally requires the per-device extension and an MPS build implementing the CUDA 13.5/r620 per-device namespace API. This is not available from the upstream GPU Fractioning release used by the original eight-GPU experiment. A driver branch number alone is not a capability check.

## Prerequisites and Tested Versions

Use a dedicated two-GPU test node with no unrelated GPU clients. Keep existing CPU workloads and cluster resources. This procedure assumes the matched KAI-Scheduler/GPU Fractioning stack and Dynamo platform are installed; it is not a driver installer. Follow [Build the KAI-Scheduler and GPU Fractioning Forks](build-gpu-fractioning-forks.md) for the base installation, including the MicroK8s socket settings. Inventory the node before any daemon or compute-mode change, and obtain the node owner's approval for driver changes or reboots.

| Component | Tested combination |
|---|---|
| GPUs | 2 × A100-SXM4-40GB, 108 SMs each, NV12 connectivity |
| Driver | 615.71.09 for the initial equal-share TP2 test; 620.69 for automatic per-device compute assignment |
| KAI-Scheduler | Fork commit `44d1d0a966465c69b1fd0d24b40b9a1ebb73699e`, [PR #2368](https://github.com/kai-scheduler/KAI-Scheduler/pull/2368) |
| GPU Fractioning base | Fork commit `af0544e4dbd1d8e03ca14fc1e06d6265fabed99b`, [PR #147](https://github.com/kai-scheduler/gpu-fractioning/pull/147) |
| Per-device extension | [Publication commit `ea8c2e6`](https://github.com/scheckerNV/kai-gpu-fractioning/tree/ea8c2e617d423ae7378085f802dfa05b7d2cd9b1); same source tree as hardware-tested commit `284d10a`, with DCO trailers added |
| Platform | Dynamo operator 1.4.2 with etcd and NATS; MicroK8s 1.35.6; containerd 2.1.6 |
| GPU integration | GPU Operator 26.7.1, host-managed driver, Container Toolkit 1.20.1, device plugin 0.20.1 |
| Model | `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Runtime | `nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:5913d0f4002c9f90f61ea9872b37719828851511af5a3fabb0ce56d0d0805f72` |
| Engine | vLLM 0.23.0, PyTorch 2.11.0+cu130, NCCL 2.28.9; TP=2, maximum model length 4096, maximum sequences 32, CUDA graphs enabled |

The runtime's CUDA toolkit remains 13.0. The tested driver supplies the newer CUDA driver and MPS capability; installing a CUDA 13.5 framework runtime was not required.

The [TP2 example files](https://github.com/ai-dynamo/dynamo/tree/main/examples/gpu-sharing/tp2) render portable manifests and provide bounded probes. They replace the original experiment's node names, cache paths, and test gates. The recorded hardware results below come from the original experiment; these documentation helpers are not a new hardware run.

Run the following commands in Bash on the GPU node, from `examples/gpu-sharing`. Standard `kubectl` must be configured for the cluster. On MicroK8s, export `KUBECTL="microk8s kubectl"`; both the shell blocks and the selector helper honor it.

```bash
set -euo pipefail
cd examples/gpu-sharing
export NAMESPACE=tp2-gpu-fractions-example
export NODE_NAME='replace-with-your-two-GPU-node-name'
export HF_CACHE_DIR=/opt/hf-cache
export PLATFORM_NAMESPACE=dynamo-system
export RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:5913d0f4002c9f90f61ea9872b37719828851511af5a3fabb0ce56d0d0805f72
read -r -a K <<< "${KUBECTL:-kubectl}"
mkdir -p results

"${K[@]}" get pods -A -o json > results/pods-before.json
"${K[@]}" get gpufractioningconfig default -o yaml > results/fractioning-before.yaml
"${K[@]}" get nodes -o yaml > results/nodes-before.yaml
nvidia-smi -q > results/gpus-before.txt
nvidia-smi topo -m | tee results/topology.txt
nvidia-smi --query-gpu=index,uuid,name,memory.total,compute_mode,driver_version --format=csv | tee results/hardware.csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name --format=csv | tee results/clients-before.csv
```

Confirm the actual GPU count, memory, and interconnect. Export `GPU0_UUID` and `GPU1_UUID` from the hardware output as full `GPU-...` strings. Keep those physical UUIDs throughout; a CUDA ordinal inside a container can have a different order.

```bash
export GPU0_UUID=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader)
export GPU1_UUID=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)
```

For the unequal-compute run, begin with a fresh MPS server after `EXCLUSIVE_PROCESS` is enabled on both GPUs. Coordinate this while the test GPUs have no clients. Do not restart a shared daemon underneath unrelated GPU work. The earlier r615 experiments did not establish startup order as the root cause of a historical bypass; this is the tested starting condition, not a claim that compute mode alone fixes isolation.

Create the new workload namespace, and prepare the cache using the existing [platform and model-download steps](deploy-kai-gpu-fractions.md#step-1-install-the-dynamo-platform). Reuse an existing platform instead of reinstalling it. If its namespace differs, set `PLATFORM_NAMESPACE` accordingly.

```bash
"${K[@]}" create namespace "$NAMESPACE"
NODE_NAME="$NODE_NAME" common/download-model.sh
```

The node-local cache must be readable by UID 1000. The download Job needs permission for its root init container and `hostPath` volume.

## 1. Request Two Fractional GPUs for One Worker

The fork's supported request is `gpu-fraction-num-devices: "2"` plus container-specific scalar compute and memory requests. The renderer emits:

```yaml
annotations:
  gpu-fraction-num-devices: "2"
  nvidia.com/container.main.gpu-compute.mode: sm-sharing
  nvidia.com/container.main.gpu-compute.request: "0.5"
  nvidia.com/container.main.gpu-memory.request: 20480Mi
  nvidia.com/container.main.gpu-memory.limit: 20480Mi
```

These scalar requests apply to **each** assigned GPU. Do not also request the integer `nvidia.com/gpu` resource or invent a per-GPU percentage annotation. Keep the container named `main`.

First render an idle worker so the probes can run before model allocation:

```bash
python3 tp2/render.py dgds --workers 1 --memory-mib 20480 --idle > results/one-idle.json
"${K[@]}" apply --dry-run=server -f results/one-idle.json
"${K[@]}" apply -f results/one-idle.json
"${K[@]}" -n "$NAMESPACE" wait --for=jsonpath='{.status.phase}'=Running pod \
  -l experiment.nvidia.com/dgd=tp2-a,experiment.nvidia.com/role=worker --timeout=300s
POD_A=$("${K[@]}" -n "$NAMESPACE" get pods -l experiment.nvidia.com/dgd=tp2-a,experiment.nvidia.com/role=worker -o jsonpath='{.items[0].metadata.name}')
"${K[@]}" -n "$NAMESPACE" get pod "$POD_A" -o yaml > results/one-assigned.yaml
```

The worker runs non-root, uses `kai-scheduler`, and has no model loaded. Running an idle container does not mean its DGD can serve inference yet.

## 2. Verify Both GPUs, Memory Sizing, and Communication

Copy the communication probe to the idle worker. It checks two distinct UUIDs, the expected SM count on each, CUDA peer access, vLLM's actual `MemorySnapshot`, and a two-rank NVIDIA Collective Communications Library (NCCL) all-reduce. Both ranks must report the sum **1 + 2 = 3**.

```bash
"${K[@]}" -n "$NAMESPACE" cp tp2/communication_probe.py "$POD_A:/tmp/communication_probe.py" -c main
"${K[@]}" -n "$NAMESPACE" exec "$POD_A" -c main -- env \
  TP2_EXPECTED_MAP="{\"$GPU0_UUID\":54,\"$GPU1_UUID\":54}" TP2_MEMORY_MIB=20480 \
  python3 /tmp/communication_probe.py | tee results/one-communication.txt
```

Inspect the printed `request_memory` source and snapshots. In the tested stack, vLLM sees the **cgroup hard limit as total memory**: 20 GiB for this single-worker configuration, rather than the physical 40 GiB. Thus `0.8 × 20 = 16 GiB`, leaving 4 GiB below the cap. With two workers at 19 GiB each, `0.8 × 19 = 15.2 GiB`, leaving 3.8 GiB. The probe fails if the total differs or free quota is insufficient. Do not reuse `0.8` when your stack reports a different total; the earlier TP1 experiment used `0.40` against a physical-memory view.

Read actual MPS settings, and identify this container's namespace in the host-backed lease file:

```bash
"${K[@]}" -n gpu-fractioning exec ds/gpu-fractioning-mpsd -- \
  nvidia-cuda-mps-control -p 3 namespace list --server=shared | tee results/namespaces.txt
"${K[@]}" -n gpu-fractioning exec ds/gpu-fractioning-mpsd -- \
  cat /run/nvidia-mps/.gpu-fractioning-namespaces.json > results/leases.json
```

Match the pod's `main` container ID to its lease. On MicroK8s, resolve the container's PID and leaf cgroup with:

```bash
CID=$("${K[@]}" -n "$NAMESPACE" get pod "$POD_A" -o jsonpath='{.status.containerStatuses[?(@.name=="main")].containerID}')
CID=${CID#containerd://}
PID=$(microk8s ctr tasks ls | awk -v id="$CID" '$1 == id {print $2}')
test -n "$PID"
CGROUP="/sys/fs/cgroup$(awk -F: '$1 == "0" {print $3}' "/proc/$PID/cgroup")"
test "$CGROUP" != /sys/fs/cgroup
test -d "$CGROUP"
```

On another container runtime, obtain that same container's PID with its configured runtime inspection tool. Use its `/proc/<PID>/cgroup` entry to identify the leaf cgroup. Read both driver limits:

```bash
nvidia-smi memory-limits --get --namespace "$CGROUP" -i "$GPU0_UUID"
nvidia-smi memory-limits --get --namespace "$CGROUP" -i "$GPU1_UUID"
```

Both hard limits must be 20,480 MiB. Environment variables or an NVML memory display alone are not proof of enforcement.

Test direct allocation rejection on each idle GPU with client memory hints removed. Measure host free memory immediately before each attempt. The probe requires at least 1 GiB more physical free memory than the attempted allocation, so physical exhaustion cannot explain rejection. CUDA code `2` means out of memory.

```bash
for GPU in "$GPU0_UUID" "$GPU1_UUID"; do
  FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  "${K[@]}" -n "$NAMESPACE" exec -i "$POD_A" -c main -- python3 - \
    --uuid "$GPU" --mib 64 --expected-code 0 --host-free-mib "$FREE" \
    --expected-sm 54 --selection dynamic --hostile --narrow \
    < tp2/allocation_probe.py > "results/one-small-control-$GPU.json"
  FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
  "${K[@]}" -n "$NAMESPACE" exec -i "$POD_A" -c main -- python3 - \
    --uuid "$GPU" --mib 24576 --expected-code 2 --host-free-mib "$FREE" \
    --expected-sm 54 --selection dynamic --hostile --narrow \
    < tp2/allocation_probe.py > "results/one-above-cap-$GPU.json"
done
```

An inconclusive headroom check is a blocker, not a pass. A successful above-cap allocation is a failure; do not proceed to model load.

## 3. Load One Model and Send One Request

Stop the idle worker before replacing its pod template. The example's worker is component index 1; the frontend is index 0.

```bash
"${K[@]}" -n "$NAMESPACE" patch dgd tp2-a --type=json \
  -p='[{"op":"replace","path":"/spec/components/1/replicas","value":0}]'
"${K[@]}" -n "$NAMESPACE" wait --for=delete pod \
  -l experiment.nvidia.com/dgd=tp2-a,experiment.nvidia.com/role=worker --timeout=300s
python3 tp2/render.py dgds --workers 1 --memory-mib 20480 > results/one-model.json
"${K[@]}" apply -f results/one-model.json
"${K[@]}" -n "$NAMESPACE" get dgd tp2-a
```

Wait for the DGD to become Ready. Save the new worker's logs and confirm TP=2, initialization of both ranks, CUDA graph capture, and no restarts. Rediscover `POD_A` because the pod changed. Read its new cgroup's limits again; the idle pod's limits do not prove the model pod's limits.

In one terminal, find the frontend Service and forward port 8000. In another, send one bounded request:

```bash
"${K[@]}" -n "$NAMESPACE" get svc
export SERVICE_A='tp2-a-frontend'
"${K[@]}" -n "$NAMESPACE" port-forward "svc/$SERVICE_A" 8000:8000
```

```bash
curl --fail-with-body -sS --max-time 180 http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-4B","messages":[{"role":"user","content":"Reply with a short greeting. /no_think"}],"max_tokens":32,"stream":false}' \
  | tee results/one-inference.json
```

Require HTTP success and nonempty generated content before adding the second worker.

## 4. Add the Second TP2 Worker

After the single-worker checks pass, stop worker A as above and wait for its pod to disappear. Render two idle workers at 19 GiB each:

```bash
python3 tp2/render.py dgds --workers 2 --idle > results/two-idle.json
"${K[@]}" apply -f results/two-idle.json
```

Repeat the placement, communication, driver-limit, and above-cap checks for **both** workers. Expect the same pair of physical UUIDs, 54 SMs and 19,456 MiB per GPU for each worker. Set `TP2_MEMORY_MIB=19456` for the communication probe. While both are idle, a 20,480 MiB attempt exceeds a 19 GiB cap; run a small successful allocation control too, using `--mib 64 --expected-code 0`. Save each worker/GPU result separately.

Stop both idle workers, wait for their pods to disappear, render `dgds --workers 2` without `--idle`, and apply. Check both DGDs, graph capture, actual caps, and restarts. Forward the second frontend on local port 8001 and issue one request to each frontend concurrently. To verify overlapping generated output, use streaming requests and record token timestamps; concurrent request start times alone do not establish overlap. This is a functionality check, not a throughput result.

## 5. Apply Unequal Compute Automatically

Keep memory at **19/19 GiB per worker**, so compute is the only allocation change. Stop both workers and verify that no GPU clients remain before changing Fractioning daemons or compute mode. Preserve both frontends.

Check out the extension in a separate source directory:

```bash
git clone https://github.com/scheckerNV/kai-gpu-fractioning.git kai-gpu-fractioning-r620
git -C kai-gpu-fractioning-r620 checkout ea8c2e617d423ae7378085f802dfa05b7d2cd9b1
export FRACTIONING_DIR="$PWD/kai-gpu-fractioning-r620"
```

Build the Fractioning images from this directory using [the build guide](build-gpu-fractioning-forks.md#step-2-build-and-push-the-images), with a new tag identifying this commit. Deploy its matching operator, fractiond, and mpsd images; KAI stays at `44d1d0a`. The tested run built images locally and imported them into MicroK8s containerd, without publishing images. Use that route when a registry is unavailable; if using a registry, confirm reachability from both build machine and node and never overwrite a tag. These docs do not provide prebuilt extension images or a driver download.

Apply the generated CRD explicitly: the prototype chart's embedded CRD does not yet contain the optional field. A later Helm upgrade can replace that schema, so verify it again after any upgrade.

```bash
"${K[@]}" apply --dry-run=server -f "$FRACTIONING_DIR/operator/config/crd/bases/gpu-fractioning.kai.scheduler_gpufractioningconfigs.yaml"
"${K[@]}" apply -f "$FRACTIONING_DIR/operator/config/crd/bases/gpu-fractioning.kai.scheduler_gpufractioningconfigs.yaml"
"${K[@]}" explain gpufractioningconfig.spec.mpsDaemon.smPartitionPolicy
python3 tp2/render.py policy > results/static-policy.json
"${K[@]}" apply -f results/static-policy.json
"${K[@]}" patch gpufractioningconfig default --type=merge \
  -p "{\"spec\":{\"mpsDaemon\":{\"smPartitionPolicy\":{\"name\":\"$NAMESPACE-static-v1\",\"key\":\"policy.json\"}}}}"
```

Persist that same `mpsDaemon.smPartitionPolicy` reference in the Fractioning Helm values for subsequent upgrades. The immutable ConfigMap is in **`gpu-fractioning`**, the operator namespace. Its rules match the workload namespace and stable worker labels and assign these fixed profiles:

| Worker | Physical GPU 0 | Physical GPU 1 | Memory per GPU |
|---|---:|---:|---:|
| A | 18 chunks = 36 SMs | 36 chunks = 72 SMs | 19 GiB |
| B | 36 chunks = 72 SMs | 18 chunks = 36 SMs | 19 GiB |

Chunk size is device-specific; these counts apply to the tested A100. The policy uses full GPU UUIDs. This extension does not teach KAI to schedule unequal shares: scheduler requests remain scalar 50% compute and 19 GiB memory per GPU. The administrator owns the fixed compute profiles, and MPS enforces them.

Wait until the operator has reconciled the **new image and policy** into the expected DaemonSet generations, then wait for those new pods to become Ready. An old Ready generation is insufficient. In the mpsd pod, check that `namespace set --help` includes `--device` and `--sm-partition`, and that all four static partitions have status `created`, not merely `configured`:

```bash
"${K[@]}" -n gpu-fractioning exec ds/gpu-fractioning-mpsd -- \
  nvidia-cuda-mps-control -p 3 namespace set --help
"${K[@]}" -n gpu-fractioning exec ds/gpu-fractioning-mpsd -- \
  nvidia-cuda-mps-control -p 3 sm-partition list --server=shared --all --format=csv \
  | tee results/partitions.csv
```

Apply `two-idle.json` again. Repeat both communication probes using expected maps **36/72** for A and **72/36** for B, matched by UUID. Read the fresh leases and each namespace's actual per-device map:

Set `MPS_NAMESPACE` to the namespace in the lease matching the current worker's container ID.

```bash
"${K[@]}" -n gpu-fractioning exec ds/gpu-fractioning-mpsd -- \
  nvidia-cuda-mps-control -p 3 namespace get "$MPS_NAMESPACE" shared sm-partitions
```

Require exact mappings and all four driver memory caps. Fractioning injects active-thread percentage **100** for static profiles; that removes an additional scalar ceiling. The assigned static partitions still limit compute. The model must not set `CUDA_MPS_SM_PARTITION`.

### Test client partition overrides

The execution probe launches eight bounded CUDA kernels and records client-visible SM IDs. It forces client compute to 100% and removes the pinned-memory hint. Test an omitted selector, a malformed selector, and a **valid peer-worker partition**. A malformed selector alone does not establish binding.

Copy the peer partition's `VISIBLE UUID` from `partitions.csv`. For example, when testing A on physical GPU 0, use B's GPU 0 token in `PEER_VISIBLE_UUID`:

```bash
"${K[@]}" -n "$NAMESPACE" exec -i "$POD_A" -c main -- python3 - \
  --uuid "$GPU0_UUID" --mode assigned --selection "$GPU0_UUID/$PEER_VISIBLE_UUID" \
  --expected-sm 36 < tp2/smid_probe.py | tee results/a-gpu0-peer-selector.jsonl
```

Repeat with `--mode correct-socket-omitted` and `--mode wrong-token`; pass the same selection argument, which those modes clear or replace. Repeat each case with `--all-visible`, on both physical GPUs, in both workers: **3 × 2 × 2 × 2 = 24 checks**. Each check must complete eight launches and retain its own expected SM count, including when it requests a larger peer partition. A smaller observed count does not prove the intended capacity, and client-visible IDs do not prove physical cross-client disjointness.

Run the memory controls with `--expected-sm` matching each assigned share. Only after these checks pass, stop the idle workers, wait for release, and apply the two-model manifests. Verify graphs, both ranks and frontend responses again. Repeat the 24 selector checks while models are resident.

## 6. Verify Automatic Reapplication

Save both workers' pod UIDs, container IDs, MPS namespaces, cgroups, driver limits, Ready states, and restart counts. Delete **only A's worker pod**:

```bash
"${K[@]}" -n "$NAMESPACE" delete pod "$POD_A"
```

Require a new A pod, container, cgroup, and MPS namespace with the same **36/72 SM** map and **19 GiB** hard limits. It must initialize the model and capture graphs without manually setting partitions, memory caps, or a per-replacement gate. B must retain its pod/container identity and remain Ready at every check. Send one request to each frontend concurrently and save both responses.

For a resident-memory enforcement check, attempt 4 GiB directly on each replacement-A GPU with client hints removed. First read its current usage and hard limit: its own remaining allowance must be below 4 GiB, while immediately measured physical free memory must exceed 5 GiB. Then use `allocation_probe.py --mib 4096 --expected-code 2 --hostile --narrow` with the current free-memory value and expected SM count. If either headroom condition is absent, report the check as inconclusive.

Fixed profiles allow one live container each. Use stop-before-start updates; a surge rollout can wait indefinitely while the old worker holds the profile. Live recovery of already-running CUDA models after an MPS daemon failure was not tested.

## Unequal Memory Is a Separate Procedure

The scheduler annotations above still provide one scalar memory value per container. This compute-policy extension does not add automatic unequal memory allocation.

Separate manual tests verified **20/30 GiB** for one TP2 worker and **17/21 versus 21/17 GiB** for two TP2 workers, with compute held at 54/54 SMs. Use the host-driver procedure in [Different Memory Limits per GPU](mps-per-gpu-limits.md#different-memory-limits-per-gpu) on each current container's leaf cgroup. Perform allocation checks before loading the model, and account for both workers' totals. Those manual cgroup limits require reapplication after container recreation; they are not evidence of automatic lifecycle support.

## Recorded Result and Stop Condition

The automatic-compute integration passed on the pinned r620 stack: two TP2 workers retained 36/72 and 72/36 SMs and 19 GiB per GPU, both captured CUDA graphs, both served inference, and worker A recovered its assignments after deliberate pod replacement. The experiment passed 48 selector checks and 18 direct memory-allocation checks, with zero unexpected model container restarts. B kept its identity throughout replacement, and 29 unrelated baseline pods retained their identities and restart counts.

Save the rendered manifests, exact commits and image digests, commands, logs, per-GPU readbacks and probe output under your run's `results/` directory. Record **PASS**, **FAIL**, or **BLOCKED** per check; never turn a missing API or inconclusive headroom check into a pass. Stop after these bounded checks. Do not add another model or start a sweep.

For cleanup, stop and delete only `tp2-a` and `tp2-b` in the namespace you created. Coordinate removing the experimental daemon policy after its clients stop; preserve unrelated namespaces, GPU Operator, KAI, and Dynamo platform resources.
