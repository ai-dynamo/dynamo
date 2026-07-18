<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GMS CUDA initialization A/B

This directory contains a narrow, switchable experiment for GMS server CUDA
initialization. The variants use the same source and workload:

- **A:** `DYN_GMS_SERVER_GPU_UUID_ISOLATION=0`. This preserves the prior
  behavior: one child per GPU, each child receives its physical ordinal in
  `--device` and inherits the server container's full GPU visibility.
- **B:** `DYN_GMS_SERVER_GPU_UUID_ISOLATION=1`. This preserves one
  allocation-owning OS child per physical GPU, but each child receives exactly
  its assigned full GPU UUID in `CUDA_VISIBLE_DEVICES` and uses `--device 0`
  inside that single-device view.

The loader remains one process with the same full eight-GPU DRA allocation in
both variants. The flag is consumed only by
`gpu_memory_service.cli.server`; setting it in the shared worker image does not
filter the loader, main worker, or any transfer thread.

This experiment does not combine GMS children, move transfers into the server,
create a CUDA context in the supervisor, change VMM layout, or change storage,
transfer, model, vLLM, FlashInfer, NCCL, or compile-cache settings.

## Source behavior and instrumentation

The server supervisor discovers devices through NVML and starts children with
`subprocess.Popen`. It does not call CUDA before launching them. Every child is
a fresh Python process.

Before this change, the loader did not explicitly call `cuInit` before loading.
Its first CUDA call in each per-device loader thread was `cudaSetDevice`. The
first of those concurrent calls initializes CUDART and the Driver implicitly.
`cudaSetDevice` remains necessary in every loader thread because the selected
device and current CUDA context are host-thread state. The later
synchronize/unmap/commit and transfer paths run from those threads and require
the assigned device's primary context to be current. Replacing it with Driver
API calls is intentionally left to a separate experiment.

With `GMS_SNAPSHOT_PROFILE=1`, the images add these timing records:

- loader: `first_process_cudart_call`, each `cuda_set_device`,
  `current_context_query`, `current_context_established`, and the concurrent
  `all_device_cuda_initialization` envelope;
- loader client: the later explicit, normally idempotent `client_cu_init`;
- server child: explicit `server_cu_init`, `allocation_manager_ready`, and
  `socket_ready`;
- restore: `restore_target_mapping_envelope`, whose `wall_start_ns` marks
  mapping start.

`first_process_cudart_call` means the first invocation claimed under the
profiling lock; concurrent scheduling means it is not asserted to be the first
call to finish.

## Known-good workload provenance

`manifests/base/dgd.yaml` is the exact payload copied from:

```text
gms-profile-evidence-20260717T195352Z/manifests/tep8-profile-dgd-r2.yaml
```

Its source payload SHA-256 is
`13172fe4d83929ba94ab9ff2768e70a77afd7fdb010f1e9cbf6b82a75a067d9e`.
Only an SPDX header was prepended in this directory.

The overlays preserve:

- DGD `g52-t8-gms-prof-r29604929787-r2` in namespace `schwinns`;
- node `cluster-0967a26d-pool-14bee067-prctr-s2877`;
- model `nvidia/GLM-5.2-NVFP4`, TP8, DP1, and EP8;
- FlashInfer MNNVL all-reduce and `flashinfer_nvlink_one_sided` MoE A2A;
- NCCL over TCP/socket with the source NCCL environment unchanged;
- `--gpu-memory-utilization 0.80`;
- `VLLM_DISABLE_COMPILE_CACHE=1`;
- PVC `snapshot-pvc`, the seven NVMe roots, and sharded-SSD queue settings;
- frontend image digest
  `sha256:ffc100b5511b5c3f5f73f5edf01e44895b531fcd192aa94761158867c5f17291`;
- checkpoint CR `checkpoint-57a124961e2a47a2cf9c2712e58a0a2b`,
  checkpoint ID `57a124961e2a47a2cf9c2712e58a0a2b`, and GMS artifact
  `/checkpoints/gms/g52-t8-gms-prof-r29604929787-r2/versions/1`.

Both overlays replace the source auto-capture job with the same retained,
explicit `checkpointRef` and preserve `startupPolicy: WaitForCheckpoint`.
Consequently, they use the normal DynamoCheckpoint restore path, create no
checkpoint job, and cannot own or delete the referenced checkpoint. Do not run
`snapshotctl` for this experiment.

The source restore evidence is under:

```text
gms-profile-evidence-20260717T195352Z/capture-r2/restore-profile-r3-parent
```

Its lifecycle record shows:

- preflight began `2026-07-17T21:52:49.181418925Z`;
- scale-up patch was sent `2026-07-17T21:53:08.145101754Z`;
- the worker became ready `2026-07-17T21:53:45.931761605Z`;
- inference ran from `2026-07-17T21:53:50.154645636Z` through
  `2026-07-17T21:53:57.085633419Z`;
- the run completed `2026-07-17T21:55:19.633165811Z`.

The semantic response is
`capture-r2/restore-profile-r3-parent/inference/response.json`: HTTP 200,
`finish_reason=stop`, with a coherent answer. In that baseline, the eight
server allocation-manager initializations took about 7.87-8.11 seconds, the
eight first loader-thread `cudaSetDevice` calls took about 6.82-6.90 seconds,
and all eight devices loaded in 28.83 seconds.

The older retained GLM 5.2 TP8/EP8 checkpoint documented in
`HANDOFF-dynamo-glm52-snapshot-gms-filter-20260713.md` remains untouched:
`glm52-tep8-gms-ssd-top-29213270567-source-v2`, checkpoint ID
`dbde5ab07f1ee977`.

## Images

The images are built from source commit
`b4b23ff041f92285c8f6ceb19c54a2217a43eb43` on branch
`experiment/gms-cuda-init-uuid-isolation-20260718`, over the exact known-good
worker base digest
`sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2`.

| Variant | Image |
|---|---|
| A | `dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-b4b23ff041f9-a@sha256:6d8c8ce078d82b47c33924c60040b813d106176a26565b205f42e55f6cb8224c` |
| B | `dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-b4b23ff041f9-b@sha256:43dc7c520bd4b019d796623e31c4544bb348ca1acb8282206a39330fa63266bf` |

Both images have `org.opencontainers.image.revision=b4b23ff04...`, identical
root filesystem layers, and byte-identical installed experiment source:

| File | SHA-256 |
|---|---|
| `gpu_memory_service/cli/server.py` | `9f7aaf19638e880b51c48bf6b8a8e6182c5cfa8bb14f1b2a5f5d10056fb73517` |
| `gpu_memory_service/cli/snapshot/loader.py` | `3af0b7f0c6dcd7040933312a385f2ae060a96abc63e7f2fd84fd233eacca7ef6` |
| `gpu_memory_service/common/utils.py` | `cffe255b930f56725f50b8e6b0bee76fbce49a61f1bbf020ae6218a7f918b5f0` |
| `gpu_memory_service/common/cuda_utils.py` | `db3d25994173dd584d5a0a04daf6261997263ae5b8c188298d7d15d26f9efa87` |
| `gpu_memory_service/server/allocations.py` | `fb7f1d947c8391fc5a8140c39f33a90e98329bf52f2ba71d54d6e98942b4f47a` |

The image config and audit label are `0` for A and `1` for B. The operator's
`EnsureServerSidecar` constructs `gms-server` from the worker main image, while
the loader is the explicit `gms-loader` container. The overlays therefore
replace those two worker images. The frontend, snapshot agent, and operator do
not contain either process and are not rebuilt.

To reproduce the builds:

```bash
./benchmarks/gms_cuda_init_ab/build-images.sh
docker push dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-b4b23ff041f9-a
docker push dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-b4b23ff041f9-b
```

## Tester procedure

The variants use the same DGD name and must run sequentially. Set the context
explicitly; never change the current context:

```bash
export CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
export NS=schwinns
export DGD=g52-t8-gms-prof-r29604929787-r2
export CKPT=checkpoint-57a124961e2a47a2cf9c2712e58a0a2b
```

First confirm the retained checkpoint is still `Ready`, then render and inspect
both variants:

```bash
kubectl --context "$CTX" -n "$NS" get dynamocheckpoint "$CKPT"
kubectl --context "$CTX" -n "$NS" kustomize \
  benchmarks/gms_cuda_init_ab/manifests/variant-a > /tmp/gms-init-a.yaml
kubectl --context "$CTX" -n "$NS" kustomize \
  benchmarks/gms_cuda_init_ab/manifests/variant-b > /tmp/gms-init-b.yaml
diff -u /tmp/gms-init-a.yaml /tmp/gms-init-b.yaml
```

The expected rendered diff consists only of two image references and two
variant annotations. Run A and collect server, loader, snapshot-agent, worker,
and lifecycle logs before scaling the worker to zero. Then run B with the same
cache-conditioning and measurement procedure:

```bash
kubectl --context "$CTX" -n "$NS" apply -k \
  benchmarks/gms_cuda_init_ab/manifests/variant-a

# Collect A evidence and validate inference before proceeding.

kubectl --context "$CTX" -n "$NS" patch dynamographdeployment "$DGD" \
  --type=json \
  -p='[{"op":"replace","path":"/spec/components/1/replicas","value":0}]'

kubectl --context "$CTX" -n "$NS" apply -k \
  benchmarks/gms_cuda_init_ab/manifests/variant-b
```

Do not delete/recreate the DGD or checkpoint, do not apply both variants
concurrently, and do not use `snapshotctl`.

## Expected comparison

The primary comparison is A versus B for `server_cu_init`, followed by
`allocation_manager_ready` and `socket_ready`, per physical GPU and as an
eight-child wall-clock envelope. B is expected to reduce Driver initialization
work or contention if full inherited visibility is the cause.

The loader's `first_process_cudart_call`, every `cuda_set_device`,
`current_context_established`, `all_device_cuda_initialization`, and
`restore_target_mapping_envelope` should remain within run-to-run noise because
the loader still sees all eight DRA GPUs and its code path is identical. A
change in loader visibility, device count, transfer time, checkpoint identity,
or inference output invalidates the comparison.
