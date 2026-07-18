<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GMS CUDA initialization A/B

This directory contains a narrow, switchable experiment for GMS server CUDA
initialization. Both variants use the exact retained-checkpoint `main` image,
retained checkpoint, and workload. Only `gms-loader` and `gms-server` use the
A/B experiment images:

- **A:** `DYN_GMS_SERVER_GPU_UUID_ISOLATION=0`. This preserves prior behavior:
  one child per GPU, each child receives its physical ordinal in `--device` and
  inherits the server container's full GPU visibility.
- **B:** `DYN_GMS_SERVER_GPU_UUID_ISOLATION=1`. This preserves one
  allocation-owning OS child per physical GPU, but each fresh-exec child
  receives one full physical UUID in `CUDA_VISIBLE_DEVICES` and uses
  `--device 0` inside that one-device view.

The loader remains one all-visible process with the same eight-GPU DRA
allocation in both variants. The flag is consumed only by
`gpu_memory_service.cli.server`; it does not filter the main worker, loader, or
transfer threads. The experiment does not combine server children, move
transfers into the server, create a CUDA context in the supervisor, or change
VMM layout, storage, checkpoint, model, vLLM, FlashInfer, NCCL, or compile-cache
behavior.

## Visibility and profile identity

The isolated supervisor discovers physical UUIDs through NVML but does not
assume NVML scope proves allocation scope:

- `NVIDIA_VISIBLE_DEVICES=none` resolves to no GPUs.
- An explicit NVIDIA UUID allocation is authoritative. CUDA UUIDs and ordinals
  may only select GPUs within that allocation.
- Numeric visibility without an explicit UUID allocation is rejected.
- `all` or `void` without authoritative UUIDs is rejected.
- This node-pinned experiment declares the exact known-good eight-UUID DRA
  allocation through `DYN_GMS_SERVER_EXPECTED_GPU_UUIDS`. Startup resolves
  those UUIDs against NVML and fails if any are absent. A host-wide NVML view
  therefore cannot add GPUs outside the declared allocation.

The exact ordinal-order UUIDs for
`cluster-0967a26d-pool-14bee067-prctr-s2877` are:

```text
GPU-02ff0cc1-647f-dee7-8365-921738e945a6
GPU-0d5ad102-eb8f-922a-173e-e91033320e0f
GPU-9c595f65-4651-0b25-f95c-09a0abd5f5fa
GPU-4fba7684-5a96-6280-91ff-b41f7484564c
GPU-3ef7c092-d55c-ca6c-0018-9fc89ed28683
GPU-32a6c51c-d07d-7513-86b5-813b64e452d2
GPU-56c0be30-f1d3-a00d-8a2a-7fa70da8037f
GPU-ce231b92-c54f-3f0c-af1c-1a696db97f51
```

Every server `SnapshotProfile` record includes the physical UUID, process PID,
and CUDA child ordinal. In B, `server_cu_init`,
`allocation_manager_ready`, and `socket_ready` therefore retain
`device=0` while remaining attributable to one physical GPU and process.
Launch and readiness logs include the same UUID/PID identity. In A, the profile
includes each physical ordinal and UUID.

## Timing semantics

Before this experiment, the loader's first CUDA call in each concurrent
per-device thread was `cudaSetDevice`. That call remains in every loader thread
because device selection and the current CUDA context are thread state needed
by later synchronize, unmap, commit, and transfer operations.

With `GMS_SNAPSHOT_PROFILE=1`, the images emit:

- loader: `first_claimed_cuda_set_device`, each `cuda_set_device`,
  `current_context_query`, `current_context_established`, and
  `all_device_cuda_initialization`;
- loader client: the later explicit, normally idempotent `client_cu_init`;
- server child: `server_cu_init`, `allocation_manager_ready`, and
  `socket_ready`;
- restore: `restore_target_mapping_envelope`, whose `wall_start_ns` marks
  mapping start.

`first_claimed_cuda_set_device` means only that a thread won the profiling
bookkeeping claim before making its unsynchronized `cudaSetDevice` call. The
lock is released before CUDA entry. Another thread may enter CUDART first, so
this phase does **not** identify the first actual CUDART entrant or the thread
that incurred process initialization. The CUDA calls are intentionally not
serialized or reordered.

`all_device_cuda_initialization` begins before task submission and ends after
each profiled thread's `cuCtxGetCurrent` validation. It includes task
scheduling, all `cudaSetDevice` calls, and current-context queries; it is not a
pure CUDART duration. The context query is profile-only instrumentation applied
equally to A and B.

## Known-good workload

`manifests/base/dgd.yaml` is the exact payload copied from:

```text
gms-profile-evidence-20260717T195352Z/manifests/tep8-profile-dgd-r2.yaml
```

Its source payload SHA-256, excluding the added three-line SPDX preamble, is
`13172fe4d83929ba94ab9ff2768e70a77afd7fdb010f1e9cbf6b82a75a067d9e`.
The overlays preserve:

- DGD `g52-t8-gms-prof-r29604929787-r2` in namespace `schwinns`;
- node `cluster-0967a26d-pool-14bee067-prctr-s2877`;
- model `nvidia/GLM-5.2-NVFP4`, TP8, DP1, and EP8;
- FlashInfer MNNVL all-reduce and `flashinfer_nvlink_one_sided` MoE A2A;
- the source TCP/socket NCCL environment;
- `--gpu-memory-utilization 0.80`;
- `VLLM_DISABLE_COMPILE_CACHE=1`;
- PVC `snapshot-pvc`, seven NVMe roots, and sharded-SSD queue settings;
- frontend digest
  `sha256:ffc100b5511b5c3f5f73f5edf01e44895b531fcd192aa94761158867c5f17291`;
- retained checkpoint `checkpoint-57a124961e2a47a2cf9c2712e58a0a2b`;
- checkpoint ID `57a124961e2a47a2cf9c2712e58a0a2b`; and
- GMS artifact
  `/checkpoints/gms/g52-t8-gms-prof-r29604929787-r2/versions/1`.

The checkpoint's retained identity is:

```text
UID:    1fb182f1-1a4c-4c51-aff0-67ab530437ea
phase:  Ready
ownerReferences: null
deletionTimestamp: null
```

Both overlays set worker replicas to zero on apply, use only the normal
`checkpointRef` restore path with `startupPolicy: WaitForCheckpoint`, and omit
the checkpoint job. `run-variant.sh` scales the existing worker component to
one only after preflight and cache eviction. Never run `snapshotctl`, and never
delete or recreate the retained checkpoint.

The retained checkpoint's successful target container used exactly:

```text
dynamoci.azurecr.io/ai-dynamo/dynamo:760e55e21e14f76d7c204920f00ea9144d819b4b-vllm-placeholder-run-29604929787-1@sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2
```

Both overlays leave `main` at that exact image reference. This is a
checkpoint-root-filesystem compatibility requirement, not an A/B variable.

## Invalid prior retries

Two prior retries replaced `main` with a derivative image that added a GMS
wheel layer, then failed while CRIU reconstructed a TUN device. Those runs
changed the checkpoint target root filesystem and are invalid measurements for
this experiment. They do not establish a node runtime fault. Discard their
diagnostic conclusion; retain the checkpoint and rerun only with the exact
`main` image above.

## Final images and provenance

The exact image source is commit:

```text
SOURCE_COMMIT=324c14408b2cb79f39d80a8d90fe4ae54182bef2
```

The A/B GMS images were built from:

```text
dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2
```

| Variant | Final image |
|---|---|
| A | `dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-324c14408b2c-a@sha256:d0ea4cc1aceeeeef5825c418999ceca00fcde20dbdbc203d4b2bc683a874708a` |
| B | `dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-324c14408b2c-b@sha256:f0e3d788dca28715674705a7f151636dae3ee868f4df5575c1e284a777a7ab0a` |

Both images have 79 identical OCI filesystem layer descriptors and these
labels:

```text
org.opencontainers.image.revision=324c14408b2cb79f39d80a8d90fe4ae54182bef2
com.nvidia.dynamo.source-archive-sha256=f744868db122a1dd43ee56291c7390e8ca937b059acaf174a24953bf9bcdce9f
com.nvidia.dynamo.dockerfile-sha256=c5d72e01a8d660a46c962b928d38cfdf7f0bdf50f7ac8f144fcfce521d9e4ac7
```

Their OCI config digests differ because A and B intentionally set different
isolation env/label values:

```text
A config: sha256:99f7e36945d1af741e1c0fc8c752e6e4f106b5e689614220bf9c5a0292e40555
B config: sha256:5857baf8f6ecb1180fc5ecf45785b0478f5bae94d929a769dc739d2b175e2b82
```

Installed source hashes are identical in A and B and match `SOURCE_COMMIT`:

| File | SHA-256 |
|---|---|
| `cli/server.py` | `8bda1b32f7a9d5ccab9bd977a08e1afbc2511640991f79218b6654ca177c5f0f` |
| `cli/runner.py` | `18ba0fca9afc6259c305e88017e9c055d7b70824b355170ed3b09f7ca0cef02e` |
| `cli/snapshot/loader.py` | `f63462f01c3a65da116eb1ddc9c26fa51c80e0e219f4f9674b6295596a5f78fa` |
| `common/utils.py` | `4f574d19467c8328d4ff1fb01914bac341418b269dde4ff350f91b50dd050fad` |
| `common/cuda_utils.py` | `db3d25994173dd584d5a0a04daf6261997263ae5b8c188298d7d15d26f9efa87` |
| `server/allocations.py` | `fb7f1d947c8391fc5a8140c39f33a90e98329bf52f2ba71d54d6e98942b4f47a` |
| `server/rpc.py` | `c81688e68b1bbd1df2353293b96be3aa81c7343cf6092168855622c5a4c57844` |

Each overlay assigns its final image only to the explicit `gms-loader` and
explicit native-init-sidecar `gms-server`. It does not assign this derivative
image to `main`. The custom server mirrors the operator-generated shape:

- name `gms-server`;
- command `python3 -m gpu_memory_service.cli.server`;
- `GMS_SOCKET_DIR=/gms-intrapod-control`;
- the `gms-intrapod-control` shared `emptyDir` mount;
- the `intrapod-shared-gpu` DRA claim; and
- `restartPolicy: Always`.

The existing operator first applies client socket and claim wiring to `main`,
then returns without adding a server when an init container named
`gms-server` already exists. Reconciliation therefore preserves the custom
experiment image and produces exactly one server. Frontend, `main`,
snapshot-agent, and operator images remain unchanged.

### Reproduce and inspect

Build only from the pinned commit. The script archives both GMS source and the
Dockerfile from that commit, so a working-tree Dockerfile cannot drift from the
revision label:

```bash
SOURCE_COMMIT=324c14408b2cb79f39d80a8d90fe4ae54182bef2 \
  ./benchmarks/gms_cuda_init_ab/build-images.sh

docker push \
  dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-324c14408b2c-a
docker push \
  dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-324c14408b2c-b
```

Inspect the live registry digests and compare OCI layers:

```bash
docker manifest inspect --verbose \
  dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-324c14408b2c-a \
  > /tmp/gms-init-a.manifest.json
docker manifest inspect --verbose \
  dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-324c14408b2c-b \
  > /tmp/gms-init-b.manifest.json

jq -r '.Descriptor.digest' /tmp/gms-init-{a,b}.manifest.json
jq -r '.OCIManifest.layers[].digest' \
  /tmp/gms-init-a.manifest.json > /tmp/gms-init-a.layers
jq -r '.OCIManifest.layers[].digest' \
  /tmp/gms-init-b.manifest.json > /tmp/gms-init-b.layers
diff -u /tmp/gms-init-a.layers /tmp/gms-init-b.layers
jq -r '.OCIManifest.config.digest' \
  /tmp/gms-init-{a,b}.manifest.json
```

Inspect source labels, filesystem diff IDs, and installed hashes:

```bash
for variant in a b; do
  image="dynamoci.azurecr.io/ai-dynamo/dynamo:gms-cuda-init-ab-324c14408b2c-$variant"
  docker pull "$image"
  docker image inspect "$image" |
    jq '.[0] | {RepoDigests, labels:.Config.Labels, env:.Config.Env,
      layers:.RootFS.Layers}'
  docker run --rm --entrypoint python3 "$image" -c '
import hashlib
import pathlib
import gpu_memory_service

root = pathlib.Path(gpu_memory_service.__file__).parent
files = [
    "cli/server.py",
    "cli/runner.py",
    "cli/snapshot/loader.py",
    "common/utils.py",
    "common/cuda_utils.py",
    "server/allocations.py",
    "server/rpc.py",
]
for name in files:
    print(hashlib.sha256((root / name).read_bytes()).hexdigest(), name)
'
done

git show \
  324c14408b2cb79f39d80a8d90fe4ae54182bef2:benchmarks/gms_cuda_init_ab/Dockerfile |
  sha256sum
git archive 324c14408b2cb79f39d80a8d90fe4ae54182bef2 \
  lib/gpu_memory_service | sha256sum
```

## Deterministic tester procedure

Run A and B sequentially from the repository root. Every cluster operation in
`run-variant.sh` uses the explicit context
`nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster` and namespace `schwinns`;
it never changes the active kube context.

First render and inspect both zero-replica overlays:

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
NS=schwinns
CKPT=checkpoint-57a124961e2a47a2cf9c2712e58a0a2b

kubectl --context "$CTX" -n "$NS" get dynamocheckpoint "$CKPT" \
  -o json |
  jq '{uid:.metadata.uid, ownerReferences:.metadata.ownerReferences,
    deletionTimestamp:.metadata.deletionTimestamp, phase:.status.phase,
    checkpointID:.status.checkpointID}'
kubectl --context "$CTX" -n "$NS" kustomize \
  benchmarks/gms_cuda_init_ab/manifests/variant-a \
  > /tmp/gms-init-a.yaml
kubectl --context "$CTX" -n "$NS" kustomize \
  benchmarks/gms_cuda_init_ab/manifests/variant-b \
  > /tmp/gms-init-b.yaml
diff -u /tmp/gms-init-a.yaml /tmp/gms-init-b.yaml
```

The raw rendered diff must contain only the `gms-loader` and `gms-server`
experiment image references and the two variant annotations. It must never
contain a `main` image difference. Both rendered worker replica counts must be
zero.

Run A:

```bash
./benchmarks/gms_cuda_init_ab/run-variant.sh \
  a evidence/gms-cuda-init-a
```

Do not begin B unless A exits zero. A's teardown waits for all matching DCD and
Deployment replica specs to reach zero, the old worker pod to disappear, and
the generated DRA `ResourceClaim` to be released. Then run B:

```bash
./benchmarks/gms_cuda_init_ab/run-variant.sh \
  b evidence/gms-cuda-init-b
```

For each variant, the runner:

1. Verifies checkpoint UID, `Ready` status, checkpoint ID, retained ownership,
   and absence of a deletion timestamp before apply, before teardown, and after
   teardown.
2. Verifies the live s2877 `ResourceSlice` has exactly the expected eight UUIDs
   and that the worker claim template requests exactly eight
   `gpu.nvidia.com` devices.
3. Applies the selected overlay at zero replicas, then confirms no old
   DCD/Deployment replica, worker pod, or matching DRA claim remains.
4. Starts/reuses the `gms-cuda-init-cache-helper-root-v2` Pod from
   `cache-helper.yaml` on s2877. The helper runs as UID/GID 0 so it can
   traverse the root-owned retained checkpoint, while dropping every
   capability and disabling privilege escalation. It has no GPU or DRA claim,
   service-account token, privileged access, or host PID/network access, and
   all checkpoint/NVMe mounts remain read-only. Before cache eviction, the
   runner records and strictly validates UID/GID 0 and full traversal of both
   exact checkpoint roots. It then streams `gms-fadvise-exact.py` into the
   helper, which calls `POSIX_FADV_DONTNEED` on exactly these trees:
   - `/checkpoints/57a124961e2a47a2cf9c2712e58a0a2b/versions/1`
   - `/checkpoints/gms/g52-t8-gms-prof-r29604929787-r2/versions/1`
   - the seven
     `/cache/nvme{2,4,5,6,7,8,9}/schwinns/g52-t8-gms-prof-r29604929787-r2`
     trees.
5. Starts timestamped snapshot-agent/operator/container log capture, a 200 ms
   host sampler, a 200 ms `nvidia-smi` sampler, and one-second pod lifecycle
   records before scaling the worker to one.
6. Captures the allocated DRA claim and waits up to 900 seconds for worker
   readiness.
7. Verifies the live `main` `imageID` equals
   `sha256:44ade91e2dc09c9732ea038b9db81bff7b3fcdc7b5a692ab1142d2ee7bde0ca2`.
   It separately verifies that `gms-loader` and the sole native-init-sidecar
   `gms-server` use A's `sha256:d0ea4cc1...` or B's
   `sha256:f0e3d788...` digest, including the exact command, socket
   environment, shared mount, DRA claim, and restart policy.
8. Runs `nvidia-smi --query-gpu=uuid` in the loader, requires exactly eight
   devices, and compares the UUID set with the expected allocation.
9. Runs the same deterministic chat-completion request after normal
   `DynamoCheckpoint` restore, requires HTTP 200, `finish_reason=stop`, nonempty
   output, and coherent references to blue light, atmosphere, and scattering.
10. Validates every `server_cu_init`, `allocation_manager_ready`, and
    `socket_ready` profile against the expected physical UUID set and a valid
    PID. For B it also requires `device=0` and PID agreement with the
    supervisor's per-UUID launch records.
11. Validates the snapshot-agent's ordinal-order DRA UUID record, collects all
    final objects/logs, terminates and waits for sampler/log-follower process
    trees, scales to zero, performs the teardown waits, and writes
    `SHA256SUMS`. The same descendant cleanup runs on failure, and checksum
    generation is skipped if cleanup cannot confirm that descendants stopped.

The helper Pod contains no checkpoint lifecycle commands. `posix_fadvise` on
read-only file descriptors requires no Linux capability, so the helper adds
none. Do not delete the old `gms-cuda-init-cache-helper` Pod, the retained
checkpoint, or any checkpoint data. Do not use `snapshotctl`. This harness-only
change does not require rebuilding either A/B image.

## Expected comparison

The primary comparison is A versus B for `server_cu_init`, followed by
`allocation_manager_ready` and `socket_ready`, per physical GPU and as an
eight-child wall-clock envelope. B is expected to reduce Driver initialization
work or contention if inherited all-GPU visibility is the cause.

The loader's `first_claimed_cuda_set_device`, every `cuda_set_device`,
`current_context_established`, `all_device_cuda_initialization`, and
`restore_target_mapping_envelope` should remain within run-to-run noise because
the loader sees all eight DRA GPUs and follows the same code path. A change in
loader visibility, device count, transfer time, checkpoint identity, live image
digest, UUID attribution, or inference coherence invalidates the comparison.
