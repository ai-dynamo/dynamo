<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 07 — NVMe disk cleanup on s2877

## TL;DR — this is prudent, not blocking

| | |
|---|---|
| Needed per root | **~53 GB** |
| `nvme2` free (worst root) | **~212 GB** |
| Verdict | **Run A fits today without any cleanup.** |

Clean up anyway for headroom across repeated runs (each run writes a fresh
`<leaf>/device-N/shards/` tree), but do **not** treat this as a prerequisite.

## Capacity math

GLM-5.2-NVFP4 is ~**372 GB** of weights in total across the 8 ranks. GMS
sharded-SSD save stripes each device's shards across all configured roots
(`snapshot/backends/sharded_ssd.py:36-46`, `snapshot/storage_client.py:142-149`).

```
372 GB total ÷ 7 roots ≈ 53 GB per root
                      ≈ 80 GB per root with a 1.5× safety margin
```

| Root | Free | Needed | Headroom |
|---|---:|---:|---|
| **nvme2** | **212 GB** (94 % full) | ~53 GB | ~4× — tight but **sufficient** |
| nvme4 | 1.73 TB | ~53 GB | ample |
| nvme5-9 | ~2.4-2.5 TB each | ~53 GB | ample |

Each root is 3.78 TB. `nvme2` is the binding constraint; everything else is
comfortable. Sharding is even, so **`nvme2` filling up fails the whole save.**

> [!TIP]
> If you only want to de-risk cheaply, deleting just `gms-storage` (2.3 TB) on
> nvme2 takes it from 212 GB → ~2.5 TB free and matches the other roots. Steps
> 1-2 below are enough; the rest is optional tidying.

## Non-negotiable exclusions

> [!CAUTION]
> **Never delete these:**
> - `/mnt/dynamo-gms-nvme2/schwinns` — 62 GB, dated **2026-07-30**, recent and
>   likely in use by the current work. This is also where the new run's leaf
>   directory will live.
> - `lost+found` on any root — filesystem metadata; removing it breaks `fsck`
>   recovery.
>
> Everything listed for deletion below is dated **Mar-May 2026** and is stale
> output from prior benchmark/bench-save runs.

## Getting a shell on the node

`kubectl exec` needs a pod that mounts the roots. The **snapshot-agent** pod
already runs on s2877 with `hostPID`/privileged access, but it does *not* mount
`/mnt/dynamo-gms-nvme*`. Two options:

**Option A — a throwaway shell pod (preferred, isolated):**

```bash
CTX=nv-prd-dgxc.teleport.sh-dynamo-nscale-dev-cluster
NS=schwinns

kubectl --context "$CTX" -n "$NS" apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: nvme-janitor
spec:
  restartPolicy: Never
  nodeSelector:
    kubernetes.io/hostname: cluster-0967a26d-pool-14bee067-prctr-s2877
  tolerations:
    - {key: dra, operator: Exists, effect: NoSchedule}
    - {key: schwinns.ai-dynamo.io/reserved, operator: Exists, effect: NoSchedule}
    - {key: schwinns.ai-dynamo.io/gms-bench, operator: Exists, effect: NoSchedule}
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  containers:
    - name: sh
      image: busybox:1.36
      command: ["sleep", "36000"]
      securityContext:
        runAsUser: 0
      volumeMounts:
        - {name: nvme2, mountPath: /mnt/nvme2}
        - {name: nvme4, mountPath: /mnt/nvme4}
        - {name: nvme5, mountPath: /mnt/nvme5}
        - {name: nvme6, mountPath: /mnt/nvme6}
        - {name: nvme7, mountPath: /mnt/nvme7}
        - {name: nvme8, mountPath: /mnt/nvme8}
        - {name: nvme9, mountPath: /mnt/nvme9}
  volumes:
    - {name: nvme2, hostPath: {path: /mnt/dynamo-gms-nvme2, type: Directory}}
    - {name: nvme4, hostPath: {path: /mnt/dynamo-gms-nvme4, type: Directory}}
    - {name: nvme5, hostPath: {path: /mnt/dynamo-gms-nvme5, type: Directory}}
    - {name: nvme6, hostPath: {path: /mnt/dynamo-gms-nvme6, type: Directory}}
    - {name: nvme7, hostPath: {path: /mnt/dynamo-gms-nvme7, type: Directory}}
    - {name: nvme8, hostPath: {path: /mnt/dynamo-gms-nvme8, type: Directory}}
    - {name: nvme9, hostPath: {path: /mnt/dynamo-gms-nvme9, type: Directory}}
EOF

kubectl --context "$CTX" -n "$NS" wait --for=condition=Ready pod/nvme-janitor --timeout=120s
J() { kubectl --context "$CTX" -n "$NS" exec nvme-janitor -- sh -c "$1"; }
```

**Option B** — if you have SSH to s2877, work directly on `/mnt/dynamo-gms-nvme*`
and substitute host paths for `/mnt/nvmeN`.

## Step 0 — Baseline (record this before touching anything)

```bash
J 'df -h /mnt/nvme2 /mnt/nvme4 /mnt/nvme5 /mnt/nvme6 /mnt/nvme7 /mnt/nvme8 /mnt/nvme9'
J 'ls -la --time-style=long-iso /mnt/nvme2'
J 'du -sh /mnt/nvme2/* 2>/dev/null | sort -rh'
```

Save that output. It is your only record of what existed.

## Cleanup, ordered least-risky first

Each step is: **measure → confirm → delete → re-measure.** Nothing here is
recoverable, so do not batch the confirmations.

### Step 1 — `testfile` (65 GB) — pure scratch, zero risk

```bash
J 'ls -la --time-style=long-iso /mnt/nvme2/testfile; du -sh /mnt/nvme2/testfile'
# CONFIRM: name is literally "testfile", date is Mar-May 2026.
J 'rm -rf /mnt/nvme2/testfile'
J 'df -h /mnt/nvme2'
```

### Step 2 — `gms-storage` (2.3 TB) — the big win

```bash
J 'du -sh /mnt/nvme2/gms-storage; ls -la --time-style=long-iso /mnt/nvme2/gms-storage | head -20'
# CONFIRM: contents dated Mar-May 2026, nothing dated 2026-07-xx.
J 'rm -rf /mnt/nvme2/gms-storage'
J 'df -h /mnt/nvme2'   # expect ~2.5 TB free
```

> [!NOTE]
> This single step is sufficient. Stop here unless you want maximum headroom.

### Step 3 — `gms-shards` (433 GB) — stale saver output

```bash
J 'du -sh /mnt/nvme2/gms-shards; ls -la --time-style=long-iso /mnt/nvme2/gms-shards | head -20'
# CONFIRM: this is prior-run shard output, NOT the current run's leaf.
J 'rm -rf /mnt/nvme2/gms-shards'
J 'df -h /mnt/nvme2'
```

### Step 4 — old benchmark artifacts (282 GB combined)

```bash
J 'du -sh /mnt/nvme2/gms_bench_save_72b /mnt/nvme2/sharded-8x72 /mnt/nvme2/gms-synth-8gpu-72g'
# CONFIRM: all are 72B-model bench artifacts, unrelated to GLM-5.2.
J 'rm -rf /mnt/nvme2/gms_bench_save_72b'
J 'rm -rf /mnt/nvme2/sharded-8x72'
J 'rm -rf /mnt/nvme2/gms-synth-8gpu-72g'
J 'df -h /mnt/nvme2'
```

### Step 5 — stale HF caches (185 GB) — check most carefully

```bash
J 'du -sh /mnt/nvme2/hf-cache /mnt/nvme2/hf-cache-pr7325a1'
J 'ls -la --time-style=long-iso /mnt/nvme2/hf-cache'
```

> [!WARNING]
> These are **node-local** caches on NVMe. The experiment's model comes from the
> **PVC `shared-model-cache`** (mounted at `/home/dynamo/.cache/huggingface`), not
> from here — confirmed by the reference's `hf-cache` volume definition. Deleting
> these must not affect the run, but verify the dates are Mar-May 2026 and that
> nothing references them before proceeding.

```bash
J 'rm -rf /mnt/nvme2/hf-cache-pr7325a1'   # PR-specific, clearly stale
J 'rm -rf /mnt/nvme2/hf-cache'            # only after the check above
J 'df -h /mnt/nvme2'
```

### Step 6 — Final verification

```bash
J 'df -h /mnt/nvme2 /mnt/nvme4 /mnt/nvme5 /mnt/nvme6 /mnt/nvme7 /mnt/nvme8 /mnt/nvme9'
J 'ls -la --time-style=long-iso /mnt/nvme2'
# MUST still exist: schwinns/ and lost+found/
J 'ls -d /mnt/nvme2/schwinns /mnt/nvme2/lost+found'
```

## Expected outcome

| Step | Reclaimed | nvme2 free after |
|---|---:|---:|
| baseline | — | 212 GB |
| 1 `testfile` | 65 GB | ~277 GB |
| 2 `gms-storage` | 2 300 GB | ~2.58 TB |
| 3 `gms-shards` | 433 GB | ~3.01 TB |
| 4 bench artifacts | 282 GB | ~3.29 TB |
| 5 HF caches | 185 GB | ~3.48 TB |
| **total** | **~3.19 TB** | **~3.48 TB of 3.78 TB** |

That brings nvme2 in line with nvme5-9 and gives ~65× headroom over the ~53 GB
the run needs.

## Pre-create the run's leaf directories

The saver creates them (`storage_client.py:_prepare_output_dir` →
`os.makedirs(..., exist_ok=True)`), but pre-creating catches permission problems
before GPU time is spent:

```bash
for n in 2 4 5 6 7 8 9; do
  J "mkdir -p /mnt/nvme$n/schwinns/gmsv1-glm52-dep8 && ls -ld /mnt/nvme$n/schwinns/gmsv1-glm52-dep8"
done
```

Those are the **host**-side paths; the saver sees them as
`/mnt/gms-ssd/nvme{N}/schwinns/gmsv1-glm52-dep8` (see `05-prior-attempts.md`).

## Teardown

```bash
kubectl --context "$CTX" -n "$NS" delete pod nvme-janitor
```

<!-- UNVERIFIED: all sizes, free-space figures, and directory dates are from the
     user's live measurement, not re-measured here. Re-run Step 0 before deleting
     anything — the `du -sh` in each step is the authoritative check. -->
