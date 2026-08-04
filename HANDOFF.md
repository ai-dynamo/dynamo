<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV-Reuse-on-Failover POC — Handoff

Working notes for the vLLM POC tracked in
[#12053](https://github.com/ai-dynamo/dynamo/issues/12053): GMS-owned GPU KV survives an
inference-engine process death, and a local shadow vLLM resumes from the preserved cache.

This branch is an **investigation branch, not a merge candidate.** It carries product-shaped
changes, throwaway debug instrumentation, and standalone harnesses side by side, in separate
commits so the layers can be pulled apart later. Read [Status](#status) first, then
[Where we are stuck](#where-we-are-stuck) — that is the live question.

**Base:** `origin/main` @ `290f96bc3e`. **Validated on:** Qwen3-0.6B, TP=1, single B200, 28 layers.

---

## Status

| Milestone | What it does | State |
|---|---|---|
| **M0** — baseline + crash model | Reproducible 2-engine failover harness; North Star reuse probe (TTFT proxy + `prefix_hits` + greedy-output correctness). Baseline is a MISS, as expected. | ✅ committed, green |
| **M1** — persist + reattach bytes | GMS `--persist-on-abort` keeps KV allocations across a writer crash; the standby reattaches the *same* physical instead of allocating a fresh pool. | ✅ committed, green |
| **M2/M3** — persist + rehydrate the index | Carries the vLLM prefix-cache index (`block_hash → block_id`) across the crash so the reattached bytes become a real HIT. | ⚠️ **mechanism works, output is wrong** |
| **Test 1** — same-engine sleep/wake | One non-shadow engine: warm → sleep → wake → re-send. | ✅ 5/5, byte-identical output |

**The one-line problem:** after failover we get a genuine prefix **HIT** on byte-perfect
reattached KV, and the output is still garbage.

---

## North Star

Warm a long prefix on the primary → SIGKILL it → the standby reattaches the preserved KV →
re-send the *same* prefix → observe a prefix-cache **HIT** with **byte-identical** output to
the pre-crash reference. M0's baseline measurement is the "before": `prefix_hits=0`, MISS.

---

## Background

Enough to read the code without spelunking.

**GMS (GPU Memory Service)** is a per-GPU daemon that owns GPU physical memory via CUDA VMM
(`cuMemCreate` / `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle` /
`cuMemMap`). Clients hold *mappings* (virtual addresses) onto server-owned *allocations*
(physical). Because the daemon owns the physical, it can outlive the engine that was writing
it. Access is arbitrated by a socket-as-lock FSM (`EMPTY → RW → COMMITTED → RO`).

**Two identities, easy to confuse:**
- `allocation_id` (str, uuid) — the *physical* identity. Used for export/import.
- `layout_slot` (int) — a positional index (`_next_layout_slot++`). Used for pairing/matching
  during remap. The scratch path resets it to 0 for moved mappings, so `layout_slot` is **not**
  a stable cross-process key; `allocation_id` is.

**Shadow failover:** two engines start; both alias their KV over a tiny scratch block (cheap
colocation). They race for an `flock`; the winner wakes and serves, the loser blocks. When the
winner dies the kernel drops the lock, the loser wakes and swaps scratch → real KV via
`prepare_scratch_for_reallocation` → `reallocate_all_handles` → `remap_all_vas`.

**Why the index is a separate problem from the bytes.** M1 makes the KV *bytes* survive. But a
prefix HIT needs vLLM's `BlockPool.cached_block_hash_to_block` index (`block_hash → block_id`),
which lives in scheduler process RAM and dies with the process. It **cannot be regenerated from
the bytes** — the hash is over token ids, not over KV contents. Hence M2/M3: write the index
through to a shared log and replay it on the standby. Both engines pin identical geometry, so
`block_id` N on the standby is the same reattached physical block as on the primary.

**Determinism (B3):** vLLM's block-hash chain roots at `NONE_HASH = os.urandom(32)` per process,
so the two engines would never agree. The harness pins `PYTHONHASHSEED=0`.

---

## Reproduce

### Environment

Single-GPU dev pod, spec in [`poc/pod.yaml`](poc/pod.yaml) (image
`dynamoci.azurecr.io/ai-dynamo/dynamo:1.4.0-ci-290f96bc3e...-vllm-runtime`, matching the base
commit). Everything runs inside that pod.

**The `PYTHONPATH` override (load-bearing).** Installed `site-packages` is root-owned and there
is no passwordless sudo in the pod, so you cannot edit `gpu_memory_service` in place:

```bash
cp -r /usr/local/lib/python3.12/dist-packages/gpu_memory_service /tmp/gmsoverride/
# edit under /tmp/gmsoverride/, then launch everything with:
export PYTHONPATH=/tmp/gmsoverride
```

Re-sync an edited file from a checkout (`kubectl cp` fails on "File exists" / perms):

```bash
kubectl exec -i <pod> -n <ns> -c dev -- \
  bash -c 'cat > /tmp/gmsoverride/gpu_memory_service/integrations/vllm/kv_index_persist.py' \
  < lib/gpu_memory_service/integrations/vllm/kv_index_persist.py
```

### Environment variables

| Var | Purpose |
|---|---|
| `GMS_KV_DEBUG=1` | Enables **all** instrumentation below. Off ⇒ zero overhead, no probes. |
| `GMS_KV_INDEX_PATH` | Shared append-only prefix-index log both engines use (M2/M3). |
| `GMS_KV_TARGET_FILE` | Where the winner dumps its stable-point KV fingerprint for the standby to compare against. |
| `PYTHONHASHSEED=0` | B3 determinism — required or no rehydrated hash ever matches. |
| `CLEAN_HANDOFF=1` | Diagnostic: `control/sleep` the winner (clean KV unmap) *before* killing it. |

### Run

```bash
# The North Star: 2-engine shadow failover (this is the one that reproduces the bug)
GMS_KV_DEBUG=1 CLEAN_HANDOFF=1 PYTHONPATH=/tmp/gmsoverride \
  bash poc/harness.sh Qwen/Qwen3-0.6B 1

# Test 1 control: single non-shadow engine, sleep/wake. Passes 5/5.
GMS_KV_DEBUG=1 PYTHONPATH=/tmp/gmsoverride \
  bash poc/test1_sleep_wake.sh Qwen/Qwen3-0.6B

# Is the KV externally readable? Engine writes, independent client reads.
GMS_KV_DEBUG=1 PYTHONPATH=/tmp/gmsoverride bash poc/test_ext_read.sh Qwen/Qwen3-0.6B

# Pure two-process GMS byte-sharing matrix (no vLLM at all). All variants pass.
bash poc/gms_ipc_matrix.sh
```

Engine sleep/wake is driven over the system port (route `/engine/{*path}` in
`system_status_server.rs`):

```bash
curl -X POST http://localhost:$DYN_SYSTEM_PORT/engine/control/sleep   -d '{"level":1}'
curl -X POST http://localhost:$DYN_SYSTEM_PORT/engine/control/wake_up -d '{}'
```

---

## The measurement rig

All of it is `GMS_KV_DEBUG`-gated and lives in the "debug-only" commit.

| Probe | Where | Answers |
|---|---|---|
| **Stable-point fingerprint** | top of `GMSWorker.sleep()` | The winner's KV *after all writes, before any unmap*. Dumps `allocation_id` + a position-sensitive fingerprint per layer to `GMS_KV_TARGET_FILE`. |
| **Reattach byte-compare** | shadow wake, post-remap, **pre-forward** | Does the standby's reattached tensor hold the winner's exact bytes at exact positions? Reports `N/28 layers byte-identical` and localizes the first divergent chunk. |
| **D1 (tensor binding)** | both sides | Does `kv_caches[i].data_ptr()` actually sit inside a GMS mapping (`off`, `handle`), or did it escape to plain torch memory? |
| **Fresh import** | both sides | Re-export/re-import the same handle at a **brand-new VA** and read it, bypassing the engine's own mapping. Catches "same VA, different physical". |
| **Identity log** | both sides | `idx / layout_slot / allocation_id` per layer, to detect positional cross-wiring. |
| **Server export log** | `allocations.py` | Which `cuMem` handle the server hands to each client. |

### `kv_fingerprint` — read this before comparing anything

A whole-tensor `abs().sum()` is **permutation-invariant** and, on a KV pool, ~97% of it is
uninitialized garbage (a 3.8k-token prompt writes ~240 of 8192 blocks; `cuMemCreate` does not
zero). Two consequences that already cost us a wrong root cause (see below):

- Comparing abssum across **different** allocations is meaningless — the garbage differs.
- Comparing abssum across the **same** allocation is valid (identical garbage both sides), so
  it is a fine *sharing* check but a useless *correctness* check.

`kv_fingerprint()` splits the flat tensor into 1024 ordered chunks, abs-sums each, and hashes
the **ordered** vector. Any permutation or offset changes the hash, and the first divergent
chunk localizes the damage.

---

## Findings

Chronological, including the dead ends — the disproof trail is the point.

### Ruled out, each by direct measurement

- **Pairing / cross-wire.** Winner and shadow bind identical `allocation_id` per layer.
- **Tensor binding.** Shadow's `kv_caches[i].data_ptr()` == a remapped VA, `off=0`, `handle != 0`.
- **Sync / unmap / commit / publication barrier.** No combination changes the outcome;
  `vmm.synchronize()` is `cuCtxSynchronize`, which the forward pass already does.
- **Crash-with-mapping.** `CLEAN_HANDOFF=1` (clean unmap before the kill) behaves identically,
  so SIGKILL-while-mapped is not the corrupter.
- **GMS itself.** `poc/gms_ipc_test.py` — 28 × 512 MiB, scratch → reallocate → remap writer that
  hangs with its session open and mapped, then gets SIGKILLed, with a scratch-reattach reader,
  and even a colocated concurrent reader — **every variant passes, byte-perfect.**
- **Server logic.** The server exports the *same* `cuMem` handle to both winner and shadow.

### ⚠️ Refuted: "cross-process VMM sharing fails between the EngineCore processes"

**We believed this for a while. It is wrong. Do not re-chase it.**

It rested on "winner writes abssum ≈ 83k, shadow reads ≈ 17M for the same `allocation_id`."
Both halves of that comparison were broken:

1. **The 83k was read too early.** It came from inside `BlockPool.cache_full_blocks`, which the
   scheduler runs at *schedule* time — **before** that step's forward pass writes the KV. It
   measured a pre-write partial (mostly zeros), not the winner's KV. Moving the read to the
   stable point (top of `sleep()`) gives ≈ 17–24M, the same order as the shadow.
2. **The remainder was garbage contamination**, per the `kv_fingerprint` note above — the two
   numbers came from *different fresh allocations*, so they were never comparable.

Disproof: `poc/test_ext_read.sh` — a shadow engine writes KV, sleeps, and a fully independent
minimal GMS client (no vLLM) imports the same allocation at a fresh VA:

```
engine model-tensor  = 17,384,260
same-proc fresh read = 17,384,260
EXTERNAL process     = 17,384,260   → MATCH
```

Cross-process sharing works.

### The reattach is byte-perfect

Real 2-engine failover, `CLEAN_HANDOFF=1`. Winner fingerprinted at its stable point; standby
re-fingerprinted after remap and **before any forward pass**:

```
winner L0 (stable):     alloc=06b720f6  vec_sha=7a763fa30cdd4ce4  abssum=17,451,918.8
shadow L0 (reattached): alloc=06b720f6  vec_sha=7a763fa30cdd4ce4  abssum=17,451,918.8
MATCH=True   first_divergent_chunk = -1/1024
```

Same allocation, identical position-sensitive hash across all 1024 chunks.

> **Pending:** the all-layer version of this compare (`N/28 layers byte-identical`) was launched
> but the result was not captured before the session ended. It is wired up and runs by default
> under `GMS_KV_DEBUG` + `GMS_KV_TARGET_FILE` — grep `REATTACH BYTE-COMPARE (ALL LAYERS)` in the
> standby log. **Anyone picking this up should read that number first**; it decides open lead #1.

---

## Where we are stuck

Everything in the chain measures healthy, and the output is still wrong:

| Signal | Value | Reading |
|---|---|---|
| Winner output (cold / hot) | `Section 241: the quick` | The winner's KV is *good* KV |
| Reattach byte-compare (L0) | `MATCH=True`, 1024/1024 chunks | Standby holds the winner's exact bytes |
| Layers non-zero after reattach | 28/28 | Nothing is empty |
| `prefix_hits` (winner → loser) | 4688 → 4688 | The HIT is real, not a miss in disguise |
| Post-failover TTFT | 29 ms (vs 129 ms cold, 28 ms hot) | Prefill genuinely skipped |
| Index entries rehydrated | 293 of 294 records | Index replay works |
| **Post-failover output** | **`2400000`** | **Garbage** |

Correct KV bytes + byte-perfect reattach + a real prefix HIT that reads them → still garbage.
That rules out the sharing and reattach layers entirely. **The defect is downstream of the
bytes** — in how the reattached-and-rehydrated engine *uses* those blocks.

---

## Open leads

**1. A cross-wired layer.** Only L0 is byte-verified so far. If any of the other 27 layers is
paired to the wrong allocation, that alone produces garbage, and a permutation-invariant abssum
would not catch it. The all-layer probe is wired and pending (above). *If it returns 28/28, this
lead is dead and #2 becomes primary.*

**2. Rehydrated blocks overwritten before the re-send.** Rehydrate re-inserts prefix hashes but
leaves those blocks at `ref_cnt=0` — cached-but-free, therefore allocatable. Between wake and
the warm-prefix re-send, the harness runs a short unrelated health inference ("capital of
France"). If the allocator hands it block_ids overlapping the rehydrated prefix blocks, that
KV is overwritten in place and the later HIT reads corrupted blocks. Suggestive: replay
persisted **294** records but installed **293** — one block was already held.

*Cheap decisive test:* drop the intervening inference so the warm-prefix re-send is the first
thing the standby serves. If the garbage clears, this is it. A real fix would pin/refcount
rehydrated blocks rather than reorder the harness.

**3. Continuation-state mismatch.** If both above come back clean, suspect what the forward pass
assumes beyond the KV blocks themselves for a resumed sequence.

---

## Gotchas and parked items

- **The debug fresh-import probe leaks a VA mapping** (reserves + maps, never unmaps). Harmless
  in the failover flow, but it can break a *same-process* wake — Test 1's post-wake leg fails
  while the probe is enabled. Debug-gated only. Clean this up before it bites.
- **`--num-gpu-blocks-override 8192` on both engines is load-bearing.** Two colocated engines
  profile slightly different `num_blocks` (69.29 vs 69.23 GiB), so their layouts differ and
  remap's size check rejects the reattach. Pinning makes the geometry line up.
- **SIGTERM to `dynamo.vllm` hangs** in graceful shutdown ("disconnect backend services") on
  main @ vLLM 0.26, so it never releases the `flock` and the standby can never wake. The harness
  SIGKILLs the whole process tree instead — which also matches the POC's crash model. **The
  graceful-shutdown regression is parked, not fixed**, and deserves its own investigation.
- **Killing only the parent strands the EngineCore**, holding ~70 GiB of KV and a live GMS
  connection, which stalls the standby's realloc past its timeout. Kill the tree.
- **`on_connect(RW)` used to clear the layout unconditionally**, which wiped the persisted
  allocations the moment the standby connected. Now gated under `persist_on_abort` (adopt path).

---

## Layout

| Path | What |
|---|---|
| `lib/gpu_memory_service/integrations/vllm/kv_index_persist.py` | M2/M3: index write-through + rehydrate. Also hosts `kv_fingerprint` / the compare helpers (debug). |
| `lib/gpu_memory_service/integrations/vllm/worker.py` | Wake/reattach branch, sleep-path fingerprint, instrumentation. |
| `lib/gpu_memory_service/server/allocations.py` | Server-side export logging (debug). |
| `poc/harness.sh` | The 2-engine North Star harness. |
| `poc/test1_sleep_wake.sh` | Single-engine sleep/wake control. Green. |
| `poc/test_ext_read.sh` + `poc/gms_ext_read.py` | Engine writes / independent client reads. |
| `poc/gms_ipc_test.py` + `poc/gms_ipc_matrix.sh` | Pure two-process GMS byte-sharing matrix, no vLLM. |
| `poc/pod.yaml` | Dev pod spec. |

Commits are layered so they can be separated: product changes, debug instrumentation, and
harnesses are each their own commit. Only the product commits are PR candidates, and they need
the instrumentation stripped first.
