# PR 2 review guide — carry the prefix-cache index across a failover

PR: https://github.com/ai-dynamo/dynamo/pull/12767 (stacked on #12648)
Branch: `gms-kv-prefix-index` · +706/−1 · 4 files

Not part of the PR. Scratch doc for review.

---

## What it is for, in one paragraph

PR 1 makes the KV **bytes** outlive an engine. A cache hit also needs the **index** —
`block_hash → block_id` — which lives in the scheduler's `BlockPool` in process RAM and
cannot be recomputed from the bytes, because the hash is over token ids rather than
content. So with PR 1 alone the standby adopts correct KV it cannot find and re-prefills
anyway. This carries the index the same way the layout is carried: written by the active
engine, read by whoever adopts next.

---

## The one idea to hold onto

The two directions of staleness are **not symmetric**, and the whole design is that
asymmetry:

| between the last capture and the crash | the capture says | consequence |
|---|---|---|
| a block was **cached** | absent | MISS → recompute. Fail-safe |
| a block was **evicted and reused** | still present | **HIT on overwritten memory** |

So additions may wait; deletions may not. Additions ride a periodic snapshot, deletions
are streamed as they happen. Every other design choice falls out of this.

---

## Files

| file | lines | what |
|---|---|---|
| `integrations/vllm/kv_index_persist.py` | +425 | all the logic |
| `integrations/vllm/worker.py` | +24 | report `(adopted, layout_hash)` per rank |
| `setup.py` | +11 | the `vllm.general_plugins` entry point |
| `tests/test_kv_index_persist.py` | +247 | 11 unit tests |

---

## Where things live at runtime

```
┌─ EngineCore process ─────────────────────────────┐
│   Scheduler → KVCacheManager → BlockPool         │ ← the INDEX. one per engine.
│                                                  │   all three hooks install here
└──────────────────┬───────────────────────────────┘
                   │ collective_rpc
        ┌──────────┴──────────┐
   ┌────▼─────┐          ┌────▼─────┐
   │ rank 0   │          │ rank 1   │                ← the BYTES. one set per GPU.
   │ GMS+ctx  │          │ GMS+ctx  │
   └──────────┘          └──────────┘
```

**Index is engine-wide, bytes are per-rank.** That asymmetry is why the log does not
scale with GPU count (one snapshot regardless of TP) and why the replay must be
all-or-nothing across ranks.

---

## Flow A — active engine captures

```
╔═ every step ═════════════════════════════════════════════════════════════╗
║  Scheduler.update_from_output returns          [patch at :390]           ║
║    └─► maybe_snapshot(scheduler)                          :163           ║
║          ├─ idle = not scheduler.has_requests()                          ║
║          ├─ skip unless idle OR interval elapsed (default 1s)            ║
║          └─► write_snapshot(pool, layout_id, path)        :75            ║
║                ├─ walk BlockPool.blocks  ← PUBLIC list, no private access║
║                ├─ keep those with block_hash is not None                 ║
║                ├─ write temp file, fsync                                 ║
║                └─ os.replace()  ← atomic; readers never see a tear       ║
║          └─ truncate the deletion list (superseded by this snapshot)     ║
╚══════════════════════════════════════════════════════════════════════════╝

╔═ on every eviction ══════════════════════════════════════════════════════╗
║  BlockPool._maybe_evict_cached_block           [patch at :358]           ║
║    ├─ read the hashes BEFORE delegating (the original resets them)       ║
║    ├─ evicted = orig_evict(...)                                          ║
║    └─ if evicted: append_deletions(hashes)                :129           ║
║         published immediately, NOT batched — this is the direction       ║
║         where lateness is a correctness bug                              ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Why the step barrier.** `cache_full_blocks` runs at *schedule* time, before the forward
pass writes that step's KV. Capturing there would name blocks whose bytes were never
written. `update_from_output` runs once the step's model output exists, and the KV writes
precede the sampler on the same CUDA stream, so having the output in hand means the
writes landed. No explicit GPU sync needed.

**Why the quiescence flush.** Snapshots ride on steps, and steps only happen while there
is work. An engine that caches a prefix and then idles would never persist it — which is
exactly the state a failover finds it in. Found by the first e2e returning a 0-entry
snapshot; not theoretical.

---

## Flow B — standby replays

```
╔═ EngineCore.resume_scheduler ═══════════════════ [patch at :407] ════════╗
║  (called from EngineCore.wake_up, AFTER model_executor.wake_up returned) ║
║                                                                          ║
║  replay_after_wake(engine_core)                            :280          ║
║    ├─► collective_rpc("gms_kv_takeover_state")                           ║
║    │      → [(adopted, layout_hash), ...] one per rank                   ║
║    │        worker.py:399                                                ║
║    │                                                                     ║
║    ├─ all(adopted)?  ── no ──► return False, DO NOT replay               ║
║    │     the index is engine-wide, the bytes are per-rank; a partial     ║
║    │     adoption is correct for some shards and garbage for one         ║
║    │                                                                     ║
║    ├─ layout_id = "|".join(hashes)   ← identity of the adopted pages     ║
║    │                                                                     ║
║    └─► replay_index(block_pool, layout_id)                 :210          ║
║          ├─ read_snapshot()                                              ║
║          ├─ snapshot layout_id != ours? ──► REFUSE                       ║
║          ├─ deleted = read_deletions()                                   ║
║          ├─ for each record not in deleted:                              ║
║          │     skip null block / out of range                            ║
║          │     skip if block already hashed or ref_cnt != 0              ║
║          │     _insert_block_hash(...)                                   ║
║          └─► _requeue_to_tail(...)                          :255         ║
║                                                                          ║
║  then orig_resume_scheduler()  ← scheduling resumes only now             ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Why `resume_scheduler` and not `wake_up`.** Three constraints, one place satisfying all:
runs in the process owning the `BlockPool`; runs after `model_executor.wake_up()` (a
blocking collective, so every rank has reattached); runs before scheduling resumes. The
earlier draft hooked `wake_up` and ran *after* the original returned — and
`resume_scheduler()` is the last thing that function does, so the replay happened after
the scheduler was resumed. It was safe only because `wake_up` is handled synchronously on
the engine loop, which is an inherited guarantee rather than one we own.

**Why the requeue.** vLLM keeps "hand out uncached blocks first" purely by queue
*position*: `free_blocks` prepends uncached and appends cached, and `get_new_blocks` pops
the head. Marking a block cached does not move it. On a freshly woken standby every block
is unhashed and queued in id order, so the reused prefix sits at the head and the next
request served is handed its blocks and overwrites them. **This was the original
garbage-output bug.** Descending block id, so the prefix's *last* blocks are evicted first
and a usable leading prefix survives.

---

## Installation

```
setup.py  →  entry_points["vllm.general_plugins"]
                    │
                    ▼  vLLM calls this from EngineCore.__init__
     enable_kv_index_persistence()                        :338
        ├─ returns immediately unless GMS_KV_INDEX_PATH is set
        ├─ _patch_eviction()      BlockPool._maybe_evict_cached_block
        ├─ _patch_step_barrier()  Scheduler.update_from_output
        └─ _patch_wake()          EngineCore.resume_scheduler
```

Importing the GMS worker module is **not** enough: at TP>1 `WorkerProc.__init__` resolves
`worker_cls` in the *child*, so the scheduler's process never imports it and the whole
mechanism is silently absent. See the open question below on whether this should be an
entry point at all.

---

## Review checklist, roughly in order of how much I'd like eyes on it

1. **`maybe_snapshot` :163** — the idle/interval gate. Is quiescence the right forcing
   condition, and is 1s a sane default?
2. **`replay_after_wake` :280** — the `all(adopted)` gate and the layout-id construction.
   Is joining per-rank hashes the right identity?
3. **`_patch_eviction` :358** — read-hashes-before-delegate ordering. Wrong order silently
   records nothing.
4. **`_requeue_to_tail` :255** — depends on a vLLM invariant that is not written down
   anywhere in vLLM.
5. **`write_snapshot` :75** — temp+fsync+rename. Is fsync per snapshot too heavy at 1s?
6. **worker.py :510** — the hash is read *after* `commit_layout()`, so a freshly sealed
   layout reports the hash it was just given. Ordering matters here.

---

## Known gaps, stated rather than buried

- **The DEL path has no e2e coverage.** It only matters for a crash *mid-burst*, before
  the next snapshot or idle flush. Unit tests cover it; provoking it deterministically
  needs crash injection at a chosen point.
- **Three monkey-patches, two against private API** (`_maybe_evict_cached_block`,
  `_insert_block_hash`). No version guard. Will need attention on vLLM upgrades.
- **Snapshot cost is O(num_gpu_blocks)**, not O(cached). Fine at 8k–40k; a much larger
  pool would want delta encoding.
- **`PYTHONHASHSEED` must be pinned identically across engines.** Fail-safe (degrades to
  MISS) but silently removes the entire benefit.
- **No e2e test ships with the PR.** Validation used `poc/harness.sh`, which is excluded.
  `tests/gpu_memory_service/test_shadow_failover.py` is the established home for one and
  currently asserts nothing about prefix reuse.

---

## Open question for the reviewer

`scheduler_cls` is a real integration hook and Dynamo already uses it
(`InstrumentedScheduler`, subclassing `AsyncScheduler`, injected in `args.py:320`). We are
*not* using it because it is a single-occupancy slot: claiming it would make KV failover
and forward-pass metrics mutually exclusive, and it cannot reach the wake hook anyway.
The alternative to the entry point is to have Dynamo's own scheduler call
`enable_kv_index_persistence()`, which keeps everything inside Dynamo at the cost of only
firing when that scheduler is injected.

---

## Validation

**Unit** — 11 tests over a `FakeBlockPool`, no GPU. Both staleness directions,
layout-mismatch refusal, free-queue placement, in-use blocks left alone, torn deletion
tail, disabled-by-default. Suite: 39 passed, 2 pre-existing failures (they need real
export FDs and a subprocess holder; they fail identically on unmodified `main`).

**End to end** — 2×B200, `poc/harness.sh`, each **16/16 with output byte-identical to a
fresh prefill**:

| run | result |
|---|---|
| TP=1 | 294 entries replayed, `prefix_hits 4688 == winner`, full HIT |
| eviction pressure, 384 blocks | partial HIT (2144), correct — eviction retired blocks |
| TP=2 | full HIT |

**Snapshot cost**, measured on B200: 2.6 ms @ 8k blocks, 8.3 ms @ 40k, 30 ms @ 160k →
0.26%–3% of serving time at the 1s default.
