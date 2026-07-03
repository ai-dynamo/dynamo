# Mutable model state and GMS memory classes

Status: near-term fix implemented (publisher rebind in `finalize_gms_write`);
medium-term plan (misc pool + KV semantics convergence) not yet scheduled.

## Problem

The publisher builds the entire model inside the GMS `weights` memory pool,
so the committed allocations contain more than parameters: torch buffers
(`expert_map`, Mamba `conv_weights`, ...) and tensor attributes (fp8 KV
scales `_k_scale`/`_v_scale`/`_q_scale`/`_prob_scale`, quantization ranges)
are swept into the same caching-allocator segments. That sweep is
intentional — a fresh importer never runs `load_weights`, so everything the
forward pass references must be materializable from GMS.

The bug is that import semantics are uniform: after publish (and after any
RO reconnect — wake, restore), the whole tag is remapped
`CU_MEM_ACCESS_FLAGS_PROT_READ`. Some captured tensors are written *after*
load:

- `post_kv_cache_wake_up` → `init_fp8_kv_scales` rewrites the fp8 scale
  attrs on every wake (any `--kv-cache-dtype fp8` model);
- fp8 range attrs are updated on the forward path.

A write through a `PROT_READ` mapping is a fatal MMU fault
(NVRM Xid 31 `FAULT_RO_VIOLATION`). Cold engines only avoid it by timing:
every write lands inside the RW load window, and the late writers first run
after a sleep/wake — or after a snapshot restore, which is how this was
found (Nemotron NVFP4-MoE, fp8 KV, DP8).

## Current lifecycle classes

| tag | lifecycle | sleep | wake/restore |
|---|---|---|---|
| `weights` | committed-immutable | unmap (bytes live server-side) | remap same VAs, `PROT_READ` |
| `kv_cache` | ephemeral RW | unmap + abort | fresh physical at preserved VAs |

Mutable-durable state (the scale/range attrs and, conservatively, all
non-parameter tensors) fits neither class.

## Near-term fix: publisher rebind (implemented)

`materialize_module_from_gms` already gives importers the right binding:
parameters bind to the shared read-only mapping; buffers/tensor attrs are
`detach().clone()`d into ordinary CUDA memory. The fix makes the publisher
adopt the same rule: at the end of `finalize_gms_write` (after
`register_module_tensors` records the GMS copies for importers, and after
the RO remap), `rebind_nonparameter_tensors` re-binds every GMS-resident
non-parameter tensor to a private clone.

Consequences:

- late writers hit ordinary CUDA memory — preserved byte-exact and
  VA-stable by cuda-checkpoint/CRIU in the snapshot flow;
- the weights tag can be honestly `PROT_READ` for every consumer,
  including snapshot-restored publishers;
- the server's committed copy can never be mutated;
- the GMS copies stay registered, so importer materialization is unchanged.

Constraints:

- the rebind must run before CUDA graph capture (clones live at new
  addresses; captured graphs bake raw pointers);
- classification is deliberately blanket — *every* non-parameter is treated
  as mutable, because proving immutability per-buffer is impractical. The
  cost is the clone bytes, logged at finalize as
  `rebound X MiB of non-parameter tensors to private memory` to keep the
  duplication visible per model.

## Medium-term plan: `misc` pool and KV semantics convergence

Rather than keeping mutable state as process-private clones forever, the
plan is a third pool (`misc`) for non-parameter tensors, paired with a
change to KV semantics so it is *not* a third lifecycle:

- **Converged mutable class**: server-side allocations whose backing is
  *reused* across sleep/wake instead of aborted and reallocated. Contents
  are best-effort on failure (KV contents are disposable after a failure
  anyway; misc contents are re-derivable: scales/ranges are recomputed at
  wake, load-time constants are re-materializable from the artifact).
  `kv_cache` moves to these semantics too — same VAs, same physical
  backing, no realloc churn on wake.
- **Weights pool becomes parameters-only** at allocation time (misc pool
  active for non-parameter creation) or via a post-load dense repack
  (packed-slabs pattern), which also drops dead load-transient blocks from
  the artifact.
- **Importer/shadow dedup**: shadows import misc read-only sources and keep
  private working copies only where they write.

Open design questions to resolve before building:

1. **Snapshot capture point**: the saver copies committed state at
   load-commit time. Mutable misc contents at dump time differ from
   load-time values. Either the contract is "restore = load-time values +
   recompute at wake" (holds for the known set), or the saver needs a
   second, checkpoint-barrier-time pass over misc.
2. **Multi-consumer isolation**: misc is written per-engine; concurrent
   engines (shadow failover) must not share writable physical pages —
   copy-on-import or per-engine instances are required.
3. **Commit/staleness semantics**: misc is uncommitted-mutable; the layout
   hash and `StaleMemoryLayoutError` rules need a class-aware definition.
4. **Placement**: non-parameter tensors are created inside model
   `__init__`/load, interleaved with parameters under one mempool context;
   routing them to misc at creation needs either a tag-switching hook or a
   post-load migration pass (which must also complete before graph
   capture).

## References

- Fault forensics: NVRM Xid 31 `FAULT_RO_VIOLATION ACCESS_TYPE_VIRT_WRITE`
  on all ranks at first post-restore wake/forward; fault addresses at the
  final 4 KiB page of 2 MiB small-pool segments (fp8 scale blocks at the
  segment tail).
- Code: `client/torch/module.py` (`rebind_nonparameter_tensors`,
  `materialize_module_from_gms`), `integrations/common/utils.py`
  (`finalize_gms_write`), `client/memory_manager.py` (VA-stable
  unmap/remap, `cumem_set_access`).
