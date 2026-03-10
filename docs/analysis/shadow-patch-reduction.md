# Shadow Patch Reduction Analysis

## Current State: 6 patches

1. `patch_initialize_kv_cache_tensors` — return `{}` during `_shadow_init_phase`
2. `patch_get_slot_mappings` — return `(None, None)` when `kv_caches` empty
3. `patch_request_memory` — bypass memory check
4. `patch_register_kv_caches` — skip NIXL registration when empty
5. `patch_allocate_kv_cache_on_wake` — add method to allocate on wake
6. `patch_cudagraph_mode_escalation` — prevent FULL mode in shadow

## Analysis: Which Are Actually Needed?

### patch_request_memory — NOT NEEDED

`request_memory()` only checks free GPU memory vs requested memory percentage.
It doesn't interact with KV cache at all. The shadow engine has enough free memory
for profiling (model weights are already loaded). The patch was added defensively
but the check never fails for shadow engines.

### patch_register_kv_caches — NOT NEEDED

`NixlConnector.register_kv_caches({})` with an empty dict is a no-op.
It doesn't crash. The loop iterates over an empty dict and does nothing.
The patch is defensive but unnecessary.

### patch_get_slot_mappings — NEEDED but can be avoided

`_get_slot_mappings` already has a natural guard (gpu_model_runner.py:3261-3266):

```python
if not (
    hasattr(self, "kv_cache_config")
    and self.kv_cache_config is not None
    and len(self.kv_cache_config.kv_cache_groups) > 0
):
    return None, None
```

If `self.kv_cache_config` is unset, it returns `(None, None)` naturally.
The problem is that `initialize_kv_cache()` sets `self.kv_cache_config` at line 6054
BEFORE calling `initialize_kv_cache_tensors`. So by the time graph capture runs,
the config exists and the guard doesn't trigger.

**Solution:** Don't set `self.kv_cache_config` during shadow init. Store it
elsewhere (e.g., `self._deferred_kv_cache_config`) and only set the real attribute
when KV cache is actually allocated on wake.

### patch_initialize_kv_cache_tensors — NEEDED (core behavior change)

This is the actual behavior we want: skip GPU memory allocation. Can't avoid this.

### patch_allocate_kv_cache_on_wake — NEEDED (core behavior change)

Need to allocate KV cache on wake using stored config. Can't avoid this.

### patch_cudagraph_mode_escalation — NEEDED for multinode only

vLLM escalates PIECEWISE to FULL_AND_PIECEWISE on the headless worker.
FULL mode requires slot_mappings which are None.

BUT: if we fix the slot_mappings issue by not setting `kv_cache_config` during
shadow init, `_get_slot_mappings` returns `(None, None)` naturally. In PIECEWISE
mode, `_build_attention_metadata` is never called (guarded by `force_attention`).
In FULL mode, `_build_attention_metadata` IS called and asserts `slot_mappings is not None`.

So FULL mode still crashes even with the natural `(None, None)` return. We still
need to prevent FULL mode captures when there's no KV cache.

## Proposed: 2 Patches

### Patch 1: `initialize_kv_cache` (replaces patches 1, 2, 3, 4)

One patch on `initialize_kv_cache` that during shadow init:
- Runs all the setup steps (attn backend, metadata builders, input batch)
- Skips `initialize_kv_cache_tensors` (no GPU allocation)
- Stores config in `_deferred_kv_cache_config` instead of `kv_cache_config`
- Registers empty KV caches with transfer group
- `self.kv_caches` stays empty, `self.kv_cache_config` stays unset
- `_get_slot_mappings` naturally returns `(None, None)` via existing guard

```python
def patched_initialize_kv_cache(self, kv_cache_config):
    if getattr(self, "_shadow_init_phase", False):
        # Run setup steps that don't need actual tensors
        kv_cache_config = deepcopy(kv_cache_config)
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)

        # Store config for wake, but DON'T set self.kv_cache_config
        # This makes _get_slot_mappings return (None, None) naturally
        self._deferred_kv_cache_config = kv_cache_config
        self._deferred_kernel_block_sizes = kernel_block_sizes

        # Register empty with transfer group
        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            kv_transfer_group.register_kv_caches({})

        return

    # Normal path unchanged
    original_initialize_kv_cache(self, kv_cache_config)
```

### Patch 2: `allocate_kv_cache_on_wake` (replaces patch 5)

Adds the wake method that uses the deferred config:

```python
def allocate_kv_cache_on_wake(self):
    config = self._deferred_kv_cache_config
    block_sizes = self._deferred_kernel_block_sizes

    # Now set the real kv_cache_config (enables _get_slot_mappings)
    self.kv_cache_config = config

    # Allocate and bind
    kv_caches = self.initialize_kv_cache_tensors(config, block_sizes)

    # Register with transfer group
    if has_kv_transfer_group():
        kv_transfer_group = get_kv_transfer_group()
        kv_transfer_group.register_kv_caches(kv_caches)

    return kv_caches
```

### What About cudagraph_mode Escalation? (patch 6)

With `kv_cache_config` unset during shadow init, `_get_slot_mappings` returns
`(None, None)`. In PIECEWISE mode, `_build_attention_metadata` is never called.
But in FULL mode it IS called and asserts `slot_mappings is not None`.

The question: does `_build_attention_metadata` even get called when
`slot_mappings_by_group` is `None`?

In `_dummy_run` at line 4788:
```python
if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
    attn_metadata, _ = self._build_attention_metadata(
        slot_mappings=slot_mappings_by_group,  # None
    )
```

Inside `_build_attention_metadata` at line 1738:
```python
assert slot_mappings is not None  # CRASH
```

So yes, FULL mode still crashes. We need to either:
a. Prevent FULL mode escalation (current approach — hotpatch in vLLM)
b. Add a guard: `if slot_mappings_by_group is None: skip attention metadata`
c. Accept that shadow mode requires PIECEWISE only

Option (b) is cleanest — add a condition in `_dummy_run`:

```python
if (force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL) and slot_mappings_by_group is not None:
    attn_metadata, _ = self._build_attention_metadata(...)
```

This is a one-line change in vLLM that's safe even for non-shadow engines
(slot_mappings_by_group is never None in normal operation). Could be upstreamed.

But for now, preventing FULL mode escalation is simpler and doesn't require
a vLLM change. It can be done inside patch 1 — clamp the cudagraph_mode
in `initialize_attn_backend` when in shadow init phase.

## Summary

**From 6 patches to 2 patches + 1 vLLM guard:**

| Before (6 patches) | After (2 patches) |
|---|---|
| patch_initialize_kv_cache_tensors | Patch 1: initialize_kv_cache |
| patch_get_slot_mappings | Eliminated (natural guard via unset kv_cache_config) |
| patch_request_memory | Eliminated (unnecessary) |
| patch_register_kv_caches | Folded into Patch 1 |
| patch_allocate_kv_cache_on_wake | Patch 2: allocate_kv_cache_on_wake |
| patch_cudagraph_mode_escalation | Folded into Patch 1 (clamp in initialize_attn_backend) |

The key insight: by NOT setting `self.kv_cache_config` during shadow init,
we leverage vLLM's existing `_get_slot_mappings` guard to return `(None, None)`,
which makes `set_forward_context` use an empty slot_mapping dict, which makes
`unified_kv_cache_update` skip KV writes. No separate patch needed.
