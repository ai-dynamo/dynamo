# Worker Set Routing Implementation

## Problem Statement

During rollouts, the frontend's PushRouter uses a single Client bound to a specific namespace (e.g., `tm-vllm-agg-qwen-3-a46fe6f1`). When new workers deploy with a new hash (e.g., `tm-vllm-agg-qwen-3-8a4cb500`) and old workers are removed, the Client can't find any instances, causing routing failures.

## Key Design Questions

### Q: Do we need prefix-based discovery?
**A: No.** The ModelWatcher already uses `DiscoveryQuery::AllModels` which discovers workers across all namespaces. The NamespaceFilter determines which to accept.

### Q: Will we have the same stale Client issue?
**A: No.** Instead of PushRouter holding a single fixed Client, it will query WorkerSetManager on each request. WorkerSetManager maintains multiple WorkerSets (one per namespace/hash), each with its own Client. When old workers are removed, the old WorkerSet is removed from the manager.

### Q: Should we conditionally use WorkerSets (only for KV mode or multi-MDC)?
**A: No.** Always use two-level selection for consistency:
1. Select WorkerSet (weighted by worker count)
2. Select worker within set (random/round-robin/KV-aware)

This provides consistent behavior, correct MDC handling, and natural rollout traffic distribution.

## Core Abstractions

### WorkerSet
- Represents workers deployed together from the same configuration
- Identified by namespace (which includes deployment hash)
- Has consistent MDC checksum
- **Owns a Client** for its specific namespace
- Provides worker selection methods (random, round-robin)

### WorkerSetManager
- Groups workers by namespace into WorkerSets
- Creates WorkerSets on-demand as workers register
- Provides weighted selection (70% to 7-worker set, 30% to 3-worker set)
- Automatically removes empty sets

### Routing Pattern (All Modes)
```
Request → Select WorkerSet (weighted) → Select Worker (mode-specific) → Route
```

## Edge Cases

### 1. KV Cache Cold Start
During rollout, new workers have empty caches while old workers have warm caches. Without weighted set selection, KV router would always pick old workers (better cache hits), starving new workers.

**Solution:** Weighted set selection forces traffic to new workers proportional to their count, allowing cache warmup.

### 2. Different MDC Checksums
Workers in different sets may have different model configs (tokenizer, templates, etc.). Using wrong MDC for preprocessing causes incorrect outputs.

**Solution:** Each WorkerSet tracks its mdcsum. Preprocessing uses the correct MDC for the target set. (Per-MDC engine registration to be addressed in watcher.rs)

### 3. Rollout Complete, Old Set Empty
When all old workers are deleted, WorkerSetManager should remove the empty set.

**Solution:** WorkerSetManager.remove_worker() auto-removes empty sets (existing behavior, inherited from MultiPoolManager).

### 4. Instance Address Resolution
PushRouter.generate_with_fault_detection() needs instance details (transport, address) from Client.instances(). Must find correct WorkerSet for a given instance_id.

**Solution:** Add WorkerSetManager.find_set_for_instance() to locate the set containing an instance.

## Implementation Plan

### Phase 1: Rename and Restructure ✅
- [x] Rename `multi_pool_manager.rs` → `worker_set_manager.rs`
- [x] Rename `worker_pool.rs` → `worker_set.rs`
- [x] Rename classes: `MultiPoolManager` → `WorkerSetManager`, `WorkerPool` → `WorkerSet`
- [x] Update all imports and references

### Phase 2: WorkerSet with Client ✅
- [x] Add `client: Client` field to WorkerSet
- [x] Add `new()` async method to create Client for the set's namespace
- [x] Add `instance_ids()`, `select_random()`, `select_round_robin()` methods
- [x] Worker count now derived from client.instance_ids() (live data)

### Phase 3: WorkerSetManager Updates ✅
- [x] Update `add_worker()` to create WorkerSets with Clients (now async)
- [x] Add `find_set_for_instance(instance_id) -> Option<Arc<WorkerSet>>`
- [x] Pass DistributedRuntime to manager for Client creation

### Phase 4: PushRouter Integration ✓
- [ ] Replace `client: Client` with `worker_set_manager: Arc<WorkerSetManager>`
- [ ] Update `select_next_worker()` to use set selection
- [ ] Update `generate_with_fault_detection()` to find set for instance
- [ ] Update `from_client()` constructors to accept WorkerSetManager

### Phase 5: Watcher Integration ✓
- [ ] Pass DistributedRuntime to `add_worker()` calls
- [ ] Fix MDC per-checksum engine registration (future: track engines by checksum)

### Phase 6: KV Router Updates ✓
- [ ] Update to use WorkerSetManager/WorkerSet naming
- [ ] Verify pool-aware scheduling still works

### Phase 7: Testing ✓
- [ ] Test rollout scenario (old → new workers)
- [ ] Test KV routing with multiple sets
- [ ] Test Random/RoundRobin modes
- [ ] Verify metrics/observability

## Implementation Status

**Current Phase:** Phase 4 - PushRouter Integration

**Blockers:** None

**Notes:**
- Discovery already handles cross-namespace worker finding (AllModels query)
- Client instance discovery remains namespace-specific (correct behavior)
- Per-MDC preprocessing will be addressed incrementally (not blocking)
