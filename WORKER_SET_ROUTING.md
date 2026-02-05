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

### Phase 4: PushRouter Integration ✅
**Approach:** Create WorkerSetPushRouter wrapper (similar to KvPushRouter)
- [x] Create `WorkerSetPushRouter` in llm crate
- [x] Wraps base PushRouter + WorkerSetManager
- [x] Implements AsyncEngine: selects set → selects worker → calls PushRouter.direct()
- [x] Update build_routed_pipeline() to use WorkerSetPushRouter in prefix mode

### Phase 5: Watcher Integration ✅
- [x] Pass DistributedRuntime to `add_worker()` calls
- [x] Made add_to_worker_set async to handle Client creation
- Note: MDC per-checksum engine registration is existing behavior (not blocking)

### Phase 6: KV Router Updates ✅
- [x] Updated to use WorkerSetManager/WorkerSet naming
- [x] Updated method names (set_weights, select_weighted)
- [x] Pool-aware scheduling logic unchanged (works with renamed classes)

### Phase 7: Testing
- [ ] Manual test: Deploy with rollout, verify traffic distribution
- [ ] Verify old namespace issue is fixed
- [ ] Test Random/RoundRobin modes work
- [ ] Check metrics show set distribution

## Implementation Status

**Current Phase:** Complete - Ready for Testing

**Blockers:** None

**Implementation Summary:**

The fix involved 4 main phases:
1. **Renamed** MultiPoolManager→WorkerSetManager, WorkerPool→WorkerSet for clearer semantics
2. **Added Client to WorkerSet**: Each set now owns a Client watching its specific namespace
3. **Updated WorkerSetManager**: Creates WorkerSets with Clients, provides set lookup methods
4. **Created WorkerSetPushRouter**: Wrapper that selects set→worker, routes via PushRouter.direct()

**Key Design Decisions:**
- WorkerSet owns a Client (not PushRouter) - fixes stale client issue
- WorkerSetPushRouter wraps PushRouter (can't modify runtime crate directly)
- Consistent two-level selection for all modes (set→worker)
- Discovery unchanged - ModelWatcher already finds all workers via AllModels

**Notes:**
- Discovery already handles cross-namespace worker finding (AllModels query)
- Client instance discovery remains namespace-specific (correct behavior)
- Per-MDC preprocessing exists but not enforced (separate concern)
- KV router already had multi-set support, now Random/RoundRobin do too
