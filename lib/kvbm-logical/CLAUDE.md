# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Build (from repo root or this directory)
cargo build -p kvbm-logical

# Run all tests (263 tests)
cargo test -p kvbm-logical --lib

# Run a single test
cargo test -p kvbm-logical --lib test_name

# Run tests in a specific module
cargo test -p kvbm-logical --lib registry::tests

# Lint
cargo clippy -p kvbm-logical --lib

# Build with test utilities exposed (for downstream crates)
cargo build -p kvbm-logical --features testing
```

## Architecture

**kvbm-logical** is the core logical block lifecycle manager for KVBM (KV Block Manager). It manages KV cache blocks for LLM inference through a type-safe state machine, registry, and pool system.

### Block Lifecycle (Type-State Pattern)

Blocks use compile-time type states to enforce valid transitions:

```
MutableBlock<T>  →  CompleteBlock<T>  →  ImmutableBlock<T>  →  WeakBlock<T>
   (Reset)             (Staged)            (Registered)        (Non-owning)
```

- **MutableBlock**: Allocated from `ResetPool`, writable. Drop returns to reset pool.
- **CompleteBlock**: Staged via `stage()`/`complete()`. Drop returns to reset pool.
- **ImmutableBlock**: Registered in the registry. Strong Arc reference prevents eviction. Drop moves to inactive pool.
- **WeakBlock**: Non-owning reference. Does not prevent eviction. Two-stage upgrade (fast Weak::upgrade + slow pool search fallback).

The type parameter `T: BlockMetadata` is a marker for the storage tier (G1=GPU, G2=CPU, G3=Disk, G4=External).

### Module Structure

- **`manager/`** — `BlockManager<T>`: Top-level orchestrator. Entry point for allocating, registering, matching, and evicting blocks. Uses builder pattern (`BlockManagerConfigBuilder`).
- **`blocks/`** — RAII guard types for each lifecycle state. All guards auto-return blocks to the correct pool on drop.
- **`registry/`** — `BlockRegistry`: Tracks registered blocks by `SequenceHash` in a `PositionalRadixTree`. Supports typed attachments, presence markers, and touch callbacks. Optional TinyLFU frequency tracking.
- **`pools/`** — Three-tier pool system:
  - `ResetPool<T>`: Free blocks (FIFO)
  - `ActivePool<T>`: In-use registered blocks
  - `InactivePool<T>`: Cached evictable blocks with pluggable backends
- **`pools/inactive/backends/`** — Eviction strategies implementing `InactivePoolBackend<T>`: `HashMapBackend`, `LruBackend`, `MultiLruBackend` (4-tier frequency-aware), `LineageBackend` (parent-chain aware).
- **`events/`** — Block event pipeline with configurable emission policies, batching, and broadcast channels.
- **`metrics/`** — Prometheus metrics with atomic counters (`BlockPoolMetrics`) and optional periodic sampling (`StatsCollector`).
- **`tinylfu.rs`** — Count-Min Sketch frequency tracker (4 hash functions, configurable decay).
- **`testing/`** — Test utilities (behind `testing` feature flag): `TestBlockBuilder`, `BlockSequenceBuilder`, `create_test_manager()`.

### Key Design Decisions

- **Synchronous core with interior mutability**: Pool operations use `parking_lot` locks, no async channels. RAII returns execute inline.
- **Registry uses weak references**: `PositionalRadixTree<Weak<BlockRegistrationHandleInner>>`. Entries auto-clean when all strong refs drop.
- **Attachment system**: Extensible typed metadata on `BlockRegistrationHandle` via `attach_unique<T>()`/`attach<T>()` — no struct modification needed.
- **`docs/advancements.md`** contains the detailed design doc comparing v1 vs kvbm-logical architecture.

## Testing

Tests use `rstest` for fixtures and `proptest` for property-based testing (`pools/block_proptest.rs`). Test utilities in `src/testing/` are gated behind the `testing` feature flag and are also available to downstream crates.
