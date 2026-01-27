# KVBM Testing Infrastructure

This directory provides reusable test utilities for writing end-to-end tests of the KVBM block management system. Use these modules to construct test fixtures without reimplementing common setup patterns.

## Builder Pattern

Follow the project-wide `derive_builder` conventions from `/CLAUDE.md`:

```rust
#[derive(Builder)]
#[builder(setter(into), build_fn(error = "anyhow::Error"))]
pub struct MyConfig {
    #[builder(default)]
    field_a: Option<String>,
    #[builder(default = "1024")]
    field_b: usize,
}

impl MyConfig {
    pub fn builder() -> MyConfigBuilder {
        MyConfigBuilder::default()
    }
}
```

### Custom Build Functions

When `build()` requires async operations, complex validation, or needs to return a different type than the config struct, use `build_fn(skip)` and implement a custom build method:

```rust
#[derive(Builder)]
#[builder(
    setter(into, strip_option),
    build_fn(skip),  // Skip auto-generated build()
    pattern = "owned"
)]
pub struct EventsPipelineConfig {
    #[builder(default)]
    instance_id: Option<InstanceId>,

    #[builder(default = "Duration::from_millis(50)")]
    batching_window: Duration,
}

impl EventsPipelineConfigBuilder {
    /// Custom async build that creates the fixture.
    pub async fn build_async(self) -> Result<EventsPipelineFixture> {
        let instance_id = self.instance_id.flatten().unwrap_or_else(InstanceId::new_v4);
        let batching_window = self.batching_window.unwrap_or(Duration::from_millis(50));
        // ... async setup ...
        Ok(EventsPipelineFixture { /* ... */ })
    }
}

// Usage:
let fixture = EventsPipelineFixture::builder()
    .batching_window(Duration::from_millis(100))
    .build_async().await?;
```

Use custom build functions when:
- Build requires async operations (subscriptions, network setup)
- Build creates a different struct than the config (config → fixture pattern)
- Complex validation that can't be expressed in `build_fn(validate)`
- Build needs to call fallible operations that aren't just field validation

## Module Overview

| Module | Purpose |
|--------|---------|
| `token_blocks` | Generate token blocks and sequence hashes |
| `managers` | Create and populate BlockManagers, `MultiInstancePopulator` |
| `nova` | Create Nova instances for distributed testing |
| `physical` | Create physical layouts, `TestAgentBuilder`, `TransferChecksums` |
| `distributed` | Create InstanceLeaders with workers, `TestSession` |
| `connector` | High-level connector test fixtures |
| `scheduler` | Scheduler integration test utilities |
| `events` | `EventsPipelineFixture` for events testing |

## Backend Configuration

The `TestAgentBuilder` provides flexible NIXL backend handling for tests:

### Required vs Optional Backends

```rust
use crate::v2::testing::physical::TestAgentBuilder;

// RDMA tests - UCX is REQUIRED (fails if unavailable)
let agent = TestAgentBuilder::new("rdma-test")
    .require_backend("UCX")  // Fails build() if UCX unavailable
    .try_backend("POSIX")    // Optional - continues without it
    .build()?;

// Disk tests - POSIX only (no GDS requirement)
let agent = TestAgentBuilder::new("disk-test")
    .try_backend("POSIX")    // POSIX is always available
    .build()?;

// Flexible test - check what's available
let agent = TestAgentBuilder::new("flexible-test")
    .try_backend("UCX")
    .try_backend("POSIX")
    .build()?;

if !agent.has_backend("UCX") {
    eprintln!("Skipping RDMA test - UCX unavailable");
    return Ok(());
}
```

### Backend Guidelines

| Test Type | Backend Strategy |
|-----------|-----------------|
| RDMA/UCX tests | `.require_backend("UCX")` |
| Disk I/O tests (G2↔G3) | `.try_backend("POSIX")` only |
| G1↔G3 direct (GPU Direct Storage) | `.try_backend("GDS_MT")` + skip if unavailable |
| General tests | Use `try_backend` and check availability |

### GDS_MT for G1↔G3 Direct Transfers

G1↔G3 direct transfers (GPU ↔ disk without CPU staging) require GDS_MT (GPU Direct Storage). Since GDS may not be available on all systems, tests using G1↔G3 direct should gracefully skip when unavailable:

```rust
let agent = TestAgentBuilder::new("g1-g3-test")
    .try_backend("GDS_MT")
    .try_backend("POSIX")  // Fallback for non-direct tests
    .build()?;

if !agent.has_backend("GDS_MT") {
    eprintln!("Skipping G1↔G3 direct test - GDS_MT unavailable");
    return Ok(());
}

// ... G1↔G3 direct transfer test ...
```

**Note:** G2↔G3 transfers (CPU-mediated) only require POSIX, which is always available.

## Reducing Test Boilerplate

### MultiInstancePopulator

Replace repeated slice→populate patterns:

```rust
use crate::v2::testing::managers::MultiInstancePopulator;

// BEFORE: 12+ lines per multi-instance test
let full_sequence = create_token_sequence(32, 16, 0);
let inst1_blocks: Vec<_> = full_sequence.blocks()[0..16].to_vec();
let inst1_hashes = populate_manager_with_blocks(&inst1.g2_manager(), &inst1_blocks)?;
let inst1_matched = inst1.g2_manager().match_blocks(&inst1_hashes);
let inst1_block_ids: Vec<_> = inst1_matched.into_iter().map(|b| b.block_id()).collect();
// ... repeat for inst2, inst3 ...

// AFTER: 5 lines total
let population = MultiInstancePopulator::new()
    .total_blocks(32)
    .block_size(16)
    .add_instance(&inst1.g2_manager(), 0..16)
    .add_instance(&inst2.g2_manager(), 16..24)
    .add_instance(&inst3.g2_manager(), 24..32)
    .build()?;

// Access results
let all_hashes = population.all_hashes();
let inst1_ids = population.instance_block_ids(0).unwrap();
let inst2_hashes = population.instance_hashes(1).unwrap();
```

### TestSession

Encapsulate session lifecycle:

```rust
use crate::v2::testing::distributed::TestSession;

// BEFORE: 6 lines repeated in many tests
let (session_id, handle) = leader.create_endpoint_session(&hashes)?;
let mut remote_handle = remote_leader.attach_session(instance_id, session_id).await?;
let state = timeout(Duration::from_secs(5), remote_handle.wait_for_ready())
    .await.expect("Timeout").expect("Ready");

// AFTER: 1 line
let session = TestSession::establish_default(&leader, &remote_leader, &hashes).await?;

// Access session state
let g2_blocks = session.g2_blocks();
let phase = session.phase();

// Execute operations
let notification = session.pull_blocks_rdma(&g2_blocks, &dst_ids).await?;
session.notify_layers_ready(0..num_layers).await?;

// Clean shutdown
session.close().await?;
```

### EventsPipelineFixture

Reduce events pipeline setup (uses custom `build_async()` for async construction):

```rust
use crate::v2::testing::events::EventsPipelineFixture;

// BEFORE: 15 lines of setup
let events_manager = Arc::new(EventsManager::builder().build());
let bus = StubBus::default();
let publisher = Arc::new(bus.publisher());
let subscriber = bus.subscriber();
let mut subscription = subscriber.subscribe("kvbm.events").await?;
let _publisher = KvbmCacheEventsPublisher::builder()
    .instance_id(12345)
    .event_stream(events_manager.subscribe())
    .publisher(publisher)
    .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
    .subject("kvbm.events")
    .build()?;

// AFTER: 3 lines (uses build_async() for async setup)
let mut fixture = EventsPipelineFixture::builder()
    .batching_window(Duration::from_millis(50))
    .build_async().await?;

// Create manager with events integration
let manager = fixture.create_manager::<G1>(100, 4);

// Receive events
let batch = fixture.flush_and_receive(Duration::from_millis(50)).await;
```

### TransferChecksums

Cleaner checksum verification:

```rust
use crate::v2::testing::physical::TransferChecksums;

// BEFORE
let src_checksums = fill_and_checksum(&src_layout, &src_ids, FillPattern::Sequential)?;
// ... execute transfer ...
verify_checksums_by_position(&src_checksums, &src_ids, &dst_layout, &dst_ids)?;

// AFTER
let src = TransferChecksums::fill_and_capture(&src_layout, &src_ids, FillPattern::Sequential)?;
// ... execute transfer ...
src.verify_against(&dst_layout, &dst_ids)?;
```

## Quick Start: Writing an E2E Test

### 1. Simple BlockManager Test

```rust
use crate::v2::testing::managers::TestManagerBuilder;
use crate::v2::testing::token_blocks;

#[tokio::test]
async fn test_block_registration() {
    // Create a G2 block manager with builder
    let manager = TestManagerBuilder::<G2>::new()
        .block_count(100)
        .block_size(4)
        .build();

    // Generate token sequence and populate
    let token_seq = token_blocks::create_token_sequence(10, 4, 0);
    let blocks = manager.allocate_blocks(10).unwrap();
    let complete_blocks: Vec<_> = blocks.into_iter()
        .zip(token_seq.blocks())
        .map(|(b, tb)| b.complete(tb.clone()).unwrap())
        .collect();
    let _registered = manager.register_blocks(complete_blocks);

    // Verify blocks can be matched
    let hashes: Vec<_> = token_seq.blocks().iter()
        .map(|b| b.kvbm_sequence_hash())
        .collect();
    let matched = manager.match_blocks(&hashes);
    assert_eq!(matched.len(), 10);
}
```

### 2. Distributed Leader Test (No Workers)

```rust
use crate::v2::testing::distributed::{create_instance_leader_pair, populate_leader_with_blocks};

#[tokio::test]
async fn test_remote_block_search() {
    // Create two connected leaders via Nova TCP
    let pair = create_instance_leader_pair(100, 4).await.unwrap();

    // Populate leader A with blocks
    let (_, hashes) = populate_leader_with_blocks(&pair.leader_a, 32, 4, 0).unwrap();

    // Leader B can search leader A remotely
    let result = pair.leader_b.leader.find_matches(&hashes).unwrap();
}
```

### 3. RDMA Transfer Test (With Workers)

```rust
use crate::v2::testing::{distributed, physical};
use dynamo_memory::StorageKind;

#[tokio::test(flavor = "multi_thread")]
async fn test_rdma_transfer() {
    let layout_config = physical::custom_config(64, 2, 2, 4, 64, 2);

    // Create decode/prefill pair with UCX workers
    let pair = distributed::create_instance_leader_pair_with_workers(
        64, 4, 2, &layout_config, StorageKind::Pinned
    ).await.unwrap();

    // Fill source blocks with test pattern
    for worker in &pair.decode.workers {
        worker.fill_g2_blocks(&[0, 1, 2], FillPattern::Sequential).unwrap();
    }

    // Execute RDMA transfer and verify checksums...
}
```

### 4. Events Pipeline Test

```rust
use crate::v2::testing::managers::TestManagerBuilder;
use crate::v2::testing::token_blocks;
use crate::v2::logical::events::{EventsManager, BatchingConfig, KvbmCacheEventsPublisher};
use crate::v2::distributed::pubsub::{StubBus, Subscriber};

#[tokio::test]
async fn test_events_pipeline() {
    // Create events manager (AllEventsPolicy is the default)
    let events_manager = Arc::new(EventsManager::builder().build());

    // Create stub pubsub for testing
    let bus = StubBus::default();
    let publisher = Arc::new(bus.publisher());
    let subscriber = bus.subscriber();

    // Subscribe BEFORE publishing (stub doesn't buffer for late subscribers)
    let mut sub = subscriber.subscribe("kvbm.events").await.unwrap();

    // Build publishing pipeline
    let _publisher = KvbmCacheEventsPublisher::builder()
        .instance_id(12345)
        .event_stream(events_manager.subscribe())
        .publisher(publisher)
        .batching_config(BatchingConfig::default().with_window(Duration::from_millis(50)))
        .subject("kvbm.events")
        .build()
        .unwrap();

    // Create block manager with events integration
    let manager = TestManagerBuilder::<G1>::new()
        .block_count(100)
        .block_size(4)
        .events_manager(events_manager.clone())
        .build();

    // Register blocks (triggers Create events)
    let token_seq = token_blocks::create_token_sequence(5, 4, 0);
    let blocks = manager.allocate_blocks(5).unwrap();
    let complete_blocks: Vec<_> = blocks.into_iter()
        .zip(token_seq.blocks())
        .map(|(b, tb)| b.complete(tb.clone()).unwrap())
        .collect();
    let _registered = manager.register_blocks(complete_blocks);

    // Wait for batch window and verify events received
    tokio::time::sleep(Duration::from_millis(100)).await;
    let msg = sub.next().await.unwrap();
    // Deserialize with rmp_serde::from_slice(&msg.payload)
}
```

## Module Reference

### `token_blocks` - Token Generation

```rust
// Create single token block
let block = create_token_block(&[1, 2, 3, 4]);
let block = create_sequential_block(100, 4);  // [100, 101, 102, 103]

// Create multi-block sequence
let sequence = create_token_sequence(32, 4, 0);  // 32 blocks, 4 tokens each
let hashes = generate_sequence_hashes(&sequence);

// Create disjoint sequences (for gap testing)
let (blocks, hashes) = create_disjoint_sequences(
    vec![(2, 0), (2, 100), (3, 200)],  // segments
    4  // block_size
);
```

### `managers` - BlockManager Setup

```rust
use crate::v2::testing::managers::{TestRegistryBuilder, TestManagerBuilder, MultiInstancePopulator};
use crate::v2::logical::manager::FrequencyTrackingCapacity;

// Simple block manager (creates registry internally)
let manager = TestManagerBuilder::<G2>::new()
    .block_count(100)
    .block_size(4)
    .build();

// With events integration via TestRegistryBuilder
let events_manager = Arc::new(EventsManager::builder().build());
let registry = TestRegistryBuilder::new()
    .events_manager(events_manager.clone())
    .build();
let manager = TestManagerBuilder::<G1>::new()
    .block_count(100)
    .block_size(4)
    .registry(registry)
    .build();

// Convenience: events_manager without explicit registry
let manager = TestManagerBuilder::<G1>::new()
    .block_count(100)
    .block_size(4)
    .events_manager(events_manager.clone())
    .build();

// With custom frequency tracking
let registry = TestRegistryBuilder::new()
    .frequency_tracking(FrequencyTrackingCapacity::Large)
    .build();
let manager = TestManagerBuilder::<G2>::new()
    .block_count(100)
    .block_size(4)
    .registry(registry)
    .build();

// Multi-instance population (see "Reducing Test Boilerplate" section)
let population = MultiInstancePopulator::new()
    .total_blocks(32)
    .block_size(16)
    .add_instance(&inst1.g2_manager(), 0..16)
    .add_instance(&inst2.g2_manager(), 16..32)
    .build()?;
```

### `physical` - Physical Layouts & Agents

```rust
use crate::v2::testing::physical::{TestAgentBuilder, TransferChecksums};

// Flexible agent with optional backends
let agent = TestAgentBuilder::new("test")
    .try_backend("UCX")
    .try_backend("POSIX")
    .build()?;

// RDMA agent with required UCX
let agent = TestAgentBuilder::new("rdma")
    .require_backend("UCX")
    .build()?;

// Check backend availability
if agent.has_backend("UCX") {
    // Run RDMA tests
}

// Transfer checksums
let src = TransferChecksums::fill_and_capture(&layout, &ids, FillPattern::Sequential)?;
// ... transfer ...
src.verify_against(&dst_layout, &dst_ids)?;
```

### `nova` - Distributed Communication

```rust
// Single Nova instance (TCP on random port)
let nova = create_nova_instance_tcp().await?;

// Connected Nova pair (bidirectional peers)
let NovaPair { nova_a, nova_b } = create_nova_pair_tcp().await?;
```

### `physical` - Physical Layouts & Transfers

```rust
// Standard test configuration
let config = standard_config(num_blocks);  // 2 layers, 16 page_size

// Custom configuration
let config = custom_config(blocks, layers, outer_dim, page_size, inner_dim, dtype_width);

// Create layouts
let agent = create_test_agent("my-agent");
let layout = create_fc_layout(agent, StorageKind::System, num_blocks);
let layout = create_lw_layout(agent, StorageKind::Pinned, num_blocks);

// Fill and verify
let checksums = fill_and_checksum(&layout, &block_ids, FillPattern::Sequential)?;
verify_checksums_by_position(&src_checksums, &src_ids, &dst_layout, &dst_ids)?;
```

### `distributed` - Leader/Worker Test Fixtures

```rust
use crate::v2::testing::distributed::{
    create_instance_leader_pair,
    create_instance_leader_pair_with_workers,
    TestSession,
};

// Simple leader pair (no workers, no RDMA)
let pair = create_instance_leader_pair(block_count, block_size).await?;
// pair.leader_a, pair.leader_b

// Leader pair with RDMA workers (for transfer tests)
let pair = create_instance_leader_pair_with_workers(
    block_count, block_size, num_workers, &layout_config, storage
).await?;
// pair.decode.workers[0].fill_g2_blocks(...)
// pair.prefill.workers[0].compute_g2_checksums(...)

// Populate leader's logical manager
let (block_ids, hashes) = pair.decode.populate_g2_blocks(num, size, start)?;

// TestSession for simplified session lifecycle
let session = TestSession::establish_default(&leader, &remote_leader, &hashes).await?;
let g2_blocks = session.g2_blocks();
session.pull_blocks_rdma(&g2_blocks, &dst_ids).await?;
session.close().await?;
```

### `events` - Events Pipeline Testing

```rust
use crate::v2::testing::events::EventsPipelineFixture;

// Create fixture with events pipeline (uses build_async for async setup)
let mut fixture = EventsPipelineFixture::builder()
    .instance_id(InstanceId::new_v4())  // Nova InstanceId
    .batching_window(Duration::from_millis(50))
    .subject("kvbm.events")
    .build_async().await?;

// Create manager with events integration
let manager = fixture.create_manager::<G1>(100, 4);

// Register blocks (triggers events)
// ...

// Receive events
let batch = fixture.flush_and_receive(Duration::from_millis(50)).await;
let batch = fixture.receive_batch_default().await;
```

## Test Patterns

### Pattern: Checksum Verification

```rust
// Fill source, capture checksums
let src_checksums = fill_and_checksum(&src_layout, &src_ids, FillPattern::Sequential)?;

// Execute transfer...

// Verify destination matches source
verify_checksums_by_position(&src_checksums, &src_ids, &dst_layout, &dst_ids)?;
```

### Pattern: Guard Block Verification

```rust
// Fill adjacent blocks with different pattern
let guard_checksums = create_guard_blocks(&layout, &guard_ids, FillPattern::Constant(0xFF))?;

// Execute transfer on other blocks...

// Verify guards unchanged
verify_guard_blocks_unchanged(&layout, &guard_ids, &guard_checksums)?;
```

### Pattern: Tiered Search (G2/G3)

```rust
// Populate G3 with all blocks
let g3_hashes = managers::populate_manager_with_blocks(g3_manager, &all_blocks)?;

// Populate G2 with subset (even positions)
let g2_hashes = managers::populate_manager_with_blocks(g2_manager, &even_blocks)?;

// Search returns G2 preferentially, G3 for the rest
let result = leader.scan_blocks(&all_hashes, true);
```

### Pattern: Layerwise Transfer

```rust
// Create endpoint session on source
let (session_id, handle) = leader.create_endpoint_session(&hashes)?;

// Remote attaches
let mut remote_handle = remote_leader.attach_session(instance_id, session_id).await?;

// Notify layers incrementally
for layer in 0..num_layers {
    handle.notify_layers_ready(0..layer+1).await?;
}

// Remote pulls
remote_handle.pull_blocks_rdma(&blocks, &dst_ids).await?;
```

## PubSub Testing

The `StubBus` provides in-memory pub/sub for testing event pipelines:

```rust
use crate::v2::distributed::pubsub::{StubBus, Publisher, Subscriber};

let bus = StubBus::default();
let publisher = bus.publisher();
let subscriber = bus.subscriber();

// Subscribe first (messages are not buffered for late subscribers)
let mut stream = subscriber.subscribe("kvbm.events").await?;

// Publish
publisher.publish("kvbm.events", Bytes::from("hello"))?;

// Receive
let msg = stream.next().await.unwrap();
```

## Configuration for Tests

Use `ConnectorTestConfig` for connector integration tests:

```rust
use crate::v2::testing::connector::ConnectorTestConfig;

// Builder API
let config = ConnectorTestConfig::new()
    .leader_cache_gb(1.0)
    .worker_tokio_threads(2);

// Or JSON API (vLLM-style)
let config = ConnectorTestConfig::from_json(r#"{
    "leader": { "cache": { "host": { "cache_size_gb": 1.0 } } }
}"#)?;

let leader_config = config.build_leader()?;
let worker_config = config.build_worker()?;
```

## Tips

1. **Use `#[tokio::test(flavor = "multi_thread")]`** for RDMA tests - UCX requires multiple threads.

2. **Use `TestManagerBuilder`** for creating block managers - it handles registry creation internally.

3. **Use `StorageKind::Pinned`** for RDMA tests - System memory may not be RDMA-capable.

4. **Prefer `AllEventsPolicy`** for testing events - `PowerOfTwoPolicy` only emits at specific positions. `AllEventsPolicy` is the default.

5. **Call `register_handlers()`** after creating InstanceLeaders for distributed communication.

6. **Use builder patterns** - `BlockRegistry::builder()`, `EventsManager::builder()`, `TestManagerBuilder::new()` all support fluent configuration.

7. **Use `TestAgentBuilder`** instead of `NixlAgent::with_backends()` for flexible backend handling - it allows graceful degradation when optional backends are unavailable.

8. **Use `MultiInstancePopulator`** for multi-instance tests - it reduces the boilerplate of populating multiple managers from a shared token sequence.

9. **Disk tests should use POSIX only** - GDS (GPU Direct Storage) is optional and may not be available on all systems.

10. **Use `TestSession`** for session lifecycle tests - it encapsulates the create→attach→wait_for_ready pattern.
