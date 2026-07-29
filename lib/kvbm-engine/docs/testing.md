# Testing Module

Test infrastructure for the kvbm-engine crate. Core block and token utilities
are re-exported from `kvbm_logical::testing` and `kvbm_physical::testing`;
this module adds engine-specific helpers for transport, sessions, offload
pipelines, and multi-instance scenarios.

## Test Helpers

### TestManagerBuilder / TestRegistryBuilder

Create test block managers and registries with synthetic physical layouts.
`TestManagerBuilder` produces a `BlockManager<T>` backed by mock memory.
`TestRegistryBuilder` produces a `BlockRegistry` pre-populated with hashes.

Use `populate_manager_with_blocks` and `create_and_populate_manager` to
quickly set up managers with pre-allocated blocks for testing.

### MessengerPair

Creates a pair of connected Velo `Messenger` instances for transport
testing without a real network. Messages sent through one messenger are
received by the other, enabling end-to-end session testing in a single
process.

```rust,ignore
let (messenger_a, messenger_b) = create_messenger_pair_tcp().await?;
```

### TestSession

Helper for testing distributed session protocols. Sets up the full session
infrastructure (dispatch maps, transport, channels) for testing
`InitiatorSession` / `ResponderSession` / `ControllableSession` interactions.

### EventsPipelineFixture

Test fixture for the offload pipeline. Provides pre-configured pipeline
stages, event managers, and block managers for testing policy evaluation,
batching, and transfer execution in isolation.

### MultiInstancePopulator

Sets up multi-instance distributed test scenarios with multiple leaders,
workers, and block managers. Populates each instance with configurable
block patterns for testing cross-instance onboarding.

```rust,ignore
let populated = MultiInstancePopulator::builder()
    .instance_count(3)
    .blocks_per_instance(100)
    .build()?
    .populate()
    .await?;
```

### Physical Test Utilities

`TestAgent` and `TestAgentBuilder` create mock `NixlAgent` instances for
testing `TransferManager` without real RDMA hardware. `TransferChecksums`
provides utilities for verifying transfer correctness.

### Token Block Helpers

The `token_blocks` module provides utilities for creating test blocks with
known token sequences, useful for verifying search and match operations.

## Writing a New Test

1. Choose the appropriate fixture for your test scope:
   - Single-instance transfer → `TestManagerBuilder` + `TestAgent`
   - Session protocol → `TestSession` + `MessengerPair`
   - Offload pipeline → `EventsPipelineFixture`
   - Multi-instance → `MultiInstancePopulator`
2. Build the fixture and populate with test data
3. Exercise the code under test
4. Assert on results and verify cleanup (blocks released, sessions closed)
