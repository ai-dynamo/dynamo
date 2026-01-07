// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_tokens::TokenBlockSequence;

use crate::KvbmSequenceHashProvider;
use crate::logical::pools::backends::{FifoReusePolicy, HashMapBackend};

use super::{
    blocks::{state::Registered, *},
    pools::*,
};

use std::sync::{Arc, Barrier};
use std::thread;

// #[test]
// fn test_mutable_access() {
//     let registry = BlockRegistry::new();
//     let handle = registry.register_sequence_hash(42);

//     #[derive(Debug, Clone, PartialEq)]
//     struct UniqueCounter(i32);

//     #[derive(Debug, Clone, PartialEq)]
//     struct MultipleCounter(i32);

//     impl UniqueCounter {
//         fn increment(&mut self) {
//             self.0 += 1;
//         }
//     }

//     impl MultipleCounter {
//         fn increment(&mut self) {
//             self.0 += 1;
//         }
//     }

//     // Test unique mutable access
//     handle.attach_unique(UniqueCounter(0)).unwrap();

//     handle.get::<UniqueCounter>().with_unique_mut(|counter| {
//         counter.increment();
//         counter.increment();
//     });

//     // Verify the change
//     let value = handle
//         .get::<UniqueCounter>()
//         .with_unique(|counter| counter.0);
//     assert_eq!(value, Some(2));

//     // Test mutable access to multiple (different type)
//     handle.attach(MultipleCounter(10)).unwrap();
//     handle.attach(MultipleCounter(20)).unwrap();

//     handle
//         .get::<MultipleCounter>()
//         .with_multiple_mut(|counters| {
//             for counter in counters {
//                 counter.increment();
//             }
//         });

//     // Verify multiple were modified
//     let total = handle
//         .get::<MultipleCounter>()
//         .with_multiple(|counters| counters.iter().map(|c| c.0).sum::<i32>());
//     assert_eq!(total, 32); // 11 + 21
// }

// #[test]
// fn test_with_all_mut_unique() {
//     let registry = BlockRegistry::new();
//     let handle = registry.register_sequence_hash(42);

//     #[derive(Debug, Clone, PartialEq)]
//     struct UniqueValue(i32);

//     impl UniqueValue {
//         fn increment(&mut self) {
//             self.0 += 1;
//         }
//     }

//     // Attach unique value
//     handle.attach_unique(UniqueValue(10)).unwrap();

//     // Test with_all_mut for unique type
//     handle
//         .get::<UniqueValue>()
//         .with_all_mut(|unique, multiple| {
//             assert!(unique.is_some());
//             assert_eq!(multiple.len(), 0);
//             if let Some(val) = unique {
//                 val.increment();
//             }
//         });

//     // Verify the change
//     let value = handle.get::<UniqueValue>().with_unique(|v| v.0);
//     assert_eq!(value, Some(11));
// }

// #[test]
// fn test_with_all_mut_multiple() {
//     let registry = BlockRegistry::new();
//     let handle = registry.register_sequence_hash(42);

//     #[derive(Debug, Clone, PartialEq)]
//     struct MultipleValue(i32);

//     impl MultipleValue {
//         fn increment(&mut self) {
//             self.0 += 1;
//         }
//     }

//     // Attach multiple values
//     handle.attach(MultipleValue(1)).unwrap();
//     handle.attach(MultipleValue(2)).unwrap();

//     // Test with_all_mut for multiple type
//     handle
//         .get::<MultipleValue>()
//         .with_all_mut(|unique, multiple| {
//             assert!(unique.is_none());
//             assert_eq!(multiple.len(), 2);
//             for val in multiple {
//                 val.increment();
//             }
//         });

//     // Verify the changes
//     let total = handle
//         .get::<MultipleValue>()
//         .with_multiple(|values| values.iter().map(|v| v.0).sum::<i32>());
//     assert_eq!(total, 5); // 2 + 3
// }

/// Tests block resurrection during the pool return transition window.
///
/// # What This Tests
///
/// This test validates the critical ability of `try_get_block()` to "resurrect" a block
/// while it's transitioning back to the inactive pool. This happens when:
/// - A `PrimaryBlock` is dropped, triggering its return function
/// - The return function holds the `Arc<Block<T, Registered>>`
/// - Another thread calls `try_get_block()` during this window
/// - The weak reference to the raw block can be upgraded before the Arc is unwrapped
///
/// # Why This Matters
///
/// The registry maintains weak references to blocks even after the `PrimaryBlock` wrapper
/// is dropped. This allows blocks to be found and reused while they're still in memory,
/// even if they're transitioning to the inactive pool. This is a key optimization:
/// - Avoids unnecessary pool insertion/removal cycles
/// - Reduces lock contention on the pool
/// - Enables lock-free block reuse in high-concurrency scenarios
///
/// # Test Strategy
///
/// The test uses two barriers to create a deterministic interleaving:
///
/// 1. **Drop Thread** drops the `PrimaryBlock`, triggering a custom return function that:
///    - Receives the `Arc<Block<T, Registered>>`
///    - Signals readiness at barrier1
///    - Waits at barrier2 for the upgrade to complete
///    - Attempts to return to pool (fails if Arc was upgraded)
///
/// 2. **Upgrade Thread** waits for the Arc to be held by return function, then:
///    - Calls `try_get_block()` which upgrades the weak reference
///    - Creates a new `PrimaryBlock` wrapping the same Arc
///    - Signals completion at barrier2
///
/// # Expected Outcome
///
/// - `try_get_block()` successfully upgrades and returns a new `PrimaryBlock`
/// - The Arc refcount becomes ≥ 2 (held by both return fn and new PrimaryBlock)
/// - `Arc::try_unwrap()` in the inactive pool's return function fails
/// - The block never makes it into the inactive pool
/// - The block remains accessible through the upgraded reference
#[test]
fn test_concurrent_try_get_block_and_drop() {
    #[derive(Debug, Clone, PartialEq)]
    struct TestData {
        value: u64,
    }

    let registry = BlockRegistry::new();

    let tokens = vec![1u32, 2, 3, 4];
    let sequence = TokenBlockSequence::from_slice(&tokens, 4, Some(42));
    let token_block = if let Some(block) = sequence.blocks().first() {
        block.clone()
    } else {
        let mut partial = sequence.into_parts().1;
        partial.commit().expect("Should be able to commit")
    };

    let seq_hash = token_block.kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    let reset_blocks: Vec<_> = (0..10).map(|i| Block::new(i, 4)).collect();
    let reset_pool = ResetPool::new(reset_blocks, 4);
    let reuse_policy = Box::new(FifoReusePolicy::new());
    let backend = Box::new(HashMapBackend::new(reuse_policy));
    let registered_pool = InactivePool::new(backend, &reset_pool);

    // Create barriers for synchronization
    // barrier1: signals that return function has the Arc
    // barrier2: signals that upgrade has completed
    let barrier1 = Arc::new(Barrier::new(2));
    let barrier2 = Arc::new(Barrier::new(2));
    let barrier1_clone = barrier1.clone();
    let barrier2_clone = barrier2.clone();

    // Create custom return function that holds the Arc at barriers
    let registered_pool_clone = registered_pool.clone();
    let pool_return_fn = Arc::new(move |block: Arc<Block<TestData, Registered>>| {
        // Signal that we have the Arc
        barrier1_clone.wait();
        // Wait for upgrade to complete
        barrier2_clone.wait();
        // Now try to return - this will fail if try_get_block upgraded the Arc
        (registered_pool_clone.return_fn())(block);
    }) as Arc<dyn Fn(Arc<Block<TestData, Registered>>) + Send + Sync>;

    // Manually create a registered block and PrimaryBlock with custom return function
    // This is necessary because register_block() now takes &InactivePool instead of pool_return_fn
    let complete_block = Block::<TestData, _>::new(0, 4)
        .complete(token_block)
        .expect("Block size should match");
    let registered_block = complete_block.register(handle.clone());

    // Create PrimaryBlock with custom return function
    let primary = PrimaryBlock::new(Arc::new(registered_block), pool_return_fn);

    // Manually attach the block to the registry for future lookups
    let immutable_block = handle.attach_block(primary);

    let handle_clone = handle.clone();
    let real_return_fn = registered_pool.return_fn();
    let registered_pool_clone2 = registered_pool.clone();

    let upgrade_thread = thread::spawn(move || {
        // Wait for return function to receive the Arc
        barrier1.wait();
        // Try to upgrade - should succeed because Arc is held by return fn
        // Use the real return function (not the custom one) to avoid deadlock
        let result = handle_clone.try_get_block::<TestData>(real_return_fn);
        // Signal that upgrade is complete
        barrier2.wait();
        result
    });

    let drop_thread = thread::spawn(move || {
        // Drop the block, which triggers the return function that waits at barriers
        drop(immutable_block);
    });

    // Get the upgraded block from try_get_block
    let upgraded_block = upgrade_thread.join().unwrap();

    drop_thread.join().unwrap();

    // Verify that try_get_block succeeded
    assert!(
        upgraded_block.is_some(),
        "Should successfully upgrade the weak reference to Arc<Block<T, Registered>>"
    );

    // Hold the block to keep Arc refcount > 1
    let _held_block = upgraded_block;

    // Verify that the block never made it to the inactive pool
    // because Arc::try_unwrap failed due to refcount >= 2
    assert_eq!(
        registered_pool_clone2.len(),
        0,
        "Block should not be in inactive pool because Arc refcount was >= 2"
    );
}
