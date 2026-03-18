// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{BlockId, KvbmSequenceHashProvider, tinylfu::TinyLFUTracker};

use super::attachments::AttachmentError;
use super::*;

use crate::blocks::{
    Block, BlockMetadata, CompleteBlock, PrimaryBlock, RegisteredBlock,
    state::{Registered, Staged},
};
use crate::pools::InactivePool;
use crate::testing::{
    self, MetadataA, MetadataB, MetadataC, TestMeta, TestPoolSetupBuilder, create_staged_block,
};

use std::any::TypeId;
use std::sync::Arc;

// Type aliases for backward compatibility with existing tests
type TestMetadata = TestMeta;

/// Wrapper for create_staged_block with original (tokens, block_id) arg order for fifo.rs
/// The original create_completed_block had args (tokens, block_id) but create_staged_block
/// uses (block_id, tokens), so we provide a wrapper function.
pub fn create_completed_block<T: BlockMetadata + std::fmt::Debug>(
    tokens: &[u32],
    block_id: BlockId,
) -> Block<T, Staged> {
    create_staged_block::<T>(block_id, tokens)
}

/// Helper to create a token block for testing (local wrapper with auto block_size)
fn create_test_token_block(tokens: &[u32]) -> dynamo_tokens::TokenBlock {
    testing::create_test_token_block(tokens, tokens.len() as u32)
}

/// Helper to create and register a block with specific metadata type
fn register_test_block<T: BlockMetadata + std::fmt::Debug>(
    registry: &BlockRegistry,
    block_id: BlockId,
    tokens: &[u32],
) -> Block<T, Registered> {
    let staged = create_staged_block::<T>(block_id, tokens);
    let handle = registry.register_sequence_hash(staged.sequence_hash());
    staged.register_with_handle(handle)
}

#[test]
fn test_type_tracking_enforcement() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    // Test: attach unique first, then try multiple (should fail)
    handle
        .attach_unique("unique_publisher".to_string())
        .unwrap();

    let result = handle.attach("listener1".to_string());
    assert_eq!(
        result,
        Err(AttachmentError::TypeAlreadyRegisteredAsUnique(
            TypeId::of::<String>()
        ))
    );

    // Test with different types: attach multiple first, then try unique (should fail)
    handle.attach(42i32).unwrap();
    handle.attach(43i32).unwrap();

    let result = handle.attach_unique(44i32);
    assert_eq!(
        result,
        Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(
            TypeId::of::<i32>()
        ))
    );
}

#[test]
fn test_different_types_usage() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    // Define some test types for better demonstration
    #[derive(Debug, Clone, PartialEq)]
    struct EventPublisher(String);

    #[derive(Debug, Clone, PartialEq)]
    struct EventListener(String);

    // Attach unique EventPublisher
    handle
        .attach_unique(EventPublisher("main_publisher".to_string()))
        .unwrap();

    // Attach multiple EventListeners
    handle
        .attach(EventListener("listener1".to_string()))
        .unwrap();
    handle
        .attach(EventListener("listener2".to_string()))
        .unwrap();

    // Retrieve by type using new API
    let publisher = handle.get::<EventPublisher>().with_unique(|p| p.clone());
    assert_eq!(
        publisher,
        Some(EventPublisher("main_publisher".to_string()))
    );

    let listeners = handle
        .get::<EventListener>()
        .with_multiple(|listeners| listeners.iter().map(|l| (*l).clone()).collect::<Vec<_>>());
    assert_eq!(listeners.len(), 2);
    assert!(listeners.contains(&EventListener("listener1".to_string())));
    assert!(listeners.contains(&EventListener("listener2".to_string())));

    // Test with_all for EventListener (should have no unique, only multiple)
    handle.get::<EventListener>().with_all(|unique, multiple| {
        assert_eq!(unique, None);
        assert_eq!(multiple.len(), 2);
    });

    // Attempting to register EventPublisher as multiple should fail
    let result = handle.attach(EventPublisher("another_publisher".to_string()));
    assert_eq!(
        result,
        Err(AttachmentError::TypeAlreadyRegisteredAsUnique(
            TypeId::of::<EventPublisher>()
        ))
    );

    // Attempting to register EventListener as unique should fail
    let result = handle.attach_unique(EventListener("unique_listener".to_string()));
    assert_eq!(
        result,
        Err(AttachmentError::TypeAlreadyRegisteredAsMultiple(
            TypeId::of::<EventListener>()
        ))
    );
}

#[test]
fn test_frequency_tracking() {
    let tracker = Arc::new(TinyLFUTracker::new(100));
    let registry = BlockRegistry::builder()
        .frequency_tracker(tracker.clone())
        .build();

    let block_1 = create_test_token_block(&[1, 2, 3, 4]);
    let seq_hash_1 = block_1.kvbm_sequence_hash();

    assert!(registry.has_frequency_tracking());
    assert_eq!(registry.count(seq_hash_1), 0);

    registry.touch(seq_hash_1);
    assert_eq!(registry.count(seq_hash_1), 1);

    registry.touch(seq_hash_1);
    registry.touch(seq_hash_1);
    assert_eq!(registry.count(seq_hash_1), 3);

    let block_2 = create_test_token_block(&[5, 6, 7, 8]);
    let seq_hash_2 = block_2.kvbm_sequence_hash();

    let _handle1 = registry.register_sequence_hash(seq_hash_2);
    assert_eq!(registry.count(seq_hash_2), 1);

    let _handle2 = registry.match_sequence_hash(seq_hash_2, true);
    assert_eq!(registry.count(seq_hash_2), 2);

    let _handle3 = registry.match_sequence_hash(seq_hash_2, false);
    assert_eq!(registry.count(seq_hash_2), 2);
}

#[test]
fn test_transfer_registration_no_tracking() {
    let tracker = Arc::new(TinyLFUTracker::new(100));
    let registry = BlockRegistry::builder()
        .frequency_tracker(tracker.clone())
        .build();

    let seq_hash_1 = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let seq_hash_2 = create_test_token_block(&[5, 6, 7, 8]).kvbm_sequence_hash();

    let _handle1 = registry.transfer_registration(seq_hash_1);
    assert_eq!(registry.count(seq_hash_1), 0);

    let _handle2 = registry.register_sequence_hash(seq_hash_2);
    assert_eq!(registry.count(seq_hash_2), 1);
}

#[test]
fn test_presence_tracking_lifecycle() {
    let registry = BlockRegistry::new();
    let complete_block = create_completed_block::<TestMetadata>(&[1, 2, 3, 4], 42);
    let handle = registry.register_sequence_hash(complete_block.sequence_hash());

    // Initially, no block is present
    assert!(!handle.has_block::<TestMetadata>());

    // Register a block - this should mark it as present
    let registered_block = complete_block.register_with_handle(handle.clone());

    // Now the block should be present
    assert!(handle.has_block::<TestMetadata>());

    // Reset the block - this should mark it as absent
    let _reset_block = registered_block.reset();

    // Now the block should not be present
    assert!(!handle.has_block::<TestMetadata>());
}

#[test]
fn test_presence_tracking_different_types() {
    let registry = BlockRegistry::new();
    let complete_block = create_completed_block::<TestMetadata>(&[100, 101, 102, 103], 42);
    let handle = registry.register_sequence_hash(complete_block.sequence_hash());

    // Register a block with MetadataA
    let _registered_a = register_test_block::<MetadataA>(&registry, 42, &[100, 101, 102, 103]);

    // MetadataA should be present, but not MetadataB
    assert!(handle.has_block::<MetadataA>());
    assert!(!handle.has_block::<MetadataB>());

    // Now register a block with MetadataB (same seq_hash, different type)
    let _registered_b = register_test_block::<MetadataB>(&registry, 42, &[100, 101, 102, 103]);

    // Both should be present now
    assert!(handle.has_block::<MetadataA>());
    assert!(handle.has_block::<MetadataB>());
}

#[test]
fn test_check_presence_api() {
    let registry = BlockRegistry::new();

    // Register blocks for hashes 100 and 300, but not 200
    let block_100 = register_test_block::<TestMetadata>(&registry, 100, &[0, 1, 2, 3]);
    let block_200 = create_completed_block::<TestMetadata>(&[10, 11, 12, 13], 200);
    let block_300 = register_test_block::<TestMetadata>(&registry, 300, &[20, 21, 22, 23]);

    let hashes = vec![
        block_100.sequence_hash(),
        block_200.sequence_hash(),
        block_300.sequence_hash(),
    ];

    // Check presence using the API
    let presence = registry.check_presence::<TestMetadata>(&hashes);

    assert_eq!(presence.len(), 3);
    assert_eq!(presence[0], (block_100.sequence_hash(), true));
    assert_eq!(presence[1], (block_200.sequence_hash(), false));
    assert_eq!(presence[2], (block_300.sequence_hash(), true));
}

#[test]
fn test_has_any_block() {
    let registry = BlockRegistry::new();
    let complete_block = create_completed_block::<MetadataB>(&[1, 2, 3, 4], 42);
    let handle = registry.register_sequence_hash(complete_block.sequence_hash());

    // No blocks initially
    let type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataB>()];
    assert!(!handle.has_any_block(&type_ids));

    // Register a block with MetadataB
    let _registered = complete_block.register_with_handle(handle.clone());

    // Now should return true because MetadataB is present
    assert!(handle.has_any_block(&type_ids));

    // Check with different types (neither A nor C is present)
    let other_type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataC>()];
    assert!(!handle.has_any_block(&other_type_ids));

    // Check with just MetadataB
    let b_only = [TypeId::of::<MetadataB>()];
    assert!(handle.has_any_block(&b_only));
}

#[test]
fn test_check_presence_any() {
    let registry = BlockRegistry::new();

    // Create three blocks:
    // hash 100: has MetadataA
    // hash 200: has MetadataB
    // hash 300: has neither
    let block_a = register_test_block::<MetadataA>(&registry, 100, &[10, 11, 12, 13]);
    let block_b = create_completed_block::<MetadataA>(&[1, 2, 3, 4], 200);
    let block_c = register_test_block::<MetadataB>(&registry, 300, &[20, 21, 22, 23]);

    let hashes = vec![
        block_a.sequence_hash(),
        block_b.sequence_hash(),
        block_c.sequence_hash(),
    ];

    // Check presence with both types
    let type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataB>()];
    let presence = registry.check_presence_any(&hashes, &type_ids);

    assert_eq!(presence.len(), 3);
    assert_eq!(presence[0], (block_a.sequence_hash(), true));
    assert_eq!(presence[1], (block_b.sequence_hash(), false));
    assert_eq!(presence[2], (block_c.sequence_hash(), true));

    // Check with only MetadataA
    let a_only = [TypeId::of::<MetadataA>()];
    let a_presence = registry.check_presence_any(&hashes, &a_only);
    assert_eq!(a_presence[0], (block_a.sequence_hash(), true));
    assert_eq!(a_presence[1], (block_b.sequence_hash(), false));
    assert_eq!(a_presence[2], (block_c.sequence_hash(), false));
}

#[test]
fn test_handle_drop_removes_registration() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();

    {
        let _handle = registry.register_sequence_hash(seq_hash);
        assert!(registry.is_registered(seq_hash));
        assert_eq!(registry.registered_count(), 1);
    }

    // Handle should be dropped and registration removed
    assert!(!registry.is_registered(seq_hash));
    assert_eq!(registry.registered_count(), 0);
}

#[test]
fn test_multiple_handles_same_sequence() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle1 = registry.register_sequence_hash(seq_hash);
    let handle2 = handle1.clone();

    drop(handle1);

    // Sequence should still be registered because handle2 exists
    assert!(registry.is_registered(seq_hash));
    assert_eq!(registry.registered_count(), 1);

    drop(handle2);

    // Now sequence should be unregistered
    assert!(!registry.is_registered(seq_hash));
    assert_eq!(registry.registered_count(), 0);
}

#[test]
fn test_mutable_access() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    #[derive(Debug, Clone, PartialEq)]
    struct UniqueCounter(i32);

    #[derive(Debug, Clone, PartialEq)]
    struct MultipleCounter(i32);

    impl UniqueCounter {
        fn increment(&mut self) {
            self.0 += 1;
        }
    }

    impl MultipleCounter {
        fn increment(&mut self) {
            self.0 += 1;
        }
    }

    // Test unique mutable access
    handle.attach_unique(UniqueCounter(0)).unwrap();

    handle.get::<UniqueCounter>().with_unique_mut(|counter| {
        counter.increment();
        counter.increment();
    });

    // Verify the change
    let value = handle
        .get::<UniqueCounter>()
        .with_unique(|counter| counter.0);
    assert_eq!(value, Some(2));

    // Test mutable access to multiple (different type)
    handle.attach(MultipleCounter(10)).unwrap();
    handle.attach(MultipleCounter(20)).unwrap();

    handle
        .get::<MultipleCounter>()
        .with_multiple_mut(|counters| {
            for counter in counters {
                counter.increment();
            }
        });

    // Verify multiple were modified
    let total = handle
        .get::<MultipleCounter>()
        .with_multiple(|counters| counters.iter().map(|c| c.0).sum::<i32>());
    assert_eq!(total, 32); // 11 + 21
}

#[test]
fn test_with_all_mut_unique() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    #[derive(Debug, Clone, PartialEq)]
    struct UniqueValue(i32);

    impl UniqueValue {
        fn increment(&mut self) {
            self.0 += 1;
        }
    }

    // Attach unique value
    handle.attach_unique(UniqueValue(10)).unwrap();

    // Test with_all_mut for unique type
    handle
        .get::<UniqueValue>()
        .with_all_mut(|unique, multiple| {
            assert!(unique.is_some());
            assert_eq!(multiple.len(), 0);
            if let Some(val) = unique {
                val.increment();
            }
        });

    // Verify the change
    let value = handle.get::<UniqueValue>().with_unique(|v| v.0);
    assert_eq!(value, Some(11));
}

#[test]
fn test_with_all_mut_multiple() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    #[derive(Debug, Clone, PartialEq)]
    struct MultipleValue(i32);

    impl MultipleValue {
        fn increment(&mut self) {
            self.0 += 1;
        }
    }

    // Attach multiple values
    handle.attach(MultipleValue(1)).unwrap();
    handle.attach(MultipleValue(2)).unwrap();

    // Test with_all_mut for multiple type
    handle
        .get::<MultipleValue>()
        .with_all_mut(|unique, multiple| {
            assert!(unique.is_none());
            assert_eq!(multiple.len(), 2);
            for val in multiple {
                val.increment();
            }
        });

    // Verify the changes
    let total = handle
        .get::<MultipleValue>()
        .with_multiple(|values| values.iter().map(|v| v.0).sum::<i32>());
    assert_eq!(total, 5); // 2 + 3
}

#[test]
fn test_concurrent_try_get_block_and_drop() {
    use crate::pools::backends::{FifoReusePolicy, HashMapBackend};
    use crate::pools::*;
    use std::sync::Barrier;
    use std::thread;

    let registry = BlockRegistry::new();

    let tokens = vec![1u32, 2, 3, 4];
    let token_block = create_test_token_block(&tokens);
    let seq_hash = token_block.kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    let reset_blocks: Vec<_> = (0..10).map(|i| Block::new(i, 4)).collect();
    let reset_pool = ResetPool::new(reset_blocks, 4, None);
    let reuse_policy = Box::new(FifoReusePolicy::new());
    let backend = Box::new(HashMapBackend::new(reuse_policy));
    let registered_pool = InactivePool::new(backend, &reset_pool, None);

    // Create barriers for synchronization
    let barrier1 = Arc::new(Barrier::new(2));
    let barrier2 = Arc::new(Barrier::new(2));
    let barrier1_clone = barrier1.clone();
    let barrier2_clone = barrier2.clone();

    // Create custom return function that holds the Arc at barriers
    let registered_pool_clone = registered_pool.clone();
    let pool_return_fn = Arc::new(move |block: Arc<Block<TestMetadata, Registered>>| {
        barrier1_clone.wait();
        barrier2_clone.wait();
        (registered_pool_clone.return_fn())(block);
    }) as Arc<dyn Fn(Arc<Block<TestMetadata, Registered>>) + Send + Sync>;

    let complete_block = Block::<TestMetadata, _>::new(0, 4).stage(seq_hash);
    let registered_block = complete_block.register_with_handle(handle.clone());

    let primary_arc = PrimaryBlock::new_attached(Arc::new(registered_block), pool_return_fn);
    let immutable_block = primary_arc as Arc<dyn RegisteredBlock<TestMetadata>>;

    let handle_clone = handle.clone();
    let real_return_fn = registered_pool.return_fn();
    let registered_pool_clone2 = registered_pool.clone();

    let upgrade_thread = thread::spawn(move || {
        barrier1.wait();
        let result = handle_clone.try_get_block::<TestMetadata>(real_return_fn);
        barrier2.wait();
        result
    });

    let drop_thread = thread::spawn(move || {
        drop(immutable_block);
    });

    let upgraded_block = upgrade_thread.join().unwrap();
    drop_thread.join().unwrap();

    assert!(
        upgraded_block.is_some(),
        "Should successfully upgrade the weak reference to Arc<Block<T, Registered>>"
    );

    let _held_block = upgraded_block;

    assert_eq!(
        registered_pool_clone2.len(),
        0,
        "Block should not be in inactive pool because Arc refcount was >= 2"
    );
}

/// Test helper to create an inactive pool with test infrastructure
fn create_test_inactive_pool() -> (
    crate::pools::ResetPool<TestMetadata>,
    InactivePool<TestMetadata>,
) {
    let setup = TestPoolSetupBuilder::default().build().unwrap();
    let (inactive_pool, reset_pool) = setup.build_pools::<TestMetadata>();
    (reset_pool, inactive_pool)
}

#[test]
fn test_attach_block_ref_called_on_inactive_promotion_allow_policy() {
    use crate::pools::*;

    let registry = BlockRegistry::new();
    let (reset_pool, inactive_pool) = create_test_inactive_pool();

    let tokens = vec![1u32, 2, 3, 4];
    let token_block = create_test_token_block(&tokens);
    let seq_hash = token_block.kvbm_sequence_hash();

    let handle = registry.register_sequence_hash(seq_hash);

    let complete_block1 = Block::<TestMetadata, _>::new(100, 4)
        .complete(&token_block)
        .expect("Block size should match");

    let complete_block1 = CompleteBlock::new(complete_block1, reset_pool.return_fn());

    let registered1 = handle.register_block(
        complete_block1,
        BlockDuplicationPolicy::Allow,
        &inactive_pool,
        None,
    );

    drop(registered1);

    assert!(
        inactive_pool.has_block(seq_hash),
        "Block should be in inactive pool after drop"
    );

    let before_result = handle.try_get_block::<TestMetadata>(inactive_pool.return_fn());
    assert!(
        before_result.is_none(),
        "try_get_block should return None before re-promotion (weak ref expired)"
    );

    let complete_block2 = Block::<TestMetadata, _>::new(200, 4)
        .complete(&token_block)
        .expect("Block size should match");

    let complete_block2 = CompleteBlock::new(complete_block2, reset_pool.return_fn());

    let registered2 = handle.register_block(
        complete_block2,
        BlockDuplicationPolicy::Allow,
        &inactive_pool,
        None,
    );

    let after_result = handle.try_get_block::<TestMetadata>(inactive_pool.return_fn());
    assert!(
        after_result.is_some(),
        "try_get_block should succeed after promotion - attach_block_ref must have been called"
    );

    drop(registered2);
    drop(after_result);
}

#[test]
fn test_attach_block_ref_called_on_inactive_promotion_reject_policy() {
    use crate::pools::*;

    let registry = BlockRegistry::new();
    let (reset_pool, inactive_pool) = create_test_inactive_pool();

    let tokens = vec![5u32, 6, 7, 8];
    let token_block = create_test_token_block(&tokens);
    let seq_hash = token_block.kvbm_sequence_hash();

    let handle = registry.register_sequence_hash(seq_hash);

    let complete_block1 = Block::<TestMetadata, _>::new(100, 4)
        .complete(&token_block)
        .expect("Block size should match");

    let complete_block1 = CompleteBlock::new(complete_block1, reset_pool.return_fn());

    let registered1 = handle.register_block(
        complete_block1,
        BlockDuplicationPolicy::Reject,
        &inactive_pool,
        None,
    );

    drop(registered1);

    assert!(inactive_pool.has_block(seq_hash));

    let before_result = handle.try_get_block::<TestMetadata>(inactive_pool.return_fn());
    assert!(before_result.is_none());

    let complete_block2 = Block::<TestMetadata, _>::new(200, 4)
        .complete(&token_block)
        .expect("Block size should match");

    let complete_block2 = CompleteBlock::new(complete_block2, reset_pool.return_fn());

    let registered2 = handle.register_block(
        complete_block2,
        BlockDuplicationPolicy::Reject,
        &inactive_pool,
        None,
    );

    let after_result = handle.try_get_block::<TestMetadata>(inactive_pool.return_fn());
    assert!(
        after_result.is_some(),
        "try_get_block should succeed after Reject policy promotion"
    );

    drop(registered2);
    drop(after_result);
}

#[test]
fn test_touch_callback_fires() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    let counter = Arc::new(AtomicU32::new(0));
    let counter_clone = counter.clone();

    handle.on_touch(Arc::new(move |hash| {
        assert_eq!(hash, seq_hash);
        counter_clone.fetch_add(1, Ordering::Relaxed);
    }));

    handle.touch();
    assert_eq!(counter.load(Ordering::Relaxed), 1);

    handle.touch();
    handle.touch();
    assert_eq!(counter.load(Ordering::Relaxed), 3);
}

#[test]
fn test_touch_multiple_callbacks() {
    use std::sync::atomic::{AtomicU32, Ordering};

    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[5, 6, 7, 8]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    let counter_a = Arc::new(AtomicU32::new(0));
    let counter_b = Arc::new(AtomicU32::new(0));
    let ca = counter_a.clone();
    let cb = counter_b.clone();

    handle.on_touch(Arc::new(move |_| {
        ca.fetch_add(1, Ordering::Relaxed);
    }));
    handle.on_touch(Arc::new(move |_| {
        cb.fetch_add(10, Ordering::Relaxed);
    }));

    handle.touch();
    assert_eq!(counter_a.load(Ordering::Relaxed), 1);
    assert_eq!(counter_b.load(Ordering::Relaxed), 10);
}

#[test]
fn test_touch_no_callbacks_is_noop() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[9, 10, 11, 12]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    // Should not panic or fail
    handle.touch();
}

#[test]
fn test_touch_callback_receives_correct_hash() {
    use parking_lot::Mutex;

    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[13, 14, 15, 16]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    let received_hash = Arc::new(Mutex::new(None));
    let rh = received_hash.clone();

    handle.on_touch(Arc::new(move |hash| {
        *rh.lock() = Some(hash);
    }));

    handle.touch();
    assert_eq!(*received_hash.lock(), Some(seq_hash));
}

#[test]
fn test_with_all_mut_no_attachments() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[50, 51, 52, 53]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    struct UnusedType(i32);

    // Call with_all_mut without attaching anything — exercises the None arm
    let result = handle.get::<UnusedType>().with_all_mut(|unique, multiple| {
        assert!(unique.is_none());
        assert_eq!(multiple.len(), 0);
        42
    });
    assert_eq!(result, 42);
}

#[test]
fn test_attachment_error_display() {
    let err_multiple = AttachmentError::TypeAlreadyRegisteredAsMultiple(TypeId::of::<String>());
    let display = format!("{}", err_multiple);
    assert!(
        display.contains("already registered as multiple"),
        "Display should describe multiple registration: {}",
        display
    );

    let err_unique = AttachmentError::TypeAlreadyRegisteredAsUnique(TypeId::of::<i32>());
    let display = format!("{}", err_unique);
    assert!(
        display.contains("already registered as unique"),
        "Display should describe unique registration: {}",
        display
    );
}

#[test]
fn test_is_from_registry() {
    let registry1 = BlockRegistry::new();
    let registry2 = BlockRegistry::new();

    let seq_hash = create_test_token_block(&[60, 61, 62, 63]).kvbm_sequence_hash();
    let handle = registry1.register_sequence_hash(seq_hash);

    assert!(
        handle.is_from_registry(&registry1),
        "Handle should be from registry1"
    );
    assert!(
        !handle.is_from_registry(&registry2),
        "Handle should NOT be from registry2"
    );
}
