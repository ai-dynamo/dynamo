// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::{KvbmSequenceHashProvider, tinylfu::TinyLFUTracker};

use super::attachments::AttachmentError;
use super::*;

use crate::testing::{self, MetadataA, MetadataB, MetadataC, TestMeta};
use crate::{BlockManager, blocks::BlockDuplicationPolicy};

use std::any::TypeId;
use std::sync::Arc;

type TestMetadata = TestMeta;

/// Helper to create a token block for testing (auto block_size).
fn create_test_token_block(tokens: &[u32]) -> dynamo_tokens::TokenBlock {
    testing::create_test_token_block(tokens, tokens.len() as u32)
}

/// Helper to construct a manager seeded with a registry that the test owns.
fn manager_with_registry<T: crate::blocks::BlockMetadata + Sync>(
    registry: BlockRegistry,
    block_count: usize,
) -> BlockManager<T> {
    BlockManager::<T>::builder()
        .block_count(block_count)
        .block_size(4)
        .registry(registry)
        .duplication_policy(BlockDuplicationPolicy::Allow)
        .build()
        .unwrap()
}

/// Allocate one block from `manager`, complete it with the given tokens,
/// register it, and return the resulting `ImmutableBlock`.
fn register_one<T: crate::blocks::BlockMetadata + Sync>(
    manager: &BlockManager<T>,
    tokens: &[u32],
) -> crate::blocks::ImmutableBlock<T> {
    let mut allocated = manager.allocate_blocks(1).expect("allocate");
    let mutable = allocated.pop().unwrap();
    let tb = create_test_token_block(tokens);
    let complete = mutable.complete(&tb).expect("complete");
    manager.register_block(complete)
}

#[test]
fn test_type_tracking_enforcement() {
    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

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

    #[derive(Debug, Clone, PartialEq)]
    struct EventPublisher(String);

    #[derive(Debug, Clone, PartialEq)]
    struct EventListener(String);

    handle
        .attach_unique(EventPublisher("main_publisher".to_string()))
        .unwrap();
    handle
        .attach(EventListener("listener1".to_string()))
        .unwrap();
    handle
        .attach(EventListener("listener2".to_string()))
        .unwrap();

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
    let manager = manager_with_registry::<TestMetadata>(registry.clone(), 2);

    let tokens = [1u32, 2, 3, 4];
    let seq_hash = create_test_token_block(&tokens).kvbm_sequence_hash();
    let pre_handle = registry.register_sequence_hash(seq_hash);

    assert!(!pre_handle.has_block::<TestMetadata>());

    let immutable = register_one::<TestMetadata>(&manager, &tokens);
    assert!(pre_handle.has_block::<TestMetadata>());

    drop(immutable);
    // Block is now in the inactive pool — still present from the registry's POV.
    assert!(pre_handle.has_block::<TestMetadata>());

    // Force eviction by allocating until the inactive block is evicted.
    let _evicted = manager
        .allocate_blocks(2)
        .expect("allocate forces eviction");
    assert!(!pre_handle.has_block::<TestMetadata>());
}

#[test]
fn test_presence_tracking_different_types() {
    let registry = BlockRegistry::new();
    let manager_a = manager_with_registry::<MetadataA>(registry.clone(), 1);
    let manager_b = manager_with_registry::<MetadataB>(registry.clone(), 1);

    let tokens = [100u32, 101, 102, 103];
    let seq_hash = create_test_token_block(&tokens).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    let _ia = register_one::<MetadataA>(&manager_a, &tokens);
    assert!(handle.has_block::<MetadataA>());
    assert!(!handle.has_block::<MetadataB>());

    let _ib = register_one::<MetadataB>(&manager_b, &tokens);
    assert!(handle.has_block::<MetadataA>());
    assert!(handle.has_block::<MetadataB>());
}

#[test]
fn test_check_presence_api() {
    let registry = BlockRegistry::new();
    let manager = manager_with_registry::<TestMetadata>(registry.clone(), 4);

    let tokens_100 = [0u32, 1, 2, 3];
    let tokens_200 = [10u32, 11, 12, 13];
    let tokens_300 = [20u32, 21, 22, 23];

    let _i100 = register_one::<TestMetadata>(&manager, &tokens_100);
    let _i300 = register_one::<TestMetadata>(&manager, &tokens_300);

    let hashes = [
        create_test_token_block(&tokens_100).kvbm_sequence_hash(),
        create_test_token_block(&tokens_200).kvbm_sequence_hash(),
        create_test_token_block(&tokens_300).kvbm_sequence_hash(),
    ];

    let presence = registry.check_presence::<TestMetadata>(&hashes);
    assert_eq!(presence.len(), 3);
    assert!(presence[0].1);
    assert!(!presence[1].1);
    assert!(presence[2].1);
}

#[test]
fn test_has_any_block() {
    let registry = BlockRegistry::new();
    let manager = manager_with_registry::<MetadataB>(registry.clone(), 1);

    let tokens = [1u32, 2, 3, 4];
    let seq_hash = create_test_token_block(&tokens).kvbm_sequence_hash();
    let handle = registry.register_sequence_hash(seq_hash);

    let type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataB>()];
    assert!(!handle.has_any_block(&type_ids));

    let _ib = register_one::<MetadataB>(&manager, &tokens);
    assert!(handle.has_any_block(&type_ids));

    let other_type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataC>()];
    assert!(!handle.has_any_block(&other_type_ids));
}

#[test]
fn test_check_presence_any() {
    let registry = BlockRegistry::new();
    let manager_a = manager_with_registry::<MetadataA>(registry.clone(), 2);
    let manager_b = manager_with_registry::<MetadataB>(registry.clone(), 2);

    let tokens_100 = [10u32, 11, 12, 13];
    let tokens_200 = [1u32, 2, 3, 4];
    let tokens_300 = [20u32, 21, 22, 23];

    let _ia = register_one::<MetadataA>(&manager_a, &tokens_100);
    let _ib = register_one::<MetadataB>(&manager_b, &tokens_300);

    let hashes = [
        create_test_token_block(&tokens_100).kvbm_sequence_hash(),
        create_test_token_block(&tokens_200).kvbm_sequence_hash(),
        create_test_token_block(&tokens_300).kvbm_sequence_hash(),
    ];

    let type_ids = [TypeId::of::<MetadataA>(), TypeId::of::<MetadataB>()];
    let presence = registry.check_presence_any(&hashes, &type_ids);
    assert!(presence[0].1);
    assert!(!presence[1].1);
    assert!(presence[2].1);

    let a_only = [TypeId::of::<MetadataA>()];
    let a_presence = registry.check_presence_any(&hashes, &a_only);
    assert!(a_presence[0].1);
    assert!(!a_presence[1].1);
    assert!(!a_presence[2].1);
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
    assert!(registry.is_registered(seq_hash));
    assert_eq!(registry.registered_count(), 1);

    drop(handle2);
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

    handle.attach_unique(UniqueCounter(0)).unwrap();
    handle.get::<UniqueCounter>().with_unique_mut(|c| {
        c.increment();
        c.increment();
    });
    let value = handle.get::<UniqueCounter>().with_unique(|c| c.0);
    assert_eq!(value, Some(2));

    handle.attach(MultipleCounter(10)).unwrap();
    handle.attach(MultipleCounter(20)).unwrap();
    handle.get::<MultipleCounter>().with_multiple_mut(|cs| {
        for c in cs {
            c.increment();
        }
    });
    let total = handle
        .get::<MultipleCounter>()
        .with_multiple(|cs| cs.iter().map(|c| c.0).sum::<i32>());
    assert_eq!(total, 32);
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

    handle.attach_unique(UniqueValue(10)).unwrap();
    handle
        .get::<UniqueValue>()
        .with_all_mut(|unique, multiple| {
            assert!(unique.is_some());
            assert_eq!(multiple.len(), 0);
            if let Some(val) = unique {
                val.increment();
            }
        });
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

    handle.attach(MultipleValue(1)).unwrap();
    handle.attach(MultipleValue(2)).unwrap();
    handle
        .get::<MultipleValue>()
        .with_all_mut(|unique, multiple| {
            assert!(unique.is_none());
            assert_eq!(multiple.len(), 2);
            for val in multiple {
                val.increment();
            }
        });
    let total = handle
        .get::<MultipleValue>()
        .with_multiple(|vs| vs.iter().map(|v| v.0).sum::<i32>());
    assert_eq!(total, 5);
}

#[test]
fn test_resurrect_promotes_inactive_block_under_allow_policy() {
    let registry = BlockRegistry::new();
    let manager = manager_with_registry::<TestMetadata>(registry, 2);

    let tokens = [1u32, 2, 3, 4];
    let imm = register_one::<TestMetadata>(&manager, &tokens);
    let weak = imm.downgrade();
    drop(imm);

    // Block is now in the inactive pool. Upgrading the weak reference
    // should resurrect it.
    let resurrected = weak.upgrade().expect("resurrection should succeed");
    assert_eq!(
        resurrected.sequence_hash(),
        create_test_token_block(&tokens).kvbm_sequence_hash()
    );
}

#[test]
fn test_resurrect_via_register_returns_existing_under_reject_policy() {
    let registry = BlockRegistry::new();
    let manager = BlockManager::<TestMetadata>::builder()
        .block_count(2)
        .block_size(4)
        .registry(registry.clone())
        .duplication_policy(BlockDuplicationPolicy::Reject)
        .build()
        .unwrap();

    let tokens = [5u32, 6, 7, 8];
    let imm1 = register_one::<TestMetadata>(&manager, &tokens);
    let imm1_id = imm1.block_id();
    let imm1_weak = imm1.downgrade();
    drop(imm1);

    // Re-register the same hash under Reject policy: should return the
    // resurrected primary, not a new block.
    let imm2 = register_one::<TestMetadata>(&manager, &tokens);
    assert_eq!(
        imm2.block_id(),
        imm1_id,
        "Reject policy should reuse the original block_id"
    );
    let upgraded = imm1_weak.upgrade().expect("weak should still upgrade");
    assert_eq!(upgraded.block_id(), imm1_id);
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
    assert!(display.contains("already registered as multiple"));

    let err_unique = AttachmentError::TypeAlreadyRegisteredAsUnique(TypeId::of::<i32>());
    let display = format!("{}", err_unique);
    assert!(display.contains("already registered as unique"));
}

#[test]
fn test_is_from_registry() {
    let registry1 = BlockRegistry::new();
    let registry2 = BlockRegistry::new();

    let seq_hash = create_test_token_block(&[60, 61, 62, 63]).kvbm_sequence_hash();
    let handle = registry1.register_sequence_hash(seq_hash);

    assert!(handle.is_from_registry(&registry1));
    assert!(!handle.is_from_registry(&registry2));
}

/// Regression: `BlockRegistrationHandleInner::drop` must not remove the registry
/// entry when a newer registration has already replaced it. The race in
/// production:
///
/// 1. Thread A drops the last `Arc<InnerA>` for `seq_hash`; `drop_in_place` runs
///    but has not yet acquired the registry's position lock.
/// 2. Thread B calls `register_sequence_hash`, finds `weak.upgrade() == None`,
///    creates `InnerB`, and overwrites the entry's `Weak` in place.
/// 3. Thread A's drop body runs and unconditionally removes the entry —
///    deleting `InnerB`'s `Weak` and silently making the live block
///    unreachable through `match_sequence_hash`.
///
/// We simulate this race deterministically by manually injecting `InnerB`'s
/// weak into the PRT entry between A's strong-count-drop and A's `Drop` body.
#[test]
fn drop_does_not_remove_entry_when_replaced_by_newer_registration() {
    use super::handle::BlockRegistrationHandleInner;

    let registry = BlockRegistry::new();
    let seq_hash = create_test_token_block(&[1, 2, 3, 4]).kvbm_sequence_hash();

    // 1. Register InnerA the normal way and keep a single strong Arc to it.
    let handle_a = registry.register_sequence_hash(seq_hash);
    let inner_a: Arc<BlockRegistrationHandleInner> = handle_a.inner.clone();
    drop(handle_a);

    // 2. Inject a foreign InnerB weak into the PRT entry, simulating Thread B's
    //    `register_sequence_hash` overwriting the slot in place. The strong
    //    Arc to InnerB is held in `inner_b` so its weak stays upgradeable.
    let inner_b = Arc::new(BlockRegistrationHandleInner::new(
        seq_hash,
        Arc::downgrade(&registry.prt),
    ));
    {
        let map = registry.prt.prefix(&seq_hash);
        let mut weak = map.get_mut(&seq_hash).expect("entry present");
        *weak = Arc::downgrade(&inner_b);
    }

    // 3. Drop InnerA. Its Drop must NOT remove InnerB's entry.
    drop(inner_a);

    // 4. InnerB must still be reachable through the registry.
    assert!(
        registry.is_registered(seq_hash),
        "stale Drop removed a newer registration's entry"
    );
    let matched = registry
        .match_sequence_hash(seq_hash, false)
        .expect("InnerB should still be reachable via match_sequence_hash");
    assert!(
        Arc::ptr_eq(&matched.inner, &inner_b),
        "match_sequence_hash returned a handle that does not point to InnerB"
    );

    // 5. Sanity: after dropping InnerB, the entry is properly cleaned up.
    drop(matched);
    drop(inner_b);
    assert!(!registry.is_registered(seq_hash));
}
