// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared test utilities for radix tree tests.

use crate::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent, WorkerId,
};

/// Creates blocks with artificial hash mapping (hash * 100) for testing.
pub fn make_blocks(hashes: Vec<u64>) -> Vec<KvCacheStoredBlockData> {
    hashes
        .iter()
        .map(|i| KvCacheStoredBlockData {
            tokens_hash: LocalBlockHash(*i),
            block_hash: ExternalSequenceBlockHash(*i * 100),
            mm_extra_info: None,
        })
        .collect()
}

pub fn add_blocks(
    hashes: Vec<u64>,
    parent_hash: Option<ExternalSequenceBlockHash>,
) -> KvCacheEventData {
    KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash,
        blocks: make_blocks(hashes),
        group_id: None,
    })
}

pub fn create_store_event(
    worker_id: WorkerId,
    event_id: u64,
    hashes: Vec<u64>,
    parent: Option<ExternalSequenceBlockHash>,
) -> RouterEvent {
    RouterEvent {
        worker_id,
        event: KvCacheEvent {
            event_id,
            data: add_blocks(hashes, parent),
            dp_rank: 0,
        },
    }
}

pub fn create_remove_event(worker_id: WorkerId, event_id: u64, hashes: Vec<u64>) -> RouterEvent {
    RouterEvent {
        worker_id,
        event: KvCacheEvent {
            event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: hashes
                    .iter()
                    .map(|i| ExternalSequenceBlockHash(*i * 100))
                    .collect(),
                group_id: None,
            }),
            dp_rank: 0,
        },
    }
}

/// Create a store event with a specific group_id (for hybrid model testing).
pub fn create_store_event_with_group(
    worker_id: WorkerId,
    event_id: u64,
    hashes: Vec<u64>,
    parent: Option<ExternalSequenceBlockHash>,
    group_id: u32,
) -> RouterEvent {
    RouterEvent {
        worker_id,
        event: KvCacheEvent {
            event_id,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: parent,
                blocks: make_blocks(hashes),
                group_id: Some(group_id),
            }),
            dp_rank: 0,
        },
    }
}

/// Create a remove event with a specific group_id (for hybrid model testing).
pub fn create_remove_event_with_group(
    worker_id: WorkerId,
    event_id: u64,
    hashes: Vec<u64>,
    group_id: u32,
) -> RouterEvent {
    RouterEvent {
        worker_id,
        event: KvCacheEvent {
            event_id,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: hashes
                    .iter()
                    .map(|i| ExternalSequenceBlockHash(*i * 100))
                    .collect(),
                group_id: Some(group_id),
            }),
            dp_rank: 0,
        },
    }
}
