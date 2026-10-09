// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `crtc_thread_exit.rs` adapted to arena-c (local campaign branch C): a thread that exits
//! while a large backlog of expired arena-c frees is queued must not run them recursively.
//!
//! crossbeam-epoch collects during a thread's final unpin, and any pin inside a deferred
//! function at that point registers a fresh participant whose first pin collects again,
//! nesting once per expired bag. arena-c's deferred frees only push ids and addresses onto
//! free lists and never pin, so the backlog must drain flat. This runs in its own test
//! binary so no other test shares the process-global collector.

use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use dynamo_kv_router::indexer::arena_c::ArenaIndexC;
use dynamo_kv_router::indexer::{SyncIndexer, WorkerTask};
use dynamo_kv_router::protocols::{
    ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
    KvCacheStoredBlockData, LocalBlockHash, RouterEvent,
};

const TOP_NODES: u64 = 20_000;

fn event(data: KvCacheEventData) -> WorkerTask {
    WorkerTask::Event(RouterEvent::new(
        0,
        KvCacheEvent {
            event_id: 0,
            data,
            dp_rank: 0,
        },
    ))
}

/// Stores `blocks` under `parent`. Every block id is unique here, so it doubles as the
/// sequence hash of the prefix it ends.
fn store(parent: Option<u64>, blocks: &[u64]) -> WorkerTask {
    let blocks = blocks
        .iter()
        .map(|&block| KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(block),
            tokens_hash: LocalBlockHash(block),
            mm_extra_info: None,
        })
        .collect();
    event(KvCacheEventData::Stored(KvCacheStoreData {
        parent_hash: parent.map(ExternalSequenceBlockHash),
        start_position: None,
        blocks,
    }))
}

fn run_lane(tree: &Arc<ArenaIndexC>, tasks: Vec<WorkerTask>) {
    let (events, receiver) = flume::unbounded();
    let lane = {
        let tree = tree.clone();
        thread::spawn(move || tree.worker(receiver, None).unwrap())
    };
    for task in tasks {
        events.send(task).unwrap();
    }
    events.send(WorkerTask::Terminate).unwrap();
    lane.join().unwrap();
}

#[test]
fn arena_c_thread_exit_with_expired_free_backlog_does_not_overflow() {
    let tree = Arc::new(ArenaIndexC::new());
    let stores = (0..TOP_NODES).flat_map(|i| {
        let base = i * 4 + 1;
        // A store diverging after `base` hangs [base + 2] as a child at offset 1 of
        // [base, base + 1], so every top run owns a child table.
        [
            store(None, &[base, base + 1]),
            store(Some(base), &[base + 2]),
        ]
    });
    run_lane(&tree, stores.collect());
    // Reclaim what building retired, so the backlog below is all detached subtrees.
    for _ in 0..1024 {
        crossbeam_epoch::pin().flush();
    }

    // A thread pinned meanwhile keeps every snapshot the clear retires from expiring.
    let (pinned_tx, pinned_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let staller = thread::spawn(move || {
        let _guard = crossbeam_epoch::pin();
        pinned_tx.send(()).unwrap();
        release_rx.recv().unwrap();
    });
    pinned_rx.recv().unwrap();
    // Clearing the only worker empties every run, so eager unlinks defer every run id,
    // array and table while the staller holds the epoch back.
    run_lane(&tree, vec![event(KvCacheEventData::Cleared)]);
    release_tx.send(()).unwrap();
    staller.join().unwrap();

    // Let the backlog expire without collecting most of it.
    for _ in 0..2 {
        crossbeam_epoch::pin().flush();
    }

    // The exiting thread's final unpin collects, because it has pinned 128 times.
    thread::Builder::new()
        .stack_size(256 * 1024)
        .spawn(|| {
            for _ in 0..128 {
                drop(crossbeam_epoch::pin());
            }
        })
        .unwrap()
        .join()
        .unwrap();
}
