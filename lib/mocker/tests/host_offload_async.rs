// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler-facing characterization tests for asynchronous host offload.
//!
//! These tests exercise only the public API that a future vLLM scheduler
//! adapter will consume. They use simulation timestamps and normalized events;
//! wall-clock timing is intentionally outside the contract.

use std::sync::{Arc, Mutex};

use dynamo_mocker::host_offload::{
    CompletedTransfer, G1Location, G2Lookup, HostBlockKey, HostOffloadConfig, HostOffloadEvent,
    HostOffloadEventSink, HostOffloadManager, LoadBlock, LoadScheduleOutcome, PrepareStoreOutcome,
    ResolvedHostOffloadPolicy, SourceFenceOutcome, SourceFenceReason, StoreBlock, SubmittedStore,
    TransferId,
};

#[derive(Default)]
struct CapturingSink(Mutex<Vec<HostOffloadEvent>>);

impl CapturingSink {
    fn snapshot(&self) -> Vec<HostOffloadEvent> {
        self.0.lock().unwrap().clone()
    }

    fn drain(&self) -> Vec<HostOffloadEvent> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl HostOffloadEventSink for CapturingSink {
    fn record(&self, event: &HostOffloadEvent) {
        self.0.lock().unwrap().push(event.clone());
    }
}

fn key(value: u8) -> HostBlockKey {
    HostBlockKey::new(0, None, [value; 32])
}

fn source(value: u64) -> G1Location {
    G1Location::new(value)
}

fn store(key_value: u8, source_value: u64) -> StoreBlock {
    StoreBlock {
        key: key(key_value),
        source: source(source_value),
    }
}

fn load(key_value: u8, destination: u64) -> LoadBlock {
    LoadBlock {
        key: key(key_value),
        destination: G1Location::new(destination),
    }
}

fn manager(capacity_blocks: usize, sink: Arc<CapturingSink>) -> HostOffloadManager {
    HostOffloadManager::with_event_sink(
        HostOffloadConfig {
            capacity_blocks,
            // At 1 GB/s, each block takes exactly one simulation millisecond.
            block_bytes: 1_000_000,
            d2h_bandwidth_gbps: 1.0,
            h2d_bandwidth_gbps: 1.0,
        },
        ResolvedHostOffloadPolicy::vllm_offloading_connector_defaults(),
        sink,
    )
    .unwrap()
}

#[test]
fn repeated_prefix_stays_pending_until_next_step_store_completes() {
    let sink = Arc::new(CapturingSink::default());
    let mut manager = manager(2, sink.clone());
    let blocks = [store(1, 101), store(2, 102)];

    assert_eq!(
        manager.prepare_store(&blocks, 5.0),
        PrepareStoreOutcome::Prepared {
            transfer_id: TransferId::new(0),
            stored_blocks: 2,
        }
    );
    assert!(manager.needs_engine_step());
    assert_eq!(manager.next_deadline(), None);
    assert_eq!(
        manager.lookup(key(1)),
        G2Lookup::Pending {
            transfer_id: TransferId::new(0),
        }
    );
    assert_eq!(
        manager.lookup(key(2)),
        G2Lookup::Pending {
            transfer_id: TransferId::new(0),
        }
    );
    assert_eq!(
        sink.snapshot(),
        vec![
            HostOffloadEvent::StorePrepared {
                at_ms: 5.0,
                transfer_id: TransferId::new(0),
                key: key(1),
                source: source(101),
                bytes: 1_000_000,
            },
            HostOffloadEvent::StorePrepared {
                at_ms: 5.0,
                transfer_id: TransferId::new(0),
                key: key(2),
                source: source(102),
                bytes: 1_000_000,
            },
        ]
    );

    // A repeated prefix lookup/store attempt while D2H is pending neither
    // makes the prefix visible nor creates a duplicate transfer.
    assert!(manager.tick(7.0).completed.is_empty());
    assert_eq!(
        manager.prepare_store(&blocks, 7.0),
        PrepareStoreOutcome::AlreadyPresent
    );
    assert_eq!(
        manager.lookup(key(1)),
        G2Lookup::Pending {
            transfer_id: TransferId::new(0)
        }
    );
    assert_eq!(sink.snapshot().len(), 2);

    assert_eq!(
        manager.submit_prepared_stores(8.0),
        vec![SubmittedStore {
            transfer_id: TransferId::new(0),
            completes_at_ms: 10.0,
        }]
    );
    assert!(!manager.needs_engine_step());
    assert_eq!(manager.next_deadline(), Some(10.0));
    assert!(manager.tick(9.0).completed.is_empty());
    assert_eq!(
        manager.lookup(key(1)),
        G2Lookup::Pending {
            transfer_id: TransferId::new(0)
        }
    );
    assert_eq!(
        manager.schedule_load(&[load(1, 201)], 9.0),
        LoadScheduleOutcome::Miss
    );

    assert_eq!(
        manager.tick(10.0).completed,
        vec![CompletedTransfer::Store {
            transfer_id: TransferId::new(0),
            blocks: blocks.to_vec(),
        }]
    );
    assert_eq!(manager.lookup(key(1)), G2Lookup::Hit);
    assert_eq!(manager.lookup(key(2)), G2Lookup::Hit);
    assert_eq!(manager.resident_snapshot(), vec![key(1), key(2)]);
    assert_eq!(
        sink.snapshot()[2..],
        [
            HostOffloadEvent::StoreQueued {
                at_ms: 8.0,
                transfer_id: TransferId::new(0),
                key: key(1),
                source: source(101),
                bytes: 1_000_000,
            },
            HostOffloadEvent::StoreQueued {
                at_ms: 8.0,
                transfer_id: TransferId::new(0),
                key: key(2),
                source: source(102),
                bytes: 1_000_000,
            },
            HostOffloadEvent::StoreCompleted {
                at_ms: 10.0,
                transfer_id: TransferId::new(0),
                key: key(1),
            },
            HostOffloadEvent::StoreCompleted {
                at_ms: 10.0,
                transfer_id: TransferId::new(0),
                key: key(2),
            },
        ]
    );
}

#[test]
fn d2h_and_h2d_lanes_overlap_but_each_direction_remains_fifo() {
    let sink = Arc::new(CapturingSink::default());
    let mut manager = manager(4, sink.clone());

    manager.prepare_store(&[store(1, 101), store(2, 102)], 0.0);
    manager.submit_prepared_stores(0.0);
    manager.tick(2.0);
    sink.drain();

    manager.prepare_store(&[store(3, 103)], 2.0);
    manager.prepare_store(&[store(4, 104)], 2.0);
    assert_eq!(
        manager.submit_prepared_stores(2.0),
        vec![
            SubmittedStore {
                transfer_id: TransferId::new(1),
                completes_at_ms: 3.0,
            },
            SubmittedStore {
                transfer_id: TransferId::new(2),
                completes_at_ms: 4.0,
            },
        ]
    );
    assert_eq!(
        manager.schedule_load(&[load(1, 201)], 2.0),
        LoadScheduleOutcome::Queued {
            transfer_id: TransferId::new(3),
            completes_at_ms: 3.0,
            loaded_blocks: 1,
        }
    );
    assert_eq!(
        manager.schedule_load(&[load(2, 202)], 2.0),
        LoadScheduleOutcome::Queued {
            transfer_id: TransferId::new(4),
            completes_at_ms: 4.0,
            loaded_blocks: 1,
        }
    );

    // The first D2H and first H2D complete together at 3 ms. Each lane's
    // second job remains serialized until 4 ms.
    assert_eq!(
        manager.tick(3.0).completed,
        vec![
            CompletedTransfer::Store {
                transfer_id: TransferId::new(1),
                blocks: vec![store(3, 103)],
            },
            CompletedTransfer::Load {
                transfer_id: TransferId::new(3),
                blocks: vec![load(1, 201)],
            },
        ]
    );
    assert_eq!(manager.next_deadline(), Some(4.0));
    assert_eq!(
        manager.tick(4.0).completed,
        vec![
            CompletedTransfer::Store {
                transfer_id: TransferId::new(2),
                blocks: vec![store(4, 104)],
            },
            CompletedTransfer::Load {
                transfer_id: TransferId::new(4),
                blocks: vec![load(2, 202)],
            },
        ]
    );

    let events = sink.snapshot();
    let completion_trace: Vec<_> = events
        .iter()
        .filter(|event| {
            matches!(
                event,
                HostOffloadEvent::StoreCompleted { .. } | HostOffloadEvent::LoadCompleted { .. }
            )
        })
        .map(|event| (event.at_ms(), event.key()))
        .collect();
    assert_eq!(
        completion_trace,
        vec![(3.0, key(3)), (3.0, key(1)), (4.0, key(4)), (4.0, key(2)),]
    );
}

#[test]
fn source_reuse_fence_reports_an_explicitly_submitted_store_deadline() {
    let sink = Arc::new(CapturingSink::default());
    let mut manager = manager(2, sink.clone());
    let blocks = [store(1, 101), store(2, 102)];

    manager.prepare_store(&blocks, 0.0);
    assert_eq!(
        manager.fence_sources(&[source(101)], SourceFenceReason::SourceReuse, 0.25),
        SourceFenceOutcome::NeedsSubmission {
            transfer_ids: vec![TransferId::new(0)]
        }
    );
    manager.submit_prepared_stores(0.25);
    let SourceFenceOutcome::Pending(fence) =
        manager.fence_sources(&[source(101)], SourceFenceReason::SourceReuse, 0.25)
    else {
        panic!("submitted store must produce a pending source fence");
    };

    assert_eq!(fence.until_ms, 2.25);
    assert_eq!(fence.transfer_ids, vec![TransferId::new(0)]);
    assert!(!manager.needs_engine_step());
    assert_eq!(manager.next_deadline(), Some(2.25));
    assert!(manager.tick(2.0).completed.is_empty());
    assert_eq!(
        manager.lookup(key(1)),
        G2Lookup::Pending {
            transfer_id: TransferId::new(0)
        }
    );
    assert_eq!(manager.tick(fence.until_ms).completed.len(), 1);
    assert_eq!(manager.lookup(key(1)), G2Lookup::Hit);

    assert_eq!(
        sink.snapshot(),
        vec![
            HostOffloadEvent::StorePrepared {
                at_ms: 0.0,
                transfer_id: TransferId::new(0),
                key: key(1),
                source: source(101),
                bytes: 1_000_000,
            },
            HostOffloadEvent::StorePrepared {
                at_ms: 0.0,
                transfer_id: TransferId::new(0),
                key: key(2),
                source: source(102),
                bytes: 1_000_000,
            },
            HostOffloadEvent::StoreQueued {
                at_ms: 0.25,
                transfer_id: TransferId::new(0),
                key: key(1),
                source: source(101),
                bytes: 1_000_000,
            },
            HostOffloadEvent::StoreQueued {
                at_ms: 0.25,
                transfer_id: TransferId::new(0),
                key: key(2),
                source: source(102),
                bytes: 1_000_000,
            },
            HostOffloadEvent::SourceFenced {
                at_ms: 0.25,
                transfer_id: TransferId::new(0),
                key: key(1),
                source: source(101),
                reason: SourceFenceReason::SourceReuse,
            },
            HostOffloadEvent::StoreCompleted {
                at_ms: 2.25,
                transfer_id: TransferId::new(0),
                key: key(1),
            },
            HostOffloadEvent::StoreCompleted {
                at_ms: 2.25,
                transfer_id: TransferId::new(0),
                key: key(2),
            },
        ]
    );
}

#[test]
fn preemption_caller_flushes_all_stores_then_fences_selected_sources() {
    let sink = Arc::new(CapturingSink::default());
    let mut manager = manager(2, sink.clone());

    manager.prepare_store(&[store(1, 101)], 0.0);
    manager.prepare_store(&[store(2, 102)], 0.0);
    manager.submit_prepared_stores(0.5);
    let SourceFenceOutcome::Pending(fence) =
        manager.fence_sources(&[source(102)], SourceFenceReason::RequestPreemption, 0.5)
    else {
        panic!("submitted store must produce a pending preemption fence");
    };

    // Both worker-local stores are flushed to the FIFO lane. Only the second
    // transfer guards the preempted request's source, so the reported fence is
    // its later deadline.
    assert_eq!(fence.until_ms, 2.5);
    assert_eq!(fence.transfer_ids, vec![TransferId::new(1)]);
    assert!(!manager.needs_engine_step());
    assert_eq!(manager.next_deadline(), Some(1.5));

    let events = sink.snapshot();
    let queued: Vec<_> = events
        .iter()
        .filter_map(|event| match event {
            HostOffloadEvent::StoreQueued {
                at_ms,
                transfer_id,
                key,
                ..
            } => Some((*at_ms, *transfer_id, *key)),
            _ => None,
        })
        .collect();
    assert_eq!(
        queued,
        vec![
            (0.5, TransferId::new(0), key(1)),
            (0.5, TransferId::new(1), key(2)),
        ]
    );
    assert!(events.iter().any(|event| {
        matches!(
            event,
            HostOffloadEvent::SourceFenced {
                at_ms: 0.5,
                transfer_id,
                key: event_key,
                source: event_source,
                reason: SourceFenceReason::RequestPreemption,
            } if *transfer_id == TransferId::new(1)
                && *event_key == key(2)
                && *event_source == source(102)
        )
    }));
    assert!(!events.iter().any(|event| {
        matches!(
            event,
            HostOffloadEvent::SourceFenced { key: event_key, .. }
                if *event_key == key(1)
        )
    }));
}

#[test]
fn capacity_retry_survives_pending_store_and_pinned_load_then_retries_same_block() {
    let sink = Arc::new(CapturingSink::default());
    let mut manager = manager(2, sink.clone());

    manager.prepare_store(&[store(1, 101)], 0.0);
    manager.submit_prepared_stores(0.0);
    manager.tick(1.0);
    sink.drain();

    // One slot is held by an unsubmitted store. The other is resident but
    // pinned by an H2D load, leaving no legal LRU victim.
    manager.prepare_store(&[store(2, 102)], 1.0);
    manager.schedule_load(&[load(1, 201)], 1.0);
    assert_eq!(
        manager.prepare_store(&[store(3, 103)], 1.0),
        PrepareStoreOutcome::RetryCapacity
    );
    assert_eq!(manager.lookup(key(3)), G2Lookup::Miss);
    assert_eq!(manager.used_blocks(), 2);
    assert!(sink.snapshot().iter().any(|event| {
        matches!(
            event,
            HostOffloadEvent::CapacityRetry {
                at_ms: 1.0,
                key: event_key,
            } if *event_key == key(3)
        )
    }));

    manager.submit_prepared_stores(1.0);
    manager.tick(2.0);
    assert_eq!(
        manager.prepare_store(&[store(3, 103)], 2.0),
        PrepareStoreOutcome::Prepared {
            transfer_id: TransferId::new(3),
            stored_blocks: 1,
        }
    );
    assert_eq!(
        manager.lookup(key(3)),
        G2Lookup::Pending {
            transfer_id: TransferId::new(3)
        }
    );
    assert!(!manager.is_resident(key(2)));
    assert!(manager.is_resident(key(1)));

    let events = sink.snapshot();
    let retry_index = events
        .iter()
        .position(|event| matches!(event, HostOffloadEvent::CapacityRetry { .. }))
        .unwrap();
    let eviction_index = events
        .iter()
        .position(|event| {
            matches!(
                event,
                HostOffloadEvent::G2Evicted {
                    at_ms: 2.0,
                    key: event_key,
                } if *event_key == key(2)
            )
        })
        .unwrap();
    let retry_store_index = events
        .iter()
        .position(|event| {
            matches!(
                event,
                HostOffloadEvent::StorePrepared {
                    at_ms: 2.0,
                    transfer_id,
                    key: event_key,
                    ..
                } if *transfer_id == TransferId::new(3) && *event_key == key(3)
            )
        })
        .unwrap();
    assert!(retry_index < eviction_index && eviction_index < retry_store_index);
}

#[test]
fn completed_load_retains_the_g2_copy_for_a_second_load() {
    let sink = Arc::new(CapturingSink::default());
    let mut manager = manager(1, sink.clone());

    manager.prepare_store(&[store(1, 101)], 0.0);
    manager.submit_prepared_stores(0.0);
    manager.tick(1.0);

    assert_eq!(
        manager.schedule_load(&[load(1, 201)], 1.0),
        LoadScheduleOutcome::Queued {
            transfer_id: TransferId::new(1),
            completes_at_ms: 2.0,
            loaded_blocks: 1,
        }
    );
    assert_eq!(manager.lookup(key(1)), G2Lookup::Hit);
    manager.tick(2.0);
    assert_eq!(manager.lookup(key(1)), G2Lookup::Hit);
    assert_eq!(manager.resident_snapshot(), vec![key(1)]);

    assert_eq!(
        manager.schedule_load(&[load(1, 202)], 2.0),
        LoadScheduleOutcome::Queued {
            transfer_id: TransferId::new(2),
            completes_at_ms: 3.0,
            loaded_blocks: 1,
        }
    );
    manager.tick(3.0);
    assert_eq!(manager.lookup(key(1)), G2Lookup::Hit);
    assert_eq!(manager.used_blocks(), 1);
    assert_eq!(manager.resident_blocks(), 1);

    let events = sink.snapshot();
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, HostOffloadEvent::StorePrepared { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter_map(|event| match event {
                HostOffloadEvent::LoadCompleted {
                    at_ms,
                    transfer_id,
                    key: event_key,
                } => Some((*at_ms, *transfer_id, *event_key)),
                _ => None,
            })
            .collect::<Vec<_>>(),
        vec![
            (2.0, TransferId::new(1), key(1)),
            (3.0, TransferId::new(2), key(1)),
        ]
    );
}
