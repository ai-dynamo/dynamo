// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `LocalKvIndexer` and worker KV query tests. They stay here with `local.rs` and the
//! request-plane DTOs until that slice moves to `dynamo-kv-indexer`.

// ============================================================================
// LocalKvIndexer tests
// ============================================================================

mod local_indexer_tests {
    use std::sync::Arc;
    use std::time::Duration;

    use tokio::time;
    use tokio_util::sync::CancellationToken;

    use crate::prelude::*;
    use crate::testing::make_store_event_with_dp_rank;
    use crate::*;

    fn make_local_store_event(event_id: u64, block_hash: u64) -> RouterEvent {
        RouterEvent::new(
            0,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(block_hash),
                        tokens_hash: LocalBlockHash(block_hash),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        )
    }

    fn make_local_remove_event(event_id: u64, block_hashes: &[u64]) -> RouterEvent {
        RouterEvent::new(
            0,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: block_hashes
                        .iter()
                        .copied()
                        .map(ExternalSequenceBlockHash)
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    fn make_local_clear_event(event_id: u64) -> RouterEvent {
        RouterEvent {
            worker_id: 0,
            state_source: None,
            session_id: None,
            storage_tier: StorageTier::Device,
            residency_domain: WireResidencyDomain::default(),
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
        }
    }

    #[tokio::test]
    async fn test_local_indexer_get_events_in_id_range_all_cases() {
        // Create indexer with small buffer (5 events max)
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        // Helper to create a test event
        let make_event = |id: u64| {
            RouterEvent::new(
                0,
                KvCacheEvent {
                    event_id: id,
                    data: KvCacheEventData::Stored(KvCacheStoreData {
                        parent_hash: None,
                        start_position: None,
                        blocks: vec![KvCacheStoredBlockData {
                            block_hash: ExternalSequenceBlockHash(id * 100),
                            tokens_hash: LocalBlockHash(id * 200),
                            mm_extra_info: None,
                        }],
                    }),
                    dp_rank: 0,
                },
            )
        };

        // Add 10 events (IDs 5-14), buffer keeps last 5: events 10-14
        for id in 5..15 {
            indexer
                .apply_event_with_buffer(make_event(id))
                .await
                .unwrap();
        }

        // Wait for events to be processed
        indexer.flush().await;

        let extract_events = |resp: WorkerKvQueryResponse| -> Vec<RouterEvent> {
            match resp {
                WorkerKvQueryResponse::Events { events: e, .. } => e,
                WorkerKvQueryResponse::TreeDump { events: e, .. } => e,
                _ => panic!("Unexpected response type: {:?}", resp),
            }
        };

        let extract_last_event_id = |resp: &WorkerKvQueryResponse| -> Option<u64> {
            match resp {
                WorkerKvQueryResponse::Events { last_event_id, .. } => Some(*last_event_id),
                WorkerKvQueryResponse::TreeDump { last_event_id, .. } => Some(*last_event_id),
                _ => None,
            }
        };

        let get_ids = |events: Vec<RouterEvent>| -> Vec<u64> {
            events.iter().map(|e| e.event.event_id).collect()
        };

        // Verify buffer state
        let buffer_events = indexer.get_all_events_in_buffer();
        assert_eq!(get_ids(buffer_events), vec![10, 11, 12, 13, 14]);

        // Buffer hits return the contiguous suffix through the buffered tail.
        let result = indexer.get_events_in_id_range(Some(11), None).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(10), Some(14)).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![10, 11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(11), Some(12)).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(11), Some(20)).await;
        assert_eq!(
            get_ids(extract_events(result.clone())),
            vec![11, 12, 13, 14]
        );
        assert_eq!(extract_last_event_id(&result), Some(14));

        let result = indexer.get_events_in_id_range(Some(12), Some(12)).await;
        assert_eq!(get_ids(extract_events(result.clone())), vec![12, 13, 14]);
        assert_eq!(extract_last_event_id(&result), Some(14));

        // Tree dump path tests
        let result = indexer.get_events_in_id_range(None, None).await;
        match result {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                ..
            } => {
                assert_eq!(events.len(), 10);
                assert_eq!(last_event_id, 14);
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }

        let result = indexer.get_events_in_id_range(Some(7), None).await;
        match result {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                ..
            } => {
                assert_eq!(events.len(), 10);
                assert_eq!(last_event_id, 14);
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }

        // Edge cases
        let result = indexer.get_events_in_id_range(Some(15), Some(10)).await;
        assert!(matches!(result, WorkerKvQueryResponse::InvalidRange { .. }));

        let result = indexer.get_events_in_id_range(Some(100), Some(200)).await;
        assert!(matches!(result, WorkerKvQueryResponse::TooNew { .. }));

        let empty_indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        let result = empty_indexer.get_events_in_id_range(None, None).await;
        match result {
            WorkerKvQueryResponse::TreeDump {
                last_event_id,
                events,
                ..
            } => {
                assert_eq!(
                    last_event_id, 0,
                    "empty buffer should return last_event_id=0"
                );
                assert!(events.is_empty(), "empty indexer should have no events");
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_local_indexer_buffer_response_starts_at_last_all_domain_clear() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            16,
        );

        for event in [
            make_local_store_event(10, 10),
            make_local_clear_event(11),
            make_local_store_event(12, 12),
            make_local_store_event(13, 13),
            make_local_clear_event(14),
            make_local_store_event(15, 15),
        ] {
            indexer.apply_event_with_buffer(event).await.unwrap();
        }
        indexer.flush().await;

        let event_ids = |response: WorkerKvQueryResponse| -> (Vec<u64>, u64) {
            match response {
                WorkerKvQueryResponse::Events {
                    events,
                    last_event_id,
                } => (
                    events.iter().map(|event| event.event.event_id).collect(),
                    last_event_id,
                ),
                other => panic!("Expected Events, got: {other:?}"),
            }
        };

        let (ids, last_event_id) = event_ids(indexer.get_events_in_id_range(Some(10), None).await);
        assert_eq!(ids, vec![14, 15]);
        assert_eq!(last_event_id, 15);

        let (ids, last_event_id) = event_ids(indexer.get_events_in_id_range(Some(12), None).await);
        assert_eq!(ids, vec![14, 15]);
        assert_eq!(last_event_id, 15);

        let (ids, last_event_id) = event_ids(indexer.get_events_in_id_range(Some(15), None).await);
        assert_eq!(ids, vec![15]);
        assert_eq!(last_event_id, 15);
    }

    #[tokio::test]
    async fn test_local_indexer_buffer_and_serialization() {
        let worker_id = 42u64;
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token, 4, metrics, 100));

        let test_event = RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        );

        local_indexer
            .apply_event_with_buffer(test_event)
            .await
            .unwrap();

        local_indexer.flush().await;

        let buffered_events = local_indexer.get_all_events_in_buffer();
        assert_eq!(buffered_events.len(), 1);
        assert_eq!(buffered_events[0].worker_id, worker_id);

        // Test serialization round-trip
        let response = WorkerKvQueryResponse::Events {
            events: buffered_events,
            last_event_id: 1,
        };
        let serialized = serde_json::to_vec(&response).unwrap();
        let deserialized: WorkerKvQueryResponse = serde_json::from_slice(&serialized).unwrap();

        let (events, last_event_id) = match deserialized {
            WorkerKvQueryResponse::Events {
                events,
                last_event_id,
            } => (events, last_event_id),
            _ => panic!("Expected Events variant"),
        };
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].worker_id, worker_id);
        assert_eq!(last_event_id, 1);
    }

    #[tokio::test]
    async fn test_local_indexer_does_not_buffer_failed_send() {
        let local_indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        let test_event = RouterEvent::new(
            7,
            KvCacheEvent {
                event_id: 1,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: None,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100),
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            },
        );

        let event_tx = local_indexer.event_sender();
        local_indexer.shutdown();
        event_tx.closed().await;

        let result = local_indexer.apply_event_with_buffer(test_event).await;
        assert!(matches!(result, Err(KvRouterError::IndexerOffline)));
        assert_eq!(local_indexer.buffer_len(), 0);

        match local_indexer.get_events_in_id_range(None, None).await {
            WorkerKvQueryResponse::TreeDumpFailed {
                last_event_id,
                message,
            } => {
                assert_eq!(last_event_id, 0);
                assert_eq!(message, "Indexer is offline");
            }
            other => panic!("Expected TreeDumpFailed, got: {other:?}"),
        }
    }

    #[test]
    fn legacy_named_messagepack_query_defaults_explicit_dump_failure_capability_off() {
        #[derive(serde::Serialize)]
        struct LegacyWorkerKvQueryRequest {
            worker_id: WorkerId,
            dp_rank: DpRank,
            start_event_id: Option<u64>,
            end_event_id: Option<u64>,
        }

        let encoded = rmp_serde::to_vec_named(&LegacyWorkerKvQueryRequest {
            worker_id: 7,
            dp_rank: 3,
            start_event_id: None,
            end_event_id: Some(9),
        })
        .unwrap();
        let decoded: WorkerKvQueryRequest = rmp_serde::from_slice(&encoded).unwrap();

        assert_eq!(decoded.worker_id, 7);
        assert_eq!(decoded.dp_rank, 3);
        assert_eq!(decoded.end_event_id, Some(9));
        assert!(!decoded.supports_tree_dump_failed);
    }

    #[tokio::test]
    async fn test_local_indexer_remove_worker_dp_rank_only_clears_target_rank() {
        let local_indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        local_indexer
            .apply_event_with_buffer(make_store_event_with_dp_rank(7, &[101], 0))
            .await
            .unwrap();
        local_indexer
            .apply_event_with_buffer(make_store_event_with_dp_rank(7, &[202], 1))
            .await
            .unwrap();
        local_indexer.flush().await;

        local_indexer.remove_worker_dp_rank(7, 0).await;
        local_indexer.flush().await;

        let events = local_indexer.dump_events().await.unwrap();
        let mut rank0 = events
            .iter()
            .filter(|event| event.worker_id == 7 && event.event.dp_rank == 0)
            .collect::<Vec<_>>();
        let mut rank1 = events
            .iter()
            .filter(|event| event.worker_id == 7 && event.event.dp_rank == 1)
            .collect::<Vec<_>>();
        rank0.sort_by_key(|event| event.event.event_id);
        rank1.sort_by_key(|event| event.event.event_id);

        assert!(rank0.is_empty());
        assert_eq!(rank1.len(), 1);
        assert!(matches!(
            &rank1[0].event.data,
            KvCacheEventData::Stored(data)
                if data.blocks.first().map(|block| block.block_hash.0) == Some(202)
        ));
    }

    #[tokio::test]
    async fn test_local_indexer_coalesces_concurrent_tree_dumps() {
        let indexer = Arc::new(LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        ));
        indexer.set_dump_build_delay(Some(Duration::from_millis(50)));

        let first = {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.get_events_in_id_range(None, None).await })
        };
        tokio::time::sleep(Duration::from_millis(10)).await;
        let second = {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.get_events_in_id_range(None, None).await })
        };

        let first = first.await.unwrap();
        let second = second.await.unwrap();

        assert!(matches!(first, WorkerKvQueryResponse::TreeDump { .. }));
        assert!(matches!(second, WorkerKvQueryResponse::TreeDump { .. }));
        assert_eq!(indexer.dump_build_count(), 1);
    }

    fn tree_dump_parts(response: WorkerKvQueryResponse) -> (Vec<RouterEvent>, u64) {
        match response {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                ..
            } => (events, last_event_id),
            other => panic!("Expected TreeDump, got: {other:?}"),
        }
    }

    async fn wait_for_dump_build(indexer: &LocalKvIndexer, count: usize) {
        while indexer.dump_build_count() < count {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn test_local_indexer_coalesced_waiter_covers_events_seen_before_waiting() {
        let indexer = Arc::new(LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            16,
        ));
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;
        indexer.set_dump_build_delay(Some(Duration::from_millis(50)));

        let builder = {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.get_events_in_id_range(None, None).await })
        };
        wait_for_dump_build(&indexer, 1).await;

        // The waiter observes these before joining the build based at event 1, so its
        // response must extend through them rather than reuse the builder's response.
        for (event_id, block_hash) in [(2, 202), (3, 303)] {
            indexer
                .apply_event_with_buffer(make_local_store_event(event_id, block_hash))
                .await
                .unwrap();
        }
        let waiter = {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.get_events_in_id_range(None, None).await })
        };

        let (builder_events, builder_last_event_id) = tree_dump_parts(builder.await.unwrap());
        let (waiter_events, waiter_last_event_id) = tree_dump_parts(waiter.await.unwrap());

        assert_eq!(indexer.dump_build_count(), 1);
        assert_eq!(builder_last_event_id, 1);
        assert_eq!(waiter_last_event_id, 3);
        assert_eq!(waiter_events.len(), builder_events.len() + 2);
        assert_eq!(&waiter_events[..builder_events.len()], &builder_events[..]);
        assert_eq!(
            waiter_events[builder_events.len()..]
                .iter()
                .map(|event| event.event.event_id)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    // Real time: the clear waits on the indexer's OS thread, during which a paused clock
    // would auto-advance past the build delay and let the build finish before the clear.
    #[tokio::test]
    async fn test_local_indexer_invalidated_build_serves_requester_without_caching() {
        let indexer = Arc::new(LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            16,
        ));
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;
        indexer.set_dump_build_delay(Some(Duration::from_millis(200)));

        let builder = {
            let indexer = indexer.clone();
            tokio::spawn(async move { indexer.get_events_in_id_range(None, None).await })
        };
        wait_for_dump_build(&indexer, 1).await;
        indexer
            .apply_event_with_buffer(make_local_clear_event(2))
            .await
            .unwrap();

        // The dump is still a valid recovery point at its base; the clear replays after it.
        let (_, last_event_id) = tree_dump_parts(builder.await.unwrap());
        assert_eq!(last_event_id, 1);
        assert_eq!(indexer.dump_build_count(), 1);
        assert!(!indexer.has_cached_recovery_snapshot());

        indexer.set_dump_build_delay(None);
        let (_, last_event_id) = tree_dump_parts(indexer.get_events_in_id_range(None, None).await);
        assert_eq!(last_event_id, 2);
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_frees_expired_snapshot_without_query() {
        // Buffer of 4 gives an append budget of 2 events past the snapshot base.
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            4,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;
        let _ = indexer.get_events_in_id_range(None, None).await;

        for event_id in 2..=3 {
            indexer
                .apply_event_with_buffer(make_local_store_event(event_id, event_id * 101))
                .await
                .unwrap();
            assert!(indexer.has_cached_recovery_snapshot());
        }
        indexer
            .apply_event_with_buffer(make_local_store_event(4, 404))
            .await
            .unwrap();
        assert!(!indexer.has_cached_recovery_snapshot());
    }

    #[tokio::test(start_paused = true)]
    async fn test_local_indexer_reuses_cached_tree_dump_without_time_expiry() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let first = indexer.get_events_in_id_range(None, None).await;
        time::advance(Duration::from_secs(60)).await;
        let second = indexer.get_events_in_id_range(None, None).await;

        assert!(matches!(first, WorkerKvQueryResponse::TreeDump { .. }));
        assert!(matches!(second, WorkerKvQueryResponse::TreeDump { .. }));
        assert_eq!(indexer.dump_build_count(), 1);
    }

    #[tokio::test]
    async fn test_local_indexer_rebuilds_when_cumulative_append_budget_exceeded() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(2, 202))
            .await
            .unwrap();
        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(3, 303))
            .await
            .unwrap();
        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(4, 404))
            .await
            .unwrap();
        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_appends_safe_tail_to_cached_dump() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let first = indexer.get_events_in_id_range(None, None).await;
        assert!(matches!(first, WorkerKvQueryResponse::TreeDump { .. }));
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_remove_event(2, &[101]))
            .await
            .unwrap();

        match indexer.get_events_in_id_range(None, None).await {
            WorkerKvQueryResponse::TreeDump {
                events,
                last_event_id,
                ..
            } => {
                assert_eq!(last_event_id, 2);
                assert!(events.iter().any(|event| event.event.event_id == 2));
                assert!(
                    events
                        .iter()
                        .any(|event| matches!(event.event.data, KvCacheEventData::Removed(_)))
                );
            }
            other => panic!("Expected TreeDump, got: {other:?}"),
        }
        assert_eq!(indexer.dump_build_count(), 1);
    }

    #[tokio::test]
    async fn test_local_indexer_invalidates_cache_on_clear() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_clear_event(2))
            .await
            .unwrap();

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_invalidates_cache_on_event_gap() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(3, 303))
            .await
            .unwrap();

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_invalidates_cache_on_missing_tail_coverage() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            1,
        );
        indexer
            .apply_event_with_buffer(make_local_store_event(1, 101))
            .await
            .unwrap();
        indexer.flush().await;

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 1);

        indexer
            .apply_event_with_buffer(make_local_store_event(2, 202))
            .await
            .unwrap();
        indexer
            .apply_event_with_buffer(make_local_store_event(3, 303))
            .await
            .unwrap();

        let _ = indexer.get_events_in_id_range(None, None).await;
        assert_eq!(indexer.dump_build_count(), 2);
    }

    #[tokio::test]
    async fn test_local_indexer_failed_dump_is_not_cached() {
        let indexer = LocalKvIndexer::new(
            CancellationToken::new(),
            4,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            5,
        );

        let dump_tx = indexer.snapshot_event_sender();
        indexer.shutdown();
        dump_tx.closed().await;

        let first = indexer.get_events_in_id_range(None, None).await;
        let second = indexer.get_events_in_id_range(None, None).await;

        assert_eq!(indexer.dump_build_count(), 2);
        assert!(matches!(
            first,
            WorkerKvQueryResponse::TreeDumpFailed {
                last_event_id: 0,
                ..
            }
        ));
        assert!(matches!(
            second,
            WorkerKvQueryResponse::TreeDumpFailed {
                last_event_id: 0,
                ..
            }
        ));
    }
}
