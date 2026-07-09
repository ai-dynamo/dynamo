// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn test_distributed_kvindexer_recovery_from_outage() {
    let worker_1_id = 1u64;
    let block_size = 4u32;
    let token = CancellationToken::new();

    let (worker_component, worker_published) = MockComponent::new();
    let local_indexer_1 = Arc::new(LocalKvIndexer::new(
        token.clone(),
        block_size,
        Arc::new(KvIndexerMetrics::new_unregistered()),
        100,
    ));
    let (worker_tx, worker_rx) = mpsc::unbounded_channel::<PlacementEvent>();
    tokio::spawn(start_event_processor(
        worker_component,
        worker_1_id,
        token.clone(),
        worker_rx,
        Some(local_indexer_1.clone()),
        Some(10),
    ));

    let router_indexer = Arc::new(KvIndexer::new(
        token.clone(),
        block_size,
        Arc::new(KvIndexerMetrics::new_unregistered()),
    ));
    let event_1 = KvCacheEvent {
        event_id: 1,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: vec![
                KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                },
                KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(101),
                    tokens_hash: LocalBlockHash(201),
                    mm_extra_info: None,
                },
            ],
        }),
        dp_rank: 0,
    };

    worker_tx
        .send(local_gpu_event(worker_1_id, event_1.clone()))
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let (subject, bytes) = {
        let published = worker_published.lock().unwrap();
        assert_eq!(published.len(), 1, "Worker should have published 1 event");
        (published[0].0.clone(), published[0].1.clone())
    };
    assert_eq!(subject, KV_EVENT_SUBJECT);

    let router_event: RouterEvent = rmp_serde::from_slice(&bytes).unwrap();
    router_indexer
        .event_sender()
        .send(router_event)
        .await
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let get_workers_tx = router_indexer.get_workers_sender();
    let mut router_has_worker = false;
    for _ in 0..20 {
        let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
        get_workers_tx
            .send(GetWorkersRequest { resp: resp_tx })
            .await
            .unwrap();
        let workers: Vec<u64> = resp_rx.await.unwrap();
        if workers.contains(&worker_1_id) {
            router_has_worker = true;
            break;
        }
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }
    assert!(
        router_has_worker,
        "Router should see worker 1 after normal operation"
    );

    match local_indexer_1.get_events_in_id_range(Some(1), None).await {
        WorkerKvQueryResponse::Events { events, .. } => {
            assert_eq!(events.len(), 1, "Local indexer should buffer 1 event");
        }
        other => panic!("Expected buffered events, got {other:?}"),
    }

    let event_2 = KvCacheEvent {
        event_id: 2,
        data: KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: vec![
                KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                },
                KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(102),
                    tokens_hash: LocalBlockHash(202),
                    mm_extra_info: None,
                },
            ],
        }),
        dp_rank: 0,
    };

    worker_tx
        .send(local_gpu_event(worker_1_id, event_2.clone()))
        .unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    {
        let published = worker_published.lock().unwrap();
        assert_eq!(
            published.len(),
            2,
            "Worker should have published 2 events total"
        );
    }

    match local_indexer_1.get_events_in_id_range(Some(1), None).await {
        WorkerKvQueryResponse::Events { events, .. } => {
            assert_eq!(
                events.len(),
                2,
                "Local indexer should have both events during outage"
            );
        }
        other => panic!("Expected buffered events, got {other:?}"),
    }

    let block_hashes_2 = vec![LocalBlockHash(200), LocalBlockHash(202)];
    let overlap = router_indexer
        .find_matches(block_hashes_2.clone())
        .await
        .unwrap();
    let router_overlap = overlap
        .scores
        .get(&dynamo_kv_router::protocols::WorkerWithDpRank::from_worker_id(worker_1_id))
        .copied()
        .unwrap_or(0);
    assert_eq!(
        router_overlap, 1,
        "Router should only see 1 shared block (not the new block from event_2)"
    );

    let response = local_indexer_1.get_events_in_id_range(Some(2), None).await;
    let missed_events = match response {
        WorkerKvQueryResponse::Events { events, .. }
        | WorkerKvQueryResponse::TreeDump { events, .. } => events,
        WorkerKvQueryResponse::Error(message) => {
            panic!("Unexpected error response: {message}")
        }
        other => panic!("Unexpected response: {other:?}"),
    };
    assert_eq!(
        missed_events.len(),
        1,
        "Should get 1 missed event (event_2 with id=2)"
    );

    for router_event in missed_events {
        router_indexer
            .event_sender()
            .send(router_event)
            .await
            .unwrap();
    }
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let overlap = router_indexer.find_matches(block_hashes_2).await.unwrap();
    let router_overlap_after = overlap
        .scores
        .get(&dynamo_kv_router::protocols::WorkerWithDpRank::from_worker_id(worker_1_id))
        .copied()
        .unwrap_or(0);
    assert_eq!(
        router_overlap_after, 2,
        "Router should now see both blocks after recovery"
    );

    token.cancel();
}
