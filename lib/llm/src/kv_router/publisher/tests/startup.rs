// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg(test)]
mod tests_startup_helpers {
    use super::*;
    use crate::utils::zmq::{bind_pub_socket, send_multipart};
    use bytes::Bytes;
    use dynamo_kv_router::indexer::{
        GetWorkersRequest, KvIndexer, KvIndexerInterface, WorkerKvQueryResponse,
    };
    use dynamo_kv_router::protocols::{ExternalSequenceBlockHash, LocalBlockHash};
    use std::sync::{Arc, Mutex};

    mod recovery;

    // Type alias to resolve clippy::type_complexity warning
    type PublishedEvents = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

    //--------------------------------------------------------------------
    // A tiny stand-in for Component that just records every publish call
    //--------------------------------------------------------------------
    #[derive(Default)]
    struct MockComponent {
        published: PublishedEvents,
    }

    impl MockComponent {
        fn new() -> (Self, PublishedEvents) {
            let published = Arc::new(Mutex::new(Vec::new()));
            (
                Self {
                    published: published.clone(),
                },
                published,
            )
        }
    }

    impl RouterEventSink for MockComponent {
        fn publish_event(
            &self,
            event: &RouterEvent,
        ) -> impl Future<Output = anyhow::Result<()>> + Send {
            let bytes = rmp_serde::to_vec(event).unwrap();
            self.published
                .lock()
                .unwrap()
                .push((KV_EVENT_SUBJECT.to_string(), bytes));
            async { Ok(()) }
        }
    }

    fn local_gpu_event(worker_id: WorkerId, event: KvCacheEvent) -> PlacementEvent {
        PlacementEvent::local_gpu(worker_id, event)
    }

    //--------------------------------------------------------------------
    // Test start_event_processor
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor() {
        let (component, published) = MockComponent::new();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
            }),
            dp_rank: 0,
        };

        let token = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        tx.send(local_gpu_event(1, event)).unwrap();
        drop(tx);

        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token,
            rx,
            None,
            Some(10_000),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        let published = published.lock().unwrap();
        assert_eq!(published.len(), 1);
        let (subject, _) = &published[0];
        assert_eq!(subject, KV_EVENT_SUBJECT);
    }

    //--------------------------------------------------------------------
    // Test start_event_processor with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_event_processor_with_local_indexer() {
        let (component, published) = MockComponent::new();

        // Create a local indexer
        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // Create BlockStored event
        let event = KvCacheEvent {
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

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        tx.send(local_gpu_event(1, event)).unwrap();
        drop(tx);

        // Start event processor with local indexer
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()), // arc::clone just increments atomic counters
            Some(10_000),
        ));

        // Wait for processing
        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Verify event was published to NATS (same as test_start_event_processor)
        {
            let published_events = published.lock().unwrap();
            assert_eq!(published_events.len(), 1);
            let (subject, _) = &published_events[0];
            assert_eq!(subject, KV_EVENT_SUBJECT);
        } // drop lock

        // Verify event was applied to local indexer
        // We can check by querying the workers that have blocks
        let get_workers_tx = local_indexer.get_workers_sender();
        let mut found = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let (resp_tx, resp_rx) = tokio::sync::oneshot::channel();
            get_workers_tx
                .send(GetWorkersRequest { resp: resp_tx })
                .await
                .unwrap();
            let workers: Vec<u64> = resp_rx.await.unwrap();

            if workers.contains(&1) {
                found = true;
                break;
            }

            // Wait before retrying
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        // Worker 1 should be in the set (we used worker_id=1)
        assert!(
            found,
            "Worker 1 was not found in the indexer after processing"
        );

        // Cleanup
        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test BlockRemoved event with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_block_removed_with_local_indexer() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // First, store a block
        let store_event = KvCacheEvent {
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
        };

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        tx.send(local_gpu_event(1, store_event)).unwrap();

        // Start event processor with local indexer
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()),
            Some(10_000),
        ));

        // Then remove same event
        let remove_event = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(100)],
            }),
            dp_rank: 0,
        };
        tx.send(local_gpu_event(1, remove_event)).unwrap();
        drop(tx);

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Local indexer should have no block
        let mut no_blocks = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let scores = local_indexer
                .find_matches(vec![LocalBlockHash(200)])
                .await
                .unwrap();
            if scores.scores.is_empty() {
                no_blocks = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(no_blocks, "worker should have no blocks after removal");

        // Global kvindexer should have recieved two events (create/remove)
        let published = published.lock().unwrap();
        assert_eq!(
            published.len(),
            2,
            "expected 2 published events, found {}",
            published.len()
        );

        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test AllBlocksCleared event with local indexer
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_all_blocks_cleared_with_local_indexer() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // Store a block
        let store_event = KvCacheEvent {
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
        };

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        tx.send(local_gpu_event(1, store_event)).unwrap();

        // Clear all blocks
        let clear_event = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Cleared,
            dp_rank: 0,
        };
        tx.send(local_gpu_event(1, clear_event)).unwrap();
        drop(tx);

        // Create event processor and wait
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            token.clone(),
            rx,
            Some(local_indexer.clone()),
            Some(10_000),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Local indexer should have no block
        let mut no_blocks = false;
        for _ in 0..20 {
            // Try up to 20 times (200ms total)
            let scores = local_indexer
                .find_matches(vec![LocalBlockHash(200)])
                .await
                .unwrap();
            if scores.scores.is_empty() {
                no_blocks = true;
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }
        assert!(no_blocks, "worker should have no blocks after clearing");

        // Global kvindexer should have recieved two events (create/remove)
        let published = published.lock().unwrap();
        assert_eq!(
            published.len(),
            2,
            "expected 2 published events, found {}",
            published.len()
        );

        token.cancel();
    }

    //--------------------------------------------------------------------
    // Test that local indexer failure doesn't break NATS publishing
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_processor_local_indexer_failure_continues() {
        let (component, published) = MockComponent::new();

        let token = CancellationToken::new();
        let metrics = Arc::new(KvIndexerMetrics::new_unregistered());
        let local_indexer = Arc::new(LocalKvIndexer::new(token.clone(), 4, metrics, 100));

        // cancel indexer immediately to simulate failure
        token.cancel();

        let event = KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1)],
            }),
            dp_rank: 0,
        };

        let new_token = CancellationToken::new();
        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        tx.send(local_gpu_event(1, event)).unwrap();
        drop(tx);

        // Despite local indexer being cancelled, event processor should continue
        let handle = tokio::spawn(start_event_processor(
            component,
            1,
            new_token,
            rx,
            Some(local_indexer),
            Some(10_000),
        ));

        tokio::time::timeout(tokio::time::Duration::from_secs(1), handle)
            .await
            .unwrap()
            .unwrap();

        // Verify event was still published to NATS despite local indexer failure
        let published_events = published.lock().unwrap();
        assert_eq!(published_events.len(), 1);
    }

    //--------------------------------------------------------------------
    // Test start_zmq_listener with a real ZMQ publisher
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_start_zmq_listener_pushes_to_channel() {
        #[derive(serde::Serialize)]
        struct MapBlockStoredEvent {
            #[serde(rename = "type")]
            event_type: &'static str,
            block_hashes: Vec<u64>,
            parent_block_hash: Option<u64>,
            token_ids: Vec<u32>,
            block_size: usize,
            group_idx: Option<u32>,
            kv_cache_spec_kind: Option<&'static str>,
        }

        // Prepare channel that listener should fill
        let (tx, mut rx) = mpsc::unbounded_channel::<PlacementEvent>();

        // Keep the unique IPC directory alive until the sockets shut down.
        let (_ipc_dir, endpoint) = unique_ipc_endpoint();
        let topic = "".to_string(); // subscribe to all

        // Publisher side - set up first
        let pub_socket = bind_pub_socket(&endpoint).await.unwrap();

        // Cancellation token so we can stop the listener
        let token = dynamo_runtime::CancellationToken::new();
        // Event ID counter for the test listener
        let next_event_id = Arc::new(AtomicU64::new(0));

        // Spawn async listener (connects to publisher bound above)
        let listener_handle = tokio::spawn({
            let token = token.clone();
            start_zmq_listener(
                endpoint.to_string(),
                topic,
                1,
                tx,
                token,
                4,
                next_event_id,
                None,
            )
        });

        // Build synthetic 3-frame message: [topic, seq(8B), payload]
        let seq: u64 = 77;

        let events = vec![
            MapBlockStoredEvent {
                event_type: "BlockStored",
                block_hashes: vec![41],
                parent_block_hash: None,
                token_ids: vec![0, 1, 2, 3],
                block_size: 4,
                group_idx: Some(1),
                kv_cache_spec_kind: Some("mamba"),
            },
            MapBlockStoredEvent {
                event_type: "BlockStored",
                block_hashes: vec![42],
                parent_block_hash: None,
                token_ids: vec![0, 1, 2, 3],
                block_size: 4,
                group_idx: None,
                kv_cache_spec_kind: None,
            },
        ];

        let batch = (0.0, events, Some(1_i32));

        let payload = Bytes::from(rmps::to_vec_named(&batch).unwrap());

        let frames = vec![
            Bytes::from("").to_vec(),
            Bytes::from(seq.to_be_bytes().to_vec()).to_vec(),
            payload.clone().to_vec(),
        ];

        // Republish on a 50ms interval until the listener forwards an event
        // (or the 5s deadline trips). ZMQ PUB drops messages destined for
        // subscribers whose SUBSCRIBE handshake has not yet completed, so a
        // one-shot send + fixed sleep is racy on contended runners.
        let event = tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
            let mut publish_interval =
                tokio::time::interval(tokio::time::Duration::from_millis(50));
            loop {
                tokio::select! {
                    event = rx.recv() => {
                        return event.expect("listener channel closed").event;
                    }
                    _ = publish_interval.tick() => {
                        send_multipart(&pub_socket, frames.clone())
                            .await
                            .expect("failed to send ZMQ test event");
                    }
                }
            }
        })
        .await
        .expect("timed out waiting for listener event");

        assert_eq!(event.event_id, 0);

        let KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            start_position,
            blocks,
        }) = event.data
        else {
            panic!("expected KvCacheStoreData");
        };

        assert!(parent_hash.is_none());
        assert!(start_position.is_none());
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_hash.0, 42);

        // Stop the listener
        token.cancel();
        let _ = listener_handle.await;
    }

    #[tokio::test]
    async fn test_start_zmq_listener_connects_before_publisher_bind() {
        let (tx, mut rx) = mpsc::unbounded_channel::<PlacementEvent>();
        // Keep the unique IPC directory alive until the sockets shut down.
        let (_ipc_dir, endpoint) = unique_ipc_endpoint();
        let topic = String::new();
        let token = dynamo_runtime::CancellationToken::new();
        let next_event_id = Arc::new(AtomicU64::new(0));

        let listener_handle = tokio::spawn({
            let token = token.clone();
            let endpoint = endpoint.clone();
            start_zmq_listener(endpoint, topic, 1, tx, token, 4, next_event_id, None)
        });

        tokio::time::sleep(tokio::time::Duration::from_millis(150)).await;
        let pub_socket = bind_pub_socket(&endpoint).await.unwrap();
        let batch = KvEventBatch {
            ts: 0.0,
            events: vec![RawKvEvent::BlockStored {
                block_hashes: vec![BlockHashValue::Unsigned(64)],
                parent_block_hash: None,
                token_ids: vec![4, 5, 6, 7],
                block_size: 4,
                medium: None,
                lora_name: None,
                cache_namespace: None,
                block_mm_infos: None,
                is_eagle: None,
                group_idx: None,
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
            }],
            data_parallel_rank: Some(0),
        };
        let payload = rmps::to_vec(&batch).unwrap();

        let event = tokio::time::timeout(tokio::time::Duration::from_secs(5), async {
            let mut publish_interval =
                tokio::time::interval(tokio::time::Duration::from_millis(50));
            loop {
                tokio::select! {
                    event = rx.recv() => {
                        return event.expect("listener channel closed").event;
                    }
                    _ = publish_interval.tick() => {
                        send_multipart(
                            &pub_socket,
                            vec![Vec::new(), 12u64.to_be_bytes().to_vec(), payload.clone()],
                        )
                        .await
                        .expect("failed to send ZMQ test event");
                    }
                }
            }
        })
        .await
        .expect("timed out waiting for listener event");

        let KvCacheEventData::Stored(KvCacheStoreData { blocks, .. }) = event.data else {
            panic!("expected KvCacheStoreData");
        };
        assert_eq!(blocks[0].block_hash.0, 64);

        token.cancel();
        let _ = listener_handle.await;
    }

    fn unique_ipc_endpoint() -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().expect("failed to create temporary ZMQ directory");
        let endpoint = format!("ipc://{}", dir.path().join("events.sock").display());
        (dir, endpoint)
    }
}
