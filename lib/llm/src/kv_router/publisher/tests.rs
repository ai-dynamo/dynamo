// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
#[allow(unused_imports)]
use bytes::Bytes;
#[allow(unused_imports)]
use dynamo_kv_router::RouterEventSink;
#[allow(unused_imports)]
use rmp_serde as rmps;
#[allow(unused_imports)]
use std::future::Future;
#[allow(unused_imports)]
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

#[cfg(test)]
mod test_event_processing {
    use super::*;
    use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};

    // ---------------------------------------------------------------------
    // create_stored_block_from_parts --------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_block_from_parts() {
        let kv_block_size = 4;
        let token_ids = vec![10, 20, 30, 40];
        let blk_hash = 0xdead_beef;

        let stored =
            create_stored_block_from_parts(kv_block_size, blk_hash, &token_ids, None, None, None);

        assert_eq!(stored.block_hash.0, blk_hash);
        let expected_hash =
            compute_block_hash_for_seq(&token_ids, 4, BlockHashOptions::default())[0];
        assert_eq!(stored.tokens_hash, expected_hash);
        assert!(stored.mm_extra_info.is_none());
    }

    // ---------------------------------------------------------------------
    // create_stored_blocks -------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_blocks_ok() {
        let kv_block_size = 4;
        // two blocks, each of size 4
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let num_block_tokens = vec![4_u64, 4_u64];
        let block_hashes = vec![111_u64, 222_u64];

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            None,
            &Arc::new(AtomicU32::new(0)),
            None,
            None,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_hash.0, 111);
        assert_eq!(blocks[1].block_hash.0, 222);
    }

    #[test]
    fn test_create_stored_blocks_wrong_size_triggers_warning() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4, 5, 6, 7];
        let num_block_tokens = vec![4_u64, 3_u64];
        let block_hashes = vec![111_u64, 222_u64];
        let warning_count = Arc::new(AtomicU32::new(0));

        let blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            None,
            &warning_count,
            None,
            None,
        );

        // should early-exit as second has mismatch
        assert!(blocks.len() == 1);
        assert!(warning_count.load(Ordering::Relaxed) == 1)
    }

    // ---------------------------------------------------------------------
    // convert_event --------------------------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_convert_event_block_stored() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10), BlockHashValue::Unsigned(11)],
            parent_block_hash: Some(BlockHashValue::Unsigned(99)),
            token_ids: vec![1, 2, 3, 4, 5, 6, 7, 8],
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };

        let out = convert_event(
            raw_evt,
            42,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &Arc::new(AtomicU32::new(0)),
        )
        .unwrap();
        assert!(matches!(out.event.data, KvCacheEventData::Stored(_)));
    }

    #[test]
    fn test_convert_event_with_lora_name() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4];

        let base_evt = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };
        let lora_evt = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: Some("my-lora".to_string()),
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };

        let wc = Arc::new(AtomicU32::new(0));
        let base_out = convert_event(
            base_evt,
            1,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &wc,
        )
        .unwrap();
        let lora_out = convert_event(
            lora_evt,
            2,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &wc,
        )
        .unwrap();

        let base_hash = match &base_out.event.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        let lora_hash = match &lora_out.event.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        assert_ne!(
            base_hash, lora_hash,
            "LoRA blocks must produce distinct tokens_hash"
        );
    }

    #[test]
    fn test_convert_event_lora_name_none_is_base_model() {
        let kv_block_size = 4;
        let token_ids = vec![1, 2, 3, 4];
        let wc = Arc::new(AtomicU32::new(0));

        let evt1 = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };
        let evt2 = RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(10)],
            parent_block_hash: None,
            token_ids: token_ids.clone(),
            block_size: 4,
            medium: None,
            lora_name: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };

        let out1 = convert_event(
            evt1,
            1,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &wc,
        )
        .unwrap();
        let out2 = convert_event(
            evt2,
            2,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &wc,
        )
        .unwrap();

        let hash1 = match &out1.event.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        let hash2 = match &out2.event.data {
            KvCacheEventData::Stored(s) => s.blocks[0].tokens_hash,
            _ => panic!("expected Stored"),
        };
        assert_eq!(
            hash1, hash2,
            "Two base-model events with same tokens should produce same hash"
        );
    }

    #[test]
    fn test_backward_compat_deserialize_map_with_lora_id_no_lora_name() {
        #[derive(serde::Serialize)]
        struct OldFormatEvent {
            #[serde(rename = "type")]
            event_type: &'static str,
            block_hashes: Vec<u64>,
            parent_block_hash: Option<u64>,
            token_ids: Vec<u32>,
            block_size: usize,
            lora_id: Option<u64>,
        }

        let payload = rmps::to_vec(&OldFormatEvent {
            event_type: "BlockStored",
            block_hashes: vec![42],
            parent_block_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: Some(5),
        })
        .unwrap();

        let event: RawKvEvent = rmps::from_slice(&payload).unwrap();
        let RawKvEvent::BlockStored { lora_name, .. } = event else {
            panic!("expected BlockStored");
        };
        assert!(
            lora_name.is_none(),
            "old-format payloads with lora_id but no lora_name should deserialize with lora_name=None"
        );
    }

    #[test]
    fn test_backward_compat_deserialize_seq_with_lora_id_no_lora_name() {
        let payload = rmps::to_vec(&(
            "BlockStored",
            vec![42_u64],
            None::<u64>,
            vec![1_u32, 2, 3, 4],
            4_usize,
            Some(5_u64), // lora_id at position 5
                         // no medium, no lora_name — simulating an old producer
        ))
        .unwrap();

        let event: RawKvEvent = rmps::from_slice(&payload).unwrap();
        let RawKvEvent::BlockStored { lora_name, .. } = event else {
            panic!("expected BlockStored");
        };
        assert!(
            lora_name.is_none(),
            "old seq-format payloads with lora_id should deserialize with lora_name=None"
        );
    }

    #[test]
    fn test_convert_event_block_removed() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::BlockRemoved {
            block_hashes: vec![BlockHashValue::Unsigned(123), BlockHashValue::Signed(456)],
            medium: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        };
        let out = convert_event(
            raw_evt,
            7,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &Arc::new(AtomicU32::new(0)),
        )
        .unwrap();

        assert!(matches!(out.event.data, KvCacheEventData::Removed(_)));
    }

    #[test]
    fn test_convert_event_all_blocks_cleared() {
        let kv_block_size = 4;
        let raw_evt = RawKvEvent::AllBlocksCleared;
        let out = convert_event(
            raw_evt,
            1,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &Arc::new(AtomicU32::new(0)),
        )
        .unwrap();
        assert!(matches!(out.event.data, KvCacheEventData::Cleared));
    }

    #[test]
    fn test_parse_mm_hash_from_extra_key() {
        assert_eq!(
            parse_mm_hash_from_extra_key(
                "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210"
            ),
            Some(0x0123_4567_89ab_cdef)
        );
        assert_eq!(parse_mm_hash_from_extra_key("123"), None);
        assert_eq!(parse_mm_hash_from_extra_key("not_a_hash"), None);
    }

    #[test]
    fn test_extra_keys_to_block_mm_infos() {
        let mm_hash =
            "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string();
        let infos = extra_keys_to_block_mm_infos(Some(vec![
            Some(vec![ExtraKeyItem::Hash(mm_hash.clone())]),
            None,
            Some(vec![
                ExtraKeyItem::Hash("invalid".to_string()),
                ExtraKeyItem::Hash(mm_hash),
            ]),
        ]))
        .expect("expected parsed MM infos");

        assert_eq!(infos.len(), 3);
        assert_eq!(
            infos[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
        assert!(infos[1].is_none());
        assert_eq!(
            infos[2].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }

    #[test]
    fn test_seq_block_stored_field8_supports_extra_keys() {
        let mm_hash =
            "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string();
        let extra_keys_payload = rmps::to_vec(&(
            "BlockStored",
            vec![10_u64],
            None::<u64>,
            vec![1_u32, 2, 3, 4],
            4_usize,
            None::<u64>,
            None::<String>,
            None::<String>,
            vec![Some(vec![mm_hash])],
        ))
        .unwrap();
        let extra_keys_event: RawKvEvent = rmps::from_slice(&extra_keys_payload).unwrap();
        let RawKvEvent::BlockStored {
            lora_name,
            block_mm_infos,
            ..
        } = extra_keys_event
        else {
            panic!("expected BlockStored");
        };
        assert!(lora_name.is_none());
        assert_eq!(
            block_mm_infos.unwrap()[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }

    #[test]
    fn test_seq_block_stored_field8_supports_tuple_extra_keys() {
        let mm_hash =
            "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string();
        let extra_keys_payload = rmps::to_vec(&(
            "BlockStored",
            vec![10_u64],
            None::<u64>,
            vec![1_u32, 2, 3, 4],
            4_usize,
            None::<u64>,
            None::<String>,
            None::<String>,
            vec![Some(vec![(mm_hash, 7_i64)])],
        ))
        .unwrap();
        let extra_keys_event: RawKvEvent = rmps::from_slice(&extra_keys_payload).unwrap();
        let RawKvEvent::BlockStored { block_mm_infos, .. } = extra_keys_event else {
            panic!("expected BlockStored");
        };
        assert_eq!(
            block_mm_infos.unwrap()[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }

    #[test]
    fn test_map_block_stored_supports_extra_keys() {
        #[derive(serde::Serialize)]
        struct MapBlockStoredEvent {
            #[serde(rename = "type")]
            event_type: &'static str,
            block_hashes: Vec<u64>,
            parent_block_hash: Option<u64>,
            token_ids: Vec<u32>,
            block_size: usize,
            lora_id: Option<u64>,
            medium: Option<String>,
            lora_name: Option<String>,
            extra_keys: Option<Vec<Option<Vec<String>>>>,
        }

        let payload = rmps::to_vec(&MapBlockStoredEvent {
            event_type: "BlockStored",
            block_hashes: vec![10],
            parent_block_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            medium: Some("GPU".to_string()),
            lora_name: None,
            extra_keys: Some(vec![Some(vec![
                "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string(),
            ])]),
        })
        .unwrap();

        let event: RawKvEvent = rmps::from_slice(&payload).unwrap();
        let RawKvEvent::BlockStored { block_mm_infos, .. } = event else {
            panic!("expected BlockStored");
        };
        assert_eq!(
            block_mm_infos.unwrap()[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }

    #[test]
    fn test_map_block_stored_supports_tuple_extra_keys() {
        type BlockTupleExtraKeys = Option<Vec<Option<Vec<(String, i64)>>>>;

        #[derive(serde::Serialize)]
        struct MapBlockStoredEvent {
            #[serde(rename = "type")]
            event_type: &'static str,
            block_hashes: Vec<u64>,
            parent_block_hash: Option<u64>,
            token_ids: Vec<u32>,
            block_size: usize,
            lora_id: Option<u64>,
            medium: Option<String>,
            lora_name: Option<String>,
            extra_keys: BlockTupleExtraKeys,
        }

        let mm_hash =
            "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210".to_string();
        let payload = rmps::to_vec(&MapBlockStoredEvent {
            event_type: "BlockStored",
            block_hashes: vec![10],
            parent_block_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_id: None,
            medium: Some("GPU".to_string()),
            lora_name: None,
            extra_keys: Some(vec![Some(vec![(mm_hash, 3)])]),
        })
        .unwrap();

        let event: RawKvEvent = rmps::from_slice(&payload).unwrap();
        let RawKvEvent::BlockStored { block_mm_infos, .. } = event else {
            panic!("expected BlockStored");
        };
        assert_eq!(
            block_mm_infos.unwrap()[0].as_ref().unwrap().mm_objects[0].mm_hash,
            0x0123_4567_89ab_cdef
        );
    }
}

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
    // Test start_zmq_listener without a real socket
    //   (feed it frames through a ZMQ PAIR tcp socket)
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

        // ZMQ TCP endpoint using localhost with an ephemeral port
        let reserved_listener = reserve_open_port();
        let endpoint = format!(
            "tcp://127.0.0.1:{}",
            reserved_listener
                .local_addr()
                .expect("failed to read reserved listener address")
                .port()
        );
        drop(reserved_listener);
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
            start_zmq_listener(endpoint.to_string(), topic, 1, tx, token, 4, next_event_id)
        });

        // Give time for the connection to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send synthetic 3-frame message: [topic, seq(8B), payload]
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

        // Send the multipart message
        send_multipart(&pub_socket, frames).await.unwrap();

        // Wait for message to be received
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Check that we received the message
        let event = rx.try_recv().expect("no message received").event;
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
        let reserved_listener = reserve_open_port();
        let endpoint = format!(
            "tcp://127.0.0.1:{}",
            reserved_listener
                .local_addr()
                .expect("failed to read reserved listener address")
                .port()
        );
        drop(reserved_listener);
        let topic = String::new();
        let token = dynamo_runtime::CancellationToken::new();
        let next_event_id = Arc::new(AtomicU64::new(0));

        let listener_handle = tokio::spawn({
            let token = token.clone();
            let endpoint = endpoint.clone();
            start_zmq_listener(endpoint, topic, 1, tx, token, 4, next_event_id)
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
                block_mm_infos: None,
                is_eagle: None,
                group_idx: None,
                kv_cache_spec_kind: None,
                kv_cache_spec_sliding_window: None,
            }],
            data_parallel_rank: Some(0),
        };
        let payload = rmps::to_vec(&batch).unwrap();

        for _ in 0..5 {
            send_multipart(
                &pub_socket,
                vec![Vec::new(), 12u64.to_be_bytes().to_vec(), payload.clone()],
            )
            .await
            .unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        let event = tokio::time::timeout(tokio::time::Duration::from_secs(5), rx.recv())
            .await
            .expect("timed out waiting for listener event")
            .expect("listener channel closed")
            .event;

        let KvCacheEventData::Stored(KvCacheStoreData { blocks, .. }) = event.data else {
            panic!("expected KvCacheStoreData");
        };
        assert_eq!(blocks[0].block_hash.0, 64);

        token.cancel();
        let _ = listener_handle.await;
    }

    fn reserve_open_port() -> std::net::TcpListener {
        std::net::TcpListener::bind("127.0.0.1:0").expect("failed to bind probe listener")
    }

    //--------------------------------------------------------------------
    // Test distributed recovery: Router queries worker's LocalKvIndexer after outage
    //--------------------------------------------------------------------
    #[tokio::test]
    async fn test_distributed_kvindexer_recovery_from_outage() {
        let worker_1_id = 1u64;
        let block_size = 4u32;
        let token = CancellationToken::new();

        // === SETUP: Worker Components ===
        let (worker_component, worker_published) = MockComponent::new();
        let local_indexer_1 = Arc::new(LocalKvIndexer::new(
            token.clone(),
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
            100, // buffer size
        ));

        let (worker_tx, worker_rx) = mpsc::unbounded_channel::<PlacementEvent>();

        // Start worker's event processor
        tokio::spawn(start_event_processor(
            worker_component,
            worker_1_id,
            token.clone(),
            worker_rx,
            Some(local_indexer_1.clone()),
            Some(10), // 10ms batching timeout
        ));

        // === SETUP: Router Components ===
        let router_indexer = Arc::new(KvIndexer::new(
            token.clone(),
            block_size,
            Arc::new(KvIndexerMetrics::new_unregistered()),
        ));

        // === STEP 1: Normal Operation ===
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

        // Simulate JetStream: forward worker's published event to router
        let (subject, bytes) = {
            let published = worker_published.lock().unwrap();
            assert_eq!(published.len(), 1, "Worker should have published 1 event");
            (published[0].0.clone(), published[0].1.clone())
        }; // drop worker_published before await
        assert_eq!(subject, KV_EVENT_SUBJECT);

        let router_event: RouterEvent = rmp_serde::from_slice(&bytes).unwrap();
        router_indexer
            .event_sender()
            .send(router_event)
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Router's indexer has event
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

        // assert: Worker's local indexer buffered event
        match local_indexer_1.get_events_in_id_range(Some(1), None).await {
            WorkerKvQueryResponse::Events { events, .. } => {
                assert_eq!(events.len(), 1, "Local indexer should buffer 1 event");
            }
            other => panic!("Expected buffered events, got {other:?}"),
        }

        // === STEP 2 & 3: Simulate Outage - Stop forwarding to router ===
        let event_2 = KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: None,
                blocks: vec![
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(100), // Shared prefix
                        tokens_hash: LocalBlockHash(200),
                        mm_extra_info: None,
                    },
                    KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(102), // New block
                        tokens_hash: LocalBlockHash(202),
                        mm_extra_info: None,
                    },
                ],
            }),
            dp_rank: 0,
        };

        worker_tx
            .send(local_gpu_event(worker_1_id, event_2.clone()))
            .unwrap(); // send to worker but not to router
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Worker published event_2 to "NATS" (MockComponent)
        {
            let published = worker_published.lock().unwrap();
            assert_eq!(
                published.len(),
                2,
                "Worker should have published 2 events total"
            );
        }

        // assert: Worker's local indexer has both events
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

        // assert: Router DOESN'T have event_2
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

        // === STEP 4 & 5: Recovery - Query worker's local indexer for missed events ===
        // In practice, the subscriber detects gaps and triggers recovery automatically.
        // Here we simulate that by querying for events after event_id=1.
        let last_known_id = 1u64; // Router only received event_1
        let response = local_indexer_1
            .get_events_in_id_range(Some(last_known_id + 1), None)
            .await;
        let missed_events = match response {
            dynamo_kv_router::indexer::WorkerKvQueryResponse::Events { events: e, .. } => e,
            dynamo_kv_router::indexer::WorkerKvQueryResponse::TreeDump { events: e, .. } => e,
            dynamo_kv_router::indexer::WorkerKvQueryResponse::Error(message) => {
                panic!("Unexpected error response: {message}")
            }
            other => panic!("Unexpected response: {:?}", other),
        };
        assert_eq!(
            missed_events.len(),
            1,
            "Should get 1 missed event (event_2 with id=2)"
        );

        // Step 5: Apply missed events to router
        for router_event in missed_events {
            router_indexer
                .event_sender()
                .send(router_event)
                .await
                .unwrap();
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // assert: Router now has complete state
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
}

#[cfg(test)]
mod test_event_dedup_filter {
    use super::*;

    fn store_data(hashes: &[u64]) -> KvCacheStoreData {
        KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: hashes
                .iter()
                .map(|&h| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(h),
                    tokens_hash: LocalBlockHash(h * 10),
                    mm_extra_info: None,
                })
                .collect(),
        }
    }

    fn remove_data(hashes: &[u64]) -> KvCacheRemoveData {
        KvCacheRemoveData {
            block_hashes: hashes
                .iter()
                .map(|&h| ExternalSequenceBlockHash(h))
                .collect(),
        }
    }

    #[test]
    fn stores_track_refcounts_for_removes() {
        let mut filter = EventDedupFilter::new();
        let data = store_data(&[1, 2, 3]);

        // Store same hashes twice — refcount should be 2
        filter.track_store(0, StorageTier::Device, &data);
        filter.track_store(0, StorageTier::Device, &data);

        // First remove — refcounts 2→1, all filtered out
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1, 2, 3]));
        assert!(result.is_none());

        // Second remove — refcounts 1→0, all pass through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1, 2, 3]));
        assert!(result.is_some());
        assert_eq!(result.unwrap().block_hashes.len(), 3);
    }

    #[test]
    fn duplicate_removes_are_filtered() {
        let mut filter = EventDedupFilter::new();

        // Store same hash twice
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // First remove — refcount 2→1, filtered out
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_none());

        // Second remove — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
        assert_eq!(result.unwrap().block_hashes.len(), 1);
    }

    #[test]
    fn store_remove_store_cycle() {
        let mut filter = EventDedupFilter::new();

        // Store hash 1
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // Remove hash 1 — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());

        // Store hash 1 again — refcount starts fresh at 1
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // Remove again — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
    }

    #[test]
    fn clear_resets_all_ranks() {
        let mut filter = EventDedupFilter::new();

        // Store on rank 0 and rank 1
        filter.track_store(0, StorageTier::Device, &store_data(&[1, 2]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1, 2]));
        filter.track_store(1, StorageTier::Device, &store_data(&[1, 2]));
        filter.track_store(1, StorageTier::Device, &store_data(&[1, 2]));

        // Clear wipes all ranks (matches indexer semantics where Cleared
        // from any rank removes all blocks for the entire worker).
        filter.clear();

        // Both ranks pass through defensively after clear
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());

        let result = filter.filter_remove(1, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
    }

    #[test]
    fn mixed_blocks_in_single_remove() {
        let mut filter = EventDedupFilter::new();

        // Hash 1: stored twice (refcount 2)
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));

        // Hash 2: stored once (refcount 1)
        filter.track_store(0, StorageTier::Device, &store_data(&[2]));

        // Hash 3: stored twice (refcount 2)
        filter.track_store(0, StorageTier::Device, &store_data(&[3]));
        filter.track_store(0, StorageTier::Device, &store_data(&[3]));

        // Remove all three — only hash 2 (refcount 1→0) passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1, 2, 3]));
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.block_hashes.len(), 1);
        assert_eq!(result.block_hashes[0], ExternalSequenceBlockHash(2));
    }

    #[test]
    fn same_hash_on_different_ranks_are_independent() {
        let mut filter = EventDedupFilter::new();

        // Store hash 1 on rank 0 (twice) and rank 1 (once)
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(0, StorageTier::Device, &store_data(&[1]));
        filter.track_store(1, StorageTier::Device, &store_data(&[1]));

        // Remove hash 1 on rank 1 — refcount 1→0, passes through
        let result = filter.filter_remove(1, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());

        // Remove hash 1 on rank 0 — refcount 2→1, filtered out
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_none());

        // Remove hash 1 on rank 0 again — refcount 1→0, passes through
        let result = filter.filter_remove(0, StorageTier::Device, remove_data(&[1]));
        assert!(result.is_some());
    }
}

#[cfg(all(test, feature = "integration"))]
mod test_integration_publisher {
    use super::*;
    use crate::kv_router::KV_METRICS_SUBJECT;
    use dynamo_kv_router::protocols::ActiveLoad;
    use dynamo_runtime::distributed_test_utils::create_test_drt_async;
    use dynamo_runtime::transports::event_plane::EventSubscriber;

    #[tokio::test]
    #[ignore] // Mark as ignored as requested, because CI's integrations still don't have NATS
    async fn test_metrics_publishing_behavior() -> Result<()> {
        // Set up runtime and namespace
        let drt = create_test_drt_async().await;
        let namespace = drt.namespace("ns2001".to_string())?;

        // Create a subscriber for the metrics events
        let mut subscriber = EventSubscriber::for_namespace(&namespace, KV_METRICS_SUBJECT)
            .await
            .unwrap()
            .typed::<ActiveLoad>();

        // Create WorkerMetricsPublisher
        let publisher = WorkerMetricsPublisher::new().unwrap();
        let worker_id = 1234;

        // Start NATS metrics publishing
        publisher.start_nats_metrics_publishing(namespace.clone(), worker_id);

        // Allow some time for the background task to start
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Test 1: Publish 10 different metrics with 0.5ms intervals
        // Only the last one should be published after 1ms of stability
        for i in 0..10 {
            let value = (i * 100) as u64;
            publisher.publish(None, None, Some(value)).unwrap();
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait a bit more than 1ms to ensure the last metric is published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify we receive exactly one event with the last metric values
        let result =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), subscriber.next())
                .await
                .unwrap();

        let (_envelope, event) = result.unwrap().unwrap(); // Unwrap the Option and the Result
        assert_eq!(event.worker_id, worker_id);
        assert_eq!(event.active_decode_blocks, None); // Worker publisher sends kv_used_blocks
        assert_eq!(event.active_prefill_tokens, None); // Worker doesn't publish prefill tokens
        assert_eq!(event.kv_used_blocks, Some(900));

        // Ensure no more events are waiting
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(no_msg.is_err(), "Expected no more messages, but found one");

        // Test 2: Publish 10 more metrics with same active_decode_blocks - should not trigger publish
        for _ in 0..10 {
            publisher.publish(None, None, Some(900)).unwrap(); // Keep same as last published
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Wait to ensure no events are published
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Verify no events are received
        let no_msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(50), subscriber.next()).await;
        assert!(
            no_msg.is_err(),
            "Expected no messages when load metrics don't change"
        );

        drt.shutdown();

        Ok(())
    }
}

#[cfg(test)]
mod batching_state_tests {
    use super::*;

    #[test]
    fn test_batching_state_default() {
        let state = BatchingState::new();
        assert!(!state.has_pending(), "Default state should have no pending");
        assert!(
            state.pending_removed.is_none(),
            "Default pending_removed should be None"
        );
        assert!(
            state.pending_stored.is_none(),
            "Default pending_stored should be None"
        );
    }

    #[test]
    fn test_batching_state_new() {
        let state = BatchingState::new();
        // last_flush_time should be set to approximately now
        let elapsed = state.last_flush_time.elapsed();
        assert!(
            elapsed < Duration::from_secs(1),
            "new() should create state with flush time set to approximately now"
        );
    }

    #[test]
    fn test_batching_state_pending_removed() {
        let mut state = BatchingState::new();
        assert!(!state.has_pending(), "Should not have pending initially");

        state.pending_removed = Some(KvCacheRemoveData {
            block_hashes: vec![],
        });
        assert!(
            state.has_pending(),
            "Should have pending after setting pending_removed"
        );
    }

    #[test]
    fn test_batching_state_pending_stored() {
        let mut state = BatchingState::new();
        assert!(!state.has_pending(), "Should not have pending initially");

        state.pending_stored = Some(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: vec![],
        });
        assert!(
            state.has_pending(),
            "Should have pending after setting pending_stored"
        );
    }

    #[test]
    fn test_batching_state_timeout() {
        let mut state = BatchingState::new();

        // Reset flush time to now so we can test timeout behavior
        state.record_flush_time();

        // Test that remaining returns positive initially (10ms timeout)
        let remaining_before = state.remaining_timeout(10);
        assert!(
            remaining_before.as_millis() > 0,
            "Should have remaining time initially"
        );

        // Test zero timeout returns zero
        let remaining_zero = state.remaining_timeout(0);
        assert_eq!(
            remaining_zero.as_millis(),
            0,
            "0 timeout should return zero"
        );
    }

    #[test]
    fn test_batching_state_record_flush_time() {
        let mut state = BatchingState::new();

        let initial_time = state.last_flush_time;

        state.record_flush_time();

        assert!(
            state.last_flush_time >= initial_time,
            "record_flush_time should update the time"
        );
    }

    #[test]
    fn test_batching_state_remaining_timeout() {
        let mut state = BatchingState::new();

        // Reset flush time to now so we can test timeout behavior
        state.record_flush_time();

        // Test that remaining returns positive initially (10ms timeout)
        let remaining = state.remaining_timeout(10);
        assert!(
            remaining.as_millis() > 0,
            "Should have remaining time initially"
        );

        // Test that with 0 timeout, returns zero
        let remaining_zero = state.remaining_timeout(0);
        assert_eq!(
            remaining_zero,
            Duration::ZERO,
            "0 timeout should return zero"
        );
    }

    #[test]
    fn test_batching_state_accumulate_removed() {
        let mut state = BatchingState::new();

        let first = KvCacheRemoveData {
            block_hashes: vec![ExternalSequenceBlockHash(1), ExternalSequenceBlockHash(2)],
        };

        state.pending_removed = Some(first);

        if let Some(ref mut pending) = state.pending_removed {
            pending
                .block_hashes
                .extend(vec![ExternalSequenceBlockHash(3)]);
        }

        let pending = state.pending_removed.as_ref().unwrap();
        assert_eq!(
            pending.block_hashes.len(),
            3,
            "Should have accumulated 3 block hashes"
        );
    }

    #[test]
    fn test_batching_state_accumulate_stored() {
        let mut state = BatchingState::new();

        let block1 = KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(1),
            tokens_hash: LocalBlockHash(100),
            mm_extra_info: None,
        };
        let first = KvCacheStoreData {
            parent_hash: Some(ExternalSequenceBlockHash(0)),
            start_position: None,
            blocks: vec![block1],
        };

        state.pending_stored = Some(first);

        let block2 = KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash(2),
            tokens_hash: LocalBlockHash(200),
            mm_extra_info: None,
        };

        if let Some(ref mut pending) = state.pending_stored {
            pending.blocks.extend(vec![block2]);
        }

        let pending = state.pending_stored.as_ref().unwrap();
        assert_eq!(pending.blocks.len(), 2, "Should have accumulated 2 blocks");
    }
}

#[cfg(test)]
mod event_processor_tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio_util::sync::CancellationToken;

    /// Mock publisher that collects published events
    #[derive(Debug, Clone)]
    struct MockPublisher {
        events: Arc<Mutex<Vec<RouterEvent>>>,
    }

    impl MockPublisher {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
            }
        }

        fn get_events(&self) -> Vec<RouterEvent> {
            self.events.lock().unwrap().clone()
        }
    }

    impl RouterEventSink for MockPublisher {
        fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
            self.events.lock().unwrap().push(event.clone());
            async { Ok(()) }
        }
    }

    fn local_gpu_event(event: KvCacheEvent) -> PlacementEvent {
        PlacementEvent::local_gpu(1, event)
    }

    fn local_host_event(event: KvCacheEvent) -> PlacementEvent {
        PlacementEvent::new(
            Placement::local_worker(1, event.dp_rank, StorageTier::HostPinned),
            event,
        )
    }

    /// Test that pushing N removed events results in batched output
    /// Uses a 10ms timeout to ensure events are batched (events sent rapidly)
    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_20() {
        test_removed_events_batching(20, Some(10)).await; // 20 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_10() {
        test_removed_events_batching(10, Some(10)).await; // 10 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_5() {
        test_removed_events_batching(5, Some(10)).await; // 5 events, 10ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_removed_events_3() {
        test_removed_events_batching(3, Some(10)).await; // 3 events, 10ms timeout
    }

    /// Helper function to test removed events batching with configurable count and timeout
    async fn test_removed_events_batching(event_count: usize, timeout_ms: Option<u64>) {
        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        for i in 0..event_count {
            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            };
            tx.send(local_gpu_event(event)).unwrap();
            // Yield to allow event processor to process the event
            tokio::task::yield_now().await;
        }

        // Wait for timeout to elapse so all events flush together as one batch
        // Add small buffer to ensure flush happens before channel close
        tokio::time::sleep(tokio::time::Duration::from_millis(
            timeout_ms.unwrap_or(0) + 1,
        ))
        .await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(
            !events.is_empty(),
            "Should have received at least one event"
        );

        // With a long timeout (100ms) and rapid event sending, all events should batch into few output events
        // (first event may flush separately, rest should batch together)
        assert!(
            events.len() <= 2,
            "With long timeout ({timeout_ms:?}), all {event_count} events should batch into at most 2 output events (got {})",
            events.len()
        );

        let total_hashes: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Removed(data) = &e.event.data {
                    data.block_hashes.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_hashes, event_count,
            "All {} block hashes should be accounted for",
            event_count
        );
    }

    /// Test sequential stored events accumulate with different counts
    /// Uses a longer timeout (100ms) to ensure events have time to batch
    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_20() {
        test_stored_events_batching(20, Some(100)).await; // 20 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_10() {
        test_stored_events_batching(10, Some(100)).await; // 10 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_5() {
        test_stored_events_batching(5, Some(100)).await; // 5 events, 100ms timeout
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_batches_stored_events_3() {
        test_stored_events_batching(3, Some(100)).await; // 3 events, 100ms timeout
    }

    /// Helper function to test stored events batching with configurable count and timeout
    async fn test_stored_events_batching(event_count: usize, timeout_ms: Option<u64>) {
        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        for i in 0..event_count {
            // For sequential batching, each event's parent_hash should be the previous event's block_hash
            let parent_hash = if i == 0 {
                Some(ExternalSequenceBlockHash(0)) // First event has parent_hash = 0
            } else {
                Some(ExternalSequenceBlockHash((i - 1) as u64)) // Subsequent events reference previous block
            };

            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash,
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(i as u64),
                        tokens_hash: LocalBlockHash(i as u64 * 100),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            };
            tx.send(local_gpu_event(event)).unwrap();
            // Small sleep to allow event processor to batch events
            tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;
        }

        // Give the processor time to process all events before closing the channel
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(
            !events.is_empty(),
            "Should have received at least one event"
        );

        // With a long timeout, events should be batched. Either 1 or can be at most 2, if the first event flushes separately due to initial timestamp.
        assert!(
            events.len() <= 2,
            "With long timeout ({timeout_ms:?}) and sequential parent hashes, all {event_count} events should batch into at most 2 output events (got {})",
            events.len()
        );
        if events.len() == 2 {
            // If we got 2 events, the first one should contain only the first block, and the second should contain the rest
            if let KvCacheEventData::Stored(data) = &events[0].event.data {
                assert_eq!(
                    data.blocks.len(),
                    1,
                    "If 2 events, first event should have 1 block (got {})",
                    data.blocks.len()
                );
            } else {
                panic!("Expected Stored event");
            }
        }

        let total_blocks: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Stored(data) = &e.event.data {
                    data.blocks.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_blocks, event_count,
            "All {} blocks should be accounted for",
            event_count
        );
    }

    /// Test non-sequential stored events trigger flush
    #[tokio::test]
    async fn test_run_event_processor_loop_non_sequential_flush() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
            // SLEEP HERE?! so that events are not batched!
        });

        for i in 0..3 {
            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: Some(ExternalSequenceBlockHash((i + 1) as u64 * 100)),
                    start_position: None,
                    blocks: vec![KvCacheStoredBlockData {
                        block_hash: ExternalSequenceBlockHash(i as u64),
                        tokens_hash: LocalBlockHash(i as u64 * 100),
                        mm_extra_info: None,
                    }],
                }),
                dp_rank: 0,
            };
            tx.send(local_gpu_event(event)).unwrap();
        }

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(!events.is_empty(), "Should have received events");

        // With non-sequential parent hashes, each event should trigger a flush
        // So we expect 3 separate events
        assert_eq!(
            events.len(),
            3,
            "Non-sequential events should trigger flush, resulting in 3 separate events"
        );

        let total_blocks: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Stored(data) = &e.event.data {
                    data.blocks.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(total_blocks, 3, "All 3 blocks should be accounted for");
    }

    /// Test that reusing an older parent hash breaks the current sequential batch.
    #[tokio::test]
    async fn test_run_event_processor_loop_reused_parent_hash_breaks_chain() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None,
                start_position: Some(10),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(1),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;

        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(1)),
                start_position: Some(11_111),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(2),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;

        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(1)),
                start_position: Some(20),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(3),
                    tokens_hash: LocalBlockHash(300),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert_eq!(
            events.len(),
            2,
            "Reused parent hash should flush the current batch before starting a new one"
        );

        if let KvCacheEventData::Stored(data) = &events[0].event.data {
            assert_eq!(
                data.blocks.len(),
                2,
                "First batch should keep the valid chain"
            );
            assert_eq!(
                data.parent_hash, None,
                "First batch should preserve the original root parent"
            );
            assert_eq!(
                data.start_position,
                Some(10),
                "First batch should preserve the original start position"
            );
        } else {
            panic!("Expected first event to be Stored");
        }

        if let KvCacheEventData::Stored(data) = &events[1].event.data {
            assert_eq!(
                data.blocks.len(),
                1,
                "Second batch should contain only the inconsistent event"
            );
            assert_eq!(
                data.parent_hash,
                Some(ExternalSequenceBlockHash(1)),
                "Second batch should preserve the reused parent hash"
            );
            assert_eq!(
                data.start_position,
                Some(20),
                "Second batch should keep the new root's start position"
            );
        } else {
            panic!("Expected second event to be Stored");
        }
    }

    /// Test that with short timeout and slow input, events are NOT batched
    /// Parametrized over different timeout values: 0ms, 0.1ms, 0.2ms
    /// All use 2ms delay between events, so each event times out before the next arrives
    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0ms() {
        test_no_batching_with_slow_input(None).await; // disabled (no timeout)
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0_1ms() {
        test_no_batching_with_slow_input(Some(1)).await; // 1ms timeout (was 0.1ms in us)
    }

    #[tokio::test]
    async fn test_run_event_processor_loop_no_batching_with_slow_input_0_2ms() {
        test_no_batching_with_slow_input(Some(2)).await; // 2ms timeout (was 0.2ms in us)
    }

    /// Helper function to test no batching with slow input
    async fn test_no_batching_with_slow_input(timeout_ms: Option<u64>) {
        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send 5 removed events with 2ms delay between each
        // Since timeout is <= 0.2ms, each event should timeout and be sent individually
        for i in 0..5 {
            let event = KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            };
            tx.send(local_gpu_event(event)).unwrap();
            // Wait 2ms between events (much longer than the timeout)
            // This ensures each event times out before the next one arrives
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        }

        // Give the processor time to process the last event
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert!(!events.is_empty(), "Should have received events");

        // With slow input (2ms delay) and short timeout, most events should be sent individually
        // We expect at least 3 separate events (showing reduced batching)
        assert!(
            events.len() >= 3,
            "With slow input (2ms delay) and timeout={timeout_ms:?}, should have at least 3 separate events (got {})",
            events.len()
        );

        let total_hashes: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Removed(data) = &e.event.data {
                    data.block_hashes.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_hashes, 5,
            "All 5 block hashes should be accounted for"
        );
    }

    /// Test that switching between Removed and Stored events causes immediate flush
    #[tokio::test]
    async fn test_event_type_switching_causes_flush() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send a Removed event
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(0)],
            }),
            dp_rank: 0,
        }))
        .unwrap();

        // Small sleep
        tokio::time::sleep(tokio::time::Duration::from_micros(100)).await;

        // Send a Stored event (should cause flush of the Removed event)
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(0)),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(1),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();

        // Give time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        // Should have 2 events: one Removed, one Stored (not batched together)
        assert_eq!(
            events.len(),
            2,
            "Switching from Removed to Stored should cause immediate flush, resulting in 2 separate events"
        );
    }

    #[tokio::test]
    async fn test_host_tier_events_are_published_and_preserved() {
        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                Some(100),
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        tx.send(local_host_event(KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(42)],
            }),
            dp_rank: 0,
        }))
        .unwrap();

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();
        assert_eq!(
            events.len(),
            1,
            "Expected a single published host-tier event"
        );
        assert_eq!(events[0].storage_tier, StorageTier::HostPinned);

        let KvCacheEventData::Removed(data) = &events[0].event.data else {
            panic!("Expected Removed event");
        };
        assert_eq!(data.block_hashes, vec![ExternalSequenceBlockHash(42)]);
    }

    #[tokio::test]
    async fn test_storage_tier_change_causes_flush() {
        let timeout_ms = Some(100);

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        tx.send(local_host_event(KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(1)],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;

        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: vec![ExternalSequenceBlockHash(2)],
            }),
            dp_rank: 0,
        }))
        .unwrap();

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();
        assert_eq!(
            events.len(),
            2,
            "Changing storage tier should flush the current batch"
        );
        assert_eq!(events[0].storage_tier, StorageTier::HostPinned);
        assert_eq!(events[1].storage_tier, StorageTier::Device);
    }

    /// Test that dp_rank change causes immediate flush
    #[tokio::test]
    async fn test_dp_rank_change_causes_flush() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send events with dp_rank=0
        for i in 0..3 {
            tx.send(local_gpu_event(KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            }))
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Send events with dp_rank=1 (should cause flush of previous batch)
        for i in 3..6 {
            tx.send(local_gpu_event(KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 1,
            }))
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Give time for processing
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        // Should have 2 events: one for dp_rank=0 batch, one for dp_rank=1 batch
        assert_eq!(
            events.len(),
            2,
            "dp_rank change should cause immediate flush, resulting in 2 separate events"
        );

        // Verify all 6 block hashes are accounted for
        let total_hashes: usize = events
            .iter()
            .map(|e| {
                if let KvCacheEventData::Removed(data) = &e.event.data {
                    data.block_hashes.len()
                } else {
                    0
                }
            })
            .sum();
        assert_eq!(
            total_hashes, 6,
            "All 6 block hashes should be accounted for"
        );

        // Verify dp_rank is correct for each batch
        assert_eq!(
            events[0].event.dp_rank, 0,
            "First batch should have dp_rank=0"
        );
        assert_eq!(
            events[1].event.dp_rank, 1,
            "Second batch should have dp_rank=1"
        );
    }

    /// Test that flushed events have correct metadata (event_id, dp_rank)
    /// This verifies that metadata is NOT overwritten before flush
    #[tokio::test]
    async fn test_flushed_events_have_correct_metadata() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send first batch: 3 events with dp_rank=0, event_ids 10-12
        for i in 0..3 {
            tx.send(local_gpu_event(KvCacheEvent {
                event_id: 10 + i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            }))
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Send second batch: 2 events with dp_rank=1, event_ids 20-21
        // This should flush the first batch with dp_rank=0
        for i in 0..2 {
            tx.send(local_gpu_event(KvCacheEvent {
                event_id: 20 + i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash((i + 3) as u64)],
                }),
                dp_rank: 1,
            }))
            .unwrap();
            tokio::task::yield_now().await;
        }

        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert_eq!(
            events.len(),
            2,
            "Should have 2 events (one per dp_rank batch)"
        );

        // First event should have dp_rank=0 and monotonic batch event_id=1
        assert_eq!(
            events[0].event.dp_rank, 0,
            "First batch should have dp_rank=0"
        );
        assert_eq!(
            events[0].event.event_id, 1,
            "First batch should have monotonic event_id=1"
        );

        // Second event should have dp_rank=1 and monotonic batch event_id=2
        assert_eq!(
            events[1].event.dp_rank, 1,
            "Second batch should have dp_rank=1"
        );
        assert_eq!(
            events[1].event.event_id, 2,
            "Second batch should have monotonic event_id=2"
        );
    }

    /// Test that events after a long idle period flush immediately (stale timer).
    /// This gives low latency for sparse important events after idle periods.
    /// After the initial stale flush, subsequent rapid events batch normally.
    #[tokio::test]
    async fn test_first_event_after_idle_flushes_immediately_then_batches() {
        let timeout_ms = Some(50); // 50ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Wait longer than timeout to simulate idle period (timer becomes stale)
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Send 3 events rapidly - first should flush immediately (stale timer),
        // remaining 2 should batch together
        for i in 0..3 {
            tx.send(local_gpu_event(KvCacheEvent {
                event_id: i as u64,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: vec![ExternalSequenceBlockHash(i as u64)],
                }),
                dp_rank: 0,
            }))
            .unwrap();
            tokio::task::yield_now().await;
        }

        // Wait for timeout to elapse so remaining batch flushes
        tokio::time::sleep(tokio::time::Duration::from_millis(60)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        // First event flushes immediately (stale timer), remaining 2 batch together
        assert_eq!(
            events.len(),
            2,
            "First event should flush immediately (stale), remaining 2 should batch"
        );

        // First event has 1 hash, second event (batch) has 2 hashes
        let first_len = if let KvCacheEventData::Removed(data) = &events[0].event.data {
            data.block_hashes.len()
        } else {
            0
        };
        let second_len = if let KvCacheEventData::Removed(data) = &events[1].event.data {
            data.block_hashes.len()
        } else {
            0
        };
        assert_eq!(first_len, 1, "First event should have 1 hash");
        assert_eq!(second_len, 2, "Second event (batched) should have 2 hashes");
    }

    /// Test that stored events with dp_rank change have correct metadata
    #[tokio::test]
    async fn test_stored_events_with_dp_rank_change_correct_metadata() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // Send first batch: 2 sequential stored events with dp_rank=0, event_ids 100-101
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 100,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(0)),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(1),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;

        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 101,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(1)),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(2),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;

        // Send second batch: 1 event with dp_rank=1, event_id=200
        // This should flush the first batch with dp_rank=0, event_id=101
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 200,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(0)),
                start_position: None,
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(100),
                    tokens_hash: LocalBlockHash(1000),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 1,
        }))
        .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert_eq!(
            events.len(),
            2,
            "Should have 2 events (one per dp_rank batch)"
        );

        // First batch: dp_rank=0, monotonic event_id=1
        assert_eq!(
            events[0].event.dp_rank, 0,
            "First batch should have dp_rank=0"
        );
        assert_eq!(
            events[0].event.event_id, 1,
            "First batch should have monotonic event_id=1"
        );

        // Second batch: dp_rank=1, monotonic event_id=2
        assert_eq!(
            events[1].event.dp_rank, 1,
            "Second batch should have dp_rank=1"
        );
        assert_eq!(
            events[1].event.event_id, 2,
            "Second batch should have monotonic event_id=2"
        );

        // Verify block counts
        if let KvCacheEventData::Stored(data) = &events[0].event.data {
            assert_eq!(data.blocks.len(), 2, "First batch should have 2 blocks");
        } else {
            panic!("Expected Stored event");
        }
        if let KvCacheEventData::Stored(data) = &events[1].event.data {
            assert_eq!(data.blocks.len(), 1, "Second batch should have 1 block");
        } else {
            panic!("Expected Stored event");
        }
    }

    /// Test that extending a batch does NOT change parent_hash
    /// First event with parent_hash=None should keep it None even if subsequent events have Some(X)
    #[tokio::test]
    async fn test_batch_parent_hash_preserved_when_extending() {
        let timeout_ms = Some(100); // 100ms timeout

        let (tx, rx) = mpsc::unbounded_channel::<PlacementEvent>();
        let publisher = MockPublisher::new();
        let publisher_clone = publisher.clone();
        let cancellation_token = CancellationToken::new();

        let handle = tokio::spawn(async move {
            run_event_processor_loop(
                publisher_clone,
                1,
                cancellation_token,
                rx,
                None,
                timeout_ms,
                DEFAULT_MAX_BATCH_BLOCKS,
            )
            .await
        });

        // First event: parent_hash=None, block_hash=1
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 0,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: None, // Root block with no parent
                start_position: Some(10),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(1),
                    tokens_hash: LocalBlockHash(100),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;

        // Second event: parent_hash=Some(1), block_hash=2 (sequential)
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 1,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(1)), // Points to previous block
                start_position: Some(999),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(2),
                    tokens_hash: LocalBlockHash(200),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();
        tokio::task::yield_now().await;

        // Third event: parent_hash=Some(2), block_hash=3 (sequential)
        tx.send(local_gpu_event(KvCacheEvent {
            event_id: 2,
            data: KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: Some(ExternalSequenceBlockHash(2)),
                start_position: Some(1_234),
                blocks: vec![KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(3),
                    tokens_hash: LocalBlockHash(300),
                    mm_extra_info: None,
                }],
            }),
            dp_rank: 0,
        }))
        .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        drop(tx);
        handle.await.unwrap();

        let events = publisher.get_events();

        assert_eq!(
            events.len(),
            1,
            "All 3 sequential events should batch into 1"
        );

        // The batch should have parent_hash=None (preserved from first event)
        if let KvCacheEventData::Stored(data) = &events[0].event.data {
            assert_eq!(data.blocks.len(), 3, "Batch should have 3 blocks");
            assert_eq!(
                data.parent_hash, None,
                "Batch parent_hash should remain None (from first event), NOT overwritten by subsequent events"
            );
            assert_eq!(
                data.start_position,
                Some(10),
                "Batch start_position should remain anchored to the first event"
            );
        } else {
            panic!("Expected Stored event");
        }
    }
}
