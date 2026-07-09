// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[cfg(test)]
mod test_event_processing {
    use super::*;
    use dynamo_kv_router::protocols::{BlockHashOptions, compute_block_hash_for_seq};
    use dynamo_kv_router::zmq_wire::StoredBlockOptions;

    #[test]
    fn direct_valkey_writer_uses_unsuffixed_dynamo_namespace() {
        assert_eq!(
            valkey_index_namespace("default-dgd-f8c2a", Some("default-dgd"), Some("f8c2a"),),
            "default-dgd"
        );
        assert_eq!(
            valkey_index_namespace("test-namespace", None, Some("worker-pool")),
            "test-namespace"
        );
        assert_eq!(
            valkey_index_namespace("test-namespace", Some("   "), Some("worker-pool")),
            "test-namespace"
        );
        assert_eq!(
            valkey_index_namespace("explicit-namespace", Some("process-default"), None),
            "explicit-namespace"
        );
        assert_eq!(
            valkey_index_namespace(
                "explicit-namespace",
                Some("process-default"),
                Some("worker-pool")
            ),
            "explicit-namespace"
        );
    }

    #[test]
    fn direct_valkey_writer_uses_small_default_batch_window() {
        assert_eq!(valkey_batching_timeout_ms(None, None), Some(1));
        assert_eq!(valkey_batching_timeout_ms(None, Some("0")), None);
        assert_eq!(valkey_batching_timeout_ms(None, Some("3")), Some(3));
        assert_eq!(valkey_batching_timeout_ms(None, Some("invalid")), Some(1));
        assert_eq!(valkey_batching_timeout_ms(Some(7), Some("0")), Some(7));
    }

    #[test]
    fn direct_valkey_writer_keeps_two_high_concurrency_waves_of_bounded_ingress() {
        assert_eq!(valkey_event_input_buffer_size(None), 131_072);
        assert_eq!(valkey_event_input_buffer_size(Some("65536")), 65_536);
        assert_eq!(valkey_event_input_buffer_size(Some("1023")), 131_072);
        assert_eq!(valkey_event_input_buffer_size(Some("invalid")), 131_072);
        assert_eq!(valkey_event_input_buffer_size(Some("2000000")), 1_048_576);
    }

    #[test]
    fn direct_valkey_writer_rejects_invalid_enable_flag() {
        for value in ["true", "1", "YES", " on "] {
            assert!(parse_valkey_worker_events_enabled(value).unwrap());
        }
        for value in ["false", "0", "NO", " off "] {
            assert!(!parse_valkey_worker_events_enabled(value).unwrap());
        }
        assert!(parse_valkey_worker_events_enabled("maybe").is_err());
        assert!(parse_valkey_worker_events_enabled("").is_err());
    }

    #[test]
    fn direct_valkey_writer_validates_required_replica_ack_count() {
        assert_eq!(parse_valkey_required_replica_acks("0").unwrap(), 0);
        assert_eq!(parse_valkey_required_replica_acks("1").unwrap(), 1);
        assert!(parse_valkey_required_replica_acks("-1").is_err());
        assert!(parse_valkey_required_replica_acks("invalid").is_err());
        assert!(parse_valkey_required_replica_acks(&(i32::MAX as u32 + 1).to_string()).is_err());
    }

    #[test]
    fn direct_valkey_gc_config_parsing_is_bounded_and_disableable() {
        assert_eq!(parse_valkey_gc_interval_ms("0").unwrap(), None);
        assert_eq!(
            parse_valkey_gc_interval_ms(" 60000 ").unwrap(),
            Some(60_000)
        );
        assert!(parse_valkey_gc_interval_ms("999").is_err());
        assert!(parse_valkey_gc_interval_ms("not-a-number").is_err());
        assert_eq!(parse_valkey_gc_inspection_budget("256").unwrap(), 256);
        assert!(parse_valkey_gc_inspection_budget("0").is_err());
        assert!(
            parse_valkey_gc_inspection_budget(&(MAX_VALKEY_GC_INSPECTION_BUDGET + 1).to_string())
                .is_err()
        );
    }

    #[test]
    fn direct_valkey_gc_initial_delay_is_deterministic_and_spread() {
        let interval_ms = 60_000;
        let nonce = 0x1234_5678_9abc_def0;
        let delay = valkey_gc_initial_delay_ms(interval_ms, nonce);
        assert_eq!(delay, valkey_gc_initial_delay_ms(interval_ms, nonce));
        assert!((interval_ms..interval_ms * 2).contains(&delay));

        let distinct_phases = (1..=16)
            .map(|owner_nonce| valkey_gc_initial_delay_ms(interval_ms, owner_nonce))
            .collect::<std::collections::BTreeSet<_>>();
        assert!(distinct_phases.len() > 12);
    }

    #[test]
    fn valkey_worker_shutdown_budget_fits_process_teardown_grace() {
        let lifecycle_budget = KV_EVENT_PUBLISHER_DRAIN_TIMEOUT
            + KV_EVENT_PUBLISHER_FORCED_WAIT_TIMEOUT
            + VALKEY_WORKER_UNREGISTER_TIMEOUT;
        assert!(lifecycle_budget < Duration::from_secs(10));
        assert_eq!(lifecycle_budget, Duration::from_millis(5_500));
    }

    // ---------------------------------------------------------------------
    // create_stored_block_from_parts --------------------------------------
    // ---------------------------------------------------------------------
    #[test]
    fn test_create_stored_block_from_parts() {
        let kv_block_size = 4;
        let token_ids = vec![10, 20, 30, 40];
        let blk_hash = 0xdead_beef;

        let stored = create_stored_block_from_parts(
            kv_block_size,
            blk_hash,
            &token_ids,
            StoredBlockOptions::default(),
        );

        assert_eq!(stored.block_hash.0, blk_hash);
        let expected_hash =
            compute_block_hash_for_seq(&token_ids, 4, BlockHashOptions::default())[0];
        assert_eq!(stored.tokens_hash, expected_hash);
        assert!(stored.mm_extra_info.is_none());
    }

    #[test]
    fn test_create_stored_block_from_parts_with_cache_namespace() {
        let kv_block_size = 4;
        let token_ids = vec![10, 20, 30, 40];
        let stored = create_stored_block_from_parts(
            kv_block_size,
            0xdead_beef,
            &token_ids,
            StoredBlockOptions {
                cache_namespace: Some("tenant-a"),
                ..Default::default()
            },
        );
        let expected_hash = compute_block_hash_for_seq(
            &token_ids,
            kv_block_size,
            BlockHashOptions {
                cache_namespace: Some("tenant-a"),
                ..Default::default()
            },
        )[0];
        let base_hash =
            compute_block_hash_for_seq(&token_ids, kv_block_size, BlockHashOptions::default())[0];

        assert_eq!(stored.tokens_hash, expected_hash);
        assert_ne!(stored.tokens_hash, base_hash);
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
            None,
            &Arc::new(AtomicU32::new(0)),
            None,
            None,
            None,
        );

        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].block_hash.0, 111);
        assert_eq!(blocks[1].block_hash.0, 222);

        let namespaced_blocks = create_stored_blocks(
            kv_block_size,
            &token_ids,
            &num_block_tokens,
            &block_hashes,
            None,
            Some("tenant-a"),
            &Arc::new(AtomicU32::new(0)),
            None,
            None,
            None,
        );
        for (block, tokens) in namespaced_blocks
            .iter()
            .zip(token_ids.chunks(kv_block_size as usize))
        {
            let expected = compute_block_hash_for_seq(
                tokens,
                kv_block_size,
                BlockHashOptions {
                    cache_namespace: Some("tenant-a"),
                    ..Default::default()
                },
            )[0];
            assert_eq!(block.tokens_hash, expected);
        }
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
            None,
            &warning_count,
            None,
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
            cache_namespace: None,
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
            None,
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
            cache_namespace: None,
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
            cache_namespace: None,
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
            None,
        )
        .unwrap();
        let lora_out = convert_event(
            lora_evt,
            2,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &wc,
            None,
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
            cache_namespace: None,
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
            cache_namespace: None,
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
            None,
        )
        .unwrap();
        let out2 = convert_event(
            evt2,
            2,
            kv_block_size,
            WorkerWithDpRank::from_worker_id(1),
            &wc,
            None,
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
            None,
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
            None,
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
