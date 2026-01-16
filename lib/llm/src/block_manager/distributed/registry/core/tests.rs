// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the registry.

#[cfg(test)]
mod integration {
    use std::sync::Arc;
    use std::time::Duration;

    use crate::block_manager::distributed::registry::core::{
        BinaryCodec, HashMapStorage, InProcessHubTransport, InProcessTransport, NoMetadata,
        OffloadStatus, QueryType, Registry, RegistryCodec, RegistryTransport, ResponseType,
        Storage, client, hub,
    };

    /// Full end-to-end test: Client <-> Hub using in-process transport.
    #[tokio::test]
    async fn test_client_hub_integration() {
        // Create hub with storage using the builder
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        storage.insert(1, 100);
        storage.insert(2, 200);

        let registry_hub = hub::<u64, u64, NoMetadata, _>(storage)
            .lease_ttl(Duration::from_secs(30))
            .build();

        // Create in-process transport pair
        let (mut hub_transport, client_handle) = InProcessHubTransport::new();

        // Spawn hub server
        let hub_task = tokio::spawn(async move {
            for _ in 0..3 {
                registry_hub.process_one(&mut hub_transport).await.ok();
            }
        });

        let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

        // Test can_offload
        let mut buf = Vec::new();
        codec
            .encode_query(&QueryType::CanOffload(vec![1, 3]), &mut buf)
            .unwrap();
        let response = client_handle.request(&buf).await.unwrap();
        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&response).unwrap();
        match decoded {
            ResponseType::CanOffload(statuses) => {
                assert_eq!(statuses[0], OffloadStatus::AlreadyStored);
                assert_eq!(statuses[1], OffloadStatus::Granted);
            }
            _ => panic!("Wrong response"),
        }

        // Test match
        buf.clear();
        codec
            .encode_query(&QueryType::Match(vec![1, 2]), &mut buf)
            .unwrap();
        let response = client_handle.request(&buf).await.unwrap();
        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&response).unwrap();
        match decoded {
            ResponseType::Match(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0], (1, 100, NoMetadata));
                assert_eq!(entries[1], (2, 200, NoMetadata));
            }
            _ => panic!("Wrong response"),
        }

        // One more query to complete hub_task
        buf.clear();
        codec
            .encode_query(&QueryType::CanOffload(vec![1]), &mut buf)
            .unwrap();
        let _ = client_handle.request(&buf).await;

        hub_task.await.unwrap();
    }

    /// Test with mock hub (simpler, no spawning).
    #[tokio::test]
    async fn test_full_registry_flow() {
        let hub_storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        hub_storage.insert(1, 100);
        hub_storage.insert(2, 200);
        hub_storage.insert(3, 300);

        let (transport, _rx) = InProcessTransport::new(move |data| {
            let codec: BinaryCodec<u64, u64, NoMetadata> = BinaryCodec::new();

            if let Some(query) = codec.decode_query(data) {
                let mut buf = Vec::new();
                match query {
                    QueryType::CanOffload(keys) => {
                        let statuses: Vec<_> = keys
                            .iter()
                            .map(|k| {
                                if hub_storage.contains(k) {
                                    OffloadStatus::AlreadyStored
                                } else {
                                    OffloadStatus::Granted
                                }
                            })
                            .collect();
                        codec
                            .encode_response(&ResponseType::CanOffload(statuses), &mut buf)
                            .unwrap();
                    }
                    QueryType::Match(keys) => {
                        let entries: Vec<_> = keys
                            .iter()
                            .filter_map(|k| hub_storage.get(k).map(|v| (*k, v, NoMetadata)))
                            .collect();
                        codec
                            .encode_response(&ResponseType::Match(entries), &mut buf)
                            .unwrap();
                    }
                }
                buf
            } else {
                Vec::new()
            }
        });

        // Build client using the builder
        let registry_client = client::<u64, u64, NoMetadata, _>(transport)
            .batch_size(100)
            .build();

        // Test can_offload
        let result = registry_client.can_offload(&[1, 2, 5, 6]).await.unwrap();
        assert_eq!(result.already_stored, vec![1, 2]);
        assert_eq!(result.can_offload, vec![5, 6]);

        // Test match_prefix
        let matched = registry_client.match_prefix(&[1, 2, 3]).await.unwrap();
        assert_eq!(matched.len(), 3);
        assert_eq!(matched[0].1, 100);
        assert_eq!(matched[1].1, 200);
        assert_eq!(matched[2].1, 300);
    }

    /// Test batched registration and flush.
    #[tokio::test]
    async fn test_register_and_flush() {
        use std::sync::Mutex;

        let received = Arc::new(Mutex::new(Vec::new()));
        let received_clone = received.clone();

        let (transport, mut rx) = InProcessTransport::new(move |_data| Vec::new());

        let received_for_task = received_clone.clone();
        tokio::spawn(async move {
            while let Some(data) = rx.recv().await {
                received_for_task.lock().unwrap().push(data);
            }
        });

        // Build client with custom batch size using the builder
        let registry_client = client::<u64, u64, NoMetadata, _>(transport)
            .batch_size(10)
            .build();

        registry_client
            .register(&[(1, 100, NoMetadata), (2, 200, NoMetadata)])
            .await
            .unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert!(received.lock().unwrap().is_empty());

        registry_client.flush().await.unwrap();

        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        assert_eq!(received.lock().unwrap().len(), 1);
    }

    /// ZMQ end-to-end test.
    ///
    /// Run manually: `cargo test -- --ignored test_zmq_e2e`
    #[tokio::test]
    #[ignore]
    async fn test_zmq_e2e() {
        use crate::block_manager::distributed::registry::core::{
            ZmqHub, ZmqHubServerConfig, ZmqTransport,
        };
        use tokio_util::sync::CancellationToken;

        let port_base = 17555;
        let config = ZmqHubServerConfig::bind_all(port_base, port_base + 1);

        // Create hub with pre-populated storage
        let storage: HashMapStorage<u64, u64> = HashMapStorage::new();
        storage.insert(1, 100);
        storage.insert(2, 200);

        let hub = ZmqHub::new(config, storage, BinaryCodec::<u64, u64, NoMetadata>::new());
        let cancel = CancellationToken::new();

        // Start hub
        let hub_cancel = cancel.clone();
        let hub_handle = tokio::spawn(async move { hub.serve(hub_cancel).await });

        // Wait for hub to bind
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Connect client
        let transport = ZmqTransport::connect_to("localhost", port_base, port_base + 1)
            .expect("Failed to connect");

        // Test query directly with transport
        let codec = BinaryCodec::<u64, u64, NoMetadata>::new();

        // Test can_offload
        let mut buf = Vec::new();
        codec
            .encode_query(&QueryType::CanOffload(vec![1, 3]), &mut buf)
            .unwrap();
        let response = transport.request(&buf).await.expect("Request failed");
        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&response).unwrap();

        match decoded {
            ResponseType::CanOffload(statuses) => {
                assert_eq!(statuses[0], OffloadStatus::AlreadyStored);
                assert_eq!(statuses[1], OffloadStatus::Granted);
            }
            _ => panic!("Wrong response type"),
        }

        // Test match
        buf.clear();
        codec
            .encode_query(&QueryType::Match(vec![1, 2]), &mut buf)
            .unwrap();
        let response = transport.request(&buf).await.expect("Match request failed");
        let decoded: ResponseType<u64, u64, NoMetadata> = codec.decode_response(&response).unwrap();

        match decoded {
            ResponseType::Match(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0], (1, 100, NoMetadata));
                assert_eq!(entries[1], (2, 200, NoMetadata));
            }
            _ => panic!("Wrong response type"),
        }

        // Shutdown
        cancel.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(1), hub_handle).await;
    }
}
