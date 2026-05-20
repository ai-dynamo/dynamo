// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test: verify Mooncake Store via raw API and MooncakeObjectBlockClient.
//!
//! This test requires a running Mooncake server. Run with:
//! ```bash
//! cargo test -p kvbm-engine --features mooncake,testing --test mooncake_integration -- --ignored
//! ```

use kvbm_config::MooncakeProtocol;
use mooncake_store::MooncakeStore;

#[test]
#[ignore = "Requires Mooncake server and NIXL environment"]
fn mooncake_integration() {
    println!("=== Mooncake Integration Test ===");

    // ---------------------------------------------------------------
    // 1. Raw store API smoke test
    // ---------------------------------------------------------------
    {
        let store = MooncakeStore::new().expect("create store");
        store
            .setup(
                "localhost",
                "http://127.0.0.1:8080/metadata",
                512 * 1024 * 1024,
                128 * 1024 * 1024,
                "tcp",
                "",
                "127.0.0.1:50051",
            )
            .expect("setup store");

        let key = "test-key-1";
        let value = b"hello mooncake from dynamo!";
        store.put(key, value, None).expect("put");
        println!("[OK] put {} = {} bytes", key, value.len());

        let got = store.get(key).expect("get");
        assert_eq!(got, value);
        println!("[OK] get {} = {} bytes, data matches", key, got.len());

        assert!(store.is_exist(key).expect("is_exist"));
        println!("[OK] is_exist({}) = true", key);

        let size = store.get_size(key).expect("get_size");
        assert_eq!(size, value.len() as i64);
        println!("[OK] get_size({}) = {}", key, size);

        let results = store
            .batch_is_exist(&["test-key-1", "test-key-2"])
            .expect("batch_is_exist");
        assert_eq!(results, vec![true, false]);
        println!("[OK] batch_is_exist = {:?}", results);

        store.remove(key, true).expect("remove");
        assert!(!store.is_exist(key).expect("is_exist after remove"));
        println!("[OK] remove({}) + is_exist = false", key);
    } // store dropped here

    // ---------------------------------------------------------------
    // 2. MooncakeObjectBlockClient::has_blocks
    // ---------------------------------------------------------------
    {
        use kvbm_config::MooncakeObjectConfig;
        use kvbm_engine::object::ObjectBlockOps;
        use kvbm_engine::object::mooncake::client::MooncakeObjectBlockClient;

        let config = MooncakeObjectConfig {
            metadata_server: "http://127.0.0.1:8080/metadata".to_string(),
            master_server_addr: "127.0.0.1:50051".to_string(),
            local_hostname: "localhost".to_string(),
            protocol: MooncakeProtocol::Tcp,
            device_name: "".to_string(),
            global_segment_size: 512 * 1024 * 1024,
            local_buffer_size: 128 * 1024 * 1024,
            namespace: Some("test-ns".to_string()),
            max_concurrent_requests: 4,
        };

        let client =
            MooncakeObjectBlockClient::new(&config).expect("create MooncakeObjectBlockClient");

        let hash_a = kvbm_engine::SequenceHash::new(42, None, 0);
        let hash_b = kvbm_engine::SequenceHash::new(99, None, 1);
        let hashes = vec![hash_a, hash_b];

        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let results = rt.block_on(client.has_blocks(hashes.clone()));

        for (hash, size) in &results {
            assert!(size.is_none(), "block {} should not exist", hash);
        }
        println!("[OK] has_blocks (empty) = {:?}", results);
    }

    println!("\n=== All integration tests passed ===");
}
