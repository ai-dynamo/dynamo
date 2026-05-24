// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test: build a real [`HostMemoryPool`] with tiny allocations
//! and no hugepages, then drive `/v1/pool` over HTTP to confirm the slab
//! count + tier + agent name make it through.
//!
//! Gated behind `testing-cuda` because the path goes through
//! `cuMemHostRegister`, which requires a working CUDA driver + at least
//! one visible device.

#![cfg(feature = "testing-cuda")]

use std::time::Duration;

use dynamo_memory::HugepageMode;
use kvbm_service::{HostMemoryPool, KvbmService, PoolConfig, PoolSizing, ServiceConfig};

const SMALL_SLAB_BYTES: u64 = 16 * 1024 * 1024;

fn small_pool_config() -> PoolConfig {
    PoolConfig {
        sizing: PoolSizing::PerNode {
            bytes_per_node: SMALL_SLAB_BYTES,
        },
        hugepage_mode: HugepageMode::Disabled,
        // No NIXL backends are configured in this test process — opt out
        // of the production-safety check so slabs can be built local-only.
        allow_no_nixl_backends: true,
        ..Default::default()
    }
}

#[tokio::test]
async fn pool_allocates_one_slab_per_host_memory_node() {
    let pool =
        HostMemoryPool::new(&small_pool_config(), "test-instance").expect("HostMemoryPool::new");
    let snapshot = pool.snapshot();

    assert!(
        !snapshot.slabs.is_empty(),
        "expected at least one slab; got {:?}",
        snapshot,
    );

    for slab in &snapshot.slabs {
        assert_eq!(slab.size_bytes, SMALL_SLAB_BYTES);
        // Agent name is Some when nixl_sys has a working DRAM backend
        // (we'd be in the `Registered` SlabStorage variant) and None on
        // stub-mode hosts where the pool falls back to local-only slabs.
        // When present, it must follow the documented format.
        if let Some(name) = &slab.agent_name {
            assert!(
                name.starts_with("kvbm-svc:test-instance:n"),
                "unexpected agent name {name}",
            );
        }
    }

    assert_eq!(
        snapshot.total_bytes,
        SMALL_SLAB_BYTES * (snapshot.slabs.len() as u64)
    );
}

#[tokio::test]
async fn http_v1_pool_returns_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let cfg = ServiceConfig {
        http_addr: "127.0.0.1:0".parse().unwrap(),
        uds_path: None,
        uds_dir: tmp.path().to_path_buf(),
        shutdown_grace_ms: None,
        pool: small_pool_config(),
    };

    let svc = KvbmService::start_with_pool(cfg, std::sync::Arc::new(kvbm_service::NoopContainer))
        .await
        .expect("start_with_pool");
    let http = svc.http_addr;

    let body: serde_json::Value = reqwest::get(format!("http://{http}/v1/pool"))
        .await
        .unwrap()
        .json()
        .await
        .unwrap();

    assert!(body["slabs"].as_array().is_some_and(|a| !a.is_empty()));
    let total = body["total_bytes"].as_u64().unwrap();
    let slab_count = body["slabs"].as_array().unwrap().len() as u64;
    assert_eq!(total, SMALL_SLAB_BYTES * slab_count);

    // Default container drops the lease, so the snapshot reports leased=false.
    assert_eq!(body["leased"], false);

    svc.shutdown_graceful(Some(Duration::from_secs(60))).await;
}

#[tokio::test]
async fn http_v1_pool_503_without_pool() {
    // Service started without a pool — /v1/pool should report 503.
    let tmp = tempfile::tempdir().unwrap();
    let cfg = ServiceConfig {
        http_addr: "127.0.0.1:0".parse().unwrap(),
        uds_path: None,
        uds_dir: tmp.path().to_path_buf(),
        shutdown_grace_ms: None,
        pool: Default::default(),
    };
    let svc = KvbmService::start(cfg).await.expect("start");
    let http = svc.http_addr;

    let resp = reqwest::get(format!("http://{http}/v1/pool"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"].as_str().is_some());

    svc.shutdown_graceful(Some(Duration::from_secs(60))).await;
}
