// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration test: build a real [`HostMemoryPool`] with tiny allocations
//! and no hugepages, then drive `/v1/pool` over HTTP to confirm the slab
//! count + tier + agent name make it through.
//!
//! Gated behind `testing-cuda` because the path goes through
//! `cuMemHostRegister`, which requires a working CUDA driver + at least
//! one visible device. The UCX NIXL plugin must also be installed in the
//! test environment — this is a project invariant; tests fail loudly if
//! UCX is missing rather than silently falling back to local-only slabs.

#![cfg(feature = "testing-cuda")]

use std::time::Duration;

use dynamo_memory::HugepageMode;
use kvbm_service::{HostMemoryPool, KvbmService, PoolConfig, ServiceConfig};

const SMALL_SLAB_BYTES: u64 = 16 * 1024 * 1024;

/// Small NIXL-backed pool config used by all alloc tests. UCX is on the
/// default backend list so production / test parity is preserved — no env
/// vars required.
fn small_pool_config() -> PoolConfig {
    PoolConfig::builder()
        .per_node_bytes(SMALL_SLAB_BYTES)
        .hugepage_mode(HugepageMode::Disabled)
        .with_ucx()
        .build()
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
        assert!(
            slab.registered,
            "slab on node {} should be NIXL-registered (UCX is in default backends); \
             got registered=false. Is libplugin_UCX.so available in this env?",
            slab.numa_node,
        );
        let name = slab
            .agent_name
            .as_deref()
            .expect("registered slab must have an agent_name");
        assert!(
            name.starts_with("kvbm-svc:test-instance:n"),
            "unexpected agent name {name}",
        );
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

    let slabs = body["slabs"].as_array().expect("snapshot has slabs array");
    assert!(!slabs.is_empty(), "expected at least one slab");
    for slab in slabs {
        assert_eq!(slab["registered"].as_bool(), Some(true));
        assert!(slab["agent_name"].as_str().is_some());
    }
    let total = body["total_bytes"].as_u64().unwrap();
    assert_eq!(total, SMALL_SLAB_BYTES * (slabs.len() as u64));

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

#[tokio::test]
async fn pool_local_only_builder_succeeds_without_nixl() {
    // The escape hatch for environments without UCX (or any NIXL libs).
    // Slabs allocate but are local-only — not reachable from remote NIXL.
    let cfg = PoolConfig::builder()
        .per_node_bytes(SMALL_SLAB_BYTES)
        .hugepage_mode(HugepageMode::Disabled)
        .local_only()
        .build();
    let pool = HostMemoryPool::new(&cfg, "test-local").expect("HostMemoryPool::new local-only");
    let snapshot = pool.snapshot();
    assert!(!snapshot.slabs.is_empty());
    for slab in &snapshot.slabs {
        assert!(
            !slab.registered,
            "local-only slab must not be NIXL-registered"
        );
        assert!(slab.agent_name.is_none());
    }
}
