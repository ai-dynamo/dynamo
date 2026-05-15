// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "testing")]

//! Integration tests for the leader control plane and its togglable modules,
//! exercised over a real velo TCP loopback pair via the public
//! `LeaderControlClient`.
//!
//! The "`Tests` module is absent without the `testing` feature" case is a
//! compile-time gate (`#[cfg(feature = "testing")]` on the module and its
//! registration); it is covered by the workspace's
//! `--no-default-features` build, not a runtime assertion here.

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use kvbm_common::SequenceHash;
use kvbm_engine::G2;
use kvbm_engine::disagg::session::testing::wait_until;
use kvbm_engine::disagg::session::{
    LifecycleEvent, MockSessionFactory, SessionFactory, SessionManager,
};
use kvbm_engine::leader::ControlPlane;
use kvbm_engine::leader::control::TestModule;
use kvbm_engine::leader::control::TransferModule;
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_logical::manager::BlockManager;
use kvbm_protocols::control::ModuleId;
use kvbm_protocols::control::client::LeaderControlClient;
use kvbm_protocols::control::modules::test::RegisterTestBlocksRequest;
use kvbm_protocols::control::modules::transfer::{SearchRequest, SearchResponse};
use tokio::runtime::Handle;
use velo::transports::tcp::TcpTransportBuilder;

const BLOCK_SIZE: usize = 16;
const G2_BLOCK_COUNT: usize = 32;

fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

/// A connected `(server, client)` velo pair plus the server's G2 manager.
struct Fixture {
    server: Arc<velo::Velo>,
    client: Arc<velo::Velo>,
    g2: Arc<BlockManager<G2>>,
}

async fn fixture() -> Fixture {
    let server = new_velo().await;
    let client = new_velo().await;
    server.register_peer(client.peer_info()).unwrap();
    client.register_peer(server.peer_info()).unwrap();

    let registry = TestRegistryBuilder::new().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry)
            .build(),
    );

    Fixture { server, client, g2 }
}

fn hashes(n: usize) -> Vec<SequenceHash> {
    hashes_from(0, n)
}

/// `n` distinct sequence hashes starting at `base` — used to produce a
/// disjoint set the G2 manager will not match.
fn hashes_from(base: u64, n: usize) -> Vec<SequenceHash> {
    (base..base + n as u64)
        .map(|i| {
            let parent = if i == 0 { None } else { Some(i - 1) };
            SequenceHash::new(i, parent, i)
        })
        .collect()
}

/// Allocate + hash-stamp + register one G2 block per hash, then drop the
/// `ImmutableBlock`s — they return to the inactive pool but stay in the
/// registry, so `match_blocks` / `scan_matches` still find them.
fn populate_g2(g2: &BlockManager<G2>, seq_hashes: &[SequenceHash]) {
    let mutables = g2
        .allocate_blocks(seq_hashes.len())
        .expect("G2 pool large enough");
    let block_size = g2.block_size();
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(seq_hashes)
        .map(|(m, h)| m.stage(*h, block_size).expect("stage"))
        .collect();
    let _immutables = g2.register_blocks(completes);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tests_module_lists_and_registers_blocks() {
    let fx = fixture().await;

    let _plane = ControlPlane::builder(fx.server.messenger().clone(), fx.server.instance_id())
        .with_module(TestModule::new(fx.g2.clone()))
        .register()
        .expect("register control plane");

    let control = LeaderControlClient::new(fx.client.messenger().clone(), fx.server.instance_id());

    // `list_modules` reports the enabled `Tests` module.
    let modules = control.list_modules().await.expect("list_modules");
    assert_eq!(modules, vec![ModuleId::Test]);

    // `register_test_blocks` allocates one G2 block per hash (all-or-nothing).
    let resp = control
        .test()
        .register_test_blocks(RegisterTestBlocksRequest {
            sequence_hashes: hashes(8),
        })
        .await
        .expect("register_test_blocks");
    assert_eq!(resp.allocated, 8);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn register_test_blocks_is_all_or_nothing_when_pool_too_small() {
    let fx = fixture().await;

    let _plane = ControlPlane::builder(fx.server.messenger().clone(), fx.server.instance_id())
        .with_module(TestModule::new(fx.g2.clone()))
        .register()
        .expect("register control plane");

    let control = LeaderControlClient::new(fx.client.messenger().clone(), fx.server.instance_id());

    // More hashes than the G2 pool can satisfy → allocated == 0.
    let resp = control
        .test()
        .register_test_blocks(RegisterTestBlocksRequest {
            sequence_hashes: hashes(G2_BLOCK_COUNT + 1),
        })
        .await
        .expect("register_test_blocks");
    assert_eq!(resp.allocated, 0);
}

// ---- transfer module -------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transfer_module_opens_session_on_match() {
    let fx = fixture().await;

    // Populate G2 so the search handlers have something to find.
    let known = hashes(5);
    populate_g2(&fx.g2, &known);

    let factory: Arc<dyn SessionFactory> = MockSessionFactory::new();
    let factory_cell: Arc<OnceLock<Arc<dyn SessionFactory>>> = Arc::new(OnceLock::new());
    assert!(factory_cell.set(factory).is_ok());
    let session_manager = SessionManager::new(Handle::current(), Duration::from_secs(30));

    let _plane = ControlPlane::builder(fx.server.messenger().clone(), fx.server.instance_id())
        .with_module(TransferModule::new(
            fx.g2.clone(),
            factory_cell,
            session_manager.clone(),
        ))
        .register()
        .expect("register control plane");

    let control = LeaderControlClient::new(fx.client.messenger().clone(), fx.server.instance_id());

    // Prefix search of known hashes → a session is opened and parked.
    let resp = control
        .transfer()
        .search_prefix(SearchRequest {
            sequence_hashes: known.clone(),
        })
        .await
        .expect("search_prefix");
    assert!(matches!(resp, SearchResponse::Session { .. }));
    assert_eq!(session_manager.len(), 1);

    // Scatter search of the same hashes → a second session.
    let resp = control
        .transfer()
        .search_scatter(SearchRequest {
            sequence_hashes: known,
        })
        .await
        .expect("search_scatter");
    assert!(matches!(resp, SearchResponse::Session { .. }));
    assert_eq!(session_manager.len(), 2);

    // Hashes the G2 manager does not have → no session created.
    let resp = control
        .transfer()
        .search_prefix(SearchRequest {
            sequence_hashes: hashes_from(10_000, 3),
        })
        .await
        .expect("search_prefix");
    assert!(matches!(resp, SearchResponse::NoBlocksFound));
    assert_eq!(session_manager.len(), 2);
}

// ---- SessionManager eviction ----------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_manager_evicts_on_terminal_lifecycle_event() {
    let manager = SessionManager::new(Handle::current(), Duration::from_secs(30));

    let factory = MockSessionFactory::new();
    let session = factory.open(uuid::Uuid::new_v4()).expect("open");
    let mock = factory.last_opened().expect("last_opened");

    manager.register(session);
    assert_eq!(manager.len(), 1);

    // A terminal lifecycle event makes the watcher evict the entry.
    mock.inject_lifecycle(LifecycleEvent::Detached {
        reason: Some("test".to_string()),
    });
    wait_until(|| manager.is_empty()).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn session_manager_evicts_on_watchdog_timeout() {
    // Short watchdog: no lifecycle event is ever injected, so only the
    // timeout path can evict the session.
    let manager = SessionManager::new(Handle::current(), Duration::from_millis(150));

    let factory = MockSessionFactory::new();
    let session = factory.open(uuid::Uuid::new_v4()).expect("open");

    manager.register(session);
    assert_eq!(manager.len(), 1);

    wait_until(|| manager.is_empty()).await;
}
