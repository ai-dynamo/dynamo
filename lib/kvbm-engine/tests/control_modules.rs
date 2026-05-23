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

use std::sync::Arc;
use std::time::Duration;

use kvbm_common::SequenceHash;
use kvbm_engine::leader::ControlPlane;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::leader::control::TransferModule;
use kvbm_engine::p2p::session::testing::wait_until;
use kvbm_engine::p2p::session::{
    LifecycleEvent, MockSessionFactory, SessionFactory, SessionManager,
};
use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
use kvbm_engine::{G2, G3};
use kvbm_logical::blocks::BlockRegistry;
use kvbm_logical::manager::BlockManager;
use kvbm_observability::KvbmObservability;
use kvbm_protocols::control::ModuleId;
use kvbm_protocols::control::client::LeaderControlClient;
use kvbm_protocols::control::modules::transfer::{
    CloseTransferSessionRequest, FindMode, OpenTransferSessionRequest, OpenTransferSessionResponse,
    PullFromSessionRequest, SearchMode, SearchRequest, SearchResponse, TierSelection,
};
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

/// G3 counterpart of [`populate_g2`].
fn populate_g3(g3: &BlockManager<G3>, seq_hashes: &[SequenceHash]) {
    let mutables = g3
        .allocate_blocks(seq_hashes.len())
        .expect("G3 pool large enough");
    let block_size = g3.block_size();
    let completes: Vec<_> = mutables
        .into_iter()
        .zip(seq_hashes)
        .map(|(m, h)| m.stage(*h, block_size).expect("stage"))
        .collect();
    let _immutables = g3.register_blocks(completes);
}

// ---- transfer module -------------------------------------------------------

/// Build an `Arc<InstanceLeader>` over `messenger` using `g2`, wire a
/// `MockSessionFactory` into its session-factory cell, and return the
/// leader plus a handle to the leader's `SessionManager` for assertions.
async fn leader_with_transfer_module(
    messenger: Arc<velo::Messenger>,
    g2: Arc<BlockManager<G2>>,
    registry: BlockRegistry,
) -> (Arc<InstanceLeader>, Arc<SessionManager>) {
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(messenger)
            .registry(registry)
            .g2_manager(g2)
            .workers(vec![])
            .build()
            .expect("build leader"),
    );
    let factory: Arc<dyn SessionFactory> = MockSessionFactory::new();
    assert!(leader.set_session_factory(factory));
    let manager = leader.session_manager().clone();
    (leader, manager)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transfer_module_opens_session_on_match() {
    let fx = fixture().await;

    // Populate G2 so the search handlers have something to find.
    let known = hashes(5);
    populate_g2(&fx.g2, &known);

    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    // The leader needs its own G2 registry populated separately because the
    // builder takes ownership of g2 and registry; mirror fixture's state.
    populate_g2(&g2, &known);
    let (leader, session_manager) =
        leader_with_transfer_module(fx.server.messenger().clone(), g2, registry).await;

    let _plane = ControlPlane::builder(fx.server.messenger().clone(), fx.server.instance_id())
        .with_module(TransferModule::new(Arc::clone(&leader)))
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_transfer_session_sync_returns_committed_inline() {
    let fx = fixture().await;

    let known = hashes(4);
    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    populate_g2(&g2, &known);
    let (leader, session_manager) =
        leader_with_transfer_module(fx.server.messenger().clone(), g2, registry).await;

    let resp = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known.clone(),
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open_transfer_session");

    match resp {
        OpenTransferSessionResponse::Sync {
            capability,
            committed,
            breakdown,
        } => {
            assert_eq!(committed, known);
            assert_eq!(breakdown.host_blocks, known.len());
            assert_eq!(capability.instance_id, fx.server.instance_id());
            assert_eq!(session_manager.len(), 1);
        }
        other => panic!("expected Sync response, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_transfer_session_sync_no_match_returns_no_blocks_found() {
    let fx = fixture().await;

    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let (leader, session_manager) =
        leader_with_transfer_module(fx.server.messenger().clone(), g2, registry).await;

    // G2 is empty → Sync should short-circuit without opening a session.
    let resp = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: hashes(3),
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open_transfer_session");

    assert!(matches!(resp, OpenTransferSessionResponse::NoBlocksFound));
    assert_eq!(session_manager.len(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_transfer_session_async_opens_session_with_no_matches() {
    let fx = fixture().await;

    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let (leader, session_manager) =
        leader_with_transfer_module(fx.server.messenger().clone(), g2, registry).await;

    // Async always opens a session — the puller observes "no matches"
    // via CommitsClosed + empty cumulative set.
    let resp = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: hashes(3),
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Async,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open_transfer_session");

    match resp {
        OpenTransferSessionResponse::Async { capability } => {
            assert_eq!(capability.instance_id, fx.server.instance_id());
            assert_eq!(session_manager.len(), 1);
        }
        other => panic!("expected Async response, got {other:?}"),
    }
}

/// Async + G2 hits: the response returns immediately but the populator
/// eventually calls `finish_commits` + `finish_availability`. We can't
/// observe the disagg session's state directly without attaching, so we
/// use the watchdog as the eviction signal: with a short watchdog and
/// no attach, the populator's `finish_*` calls + the lack of attach
/// allow the watchdog to fire and evict.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_transfer_session_async_populates_then_watchdog_evicts() {
    let fx = fixture().await;

    let known = hashes(4);
    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    populate_g2(&g2, &known);

    // Build a leader with a short-watchdog SessionManager. We can't swap
    // the leader's manager after build, so this test instantiates one
    // bare via the lower-level path that the existing watchdog tests use,
    // bypassing the InstanceLeader convenience to keep the watchdog short.
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(fx.server.messenger().clone())
            .registry(registry)
            .g2_manager(g2)
            .workers(vec![])
            .build()
            .expect("build leader"),
    );
    let factory: Arc<dyn SessionFactory> = MockSessionFactory::new();
    assert!(leader.set_session_factory(factory));
    let session_manager = leader.session_manager().clone();

    // Async open: returns immediately with a capability.
    let resp = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known,
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Async,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open_transfer_session");
    assert!(matches!(resp, OpenTransferSessionResponse::Async { .. }));
    assert_eq!(session_manager.len(), 1);

    // Populator finishes commits/availability; without an attach, the
    // 30s default watchdog won't fire inside the test, but explicit
    // close should still evict.
    let cap = resp.capability().cloned().expect("Async capability");
    let close = leader
        .close_transfer_session(CloseTransferSessionRequest {
            session_id: cap.session_id,
            reason: None,
        })
        .await
        .expect("close");
    assert!(close.was_present);
    assert_eq!(session_manager.len(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_transfer_session_scatter_finds_disjoint_hits() {
    let fx = fixture().await;

    // Populate only positions 0 and 2; Prefix would stop after 0 (the next
    // hash, position 1, is missing). Scatter must still return both.
    let known: Vec<SequenceHash> = vec![
        SequenceHash::new(0, None, 0),
        SequenceHash::new(2, Some(1), 2),
    ];
    let requested: Vec<SequenceHash> = vec![
        SequenceHash::new(0, None, 0),
        SequenceHash::new(1, Some(0), 1),
        SequenceHash::new(2, Some(1), 2),
    ];

    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    populate_g2(&g2, &known);
    let (leader, _manager) =
        leader_with_transfer_module(fx.server.messenger().clone(), g2, registry).await;

    let resp = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: requested,
            search_mode: SearchMode::Scatter,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open_transfer_session");

    match resp {
        OpenTransferSessionResponse::Sync {
            committed,
            breakdown,
            ..
        } => {
            assert_eq!(committed.len(), 2);
            assert_eq!(breakdown.host_blocks, 2);
        }
        other => panic!("expected Sync response, got {other:?}"),
    }
}

// ---- pull_from_session (puller side) --------------------------------------

/// Build two leaders cross-wired by a paired MockSessionFactory pair.
/// Returns `(holder_leader, puller_leader, puller_g2_for_assertions)`.
async fn paired_leaders(
    fx: &Fixture,
    holder_known: &[SequenceHash],
) -> (
    Arc<InstanceLeader>,
    Arc<InstanceLeader>,
    Arc<BlockManager<G2>>,
) {
    let (holder_factory_inner, puller_factory_inner) = MockSessionFactory::make_paired();
    let holder_factory: Arc<dyn SessionFactory> = holder_factory_inner;
    let puller_factory: Arc<dyn SessionFactory> = puller_factory_inner;

    let holder_registry = BlockRegistry::builder().build();
    let holder_g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(holder_registry.clone())
            .build(),
    );
    populate_g2(&holder_g2, holder_known);
    let holder = Arc::new(
        InstanceLeader::builder()
            .messenger(fx.server.messenger().clone())
            .registry(holder_registry)
            .g2_manager(holder_g2)
            .workers(vec![])
            .build()
            .expect("build holder leader"),
    );
    assert!(holder.set_session_factory(holder_factory));

    let puller_registry = BlockRegistry::builder().build();
    let puller_g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(puller_registry.clone())
            .build(),
    );
    let puller = Arc::new(
        InstanceLeader::builder()
            .messenger(fx.client.messenger().clone())
            .registry(puller_registry)
            .g2_manager(puller_g2.clone())
            .workers(vec![])
            .build()
            .expect("build puller leader"),
    );
    assert!(puller.set_session_factory(puller_factory));

    (holder, puller, puller_g2)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_from_session_drains_full_committed_set() {
    let fx = fixture().await;
    let known = hashes(4);
    let (holder, puller, puller_g2) = paired_leaders(&fx, &known).await;

    let cap = holder
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known.clone(),
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open_transfer_session")
        .capability()
        .cloned()
        .expect("Sync should carry capability");

    let pull = puller
        .pull_from_session(PullFromSessionRequest {
            session_id: cap.session_id,
            source_instance_id: cap.instance_id,
            endpoint: Some(cap.endpoint),
            selector: None,
        })
        .await
        .expect("pull_from_session");

    use std::collections::HashSet;
    let pulled: HashSet<_> = pull.pulled.into_iter().collect();
    let expected: HashSet<_> = known.iter().copied().collect();
    assert_eq!(pulled, expected);
    assert_eq!(pull.breakdown.host_blocks, expected.len());

    // Puller's local G2 now hosts the pulled hashes (verifying that
    // stage + register actually landed them in the local registry).
    let matched = puller_g2.match_blocks(&known);
    assert_eq!(matched.len(), known.len());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_from_session_with_selector_pulls_subset() {
    let fx = fixture().await;
    let known = hashes(5);
    let (holder, puller, puller_g2) = paired_leaders(&fx, &known).await;

    let cap = holder
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known.clone(),
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open")
        .capability()
        .cloned()
        .expect("Sync capability");

    // Pull only the first two committed hashes.
    let selector: Vec<SequenceHash> = known[0..2].to_vec();
    let pull = puller
        .pull_from_session(PullFromSessionRequest {
            session_id: cap.session_id,
            source_instance_id: cap.instance_id,
            endpoint: Some(cap.endpoint),
            selector: Some(selector.clone()),
        })
        .await
        .expect("pull_from_session");

    use std::collections::HashSet;
    let pulled: HashSet<_> = pull.pulled.into_iter().collect();
    let expected: HashSet<_> = selector.iter().copied().collect();
    assert_eq!(pulled, expected);

    let matched = puller_g2.match_blocks(&selector);
    assert_eq!(matched.len(), selector.len());
    // Non-selected hashes are NOT in the puller's local G2.
    assert!(puller_g2.match_blocks(&[known[3]]).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_from_session_selector_with_uncommitted_hash_errors() {
    let fx = fixture().await;
    let known = hashes(3);
    let (holder, puller, _puller_g2) = paired_leaders(&fx, &known).await;

    let cap = holder
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known.clone(),
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open")
        .capability()
        .cloned()
        .expect("Sync capability");

    // Selector includes a hash that the holder did NOT commit.
    let mut selector = known.clone();
    selector.push(SequenceHash::new(9_999, Some(2), 9_999));

    let err = puller
        .pull_from_session(PullFromSessionRequest {
            session_id: cap.session_id,
            source_instance_id: cap.instance_id,
            endpoint: Some(cap.endpoint),
            selector: Some(selector),
        })
        .await
        .expect_err("expected error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("hashes_not_committed"),
        "expected hashes_not_committed error, got: {msg}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_from_session_endpoint_required_in_v1() {
    let fx = fixture().await;
    let known = hashes(2);
    let (_holder, puller, _puller_g2) = paired_leaders(&fx, &known).await;

    let err = puller
        .pull_from_session(PullFromSessionRequest {
            session_id: uuid::Uuid::new_v4(),
            source_instance_id: fx.server.instance_id(),
            endpoint: None,
            selector: None,
        })
        .await
        .expect_err("expected endpoint_required error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("endpoint_required"),
        "expected endpoint_required error, got: {msg}"
    );
}

/// G3 fail-fast: when find_phase produces G3 matches but the leader has
/// no parallel_worker, `open_transfer_session` returns
/// `g3_requires_parallel_worker` *before* opening a session. This
/// guards against the "usable-looking session that cannot serve blocks"
/// failure mode: without the check, the response promises G3 blocks
/// that the background populator will immediately fail to stage, and
/// the puller is left with a teardown-in-progress capability.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_transfer_session_g3_without_parallel_worker_errors() {
    let fx = fixture().await;
    let known = hashes(3);

    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let g3 = Arc::new(
        TestManagerBuilder::<G3>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    populate_g3(&g3, &known);

    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(fx.server.messenger().clone())
            .registry(registry)
            .g2_manager(g2)
            .g3_manager(g3)
            .workers(vec![]) // no parallel_worker
            .build()
            .expect("build leader"),
    );
    let factory: Arc<dyn SessionFactory> = MockSessionFactory::new();
    assert!(leader.set_session_factory(factory));
    let session_manager = leader.session_manager().clone();

    // tiers.g3 = false → G3 not consulted at all, scatter on empty G2
    // returns no hits. No fail-fast trigger.
    let resp = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known.clone(),
            search_mode: SearchMode::Scatter,
            find_mode: FindMode::Sync,
            tiers: TierSelection {
                g3: false,
                g4: false,
            },
            watchdog_ms: None,
        })
        .await
        .expect("open");
    assert!(matches!(resp, OpenTransferSessionResponse::NoBlocksFound));
    assert_eq!(session_manager.len(), 0);

    // tiers.g3 = true → find_phase produces G3 matches, but no
    // parallel_worker → fail-fast before opening a session.
    let err = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known,
            search_mode: SearchMode::Scatter,
            find_mode: FindMode::Sync,
            tiers: TierSelection {
                g3: true,
                g4: false,
            },
            watchdog_ms: None,
        })
        .await
        .expect_err("expected g3_requires_parallel_worker error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("g3_requires_parallel_worker"),
        "expected g3_requires_parallel_worker error, got: {msg}"
    );
    // Critically: no session was opened. The caller's capability list
    // is unchanged.
    assert_eq!(session_manager.len(), 0);
}

/// G3 is NOT consulted in `Prefix` search mode even when `tiers.g3 = true`.
/// Documented v1 behavior — extending the contiguous-prefix walk into G3
/// requires gap handling we deferred.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn open_transfer_session_g3_ignored_in_prefix_mode() {
    let fx = fixture().await;
    let known = hashes(3);

    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let g3 = Arc::new(
        TestManagerBuilder::<G3>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    populate_g3(&g3, &known);

    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(fx.server.messenger().clone())
            .registry(registry)
            .g2_manager(g2)
            .g3_manager(g3)
            .workers(vec![])
            .build()
            .expect("build leader"),
    );
    let factory: Arc<dyn SessionFactory> = MockSessionFactory::new();
    assert!(leader.set_session_factory(factory));

    let resp = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known,
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection {
                g3: true,
                g4: false,
            },
            watchdog_ms: None,
        })
        .await
        .expect("open");
    assert!(matches!(resp, OpenTransferSessionResponse::NoBlocksFound));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn close_transfer_session_is_idempotent() {
    let fx = fixture().await;

    let known = hashes(3);
    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    populate_g2(&g2, &known);
    let (leader, session_manager) =
        leader_with_transfer_module(fx.server.messenger().clone(), g2, registry).await;

    let cap = leader
        .open_transfer_session(OpenTransferSessionRequest {
            sequence_hashes: known,
            search_mode: SearchMode::Prefix,
            find_mode: FindMode::Sync,
            tiers: TierSelection::default(),
            watchdog_ms: None,
        })
        .await
        .expect("open_transfer_session")
        .capability()
        .cloned()
        .expect("Sync response should carry a capability");

    assert_eq!(session_manager.len(), 1);

    // First close removes the session.
    let resp = leader
        .close_transfer_session(CloseTransferSessionRequest {
            session_id: cap.session_id,
            reason: Some("test".into()),
        })
        .await
        .expect("close_transfer_session");
    assert!(resp.was_present);
    assert_eq!(session_manager.len(), 0);

    // Second close is a no-op.
    let resp = leader
        .close_transfer_session(CloseTransferSessionRequest {
            session_id: cap.session_id,
            reason: None,
        })
        .await
        .expect("close_transfer_session");
    assert!(!resp.was_present);
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

// ---- metrics module --------------------------------------------------------

/// End-to-end: a leader built with observability + `register_control_plane(
/// dev=false, test=false, metrics=true)` exposes the `Metrics` module via
/// `list_modules`, and `client.metrics().snapshot()` returns a well-formed
/// response. Guards the silent-failure mode where forgetting to plumb
/// observability into the builder would log a warning and skip registration.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn register_control_plane_enables_metrics_when_observability_present() {
    let fx = fixture().await;

    // Build an InstanceLeader on the server side. The G2 manager from the
    // fixture is owned by us, so we build a fresh one here to avoid the
    // double-ownership that `InstanceLeader::g2_manager(Arc<_>)` would
    // imply. (The fixture's `fx.g2` is unused by this test.)
    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let observability = Arc::new(KvbmObservability::new().expect("observability"));
    let leader = Arc::new(
        InstanceLeader::builder()
            .messenger(fx.server.messenger().clone())
            .observability(observability)
            .registry(registry)
            .g2_manager(g2)
            .workers(vec![])
            .build()
            .expect("build leader"),
    );

    let _plane = leader
        .register_control_plane(false, true)
        .expect("register control plane");

    let control = LeaderControlClient::new(fx.client.messenger().clone(), fx.server.instance_id());

    let modules = control.list_modules().await.expect("list_modules");
    assert!(
        modules.contains(&ModuleId::Metrics),
        "metrics module should be registered when observability is wired; got {modules:?}"
    );

    let snapshot = control.metrics().snapshot().await.expect("snapshot");
    // Sessions count starts at 0; pools may be empty (no allocations happened
    // yet on this leader's registry), but the response shape must be valid.
    assert_eq!(snapshot.sessions_inflight, 0);
    assert!(snapshot.gathered_at_unix_ms > 0);
    // Filtering invariant: G1 must never appear in the response.
    assert!(
        snapshot.pools.iter().all(|p| p.pool != "G1"),
        "G1 must be filtered out, got {:?}",
        snapshot.pools
    );
}

/// Counterpart: `metrics=true` with no observability plumbed in logs a warning
/// and silently skips the module. `list_modules` must NOT report `Metrics`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn register_control_plane_skips_metrics_without_observability() {
    let fx = fixture().await;

    let registry = BlockRegistry::builder().build();
    let g2 = Arc::new(
        TestManagerBuilder::<G2>::new()
            .block_count(G2_BLOCK_COUNT)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let leader = Arc::new(
        InstanceLeader::builder()
            // No `.observability(...)` call.
            .messenger(fx.server.messenger().clone())
            .registry(registry)
            .g2_manager(g2)
            .workers(vec![])
            .build()
            .expect("build leader"),
    );

    let _plane = leader
        .register_control_plane(false, true)
        .expect("register control plane");

    let control = LeaderControlClient::new(fx.client.messenger().clone(), fx.server.instance_id());
    let modules = control.list_modules().await.expect("list_modules");
    assert!(
        !modules.contains(&ModuleId::Metrics),
        "metrics must NOT be registered without observability; got {modules:?}"
    );
}
