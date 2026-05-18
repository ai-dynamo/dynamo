// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reproducer suite for hub-side `BlockLayoutMode` compatibility enforcement.
//!
//! These tests drive the design contract before any production code lands
//! (per `feedback_reproducer_first`). They reference:
//!
//! - `kvbm_common::shape::CanonicalBlockShape` (added in Phase 1)
//! - `kvbm_protocols::control::layout_compat::LayoutCompatPayload` (Phase 2)
//! - `kvbm_hub::protocol::ConditionalDisaggConfig.layout_compat` (Phase 5)
//! - `kvbm_hub::ConditionalDisaggManager` baseline state (Phase 6)
//!
//! All seven tests must FAIL TO COMPILE on `wt/kvcc` HEAD as of plan
//! Phase 0. After Phases 1–6 land they must pass without modification.

use std::net::{IpAddr, Ipv4Addr};
use std::sync::Arc;

use kvbm_common::shape::CanonicalBlockShape;
use kvbm_common::{BlockLayoutMode, KvBlockLayout};
use kvbm_hub::protocol::{
    ConditionalDisaggConfig, ConditionalDisaggRole, ErrorBody, Feature, P2pConfig, RegisterRequest,
    paths, peers_by_instance,
};
use kvbm_hub::{ConditionalDisaggManager, FeatureManager, HubServer, P2pManager};
use kvbm_protocols::control::LayoutConfigDescription;
use kvbm_protocols::control::layout_compat::LayoutCompatPayload;
use velo_ext::{InstanceId, PeerInfo, WorkerAddress};

// ---- fixtures --------------------------------------------------------------

async fn start_server_with_cd() -> (HubServer, Arc<ConditionalDisaggManager>) {
    let cd: Arc<ConditionalDisaggManager> = Arc::new(ConditionalDisaggManager::new());
    let p2p: Arc<P2pManager> = Arc::new(P2pManager::new());
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_feature_manager(p2p as Arc<dyn FeatureManager>)
        .add_feature_manager(Arc::clone(&cd) as Arc<dyn FeatureManager>)
        .serve()
        .await
        .expect("start CD hub");
    (server, cd)
}

fn make_peer() -> PeerInfo {
    PeerInfo::new(
        InstanceId::new_v4(),
        WorkerAddress::from_encoded(b"test-addr".to_vec()),
    )
}

fn control_url(server: &HubServer, path: &str) -> String {
    format!("http://{}{}", server.control_addr(), path)
}

fn discovery_url(server: &HubServer, path: &str) -> String {
    format!("http://{}{}", server.discovery_addr(), path)
}

fn http() -> reqwest::Client {
    reqwest::Client::new()
}

/// Default canonical aggregate used across the reproducers. Values are
/// chosen to mirror a TP=2 / PP=1 leader with 64 heads, layer count 32,
/// head_dim 128, fp16 dtype, page_size 16.
fn default_canonical() -> CanonicalBlockShape {
    CanonicalBlockShape {
        num_layers_total: 32,
        outer_dim: 2,
        page_size: 16,
        num_heads_total: 64,
        head_dim: 128,
        dtype_width_bytes: 2,
    }
}

/// Default per-worker layout config consistent with [`default_canonical`].
/// `num_heads` is per-worker (TP=2 → 32 per rank). `inner_dim = num_heads *
/// head_dim` so the operational equality predicate has the full picture.
fn default_layout_config() -> LayoutConfigDescription {
    LayoutConfigDescription {
        num_blocks: 1024,
        num_layers: 32,
        outer_dim: 2,
        page_size: 16,
        inner_dim: 32 * 128,
        alignment: 256,
        dtype_width_bytes: 2,
        num_heads: Some(32),
    }
}

fn payload(mode: BlockLayoutMode) -> LayoutCompatPayload {
    LayoutCompatPayload {
        mode,
        canonical: Some(default_canonical()),
        per_worker_layout: match mode {
            BlockLayoutMode::Operational => KvBlockLayout::OperationalNHD,
            BlockLayoutMode::Universal => KvBlockLayout::Universal,
        },
        per_worker_config: default_layout_config(),
        tp_size: 2,
        pp_size: 1,
    }
}

/// Construct a CD+P2P bundle for the common "leader with a role and a
/// layout_compat payload" case. Pass `None` for `layout` to test the
/// CD-without-P2P rejection path (legacy).
fn cd_features(role: ConditionalDisaggRole, layout: Option<LayoutCompatPayload>) -> Vec<Feature> {
    let cd = Feature::ConditionalDisagg(ConditionalDisaggConfig { role });
    match layout {
        Some(payload) => vec![Feature::P2P(P2pConfig { layout_compat: payload }), cd],
        None => vec![cd],
    }
}

async fn post_register(
    server: &HubServer,
    peer: &PeerInfo,
    role: ConditionalDisaggRole,
    layout: Option<LayoutCompatPayload>,
) -> reqwest::Response {
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: cd_features(role, layout),
    };
    http()
        .post(control_url(server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .expect("POST /v1/instances")
}

// ---- HTTP-level rejection tests --------------------------------------------

#[tokio::test]
async fn cross_mode_rejected_at_register() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload(BlockLayoutMode::Operational)),
    )
    .await;
    assert_eq!(resp.status(), 200, "prefill (operational) should register");

    let resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(payload(BlockLayoutMode::Universal)),
    )
    .await;
    assert_eq!(
        resp.status(),
        400,
        "decode with mismatched mode should reject"
    );
    let body: ErrorBody = resp.json().await.unwrap();
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("mode") || msg.contains("operational") || msg.contains("universal"),
        "rejection reason should name the mode mismatch, got: {}",
        body.message
    );

    // Rollback: decode entry must not be registered.
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(d_peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404, "decode entry should be rolled back");

    // Prefill survives.
    let resp = http()
        .get(discovery_url(
            &server,
            &peers_by_instance(p_peer.instance_id()),
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200, "prefill entry should survive rejection");
}

#[tokio::test]
async fn operational_mismatch_rejected_at_register() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    // Prefill: baseline.
    let resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload(BlockLayoutMode::Operational)),
    )
    .await;
    assert_eq!(resp.status(), 200);

    // Decode: same mode, divergent canonical (different num_heads_total).
    let mut mismatched = payload(BlockLayoutMode::Operational);
    mismatched.canonical.as_mut().unwrap().num_heads_total = 48;
    let resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(mismatched),
    )
    .await;
    assert_eq!(resp.status(), 400, "divergent canonical should reject");
    let body: ErrorBody = resp.json().await.unwrap();
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("num_heads") || msg.contains("canonical"),
        "rejection reason should name canonical mismatch, got: {}",
        body.message
    );
}

#[tokio::test]
async fn universal_mismatch_rejected_at_register() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload(BlockLayoutMode::Universal)),
    )
    .await;
    assert_eq!(resp.status(), 200);

    let mut mismatched = payload(BlockLayoutMode::Universal);
    mismatched.canonical.as_mut().unwrap().head_dim = 64;
    let resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(mismatched),
    )
    .await;
    assert_eq!(resp.status(), 400);
    let body: ErrorBody = resp.json().await.unwrap();
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("head_dim") || msg.contains("canonical"),
        "rejection should name head_dim mismatch, got: {}",
        body.message
    );
}

#[tokio::test]
async fn legacy_payload_without_layout_compat_rejected() {
    // c2: the hub gate is mandatory — CD registers without the
    // accompanying Feature::P2P (which carries layout_compat) must
    // be rejected at the server before any FeatureManager runs.
    let (server, _cd) = start_server_with_cd().await;
    let peer = make_peer();

    let resp = post_register(&server, &peer, ConditionalDisaggRole::Prefill, None).await;
    assert_eq!(
        resp.status(),
        400,
        "CD register without Feature::P2P (no layout_compat) must reject"
    );
    let body: ErrorBody = resp.json().await.unwrap();
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("p2p"),
        "rejection reason should name the missing P2P feature, got: {}",
        body.message
    );
}

/// Reproducer for c2: CD without P2P in the same register request must
/// be rejected at the server pre-dispatch layer. Distinct from the
/// "no layout_compat" case above — this asserts the cross-feature
/// dependency check fires even when CD's own config is well-formed.
#[tokio::test]
async fn cd_register_without_p2p_feature_is_rejected() {
    let (server, cd) = start_server_with_cd().await;
    let peer = make_peer();

    // Build a request with ONLY Feature::ConditionalDisagg, no P2P.
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![Feature::ConditionalDisagg(ConditionalDisaggConfig {
            role: ConditionalDisaggRole::Prefill,
        })],
    };
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .expect("POST /v1/instances");

    assert_eq!(resp.status(), 400, "CD without P2P must reject");
    let body: ErrorBody = resp.json().await.unwrap();
    assert!(
        body.message.to_lowercase().contains("p2p"),
        "rejection reason should name the missing P2P feature, got: {}",
        body.message
    );

    // No registration state should have leaked through.
    assert!(cd.snapshot().prefill.is_empty());
    assert!(cd.snapshot().decode.is_empty());
}

/// Reproducer for c2: P2P-only register with a self-inconsistent
/// layout_compat payload must reject (validate_self path, now owned
/// by P2pManager).
#[tokio::test]
async fn p2p_register_with_self_inconsistent_payload_is_rejected() {
    let (server, _cd) = start_server_with_cd().await;
    let peer = make_peer();

    let mut bad = payload(BlockLayoutMode::Universal);
    bad.canonical = None; // Universal mode requires a canonical aggregate.

    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![Feature::P2P(P2pConfig { layout_compat: bad })],
    };
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .expect("POST /v1/instances");

    assert_eq!(resp.status(), 400, "self-inconsistent P2P payload must reject");
    let body: ErrorBody = resp.json().await.unwrap();
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("internally inconsistent") || msg.contains("canonical"),
        "rejection reason should name validate_self failure, got: {}",
        body.message
    );
}

#[tokio::test]
async fn matching_shapes_accepted_across_roles() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let p_resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload(BlockLayoutMode::Universal)),
    )
    .await;
    assert_eq!(p_resp.status(), 200);

    let d_resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(payload(BlockLayoutMode::Universal)),
    )
    .await;
    assert_eq!(
        d_resp.status(),
        200,
        "matching universal payload across roles should accept"
    );
}

// ---- Unit-level manager state machine tests --------------------------------
//
// The layout-compat baseline lives in P2pManager after c2, so the
// state-machine tests below talk to P2pManager directly (not
// ConditionalDisaggManager).

fn p2p_feature(layout: LayoutCompatPayload) -> Feature {
    Feature::P2P(P2pConfig {
        layout_compat: layout,
    })
}

#[tokio::test]
async fn same_shape_idempotent_on_manager() {
    let mgr = P2pManager::new();
    let p_id = InstanceId::new_v4();
    let d_id = InstanceId::new_v4();

    mgr.on_register(p_id, &p2p_feature(payload(BlockLayoutMode::Operational)))
        .await
        .expect("baseline accepted");

    mgr.on_register(d_id, &p2p_feature(payload(BlockLayoutMode::Operational)))
        .await
        .expect("matching shape on second registration accepted");
}

#[tokio::test]
async fn baseline_cleared_when_population_zero() {
    let mgr = P2pManager::new();
    let p_id = InstanceId::new_v4();

    // First instance under Operational defines the baseline.
    mgr.on_register(p_id, &p2p_feature(payload(BlockLayoutMode::Operational)))
        .await
        .expect("first register accepted");
    assert!(mgr.has_baseline());

    // Last P2P instance leaves → baseline must clear so a fresh universal
    // group can take over without bouncing the hub.
    mgr.on_unregister(p_id);
    assert!(!mgr.has_baseline(), "baseline must clear when empty");

    let p2 = InstanceId::new_v4();
    mgr.on_register(p2, &p2p_feature(payload(BlockLayoutMode::Universal)))
        .await
        .expect("re-register with different mode after empty should accept");
}

#[tokio::test]
async fn operational_rejects_distinct_custom_permutations_at_hub() {
    // Codex stop-time review caught this: `KvBlockLayout::name()`
    // flattens every `Custom([..])` variant to `"custom"`, so a
    // string-typed wire field would silently accept two leaders with
    // different inner permutations. The wire type now carries
    // `KvBlockLayout` directly; this test pins the contract through the
    // hub's HTTP register surface.
    use kvbm_common::BlockDim;

    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let mut p_payload = payload(BlockLayoutMode::Operational);
    p_payload.per_worker_layout = KvBlockLayout::Custom([
        BlockDim::Layer,
        BlockDim::Outer,
        BlockDim::Page,
        BlockDim::Head,
    ]);
    let resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(p_payload),
    )
    .await;
    assert_eq!(
        resp.status(),
        200,
        "first Custom permutation should set the baseline"
    );

    let mut d_payload = payload(BlockLayoutMode::Operational);
    d_payload.per_worker_layout = KvBlockLayout::Custom([
        BlockDim::Outer,
        BlockDim::Layer,
        BlockDim::Page,
        BlockDim::Head,
    ]);
    let resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(d_payload),
    )
    .await;
    assert_eq!(
        resp.status(),
        400,
        "second instance with a *different* Custom permutation must reject"
    );
    let body: ErrorBody = resp.json().await.unwrap();
    assert!(
        body.message.contains("KvBlockLayout"),
        "rejection reason must name the per-worker layout mismatch, got: {}",
        body.message,
    );
}

#[tokio::test]
async fn first_universal_baseline_must_be_self_consistent() {
    // Codex stop-time review (round 2): the manager previously stored
    // the first payload as baseline without validating it. A first
    // universal payload with `canonical = None` would set an
    // unverifiable baseline and silently accept every subsequent peer
    // in the same mode. The fix runs `validate_self` before the first
    // baseline is stored; this test proves the rejection at the HTTP
    // boundary and confirms base registration is rolled back.
    let (server, cd) = start_server_with_cd().await;
    let peer = make_peer();

    let mut bad_first = payload(BlockLayoutMode::Universal);
    bad_first.canonical = None;
    let resp = post_register(
        &server,
        &peer,
        ConditionalDisaggRole::Prefill,
        Some(bad_first),
    )
    .await;
    assert_eq!(
        resp.status(),
        400,
        "first universal payload with canonical=None must reject"
    );
    let body: ErrorBody = resp.json().await.unwrap();
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("internally inconsistent") || msg.contains("canonical"),
        "rejection should name the missing canonical, got: {}",
        body.message
    );

    // Base registration rolled back; CD set still empty.
    assert!(cd.snapshot().prefill.is_empty());
    assert!(cd.snapshot().decode.is_empty());

    // The hub must not have stored a malformed baseline — a
    // subsequent well-formed universal registration must succeed.
    let good_peer = make_peer();
    let resp = post_register(
        &server,
        &good_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload(BlockLayoutMode::Universal)),
    )
    .await;
    assert_eq!(
        resp.status(),
        200,
        "well-formed universal must register after malformed baseline was rejected"
    );
}

#[tokio::test]
async fn first_universal_baseline_rejects_unknown_kv_block_layout() {
    // Universal mode requires every axis to be labeled. A first
    // payload with `KvBlockLayout::Unknown` is internally inconsistent
    // and must reject before setting the baseline.
    let (server, _cd) = start_server_with_cd().await;
    let peer = make_peer();

    let mut bad_first = payload(BlockLayoutMode::Universal);
    bad_first.per_worker_layout = KvBlockLayout::Unknown;
    let resp = post_register(
        &server,
        &peer,
        ConditionalDisaggRole::Prefill,
        Some(bad_first),
    )
    .await;
    assert_eq!(
        resp.status(),
        400,
        "first universal payload with Unknown KvBlockLayout must reject"
    );
}
