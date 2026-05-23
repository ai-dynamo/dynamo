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
use std::time::SystemTime;

use kvbm_common::shape::CanonicalBlockShape;
use kvbm_common::{BlockLayoutMode, KvBlockLayout};
use kvbm_hub::protocol::{
    ConditionalDisaggConfig, ConditionalDisaggRole, ErrorBody, Feature, P2pConfig, RegisterRequest,
    instance_by_id, instance_describe, paths, peers_by_instance,
};
use kvbm_hub::{
    ConditionalDisaggManager, ControlPlaneManager, FeatureManager, HubServer, P2pManager,
};
use kvbm_protocols::control::layout_compat::LayoutCompatPayload;
use kvbm_protocols::control::{HostInfo, InstanceDescription, LayoutConfigDescription};
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

fn payload(mode: BlockLayoutMode) -> LayoutCompatPayload {
    payload_with_topology(mode, 2, 1)
}

/// Build a payload with an explicit `(tp_size, pp_size)` decomposition.
///
/// The canonical aggregate stays fixed (matches [`default_canonical`])
/// while `per_worker_config` is rebuilt from
/// `(num_heads_total / tp_size, num_layers_total / pp_size, …)` so the
/// per-worker shapes are consistent with the requested decomposition.
///
/// Used by the c4 cross-topology tests to exercise the operative
/// Universal-mode invariant: same canonical, different `(TP, PP)`
/// decomposition → hub accepts.
fn payload_with_topology(
    mode: BlockLayoutMode,
    tp_size: usize,
    pp_size: usize,
) -> LayoutCompatPayload {
    let canonical = default_canonical();
    assert!(
        canonical.num_heads_total.is_multiple_of(tp_size),
        "payload_with_topology: tp_size={tp_size} must divide num_heads_total={}",
        canonical.num_heads_total
    );
    assert!(
        canonical.num_layers_total.is_multiple_of(pp_size),
        "payload_with_topology: pp_size={pp_size} must divide num_layers_total={}",
        canonical.num_layers_total
    );
    let num_heads = canonical.num_heads_total / tp_size;
    let num_layers = canonical.num_layers_total / pp_size;
    LayoutCompatPayload {
        mode,
        canonical: Some(canonical),
        per_worker_layout: match mode {
            BlockLayoutMode::Operational => KvBlockLayout::OperationalNHD,
            BlockLayoutMode::Universal => KvBlockLayout::Universal,
        },
        per_worker_config: LayoutConfigDescription {
            num_blocks: 1024,
            num_layers,
            outer_dim: canonical.outer_dim,
            page_size: canonical.page_size,
            inner_dim: num_heads * canonical.head_dim,
            alignment: 256,
            dtype_width_bytes: canonical.dtype_width_bytes,
            num_heads: Some(num_heads),
        },
        tp_size,
        pp_size,
    }
}

/// Construct a CD+P2P bundle for the common "leader with a role and a
/// layout_compat payload" case. Pass `None` for `layout` to test the
/// CD-without-P2P rejection path (legacy).
fn cd_features(role: ConditionalDisaggRole, layout: Option<LayoutCompatPayload>) -> Vec<Feature> {
    let cd = Feature::ConditionalDisagg(ConditionalDisaggConfig { role });
    match layout {
        Some(payload) => vec![
            Feature::P2P(P2pConfig {
                layout_compat: payload,
            }),
            cd,
        ],
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
        runtime: None,
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
        runtime: None,
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
        runtime: None,
    };
    let resp = http()
        .post(control_url(&server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .expect("POST /v1/instances");

    assert_eq!(
        resp.status(),
        400,
        "self-inconsistent P2P payload must reject"
    );
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

// ---- c4: cross-topology Universal-mode aggregate equality ------------------
//
// Universal mode's operative invariant: two leaders with the same
// un-sharded canonical aggregate are compatible at the hub gate
// regardless of how they slice the model across TP × PP worker grids.
// Same model, different decomposition. The three accept cases below
// form a triangle (TP=2,PP=2) ↔ (TP=4,PP=1) ↔ (TP=1,PP=4) over
// `default_canonical()` (num_heads_total=64, num_layers_total=32).
//
// Dependencies (any regression of these surfaces here):
//   - c1: single `KvBlockLayout::Universal` variant.
//   - c2: mandatory P2P gate (Feature::P2P carrying layout_compat).
//   - c3: connector pins G2/G3 to Universal in Universal mode, so the
//     wire payload's per_worker_layout is Universal on both sides.
//   - layout_compat predicate: Universal mode compares ONLY canonical,
//     ignoring tp_size/pp_size/per_worker_layout/per_worker_config.
//     If that ever changes, c4-1/2/3 fail loudly.

#[tokio::test]
async fn universal_accepts_cross_topology_tp2_pp2_vs_tp4_pp1() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let p_resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload_with_topology(BlockLayoutMode::Universal, 2, 2)),
    )
    .await;
    assert_eq!(p_resp.status(), 200, "prefill (TP=2, PP=2) must register");

    let d_resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(payload_with_topology(BlockLayoutMode::Universal, 4, 1)),
    )
    .await;
    assert_eq!(
        d_resp.status(),
        200,
        "decode (TP=4, PP=1) with same canonical aggregate must register",
    );
}

#[tokio::test]
async fn universal_accepts_cross_topology_tp2_pp2_vs_tp1_pp4() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let p_resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload_with_topology(BlockLayoutMode::Universal, 2, 2)),
    )
    .await;
    assert_eq!(p_resp.status(), 200);

    let d_resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(payload_with_topology(BlockLayoutMode::Universal, 1, 4)),
    )
    .await;
    assert_eq!(
        d_resp.status(),
        200,
        "decode (TP=1, PP=4) with same canonical aggregate must register",
    );
}

#[tokio::test]
async fn universal_accepts_cross_topology_tp4_pp1_vs_tp1_pp4() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let p_resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload_with_topology(BlockLayoutMode::Universal, 4, 1)),
    )
    .await;
    assert_eq!(p_resp.status(), 200);

    let d_resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(payload_with_topology(BlockLayoutMode::Universal, 1, 4)),
    )
    .await;
    assert_eq!(
        d_resp.status(),
        200,
        "(TP=4, PP=1) ↔ (TP=1, PP=4) with same canonical must register",
    );
}

/// Negative case: same (TP, PP) topology but different canonical
/// aggregate → universal mode still rejects. Pins the canonical-
/// equality requirement as the only thing universal mode checks.
#[tokio::test]
async fn universal_rejects_cross_topology_with_different_aggregate() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let p_resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload_with_topology(BlockLayoutMode::Universal, 2, 2)),
    )
    .await;
    assert_eq!(p_resp.status(), 200);

    let mut mismatched = payload_with_topology(BlockLayoutMode::Universal, 2, 2);
    mismatched.canonical.as_mut().unwrap().num_heads_total = 48;
    // Keep per_worker_config consistent with the mutated canonical so
    // we're testing canonical mismatch, not internal payload incoherence.
    mismatched.per_worker_config.num_heads = Some(24);
    mismatched.per_worker_config.inner_dim = 24 * 128;

    let d_resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(mismatched),
    )
    .await;
    assert_eq!(
        d_resp.status(),
        400,
        "different canonical num_heads_total must reject under universal",
    );
    let body: ErrorBody = resp_to_body(d_resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("num_heads") || msg.contains("canonical"),
        "rejection should name the diverging field; got: {}",
        body.message
    );
}

/// Operational mode is strict: even when canonical aggregates match,
/// different `(TP, PP)` decompositions reject because Operational
/// compares per-worker fields (including tp_size). Counter-test to
/// the universal_accepts_cross_topology_* cases — proves mode is the
/// dominant axis for "is cross-topology allowed?".
#[tokio::test]
async fn operational_rejects_cross_topology_even_when_canonical_matches() {
    let (server, _cd) = start_server_with_cd().await;
    let p_peer = make_peer();
    let d_peer = make_peer();

    let p_resp = post_register(
        &server,
        &p_peer,
        ConditionalDisaggRole::Prefill,
        Some(payload_with_topology(BlockLayoutMode::Operational, 2, 2)),
    )
    .await;
    assert_eq!(p_resp.status(), 200);

    let d_resp = post_register(
        &server,
        &d_peer,
        ConditionalDisaggRole::Decode,
        Some(payload_with_topology(BlockLayoutMode::Operational, 4, 1)),
    )
    .await;
    assert_eq!(
        d_resp.status(),
        400,
        "Operational mode rejects different (TP, PP) decomposition even \
         when canonical matches — tp_size is compared in Operational",
    );
}

/// Small inline helper for tests that destructure the response after
/// status assertion. Pulls the body out as ErrorBody so the test can
/// assert on the error message contents.
async fn resp_to_body(resp: reqwest::Response) -> ErrorBody {
    resp.json::<ErrorBody>()
        .await
        .expect("error response must deserialize as ErrorBody")
}

// ---- c5: describe-push layout_compat re-validation -------------------------
//
// `InstanceDescription` gains `Option<LayoutCompatPayload>`. The hub's
// `POST /v1/instances/{id}/describe` re-validates a `Some(payload)` against
// the stored `P2pManager` baseline. Defence-in-depth on top of the c2
// register-time gate: catches drift between a leader's register-time
// announcement and its describe-time announcement.
//
// Decision (per c5 plan): describe-with-Some-payload but no baseline
// registered is a protocol violation (HTTP 400). `None` payload is the
// legacy / pre-stamping snapshot path and bypasses validation.

async fn start_server_with_cpm_and_p2p() -> (HubServer, Arc<ControlPlaneManager>, Arc<P2pManager>) {
    let cpm: Arc<ControlPlaneManager> = Arc::new(ControlPlaneManager::new());
    let p2p: Arc<P2pManager> = Arc::new(P2pManager::new());
    cpm.set_p2p_manager(Arc::clone(&p2p));
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_feature_manager(Arc::clone(&p2p) as Arc<dyn FeatureManager>)
        .add_feature_manager(Arc::clone(&cpm) as Arc<dyn FeatureManager>)
        .serve()
        .await
        .expect("start hub with CPM + P2P");
    (server, cpm, p2p)
}

fn describe_payload(
    instance_id: InstanceId,
    layout_compat: Option<LayoutCompatPayload>,
) -> InstanceDescription {
    InstanceDescription {
        instance_id: instance_id.to_string(),
        worker_ids: vec![1],
        hub_instance_id: None,
        block_size: Some(16),
        parallelism: None,
        tier_capacity: Vec::new(),
        workers: Vec::new(),
        modules: Vec::new(),
        role: None,
        config: None,
        host: HostInfo {
            hostname: "c5-test".into(),
            pid: 1,
        },
        started_at: SystemTime::UNIX_EPOCH,
        layout_compat,
    }
}

async fn post_p2p_register(
    server: &HubServer,
    peer: &PeerInfo,
    layout: LayoutCompatPayload,
) -> reqwest::Response {
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![Feature::P2P(P2pConfig {
            layout_compat: layout,
        })],
        runtime: None,
    };
    http()
        .post(control_url(server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .expect("POST /v1/instances")
}

async fn post_bare_register(server: &HubServer, peer: &PeerInfo) -> reqwest::Response {
    let req = RegisterRequest {
        peer_info: peer.clone(),
        features: vec![],
        runtime: None,
    };
    http()
        .post(control_url(server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .expect("POST /v1/instances")
}

async fn post_describe(
    server: &HubServer,
    instance_id: InstanceId,
    body: &InstanceDescription,
) -> reqwest::Response {
    http()
        .post(control_url(server, &instance_describe(instance_id)))
        .json(body)
        .send()
        .await
        .expect("POST /describe")
}

#[tokio::test]
async fn describe_push_with_baseline_match_accepts() {
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer = make_peer();
    let baseline = payload(BlockLayoutMode::Operational);

    let resp = post_p2p_register(&server, &peer, baseline.clone()).await;
    assert_eq!(resp.status(), 200, "P2P register sets baseline");

    let descr = describe_payload(peer.instance_id(), Some(baseline));
    let resp = post_describe(&server, peer.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        200,
        "describe with matching layout_compat must accept"
    );
}

#[tokio::test]
async fn describe_push_with_baseline_divergence_rejects() {
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer = make_peer();
    let baseline = payload_with_topology(BlockLayoutMode::Operational, 2, 1);

    let resp = post_p2p_register(&server, &peer, baseline).await;
    assert_eq!(resp.status(), 200);

    // Divergent tp_size — operational mode treats tp_size as part of
    // equality (per check_layout_compat doc).
    let divergent = payload_with_topology(BlockLayoutMode::Operational, 4, 1);
    let descr = describe_payload(peer.instance_id(), Some(divergent));
    let resp = post_describe(&server, peer.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        400,
        "describe with divergent tp_size must reject under operational"
    );
    let body = resp_to_body(resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("tp_size") || msg.contains("baseline") || msg.contains("diverge"),
        "rejection should reference the divergence; got: {}",
        body.message
    );
}

#[tokio::test]
async fn describe_push_with_mode_divergence_rejects() {
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer = make_peer();
    let baseline = payload(BlockLayoutMode::Universal);

    let resp = post_p2p_register(&server, &peer, baseline).await;
    assert_eq!(resp.status(), 200);

    let divergent = payload(BlockLayoutMode::Operational);
    let descr = describe_payload(peer.instance_id(), Some(divergent));
    let resp = post_describe(&server, peer.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        400,
        "describe with divergent mode must reject"
    );
    let body = resp_to_body(resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("mode") || msg.contains("universal") || msg.contains("operational"),
        "rejection should reference the mode mismatch; got: {}",
        body.message
    );
}

#[tokio::test]
async fn describe_push_without_p2p_membership_rejects() {
    // Decision 1: describe carrying Some(layout_compat) from a leader
    // that did not register Feature::P2P is a protocol violation, HTTP
    // 400. The membership check fires before any baseline lookup —
    // here `inner.instances` is empty, so the rejection cites the
    // missing P2P registration rather than an absent baseline.
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer = make_peer();

    let resp = post_bare_register(&server, &peer).await;
    assert_eq!(resp.status(), 200, "bare register (no P2P) succeeds");

    let descr = describe_payload(
        peer.instance_id(),
        Some(payload(BlockLayoutMode::Operational)),
    );
    let resp = post_describe(&server, peer.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        400,
        "describe with layout_compat from non-P2P-member must reject"
    );
    let body = resp_to_body(resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("not registered") && msg.contains("p2p"),
        "rejection should reference the non-membership; got: {}",
        body.message
    );
}

#[tokio::test]
async fn describe_push_none_layout_compat_passes() {
    // Pre-stamping snapshot — None on layout_compat skips the
    // validation hook. Matches the existing Option-for-pre-stamping
    // contract on block_size / parallelism.
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer = make_peer();
    let baseline = payload(BlockLayoutMode::Operational);

    let resp = post_p2p_register(&server, &peer, baseline).await;
    assert_eq!(resp.status(), 200);

    let descr = describe_payload(peer.instance_id(), None);
    let resp = post_describe(&server, peer.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        200,
        "describe with None layout_compat must accept (pre-stamping path)"
    );
}

#[tokio::test]
async fn describe_push_none_without_baseline_passes() {
    // Legacy describe path: instance has no P2P registration AND
    // describe carries no layout_compat. Validation is a no-op.
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer = make_peer();

    let resp = post_bare_register(&server, &peer).await;
    assert_eq!(resp.status(), 200);

    let descr = describe_payload(peer.instance_id(), None);
    let resp = post_describe(&server, peer.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        200,
        "legacy describe (None + no baseline) must accept"
    );
}

#[tokio::test]
async fn describe_push_after_p2p_unregister_not_found() {
    // Regression safety net: P2pManager::on_unregister clears the
    // baseline when the last P2P instance leaves. The base registry
    // entry is also evicted on DELETE. POST /describe for an
    // unregistered instance returns 404 (existing behaviour) — proves
    // the post_describe handler keeps the registry-recheck guard ahead
    // of the new layout_compat hook.
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer = make_peer();
    let baseline = payload(BlockLayoutMode::Operational);

    let resp = post_p2p_register(&server, &peer, baseline.clone()).await;
    assert_eq!(resp.status(), 200);

    let resp = http()
        .delete(control_url(&server, &instance_by_id(peer.instance_id())))
        .send()
        .await
        .expect("DELETE /v1/instances/{id}");
    assert!(
        resp.status().is_success(),
        "unregister should succeed, got {}",
        resp.status()
    );

    let descr = describe_payload(peer.instance_id(), Some(baseline));
    let resp = post_describe(&server, peer.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        404,
        "describe for unregistered instance returns 404"
    );
}

// ---- c5 follow-on: force-pull describe must NOT bypass the gate -----------
//
// post_describe (the leader-push path) was the original c5 site. Two
// velo-pull paths bypassed the gate by writing the pulled payload
// directly to the cache:
//   - GET /v1/instances/{id}/describe?force=true (operator refresh)
//   - POST /v1/instances/{id}/control/core/describe_instance
// Both call commit_describe_if_registered directly. A force-pull
// returning a divergent layout_compat would slip past the c5 gate.
//
// The fix factored a shared `validate_describe_layout` helper invoked
// from all three sites. These reproducers exercise both velo-pull
// paths against a peer whose velo handler returns a divergent
// payload, asserting HTTP 400.

use kvbm_protocols::control::{ControlReply, DESCRIBE_INSTANCE_HANDLER, DescribeInstanceRequest};
use std::time::Duration;
use velo::Handler;
use velo::transports::tcp::TcpTransportBuilder;

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

async fn new_velo_peer() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

/// Hub fixture wiring CPM + P2P + velo transport — needed for the
/// force-pull reproducers. The plain `start_server_with_cpm_and_p2p`
/// fixture omits velo, which makes force-pull return 500 before
/// reaching the gate; the bypass concern only exists when velo IS
/// wired and the pull succeeds.
async fn start_server_with_cpm_and_p2p_and_velo()
-> (HubServer, Arc<ControlPlaneManager>, Arc<P2pManager>) {
    let transport = new_velo_transport();
    let cpm: Arc<ControlPlaneManager> = Arc::new(ControlPlaneManager::new());
    let p2p: Arc<P2pManager> = Arc::new(P2pManager::new());
    cpm.set_p2p_manager(Arc::clone(&p2p));
    let server = kvbm_hub::create_server_builder()
        .bind_addr(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .discovery_port(0)
        .control_port(0)
        .add_transport(transport as Arc<dyn velo::Transport>)
        .add_feature_manager(Arc::clone(&p2p) as Arc<dyn FeatureManager>)
        .add_feature_manager(Arc::clone(&cpm) as Arc<dyn FeatureManager>)
        .heartbeat_interval(Duration::from_secs(3600))
        .heartbeat_max_failures(u32::MAX)
        .registration_ttl(Duration::from_secs(3600))
        .serve()
        .await
        .expect("start hub with CPM + P2P + velo");
    (server, cpm, p2p)
}

fn install_canned_describe(peer: &velo::Velo, payload: InstanceDescription) {
    peer.register_handler(
        Handler::typed_unary_async::<DescribeInstanceRequest, _, _, _>(
            DESCRIBE_INSTANCE_HANDLER,
            move |_ctx| {
                let payload = payload.clone();
                async move { Ok(ControlReply::Ok(payload)) }
            },
        )
        .build(),
    )
    .unwrap();
}

async fn register_peer_with_p2p(
    server: &HubServer,
    peer: &velo::Velo,
    layout: LayoutCompatPayload,
) {
    let req = RegisterRequest {
        peer_info: peer.peer_info(),
        features: vec![Feature::P2P(P2pConfig {
            layout_compat: layout,
        })],
        runtime: None,
    };
    let resp = http()
        .post(control_url(server, paths::INSTANCES))
        .json(&req)
        .send()
        .await
        .expect("POST /v1/instances");
    assert_eq!(resp.status(), 200, "register peer with P2P feature");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn force_pull_describe_with_baseline_divergence_rejects() {
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p_and_velo().await;
    let peer = new_velo_peer().await;

    // Baseline: register peer with P1.
    let baseline = payload_with_topology(BlockLayoutMode::Operational, 2, 1);
    register_peer_with_p2p(&server, &peer, baseline).await;

    // Install velo handler returning a payload with divergent layout_compat (P2).
    let divergent = payload_with_topology(BlockLayoutMode::Operational, 4, 1);
    let pulled = describe_payload(peer.instance_id(), Some(divergent));
    install_canned_describe(&peer, pulled);

    // Force-pull. Hub fetches from peer's velo handler; the validator
    // must reject before commit.
    let url = control_url(
        &server,
        &format!("{}?force=true", instance_describe(peer.instance_id())),
    );
    let resp = http().get(&url).send().await.expect("GET /describe?force");
    assert_eq!(
        resp.status(),
        400,
        "force-pull with divergent layout_compat must reject"
    );
    let body = resp_to_body(resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("tp_size") || msg.contains("baseline") || msg.contains("diverge"),
        "rejection should reference the divergence; got: {}",
        body.message
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn core_describe_instance_with_baseline_divergence_rejects() {
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p_and_velo().await;
    let peer = new_velo_peer().await;

    let baseline = payload(BlockLayoutMode::Universal);
    register_peer_with_p2p(&server, &peer, baseline).await;

    // Divergent mode: Operational against Universal baseline.
    let divergent = payload(BlockLayoutMode::Operational);
    let pulled = describe_payload(peer.instance_id(), Some(divergent));
    install_canned_describe(&peer, pulled);

    // POST /v1/instances/{id}/control/core/describe_instance — the
    // typed velo-pull endpoint. Same bypass concern as force=true.
    let url = control_url(
        &server,
        &format!(
            "/v1/instances/{}/control/core/describe_instance",
            peer.instance_id()
        ),
    );
    let resp = http()
        .post(&url)
        .json(&serde_json::json!({}))
        .send()
        .await
        .expect("POST /control/core/describe_instance");
    assert_eq!(
        resp.status(),
        400,
        "core_describe_instance pull with divergent layout_compat must reject"
    );
    let body = resp_to_body(resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("mode") || msg.contains("universal") || msg.contains("operational"),
        "rejection should reference the mode mismatch; got: {}",
        body.message
    );
}

// ---- c5 follow-on: per-instance P2P membership check ---------------------
//
// Codex stop-time finding: `check_describe_layout` runs the candidate
// against the baseline without verifying that the pushing leader is
// itself a P2P-registered member. With two leaders in play (one P2P
// registers and sets the baseline, the other bare-registers), the
// non-P2P leader can push a describe with `Some(layout_compat)` and
// either get silently accepted (matching layout) or rejected with a
// misleading "diverges from baseline" message (divergent layout).
//
// Fix: `check_describe_layout` rejects with a "not registered with
// Feature::P2P" error before running the layout check when the pushing
// instance is absent from `inner.instances`.

#[tokio::test]
async fn describe_push_from_non_p2p_member_with_matching_baseline_rejects() {
    // Two leaders: A registers with P2P (sets baseline), B bare-registers.
    // B pushes a describe whose `layout_compat` matches the baseline. The
    // payload is "valid" against the baseline, but B is not a P2P member
    // and has no business stamping `layout_compat` in the first place.
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer_a = make_peer();
    let peer_b = make_peer();
    let baseline = payload(BlockLayoutMode::Operational);

    let resp = post_p2p_register(&server, &peer_a, baseline.clone()).await;
    assert_eq!(resp.status(), 200, "P2P register sets baseline");

    let resp = post_bare_register(&server, &peer_b).await;
    assert_eq!(resp.status(), 200, "bare register (no P2P) for B");

    let descr = describe_payload(peer_b.instance_id(), Some(baseline));
    let resp = post_describe(&server, peer_b.instance_id(), &descr).await;
    assert_eq!(
        resp.status(),
        400,
        "describe-push from a non-P2P-member must reject even if layout matches baseline"
    );
    let body = resp_to_body(resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("not registered") && msg.contains("p2p"),
        "rejection should reference the non-membership; got: {}",
        body.message
    );
}

#[tokio::test]
async fn describe_push_from_non_p2p_member_with_divergent_baseline_rejects_with_membership_reason()
{
    // Same two-leader setup, but B pushes a divergent layout. Current
    // code rejects with "diverges from baseline" — semantically wrong:
    // B was never a member, so divergence against the baseline isn't
    // the failure mode. Fix flips the message to the membership reason.
    let (server, _cpm, _p2p) = start_server_with_cpm_and_p2p().await;
    let peer_a = make_peer();
    let peer_b = make_peer();
    let baseline = payload_with_topology(BlockLayoutMode::Operational, 2, 1);

    let resp = post_p2p_register(&server, &peer_a, baseline).await;
    assert_eq!(resp.status(), 200);

    let resp = post_bare_register(&server, &peer_b).await;
    assert_eq!(resp.status(), 200);

    let divergent = payload_with_topology(BlockLayoutMode::Operational, 4, 1);
    let descr = describe_payload(peer_b.instance_id(), Some(divergent));
    let resp = post_describe(&server, peer_b.instance_id(), &descr).await;
    assert_eq!(resp.status(), 400);
    let body = resp_to_body(resp).await;
    let msg = body.message.to_lowercase();
    assert!(
        msg.contains("not registered") && msg.contains("p2p"),
        "rejection should reference non-membership (not 'diverges from baseline'); got: {}",
        body.message
    );
}
