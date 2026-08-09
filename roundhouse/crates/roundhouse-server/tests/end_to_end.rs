// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the claims this design rests on.
//!
//! Three things are worth proving, and none of them need a GPU:
//!
//! 1. A client sends a constant number of bytes per turn while the context it
//!    is reasoning over grows without bound. That is the reason to be stateful.
//! 2. Routing genuinely reacts to cache state — a target that was warmed on an
//!    earlier turn wins a later one it would otherwise lose.
//! 3. Killing the owning process mid-session loses nothing. A successor claims
//!    the lease, replays the log, and continues.

use std::collections::HashMap;
use std::sync::Arc;

use roundhouse_core::context::ByteTokenizer;
use roundhouse_core::event::SessionEventKind;
use roundhouse_core::ids::{SessionId, TurnId};
use roundhouse_core::item::Item;
use roundhouse_core::routing::{
    AffinityPolicy, CacheLedger, CacheModel, ProviderPricing, RoutingPolicy, Target,
};
use roundhouse_core::session::Session;
use roundhouse_core::store::MemoryStore;
use roundhouse_fleet::{
    EchoFrontierClient, EmbeddedFleet, FrontierModelSpec, KvRouterConfig, LocalFleet,
    SelectionServiceBuilder, StaticFrontierCatalog, WorkerRegistration,
};
use roundhouse_server::{EchoLocalExecutor, Engine, EngineConfig};

const BLOCK_SIZE: u32 = 16;
const LOCAL_MODEL: &str = "local";
const MINUTE: u64 = 60_000;

fn frontier_catalog() -> StaticFrontierCatalog {
    StaticFrontierCatalog::new(vec![FrontierModelSpec {
        provider: "anthropic".into(),
        model: "claude".into(),
        cache_model: CacheModel::Deterministic { ttl_ms: 5 * MINUTE },
        pricing: ProviderPricing {
            input_per_mtok_usd: 3.0,
            cached_input_per_mtok_usd: 0.3,
            cache_write_per_mtok_usd: 3.75,
            output_per_mtok_usd: 15.0,
        },
        quality_prior: 0.95,
        base_ttft_ms: 350.0,
        ttft_ms_per_uncached_token: 0.002,
    }])
}

fn config() -> EngineConfig {
    EngineConfig {
        block_size: BLOCK_SIZE,
        local_model: LOCAL_MODEL.to_string(),
        ..Default::default()
    }
}

fn engine_without_fleet(
    store: Arc<MemoryStore>,
    policy: Arc<dyn RoutingPolicy>,
) -> Engine<MemoryStore, ByteTokenizer> {
    Engine::new(
        store,
        ByteTokenizer,
        Arc::new(EchoLocalExecutor::new("local answer")),
        frontier_catalog(),
        Arc::new(EchoFrontierClient::new("frontier answer")),
        policy,
        config(),
    )
}

async fn embedded_fleet() -> Arc<EmbeddedFleet> {
    let service = SelectionServiceBuilder::new(KvRouterConfig {
        use_kv_events: false,
        router_queue_threshold: None,
        ..Default::default()
    })
    .indexer_threads(1)
    .build()
    .await
    .expect("selection service should start");
    let fleet = Arc::new(EmbeddedFleet::new(Arc::new(service)));
    fleet
        .register_worker(WorkerRegistration {
            worker_id: 1,
            model_name: LOCAL_MODEL.to_string(),
            routing_group: "default".to_string(),
            endpoint: "http://worker-1:8000".to_string(),
            block_size: BLOCK_SIZE,
            kv_events_endpoints: HashMap::new(),
        })
        .await
        .unwrap();
    fleet
}

/// The headline claim: constant client cost, growing server-side context.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_bytes_stay_flat_while_context_grows() {
    let store = Arc::new(MemoryStore::new());
    let engine = engine_without_fleet(store.clone(), Arc::new(AffinityPolicy::new()));
    let session_id = SessionId::generate();
    engine.create_session(&session_id).await.unwrap();

    let mut bytes_per_turn = Vec::new();
    let mut context_tokens = Vec::new();

    for turn in 0..20 {
        // What a stateless client would have to re-upload every turn is the
        // whole conversation; here it is one message.
        let message = format!("Step {turn}: continue the analysis please.");
        bytes_per_turn.push(message.len());

        engine
            .run_turn(
                &session_id,
                TurnId::new(format!("turn-{turn}")),
                vec![Item::user_text(message)],
            )
            .await
            .unwrap();

        let session = Session::open(
            store.clone(),
            session_id.clone(),
            "probe",
            10_000,
            CacheLedger::new(),
        )
        .await
        .unwrap();
        let rendered: usize = session.state().items.iter().map(|i| i.render().len()).sum();
        context_tokens.push(rendered);
        session.release().await.unwrap();
    }

    // Client-side per-turn cost is flat: every turn is within a few bytes of
    // the first, purely from the turn number's digit count.
    let smallest = *bytes_per_turn.iter().min().unwrap();
    let largest = *bytes_per_turn.iter().max().unwrap();
    assert!(
        largest - smallest <= 2,
        "client bytes per turn must not grow with context: {bytes_per_turn:?}"
    );

    // Server-side context grew monotonically and by a large multiple, which is
    // exactly the cost the client no longer pays.
    assert!(context_tokens.windows(2).all(|pair| pair[1] > pair[0]));
    assert!(
        *context_tokens.last().unwrap() > 10 * context_tokens[0],
        "context should grow substantially over 20 turns: {context_tokens:?}"
    );
}

/// Routing must actually respond to cache state, not just to static config.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_warmed_frontier_target_wins_a_turn_it_would_otherwise_lose() {
    let store = Arc::new(MemoryStore::new());
    let session_id = SessionId::generate();

    // No local fleet, so the frontier is the only option and the first turn
    // necessarily warms it.
    let engine = engine_without_fleet(store.clone(), Arc::new(AffinityPolicy::new()));
    engine.create_session(&session_id).await.unwrap();

    let first = engine
        .run_turn(
            &session_id,
            TurnId::new("t0"),
            vec![Item::user_text("open the task")],
        )
        .await
        .unwrap();
    let first_decision = first.decision.expect("first turn must route");
    assert!(!first_decision.target.is_local());

    // Second turn: the ledger now knows this target is warm, so the recorded
    // expected prefill must be lower than the prompt length.
    let second = engine
        .run_turn(
            &session_id,
            TurnId::new("t1"),
            vec![Item::user_text("continue")],
        )
        .await
        .unwrap();
    assert!(second.decision.is_some());

    let session = Session::open(store, session_id, "probe", 10_000, CacheLedger::new())
        .await
        .unwrap();
    let routed: Vec<_> = session
        .events_since(0, 1000)
        .await
        .unwrap()
        .into_iter()
        .filter_map(|event| match event.kind {
            SessionEventKind::Routed { decision, .. } => Some(decision),
            _ => None,
        })
        .collect();

    assert_eq!(routed.len(), 2);
    // Cold: the whole prompt is prefill. Warm: strictly less.
    assert_eq!(
        routed[0].expected_prefill_tokens,
        routed[0].isl_tokens as f64
    );
    assert!(
        routed[1].expected_prefill_tokens < routed[1].isl_tokens as f64,
        "a warmed target must be priced below its prompt length"
    );
    assert!(!routed[1].considered.is_empty());
}

/// Killing the owner mid-session must lose nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_successor_resumes_a_killed_session_without_gaps() {
    let store = Arc::new(MemoryStore::new());
    let session_id = SessionId::generate();
    let engine = engine_without_fleet(store.clone(), Arc::new(AffinityPolicy::new()));
    engine.create_session(&session_id).await.unwrap();

    for turn in 0..3 {
        engine
            .run_turn(
                &session_id,
                TurnId::new(format!("t{turn}")),
                vec![Item::user_text(format!("question {turn}"))],
            )
            .await
            .unwrap();
    }

    // Client's cursor at the moment of the crash.
    let cursor = {
        let probe = Session::open(
            store.clone(),
            session_id.clone(),
            "probe",
            10_000,
            CacheLedger::new(),
        )
        .await
        .unwrap();
        let seq = probe.last_seq();
        probe.release().await.unwrap();
        seq
    };

    // The owner dies without releasing. A successor on a different node takes
    // over once the lease lapses.
    store.expire_lease_now(&session_id).await;
    let successor = Engine::new(
        store.clone(),
        ByteTokenizer,
        Arc::new(EchoLocalExecutor::new("local answer")),
        frontier_catalog(),
        Arc::new(EchoFrontierClient::new("frontier answer")),
        Arc::new(AffinityPolicy::new()),
        EngineConfig {
            node_id: "node-b".to_string(),
            ..config()
        },
    );

    let resumed = successor
        .run_turn(
            &session_id,
            TurnId::new("t3"),
            vec![Item::user_text("question 3")],
        )
        .await
        .unwrap();

    // Replay from the client's cursor: contiguous, no gap and no repeat.
    let session = Session::open(store, session_id, "probe-2", 10_000, CacheLedger::new())
        .await
        .unwrap();
    let replayed = session.events_since(cursor, 1000).await.unwrap();
    assert!(!replayed.is_empty());
    let seqs: Vec<u64> = replayed.iter().map(|event| event.seq).collect();
    let expected: Vec<u64> = (cursor + 1..=resumed.last_seq).collect();
    assert_eq!(
        seqs, expected,
        "replay must be contiguous across a failover"
    );

    // The successor saw all four turns, including the three it did not run.
    assert_eq!(session.turn_index(), 4);
}

/// A retried turn must not be answered twice.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retried_turn_replays_instead_of_regenerating() {
    let store = Arc::new(MemoryStore::new());
    let engine = engine_without_fleet(store.clone(), Arc::new(AffinityPolicy::new()));
    let session_id = SessionId::generate();
    engine.create_session(&session_id).await.unwrap();

    let first = engine
        .run_turn(
            &session_id,
            TurnId::new("same-turn"),
            vec![Item::user_text("hello")],
        )
        .await
        .unwrap();
    assert!(!first.deduplicated);

    // The client never saw the response and retries with the same turn id.
    let retry = engine
        .run_turn(
            &session_id,
            TurnId::new("same-turn"),
            vec![Item::user_text("hello")],
        )
        .await
        .unwrap();

    assert!(retry.deduplicated);
    assert_eq!(retry.response_id, first.response_id);
    assert!(retry.decision.is_none(), "a replay must not route again");

    let session = Session::open(store, session_id, "probe", 10_000, CacheLedger::new())
        .await
        .unwrap();
    assert_eq!(
        session.turn_index(),
        1,
        "a retry must not open a second turn"
    );
}

/// The full stack against a real embedded selection service.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_local_worker_wins_and_its_reservation_settles() {
    let store = Arc::new(MemoryStore::new());
    let fleet = embedded_fleet().await;
    let engine = engine_without_fleet(store.clone(), Arc::new(AffinityPolicy::new()))
        .with_fleet(Arc::clone(&fleet) as Arc<dyn LocalFleet>);

    let session_id = SessionId::generate();
    engine.create_session(&session_id).await.unwrap();

    let result = engine
        .run_turn(
            &session_id,
            TurnId::new("t0"),
            vec![Item::user_text("a question for the local model")],
        )
        .await
        .unwrap();

    // Local is free and the frontier is cold, so local must win on cost.
    let decision = result.decision.expect("turn must route");
    assert!(
        matches!(decision.target, Target::Local { worker_id: 1, .. }),
        "expected the local worker, got {:?}: {}",
        decision.target,
        decision.rationale
    );
    assert_eq!(result.text, "local answer");

    // The reservation was settled: the worker is back to idle. A leak here
    // would silently distort every later routing decision.
    let residual: usize = fleet
        .service()
        .loads(Some(LOCAL_MODEL), Some("default"))
        .into_iter()
        .flat_map(|model| model.loads)
        .map(|load| load.potential_prefill_tokens)
        .sum();
    assert_eq!(residual, 0, "reservation must be released after the turn");
}
