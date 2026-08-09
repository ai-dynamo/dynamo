// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Drives a real in-process Dynamo `SelectionService`.
//!
//! No GPUs, no worker processes, no HTTP: the selection plane runs inside the
//! test binary and is called through plain async methods. What this establishes
//! is the load-bearing claim of the design — that a turn can be *priced*
//! against the local fleet without booking anything, so the same turn can be
//! compared against a frontier model and sent elsewhere at no cost.

use std::collections::HashMap;
use std::sync::Arc;

use roundhouse_core::context::{ByteTokenizer, ContextAssembler};
use roundhouse_core::item::Item;
use roundhouse_fleet::{
    EmbeddedFleet, FleetQuery, KvRouterConfig, LocalFleet, SelectionServiceBuilder,
    WorkerRegistration,
};

const BLOCK_SIZE: u32 = 16;
const MODEL: &str = "test-model";

/// Build an embedded fleet with replica sync disabled.
///
/// At one instance the selector-to-selector ZMQ path never runs: no
/// `replica_sync` call means no PUB socket, no peer list, and none of the
/// O(N^2) peer-mesh configuration that multi-replica deployments need.
async fn embedded_fleet() -> Arc<EmbeddedFleet> {
    let config = KvRouterConfig {
        // Workers here publish no KV events, so the indexer stays empty and
        // overlap comes only from active-sequence tracking.
        use_kv_events: false,
        router_queue_threshold: None,
        ..Default::default()
    };
    let service = SelectionServiceBuilder::new(config)
        .indexer_threads(1)
        .build()
        .await
        .expect("embedded selection service should start");
    Arc::new(EmbeddedFleet::new(Arc::new(service)))
}

async fn register(fleet: &EmbeddedFleet, worker_id: u64) {
    fleet
        .register_worker(WorkerRegistration {
            worker_id,
            model_name: MODEL.to_string(),
            routing_group: "default".to_string(),
            endpoint: format!("http://worker-{worker_id}:8000"),
            block_size: BLOCK_SIZE,
            kv_events_endpoints: HashMap::new(),
        })
        .await
        .expect("worker registration should succeed");
}

fn query_from(assembler: &ContextAssembler<ByteTokenizer>) -> FleetQuery {
    FleetQuery {
        model_name: MODEL.to_string(),
        routing_group: "default".to_string(),
        block_hashes: assembler
            .buffer()
            .block_hashes()
            .iter()
            .map(|hash| hash.0)
            .collect(),
        sequence_hashes: assembler.buffer().sequence_hashes().to_vec(),
        isl_tokens: assembler.buffer().isl_tokens(),
        expected_output_tokens: Some(128),
        session_id: Some("sess_integration".to_string()),
    }
}

fn assembler_with_turns(turns: usize) -> ContextAssembler<ByteTokenizer> {
    let mut assembler = ContextAssembler::new(ByteTokenizer, BLOCK_SIZE);
    assembler.push(Item::system_text(
        "You are a careful assistant working through a long task.",
    ));
    for turn in 0..turns {
        assembler.push(Item::user_text(format!(
            "Step {turn}: please continue the analysis with enough text to fill blocks."
        )));
    }
    assembler
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_empty_fleet_prices_as_unavailable_rather_than_failing() {
    let fleet = embedded_fleet().await;
    let assembler = assembler_with_turns(1);

    // No workers registered. This is a routing input, not an error: the
    // frontier path may still be viable, and the policy layer decides.
    let quote = fleet.price(&query_from(&assembler)).await.unwrap();
    assert!(quote.is_none(), "an empty fleet must not yield a quote");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pricing_is_query_only_and_books_no_load() {
    let fleet = embedded_fleet().await;
    register(&fleet, 1).await;
    let assembler = assembler_with_turns(4);
    let query = query_from(&assembler);

    let first = fleet.price(&query).await.unwrap().expect("a worker is registered");
    let second = fleet.price(&query).await.unwrap().expect("a worker is registered");

    // Distinct pending selections, and neither booked anything -- the whole
    // point of the select/reserve split. An implementation that booked on
    // select would make comparison shopping impossible.
    assert_ne!(first.selection_id, second.selection_id);
    assert_eq!(first.worker_id, 1);
    assert_eq!(first.isl_tokens, assembler.buffer().isl_tokens());

    let loads = fleet
        .service()
        .loads(Some(MODEL), Some("default"))
        .into_iter()
        .flat_map(|model| model.loads)
        .map(|load| load.potential_prefill_tokens)
        .sum::<usize>();
    assert_eq!(loads, 0, "select must not book prefill load");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_quote_can_be_abandoned_for_a_frontier_target() {
    let fleet = embedded_fleet().await;
    register(&fleet, 1).await;
    let assembler = assembler_with_turns(3);

    // Price the local option, then walk away -- the frontier won this turn.
    let _abandoned = fleet.price(&query_from(&assembler)).await.unwrap().unwrap();

    // The fleet is undisturbed: a later turn prices exactly as if the
    // abandoned quote never happened.
    let later = fleet.price(&query_from(&assembler)).await.unwrap().unwrap();
    assert_eq!(later.effective_prefill_tokens, _abandoned.effective_prefill_tokens);
    assert_eq!(later.load, Some(0.0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_full_reservation_lifecycle_settles_cleanly() {
    let fleet = embedded_fleet().await;
    register(&fleet, 1).await;
    let assembler = assembler_with_turns(5);

    let quote = fleet.price(&query_from(&assembler)).await.unwrap().unwrap();
    let selection_id = quote.selection_id.clone();

    let reservation = Arc::clone(&fleet).reserve(&quote).await.unwrap();
    assert_eq!(reservation.selection_id(), selection_id);

    // Booking is visible as load; this is what a second, concurrent turn would
    // see and weigh against a frontier alternative.
    let booked = fleet
        .service()
        .loads(Some(MODEL), Some("default"))
        .into_iter()
        .flat_map(|model| model.loads)
        .map(|load| load.potential_prefill_tokens)
        .sum::<usize>();
    assert!(booked > 0, "a reservation must register prefill load");

    reservation.prefill_complete().await.unwrap();
    reservation.output_block().await.unwrap();
    reservation.release().await.unwrap();

    // Settled: load returns to zero. Skipping any of these calls would leave
    // the router permanently overestimating this worker.
    let after = fleet
        .service()
        .loads(Some(MODEL), Some("default"))
        .into_iter()
        .flat_map(|model| model.loads)
        .map(|load| load.potential_prefill_tokens)
        .sum::<usize>();
    assert_eq!(after, 0, "release must return the worker to idle");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_selection_id_can_only_be_booked_once() {
    let fleet = embedded_fleet().await;
    register(&fleet, 1).await;
    let assembler = assembler_with_turns(2);

    let quote = fleet.price(&query_from(&assembler)).await.unwrap().unwrap();
    let reservation = Arc::clone(&fleet).reserve(&quote).await.unwrap();

    // Replaying a consumed selection must fail rather than double-book. This
    // is what makes a client retry after a dropped connection safe.
    let replay = Arc::clone(&fleet).reserve(&quote).await;
    assert!(replay.is_err(), "a consumed selection must not book twice");

    reservation.release().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn growing_context_costs_more_prefill_and_the_quote_tracks_it() {
    let fleet = embedded_fleet().await;
    register(&fleet, 1).await;

    let short = assembler_with_turns(2);
    let long = assembler_with_turns(20);

    let short_quote = fleet.price(&query_from(&short)).await.unwrap().unwrap();
    let long_quote = fleet.price(&query_from(&long)).await.unwrap().unwrap();

    assert!(long_quote.isl_tokens > short_quote.isl_tokens);
    // With a cold cache, effective prefill tracks ISL. The interesting case --
    // effective prefill far *below* ISL -- needs a warm worker, which requires
    // KV events from a real engine.
    assert!(
        long_quote.effective_prefill_tokens > short_quote.effective_prefill_tokens,
        "a longer cold context must cost more prefill"
    );

    // Both project onto the shared axis the policy compares on.
    let candidate = long_quote.to_candidate(0.6, 80.0);
    assert_eq!(candidate.expected_cost_usd, 0.0);
    assert!(candidate.load.is_some(), "local load must be observable");
}
