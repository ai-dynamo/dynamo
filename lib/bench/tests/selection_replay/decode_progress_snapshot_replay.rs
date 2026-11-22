// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconstruct the public scheduler observations at a recorded decision boundary.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_kv_events::event::RouterEvent;
use dynamo_kv_events::residency::StorageTier;
use dynamo_kv_router::plugins::prelude::*;
use dynamo_scheduler::host::{
    PromptRequest, SelectionAdmission, SelectionOperation, SelectionOutcome, SessionBinding,
};
use dynamo_slot_tracker::host::DecodeProgressUpdate;
use serde::{Deserialize, Serialize};

use super::corpus::{TraceFormat, WorkloadParams};
use super::host::{BenchHost, router_config};
use super::kv_cache::{KvCaches, PromptBlocks};

const INPUTS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/decode_progress_replay"
);

fn warm() -> WorkerWithDpRank {
    WorkerWithDpRank::new(1, 0)
}

fn cold() -> WorkerWithDpRank {
    WorkerWithDpRank::new(2, 0)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    block_size: u32,
    foreground_prompt_tokens: usize,
    warm_overlap_blocks: usize,
    workers_kv_capacity: usize,
    requests: Vec<Owner>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Owner {
    prompt_tokens: usize,
    output_limit: u32,
    generated_tokens: usize,
}

#[derive(Serialize)]
struct Reading {
    active_requests: u64,
    warm_overlap_blocks: f64,
    observed_requests: usize,
    remaining_token_budget: usize,
}

struct ProgressReader(Arc<Mutex<Option<Reading>>>);

impl WorkerPicker for ProgressReader {
    fn required_worker_inputs(&self) -> WorkerInputs {
        use dynamo_kv_router::plugins::worker_selection::experimental::decode_progress::DECODE_PROGRESS;
        WorkerInputs::CACHE | WorkerInputs::ACTIVE_REQUESTS | DECODE_PROGRESS
    }

    fn pick(
        &self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        use dynamo_kv_router::plugins::worker_selection::experimental::decode_progress::DecodeProgressByRow;
        let row_for = |worker| {
            input
                .candidates()
                .iter()
                .position(|row| row.worker() == worker)
                .unwrap()
        };
        if let Some(pinned) = context.pinned_worker() {
            return Ok(row_for(pinned));
        }
        let warm_row = row_for(warm());
        let cold_row = row_for(cold());
        let cache = input.cache().unwrap();
        let progress = input.decode_progress(warm_row).unwrap();
        assert_eq!(input.active_requests(cold_row), Some(0));
        assert_eq!(cache.get(cold_row).unwrap().device_overlap_blocks(), 0.0);
        assert!(input.decode_progress(cold_row).is_none());
        let reading = Reading {
            active_requests: input.active_requests(warm_row).unwrap(),
            warm_overlap_blocks: cache.get(warm_row).unwrap().device_overlap_blocks(),
            observed_requests: progress.observed_requests(),
            remaining_token_budget: progress.remaining_token_budget(),
        };
        assert!(self.0.lock().unwrap().replace(reading).is_none());
        // This policy only reads the input contract. The baseline run makes the scored choice.
        Ok(cold_row)
    }
}

fn params(case: &Case) -> WorkloadParams {
    WorkloadParams {
        trace_format: TraceFormat::Mooncake,
        trace_block_size: None,
        lanes: 1,
        workers: 2,
        dp_size: 1,
        block_size: case.block_size,
        ignore_trace_delays: false,
        warmup_ms: 1,
        window_ms: 1,
        num_gpu_blocks: case.workers_kv_capacity,
        max_num_batched_tokens: 32_770,
        max_num_seqs: case.requests.len(),
        engine_speedup: 1.0,
    }
}

async fn select(
    host: &BenchHost,
    tokens: Vec<u32>,
    admission: SelectionAdmission,
    pin: Option<WorkerWithDpRank>,
    cap: Option<u32>,
) -> dynamo_scheduler::host::Selected {
    let prompt = PromptRequest {
        token_ids: Some(tokens),
        is_eagle: Some(false),
        ..Default::default()
    };
    let result = host
        .service
        .core()
        .run_selection(SelectionOperation {
            key: host.partition.key().clone(),
            accounting_attribution: Default::default(),
            prompt: prompt.view(),
            router_config_override: None,
            expected_output_tokens: cap,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            session: SessionBinding::None,
            affinity_target: None,
            pinned_worker: pin,
            allowed_worker_ids: None,
            routing_constraints: Default::default(),
            admission,
            track_active_blocks: host.track_active_blocks,
            return_routing_hashes: false,
            replay_id: None,
        })
        .await
        .result
        .unwrap();
    let SelectionOutcome::Selected(selected) = result else {
        panic!("unexpected queue rejection")
    };
    selected
}

async fn rebuild(case: &Case, host: BenchHost, check_headroom: bool) -> WorkerWithDpRank {
    let mut prompt = vec![1; case.foreground_prompt_tokens];
    prompt[case.warm_overlap_blocks * case.block_size as usize..].fill(2);
    let prefix = PromptBlocks::from_tokens(
        &prompt[..case.warm_overlap_blocks * case.block_size as usize],
        case.block_size,
    );
    let caches = KvCaches::new(2, 1, case.workers_kv_capacity, 1);
    let mut events = Vec::new();
    caches.prefill(0, &prefix, &mut Vec::new(), |batch| events = batch);
    let expected_events = events.len() as u64;
    for event in events {
        host.indexer
            .apply_event_routed(RouterEvent::with_storage_tier(
                1,
                event,
                StorageTier::Device,
            ))
            .await
            .unwrap();
    }
    tokio::time::timeout(Duration::from_secs(2), async {
        while host
            .index_events_applied()
            .get("stored/ok")
            .copied()
            .unwrap_or(0)
            < expected_events
        {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap();

    if check_headroom {
        let response = select(
            &host,
            prompt.clone(),
            SelectionAdmission::Query { request_id: None },
            None,
            Some(8),
        )
        .await
        .response;
        assert_eq!(response.best_worker, warm());
        assert_eq!(
            response.cached_tokens,
            case.warm_overlap_blocks * case.block_size as usize
        );
    }
    let scheduler = host.partition.scheduler();
    let mut bookings = Vec::new();
    for (index, owner) in case.requests.iter().enumerate() {
        let selected = select(
            &host,
            vec![index as u32 + 3; owner.prompt_tokens],
            SelectionAdmission::Lease {
                request_id: format!("background-{index}"),
            },
            Some(warm()),
            Some(owner.output_limit),
        )
        .await;
        assert_eq!(selected.response.best_worker, warm());
        let booking = selected.booking.unwrap().commit();
        scheduler
            .mark_prefill_completed_if_booking(&booking)
            .await
            .unwrap();
        bookings.push(booking);
    }
    for (booking, owner) in bookings.iter().zip(&case.requests) {
        scheduler
            .update_output_if_booking(
                booking,
                0,
                None,
                Some(DecodeProgressUpdate::observed(
                    owner.output_limit,
                    owner.generated_tokens,
                )),
            )
            .await
            .unwrap();
    }
    let selected = select(
        &host,
        prompt,
        SelectionAdmission::Query { request_id: None },
        None,
        Some(8),
    )
    .await;
    assert!(selected.booking.is_none());
    for booking in bookings {
        scheduler
            .booking_cleanup()
            .enqueue_acknowledged(booking)
            .wait()
            .await
            .unwrap();
    }
    host.shutdown().await;
    selected.response.best_worker
}

fn config(file: &str) -> KvRouterConfig {
    let path = Path::new(INPUTS).join(file).display().to_string();
    router_config(Some(
        &serde_json::json!({"router_policy_config": path}).to_string(),
    ))
    .unwrap()
}

#[tokio::test]
async fn replays_decode_progress_decision_snapshots() {
    let fixture: Fixture = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/testdata/decode_progress_replay/selection-inputs.json"
    )))
    .unwrap();
    let mut rows = Vec::new();
    for case in fixture.cases {
        let params = params(&case);
        let mut choices = BTreeMap::new();
        for (name, router) in [
            ("default", router_config(None).unwrap()),
            ("two_tier", config("baseline.yaml")),
        ] {
            let builtin = BenchHost::start(&router, Default::default(), &params)
                .await
                .unwrap();
            let choice = rebuild(&case, builtin, true).await;
            let worker = if choice == warm() {
                "warm"
            } else {
                assert_eq!(choice, cold());
                "cold"
            };
            choices.insert(name, worker);
        }

        let observed = Arc::new(Mutex::new(None));
        let capture = Arc::clone(&observed);
        let mut registry = dynamo_router_policies::default_registry();
        registry
            .register_worker_selection(
                "replay-progress-reader",
                Arc::new(move |_| {
                    let capture = Arc::clone(&capture);
                    Ok(Arc::new(move |config, role, _| {
                        WorkerSelectionPolicy::new(
                            config.clone(),
                            role.default_selector_label(),
                            Vec::new(),
                            Box::new(ProgressReader(Arc::clone(&capture))),
                        )
                    }))
                }),
            )
            .unwrap();
        let probe = BenchHost::start_with_registry(
            &config("reader.yaml"),
            Default::default(),
            &params,
            registry,
        )
        .await
        .unwrap();
        rebuild(&case, probe, false).await;
        let reading = observed.lock().unwrap().take().unwrap();
        assert_eq!(reading.active_requests, case.requests.len() as u64);
        assert_eq!(reading.observed_requests, case.requests.len());
        assert_eq!(reading.warm_overlap_blocks, case.warm_overlap_blocks as f64);
        assert_eq!(
            reading.remaining_token_budget,
            case.requests
                .iter()
                .map(|r| r.output_limit as usize - r.generated_tokens)
                .sum::<usize>()
        );
        rows.push(serde_json::json!({"id": case.id, "choices": choices, "visible": reading}));
    }
    assert_eq!(rows.len(), 4);
    if let Some(output) = std::env::var_os("DYNAMO_DECODE_REPLAY_OUTPUT") {
        std::fs::write(output, serde_json::to_string_pretty(&rows).unwrap() + "\n").unwrap();
    }
}
