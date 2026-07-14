// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-worker `SteppableEngine` behavior, which the crate's other steppable
//! tests reach only through cancellation.

use dynamo_mocker::common::protocols::{DirectRequest, EngineType, MockEngineArgs};
use dynamo_mocker::loadgen::{EngineEvent, SteppableEngine, SteppableReplay};
use dynamo_mocker::replay::ReplayTerminalStatus;
use uuid::Uuid;

fn request(uuid: u128, max_output_tokens: usize) -> DirectRequest {
    DirectRequest {
        tokens: (0..64).collect(),
        max_output_tokens,
        uuid: Some(Uuid::from_u128(uuid)),
        ..Default::default()
    }
}

// Distinct token ids per uuid so the oversized prompt cannot alias a cached one.
fn reject_request(uuid: u128, prompt_tokens: u32) -> DirectRequest {
    let base = uuid as u32 * 100_000;
    DirectRequest {
        tokens: (base..base + prompt_tokens).collect(),
        max_output_tokens: 8,
        uuid: Some(Uuid::from_u128(uuid)),
        ..Default::default()
    }
}

// 16-token budget (4 blocks * block_size 4): an oversized prompt overflows it.
fn reject_args() -> MockEngineArgs {
    MockEngineArgs::builder()
        .engine_type(EngineType::Trtllm)
        .block_size(4)
        .num_gpu_blocks(4)
        .enable_prefix_caching(false)
        .speedup_ratio(1000.0)
        .build()
        .unwrap()
}

fn drive_to_idle(engine: &mut SteppableEngine) -> Vec<EngineEvent> {
    let mut events = Vec::new();
    let mut steps = 0;
    while !engine.is_idle() {
        events.extend(engine.step().unwrap().events);
        steps += 1;
        assert!(steps < 1_000, "engine did not drain");
    }
    events
}

#[test]
fn drives_request_to_terminal() {
    for (args, req, expected, completed, output_tokens) in [
        (
            MockEngineArgs::default(),
            request(1, 8),
            ReplayTerminalStatus::Completed,
            1,
            8,
        ),
        (
            reject_args(),
            reject_request(1, 200),
            ReplayTerminalStatus::Rejected,
            0,
            0,
        ),
    ] {
        let mut engine = SteppableEngine::new(args);
        let uuid = engine.submit(req).unwrap();
        let terminal = drive_to_idle(&mut engine)
            .into_iter()
            .find(|e| e.uuid == uuid && e.terminal_status.is_some())
            .expect("request must reach a terminal event");
        assert_eq!(terminal.terminal_status, Some(expected));

        let report = engine.take_report(engine.now_ms());
        assert_eq!(report.request_counts.completed_requests, completed);
        assert_eq!(report.request_counts.total_output_tokens, output_tokens);
    }
}

#[test]
fn reports_per_request_measurements_after_completion() {
    let mut engine = SteppableEngine::new(MockEngineArgs::default());
    engine.set_capture_per_request(true);
    let uuid = engine.submit(request(1, 8)).unwrap();
    assert_eq!(engine.in_flight(), 1);

    drive_to_idle(&mut engine);

    assert_eq!(engine.in_flight(), 0);
    assert_eq!(engine.actual_output_length(uuid), Some(8));
    assert!(engine.request_latencies(uuid).is_some());
}
