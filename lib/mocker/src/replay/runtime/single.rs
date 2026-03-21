// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::core::ReplayWorkerCore;
use crate::common::protocols::{DirectRequest, MockEngineArgs};
use crate::replay::{TraceCollector, TraceSimulationReport};
use std::collections::VecDeque;
use validator::Validate;

pub(crate) fn simulate_trace_single(
    args: MockEngineArgs,
    mut requests: Vec<DirectRequest>,
) -> anyhow::Result<TraceSimulationReport> {
    args.validate()?;

    requests.sort_by(|left, right| {
        let left_ts = left
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        let right_ts = right
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        left_ts.total_cmp(&right_ts)
    });

    let first_arrival_ms = requests
        .first()
        .and_then(|request| request.arrival_timestamp_ms)
        .ok_or_else(|| anyhow::anyhow!("trace replay requires at least one timestamped request"))?;
    let mut pending = VecDeque::from(
        requests
            .into_iter()
            .map(|mut request| {
                let arrival_timestamp_ms = request
                    .arrival_timestamp_ms
                    .expect("trace replay requests must have an arrival timestamp")
                    - first_arrival_ms;
                request.arrival_timestamp_ms = Some(arrival_timestamp_ms);
                request
            })
            .collect::<Vec<_>>(),
    );

    let mut worker = ReplayWorkerCore::new(&args);
    let mut collector = TraceCollector::default();
    let mut current_time_ms = 0.0;

    while !pending.is_empty() || !worker.is_empty() {
        enqueue_trace_arrivals(&mut pending, &mut worker, &mut collector, current_time_ms);

        if worker.is_empty() {
            let Some(next_arrival_ms) = pending
                .front()
                .and_then(|request| request.arrival_timestamp_ms)
            else {
                break;
            };
            current_time_ms = next_arrival_ms;
            enqueue_trace_arrivals(&mut pending, &mut worker, &mut collector, current_time_ms);
            continue;
        }

        let prefill_time = worker.run_prefill_step(&args, &mut collector, current_time_ms);
        current_time_ms += prefill_time.as_secs_f64() * 1000.0;
        enqueue_trace_arrivals(&mut pending, &mut worker, &mut collector, current_time_ms);

        let decode_time = worker.run_decode_step(&args, &mut collector, current_time_ms);
        current_time_ms += decode_time.as_secs_f64() * 1000.0;
    }

    Ok(collector.finish())
}

pub(crate) fn simulate_concurrency_single(
    args: MockEngineArgs,
    requests: Vec<DirectRequest>,
    max_in_flight: usize,
) -> anyhow::Result<TraceSimulationReport> {
    args.validate()?;

    let mut pending = VecDeque::from(requests);
    let mut worker = ReplayWorkerCore::new(&args);
    let mut collector = TraceCollector::default();
    let mut current_time_ms = 0.0;

    while !pending.is_empty() || !worker.is_empty() {
        enqueue_concurrency_arrivals(
            &mut pending,
            &mut worker,
            &mut collector,
            current_time_ms,
            max_in_flight,
        );

        if worker.is_empty() {
            break;
        }

        let prefill_time = worker.run_prefill_step(&args, &mut collector, current_time_ms);
        current_time_ms += prefill_time.as_secs_f64() * 1000.0;

        let decode_time = worker.run_decode_step(&args, &mut collector, current_time_ms);
        current_time_ms += decode_time.as_secs_f64() * 1000.0;
    }

    Ok(collector.finish())
}

fn enqueue_trace_arrivals(
    pending: &mut VecDeque<DirectRequest>,
    worker: &mut ReplayWorkerCore,
    collector: &mut TraceCollector,
    current_time_ms: f64,
) {
    loop {
        let Some(next_arrival_ms) = pending
            .front()
            .and_then(|request| request.arrival_timestamp_ms)
        else {
            break;
        };
        if next_arrival_ms > current_time_ms {
            break;
        }

        let request = pending
            .pop_front()
            .expect("front request must exist when arrival is available");
        let arrival_ms = request
            .arrival_timestamp_ms
            .expect("trace replay requests must have an arrival timestamp");
        let input_length = request.tokens.len();
        let output_length = request.max_output_tokens;
        let uuid = worker.receive(request);
        collector.on_arrival(uuid, arrival_ms, input_length, output_length);
    }
}

fn enqueue_concurrency_arrivals(
    pending: &mut VecDeque<DirectRequest>,
    worker: &mut ReplayWorkerCore,
    collector: &mut TraceCollector,
    current_time_ms: f64,
    max_in_flight: usize,
) {
    while worker.num_requests() < max_in_flight {
        let Some(mut request) = pending.pop_front() else {
            break;
        };

        request.arrival_timestamp_ms = Some(current_time_ms);
        let input_length = request.tokens.len();
        let output_length = request.max_output_tokens;
        let uuid = worker.receive(request);
        collector.on_arrival(uuid, current_time_ms, input_length, output_length);
    }
}
