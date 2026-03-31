// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio::time::Instant;

use crate::common::protocols::OutputSignal;
use crate::replay::router::ReplayRouter;
use crate::replay::{TraceCollector, TraceSimulationReport};
use crate::scheduler::AdmissionEvent;

use super::state::{ArrivalEvent, RequestRegistry, SharedLiveRuntimeStats, now_ms};

async fn process_output_signal(
    output: OutputSignal,
    batch_time_ms: f64,
    collector: &mut TraceCollector,
    requests: &RequestRegistry,
    router: &ReplayRouter,
    stats: &SharedLiveRuntimeStats,
) {
    collector.on_token(output.uuid, batch_time_ms);

    let Some(state) = requests.get(&output.uuid) else {
        return;
    };

    if state.mark_first_token_once() {
        match router.on_first_token(output.uuid).await {
            Ok(true) => stats.record_prefill_marked(),
            Ok(false) => {}
            Err(error) => tracing::warn!(
                uuid = %output.uuid,
                error = %error,
                "online replay failed to mark prefill completed"
            ),
        }
    }

    if !output.completed || !state.mark_completed_once() {
        return;
    }

    match router.on_complete(output.uuid).await {
        Ok(true) => stats.record_freed(),
        Ok(false) => {}
        Err(error) => tracing::warn!(
            uuid = %output.uuid,
            error = %error,
            "online replay failed to free completed request"
        ),
    }
    state.notify_completion();
}

pub(super) async fn run_demux(
    start: Instant,
    mut arrival_rx: mpsc::UnboundedReceiver<ArrivalEvent>,
    mut admission_rx: mpsc::UnboundedReceiver<AdmissionEvent>,
    mut output_rx: mpsc::UnboundedReceiver<Vec<OutputSignal>>,
    requests: RequestRegistry,
    router: Arc<ReplayRouter>,
    stats: Arc<SharedLiveRuntimeStats>,
) -> TraceSimulationReport {
    let mut collector = TraceCollector::default();
    let mut arrivals_open = true;
    let mut admissions_open = true;
    let mut outputs_open = true;

    loop {
        if !arrivals_open && !admissions_open && !outputs_open {
            break;
        }

        tokio::select! {
            biased;
            arrival = arrival_rx.recv(), if arrivals_open => {
                match arrival {
                    Some(arrival) => collector.on_arrival(
                        arrival.uuid,
                        arrival.at_ms,
                        arrival.input_tokens,
                        arrival.output_tokens,
                    ),
                    None => arrivals_open = false,
                }
            }
            admission = admission_rx.recv(), if admissions_open => {
                match admission {
                    Some(admission) => {
                        collector.on_admit(admission.uuid, now_ms(start), admission.reused_input_tokens);
                    }
                    None => admissions_open = false,
                }
            }
            output = output_rx.recv(), if outputs_open => {
                match output {
                    Some(output_batch) => {
                        let batch_time_ms = now_ms(start);
                        for output in output_batch {
                            process_output_signal(
                                output,
                                batch_time_ms,
                                &mut collector,
                                &requests,
                                &router,
                                &stats,
                            )
                            .await;
                        }
                    }
                    None => outputs_open = false,
                }
            }
        }
    }

    collector.finish().with_wall_time_ms(now_ms(start))
}
