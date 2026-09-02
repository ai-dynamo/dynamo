// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publication of neutral generalized-engine effects through Dynamo sinks.

use super::*;

#[derive(Clone)]
pub(super) struct RankDispatch {
    pub(super) external_dp_rank: u32,
    pub(super) event_tx: Option<SchedulerEventSender>,
    pub(super) kv_event_publishers: KvEventPublishers,
    pub(super) fpm_publisher: FpmPublisher,
    pub(super) lifecycle_tx: mpsc::Sender<SchedulerLifecycleEvent>,
    pub(super) metrics_tx: watch::Sender<MockerMetrics>,
}

struct PendingRankPublication {
    dp_rank: u32,
    lifecycle: Vec<SchedulerLifecycleEvent>,
    metrics: Metrics,
}

// Keep the timing inline: boxing this diagnostic would add an allocation to
// the completion path being measured.
#[allow(clippy::large_enum_variant)]
enum OutputPublication {
    Delivered {
        failed_requests: Vec<Uuid>,
        timing: OutputPublishTiming,
    },
    Cancelled,
}

enum CompletionDispatch {
    Completed,
    Cancelled,
}

const COMPLETION_TIMING_LOG_INTERVAL_PASSES: u64 = 512;

#[derive(Debug, Default, Clone, Copy)]
struct CompletionEventTiming {
    admission: std::time::Duration,
    reserve: std::time::Duration,
    reserve_waited: bool,
    predecessor_busy: std::time::Duration,
    receiver_wake: std::time::Duration,
}

#[derive(Debug, Default, Clone, Copy)]
pub(super) struct RouterPublishTiming {
    kv_publish: std::time::Duration,
    fpm_publish: std::time::Duration,
    completion_kv_events: u64,
    deferred_kv_events: u64,
}

#[derive(Debug, Default)]
struct CompletionPassTiming {
    completion_boundary: std::time::Duration,
    event: CompletionEventTiming,
    body_wall: std::time::Duration,
    kv_publish: std::time::Duration,
    fpm_publish: std::time::Duration,
    output_convert: std::time::Duration,
    output_publish_wall: std::time::Duration,
    output_admission: std::time::Duration,
    output_reserve: std::time::Duration,
    output_predecessor_busy: std::time::Duration,
    output_receiver_wake: std::time::Duration,
    output_gate_wait: std::time::Duration,
    output_route_wall: std::time::Duration,
    output_route_cpu_valid_wall: std::time::Duration,
    output_route_thread_cpu: std::time::Duration,
    output_route_wall_minus_thread_cpu_ms: f64,
    output_dispatcher_residual: std::time::Duration,
    output_ack_wake: std::time::Duration,
    output_residual_ms: f64,
    lifecycle_convert: std::time::Duration,
    failure_cleanup: std::time::Duration,
    lifecycle_publish: std::time::Duration,
    metrics_publish: std::time::Duration,
    before_finish: std::time::Duration,
    body_residual_ms: f64,
    finish_admission: std::time::Duration,
    finish_reserve: std::time::Duration,
    finish_send_sync: std::time::Duration,
    finish_actor_same_thread_send_sync: std::time::Duration,
    finish_actor_cross_thread_send_sync: std::time::Duration,
    finish_actor_wake: std::time::Duration,
    finish_actor_same_thread_wake: std::time::Duration,
    finish_actor_cross_thread_wake: std::time::Duration,
    finish_actor_same_thread_cpu_valid_wake: std::time::Duration,
    finish_actor_same_thread_cpu_valid_probe_wall: std::time::Duration,
    finish_actor_same_thread_cpu_time: std::time::Duration,
    finish_actor_same_thread_probe_wall_minus_cpu_ms: f64,
    finish_actor_same_thread_probe_edge_ms: f64,
    finish_return_wake: std::time::Duration,
    finish_return_same_thread_wake: std::time::Duration,
    finish_return_cross_thread_wake: std::time::Duration,
    boundary_identity_residual_ms: f64,
    rank_batches: u64,
    event_reserve_waited_passes: u64,
    output_batches: u64,
    output_reserve_waited_batches: u64,
    output_signals: u64,
    uninstrumented_output_signals: u64,
    terminal_signals: u64,
    route_found: u64,
    route_missing: u64,
    delivered: u64,
    full: u64,
    closed: u64,
    terminal_removals: u64,
    route_cpu_valid_batches: u64,
    route_cpu_invalid_batches: u64,
    cleanup_commands: u64,
    completion_kv_events: u64,
    deferred_kv_events: u64,
    lifecycle_events: u64,
    finish_reserve_waited_passes: u64,
    finish_actor_same_thread_passes: u64,
    finish_actor_cross_thread_passes: u64,
    finish_actor_same_thread_cpu_valid_passes: u64,
    finish_actor_same_thread_cpu_invalid_passes: u64,
    finish_actor_same_thread_endpoint_same_cpu_passes: u64,
    finish_actor_same_thread_endpoint_changed_cpu_passes: u64,
    finish_actor_same_thread_endpoint_cpu_invalid_passes: u64,
    finish_actor_sender_cpu_mask: u64,
    finish_actor_receiver_cpu_mask: u64,
    finish_actor_sender_cpu_id_overflow_passes: u64,
    finish_actor_receiver_cpu_id_overflow_passes: u64,
    finish_return_same_thread_passes: u64,
    finish_return_cross_thread_passes: u64,
}

#[derive(Debug, Default)]
struct CompletionTimingDiagnostics {
    total_passes: u64,
    interval_passes: u64,
    timing: CompletionPassTiming,
}

fn duration_ms(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

impl CompletionPassTiming {
    fn add_output(&mut self, timing: OutputPublishTiming) {
        self.output_batches += 1;
        self.output_publish_wall += timing.wall;
        self.output_admission += timing.admission;
        self.output_reserve += timing.reserve;
        self.output_reserve_waited_batches += u64::from(timing.reserve_waited);
        self.output_predecessor_busy += timing.predecessor_busy;
        self.output_receiver_wake += timing.receiver_wake;
        self.output_gate_wait += timing.gate_wait;
        self.output_route_wall += timing.route_wall;
        self.output_dispatcher_residual += timing.dispatcher_residual;
        self.output_ack_wake += timing.ack_wake;
        self.output_residual_ms += timing.residual_ms;
        self.output_signals += timing.signals;
        self.uninstrumented_output_signals += timing.uninstrumented_signals;
        self.terminal_signals += timing.terminals;
        self.route_found += timing.route_found;
        self.route_missing += timing.route_missing;
        self.delivered += timing.delivered;
        self.full += timing.full;
        self.closed += timing.closed;
        self.terminal_removals += timing.terminal_removals;
        if let Some(cpu) = timing.route_thread_cpu {
            self.route_cpu_valid_batches += 1;
            self.output_route_cpu_valid_wall += timing.route_wall;
            self.output_route_thread_cpu += cpu;
            self.output_route_wall_minus_thread_cpu_ms +=
                (timing.route_wall.as_secs_f64() - cpu.as_secs_f64()) * 1_000.0;
        } else {
            self.route_cpu_invalid_batches += 1;
        }
    }

    fn record_finish(&mut self, timing: BoundaryFinishTiming) {
        self.completion_boundary = timing.completion_boundary;
        self.finish_admission = timing.admission;
        self.finish_reserve = timing.reserve;
        self.finish_reserve_waited_passes = u64::from(timing.reserve_waited);
        self.finish_send_sync = timing.send_sync;
        self.finish_actor_wake = timing.actor_wake;
        if timing.actor_same_thread {
            self.finish_actor_same_thread_passes = 1;
            self.finish_actor_same_thread_send_sync = timing.send_sync;
            self.finish_actor_same_thread_wake = timing.actor_wake;
            if let Some(cpu_time) = timing.actor_same_thread_cpu {
                self.finish_actor_same_thread_cpu_valid_passes = 1;
                self.finish_actor_same_thread_cpu_valid_wake = timing.actor_wake;
                self.finish_actor_same_thread_cpu_valid_probe_wall = timing.actor_probe_wall;
                self.finish_actor_same_thread_cpu_time = cpu_time;
                self.finish_actor_same_thread_probe_wall_minus_cpu_ms =
                    duration_ms(timing.actor_probe_wall) - duration_ms(cpu_time);
                self.finish_actor_same_thread_probe_edge_ms =
                    duration_ms(timing.actor_probe_wall) - duration_ms(timing.actor_wake);
            } else {
                self.finish_actor_same_thread_cpu_invalid_passes = 1;
            }
            match (timing.actor_sender_cpu_id, timing.actor_receiver_cpu_id) {
                (Some(sender), Some(receiver)) if sender == receiver => {
                    self.finish_actor_same_thread_endpoint_same_cpu_passes = 1;
                }
                (Some(_), Some(_)) => {
                    self.finish_actor_same_thread_endpoint_changed_cpu_passes = 1;
                }
                _ => {
                    self.finish_actor_same_thread_endpoint_cpu_invalid_passes = 1;
                }
            }
        } else {
            self.finish_actor_cross_thread_passes = 1;
            self.finish_actor_cross_thread_send_sync = timing.send_sync;
            self.finish_actor_cross_thread_wake = timing.actor_wake;
        }
        record_cpu_id(
            timing.actor_sender_cpu_id,
            &mut self.finish_actor_sender_cpu_mask,
            &mut self.finish_actor_sender_cpu_id_overflow_passes,
        );
        record_cpu_id(
            timing.actor_receiver_cpu_id,
            &mut self.finish_actor_receiver_cpu_mask,
            &mut self.finish_actor_receiver_cpu_id_overflow_passes,
        );
        self.finish_return_wake = timing.return_wake;
        if timing.return_same_thread {
            self.finish_return_same_thread_passes = 1;
            self.finish_return_same_thread_wake = timing.return_wake;
        } else {
            self.finish_return_cross_thread_passes = 1;
            self.finish_return_cross_thread_wake = timing.return_wake;
        }
    }
}

impl CompletionTimingDiagnostics {
    fn record(&mut self, timing: CompletionPassTiming) {
        self.total_passes += 1;
        self.interval_passes += 1;
        self.timing.add_assign(timing);
        if self.interval_passes < COMPLETION_TIMING_LOG_INTERVAL_PASSES {
            return;
        }

        let timing = &self.timing;
        tracing::info!(
            target: "dynamo_mocker::completion_timing",
            event = "mocker_completion_timing",
            total_passes = self.total_passes,
            interval_passes = self.interval_passes,
            completion_boundary_total_ms = duration_ms(timing.completion_boundary),
            event_admission_total_ms = duration_ms(timing.event.admission),
            event_reserve_total_ms = duration_ms(timing.event.reserve),
            event_admission_nonreserve_total_ms = duration_ms(
                timing.event.admission.saturating_sub(timing.event.reserve)
            ),
            event_predecessor_busy_total_ms = duration_ms(timing.event.predecessor_busy),
            event_receiver_wake_total_ms = duration_ms(timing.event.receiver_wake),
            body_wall_total_ms = duration_ms(timing.body_wall),
            kv_publish_total_ms = duration_ms(timing.kv_publish),
            fpm_publish_total_ms = duration_ms(timing.fpm_publish),
            output_convert_total_ms = duration_ms(timing.output_convert),
            output_publish_wall_total_ms = duration_ms(timing.output_publish_wall),
            output_admission_total_ms = duration_ms(timing.output_admission),
            output_reserve_total_ms = duration_ms(timing.output_reserve),
            output_admission_nonreserve_total_ms = duration_ms(
                timing.output_admission.saturating_sub(timing.output_reserve)
            ),
            output_predecessor_busy_total_ms = duration_ms(timing.output_predecessor_busy),
            output_receiver_wake_total_ms = duration_ms(timing.output_receiver_wake),
            output_gate_wait_total_ms = duration_ms(timing.output_gate_wait),
            output_route_wall_total_ms = duration_ms(timing.output_route_wall),
            output_route_cpu_valid_wall_total_ms = duration_ms(timing.output_route_cpu_valid_wall),
            output_route_thread_cpu_total_ms = duration_ms(timing.output_route_thread_cpu),
            output_route_wall_minus_thread_cpu_total_ms = timing.output_route_wall_minus_thread_cpu_ms,
            output_dispatcher_residual_total_ms = duration_ms(timing.output_dispatcher_residual),
            output_ack_wake_total_ms = duration_ms(timing.output_ack_wake),
            output_residual_total_ms = timing.output_residual_ms,
            lifecycle_convert_total_ms = duration_ms(timing.lifecycle_convert),
            failure_cleanup_total_ms = duration_ms(timing.failure_cleanup),
            lifecycle_publish_total_ms = duration_ms(timing.lifecycle_publish),
            metrics_publish_total_ms = duration_ms(timing.metrics_publish),
            before_finish_total_ms = duration_ms(timing.before_finish),
            body_residual_total_ms = timing.body_residual_ms,
            finish_admission_total_ms = duration_ms(timing.finish_admission),
            finish_reserve_total_ms = duration_ms(timing.finish_reserve),
            finish_admission_nonreserve_total_ms = duration_ms(
                timing.finish_admission.saturating_sub(timing.finish_reserve)
            ),
            finish_send_sync_total_ms = duration_ms(timing.finish_send_sync),
            finish_actor_same_thread_send_sync_total_ms = duration_ms(
                timing.finish_actor_same_thread_send_sync
            ),
            finish_actor_cross_thread_send_sync_total_ms = duration_ms(
                timing.finish_actor_cross_thread_send_sync
            ),
            finish_send_sync_thread_identity_residual_total_ms = duration_ms(
                timing.finish_send_sync
            ) - duration_ms(timing.finish_actor_same_thread_send_sync)
                - duration_ms(timing.finish_actor_cross_thread_send_sync),
            finish_actor_wake_total_ms = duration_ms(timing.finish_actor_wake),
            finish_actor_same_thread_wake_total_ms = duration_ms(
                timing.finish_actor_same_thread_wake
            ),
            finish_actor_cross_thread_wake_total_ms = duration_ms(
                timing.finish_actor_cross_thread_wake
            ),
            finish_actor_same_thread_cpu_valid_wake_total_ms = duration_ms(
                timing.finish_actor_same_thread_cpu_valid_wake
            ),
            finish_actor_same_thread_cpu_valid_probe_wall_total_ms = duration_ms(
                timing.finish_actor_same_thread_cpu_valid_probe_wall
            ),
            finish_actor_same_thread_cpu_time_total_ms = duration_ms(
                timing.finish_actor_same_thread_cpu_time
            ),
            finish_actor_same_thread_probe_wall_minus_cpu_total_ms = timing
                .finish_actor_same_thread_probe_wall_minus_cpu_ms,
            finish_actor_same_thread_probe_edge_total_ms = timing
                .finish_actor_same_thread_probe_edge_ms,
            finish_return_wake_total_ms = duration_ms(timing.finish_return_wake),
            finish_return_same_thread_wake_total_ms = duration_ms(
                timing.finish_return_same_thread_wake
            ),
            finish_return_cross_thread_wake_total_ms = duration_ms(
                timing.finish_return_cross_thread_wake
            ),
            boundary_identity_residual_total_ms = timing.boundary_identity_residual_ms,
            rank_batches = timing.rank_batches,
            event_reserve_waited_passes = timing.event_reserve_waited_passes,
            output_batches = timing.output_batches,
            output_reserve_waited_batches = timing.output_reserve_waited_batches,
            output_signals = timing.output_signals,
            uninstrumented_output_signals = timing.uninstrumented_output_signals,
            terminal_signals = timing.terminal_signals,
            route_found = timing.route_found,
            route_missing = timing.route_missing,
            delivered = timing.delivered,
            full = timing.full,
            closed = timing.closed,
            terminal_removals = timing.terminal_removals,
            route_cpu_valid_batches = timing.route_cpu_valid_batches,
            route_cpu_invalid_batches = timing.route_cpu_invalid_batches,
            cleanup_commands = timing.cleanup_commands,
            completion_kv_events = timing.completion_kv_events,
            deferred_kv_events = timing.deferred_kv_events,
            lifecycle_events = timing.lifecycle_events,
            finish_reserve_waited_passes = timing.finish_reserve_waited_passes,
            finish_actor_same_thread_passes = timing.finish_actor_same_thread_passes,
            finish_actor_cross_thread_passes = timing.finish_actor_cross_thread_passes,
            finish_actor_thread_count_identity_residual = self.interval_passes as i64
                - timing.finish_actor_same_thread_passes as i64
                - timing.finish_actor_cross_thread_passes as i64,
            finish_actor_thread_wake_identity_residual_total_ms = duration_ms(
                timing.finish_actor_wake
            ) - duration_ms(timing.finish_actor_same_thread_wake)
                - duration_ms(timing.finish_actor_cross_thread_wake),
            finish_actor_same_thread_cpu_valid_passes = timing
                .finish_actor_same_thread_cpu_valid_passes,
            finish_actor_same_thread_cpu_invalid_passes = timing
                .finish_actor_same_thread_cpu_invalid_passes,
            finish_actor_same_thread_cpu_count_identity_residual = timing
                .finish_actor_same_thread_passes as i64
                - timing.finish_actor_same_thread_cpu_valid_passes as i64
                - timing.finish_actor_same_thread_cpu_invalid_passes as i64,
            finish_actor_same_thread_probe_cpu_identity_residual_total_ms = duration_ms(
                timing.finish_actor_same_thread_cpu_valid_probe_wall
            ) - duration_ms(timing.finish_actor_same_thread_cpu_time)
                - timing.finish_actor_same_thread_probe_wall_minus_cpu_ms,
            finish_actor_same_thread_probe_wake_identity_residual_total_ms = duration_ms(
                timing.finish_actor_same_thread_cpu_valid_probe_wall
            ) - duration_ms(timing.finish_actor_same_thread_cpu_valid_wake)
                - timing.finish_actor_same_thread_probe_edge_ms,
            finish_actor_same_thread_endpoint_same_cpu_passes = timing
                .finish_actor_same_thread_endpoint_same_cpu_passes,
            finish_actor_same_thread_endpoint_changed_cpu_passes = timing
                .finish_actor_same_thread_endpoint_changed_cpu_passes,
            finish_actor_same_thread_endpoint_cpu_invalid_passes = timing
                .finish_actor_same_thread_endpoint_cpu_invalid_passes,
            finish_actor_same_thread_endpoint_cpu_count_identity_residual = timing
                .finish_actor_same_thread_passes as i64
                - timing.finish_actor_same_thread_endpoint_same_cpu_passes as i64
                - timing.finish_actor_same_thread_endpoint_changed_cpu_passes as i64
                - timing.finish_actor_same_thread_endpoint_cpu_invalid_passes as i64,
            finish_actor_sender_cpu_mask = timing.finish_actor_sender_cpu_mask,
            finish_actor_receiver_cpu_mask = timing.finish_actor_receiver_cpu_mask,
            finish_actor_sender_cpu_id_overflow_passes = timing
                .finish_actor_sender_cpu_id_overflow_passes,
            finish_actor_receiver_cpu_id_overflow_passes = timing
                .finish_actor_receiver_cpu_id_overflow_passes,
            finish_return_same_thread_passes = timing.finish_return_same_thread_passes,
            finish_return_cross_thread_passes = timing.finish_return_cross_thread_passes,
            finish_return_thread_count_identity_residual = self.interval_passes as i64
                - timing.finish_return_same_thread_passes as i64
                - timing.finish_return_cross_thread_passes as i64,
            finish_return_thread_wake_identity_residual_total_ms = duration_ms(
                timing.finish_return_wake
            ) - duration_ms(timing.finish_return_same_thread_wake)
                - duration_ms(timing.finish_return_cross_thread_wake),
            "mocker completion timing interval"
        );

        let total_passes = self.total_passes;
        *self = Self {
            total_passes,
            ..Self::default()
        };
    }
}

impl CompletionPassTiming {
    fn add_assign(&mut self, other: Self) {
        self.completion_boundary += other.completion_boundary;
        self.event.admission += other.event.admission;
        self.event.reserve += other.event.reserve;
        self.event.reserve_waited |= other.event.reserve_waited;
        self.event.predecessor_busy += other.event.predecessor_busy;
        self.event.receiver_wake += other.event.receiver_wake;
        self.body_wall += other.body_wall;
        self.kv_publish += other.kv_publish;
        self.fpm_publish += other.fpm_publish;
        self.output_convert += other.output_convert;
        self.output_publish_wall += other.output_publish_wall;
        self.output_admission += other.output_admission;
        self.output_reserve += other.output_reserve;
        self.output_predecessor_busy += other.output_predecessor_busy;
        self.output_receiver_wake += other.output_receiver_wake;
        self.output_gate_wait += other.output_gate_wait;
        self.output_route_wall += other.output_route_wall;
        self.output_route_cpu_valid_wall += other.output_route_cpu_valid_wall;
        self.output_route_thread_cpu += other.output_route_thread_cpu;
        self.output_route_wall_minus_thread_cpu_ms += other.output_route_wall_minus_thread_cpu_ms;
        self.output_dispatcher_residual += other.output_dispatcher_residual;
        self.output_ack_wake += other.output_ack_wake;
        self.output_residual_ms += other.output_residual_ms;
        self.lifecycle_convert += other.lifecycle_convert;
        self.failure_cleanup += other.failure_cleanup;
        self.lifecycle_publish += other.lifecycle_publish;
        self.metrics_publish += other.metrics_publish;
        self.before_finish += other.before_finish;
        self.body_residual_ms += other.body_residual_ms;
        self.finish_admission += other.finish_admission;
        self.finish_reserve += other.finish_reserve;
        self.finish_send_sync += other.finish_send_sync;
        self.finish_actor_same_thread_send_sync += other.finish_actor_same_thread_send_sync;
        self.finish_actor_cross_thread_send_sync += other.finish_actor_cross_thread_send_sync;
        self.finish_actor_wake += other.finish_actor_wake;
        self.finish_actor_same_thread_wake += other.finish_actor_same_thread_wake;
        self.finish_actor_cross_thread_wake += other.finish_actor_cross_thread_wake;
        self.finish_actor_same_thread_cpu_valid_wake +=
            other.finish_actor_same_thread_cpu_valid_wake;
        self.finish_actor_same_thread_cpu_valid_probe_wall +=
            other.finish_actor_same_thread_cpu_valid_probe_wall;
        self.finish_actor_same_thread_cpu_time += other.finish_actor_same_thread_cpu_time;
        self.finish_actor_same_thread_probe_wall_minus_cpu_ms +=
            other.finish_actor_same_thread_probe_wall_minus_cpu_ms;
        self.finish_actor_same_thread_probe_edge_ms += other.finish_actor_same_thread_probe_edge_ms;
        self.finish_return_wake += other.finish_return_wake;
        self.finish_return_same_thread_wake += other.finish_return_same_thread_wake;
        self.finish_return_cross_thread_wake += other.finish_return_cross_thread_wake;
        self.boundary_identity_residual_ms += other.boundary_identity_residual_ms;
        self.rank_batches += other.rank_batches;
        self.event_reserve_waited_passes += other.event_reserve_waited_passes;
        self.output_batches += other.output_batches;
        self.output_reserve_waited_batches += other.output_reserve_waited_batches;
        self.output_signals += other.output_signals;
        self.uninstrumented_output_signals += other.uninstrumented_output_signals;
        self.terminal_signals += other.terminal_signals;
        self.route_found += other.route_found;
        self.route_missing += other.route_missing;
        self.delivered += other.delivered;
        self.full += other.full;
        self.closed += other.closed;
        self.terminal_removals += other.terminal_removals;
        self.route_cpu_valid_batches += other.route_cpu_valid_batches;
        self.route_cpu_invalid_batches += other.route_cpu_invalid_batches;
        self.cleanup_commands += other.cleanup_commands;
        self.completion_kv_events += other.completion_kv_events;
        self.deferred_kv_events += other.deferred_kv_events;
        self.lifecycle_events += other.lifecycle_events;
        self.finish_reserve_waited_passes += other.finish_reserve_waited_passes;
        self.finish_actor_same_thread_passes += other.finish_actor_same_thread_passes;
        self.finish_actor_cross_thread_passes += other.finish_actor_cross_thread_passes;
        self.finish_actor_same_thread_cpu_valid_passes +=
            other.finish_actor_same_thread_cpu_valid_passes;
        self.finish_actor_same_thread_cpu_invalid_passes +=
            other.finish_actor_same_thread_cpu_invalid_passes;
        self.finish_actor_same_thread_endpoint_same_cpu_passes +=
            other.finish_actor_same_thread_endpoint_same_cpu_passes;
        self.finish_actor_same_thread_endpoint_changed_cpu_passes +=
            other.finish_actor_same_thread_endpoint_changed_cpu_passes;
        self.finish_actor_same_thread_endpoint_cpu_invalid_passes +=
            other.finish_actor_same_thread_endpoint_cpu_invalid_passes;
        self.finish_actor_sender_cpu_mask |= other.finish_actor_sender_cpu_mask;
        self.finish_actor_receiver_cpu_mask |= other.finish_actor_receiver_cpu_mask;
        self.finish_actor_sender_cpu_id_overflow_passes +=
            other.finish_actor_sender_cpu_id_overflow_passes;
        self.finish_actor_receiver_cpu_id_overflow_passes +=
            other.finish_actor_receiver_cpu_id_overflow_passes;
        self.finish_return_same_thread_passes += other.finish_return_same_thread_passes;
        self.finish_return_cross_thread_passes += other.finish_return_cross_thread_passes;
    }
}

fn record_cpu_id(cpu_id: Option<u32>, mask: &mut u64, overflow_passes: &mut u64) {
    let Some(cpu_id) = cpu_id else {
        return;
    };
    if let Some(bit) = 1_u64.checked_shl(cpu_id) {
        *mask |= bit;
    } else {
        *overflow_passes += 1;
    }
}

#[derive(Default)]
pub(super) struct DeferredCommandPublication {
    pub(super) kv: Vec<KvEvent>,
    pub(super) metrics: Option<Metrics>,
}

pub(super) async fn run_effect_dispatcher(
    mut events: mpsc::Receiver<GroupedLiveEvent>,
    ranks: Vec<RankDispatch>,
    compatibility: Arc<CompatibilityState>,
    pending: Arc<Mutex<HashMap<u64, PendingCommand>>>,
    cancel: CancellationToken,
    completion_tracker: CompletionBoundaryTracker,
) -> Result<()> {
    let mut deferred_commands = (0..ranks.len())
        .map(|_| DeferredCommandPublication::default())
        .collect::<Vec<_>>();
    let mut timing_diagnostics = CompletionTimingDiagnostics::default();
    let mut last_handler_finished_at = None;
    loop {
        let event = tokio::select! {
            biased;
            _ = cancel.cancelled() => return Ok(()),
            event = events.recv() => event,
        };
        let Some(event) = event else {
            return Ok(());
        };
        let dequeued_at = tokio::time::Instant::now();
        match event {
            GroupedLiveEvent::CommandApplied {
                command_id,
                pass_in_flight,
                is_request_cancellation,
                effects,
                ..
            } => {
                dispatch_command_effects(
                    command_id,
                    effects,
                    pass_in_flight,
                    is_request_cancellation,
                    &ranks,
                    &compatibility,
                    &pending,
                    &mut deferred_commands,
                )
                .await?;
            }
            GroupedLiveEvent::PassStarted(started) => {
                for rank in started.by_rank {
                    let dispatch = rank_dispatch(&ranks, rank.dp_rank)?;
                    dispatch.publish_admissions(rank.effects.admissions).await?;
                    dispatch.publish_kv(rank.effects.kv_events);
                }
            }
            GroupedLiveEvent::PassCompleted {
                completed,
                boundary,
                completion_started_at,
                event_enqueued_at,
                event_reserve,
                event_reserve_waited,
            } => {
                let _completion_guard = completion_tracker.enter();
                let event_timing = CompletionEventTiming {
                    admission: event_enqueued_at.saturating_duration_since(completion_started_at),
                    reserve: event_reserve,
                    reserve_waited: event_reserve_waited,
                    predecessor_busy: last_handler_finished_at
                        .map(|finished_at: tokio::time::Instant| {
                            finished_at.saturating_duration_since(event_enqueued_at)
                        })
                        .unwrap_or_default(),
                    receiver_wake: dequeued_at.saturating_duration_since(
                        last_handler_finished_at
                            .map(|finished_at| finished_at.max(event_enqueued_at))
                            .unwrap_or(event_enqueued_at),
                    ),
                };
                if let Some(timing) = dispatch_pass_completion(
                    completed,
                    boundary,
                    &ranks,
                    &compatibility,
                    &mut deferred_commands,
                    &cancel,
                    &completion_tracker,
                    event_timing,
                )
                .await?
                {
                    timing_diagnostics.record(timing);
                }
                ensure!(
                    deferred_commands
                        .iter()
                        .all(|deferred| deferred.kv.is_empty() && deferred.metrics.is_none()),
                    "grouped pass completion omitted deferred command effects for a rank"
                );
            }
        }
        last_handler_finished_at = Some(tokio::time::Instant::now());
    }
}

// The diagnostic event timing is deliberately carried beside the existing
// completion context so the production dispatch sequence stays unchanged.
#[allow(clippy::too_many_arguments)]
async fn dispatch_pass_completion(
    completed: EnginePassCompleted<PassCompletionEffects>,
    boundary: GroupedPassBoundary,
    ranks: &[RankDispatch],
    compatibility: &CompatibilityState,
    deferred_commands: &mut [DeferredCommandPublication],
    cancel: &CancellationToken,
    completion_tracker: &CompletionBoundaryTracker,
    event_timing: CompletionEventTiming,
) -> Result<Option<CompletionPassTiming>> {
    let body_started_at = tokio::time::Instant::now();
    let mut timing = CompletionPassTiming {
        event: event_timing,
        event_reserve_waited_passes: u64::from(event_timing.reserve_waited),
        ..CompletionPassTiming::default()
    };
    let dispatch_result = async {
        let mut publications = Vec::with_capacity(completed.effects.by_rank.len());
        let mut delivery_failures = Vec::new();
        for rank in completed.effects.by_rank {
            timing.rank_batches += 1;
            let dispatch = rank_dispatch(ranks, rank.dp_rank)?;
            let effects = rank.effects;
            let router_timing = publish_pass_router_effects(
                dispatch,
                effects.kv_events,
                &mut deferred_commands
                    .get_mut(rank.dp_rank as usize)
                    .context("deferred command effect rank is out of range")?
                    .kv,
                effects.forward_pass_metrics,
            );
            timing.kv_publish += router_timing.kv_publish;
            timing.fpm_publish += router_timing.fpm_publish;
            timing.completion_kv_events += router_timing.completion_kv_events;
            timing.deferred_kv_events += router_timing.deferred_kv_events;
            let output_convert_started_at = tokio::time::Instant::now();
            let outputs = effects
                .outputs
                .into_iter()
                .map(|output| compatibility.output_signal(output))
                .collect::<Vec<_>>();
            timing.output_convert += output_convert_started_at.elapsed();
            match dispatch.publish_outputs(outputs).await? {
                OutputPublication::Delivered {
                    failed_requests,
                    timing: output_timing,
                } => {
                    if output_timing.signals > 0 {
                        timing.add_output(output_timing);
                    }
                    delivery_failures.extend(
                        failed_requests
                            .into_iter()
                            .map(|request_id| (rank.dp_rank, request_id)),
                    );
                }
                OutputPublication::Cancelled => return Ok(CompletionDispatch::Cancelled),
            }
            let lifecycle_convert_started_at = tokio::time::Instant::now();
            let lifecycle = effects
                .lifecycle_events
                .into_iter()
                .map(|event| compatibility.lifecycle_event(event))
                .collect::<Result<Vec<_>>>()?;
            timing.lifecycle_convert += lifecycle_convert_started_at.elapsed();
            timing.lifecycle_events += lifecycle.len() as u64;
            publications.push(PendingRankPublication {
                dp_rank: rank.dp_rank,
                lifecycle,
                // A command applied mid-pass snapshots state before the
                // grouped boundary has completed. Consume that deferred
                // snapshot, but always publish the authoritative metrics
                // refreshed by `complete_pass`/`complete_idle_group_pass`.
                metrics: completion_metrics(
                    &mut deferred_commands[rank.dp_rank as usize].metrics,
                    effects.metrics,
                ),
            });
        }

        let failure_cleanup_started_at = tokio::time::Instant::now();
        for (dp_rank, request_id) in delivery_failures {
            timing.cleanup_commands += 1;
            let command_result = boundary
                .apply_command(EngineSchedulerCommand::new(
                    dp_rank,
                    Command::CancelRequest {
                        request_id,
                        discard_pending_output: true,
                    },
                ))
                .await;
            // The output transport no longer owns this request regardless of
            // whether the engine had already retired it.
            compatibility.apply_cleanup(Cleanup::Request(request_id));
            let effects = command_result?;
            merge_boundary_command_effects(effects, ranks, compatibility, &mut publications)?;
        }
        timing.failure_cleanup += failure_cleanup_started_at.elapsed();

        for publication in publications {
            let dispatch = rank_dispatch(ranks, publication.dp_rank)?;
            let lifecycle_publish_started_at = tokio::time::Instant::now();
            dispatch.publish_lifecycle(publication.lifecycle).await;
            timing.lifecycle_publish += lifecycle_publish_started_at.elapsed();
            let metrics_publish_started_at = tokio::time::Instant::now();
            dispatch.publish_metrics(publication.metrics);
            timing.metrics_publish += metrics_publish_started_at.elapsed();
        }
        Ok(CompletionDispatch::Completed)
    }
    .await;

    // Always release the actor, including sink/conversion error paths. The
    // primary publication error remains the one returned to the supervisor.
    // A cancellation observed by the ordered output lane means the actor is
    // already shutting down, so there is no boundary left to release.
    let finish_result = if matches!(&dispatch_result, Ok(CompletionDispatch::Cancelled)) {
        Ok(None)
    } else {
        let before_finish_started_at = tokio::time::Instant::now();
        completion_tracker.before_finish().await;
        timing.before_finish += before_finish_started_at.elapsed();
        timing.body_wall = body_started_at.elapsed();
        let body_accounted = timing
            .kv_publish
            .saturating_add(timing.fpm_publish)
            .saturating_add(timing.output_convert)
            .saturating_add(timing.output_publish_wall)
            .saturating_add(timing.lifecycle_convert)
            .saturating_add(timing.failure_cleanup)
            .saturating_add(timing.lifecycle_publish)
            .saturating_add(timing.metrics_publish)
            .saturating_add(timing.before_finish);
        timing.body_residual_ms =
            (timing.body_wall.as_secs_f64() - body_accounted.as_secs_f64()) * 1_000.0;
        finish_boundary_or_cancel(boundary.finish(), cancel).await
    };
    match dispatch_result {
        Err(error) => Err(error),
        Ok(CompletionDispatch::Cancelled) => Ok(None),
        Ok(CompletionDispatch::Completed) => {
            let Some(finish) = finish_result? else {
                return Ok(None);
            };
            timing.record_finish(finish);
            let boundary_accounted = timing
                .event
                .admission
                .saturating_add(timing.event.predecessor_busy)
                .saturating_add(timing.event.receiver_wake)
                .saturating_add(timing.body_wall)
                .saturating_add(timing.finish_admission)
                .saturating_add(timing.finish_actor_wake);
            timing.boundary_identity_residual_ms = (timing.completion_boundary.as_secs_f64()
                - boundary_accounted.as_secs_f64())
                * 1_000.0;
            Ok(Some(timing))
        }
    }
}

async fn finish_boundary_or_cancel<F>(
    finish: F,
    cancel: &CancellationToken,
) -> Result<Option<BoundaryFinishTiming>>
where
    F: std::future::Future<Output = Result<BoundaryFinishTiming>>,
{
    tokio::pin!(finish);
    tokio::select! {
        biased;
        result = &mut finish => result.map(Some),
        _ = cancel.cancelled() => Ok(None),
    }
}

pub(super) fn completion_metrics(deferred: &mut Option<Metrics>, completed: Metrics) -> Metrics {
    deferred.take();
    completed
}

fn merge_boundary_command_effects(
    effects: EngineEffects<CommandEffects>,
    ranks: &[RankDispatch],
    compatibility: &CompatibilityState,
    publications: &mut [PendingRankPublication],
) -> Result<()> {
    ensure!(
        effects.by_rank.len() == 1,
        "output-delivery cleanup returned {} rank effect batches",
        effects.by_rank.len()
    );
    let rank = effects
        .by_rank
        .into_iter()
        .next()
        .expect("one rank effect was validated");
    let dispatch = rank_dispatch(ranks, rank.dp_rank)?;
    let effects = rank.effects;
    dispatch.publish_kv(effects.kv_events);
    let lifecycle = effects
        .lifecycle_events
        .into_iter()
        .map(|event| compatibility.lifecycle_event(event))
        .collect::<Result<Vec<_>>>()?;
    let publication = publications
        .iter_mut()
        .find(|publication| publication.dp_rank == rank.dp_rank)
        .context("output-delivery cleanup referenced a rank absent from pass completion")?;
    publication.lifecycle.extend(lifecycle);
    publication.metrics = effects.metrics;
    Ok(())
}

pub(super) fn publish_pass_router_effects(
    dispatch: &RankDispatch,
    completion_kv: Vec<KvEvent>,
    deferred_command_kv: &mut Vec<KvEvent>,
    fpm: ForwardPassMetrics,
) -> RouterPublishTiming {
    let completion_kv_events = completion_kv.len() as u64;
    let deferred_kv_events = deferred_command_kv.len() as u64;
    let kv_publish_started_at = tokio::time::Instant::now();
    dispatch.publish_kv(completion_kv);
    dispatch.publish_kv(std::mem::take(deferred_command_kv));
    let kv_publish = kv_publish_started_at.elapsed();
    let fpm_publish_started_at = tokio::time::Instant::now();
    dispatch.publish_fpm(fpm);
    RouterPublishTiming {
        kv_publish,
        fpm_publish: fpm_publish_started_at.elapsed(),
        completion_kv_events,
        deferred_kv_events,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn dispatch_command_effects(
    command_id: u64,
    effects: EngineEffects<CommandEffects>,
    pass_in_flight: bool,
    is_request_cancellation: bool,
    ranks: &[RankDispatch],
    compatibility: &CompatibilityState,
    pending: &Mutex<HashMap<u64, PendingCommand>>,
    deferred_commands: &mut [DeferredCommandPublication],
) -> Result<()> {
    ensure!(
        effects.by_rank.len() == 1,
        "native command {command_id} returned {} rank effect batches",
        effects.by_rank.len()
    );
    let rank = effects
        .by_rank
        .into_iter()
        .next()
        .expect("one rank effect was validated");
    let dispatch = rank_dispatch(ranks, rank.dp_rank)?;
    let mut effects = rank.effects;
    let immediate_empty_metrics = (pass_in_flight
        && is_request_cancellation
        && effects.result == CommandResult::Applied
        && effects.metrics.running_requests == 0
        && effects.metrics.waiting_requests == 0)
        .then(|| effects.metrics.clone());
    if pass_in_flight {
        ensure!(
            effects.lifecycle_events.is_empty(),
            "mid-pass native command {command_id} produced lifecycle effects"
        );
        let deferred = deferred_commands
            .get_mut(rank.dp_rank as usize)
            .context("deferred command effect rank is out of range")?;
        deferred.kv.append(&mut effects.kv_events);
        deferred.metrics = Some(effects.metrics.clone());
    } else {
        dispatch.publish_kv(std::mem::take(&mut effects.kv_events));
    }
    for request_id in effects.retired_requests.drain(..) {
        compatibility.apply_cleanup(Cleanup::Request(request_id));
    }
    let lifecycle = effects
        .lifecycle_events
        .into_iter()
        .map(|event| compatibility.lifecycle_event(event))
        .collect::<Result<Vec<_>>>()?;
    let result = scheduler_command_result(effects.result, effects.suppressed_pending_output);
    let pending = pending.lock().remove(&command_id);
    if let Some(pending) = pending {
        // Idle command KV effects are visible before acknowledgement. Mid-pass
        // effects are acknowledged immediately but stay hidden until the
        // current grouped pass reaches its completion boundary.
        if let Some(reply) = pending.reply {
            let _ = reply.send(Ok(SchedulerCommandEffects {
                result,
                lifecycle_events: Vec::new(),
                kv_events: Vec::new(),
            }));
        }
        if !pass_in_flight {
            dispatch.publish_lifecycle(lifecycle).await;
            dispatch.publish_metrics(effects.metrics);
        }
        if effects.suppressed_pending_output {
            for cleanup in pending.on_suppressed_output {
                compatibility.apply_cleanup(cleanup);
            }
        }
        for cleanup in pending.on_success {
            compatibility.apply_cleanup(cleanup);
        }
    } else if !pass_in_flight {
        dispatch.publish_lifecycle(lifecycle).await;
        dispatch.publish_metrics(effects.metrics);
    }
    if let Some(metrics) = immediate_empty_metrics {
        // Match the historical live boundary: a cancellation that removes
        // the last scheduler-owned request updates occupancy immediately,
        // even though the modeled pass still owns its completion boundary.
        // KV and lifecycle effects remain deferred with that pass.
        dispatch.publish_metrics(metrics);
    }
    Ok(())
}

fn scheduler_command_result(
    result: CommandResult,
    suppressed_pending_output: bool,
) -> SchedulerCommandResult {
    match result {
        CommandResult::Submitted(request_id) => SchedulerCommandResult::Submitted(request_id),
        CommandResult::DestinationAccepted { request_id } => {
            SchedulerCommandResult::DestinationAccepted { request_id }
        }
        CommandResult::Applied => SchedulerCommandResult::Applied,
        // The native request may already have retired into the in-flight pass
        // while its output is still pending. Suppressing that retained output
        // is observable cancellation work at Dynamo's compatibility boundary.
        CommandResult::Noop if suppressed_pending_output => SchedulerCommandResult::Applied,
        CommandResult::Noop => SchedulerCommandResult::Noop,
    }
}

fn rank_dispatch(ranks: &[RankDispatch], dp_rank: u32) -> Result<&RankDispatch> {
    ranks.get(dp_rank as usize).ok_or_else(|| {
        anyhow!(
            "grouped live effect referenced DP rank {dp_rank}, but only {} ranks exist",
            ranks.len()
        )
    })
}

impl RankDispatch {
    async fn publish_admissions(&self, admissions: Vec<Admission>) -> Result<()> {
        let Some(sender) = self.event_tx.as_ref() else {
            return Ok(());
        };
        let admissions = admissions
            .into_iter()
            .map(|admission| AdmissionEvent {
                uuid: admission.request_id,
                reused_input_tokens: admission.reused_input_tokens,
            })
            .collect::<Vec<_>>();
        match sender.send_admissions(&admissions).await {
            Ok(()) | Err(SchedulerEventSendError::Cancelled) => Ok(()),
            Err(SchedulerEventSendError::OrderedLaneClosed) => {
                bail!("grouped live ordered admission lane is closed")
            }
            Err(SchedulerEventSendError::OutputClosed(_)) => {
                bail!("grouped live admission unexpectedly used an output-only lane")
            }
        }
    }

    /// Publish output and return requests whose output-only consumer closed.
    async fn publish_outputs(&self, outputs: Vec<OutputSignal>) -> Result<OutputPublication> {
        let output_signals = outputs.len() as u64;
        let Some(sender) = self.event_tx.as_ref() else {
            return Ok(OutputPublication::Delivered {
                failed_requests: Vec::new(),
                timing: OutputPublishTiming {
                    signals: output_signals,
                    uninstrumented_signals: output_signals,
                    ..OutputPublishTiming::default()
                },
            });
        };
        if outputs.is_empty() {
            return Ok(OutputPublication::Delivered {
                failed_requests: Vec::new(),
                timing: OutputPublishTiming::default(),
            });
        }
        match sender.send_outputs_timed(outputs).await {
            Ok(result) => Ok(OutputPublication::Delivered {
                failed_requests: result
                    .failed
                    .into_iter()
                    .map(|signal| signal.uuid)
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect(),
                timing: result.timing,
            }),
            Err(SchedulerEventSendError::OutputClosed(signals)) => {
                let output_signals = signals.len() as u64;
                Ok(OutputPublication::Delivered {
                    failed_requests: signals
                        .into_iter()
                        .map(|signal| signal.uuid)
                        .collect::<BTreeSet<_>>()
                        .into_iter()
                        .collect(),
                    timing: OutputPublishTiming {
                        signals: output_signals,
                        uninstrumented_signals: output_signals,
                        ..OutputPublishTiming::default()
                    },
                })
            }
            Err(SchedulerEventSendError::OrderedLaneClosed) => {
                bail!("grouped live ordered output lane is closed")
            }
            Err(SchedulerEventSendError::Cancelled) => Ok(OutputPublication::Cancelled),
        }
    }

    fn publish_kv(&self, events: Vec<KvEvent>) {
        if events.is_empty() {
            return;
        }
        let mut raw_events = Vec::with_capacity(events.len());
        for event in events {
            if event.dp_rank != self.external_dp_rank {
                tracing::warn!(
                    expected_dp_rank = self.external_dp_rank,
                    event_dp_rank = event.dp_rank,
                    "dropping native KV event with mismatched DP rank"
                );
                continue;
            }
            let (event, block_token_ids) = dynamo_kv_event(event);
            raw_events.push(RawKvEvent {
                event,
                block_token_ids,
                storage_tier: StorageTier::Device,
            });
        }
        let normal_events = raw_events
            .iter()
            .map(|event| (event.event.clone(), event.storage_tier))
            .collect();
        if let Err(error) = self
            .kv_event_publishers
            .publish_event_sink_batch_only(normal_events)
        {
            tracing::warn!(dp_rank = self.external_dp_rank, error = ?error, "failed to publish grouped native KV events");
        }
        if let Err(error) = self.kv_event_publishers.publish_raw_batch(raw_events) {
            tracing::warn!(dp_rank = self.external_dp_rank, error = ?error, "failed to publish grouped raw KV events");
        }
    }

    fn publish_fpm(&self, metrics: ForwardPassMetrics) {
        let snapshot = dynamo_forward_pass_snapshot(self.external_dp_rank, metrics);
        if let Err(error) = self.fpm_publisher.publish(snapshot) {
            tracing::warn!(dp_rank = self.external_dp_rank, error = ?error, "failed to publish grouped forward-pass metrics");
        }
    }

    async fn publish_lifecycle(&self, events: Vec<SchedulerLifecycleEvent>) {
        for event in events {
            if self.lifecycle_tx.send(event).await.is_err() {
                return;
            }
        }
    }

    pub(super) fn publish_metrics(&self, metrics: Metrics) {
        let mut metrics = MockerMetrics {
            dp_rank: metrics.dp_rank,
            active_decode_blocks: metrics.active_blocks,
            total_blocks: metrics.total_blocks,
            gpu_cache_usage_perc: metrics.cache_usage,
            running_requests: metrics.running_requests,
            waiting_requests: metrics.waiting_requests,
            vllm_preemptions_total: metrics.preemptions_total,
            sglang_cache_hit_tokens: metrics.sglang_cache_hit_tokens,
            sglang_cache_total_tokens: metrics.sglang_cache_total_tokens,
        };
        self.metrics_tx.send_modify(|current| {
            // NOTE: This is a semantic latch, not optional smoothing. SGLang's cache fields are
            // per-prefill observations, while `watch` retains only the latest value. DO NOT let a
            // decode or idle snapshot with no prefill tokens erase a meaningful observation before
            // consumers can read it. A real miss has a nonzero total and must replace the latch.
            // This latch is specific to SGLang's aggregate cache observation; do not generalize it.
            if metrics.sglang_cache_total_tokens == 0 {
                metrics.sglang_cache_hit_tokens = current.sglang_cache_hit_tokens;
                metrics.sglang_cache_total_tokens = current.sglang_cache_total_tokens;
            }
            *current = metrics;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finish_timing(
        actor_same_thread: bool,
        return_same_thread: bool,
        actor_wake_ms: u64,
        return_wake_ms: u64,
    ) -> BoundaryFinishTiming {
        BoundaryFinishTiming {
            admission: std::time::Duration::from_millis(1),
            reserve: std::time::Duration::ZERO,
            reserve_waited: false,
            send_sync: std::time::Duration::from_millis(1),
            actor_wake: std::time::Duration::from_millis(actor_wake_ms),
            actor_probe_wall: std::time::Duration::from_millis(actor_wake_ms + 1),
            actor_same_thread,
            actor_same_thread_cpu: actor_same_thread.then(|| std::time::Duration::from_millis(2)),
            actor_sender_cpu_id: Some(3),
            actor_receiver_cpu_id: Some(if actor_same_thread { 3 } else { 5 }),
            completion_boundary: std::time::Duration::from_millis(20),
            return_wake: std::time::Duration::from_millis(return_wake_ms),
            return_same_thread,
        }
    }

    #[test]
    fn finish_thread_path_totals_form_additive_identities() {
        let mut aggregate = CompletionPassTiming::default();
        let mut same_actor_cross_return = CompletionPassTiming::default();
        same_actor_cross_return.record_finish(finish_timing(true, false, 3, 5));
        aggregate.add_assign(same_actor_cross_return);
        let mut cross_actor_same_return = CompletionPassTiming::default();
        cross_actor_same_return.record_finish(finish_timing(false, true, 7, 11));
        aggregate.add_assign(cross_actor_same_return);

        assert_eq!(
            aggregate.finish_actor_same_thread_passes + aggregate.finish_actor_cross_thread_passes,
            2
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_wake + aggregate.finish_actor_cross_thread_wake,
            aggregate.finish_actor_wake
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_send_sync
                + aggregate.finish_actor_cross_thread_send_sync,
            aggregate.finish_send_sync
        );
        assert_eq!(aggregate.finish_actor_same_thread_cpu_valid_passes, 1);
        assert_eq!(aggregate.finish_actor_same_thread_cpu_invalid_passes, 0);
        assert_eq!(
            aggregate.finish_actor_same_thread_cpu_valid_wake,
            std::time::Duration::from_millis(3)
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_cpu_valid_probe_wall,
            std::time::Duration::from_millis(4)
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_cpu_time,
            std::time::Duration::from_millis(2)
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_probe_wall_minus_cpu_ms,
            2.0
        );
        assert_eq!(aggregate.finish_actor_same_thread_probe_edge_ms, 1.0);
        assert_eq!(
            aggregate.finish_actor_same_thread_endpoint_same_cpu_passes,
            1
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_endpoint_changed_cpu_passes,
            0
        );
        assert_eq!(aggregate.finish_actor_sender_cpu_mask, 1_u64 << 3);
        assert_eq!(
            aggregate.finish_actor_receiver_cpu_mask,
            (1_u64 << 3) | (1_u64 << 5)
        );
        assert_eq!(
            aggregate.finish_return_same_thread_passes
                + aggregate.finish_return_cross_thread_passes,
            2
        );
        assert_eq!(
            aggregate.finish_return_same_thread_wake + aggregate.finish_return_cross_thread_wake,
            aggregate.finish_return_wake
        );
        assert_eq!(
            aggregate.finish_send_sync,
            std::time::Duration::from_millis(2)
        );
    }

    #[test]
    fn finish_cpu_probe_classifies_changed_and_invalid_samples() {
        let mut aggregate = CompletionPassTiming::default();

        let mut changed = finish_timing(true, true, 5, 1);
        changed.actor_same_thread_cpu = Some(std::time::Duration::from_millis(3));
        changed.actor_sender_cpu_id = Some(7);
        changed.actor_receiver_cpu_id = Some(9);
        let mut changed_timing = CompletionPassTiming::default();
        changed_timing.record_finish(changed);
        aggregate.add_assign(changed_timing);

        let mut invalid = finish_timing(true, true, 6, 1);
        invalid.actor_same_thread_cpu = None;
        invalid.actor_sender_cpu_id = None;
        invalid.actor_receiver_cpu_id = Some(70);
        let mut invalid_timing = CompletionPassTiming::default();
        invalid_timing.record_finish(invalid);
        aggregate.add_assign(invalid_timing);

        assert_eq!(aggregate.finish_actor_same_thread_passes, 2);
        assert_eq!(aggregate.finish_actor_same_thread_cpu_valid_passes, 1);
        assert_eq!(aggregate.finish_actor_same_thread_cpu_invalid_passes, 1);
        assert_eq!(
            aggregate.finish_actor_same_thread_endpoint_same_cpu_passes,
            0
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_endpoint_changed_cpu_passes,
            1
        );
        assert_eq!(
            aggregate.finish_actor_same_thread_endpoint_cpu_invalid_passes,
            1
        );
        assert_eq!(aggregate.finish_actor_sender_cpu_mask, 1_u64 << 7);
        assert_eq!(aggregate.finish_actor_receiver_cpu_mask, 1_u64 << 9);
        assert_eq!(aggregate.finish_actor_sender_cpu_id_overflow_passes, 0);
        assert_eq!(aggregate.finish_actor_receiver_cpu_id_overflow_passes, 1);
    }

    #[tokio::test]
    async fn ready_boundary_error_wins_over_cancellation() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        let error = finish_boundary_or_cancel(
            async { Err::<BoundaryFinishTiming, _>(anyhow!("unexpected boundary failure")) },
            &cancel,
        )
        .await
        .unwrap_err();

        assert!(error.to_string().contains("unexpected boundary failure"));
    }

    #[tokio::test]
    async fn cancellation_ends_a_pending_boundary_wait_orderly() {
        let cancel = CancellationToken::new();
        cancel.cancel();

        finish_boundary_or_cancel(
            std::future::pending::<Result<BoundaryFinishTiming>>(),
            &cancel,
        )
        .await
        .unwrap();
    }
}
