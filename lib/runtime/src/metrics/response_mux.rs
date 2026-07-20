// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded-cardinality metrics for the multiplexed TCP response transport.

use once_cell::sync::{Lazy, OnceCell};
use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGaugeVec, Opts,
};

use crate::MetricsRegistry;

pub static ACTIVE_CONNECTIONS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_active_connections",
            "Active physical TCP response-mux connections",
        ),
        &["role"],
    )
    .expect("response mux active connection gauge")
});

pub static CONNECTIONS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_connections_total",
            "Physical TCP response-mux connection lifecycle events",
        ),
        &["role", "result"],
    )
    .expect("response mux connection counter")
});

pub static ACTIVE_STREAMS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_active_streams",
            "Active logical response streams",
        ),
        &["role"],
    )
    .expect("response mux active stream gauge")
});

pub static SETUP_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_setup_seconds",
            "Time from request dispatch to logical response-stream prologue",
        )
        .buckets(vec![
            0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
        ]),
    )
    .expect("response mux setup histogram")
});

pub static FRAMES_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_frames_total",
            "Multiplexed response frames by wire direction and type",
        ),
        &["direction", "frame_type"],
    )
    .expect("response mux frame counter")
});

/// Pre-bound frame counters for one wire direction. Keeping these handles next
/// to a connection avoids a label-map lookup for every generated token.
pub struct FrameCounters {
    prologue: IntCounter,
    data: IntCounter,
    end: IntCounter,
    stop: IntCounter,
    kill: IntCounter,
    window_update: IntCounter,
    reset: IntCounter,
    connection_ack: IntCounter,
}

impl FrameCounters {
    pub fn for_direction(direction: &str) -> Self {
        let counter = |frame_type| {
            FRAMES_TOTAL
                .with_label_values(&[direction, frame_type])
                .clone()
        };
        Self {
            prologue: counter("prologue"),
            data: counter("data"),
            end: counter("end"),
            stop: counter("stop"),
            kill: counter("kill"),
            window_update: counter("window_update"),
            reset: counter("reset"),
            connection_ack: counter("connection_ack"),
        }
    }

    pub fn inc(&self, frame_type: &str) {
        match frame_type {
            "prologue" => self.prologue.inc(),
            "data" => self.data.inc(),
            "end" => self.end.inc(),
            "stop" => self.stop.inc(),
            "kill" => self.kill.inc(),
            "window_update" => self.window_update.inc(),
            "reset" => self.reset.inc(),
            "connection_ack" => self.connection_ack.inc(),
            _ => debug_assert!(false, "unknown response mux frame type {frame_type}"),
        }
    }
}

pub static WRITER_QUEUE_DEPTH: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_writer_queue_depth",
            "Queued response-mux frames waiting for the shared writer",
        ),
        &["role"],
    )
    .expect("response mux queue gauge")
});

pub static QUEUED_BYTES: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_queued_bytes",
            "Encoded response bytes queued in connection writers",
        ),
        &["role"],
    )
    .expect("response mux queued byte gauge")
});

pub static FRAMES_PER_WRITE: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_frames_per_write",
            "Logical response-mux frames encoded into each physical write",
        )
        .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
        &["role"],
    )
    .expect("response mux frames-per-write histogram")
});

pub static BATCH_BYTES: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_batch_bytes",
            "Encoded response bytes in each physical TCP write",
        )
        .buckets(vec![
            64.0, 256.0, 1024.0, 4096.0, 16_384.0, 65_536.0, 262_144.0,
        ]),
        &["role"],
    )
    .expect("response mux batch byte histogram")
});

pub static BATCH_WAIT_SECONDS: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_batch_wait_seconds",
            "Observed userspace wait from first selected data frame to write",
        )
        .buckets(vec![0.0, 0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.1]),
        &["role"],
    )
    .expect("response mux batch wait histogram")
});

pub static CONFIGURED_BATCH_INTERVAL_MS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_configured_batch_interval_ms",
            "Configured response data batching interval in milliseconds",
        ),
        &["role"],
    )
    .expect("response mux configured batch interval gauge")
});

pub static WRITE_CALLS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_write_calls_total",
            "Physical response-mux TCP write calls",
        ),
        &["role"],
    )
    .expect("response mux write call counter")
});

pub static DATA_SEGMENTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_data_segments_total",
            "Kernel TCP data segments sent on response sockets when diagnostic packet metrics are enabled",
        ),
        &["transport", "role"],
    )
    .expect("response TCP data segment counter")
});

pub static QUEUE_RESIDENCE_SECONDS: Lazy<HistogramVec> = Lazy::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_queue_residence_seconds",
            "Time logical response frames wait before their physical write",
        )
        .buckets(vec![
            0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1, 1.0,
        ]),
        &["role"],
    )
    .expect("response mux queue residence histogram")
});

pub static RESETS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_resets_total",
            "Logical response streams reset by role and low-cardinality reason",
        ),
        &["role", "reason"],
    )
    .expect("response mux reset counter")
});

pub static RECONNECTS_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_reconnects_total",
            "Replacement physical response-mux connections",
        ),
        &["role"],
    )
    .expect("response mux reconnect counter")
});

pub static FLOW_CONTROL_STALL_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_flow_control_stall_seconds",
            "Time response producers wait for stream-local credits",
        )
        .buckets(vec![0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
    )
    .expect("response mux flow-control histogram")
});

pub static CONNECTION_FLOW_CONTROL_STALL_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_connection_flow_control_stall_seconds",
            "Time response producers wait for physical-connection Data credits",
        )
        .buckets(vec![0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
    )
    .expect("response mux connection flow-control histogram")
});

pub static WRITER_ADMISSION_STALL_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_writer_admission_stall_seconds",
            "Time response producers wait for their stream-local writer queue",
        )
        .buckets(vec![0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
    )
    .expect("response mux writer-admission histogram")
});

pub static QUEUED_BYTE_ADMISSION_STALL_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_queued_byte_admission_stall_seconds",
            "Time response producers wait for connection-wide queued-byte capacity",
        )
        .buckets(vec![0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]),
    )
    .expect("response mux queued-byte admission histogram")
});

pub static STREAM_WRITER_QUEUE_OCCUPANCY: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_stream_writer_queue_occupancy",
            "Stream-local writer queue occupancy after admission",
        )
        .buckets(vec![1.0, 2.0, 4.0, 8.0]),
    )
    .expect("response mux stream writer queue occupancy histogram")
});

pub static READY_STREAMS: Lazy<IntGaugeVec> = Lazy::new(|| {
    IntGaugeVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_ready_streams",
            "Logical streams currently scheduled on a fair connection writer",
        ),
        &["role"],
    )
    .expect("response mux ready stream gauge")
});

pub static PRIORITY_QUEUE_RESIDENCE_SECONDS: Lazy<Histogram> = Lazy::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_tcp_response_mux_priority_queue_residence_seconds",
            "Time prologue and reset frames wait in the priority writer lane",
        )
        .buckets(vec![0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]),
    )
    .expect("response mux priority queue residence histogram")
});

pub static ROUND_ROBIN_TURNS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    IntCounter::new(
        "dynamo_tcp_response_mux_round_robin_turns_total",
        "Frames selected through the fair per-stream writer ring",
    )
    .expect("response mux round-robin turn counter")
});

pub static WINDOW_UPDATES_TOTAL: Lazy<IntCounterVec> = Lazy::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_tcp_response_mux_window_updates_total",
            "Response-mux window update frames",
        ),
        &["direction"],
    )
    .expect("response mux window-update counter")
});

pub static CONNECTION_LOST_STREAMS_TOTAL: Lazy<IntCounter> = Lazy::new(|| {
    IntCounter::new(
        "dynamo_tcp_response_mux_connection_lost_streams_total",
        "Logical streams failed by physical response-mux connection loss",
    )
    .expect("response mux connection-lost stream counter")
});

static REGISTERED: OnceCell<()> = OnceCell::new();

pub fn ensure_registered(registry: &MetricsRegistry) {
    let _ = REGISTERED.get_or_init(|| {
        registry.add_metric_or_warn(
            Box::new(ACTIVE_CONNECTIONS.clone()),
            "response_mux_active_connections",
        );
        registry.add_metric_or_warn(
            Box::new(CONNECTIONS_TOTAL.clone()),
            "response_mux_connections_total",
        );
        registry.add_metric_or_warn(
            Box::new(ACTIVE_STREAMS.clone()),
            "response_mux_active_streams",
        );
        registry.add_metric_or_warn(
            Box::new(SETUP_SECONDS.clone()),
            "response_mux_setup_seconds",
        );
        registry.add_metric_or_warn(Box::new(FRAMES_TOTAL.clone()), "response_mux_frames_total");
        registry.add_metric_or_warn(
            Box::new(WRITER_QUEUE_DEPTH.clone()),
            "response_mux_writer_queue_depth",
        );
        registry.add_metric_or_warn(Box::new(QUEUED_BYTES.clone()), "response_mux_queued_bytes");
        registry.add_metric_or_warn(
            Box::new(FRAMES_PER_WRITE.clone()),
            "response_mux_frames_per_write",
        );
        registry.add_metric_or_warn(Box::new(BATCH_BYTES.clone()), "response_mux_batch_bytes");
        registry.add_metric_or_warn(
            Box::new(BATCH_WAIT_SECONDS.clone()),
            "response_mux_batch_wait_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(CONFIGURED_BATCH_INTERVAL_MS.clone()),
            "response_mux_configured_batch_interval_ms",
        );
        registry.add_metric_or_warn(
            Box::new(WRITE_CALLS_TOTAL.clone()),
            "response_mux_write_calls_total",
        );
        registry.add_metric_or_warn(
            Box::new(DATA_SEGMENTS_TOTAL.clone()),
            "response_data_segments_total",
        );
        registry.add_metric_or_warn(
            Box::new(QUEUE_RESIDENCE_SECONDS.clone()),
            "response_mux_queue_residence_seconds",
        );
        registry.add_metric_or_warn(Box::new(RESETS_TOTAL.clone()), "response_mux_resets_total");
        registry.add_metric_or_warn(
            Box::new(RECONNECTS_TOTAL.clone()),
            "response_mux_reconnects_total",
        );
        registry.add_metric_or_warn(
            Box::new(FLOW_CONTROL_STALL_SECONDS.clone()),
            "response_mux_flow_control_stall_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(CONNECTION_FLOW_CONTROL_STALL_SECONDS.clone()),
            "response_mux_connection_flow_control_stall_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(WRITER_ADMISSION_STALL_SECONDS.clone()),
            "response_mux_writer_admission_stall_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(QUEUED_BYTE_ADMISSION_STALL_SECONDS.clone()),
            "response_mux_queued_byte_admission_stall_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(STREAM_WRITER_QUEUE_OCCUPANCY.clone()),
            "response_mux_stream_writer_queue_occupancy",
        );
        registry.add_metric_or_warn(
            Box::new(READY_STREAMS.clone()),
            "response_mux_ready_streams",
        );
        registry.add_metric_or_warn(
            Box::new(PRIORITY_QUEUE_RESIDENCE_SECONDS.clone()),
            "response_mux_priority_queue_residence_seconds",
        );
        registry.add_metric_or_warn(
            Box::new(ROUND_ROBIN_TURNS_TOTAL.clone()),
            "response_mux_round_robin_turns_total",
        );
        registry.add_metric_or_warn(
            Box::new(WINDOW_UPDATES_TOTAL.clone()),
            "response_mux_window_updates_total",
        );
        registry.add_metric_or_warn(
            Box::new(CONNECTION_LOST_STREAMS_TOTAL.clone()),
            "response_mux_connection_lost_streams_total",
        );
    });
}
