// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Low-cardinality metrics for the fixed-lane QUIC response transport.

use std::sync::{Arc, LazyLock, OnceLock};

use parking_lot::Mutex;
use prometheus::{Histogram, HistogramOpts, IntCounter, IntGauge};

use crate::MetricsRegistry;

pub static CONNECTIONS_ESTABLISHED: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "dynamo_quic_response_connections_established_total",
        "QUIC response connections established",
    )
    .unwrap()
});
pub static CONNECTIONS_CLOSED: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "dynamo_quic_response_connections_closed_total",
        "QUIC response connections closed",
    )
    .unwrap()
});
pub static BATCHES: LazyLock<IntCounter> = LazyLock::new(|| {
    IntCounter::new(
        "dynamo_quic_response_batches_total",
        "Vectored QUIC response writes",
    )
    .unwrap()
});
pub static FRAMES_PER_BATCH: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_quic_response_frames_per_batch",
            "Logical response frames per vectored QUIC write",
        )
        .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 63.0, 64.0]),
    )
    .unwrap()
});
pub static BATCH_WAIT_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_quic_response_batch_wait_seconds",
            "Time from the first bulk drain until a QUIC response batch is submitted",
        )
        .buckets(vec![
            0.0, 0.000_01, 0.000_05, 0.000_1, 0.000_25, 0.000_5, 0.001, 0.005,
        ]),
    )
    .unwrap()
});
pub static BLOCKED_ENQUEUE_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_quic_response_blocked_enqueue_seconds",
            "Producer wait time after a bounded QUIC lane queue is actually full",
        )
        .buckets(vec![0.000_001, 0.000_01, 0.000_1, 0.001, 0.01, 0.1, 1.0]),
    )
    .unwrap()
});

macro_rules! transport_gauge {
    ($name:ident, $metric:literal, $help:literal) => {
        static $name: LazyLock<IntGauge> = LazyLock::new(|| IntGauge::new($metric, $help).unwrap());
    };
}

transport_gauge!(
    UDP_TX_DATAGRAMS,
    "dynamo_quic_response_udp_tx_datagrams",
    "Current aggregate QUIC UDP datagrams transmitted"
);
transport_gauge!(
    UDP_RX_DATAGRAMS,
    "dynamo_quic_response_udp_rx_datagrams",
    "Current aggregate QUIC UDP datagrams received"
);
transport_gauge!(
    LOST_PACKETS,
    "dynamo_quic_response_lost_packets",
    "Current aggregate QUIC packets lost"
);
transport_gauge!(
    LOST_BYTES,
    "dynamo_quic_response_lost_bytes",
    "Current aggregate QUIC bytes lost"
);
transport_gauge!(
    CONGESTION_EVENTS,
    "dynamo_quic_response_congestion_events",
    "Current aggregate QUIC congestion events"
);
transport_gauge!(
    STREAM_DATA_BLOCKED,
    "dynamo_quic_response_stream_data_blocked_frames",
    "Current aggregate QUIC STREAM_DATA_BLOCKED frames"
);
transport_gauge!(
    RTT_MICROSECONDS,
    "dynamo_quic_response_rtt_microseconds",
    "Sum of current QUIC connection RTT estimates in microseconds"
);

static CONNECTIONS: LazyLock<Mutex<Vec<quinn::Connection>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static REGISTERED: OnceLock<()> = OnceLock::new();

pub fn track_connection(connection: quinn::Connection) {
    CONNECTIONS_ESTABLISHED.inc();
    CONNECTIONS.lock().push(connection.clone());
    tokio::spawn(async move {
        let _ = connection.closed().await;
        CONNECTIONS_CLOSED.inc();
    });
}

pub fn ensure_registered(registry: &MetricsRegistry) {
    REGISTERED.get_or_init(|| {
        macro_rules! register {
            ($metric:ident) => {
                registry.add_metric_or_warn(Box::new($metric.clone()), stringify!($metric));
            };
        }
        register!(CONNECTIONS_ESTABLISHED);
        register!(CONNECTIONS_CLOSED);
        register!(BATCHES);
        register!(FRAMES_PER_BATCH);
        register!(BATCH_WAIT_SECONDS);
        register!(BLOCKED_ENQUEUE_SECONDS);
        register!(UDP_TX_DATAGRAMS);
        register!(UDP_RX_DATAGRAMS);
        register!(LOST_PACKETS);
        register!(LOST_BYTES);
        register!(CONGESTION_EVENTS);
        register!(STREAM_DATA_BLOCKED);
        register!(RTT_MICROSECONDS);
        registry.add_update_callback(Arc::new(|| {
            update_transport_stats();
            Ok(())
        }));
    });
}

fn update_transport_stats() {
    let mut udp_tx = 0_u64;
    let mut udp_rx = 0_u64;
    let mut lost_packets = 0_u64;
    let mut lost_bytes = 0_u64;
    let mut congestion_events = 0_u64;
    let mut stream_data_blocked = 0_u64;
    let mut rtt_us = 0_u64;
    let mut connections = CONNECTIONS.lock();
    connections.retain(|connection| connection.close_reason().is_none());
    for connection in connections.iter() {
        let stats = connection.stats();
        udp_tx = udp_tx.saturating_add(stats.udp_tx.datagrams);
        udp_rx = udp_rx.saturating_add(stats.udp_rx.datagrams);
        lost_packets = lost_packets.saturating_add(stats.path.lost_packets);
        lost_bytes = lost_bytes.saturating_add(stats.path.lost_bytes);
        congestion_events = congestion_events.saturating_add(stats.path.congestion_events);
        stream_data_blocked =
            stream_data_blocked.saturating_add(stats.frame_tx.stream_data_blocked);
        rtt_us = rtt_us.saturating_add(stats.path.rtt.as_micros() as u64);
    }
    UDP_TX_DATAGRAMS.set(as_i64(udp_tx));
    UDP_RX_DATAGRAMS.set(as_i64(udp_rx));
    LOST_PACKETS.set(as_i64(lost_packets));
    LOST_BYTES.set(as_i64(lost_bytes));
    CONGESTION_EVENTS.set(as_i64(congestion_events));
    STREAM_DATA_BLOCKED.set(as_i64(stream_data_blocked));
    RTT_MICROSECONDS.set(as_i64(rtt_us));
}

fn as_i64(value: u64) -> i64 {
    value.min(i64::MAX as u64) as i64
}
