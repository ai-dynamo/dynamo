// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Low-cardinality metrics for the fixed-lane QUIC response transport.

use std::sync::{Arc, LazyLock, Weak};

use parking_lot::Mutex;
use prometheus::{
    Histogram, HistogramOpts, HistogramVec, IntCounter, IntCounterVec, IntGauge, Opts,
};

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
        .buckets(vec![
            1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 63.0, 64.0, 96.0, 128.0, 192.0, 255.0, 256.0,
            512.0, 1_024.0, 2_048.0, 4_096.0,
        ]),
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
pub static FIRST_RESPONSE_QUEUE_DWELL_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_quic_response_first_response_queue_dwell_seconds",
            "Time a prologue or first data frame spends in the mocker lane queue",
        )
        .buckets(vec![
            0.000_01, 0.000_1, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
        ]),
        &["kind"],
    )
    .unwrap()
});
pub static FIRST_RESPONSE_BLOCKED_ENQUEUE_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    HistogramVec::new(
        HistogramOpts::new(
            "dynamo_quic_response_first_response_blocked_enqueue_seconds",
            "Full-lane-queue wait for a prologue or first data frame",
        )
        .buckets(vec![
            0.000_01, 0.000_1, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
        ]),
        &["kind"],
    )
    .unwrap()
});
pub static WRITER_WRITE_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_quic_response_writer_write_seconds",
            "Time for one mocker QUIC lane write_all_chunks call",
        )
        .buckets(vec![
            0.000_01, 0.000_1, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
        ]),
    )
    .unwrap()
});
pub static SETUP_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_quic_response_setup_seconds",
            "Time from frontend response registration to QUIC prologue arrival",
        )
        .buckets(vec![
            0.000_1, 0.000_5, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
        ]),
    )
    .unwrap()
});
pub static FIRST_DATA_AFTER_PROLOGUE_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_quic_response_first_data_after_prologue_seconds",
            "Time from QUIC prologue arrival to first data-frame arrival",
        )
        .buckets(vec![
            0.000_01, 0.000_1, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0,
        ]),
    )
    .unwrap()
});
pub static FIRST_DATA_DELIVERY_SECONDS: LazyLock<Histogram> = LazyLock::new(|| {
    Histogram::with_opts(
        HistogramOpts::new(
            "dynamo_quic_response_first_data_delivery_seconds",
            "Time to deliver the first data frame into its frontend response mailbox",
        )
        .buckets(vec![
            0.000_001, 0.000_01, 0.000_1, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0,
        ]),
    )
    .unwrap()
});
pub static SERVER_MAILBOX_RESETS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_quic_response_server_mailbox_resets_total",
            "Logical responses reset because the frontend response mailbox was unavailable",
        ),
        &["reason"],
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
type PrometheusRegistryLock = std::sync::RwLock<prometheus::Registry>;
static REGISTERED: LazyLock<Mutex<Vec<Weak<PrometheusRegistryLock>>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

pub fn track_connection(connection: quinn::Connection) {
    CONNECTIONS_ESTABLISHED.inc();
    CONNECTIONS.lock().push(connection.clone());
    tokio::spawn(async move {
        let connection_id = connection.stable_id();
        let _ = connection.closed().await;
        CONNECTIONS_CLOSED.inc();
        CONNECTIONS
            .lock()
            .retain(|candidate| candidate.stable_id() != connection_id);
    });
}

#[cfg(test)]
pub(crate) fn is_connection_tracked(connection_id: usize) -> bool {
    CONNECTIONS
        .lock()
        .iter()
        .any(|connection| connection.stable_id() == connection_id)
}

pub fn ensure_registered(registry: &MetricsRegistry) {
    let mut registered = REGISTERED.lock();
    registered.retain(|candidate| candidate.strong_count() != 0);
    if !registered.iter().any(|candidate| {
        candidate
            .upgrade()
            .is_some_and(|current| Arc::ptr_eq(&current, &registry.prometheus_registry))
    }) {
        registered.push(Arc::downgrade(&registry.prometheus_registry));
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
        register!(FIRST_RESPONSE_QUEUE_DWELL_SECONDS);
        register!(FIRST_RESPONSE_BLOCKED_ENQUEUE_SECONDS);
        register!(WRITER_WRITE_SECONDS);
        register!(SETUP_SECONDS);
        register!(FIRST_DATA_AFTER_PROLOGUE_SECONDS);
        register!(FIRST_DATA_DELIVERY_SECONDS);
        register!(SERVER_MAILBOX_RESETS_TOTAL);
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
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_metrics_and_callback_once_per_registry() {
        let first = MetricsRegistry::new();
        let second = MetricsRegistry::new();
        ensure_registered(&first);
        ensure_registered(&first);
        ensure_registered(&second);

        for registry in [&first, &second] {
            let names = registry
                .prometheus_registry
                .read()
                .unwrap()
                .gather()
                .into_iter()
                .map(|family| family.name().to_string())
                .collect::<Vec<_>>();
            assert!(
                names
                    .iter()
                    .any(|name| name == "dynamo_quic_response_batches_total")
            );
            assert_eq!(
                registry.prometheus_update_callbacks.read().unwrap().len(),
                1
            );
        }
    }
}
