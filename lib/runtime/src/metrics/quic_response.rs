// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Operational metrics for the fixed-lane QUIC response transport.

use std::sync::LazyLock;

use prometheus::{IntCounterVec, IntGauge, Opts, core::Collector};

use crate::MetricsRegistry;

pub static ACTIVE_CONNECTIONS: LazyLock<IntGauge> = LazyLock::new(|| {
    IntGauge::new(
        "dynamo_quic_response_active_connections",
        "Current active QUIC response connections",
    )
    .expect("QUIC response active-connections gauge")
});

pub static BUNDLE_FAILURES_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_quic_response_bundle_failures_total",
            "QUIC response connection bundle failures",
        ),
        &["side"],
    )
    .expect("QUIC response bundle-failures counter")
});

pub static SERVER_MAILBOX_RESETS_TOTAL: LazyLock<IntCounterVec> = LazyLock::new(|| {
    IntCounterVec::new(
        Opts::new(
            "dynamo_quic_response_server_mailbox_resets_total",
            "Logical responses reset because the frontend response mailbox was unavailable",
        ),
        &["reason"],
    )
    .expect("QUIC response mailbox-resets counter")
});

pub fn track_connection(connection: quinn::Connection) {
    ACTIVE_CONNECTIONS.inc();
    tokio::spawn(async move {
        let _ = connection.closed().await;
        ACTIVE_CONNECTIONS.dec();
    });
}

pub fn record_bundle_failure(side: &'static str) {
    BUNDLE_FAILURES_TOTAL.with_label_values(&[side]).inc();
}

pub fn ensure_registered(registry: &MetricsRegistry) {
    for side in ["frontend", "worker"] {
        BUNDLE_FAILURES_TOTAL.with_label_values(&[side]);
    }
    for reason in ["full", "closed"] {
        SERVER_MAILBOX_RESETS_TOTAL.with_label_values(&[reason]);
    }

    let collectors: [Box<dyn Collector>; 3] = [
        Box::new(ACTIVE_CONNECTIONS.clone()),
        Box::new(BUNDLE_FAILURES_TOTAL.clone()),
        Box::new(SERVER_MAILBOX_RESETS_TOTAL.clone()),
    ];
    let registry = registry.prometheus_registry.write().unwrap();
    for collector in collectors {
        match registry.register(collector) {
            Ok(()) | Err(prometheus::Error::AlreadyReg) => {}
            Err(error) => tracing::warn!(%error, "failed to register QUIC response metric"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registers_operational_metrics_in_each_registry() {
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
            for expected in [
                "dynamo_quic_response_active_connections",
                "dynamo_quic_response_bundle_failures_total",
                "dynamo_quic_response_server_mailbox_resets_total",
            ] {
                assert_eq!(
                    names
                        .iter()
                        .filter(|name| name.as_str() == expected)
                        .count(),
                    1
                );
            }
            assert!(
                registry
                    .prometheus_update_callbacks
                    .read()
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
