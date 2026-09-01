// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::time::Duration;

const CBI1_ENVELOPE_HEADROOM: usize = 64 * 1024;
// Keep configured deadlines inside a conservative one-year horizon, below
// Tokio's roughly two-year millisecond timer-wheel range.
const MAX_TIMER_DURATION_MS: u64 = 365 * 24 * 60 * 60 * 1_000;
// Tokio's broadcast channel retains this many load-update messages and rounds the
// allocation up to a power of two. Keep the operator-controlled allocation bounded.
const MAX_LOAD_FANOUT_CAPACITY: usize = 65_536;

#[derive(Debug, Clone)]
pub struct KvDcRelayTransportConfig {
    pub bind: SocketAddr,
    pub max_message_bytes: usize,
    pub keepalive_interval_ms: u64,
    pub keepalive_timeout_ms: u64,
    pub pool_heartbeat_interval_ms: u64,
    pub readiness_heartbeat_interval_ms: u64,
    pub snapshot_progress_timeout_ms: u64,
    pub load_window_ms: u64,
    pub load_fanout_capacity: usize,
    pub publication_queue_capacity: usize,
    pub publication_queue_bytes: usize,
    pub publication_encoding_concurrency: usize,
    pub max_catalog_subscribers: usize,
    pub max_pool_streams_total: usize,
    pub max_subscribers_per_pool: usize,
    pub max_initialized_pool_hubs: usize,
    pub max_readiness_subscribers: usize,
    pub max_load_subscribers: usize,
}

impl KvDcRelayTransportConfig {
    /// Plaintext gRPC transport configuration with default tuning bounds.
    pub fn new(bind: SocketAddr) -> Self {
        Self {
            bind,
            max_message_bytes: 8 * 1024 * 1024,
            keepalive_interval_ms: 20_000,
            keepalive_timeout_ms: 10_000,
            pool_heartbeat_interval_ms: 10_000,
            readiness_heartbeat_interval_ms: 10_000,
            snapshot_progress_timeout_ms: 60_000,
            load_window_ms: 1_000,
            load_fanout_capacity: 16,
            publication_queue_capacity: 16,
            publication_queue_bytes: 16 * 1024 * 1024,
            publication_encoding_concurrency: 2,
            max_catalog_subscribers: 64,
            max_pool_streams_total: 64,
            max_subscribers_per_pool: 64,
            max_initialized_pool_hubs: 64,
            max_readiness_subscribers: 64,
            max_load_subscribers: 64,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        let now = tokio::time::Instant::now();
        for (name, millis) in [
            ("keepalive_interval_ms", self.keepalive_interval_ms),
            ("keepalive_timeout_ms", self.keepalive_timeout_ms),
            (
                "pool_heartbeat_interval_ms",
                self.pool_heartbeat_interval_ms,
            ),
            (
                "readiness_heartbeat_interval_ms",
                self.readiness_heartbeat_interval_ms,
            ),
            (
                "snapshot_progress_timeout_ms",
                self.snapshot_progress_timeout_ms,
            ),
            ("load_window_ms", self.load_window_ms),
        ] {
            anyhow::ensure!(millis != 0, "KV DC Relay WAN {name} must be positive");
            anyhow::ensure!(
                millis <= MAX_TIMER_DURATION_MS,
                "KV DC Relay WAN {name} value {millis} exceeds the maximum timer duration {MAX_TIMER_DURATION_MS} ms"
            );
            anyhow::ensure!(
                now.checked_add(Duration::from_millis(millis)).is_some(),
                "KV DC Relay WAN {name} value {millis} exceeds the Tokio instant range"
            );
        }
        anyhow::ensure!(
            self.load_fanout_capacity != 0
                && self.publication_queue_capacity != 0
                && self.publication_encoding_concurrency != 0,
            "KV DC Relay WAN queue and encoding limits must be positive"
        );
        anyhow::ensure!(
            self.max_catalog_subscribers != 0
                && self.max_pool_streams_total != 0
                && self.max_subscribers_per_pool != 0
                && self.max_initialized_pool_hubs != 0
                && self.max_readiness_subscribers != 0
                && self.max_load_subscribers != 0,
            "KV DC Relay WAN stream and publication limits must be positive"
        );
        anyhow::ensure!(
            self.load_fanout_capacity <= MAX_LOAD_FANOUT_CAPACITY,
            "KV DC Relay WAN load_fanout_capacity {} exceeds the maximum buffered update capacity {}",
            self.load_fanout_capacity,
            MAX_LOAD_FANOUT_CAPACITY
        );
        for (name, capacity) in [
            (
                "publication_queue_capacity",
                self.publication_queue_capacity,
            ),
            (
                "publication_encoding_concurrency",
                self.publication_encoding_concurrency,
            ),
            ("max_catalog_subscribers", self.max_catalog_subscribers),
            ("max_pool_streams_total", self.max_pool_streams_total),
            ("max_initialized_pool_hubs", self.max_initialized_pool_hubs),
            ("max_readiness_subscribers", self.max_readiness_subscribers),
            ("max_load_subscribers", self.max_load_subscribers),
        ] {
            anyhow::ensure!(
                capacity <= tokio::sync::Semaphore::MAX_PERMITS,
                "KV DC Relay WAN {name} value {capacity} exceeds the Tokio semaphore limit {}",
                tokio::sync::Semaphore::MAX_PERMITS
            );
        }
        let minimum_message =
            super::protocol::wire::images::IMAGES_MAX_FRAME_BYTES + CBI1_ENVELOPE_HEADROOM;
        anyhow::ensure!(
            self.max_message_bytes >= minimum_message,
            "KV DC Relay WAN max_message_bytes {} is below the CBI1 frame requirement {}",
            self.max_message_bytes,
            minimum_message
        );
        let minimum_queue = super::protocol::wire::images::IMAGES_MAX_FRAME_BYTES + 256;
        anyhow::ensure!(
            self.publication_queue_bytes >= minimum_queue,
            "KV DC Relay WAN publication_queue_bytes {} cannot hold one maximum frame of {} bytes",
            self.publication_queue_bytes,
            minimum_queue
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_config() -> KvDcRelayTransportConfig {
        KvDcRelayTransportConfig::new("127.0.0.1:0".parse().unwrap())
    }

    #[test]
    fn valid_transport_configuration_is_accepted() {
        valid_config().validate().unwrap();
    }

    #[test]
    fn positive_limits_are_required() {
        let mut config = valid_config();
        config.max_pool_streams_total = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.max_subscribers_per_pool = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.max_initialized_pool_hubs = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.publication_queue_capacity = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.keepalive_timeout_ms = 0;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.snapshot_progress_timeout_ms = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn arbitrary_bind_address_is_preserved() {
        let wildcard: SocketAddr = "0.0.0.0:5561".parse().unwrap();
        let config = KvDcRelayTransportConfig::new(wildcard);
        assert_eq!(config.bind, wildcard);
        config.validate().unwrap();
    }

    #[test]
    fn message_and_queue_limits_must_fit_a_maximum_cbi1_frame() {
        let mut config = valid_config();
        config.max_message_bytes = super::super::protocol::wire::images::IMAGES_MAX_FRAME_BYTES;
        assert!(config.validate().is_err());

        let mut config = valid_config();
        config.publication_queue_bytes =
            super::super::protocol::wire::images::IMAGES_MAX_FRAME_BYTES;
        assert!(config.validate().is_err());
    }

    #[test]
    fn tokio_channel_and_semaphore_capacities_are_bounded() {
        macro_rules! assert_invalid {
            ($field:ident, $value:expr) => {{
                let mut config = valid_config();
                config.$field = $value;
                let error = config.validate().unwrap_err().to_string();
                assert!(
                    error.contains(stringify!($field)),
                    "unexpected validation error for {}: {error}",
                    stringify!($field)
                );
            }};
        }

        assert_invalid!(
            load_fanout_capacity,
            MAX_LOAD_FANOUT_CAPACITY.checked_add(1).unwrap()
        );

        let above_semaphore_limit = tokio::sync::Semaphore::MAX_PERMITS.checked_add(1).unwrap();
        assert_invalid!(publication_queue_capacity, above_semaphore_limit);
        assert_invalid!(publication_encoding_concurrency, above_semaphore_limit);
        assert_invalid!(max_catalog_subscribers, above_semaphore_limit);
        assert_invalid!(max_pool_streams_total, above_semaphore_limit);
        assert_invalid!(max_initialized_pool_hubs, above_semaphore_limit);
        assert_invalid!(max_readiness_subscribers, above_semaphore_limit);
        assert_invalid!(max_load_subscribers, above_semaphore_limit);
    }

    #[test]
    fn duration_values_must_fit_tokio_instant() {
        macro_rules! assert_invalid {
            ($field:ident) => {{
                let mut config = valid_config();
                config.$field = u64::MAX;
                let error = config.validate().unwrap_err().to_string();
                assert!(
                    error.contains(stringify!($field)),
                    "unexpected validation error for {}: {error}",
                    stringify!($field)
                );
            }};
        }

        assert_invalid!(keepalive_interval_ms);
        assert_invalid!(keepalive_timeout_ms);
        assert_invalid!(pool_heartbeat_interval_ms);
        assert_invalid!(readiness_heartbeat_interval_ms);
        assert_invalid!(snapshot_progress_timeout_ms);
        assert_invalid!(load_window_ms);
    }

    #[test]
    fn maximum_timer_duration_is_accepted() {
        let mut config = valid_config();
        config.keepalive_interval_ms = MAX_TIMER_DURATION_MS;
        config.keepalive_timeout_ms = MAX_TIMER_DURATION_MS;
        config.pool_heartbeat_interval_ms = MAX_TIMER_DURATION_MS;
        config.readiness_heartbeat_interval_ms = MAX_TIMER_DURATION_MS;
        config.snapshot_progress_timeout_ms = MAX_TIMER_DURATION_MS;
        config.load_window_ms = MAX_TIMER_DURATION_MS;
        config.validate().unwrap();
    }
}
