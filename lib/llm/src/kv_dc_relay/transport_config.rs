// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::path::PathBuf;

const CBI1_ENVELOPE_HEADROOM: usize = 64 * 1024;

#[derive(Debug, Clone)]
pub struct KvDcRelayTransportConfig {
    pub bind: SocketAddr,
    pub tls_server_cert: PathBuf,
    pub tls_server_key: PathBuf,
    pub tls_client_ca: PathBuf,
    pub max_message_bytes: usize,
    pub keepalive_interval_ms: u64,
    pub keepalive_timeout_ms: u64,
    pub pool_heartbeat_interval_ms: u64,
    pub readiness_heartbeat_interval_ms: u64,
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
    /// Transport configuration with required WAN material and default tuning bounds.
    pub fn new(
        bind: SocketAddr,
        tls_server_cert: PathBuf,
        tls_server_key: PathBuf,
        tls_client_ca: PathBuf,
    ) -> Self {
        Self {
            bind,
            tls_server_cert,
            tls_server_key,
            tls_client_ca,
            max_message_bytes: 8 * 1024 * 1024,
            keepalive_interval_ms: 20_000,
            keepalive_timeout_ms: 10_000,
            pool_heartbeat_interval_ms: 10_000,
            readiness_heartbeat_interval_ms: 10_000,
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
        anyhow::ensure!(
            !self.tls_server_cert.as_os_str().is_empty(),
            "KV DC Relay WAN transport requires a TLS server certificate"
        );
        anyhow::ensure!(
            !self.tls_server_key.as_os_str().is_empty(),
            "KV DC Relay WAN transport requires a TLS server key"
        );
        anyhow::ensure!(
            !self.tls_client_ca.as_os_str().is_empty(),
            "KV DC Relay WAN transport requires a client CA for mTLS"
        );
        anyhow::ensure!(
            self.keepalive_interval_ms != 0 && self.keepalive_timeout_ms != 0,
            "KV DC Relay WAN keepalive values must be positive"
        );
        anyhow::ensure!(
            self.pool_heartbeat_interval_ms != 0 && self.readiness_heartbeat_interval_ms != 0,
            "KV DC Relay WAN heartbeat intervals must be positive"
        );
        anyhow::ensure!(
            self.load_window_ms != 0,
            "KV DC Relay WAN load window must be positive"
        );
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
        KvDcRelayTransportConfig::new(
            "127.0.0.1:0".parse().unwrap(),
            PathBuf::from("server.crt"),
            PathBuf::from("server.key"),
            PathBuf::from("ca.crt"),
        )
    }

    #[test]
    fn valid_transport_configuration_is_accepted() {
        valid_config().validate().unwrap();
    }

    #[test]
    fn mtls_material_and_positive_limits_are_required() {
        let mut config = valid_config();
        config.tls_client_ca = PathBuf::new();
        assert!(config.validate().is_err());

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
}
