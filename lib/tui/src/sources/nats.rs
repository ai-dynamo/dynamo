// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NATS monitoring source.
//!
//! Connects to a NATS server and periodically polls for connection
//! statistics and JetStream stream information. Sends `AppEvent::NatsUpdate`
//! events with the latest stats.

use anyhow::{Context, Result};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing as log;

use super::{AppEvent, Source};
use crate::model::{NatsStats, StreamInfo};

/// Configuration for the NATS source.
#[derive(Debug, Clone)]
pub struct NatsConfig {
    pub server_url: String,
    pub poll_interval: Duration,
}

impl NatsConfig {
    /// Build config from environment variables, with optional CLI overrides.
    pub fn from_env(cli_server: Option<&str>, cli_interval: Option<Duration>) -> Self {
        let server_url = cli_server
            .map(|s| s.to_string())
            .or_else(|| std::env::var("NATS_SERVER").ok())
            .unwrap_or_else(|| "nats://localhost:4222".to_string());

        Self {
            server_url,
            poll_interval: cli_interval.unwrap_or(Duration::from_secs(2)),
        }
    }
}

/// NATS monitoring source that polls for connection and stream statistics.
pub struct NatsSource {
    config: NatsConfig,
}

impl NatsSource {
    pub fn new(config: NatsConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Source for NatsSource {
    async fn run(
        self: Box<Self>,
        tx: tokio::sync::mpsc::Sender<AppEvent>,
        cancel: CancellationToken,
    ) {
        if let Err(e) = run_nats_source(self.config, tx.clone(), cancel.clone()).await {
            let _ = tx
                .send(AppEvent::SourceError {
                    source: "nats".into(),
                    message: format!("{e:#}"),
                })
                .await;
        }
    }
}

async fn run_nats_source(
    config: NatsConfig,
    tx: tokio::sync::mpsc::Sender<AppEvent>,
    cancel: CancellationToken,
) -> Result<()> {
    let client = async_nats::connect(&config.server_url)
        .await
        .context("Failed to connect to NATS")?;

    let js_ctx = async_nats::jetstream::new(client.clone());

    log::info!("Connected to NATS at {}", config.server_url);

    let mut interval = tokio::time::interval(config.poll_interval);

    loop {
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = interval.tick() => {
                let stats = poll_nats_stats(&client, &js_ctx).await;
                let _ = tx.send(AppEvent::NatsUpdate(stats)).await;
            }
        }
    }

    Ok(())
}

async fn poll_nats_stats(
    client: &async_nats::Client,
    js_ctx: &async_nats::jetstream::Context,
) -> NatsStats {
    let info = client.server_info();
    let stats = client.statistics();

    // Collect stream info
    let mut streams = Vec::new();
    if let Ok(stream_names) = collect_stream_names(js_ctx).await {
        for name in stream_names {
            if let Ok(mut stream) = js_ctx.get_stream(&name).await
                && let Ok(info) = stream.info().await {
                    streams.push(StreamInfo {
                        name,
                        consumer_count: info.state.consumer_count,
                        message_count: info.state.messages,
                    });
                }
        }
    }

    NatsStats {
        connected: true,
        server_id: info.server_id.clone(),
        msgs_in: stats.in_messages.load(std::sync::atomic::Ordering::Relaxed),
        msgs_out: stats.out_messages.load(std::sync::atomic::Ordering::Relaxed),
        bytes_in: stats.in_bytes.load(std::sync::atomic::Ordering::Relaxed),
        bytes_out: stats.out_bytes.load(std::sync::atomic::Ordering::Relaxed),
        streams,
    }
}

async fn collect_stream_names(
    js_ctx: &async_nats::jetstream::Context,
) -> Result<Vec<String>> {
    use futures::TryStreamExt;
    let names: Vec<String> = js_ctx.stream_names().try_collect().await?;
    Ok(names)
}

/// Mock NATS source for testing.
#[cfg(test)]
pub mod mock {
    use super::*;

    pub struct MockNatsSource {
        pub stats: NatsStats,
    }

    #[async_trait::async_trait]
    impl Source for MockNatsSource {
        async fn run(
            self: Box<Self>,
            tx: tokio::sync::mpsc::Sender<AppEvent>,
            cancel: CancellationToken,
        ) {
            let _ = tx.send(AppEvent::NatsUpdate(self.stats.clone())).await;
            cancel.cancelled().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nats_config_defaults() {
        // Use CLI override for predictable defaults
        let config = NatsConfig::from_env(Some("nats://localhost:4222"), None);
        assert_eq!(config.server_url, "nats://localhost:4222");
        assert_eq!(config.poll_interval, Duration::from_secs(2));
    }

    #[test]
    fn test_nats_config_cli_override() {
        let config = NatsConfig::from_env(
            Some("nats://custom:4222"),
            Some(Duration::from_secs(5)),
        );
        assert_eq!(config.server_url, "nats://custom:4222");
        assert_eq!(config.poll_interval, Duration::from_secs(5));
    }

    #[test]
    fn test_nats_config_interval_only_override() {
        let config = NatsConfig::from_env(
            Some("nats://localhost:4222"),
            Some(Duration::from_millis(500)),
        );
        assert_eq!(config.poll_interval, Duration::from_millis(500));
    }

    #[tokio::test]
    async fn test_mock_nats_source_sends_event() {
        use crate::model::StreamInfo;

        let stats = NatsStats {
            connected: true,
            server_id: "mock-server".into(),
            msgs_in: 1000,
            msgs_out: 500,
            bytes_in: 1024 * 1024,
            bytes_out: 512 * 1024,
            streams: vec![StreamInfo {
                name: "test-stream".into(),
                consumer_count: 2,
                message_count: 100,
            }],
        };

        let source = Box::new(mock::MockNatsSource { stats: stats.clone() });
        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let cancel = CancellationToken::new();

        let cancel_clone = cancel.clone();
        let handle = tokio::spawn(async move {
            source.run(tx, cancel_clone).await;
        });

        let event = rx.recv().await.unwrap();
        match event {
            AppEvent::NatsUpdate(received) => {
                assert!(received.connected);
                assert_eq!(received.server_id, "mock-server");
                assert_eq!(received.msgs_in, 1000);
                assert_eq!(received.bytes_out, 512 * 1024);
                assert_eq!(received.streams.len(), 1);
                assert_eq!(received.streams[0].name, "test-stream");
                assert_eq!(received.streams[0].consumer_count, 2);
            }
            other => panic!("Expected NatsUpdate, got {:?}", other),
        }

        cancel.cancel();
        handle.await.unwrap();
    }
}
