// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ETCD discovery source.
//!
//! Connects to ETCD, performs an initial snapshot of all registered
//! instances and model cards, then watches for real-time updates.
//! Translates flat key-value entries into the hierarchical Namespace
//! tree model and sends `AppEvent::DiscoveryUpdate` events.

use anyhow::{Context, Result};
use etcd_client::{GetOptions, WatchOptions};
use tokio_util::sync::CancellationToken;
use tracing as log;

use super::{AppEvent, Source};
use crate::model::{build_tree, parse_endpoint_key, parse_model_key};

const INSTANCES_PREFIX: &str = "v1/instances/";
const MODELS_PREFIX: &str = "v1/mdc/";

/// Configuration for the ETCD source.
#[derive(Debug, Clone)]
pub struct EtcdConfig {
    pub endpoints: Vec<String>,
    pub username: Option<String>,
    pub password: Option<String>,
}

impl EtcdConfig {
    /// Build config from environment variables, with optional CLI overrides.
    pub fn from_env(cli_endpoints: Option<&str>) -> Self {
        let endpoints = cli_endpoints
            .map(|e| e.to_string())
            .or_else(|| std::env::var("ETCD_ENDPOINTS").ok())
            .unwrap_or_else(|| "http://localhost:2379".to_string());

        let endpoints: Vec<String> = endpoints.split(',').map(|s| s.trim().to_string()).collect();

        Self {
            endpoints,
            username: std::env::var("ETCD_AUTH_USERNAME").ok(),
            password: std::env::var("ETCD_AUTH_PASSWORD").ok(),
        }
    }
}

/// ETCD discovery source that watches for namespace/component/endpoint changes.
pub struct EtcdSource {
    config: EtcdConfig,
}

impl EtcdSource {
    pub fn new(config: EtcdConfig) -> Self {
        Self { config }
    }
}

#[async_trait::async_trait]
impl Source for EtcdSource {
    async fn run(
        self: Box<Self>,
        tx: tokio::sync::mpsc::Sender<AppEvent>,
        cancel: CancellationToken,
    ) {
        if let Err(e) = run_etcd_source(self.config, tx.clone(), cancel.clone()).await {
            let _ = tx
                .send(AppEvent::SourceError {
                    source: "etcd".into(),
                    message: format!("{e:#}"),
                })
                .await;
        }
    }
}

async fn run_etcd_source(
    config: EtcdConfig,
    tx: tokio::sync::mpsc::Sender<AppEvent>,
    cancel: CancellationToken,
) -> Result<()> {
    let connect_options = if let (Some(user), Some(pass)) = (&config.username, &config.password) {
        Some(etcd_client::ConnectOptions::new().with_user(user, pass))
    } else {
        None
    };

    let mut client = if let Some(opts) = connect_options {
        etcd_client::Client::connect(&config.endpoints, Some(opts)).await
    } else {
        etcd_client::Client::connect(&config.endpoints, None).await
    }
    .context("Failed to connect to ETCD")?;

    log::info!("Connected to ETCD at {:?}", config.endpoints);

    // State: flat maps of entries
    let mut endpoint_entries: Vec<(String, String, String, u64)> = Vec::new();
    let mut model_entries: Vec<(String, String, String)> = Vec::new();

    // Initial snapshot
    snapshot_endpoints(&mut client, &mut endpoint_entries).await?;
    snapshot_models(&mut client, &mut model_entries).await?;

    let tree = build_tree(&endpoint_entries, &model_entries);
    let _ = tx.send(AppEvent::DiscoveryUpdate(tree)).await;

    // Watch for changes
    let (mut watcher, mut watch_stream) = client
        .watch("v1/", Some(WatchOptions::new().with_prefix()))
        .await
        .context("Failed to create ETCD watch")?;

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                let _ = watcher.cancel().await;
                break;
            }
            msg = watch_stream.message() => {
                match msg {
                    Ok(Some(resp)) => {
                        let mut changed = false;
                        for event in resp.events() {
                            if let Some(kv) = event.kv() {
                                let key = kv.key_str().unwrap_or_default();
                                match event.event_type() {
                                    etcd_client::EventType::Put => {
                                        if let Some(parsed) = parse_endpoint_key(key) {
                                            // Remove old entry with same instance_id, then add
                                            endpoint_entries.retain(|e| e.3 != parsed.3 || e.0 != parsed.0 || e.1 != parsed.1 || e.2 != parsed.2);
                                            endpoint_entries.push(parsed);
                                            changed = true;
                                        } else if let Some(parsed) = parse_model_key(key)
                                            && !model_entries.contains(&parsed) {
                                                model_entries.push(parsed);
                                                changed = true;
                                            }
                                    }
                                    etcd_client::EventType::Delete => {
                                        if let Some(parsed) = parse_endpoint_key(key) {
                                            let before = endpoint_entries.len();
                                            endpoint_entries.retain(|e| e.3 != parsed.3 || e.0 != parsed.0 || e.1 != parsed.1 || e.2 != parsed.2);
                                            changed = endpoint_entries.len() != before;
                                        } else if let Some(parsed) = parse_model_key(key) {
                                            let before = model_entries.len();
                                            model_entries.retain(|e| *e != parsed);
                                            changed = model_entries.len() != before;
                                        }
                                    }
                                }
                            }
                        }
                        if changed {
                            let tree = build_tree(&endpoint_entries, &model_entries);
                            let _ = tx.send(AppEvent::DiscoveryUpdate(tree)).await;
                        }
                    }
                    Ok(None) => break,
                    Err(e) => {
                        let _ = tx.send(AppEvent::SourceError {
                            source: "etcd".into(),
                            message: format!("Watch error: {e:#}"),
                        }).await;
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

async fn snapshot_endpoints(
    client: &mut etcd_client::Client,
    entries: &mut Vec<(String, String, String, u64)>,
) -> Result<()> {
    let resp = client
        .get(INSTANCES_PREFIX, Some(GetOptions::new().with_prefix()))
        .await
        .context("Failed to get ETCD instances")?;

    for kv in resp.kvs() {
        let key = kv.key_str().unwrap_or_default();
        if let Some(parsed) = parse_endpoint_key(key) {
            entries.push(parsed);
        }
    }

    log::info!("Snapshot: {} endpoint instances", entries.len());
    Ok(())
}

async fn snapshot_models(
    client: &mut etcd_client::Client,
    entries: &mut Vec<(String, String, String)>,
) -> Result<()> {
    let resp = client
        .get(MODELS_PREFIX, Some(GetOptions::new().with_prefix()))
        .await
        .context("Failed to get ETCD model cards")?;

    for kv in resp.kvs() {
        let key = kv.key_str().unwrap_or_default();
        if let Some(parsed) = parse_model_key(key)
            && !entries.contains(&parsed)
        {
            entries.push(parsed);
        }
    }

    log::info!("Snapshot: {} model entries", entries.len());
    Ok(())
}

/// Mock ETCD source for testing. Sends a single pre-built discovery update.
#[cfg(test)]
pub mod mock {
    use super::*;
    use crate::model::Namespace;

    pub struct MockEtcdSource {
        pub namespaces: Vec<Namespace>,
    }

    #[async_trait::async_trait]
    impl Source for MockEtcdSource {
        async fn run(
            self: Box<Self>,
            tx: tokio::sync::mpsc::Sender<AppEvent>,
            cancel: CancellationToken,
        ) {
            let _ = tx
                .send(AppEvent::DiscoveryUpdate(self.namespaces.clone()))
                .await;
            // Wait until cancelled
            cancel.cancelled().await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_etcd_config_defaults() {
        // Use CLI override to ensure predictable defaults
        let config = EtcdConfig::from_env(Some("http://localhost:2379"));
        assert_eq!(config.endpoints, vec!["http://localhost:2379"]);
        assert!(config.username.is_none());
    }

    #[test]
    fn test_etcd_config_cli_override() {
        let config = EtcdConfig::from_env(Some("http://etcd1:2379,http://etcd2:2379"));
        assert_eq!(config.endpoints.len(), 2);
        assert_eq!(config.endpoints[0], "http://etcd1:2379");
    }

    #[test]
    fn test_etcd_config_comma_separated_with_spaces() {
        let config = EtcdConfig::from_env(Some("http://a:2379 , http://b:2379 , http://c:2379"));
        assert_eq!(config.endpoints.len(), 3);
        assert_eq!(config.endpoints[0], "http://a:2379");
        assert_eq!(config.endpoints[1], "http://b:2379");
        assert_eq!(config.endpoints[2], "http://c:2379");
    }

    #[test]
    fn test_etcd_config_single_endpoint() {
        let config = EtcdConfig::from_env(Some("http://single:2379"));
        assert_eq!(config.endpoints.len(), 1);
    }

    #[tokio::test]
    async fn test_mock_etcd_source_sends_event() {
        use crate::model::*;

        let namespaces = vec![Namespace {
            name: "test-ns".into(),
            components: vec![Component {
                name: "backend".into(),
                endpoints: vec![Endpoint {
                    name: "generate".into(),
                    instance_count: 1,
                    status: HealthStatus::Ready,
                }],
                instance_count: 1,
                status: HealthStatus::Ready,
                models: vec![],
            }],
        }];

        let source = Box::new(mock::MockEtcdSource {
            namespaces: namespaces.clone(),
        });

        let (tx, mut rx) = tokio::sync::mpsc::channel(16);
        let cancel = CancellationToken::new();

        let cancel_clone = cancel.clone();
        let handle = tokio::spawn(async move {
            source.run(tx, cancel_clone).await;
        });

        // Should receive the discovery update
        let event = rx.recv().await.unwrap();
        match event {
            AppEvent::DiscoveryUpdate(ns) => {
                assert_eq!(ns.len(), 1);
                assert_eq!(ns[0].name, "test-ns");
                assert_eq!(ns[0].components[0].name, "backend");
            }
            other => panic!("Expected DiscoveryUpdate, got {:?}", other),
        }

        cancel.cancel();
        handle.await.unwrap();
    }
}
