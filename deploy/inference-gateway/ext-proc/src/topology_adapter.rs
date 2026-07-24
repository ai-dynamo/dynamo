// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reconciles discovered vLLM pods with the selection-service worker catalog.
//!
//! The adapter converts each ready pod into a [`WorkerRegistration`] using its
//! resolved endpoints and configured defaults, then passes the desired worker
//! set to the [`Selector`].

use std::collections::HashMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::pod_discovery::{PodDiscovery, RawWorker};
use crate::selector::{Selector, WorkerRegistration};

#[derive(Debug, Clone)]
pub struct RegistrationDefaults {
    pub model_name: String,
    pub block_size: u32,
    pub total_kv_blocks: Option<u64>,
    pub max_num_batched_tokens: Option<u64>,
}

impl RegistrationDefaults {
    pub fn from_config(cfg: &EppStandaloneConfig) -> Self {
        Self {
            model_name: cfg.model_name.clone(),
            block_size: cfg.block_size,
            total_kv_blocks: cfg.total_kv_blocks,
            max_num_batched_tokens: cfg.max_num_batched_tokens,
        }
    }
}

/// Background task that keeps the selector catalog in sync with the reflector.
/// Dropping the adapter cancels the task so it stops promptly and releases its
/// `Selector`/`PodDiscovery` handles.
pub struct TopologyAdapter {
    cancel: CancellationToken,
}

impl TopologyAdapter {
    pub fn spawn(
        reflector: PodDiscovery,
        selector: Arc<Selector>,
        defaults: RegistrationDefaults,
    ) -> Self {
        let cancel = CancellationToken::new();
        let cancel_child = cancel.clone();
        tokio::spawn(async move {
            let mut pod_changes = reflector.subscribe_changes();
            loop {
                reconcile_once(&reflector, selector.as_ref(), &defaults).await;
                tokio::select! {
                    _ = cancel_child.cancelled() => break,
                    // Re-reconcile on a pod change. Exit if the sender drops
                    // (reflector gone).
                    changed = pod_changes.changed() => {
                        if changed.is_err() {
                            tracing::warn!(
                                "Reflector change channel closed; clearing selector topology"
                            );
                            if let Err(e) = selector.reconcile(&[]).await {
                                tracing::warn!(
                                    error = %e,
                                    "Failed to clear selector topology after reflector stopped"
                                );
                            }
                            break;
                        }
                    }
                }
            }
        });
        Self { cancel }
    }
}

impl Drop for TopologyAdapter {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Run one reconcile pass: build the desired catalog from the Ready pods and
/// hand it to the selector, which owns the actual-vs-desired diff.
async fn reconcile_once(
    reflector: &PodDiscovery,
    selector: &Selector,
    defaults: &RegistrationDefaults,
) {
    let desired: Vec<WorkerRegistration> = reflector
        .ready_workers()
        .into_iter()
        .map(|w| build_registration(w, defaults))
        .collect();

    if let Err(e) = selector.reconcile(&desired).await {
        tracing::warn!(error = %e, "Selector reconcile failed; will retry on next change");
    }
}

fn build_registration(w: RawWorker, defaults: &RegistrationDefaults) -> WorkerRegistration {
    let mut kv_events_endpoints = HashMap::new();
    kv_events_endpoints.insert(0u32, w.kv_events_endpoint);

    WorkerRegistration {
        worker_id: w.worker_id,
        model_name: defaults.model_name.clone(),
        endpoint: w.http_endpoint,
        block_size: defaults.block_size,
        kv_events_endpoints,
        replay_endpoint: w.replay_endpoint,
        total_kv_blocks: defaults.total_kv_blocks,
        max_num_batched_tokens: defaults.max_num_batched_tokens,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::epp_standalone_config::TokenizerProtocol;

    fn config() -> EppStandaloneConfig {
        EppStandaloneConfig {
            selector_threads: 1,
            peer_service: None,
            inference_pool_name: "test-pool".to_string(),
            namespace: "test-ns".to_string(),
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_service_url: "http://vllm-render:8000".to_string(),
            tokenizer_protocol: TokenizerProtocol::VllmRender,
            tokenizer_max_response_bytes: 16 * 1024 * 1024,
            tokenization_timeout_ms: 5_000,
            block_size: 16,
            kv_event_port: 5557,
            replay_port: None,
            total_kv_blocks: Some(1000),
            max_num_batched_tokens: Some(8192),
        }
    }

    fn defaults() -> RegistrationDefaults {
        RegistrationDefaults {
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            block_size: 16,
            total_kv_blocks: Some(1000),
            max_num_batched_tokens: None,
        }
    }

    fn worker(id: u64, ip: &str) -> RawWorker {
        RawWorker {
            worker_id: id,
            pod_name: format!("vllm-{id}"),
            pod_ip: ip.to_string(),
            http_endpoint: format!("http://{ip}:8000"),
            kv_events_endpoint: format!("tcp://{ip}:5557"),
            replay_endpoint: None,
        }
    }

    #[test]
    fn registration_maps_env_and_endpoints() {
        let reg = build_registration(worker(7, "10.0.0.1"), &defaults());
        assert_eq!(reg.worker_id, 7);
        assert_eq!(reg.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(reg.endpoint, "http://10.0.0.1:8000");
        assert_eq!(reg.block_size, 16);
        assert_eq!(
            reg.kv_events_endpoints.get(&0).unwrap(),
            "tcp://10.0.0.1:5557"
        );
        assert_eq!(reg.total_kv_blocks, Some(1000));
    }

    #[tokio::test]
    async fn channel_close_clears_selector_topology() {
        let selector = Arc::new(
            Selector::new(&config())
                .await
                .expect("selector should build"),
        );
        let (discovery, changes_tx) = PodDiscovery::for_test(vec![worker(7, "10.0.0.1")]);
        let adapter = TopologyAdapter::spawn(discovery, selector.clone(), defaults());

        tokio::time::timeout(Duration::from_secs(1), async {
            while !selector.any_ready().await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("initial topology was not reconciled");

        // There is no unseen generation when the sole sender closes.
        drop(changes_tx);

        tokio::time::timeout(Duration::from_secs(1), async {
            while selector.any_ready().await {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal empty topology was not reconciled");

        drop(adapter);
    }
}
