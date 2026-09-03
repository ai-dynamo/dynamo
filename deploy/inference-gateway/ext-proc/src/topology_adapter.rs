// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feeds discovered vLLM pods into the selection-service worker catalog.
//!
//! The pod reflector is exposed as a [`WorkerCatalogSource`]: each ready pod
//! becomes a [`WorkerRequest`] from its resolved endpoints and the configured
//! defaults, and a [`CatalogReconciler`] keeps the catalog in step.

use std::sync::Arc;

use async_trait::async_trait;
use dynamo_kv_router::DEFAULT_ROUTING_GROUP;
use dynamo_kv_router::services::selection::{
    CatalogReconciler, WorkerCatalogSource, WorkerRequest,
};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::pod_discovery::{PodDiscovery, RawWorker};
use crate::selector::Selector;

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

/// The pod reflector as worker membership: every `Ready`, pool-selected pod.
/// When the reflector stops, the source yields one empty snapshot so selection
/// fails closed rather than routing to pods nobody is watching.
struct PodReflectorSource {
    reflector: PodDiscovery,
    changes: watch::Receiver<u64>,
    defaults: RegistrationDefaults,
    primed: bool,
    closed: bool,
}

#[async_trait]
impl WorkerCatalogSource for PodReflectorSource {
    async fn next_snapshot(&mut self) -> Option<Vec<WorkerRequest>> {
        if self.closed {
            return None;
        }
        if self.primed && self.changes.changed().await.is_err() {
            tracing::warn!("reflector change channel closed; clearing selector topology");
            self.closed = true;
            return Some(Vec::new());
        }
        self.primed = true;
        Some(
            self.reflector
                .ready_workers()
                .into_iter()
                .map(|worker| worker_request(worker, &self.defaults))
                .collect(),
        )
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
        let source = PodReflectorSource {
            changes: reflector.subscribe_changes(),
            reflector,
            defaults,
            primed: false,
            closed: false,
        };
        tokio::spawn(
            CatalogReconciler::new(Arc::clone(selector.core())).run(source, cancel.child_token()),
        );
        Self { cancel }
    }
}

impl Drop for TopologyAdapter {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

fn worker_request(w: RawWorker, defaults: &RegistrationDefaults) -> WorkerRequest {
    WorkerRequest {
        worker_id: w.worker_id,
        model_name: defaults.model_name.clone(),
        routing_group: DEFAULT_ROUTING_GROUP.to_string(),
        endpoint: Some(w.http_endpoint),
        block_size: Some(defaults.block_size),
        data_parallel_start_rank: Some(0),
        data_parallel_size: Some((w.kv_events_endpoints.len() as u32).max(1)),
        kv_events_endpoints: w.kv_events_endpoints,
        replay_endpoint: w.replay_endpoint,
        total_kv_blocks: defaults.total_kv_blocks,
        max_num_batched_tokens: defaults.max_num_batched_tokens,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::time::Duration;

    use super::*;
    use crate::epp_standalone_config::RendererProtocol;

    fn config() -> EppStandaloneConfig {
        EppStandaloneConfig {
            selector_threads: 1,
            peer_replication: None,
            inference_pool_name: "test-pool".to_string(),
            namespace: "test-ns".to_string(),
            model_name: "Qwen/Qwen3-0.6B".to_string(),
            tokenizer_service_url: "http://vllm-render:8000".to_string(),
            renderer_protocol: RendererProtocol::VllmRender,
            tokenizer_max_response_bytes: 16 * 1024 * 1024,
            tokenization_timeout_ms: 5_000,
            block_size: 16,
            data_parallel_size: 1,
            kv_event_port_stride: 1,
            kv_event_port: 5557,
            replay_port: None,
            total_kv_blocks: Some(1000),
            max_num_batched_tokens: Some(8192),
            max_inflight_requests: 1024,
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
            kv_events_endpoints: HashMap::from([(0, format!("tcp://{ip}:5557"))]),
            replay_endpoint: None,
        }
    }

    #[test]
    fn registration_maps_env_and_endpoints() {
        let mut raw = worker(7, "10.0.0.1");
        raw.kv_events_endpoints
            .insert(1, "tcp://10.0.0.1:5558".to_string());
        let request = worker_request(raw, &defaults());
        assert_eq!(request.worker_id, 7);
        assert_eq!(request.model_name, "Qwen/Qwen3-0.6B");
        assert_eq!(request.endpoint.as_deref(), Some("http://10.0.0.1:8000"));
        assert_eq!(request.block_size, Some(16));
        assert_eq!(request.data_parallel_size, Some(2));
        assert_eq!(
            request.kv_events_endpoints.get(&0).unwrap(),
            "tcp://10.0.0.1:5557"
        );
        assert_eq!(request.total_kv_blocks, Some(1000));
    }

    #[tokio::test]
    async fn channel_close_clears_selector_topology() {
        let selector = Arc::new(
            Selector::new(
                &config(),
                dynamo_kv_router::services::selection::WorkerSelectionPolicyRegistry::default(),
            )
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
