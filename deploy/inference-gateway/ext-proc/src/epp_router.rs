// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-only (selector) endpoint picker.
//!
//! This is the runtime-free counterpart to [`crate::epp::Router`]. It runs with
//! no Dynamo `DistributedRuntime`, no etcd/NATS, and no embedded KV router.
//! Instead it composes:
//!
//! - an offline [`OpenAIPreprocessor`] built from `DYN_MODEL_NAME` (tokenization
//!   for routing only),
//! - a [`PodDiscovery`] that discovers Ready raw vLLM pods from Kubernetes,
//! - a [`TopologyAdapter`] that registers those pods into the selection service,
//!   and
//! - a [`SelectorFleet`] (EndpointSlice-discovered HTTP selection backend) that
//!   asks the selection service to pick a worker.
//!
//! On each request it tokenizes the prompt, asks the selection service for a
//! worker constrained to the currently-Ready pods, and tells Envoy where to send
//! the request via routing headers. Aggregated serving only.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use dynamo_llm::preprocessor::OpenAIPreprocessor;

use crate::epp_config::EppConfig;
use crate::offline_preprocessor::build_offline_preprocessor;
use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo};
use crate::pod_discovery::PodDiscovery;
use crate::selection_backend::{SelectRequest, SelectionBackend};
use crate::selector_fleet::SelectorFleet;
use crate::topology_adapter::{RegistrationDefaults, TopologyAdapter};

/// Best-effort bound on how long startup waits for a selection-service replica
/// to report ready before serving anyway (readiness is also enforced per-pick).
const SELECTOR_READY_WAIT: Duration = Duration::from_secs(30);

/// Router-only endpoint picker backed by the standalone selection service.
pub struct EppRouter {
    preprocessor: Arc<OpenAIPreprocessor>,
    reflector: Arc<PodDiscovery>,
    backend: Arc<dyn SelectionBackend>,
    // Kept alive for the lifetime of the router; the reconcile loop runs on it.
    _adapter: TopologyAdapter,
    reflector_ready: Arc<AtomicBool>,
    model_name: String,
}

impl EppRouter {
    /// Assemble the router-only runtime from the validated selector config.
    pub async fn from_selector(cfg: EppConfig) -> Result<Self> {
        let preprocessor = build_offline_preprocessor(&cfg.model_name, cfg.block_size).await?;

        let (reflector, reflector_ready) = PodDiscovery::spawn(&cfg).await?;
        let reflector = Arc::new(reflector);

        let backend: Arc<dyn SelectionBackend> = Arc::new(SelectorFleet::spawn(&cfg).await?);

        let defaults = RegistrationDefaults::from_config(&cfg);
        let adapter = TopologyAdapter::spawn(reflector.clone(), backend.clone(), defaults);

        // Best-effort wait for the selector to admit at least one worker. The
        // reconcile loop must first see Ready pods and register them, after
        // which the selector reports ready. Per-pick checks still apply, so we
        // never block startup beyond the bound.
        wait_for_selector_ready(backend.as_ref()).await;

        Ok(Self {
            preprocessor,
            reflector,
            backend,
            _adapter: adapter,
            reflector_ready,
            model_name: cfg.model_name,
        })
    }

    /// Readiness flag for the pod reflector; gates gRPC health SERVING.
    pub fn reflector_ready(&self) -> Arc<AtomicBool> {
        self.reflector_ready.clone()
    }

    /// Tokenize a chat-completions request body for routing. Returns
    /// `(token_ids, priority_jump, strict_priority)`.
    fn tokenize(&self, request_json: &str) -> Result<(Vec<u32>, f64, u32)> {
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_str(request_json)?;
        let priority_jump = extract_priority_jump(&request);
        let strict_priority = extract_strict_priority(&request);
        let formatted_prompt = self
            .preprocessor
            .apply_template(&request)?
            .unwrap_or_default();
        let encoding = self.preprocessor.tokenize(&formatted_prompt)?;
        Ok((
            encoding.token_ids().to_vec(),
            priority_jump,
            strict_priority,
        ))
    }
}

async fn wait_for_selector_ready(backend: &dyn SelectionBackend) {
    let deadline = tokio::time::Instant::now() + SELECTOR_READY_WAIT;
    loop {
        if backend.any_ready().await {
            tracing::info!("Selection service reports ready");
            return;
        }
        if tokio::time::Instant::now() >= deadline {
            tracing::warn!(
                "Selection service not ready after {}s; serving anyway (per-pick checks apply)",
                SELECTOR_READY_WAIT.as_secs()
            );
            return;
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
}

#[tonic::async_trait]
impl EndpointPicker for EppRouter {
    async fn pick(
        &self,
        req: &RequestInfo,
        _endpoints: &[Endpoint],
    ) -> Result<PickResult, PickError> {
        if !self.reflector_ready.load(Ordering::Acquire) {
            return Err(PickError::RoutingFailed(
                "pod reflector cache not ready".to_string(),
            ));
        }

        let allowed = self.reflector.ready_worker_ids();
        if allowed.is_empty() {
            return Err(PickError::NoEndpoints);
        }

        // Body-less requests (no prompt to tokenize) route to any Ready worker.
        if req.body.is_empty() {
            let worker_id = *allowed.iter().next().ok_or(PickError::NoEndpoints)?;
            let endpoint = self
                .reflector
                .resolve_endpoint(worker_id)
                .ok_or(PickError::NoEndpoints)?;
            return Ok(PickResult {
                endpoint,
                headers: vec![
                    (
                        "x-dynamo-worker-instance-id".to_string(),
                        worker_id.to_string(),
                    ),
                    (
                        "x-dynamo-routing-mode".to_string(),
                        "aggregated".to_string(),
                    ),
                ],
                ..Default::default()
            });
        }

        let body = std::str::from_utf8(&req.body)
            .map_err(|e| PickError::TokenizationFailed(format!("request body not UTF-8: {e}")))?;
        let (tokens, priority_jump, strict_priority) = self
            .tokenize(body)
            .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;

        let select_req = SelectRequest {
            model_name: self.model_name.clone(),
            selection_id: Some(req.request_id.clone()),
            token_ids: tokens,
            allowed_worker_ids: Some(allowed),
            priority_jump: (priority_jump > 0.0).then_some(priority_jump),
            strict_priority: (strict_priority > 0).then_some(strict_priority),
        };

        let resp = self
            .backend
            .select(&select_req)
            .await
            .map_err(|e| PickError::RoutingFailed(e.to_string()))?;

        // The reflector is the source of truth for the routable address (the
        // pod IP can change between registration and pick). Fall back to the
        // selector-reported endpoint, stripping the URL scheme for Envoy.
        let endpoint = self
            .reflector
            .resolve_endpoint(resp.worker_id)
            .unwrap_or_else(|| strip_scheme(&resp.endpoint).to_string());

        Ok(PickResult {
            endpoint,
            headers: vec![
                (
                    "x-dynamo-worker-instance-id".to_string(),
                    resp.worker_id.to_string(),
                ),
                ("x-dynamo-dp-rank".to_string(), resp.dp_rank.to_string()),
                (
                    "x-dynamo-routing-mode".to_string(),
                    "aggregated".to_string(),
                ),
            ],
            // Worker re-tokenizes the forwarded request (llm-d parity); the EPP
            // tokenizes only for routing, so no token_ids are injected.
            token_ids: None,
            ..Default::default()
        })
    }
}

fn strip_scheme(endpoint: &str) -> &str {
    endpoint
        .strip_prefix("http://")
        .or_else(|| endpoint.strip_prefix("https://"))
        .unwrap_or(endpoint)
}

fn extract_priority_jump(
    request: &dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest,
) -> f64 {
    request
        .nvext
        .as_ref()
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| {
            h.priority
                .map(|p| p.max(0) as f64)
                .or(h.latency_sensitivity)
        })
        .unwrap_or(0.0)
}

fn extract_strict_priority(
    request: &dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest,
) -> u32 {
    request
        .nvext
        .as_ref()
        .and_then(|n| n.agent_hints.as_ref())
        .and_then(|h| h.strict_priority)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_scheme_handles_all_forms() {
        assert_eq!(strip_scheme("http://10.0.0.1:8000"), "10.0.0.1:8000");
        assert_eq!(strip_scheme("https://10.0.0.1:8000"), "10.0.0.1:8000");
        assert_eq!(strip_scheme("10.0.0.1:8000"), "10.0.0.1:8000");
    }
}
