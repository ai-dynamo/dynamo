// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone (selector) endpoint picker.
//!
//! This is the runtime-free counterpart to [`crate::epp::Router`]. It runs with
//! no Dynamo `DistributedRuntime`, no etcd/NATS, and no embedded KV router.
//! Instead it composes:
//!
//! - a [`VllmRenderClient`] configured by `DYN_EPP_VLLM_RENDER_URL`
//!   (tokenization for routing only),
//! - a [`PodDiscovery`] that discovers Ready raw vLLM pods from Kubernetes,
//! - a [`TopologyAdapter`] that registers those pods into the selector, and
//! - a [`Selector`] (in-process, runtime-free selection service) that picks a
//!   worker.
//!
//! On each request it tokenizes the prompt, asks the selection service for a
//! worker constrained to the currently-Ready pods, and tells Envoy where to send
//! the request via routing headers. Aggregated serving only.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use dynamo_llm::protocols::common::extensions::{HEADER_TENANT_ID, request_cache_salt};

use crate::epp_standalone_config::EppStandaloneConfig;
use crate::picker::{Endpoint, EndpointPicker, PickError, PickResult, RequestInfo};
use crate::pod_discovery::PodDiscovery;
use crate::selector::{SelectRequest, Selector};
use crate::topology_adapter::{RegistrationDefaults, TopologyAdapter};
use crate::vllm_render_client::VllmRenderClient;

/// Best-effort bound on how long startup waits for a selection-service replica
/// to report ready before serving anyway (readiness is also enforced per-pick).
const SELECTOR_READY_WAIT: Duration = Duration::from_secs(30);

/// Standalone endpoint picker backed by the standalone selection service.
pub struct EppRouter {
    renderer: VllmRenderClient,
    reflector: Arc<PodDiscovery>,
    selector: Arc<Selector>,
    // Kept alive for the lifetime of the router; the reconcile loop runs on it.
    _adapter: TopologyAdapter,
    reflector_ready: Arc<AtomicBool>,
    model_name: String,
}

impl EppRouter {
    /// Assemble the standalone runtime from the validated selector config.
    pub async fn from_selector(cfg: EppStandaloneConfig) -> Result<Self> {
        let renderer = VllmRenderClient::new(
            &cfg.vllm_render_url,
            Duration::from_millis(cfg.tokenization_timeout_ms),
        )?;

        let (reflector, reflector_ready) = PodDiscovery::spawn(&cfg).await?;
        let reflector = Arc::new(reflector);

        let selector = Arc::new(Selector::new(&cfg).await?);

        let defaults = RegistrationDefaults::from_config(&cfg);
        let adapter =
            TopologyAdapter::spawn(reflector.as_ref().clone(), selector.clone(), defaults);

        // Best-effort wait for the selector to admit at least one worker. The
        // reconcile loop must first see Ready pods and register them, after
        // which the selector reports ready. Per-pick checks still apply, so we
        // never block startup beyond the bound.
        wait_for_selector_ready(selector.as_ref()).await;

        Ok(Self {
            renderer,
            reflector,
            selector,
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
    /// `(token_ids, priority_jump, strict_priority, cache_salt)`, where
    /// `cache_salt` is the body-derived KV cache namespace (`nvext.cache_salt`,
    /// with a legacy top-level fallback) via Dynamo's public precedence rules.
    async fn tokenize(&self, request_body: &[u8]) -> Result<(Vec<u32>, f64, u32, Option<String>)> {
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_slice(request_body)?;
        let priority_jump = extract_priority_jump(&request);
        let strict_priority = extract_strict_priority(&request);
        let cache_salt = request_cache_salt(&request).map(str::to_owned);
        let token_ids = self.renderer.render_chat(request_body).await?;
        Ok((token_ids, priority_jump, strict_priority, cache_salt))
    }

    /// Intersect `allowed` Ready worker IDs with an Envoy `candidate_subset`
    /// (endpoint addresses, `ip:port` or bare `ip`). The reflector's endpoints
    /// are scheme-less `ip:port`, so a worker matches when the subset contains
    /// its full `ip:port` or its bare `ip`. An empty result for a non-empty
    /// subset means no Ready pod matched the hint.
    fn subset_worker_ids(
        &self,
        allowed: &HashSet<u64>,
        candidate_subset: &[String],
    ) -> HashSet<u64> {
        let candidates: HashSet<&str> = candidate_subset.iter().map(String::as_str).collect();
        allowed
            .iter()
            .copied()
            .filter(|worker_id| {
                self.reflector
                    .resolve_endpoint(*worker_id)
                    .is_some_and(|endpoint| endpoint_in_subset(&endpoint, &candidates))
            })
            .collect()
    }
}

/// True if a scheme-less `ip:port` endpoint is covered by an Envoy subset,
/// matching either the full `ip:port` or the bare `ip`.
fn endpoint_in_subset(endpoint: &str, candidates: &HashSet<&str>) -> bool {
    let ip = endpoint.split(':').next().unwrap_or("");
    candidates.contains(endpoint) || candidates.contains(ip)
}

/// Extract the `x-tenant-id` routing header (case-insensitive, non-empty). The
/// Dynamo frontend maps this header onto `cache_salt`, taking precedence over
/// any body value.
fn tenant_id_header(headers: &[(String, String)]) -> Option<String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(HEADER_TENANT_ID))
        .map(|(_, v)| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

async fn wait_for_selector_ready(selector: &Selector) {
    let deadline = tokio::time::Instant::now() + SELECTOR_READY_WAIT;
    loop {
        if selector.any_ready().await {
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

        let mut allowed = self.reflector.ready_worker_ids();
        if allowed.is_empty() {
            return Err(PickError::NoEndpoints);
        }

        // Honor Envoy's InferencePool subset hint
        // (`x-gateway-destination-endpoint-subset`): constrain routing to Ready
        // workers inside the requested subset. A non-empty subset that matches no
        // Ready pod must not fall back to the full set — refuse rather than route
        // outside the subset.
        if !req.candidate_subset.is_empty() {
            allowed = self.subset_worker_ids(&allowed, &req.candidate_subset);
            if allowed.is_empty() {
                tracing::warn!(
                    subset = ?req.candidate_subset,
                    "No Ready pod matches the subset hint; refusing to route outside the subset"
                );
                return Err(PickError::NoEndpoints);
            }
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

        let (tokens, priority_jump, strict_priority, body_salt) = self
            .tokenize(&req.body)
            .await
            .map_err(|e| PickError::TokenizationFailed(e.to_string()))?;

        // KV cache-isolation namespace: the `x-tenant-id` header overrides the
        // body's `cache_salt`, matching the frontend's header-routing precedence.
        let cache_salt = tenant_id_header(&req.headers).or(body_salt);

        let select_req = SelectRequest {
            model_name: self.model_name.clone(),
            selection_id: Some(req.request_id.clone()),
            token_ids: tokens,
            allowed_worker_ids: Some(allowed),
            priority_jump: (priority_jump > 0.0).then_some(priority_jump),
            strict_priority: (strict_priority > 0).then_some(strict_priority),
            cache_salt,
        };

        // Atomic select-and-reserve: books the request's load on the chosen
        // worker. Releasing the booking over the request lifecycle
        // (`on_request_complete` / `on_prefill_complete`) is wired in a follow-up.
        let resp = self
            .selector
            .select_and_reserve(select_req)
            .await
            .map_err(|e| PickError::RoutingFailed(e.to_string()))?;

        // The reflector is the source of truth for both the routable address and
        // readiness. If it can no longer resolve the selected worker, the pod left
        // Ready/the pool in the race between building `allowed` and now, so the
        // selection is stale: refuse rather than route to a draining pod or a
        // stale registration-time address. Mirrors the body-less path above.
        let Some(endpoint) = self.reflector.resolve_endpoint(resp.worker_id) else {
            tracing::warn!(
                worker_id = resp.worker_id,
                "Selected worker no longer resolvable in reflector; treating selection as stale"
            );
            return Err(PickError::NoEndpoints);
        };

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
    fn tenant_id_header_is_case_insensitive_and_trims() {
        let headers = vec![("X-Tenant-Id".to_string(), "  tenant-a  ".to_string())];
        assert_eq!(tenant_id_header(&headers), Some("tenant-a".to_string()));
    }

    #[test]
    fn tenant_id_header_absent_or_empty_is_none() {
        assert_eq!(tenant_id_header(&[]), None);
        let empty = vec![("x-tenant-id".to_string(), "   ".to_string())];
        assert_eq!(tenant_id_header(&empty), None);
    }

    #[test]
    fn endpoint_in_subset_matches_ip_port_or_bare_ip() {
        let candidates: HashSet<&str> = ["10.0.0.1:8000", "10.0.0.2"].into_iter().collect();
        // Full ip:port match.
        assert!(endpoint_in_subset("10.0.0.1:8000", &candidates));
        // Bare-ip match (subset lists just the IP).
        assert!(endpoint_in_subset("10.0.0.2:8000", &candidates));
        // Subset pinned a full ip:port, so a different port on that IP does NOT match.
        assert!(!endpoint_in_subset("10.0.0.1:9999", &candidates));
        // Unrelated endpoint does not match.
        assert!(!endpoint_in_subset("10.0.0.3:8000", &candidates));
    }
}
