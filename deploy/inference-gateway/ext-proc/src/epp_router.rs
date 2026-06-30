// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone (selector) endpoint picker.
//!
//! This is the runtime-free counterpart to [`crate::epp::Router`]. It runs with
//! no Dynamo `DistributedRuntime`, no etcd/NATS, and no embedded KV router.
//! Instead it composes:
//!
//! - a [`VllmRenderClient`] configured by `DYN_EPP_TOKENIZER_SERVICE_URL`
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
use crate::vllm_render_client::{VllmRenderClient, VllmRenderError};

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
    /// Peer-discovery readiness (replicated mode only): `None` when replication
    /// is off, else a flag that latches `true` after the initial peer-set sync.
    /// ANDed with `reflector_ready` to form the health signal, so a replicated
    /// pod is not marked SERVING with a local-only load view.
    peer_ready: Option<Arc<AtomicBool>>,
    model_name: String,
}

impl EppRouter {
    /// Assemble the standalone runtime from the validated selector config.
    pub async fn from_selector(cfg: EppStandaloneConfig) -> Result<Self> {
        let renderer = VllmRenderClient::new(
            &cfg.tokenizer_service_url,
            Duration::from_millis(cfg.tokenization_timeout_ms),
            cfg.tokenizer_max_response_bytes,
        )?;

        let (reflector, reflector_ready) = PodDiscovery::spawn(&cfg).await?;
        let reflector = Arc::new(reflector);

        let selector = Arc::new(Selector::new(&cfg).await?);
        let peer_ready = selector.peer_ready();

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
            peer_ready,
            model_name: cfg.model_name,
        })
    }

    /// Overall EPP readiness for the gRPC health signal: the pod reflector is
    /// ready (workers synced + pool resolved) AND, in replicated mode, the peer
    /// set has finished its initial sync. Polled by the health mirror in `main`.
    pub fn is_ready(&self) -> bool {
        compute_ready(
            self.reflector_ready.load(Ordering::Acquire),
            self.peer_ready.as_ref().map(|p| p.load(Ordering::Acquire)),
        )
    }

    /// Tokenize a chat-completions request body for routing. Returns
    /// `(token_ids, priority_jump, strict_priority, cache_salt)`, where
    /// `cache_salt` is the body-derived KV cache namespace (`nvext.cache_salt`,
    /// with a legacy top-level fallback) via Dynamo's public precedence rules.
    async fn tokenize(
        &self,
        request_body: bytes::Bytes,
    ) -> Result<(Vec<u32>, f64, u32, Option<String>), TokenizeError> {
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_slice(&request_body).map_err(TokenizeError::InvalidBody)?;
        let priority_jump = extract_priority_jump(&request);
        let strict_priority = extract_strict_priority(&request);
        let cache_salt = request_cache_salt(&request).map(str::to_owned);
        // Moves the `Bytes` into reqwest (zero-copy) rather than copying.
        let token_ids = self
            .renderer
            .render_chat(request_body)
            .await
            .map_err(TokenizeError::Render)?;
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
        // Filter under a single snapshot borrow; the predicate borrows each
        // endpoint, so we don't clone one per candidate just to test membership.
        self.reflector
            .filter_workers_by_endpoint(allowed, |endpoint| {
                endpoint_in_subset(endpoint, &candidates)
            })
    }
}

/// Combine the pod-reflector readiness with the optional peer-discovery readiness
/// into the overall EPP health signal. `peer_ready = None` means replication is
/// disabled (single replica, no peer gate), so readiness is exactly pod
/// readiness; `Some(false)` holds the pod NOT_SERVING until the initial peer sync.
fn compute_ready(pod_ready: bool, peer_ready: Option<bool>) -> bool {
    pod_ready && peer_ready.unwrap_or(true)
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
            // Routing is driven entirely by the destination mutation the server
            // derives from `endpoint`; aggregated raw-vLLM workers consume no
            // `x-dynamo-*` routing headers, so none are emitted.
            return Ok(PickResult {
                endpoint,
                ..Default::default()
            });
        }

        let (tokens, priority_jump, strict_priority, body_salt) = self
            .tokenize(req.body.clone())
            .await
            .map_err(|e| e.into_pick_error(&req.request_id))?;

        // KV cache-isolation namespace: the `x-tenant-id` header overrides the
        // body's `cache_salt`, matching the frontend's header-routing precedence.
        let cache_salt = tenant_id_header(&req.headers).or(body_salt);

        // Mint an EPP-side reservation id (not the gateway request id): the
        // booking key can't collide with a reused `x-request-id`, and it stays
        // EPP-known so it's releasable even if the reserve response is lost. It
        // rides back on `PickResult::reservation_id`, so the server hands it to
        // the lifecycle callbacks without any shared, request-id-keyed map.
        let reservation_id = uuid::Uuid::new_v4().to_string();

        let select_req = SelectRequest {
            model_name: self.model_name.clone(),
            reservation_id: reservation_id.clone(),
            token_ids: tokens,
            allowed_worker_ids: Some(allowed),
            priority_jump: (priority_jump > 0.0).then_some(priority_jump),
            strict_priority: (strict_priority > 0).then_some(strict_priority),
            cache_salt,
        };

        let resp = match self.selector.select_and_reserve(select_req).await {
            Ok(resp) => resp,
            Err(e) => {
                // Best-effort release any booking the selector may have created
                // before the response was lost, so no reservation leaks. A failed
                // pick never reaches the server's completion callback.
                if let Err(cleanup) = self.selector.free_reservation(&reservation_id).await {
                    tracing::debug!(request_id = %req.request_id, %reservation_id, error = %cleanup, "reservation cleanup after failed reserve");
                }
                return Err(PickError::RoutingFailed(e.to_string()));
            }
        };

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
            // The booking succeeded but we are not routing it: release it so the
            // stale selection does not leak load (a failed pick never triggers
            // `on_request_complete`).
            if let Err(cleanup) = self.selector.free_reservation(&reservation_id).await {
                tracing::debug!(request_id = %req.request_id, %reservation_id, error = %cleanup, "reservation cleanup after stale selection");
            }
            return Err(PickError::NoEndpoints);
        };

        // Routing is driven by the destination mutation the server derives from
        // `endpoint`; aggregated raw-vLLM workers consume no `x-dynamo-*` routing
        // headers, so none are emitted here. (Disaggregated routing will add its
        // own header contract alongside the consumer that reads it.)
        Ok(PickResult {
            endpoint,
            // Worker re-tokenizes the forwarded request (llm-d parity); the EPP
            // tokenizes only for routing, so no token_ids are injected.
            token_ids: None,
            // Ride the booking id back to the server so the lifecycle callbacks
            // free this exact reservation for this stream — no shared map.
            reservation_id: Some(reservation_id),
            ..Default::default()
        })
    }

    /// The gateway signalled the response is complete: release the booking made
    /// in `pick`. `booking_id` is the reservation id `pick` returned for this
    /// stream. `free_reservation` is idempotent, so a request that never booked
    /// (e.g. body-less routing, where `booking_id` is the request id) is a no-op.
    async fn on_request_complete(&self, booking_id: &str) {
        if let Err(e) = self.selector.free_reservation(booking_id).await {
            tracing::warn!(reservation_id = booking_id, error = %e, "Failed to free reservation");
        }
    }

    /// First token generated: release this request's prefill load while leaving
    /// its decode load booked until `on_request_complete`. Applies in aggregated
    /// serving too (drops the transient prefill contribution). `booking_id` is the
    /// reservation id `pick` returned; `prefill_complete` is idempotent.
    async fn on_prefill_complete(&self, booking_id: &str) {
        if let Err(e) = self.selector.prefill_complete(booking_id).await {
            tracing::warn!(reservation_id = booking_id, error = %e, "Failed to mark prefill complete");
        }
    }
}

/// Why tokenizing a request for routing failed. Kept typed so the picker can map
/// each cause to the correct HTTP status instead of collapsing everything to 400.
enum TokenizeError {
    /// The request body could not be parsed — a genuine client (400) error.
    InvalidBody(serde_json::Error),
    /// The vLLM render call failed; the specific variant decides the status.
    Render(VllmRenderError),
}

impl TokenizeError {
    /// Map to a client-safe [`PickError`], logging the detailed cause (which may
    /// include upstream URLs/bodies) server-side rather than returning it.
    fn into_pick_error(self, request_id: &str) -> PickError {
        match self {
            // The serde message describes the client's own JSON, not our
            // internals, so it is safe to surface as a 400.
            TokenizeError::InvalidBody(e) => {
                PickError::TokenizationFailed(format!("invalid request body: {e}"))
            }
            TokenizeError::Render(e) => {
                tracing::warn!(request_id, error = %e, "vLLM render failed");
                match e {
                    VllmRenderError::Unavailable { .. } => PickError::TokenizerUnavailable,
                    VllmRenderError::Timeout { .. } => PickError::TokenizerTimeout,
                    // A 4xx from the renderer means the client's payload was
                    // rejected (client error → 400); a 5xx is the renderer's
                    // fault (→ 502), as is a response that breaks its contract.
                    VllmRenderError::UpstreamStatus { status, .. } if status.is_client_error() => {
                        PickError::TokenizationFailed(
                            "request rejected by tokenization service".to_string(),
                        )
                    }
                    // A too-large or contract-breaking success is the renderer's
                    // fault (→ 502), same as a 5xx.
                    VllmRenderError::UpstreamStatus { .. }
                    | VllmRenderError::InvalidResponse { .. }
                    | VllmRenderError::ResponseTooLarge { .. } => PickError::TokenizerUpstreamError,
                }
            }
        }
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
    fn compute_ready_gates_on_pod_and_peer() {
        // No replication (peer_ready = None): readiness == pod readiness.
        assert!(compute_ready(true, None));
        assert!(!compute_ready(false, None));
        // Replicated: both must be ready. A pod that is worker-ready but hasn't
        // finished its initial peer sync stays NOT_SERVING.
        assert!(compute_ready(true, Some(true)));
        assert!(!compute_ready(true, Some(false)));
        assert!(!compute_ready(false, Some(true)));
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
