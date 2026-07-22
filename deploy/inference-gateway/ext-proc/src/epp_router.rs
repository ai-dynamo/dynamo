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
use dynamo_llm::protocols::common::extensions::{
    HEADER_REQUEST_PRIORITY, HEADER_REQUEST_STRICT_PRIORITY, HEADER_TENANT_ID, request_cache_salt,
    resolve_request_priority,
};

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

        // Best-effort, bounded wait for the selector to admit a Ready worker;
        // per-pick checks still apply, so startup never blocks past the bound.
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

    /// Tokenize a chat body for routing → `(token_ids, priority_jump,
    /// strict_priority, cache_salt)`. Priority uses header-over-body precedence
    /// via [`resolve_request_priority`]; `cache_salt` via [`request_cache_salt`].
    async fn tokenize(
        &self,
        request_body: bytes::Bytes,
        priority_header: Option<String>,
        strict_priority_header: Option<String>,
    ) -> Result<(Vec<u32>, Option<f64>, Option<u32>, Option<String>), TokenizeError> {
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_slice(&request_body).map_err(TokenizeError::InvalidBody)?;
        let resolved = resolve_request_priority(
            request.nvext.as_ref().and_then(|n| n.agent_hints.as_ref()),
            priority_header.as_deref(),
            strict_priority_header.as_deref(),
        );
        let cache_salt = request_cache_salt(&request).map(str::to_owned);
        // Moves the `Bytes` into reqwest (zero-copy) rather than copying.
        let token_ids = self
            .renderer
            .render_chat(request_body)
            .await
            .map_err(TokenizeError::Render)?;
        Ok((
            token_ids,
            resolved.priority_jump,
            resolved.strict_priority,
            cache_salt,
        ))
    }

    /// Intersect `allowed` Ready workers with an Envoy `candidate_subset`. The
    /// reflector's endpoints are scheme-less `ip:port`, so a worker matches the
    /// subset's full `ip:port` or bare `ip`; empty means nothing matched.
    fn subset_worker_ids(
        &self,
        allowed: &HashSet<u64>,
        candidate_subset: &[String],
    ) -> HashSet<u64> {
        let candidates: HashSet<&str> = candidate_subset.iter().map(String::as_str).collect();
        // Single borrow; the predicate borrows each endpoint (no per-candidate clone).
        self.reflector
            .filter_workers_by_endpoint(allowed, |endpoint| {
                endpoint_in_subset(endpoint, &candidates)
            })
    }
}

/// Overall EPP health: pod readiness AND, when replicated (`peer_ready = Some`),
/// the initial peer sync. `None` means no replication → pod readiness alone.
fn compute_ready(pod_ready: bool, peer_ready: Option<bool>) -> bool {
    pod_ready && peer_ready.unwrap_or(true)
}

/// True if a scheme-less `ip:port` endpoint is covered by an Envoy subset,
/// matching either the full `ip:port` or the bare `ip`.
fn endpoint_in_subset(endpoint: &str, candidates: &HashSet<&str>) -> bool {
    let ip = endpoint.split(':').next().unwrap_or("");
    candidates.contains(endpoint) || candidates.contains(ip)
}

/// Case-insensitive lookup of the first non-empty, trimmed value for `name`.
fn first_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.trim())
        .filter(|v| !v.is_empty())
}

/// Extract the `x-tenant-id` routing header (case-insensitive, non-empty). The
/// Dynamo frontend maps this header onto `cache_salt`, taking precedence over
/// any body value.
fn tenant_id_header(headers: &[(String, String)]) -> Option<String> {
    first_header(headers, HEADER_TENANT_ID).map(str::to_owned)
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

        if !self.reflector.has_ready_workers() {
            return Err(PickError::NoEndpoints);
        }

        // Only the subset-hint path needs an explicit id set. On the ordinary
        // path we pass `None` and let the selector schedule over its own catalog
        // (the stale-selection guard below still validates the chosen worker), so
        // no O(worker-count) set is built per request.
        let allowed: Option<HashSet<u64>> = if req.candidate_subset.is_empty() {
            None
        } else {
            // Honor Envoy's subset hint (`x-gateway-destination-endpoint-subset`):
            // constrain to Ready workers in the subset, refusing (not falling back
            // to the full set) when nothing matches.
            let filtered =
                self.subset_worker_ids(&self.reflector.ready_worker_ids(), &req.candidate_subset);
            if filtered.is_empty() {
                tracing::warn!(
                    subset = ?req.candidate_subset,
                    "No Ready pod matches the subset hint; refusing to route outside the subset"
                );
                return Err(PickError::NoEndpoints);
            }
            Some(filtered)
        };

        // Body-less requests (no prompt to tokenize) route to any Ready worker,
        // staying inside the subset when one was given.
        if req.body.is_empty() {
            let endpoint = match &allowed {
                Some(ids) => {
                    let worker_id = *ids.iter().next().ok_or(PickError::NoEndpoints)?;
                    self.reflector
                        .resolve_endpoint(worker_id)
                        .ok_or(PickError::NoEndpoints)?
                }
                None => self
                    .reflector
                    .resolve_any_endpoint()
                    .ok_or(PickError::NoEndpoints)?,
            };
            // Routing comes from the destination mutation; aggregated raw-vLLM
            // workers read no `x-dynamo-*` headers, so none are emitted.
            return Ok(PickResult {
                endpoint,
                ..Default::default()
            });
        }

        // Header-over-body priority (via the shared resolver), honored here as on
        // the frontend path.
        let priority_header =
            first_header(&req.headers, HEADER_REQUEST_PRIORITY).map(str::to_owned);
        let strict_priority_header =
            first_header(&req.headers, HEADER_REQUEST_STRICT_PRIORITY).map(str::to_owned);
        let (tokens, priority_jump, strict_priority, body_salt) = self
            .tokenize(req.body.clone(), priority_header, strict_priority_header)
            .await
            .map_err(|e| e.into_pick_error(&req.request_id))?;

        // KV cache-isolation namespace: the `x-tenant-id` header overrides the
        // body's `cache_salt`, matching the frontend's header-routing precedence.
        let cache_salt = tenant_id_header(&req.headers).or(body_salt);

        // EPP-minted booking key (not the reused `x-request-id`): stays
        // EPP-known/releasable and rides back on `PickResult::reservation_id`,
        // so the server frees it via the callbacks without a shared map.
        let reservation_id = uuid::Uuid::new_v4().to_string();

        let select_req = SelectRequest {
            model_name: self.model_name.clone(),
            reservation_id: reservation_id.clone(),
            token_ids: tokens,
            // `None` on the ordinary path: the selector schedules over its
            // catalog; `Some` only carries an Envoy subset constraint.
            allowed_worker_ids: allowed,
            // Effective header-over-body values; `None` only when unset everywhere.
            priority_jump,
            strict_priority,
            cache_salt,
        };

        let resp = match self.selector.select_and_reserve(select_req).await {
            Ok(resp) => resp,
            Err(e) => {
                // Release any booking made before the response was lost (a failed
                // pick never reaches the completion callback).
                if let Err(cleanup) = self.selector.free_reservation(&reservation_id).await {
                    tracing::debug!(request_id = %req.request_id, %reservation_id, error = %cleanup, "reservation cleanup after failed reserve");
                }
                return Err(PickError::RoutingFailed(e.to_string()));
            }
        };

        // The reflector owns the address + readiness. If it can no longer resolve
        // the selected worker, the pod left Ready in the race, so the selection is
        // stale: refuse rather than route to a stale address.
        let Some(endpoint) = self.reflector.resolve_endpoint(resp.worker_id) else {
            tracing::warn!(
                worker_id = resp.worker_id,
                "Selected worker no longer resolvable in reflector; treating selection as stale"
            );
            // Booked but not routed; release so the stale selection doesn't leak.
            if let Err(cleanup) = self.selector.free_reservation(&reservation_id).await {
                tracing::debug!(request_id = %req.request_id, %reservation_id, error = %cleanup, "reservation cleanup after stale selection");
            }
            return Err(PickError::NoEndpoints);
        };

        // Routing comes from the destination mutation; aggregated raw-vLLM workers
        // read no `x-dynamo-*` headers. (Disaggregated will add its own contract.)
        Ok(PickResult {
            endpoint,
            // Worker re-tokenizes the forwarded request (llm-d parity); no inject.
            token_ids: None,
            // Booking id for the server's lifecycle callbacks (no shared map).
            reservation_id: Some(reservation_id),
            ..Default::default()
        })
    }

    /// Response complete: release the booking from `pick`. `booking_id` is that
    /// reservation id; `free_reservation` is idempotent (body-less pick → no-op).
    async fn on_request_complete(&self, booking_id: &str) {
        if let Err(e) = self.selector.free_reservation(booking_id).await {
            tracing::warn!(reservation_id = booking_id, error = %e, "Failed to free reservation");
        }
    }

    /// First token: release prefill load, keep decode booked until completion.
    /// `booking_id` is `pick`'s reservation id; `prefill_complete` is idempotent.
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
