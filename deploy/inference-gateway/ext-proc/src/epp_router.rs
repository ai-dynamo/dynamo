// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone (selector) endpoint picker.
//!
//! This is the runtime-free counterpart to [`crate::epp::Router`]. It runs with
//! no Dynamo `DistributedRuntime`, no etcd/NATS, and no embedded KV router.
//! Instead it composes:
//!
//! - a [`VllmRenderClient`] tokenization,
//! - a [`PodDiscovery`] that discovers Ready raw vLLM pods from Kubernetes,
//! - a [`TopologyAdapter`] that registers those pods into the selector, and
//! - a [`Selector`] (in-process, runtime-free selection service) that picks a
//!   worker.
//!
//! On each request it tokenizes the prompt, asks the selection service for a
//! worker constrained to the currently-Ready pods, and tells Envoy where to send
//! the request via routing headers.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::Result;
use dynamo_llm::protocols::common::extensions::{
    HEADER_REQUEST_PRIORITY, HEADER_REQUEST_STRICT_PRIORITY, resolve_request_priority,
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

        // Best-effort, bounded wait for the selector to admit a Ready worker
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
    /// strict_priority)`. Priority uses header-over-body precedence via
    /// [`resolve_request_priority`].
    async fn tokenize(
        &self,
        request_body: bytes::Bytes,
        priority_header: Option<String>,
        strict_priority_header: Option<String>,
    ) -> Result<(Vec<u32>, Option<f64>, Option<u32>), TokenizeError> {
        let request: dynamo_llm::types::openai::chat_completions::NvCreateChatCompletionRequest =
            serde_json::from_slice(&request_body).map_err(TokenizeError::InvalidBody)?;
        let resolved = resolve_request_priority(
            request.nvext.as_ref().and_then(|n| n.agent_hints.as_ref()),
            priority_header.as_deref(),
            strict_priority_header.as_deref(),
        );
        // Moves the `Bytes` into reqwest (zero-copy) rather than copying.
        let token_ids = self
            .renderer
            .render_chat(request_body)
            .await
            .map_err(TokenizeError::Render)?;
        Ok((token_ids, resolved.priority_jump, resolved.strict_priority))
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

        // Ordinary path: pass `None` so the SelectionService schedules over its
        // own catalog ("selector owns eligibility") — no O(worker-count) id set is
        // built per request. We accept that the catalog lags the reflector by ~ms
        // after a pod event: the system already tolerates far larger staleness
        // (pod readiness), and the post-select `resolve_endpoint` guard still
        // refuses to route to a worker the reflector can no longer resolve. The
        // freshness-preserving alternative (re-assert the ready set every request)
        // would need an `Arc`-shared set threaded through the core to stay O(1) —
        // not worth the complexity. Only a subset hint (info the selector lacks)
        // needs an explicit id set, built lazily below.
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
        let (tokens, priority_jump, strict_priority) = self
            .tokenize(req.body.clone(), priority_header, strict_priority_header)
            .await
            .map_err(|e| e.into_pick_error(&req.request_id))?;

        // EPP-minted booking key (not the reused `x-request-id`): stays
        // EPP-known/releasable and rides back on `PickResult::reservation_id`,
        // so the server frees it via the callbacks without a shared map.
        let reservation_id = uuid::Uuid::new_v4().to_string();

        // Free the booking if this pick is dropped before it is adopted — the
        // ext-proc stream can close after the scheduler booked but before the
        // server stores `booking_id`, and a booked (past-queue) reservation is not
        // reclaimed by the queue's drop-retraction. Disarmed on the handled paths
        // below; until then, dropping this future frees the reservation.
        let mut reservation_guard = {
            let selector = self.selector.clone();
            let reservation_id = reservation_id.clone();
            ReservationGuard::new(move || {
                tokio::spawn(async move {
                    if let Err(e) = selector.free_reservation(&reservation_id).await {
                        tracing::debug!(%reservation_id, error = %e, "reservation cleanup on dropped pick");
                    }
                });
            })
        };

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
        };

        // On either error return below the guard (still armed) frees the booking.
        let resp = match self.selector.select_and_reserve(select_req).await {
            Ok(resp) => resp,
            Err(e) => return Err(PickError::RoutingFailed(e.to_string())),
        };

        // The reflector owns the address + readiness. If it can no longer resolve
        // the selected worker, the pod left Ready in the race, so the selection is
        // stale: refuse rather than route to a stale address.
        let Some(endpoint) = self.reflector.resolve_endpoint(resp.worker_id) else {
            tracing::warn!(
                worker_id = resp.worker_id,
                "Selected worker no longer resolvable in reflector; treating selection as stale"
            );
            return Err(PickError::NoEndpoints);
        };

        // Success: the caller adopts `reservation_id` synchronously (there is no
        // await between this return and the server storing `booking_id`), so the
        // lifecycle callbacks now own the free — disarm the guard.
        reservation_guard.disarm();

        // Routing comes from the destination mutation; aggregated raw workers
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

/// RAII cleanup for a minted reservation. Armed when `reservation_id` is minted;
/// if the pick future is dropped before the result is adopted (ext-proc stream
/// closed after a booking), `Drop` runs the cleanup (which schedules an
/// idempotent `free_reservation`). Disarmed once the pick is handled, so a
/// successful, adopted pick or an error return does not double-free.
struct ReservationGuard {
    on_drop: Option<Box<dyn FnOnce() + Send>>,
}

impl ReservationGuard {
    fn new(on_drop: impl FnOnce() + Send + 'static) -> Self {
        Self {
            on_drop: Some(Box::new(on_drop)),
        }
    }

    fn disarm(&mut self) {
        self.on_drop = None;
    }
}

impl Drop for ReservationGuard {
    fn drop(&mut self) {
        if let Some(on_drop) = self.on_drop.take() {
            on_drop();
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

    #[test]
    fn reservation_guard_frees_on_drop_unless_disarmed() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        // Dropped while armed — the pick future cancelled after the scheduler
        // booked but before the server adopts the result: cleanup runs.
        let fired = Arc::new(AtomicBool::new(false));
        {
            let fired = fired.clone();
            let _guard = ReservationGuard::new(move || fired.store(true, Ordering::SeqCst));
        }
        assert!(fired.load(Ordering::SeqCst));

        // Disarmed (successful, adopted pick): cleanup does not run.
        let fired = Arc::new(AtomicBool::new(false));
        {
            let fired = fired.clone();
            let mut guard = ReservationGuard::new(move || fired.store(true, Ordering::SeqCst));
            guard.disarm();
        }
        assert!(!fired.load(Ordering::SeqCst));
    }
}
