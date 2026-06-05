// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pluggable execution backends for the prefill router.
//!
//! The selector picks a [`super::selection::WorkerSlot`]; the slot's
//! [`PrefillExecutionBackend`] is what actually sends the request to the
//! worker. Two transports are supported: HTTP (POST `/v1/completions`
//! against a registered vLLM frontend) and velo (typed unary call to a
//! worker that hosts the [`super::protocol::PREFILL_DISPATCH_HANDLER`]
//! handler on its own velo participant).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use kvbm_protocols::disagg::TransferParams;
use reqwest::StatusCode;
use serde_json::json;
use velo::Messenger;
use velo_ext::InstanceId;

use super::dispatcher::DispatchOutcome;
use super::protocol::{
    PREFILL_DISPATCH_HANDLER, PrefillDispatchRequest, PrefillDispatchResponse, VllmHttpEndpoint,
};
use crate::protocol::PrefillRequest;

/// Wall-clock guard on a single velo unary call. Caps the wait in case a
/// worker dies between heartbeat sweeps so a single request can't pin the
/// dispatcher task indefinitely; the TTL reaper is what eventually evicts
/// the dead peer.
const VELO_DISPATCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Transport-specific delivery of a single [`PrefillRequest`].
///
/// Implementations encapsulate everything between "router has chosen this
/// worker" and "the worker has accepted (or rejected) the request" — body
/// shaping, network I/O, status mapping. Selection bookkeeping
/// (per-worker counters, fleet permits) lives in
/// [`super::selection::Selector`] and is opaque to the backend.
#[async_trait]
pub trait PrefillExecutionBackend: Send + Sync {
    /// The hub `InstanceId` of the worker this backend reaches. For HTTP
    /// it is the tracking key (the transport target is the configured
    /// `base_url`); for velo it *is* the addressing key.
    fn instance_id(&self) -> InstanceId;

    /// Stable label for logs / debug output (`"http"`, `"velo"`).
    fn label(&self) -> &'static str;

    /// Send `req` to the worker and return how the worker responded.
    async fn execute(&self, req: PrefillRequest) -> Result<DispatchOutcome>;
}

/// HTTP execution backend: POSTs each [`PrefillRequest`] to a registered
/// vLLM frontend's `/v1/completions` endpoint.
///
/// Request body shape (same as the legacy single-target dispatcher so the
/// connector's `slot.transfer_params()` round-trips):
///
/// ```json
/// {
///   "model": "<model-id>",
///   "prompt": [<token-id>, ...],
///   "max_tokens": 1,
///   "kv_transfer_params": { "remote_prefill": { ... } }
/// }
/// ```
pub struct HttpExecutionBackend {
    instance_id: InstanceId,
    client: reqwest::Client,
    base_url: String,
    model: String,
}

impl std::fmt::Debug for HttpExecutionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HttpExecutionBackend")
            .field("instance_id", &self.instance_id)
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .finish()
    }
}

impl HttpExecutionBackend {
    /// Construct a backend pinned to the worker at `endpoint`. Trailing
    /// slashes on `base_url` are stripped.
    pub fn new(instance_id: InstanceId, endpoint: VllmHttpEndpoint) -> Result<Arc<Self>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("build reqwest client for HttpExecutionBackend")?;
        Ok(Arc::new(Self {
            instance_id,
            client,
            base_url: endpoint.base_url.trim_end_matches('/').to_string(),
            model: endpoint.model,
        }))
    }

    /// Resolved base URL (trailing slash stripped).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Model name passed in dispatched POST bodies.
    pub fn model(&self) -> &str {
        &self.model
    }
}

#[async_trait]
impl PrefillExecutionBackend for HttpExecutionBackend {
    fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    fn label(&self) -> &'static str {
        "http"
    }

    async fn execute(&self, request: PrefillRequest) -> Result<DispatchOutcome> {
        let url = format!("{}/v1/completions", self.base_url);
        let transfer_params =
            kvbm_protocols::disagg::TransferParams::remote_prefill(request.remote_prefill_params());
        let body = json!({
            "model": self.model,
            "prompt": request.token_ids,
            "max_tokens": 1,
            "kv_transfer_params": transfer_params,
        });

        let resp = match self.client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(err) => {
                return Ok(DispatchOutcome::Rejected {
                    reason: format!("POST {url} failed: {err}"),
                });
            }
        };

        let status = resp.status();
        if status == StatusCode::OK {
            Ok(DispatchOutcome::Accepted)
        } else {
            let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            Ok(DispatchOutcome::Rejected {
                reason: format!("POST {url} returned {status}: {body}"),
            })
        }
    }
}

/// Velo execution backend: sends a typed unary call to a worker that
/// registered a [`PREFILL_DISPATCH_HANDLER`] handler on its own velo
/// participant. The worker is addressed by the `instance_id` it advertised
/// at registration time; the hub's velo `Messenger` resolves the peer via
/// the [`crate::registry::PeerRegistry`] discovery layer.
pub struct VeloExecutionBackend {
    instance_id: InstanceId,
    messenger: Arc<Messenger>,
}

impl std::fmt::Debug for VeloExecutionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VeloExecutionBackend")
            .field("instance_id", &self.instance_id)
            .finish()
    }
}

impl VeloExecutionBackend {
    /// Wrap an existing hub-side [`Messenger`] (the one the
    /// [`crate::HubServer`] built for active messaging) targeting the
    /// worker at `instance_id`.
    pub fn new(instance_id: InstanceId, messenger: Arc<Messenger>) -> Arc<Self> {
        Arc::new(Self {
            instance_id,
            messenger,
        })
    }
}

#[async_trait]
impl PrefillExecutionBackend for VeloExecutionBackend {
    fn instance_id(&self) -> InstanceId {
        self.instance_id
    }

    fn label(&self) -> &'static str {
        "velo"
    }

    async fn execute(&self, request: PrefillRequest) -> Result<DispatchOutcome> {
        // Pre-wrap the kv_transfer_params blob so the receiving worker can
        // hand it straight to `sampling_params.extra_args` without
        // re-deriving — mirrors the HTTP backend's `/v1/completions` body.
        let dispatch = PrefillDispatchRequest {
            request_id: request.request_id.clone(),
            token_ids: request.token_ids.clone(),
            kv_transfer_params: TransferParams::remote_prefill(request.remote_prefill_params()),
        };

        let call = self
            .messenger
            .typed_unary::<PrefillDispatchResponse>(PREFILL_DISPATCH_HANDLER)
            .with_context(|| format!("typed_unary({PREFILL_DISPATCH_HANDLER}) builder"))?
            .payload(&dispatch)
            .context("encoding PrefillDispatchRequest")?
            .instance(self.instance_id)
            .send();

        let result = tokio::time::timeout(VELO_DISPATCH_TIMEOUT, call).await;
        let resp = match result {
            Ok(Ok(resp)) => resp,
            Ok(Err(err)) => {
                return Ok(DispatchOutcome::Rejected {
                    reason: format!(
                        "velo unary to {target} failed: {err}",
                        target = self.instance_id
                    ),
                });
            }
            Err(_) => {
                return Ok(DispatchOutcome::Rejected {
                    reason: format!(
                        "velo unary to {target} timed out after {:?}",
                        VELO_DISPATCH_TIMEOUT,
                        target = self.instance_id
                    ),
                });
            }
        };
        if resp.ok {
            Ok(DispatchOutcome::Accepted)
        } else {
            Ok(DispatchOutcome::Rejected {
                reason: resp
                    .error
                    .unwrap_or_else(|| "velo worker returned ok=false with no reason".into()),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Json, Router, http::StatusCode as AxumStatus, routing::post};
    use kvbm_protocols::disagg::DISAGG_PROTOCOL_VERSION;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    fn make_request(id: &str) -> PrefillRequest {
        use kvbm_protocols::disagg::KvHashingRequestEnvelope;
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: id.to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            token_ids: vec![1, 2, 3],
            num_provided_tokens: 0,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: None,
        }
    }

    async fn spawn_stub_vllm(
        status: AxumStatus,
    ) -> (
        String,
        Arc<parking_lot::Mutex<Vec<serde_json::Value>>>,
        Arc<AtomicUsize>,
        JoinHandle<()>,
    ) {
        let bodies = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let count = Arc::new(AtomicUsize::new(0));
        let bodies_state = Arc::clone(&bodies);
        let count_state = Arc::clone(&count);

        let app = Router::new().route(
            "/v1/completions",
            post(move |Json(payload): Json<serde_json::Value>| {
                let bodies = Arc::clone(&bodies_state);
                let count = Arc::clone(&count_state);
                async move {
                    bodies.lock().push(payload);
                    count.fetch_add(1, Ordering::SeqCst);
                    if status == AxumStatus::OK {
                        (status, "{}".to_string())
                    } else {
                        (status, "stub error".to_string())
                    }
                }
            }),
        );

        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = TcpListener::bind(addr).await.unwrap();
        let local = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{}", local), bodies, count, handle)
    }

    #[tokio::test]
    async fn http_backend_posts_with_expected_body() {
        let (base, bodies, count, _server) = spawn_stub_vllm(AxumStatus::OK).await;
        let backend = HttpExecutionBackend::new(
            InstanceId::new_v4(),
            VllmHttpEndpoint {
                base_url: base,
                model: "Qwen/Qwen3-0.6B".into(),
            },
        )
        .unwrap();
        assert_eq!(backend.label(), "http");

        let outcome = backend.execute(make_request("r1")).await.unwrap();
        assert_eq!(outcome, DispatchOutcome::Accepted);
        assert_eq!(count.load(Ordering::SeqCst), 1);

        let body = bodies.lock();
        let payload = &body[0];
        assert_eq!(payload["model"].as_str(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(payload["max_tokens"].as_u64(), Some(1));
        assert!(payload["prompt"].is_array());
        let kvtp: kvbm_protocols::disagg::TransferParams =
            serde_json::from_value(payload["kv_transfer_params"].clone())
                .expect("kv_transfer_params must deserialize as TransferParams");
        assert!(kvtp.remote_prefill.is_some());
    }

    #[tokio::test]
    async fn http_backend_marks_5xx_as_rejected() {
        let (base, _bodies, _count, _server) =
            spawn_stub_vllm(AxumStatus::INTERNAL_SERVER_ERROR).await;
        let backend = HttpExecutionBackend::new(
            InstanceId::new_v4(),
            VllmHttpEndpoint {
                base_url: base,
                model: "m".into(),
            },
        )
        .unwrap();
        let outcome = backend.execute(make_request("r1")).await.unwrap();
        match outcome {
            DispatchOutcome::Rejected { reason } => assert!(reason.contains("500")),
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_backend_marks_unreachable_as_rejected() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let backend = HttpExecutionBackend::new(
            InstanceId::new_v4(),
            VllmHttpEndpoint {
                base_url: format!("http://{addr}"),
                model: "m".into(),
            },
        )
        .unwrap();
        let outcome = backend.execute(make_request("r1")).await.unwrap();
        assert!(matches!(outcome, DispatchOutcome::Rejected { .. }));
    }

    // ============================================================
    // VeloExecutionBackend — paired-velo loopback tests.
    // ============================================================

    use velo::Handler;
    use velo::transports::tcp::TcpTransportBuilder;

    fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        Arc::new(
            TcpTransportBuilder::new()
                .from_listener(listener)
                .unwrap()
                .build()
                .unwrap(),
        )
    }

    async fn new_velo() -> Arc<velo::Velo> {
        velo::Velo::builder()
            .add_transport(new_velo_transport())
            .build()
            .await
            .unwrap()
    }

    /// Stand up two paired velo participants. `hub` is the sender side,
    /// `worker` is the receiver side that registers a stub
    /// `PREFILL_DISPATCH_HANDLER` returning `response`. Each side learns
    /// the other's PeerInfo, mimicking what the real hub registry does.
    async fn paired_velos(response: PrefillDispatchResponse) -> (Arc<velo::Velo>, Arc<velo::Velo>) {
        let hub = new_velo().await;
        let worker = new_velo().await;
        hub.register_peer(worker.peer_info()).unwrap();
        worker.register_peer(hub.peer_info()).unwrap();

        let response = Arc::new(response);
        let handler =
            Handler::typed_unary_async::<PrefillDispatchRequest, PrefillDispatchResponse, _, _>(
                PREFILL_DISPATCH_HANDLER,
                move |_ctx| {
                    let response = Arc::clone(&response);
                    async move { Ok((*response).clone()) }
                },
            )
            .build();
        worker.messenger().register_handler(handler).unwrap();
        (hub, worker)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn velo_backend_accepts_when_worker_says_ok() {
        let (hub, worker) = paired_velos(PrefillDispatchResponse {
            ok: true,
            error: None,
        })
        .await;
        let backend = VeloExecutionBackend::new(worker.instance_id(), hub.messenger().clone());
        assert_eq!(backend.label(), "velo");
        let outcome = backend.execute(make_request("r1")).await.unwrap();
        assert_eq!(outcome, DispatchOutcome::Accepted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn velo_backend_propagates_worker_error() {
        let (hub, worker) = paired_velos(PrefillDispatchResponse {
            ok: false,
            error: Some("engine boom".into()),
        })
        .await;
        let backend = VeloExecutionBackend::new(worker.instance_id(), hub.messenger().clone());
        match backend.execute(make_request("r1")).await.unwrap() {
            DispatchOutcome::Rejected { reason } => assert!(reason.contains("engine boom")),
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn velo_backend_carries_kv_transfer_params_on_the_wire() {
        // The worker handler captures the wire request and asserts it
        // carries a non-empty kv_transfer_params with remote_prefill set —
        // proving the hub's symmetric pre-wrap reaches the worker.
        let hub = new_velo().await;
        let worker = new_velo().await;
        hub.register_peer(worker.peer_info()).unwrap();
        worker.register_peer(hub.peer_info()).unwrap();

        let captured: Arc<parking_lot::Mutex<Option<PrefillDispatchRequest>>> =
            Arc::new(parking_lot::Mutex::new(None));
        let captured_state = Arc::clone(&captured);
        let handler =
            Handler::typed_unary_async::<PrefillDispatchRequest, PrefillDispatchResponse, _, _>(
                PREFILL_DISPATCH_HANDLER,
                move |ctx| {
                    let captured = Arc::clone(&captured_state);
                    async move {
                        *captured.lock() = Some(ctx.input.clone());
                        Ok(PrefillDispatchResponse {
                            ok: true,
                            error: None,
                        })
                    }
                },
            )
            .build();
        worker.messenger().register_handler(handler).unwrap();

        let backend = VeloExecutionBackend::new(worker.instance_id(), hub.messenger().clone());
        backend.execute(make_request("r1")).await.unwrap();

        let req = captured.lock().clone().expect("worker received request");
        assert_eq!(req.request_id, "r1");
        assert_eq!(req.token_ids, vec![1, 2, 3]);
        assert!(req.kv_transfer_params.remote_prefill.is_some());
    }
}
