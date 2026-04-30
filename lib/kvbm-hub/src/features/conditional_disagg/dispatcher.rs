// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub-side prefill-request dispatcher.
//!
//! When a decode worker enqueues a [`PrefillRequest`] via the CD prefill
//! queue, the hub-side dispatcher worker pops items off the queue and hands
//! each one to a [`PrefillRequestDispatcher`] implementation that decides
//! what to *do* with it. The trait split keeps queue-drain orchestration
//! independent of the action: production wires
//! [`HttpVllmDispatcher`] (POST to a registered prefill instance's vLLM
//! `/v1/completions` HTTP frontend with the `kv_transfer_params` blob);
//! tests substitute [`RecordingDispatcher`].

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use parking_lot::Mutex;
use reqwest::StatusCode;
use serde_json::json;

use crate::protocol::PrefillRequest;

/// Outcome reported by a dispatcher implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchOutcome {
    /// The request was accepted by the downstream prefill peer.
    Accepted,
    /// The dispatcher could not place the request — e.g. no eligible
    /// peer, peer rejected, transport error. The hub worker will surface
    /// this to whoever cares (logging today; failure-marker callback in a
    /// future iteration).
    Rejected { reason: String },
}

/// Hub-side action that dispatches a dequeued [`PrefillRequest`] to a
/// prefill participant.
///
/// Implementations must be cheap to clone behind `Arc` and safe to call
/// concurrently from multiple worker tasks (today the hub spawns a single
/// dispatcher task, but the trait is shaped to allow scaling out without
/// changing call sites).
pub trait PrefillRequestDispatcher: Send + Sync {
    fn dispatch(&self, request: PrefillRequest) -> BoxFuture<'_, Result<DispatchOutcome>>;
}

/// Test-only dispatcher that records every received [`PrefillRequest`]
/// for later assertion. Returns `Accepted` for every request.
pub struct RecordingDispatcher {
    received: Mutex<VecDeque<PrefillRequest>>,
}

impl RecordingDispatcher {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            received: Mutex::new(VecDeque::new()),
        })
    }

    /// Snapshot of all dispatched requests, in arrival order. Does not
    /// drain the recorder — repeat calls return the same items.
    pub fn recorded(&self) -> Vec<PrefillRequest> {
        self.received.lock().iter().cloned().collect()
    }

    /// Pop the oldest recorded request, blocking on a Mutex but never
    /// awaiting. Returns `None` if nothing has arrived yet.
    pub fn pop(&self) -> Option<PrefillRequest> {
        self.received.lock().pop_front()
    }

    pub fn len(&self) -> usize {
        self.received.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.received.lock().is_empty()
    }
}

impl Default for RecordingDispatcher {
    fn default() -> Self {
        Self {
            received: Mutex::new(VecDeque::new()),
        }
    }
}

impl PrefillRequestDispatcher for RecordingDispatcher {
    fn dispatch(&self, request: PrefillRequest) -> BoxFuture<'_, Result<DispatchOutcome>> {
        Box::pin(async move {
            self.received.lock().push_back(request);
            Ok(DispatchOutcome::Accepted)
        })
    }
}

/// Production dispatcher: POSTs each [`PrefillRequest`] to a prefill
/// instance's vLLM `/v1/completions` HTTP frontend with the
/// `kv_transfer_params` blob attached so the prefill connector wrapper
/// picks it up via `slot_transfer_params`.
///
/// The request body shape:
///
/// ```json
/// {
///   "model": "<model-id>",
///   "prompt": null,                        // tokens carried in prompt_token_ids
///   "max_tokens": 1,                       // prefill-only; we only need KV state
///   "prompt_token_ids": [...],
///   "kv_transfer_params": {                // surfaced to the connector;
///     "remote_prefill": { ... }            // shape == kvbm_disagg_protocol::TransferParams
///   }
/// }
/// ```
///
/// The `kv_transfer_params` value is serialized directly from
/// [`kvbm_disagg_protocol::TransferParams`] so the prefill connector's
/// `slot.transfer_params()` (which `serde_json::from_value::<TransferParams>`)
/// round-trips it without translation.
///
/// The `model` field is configured at construction; multi-prefill load
/// balancing is out of scope for this impl (single prefill URL +
/// model). When that becomes interesting, swap in a router-aware
/// dispatcher that holds a registry-backed list of prefill peers.
pub struct HttpVllmDispatcher {
    client: reqwest::Client,
    /// Base URL of the prefill vLLM server (e.g.
    /// `http://127.0.0.1:8000`). The dispatcher appends
    /// `/v1/completions`.
    base_url: String,
    /// Model name passed in the request body; must match what the
    /// prefill vLLM was started with.
    model: String,
}

impl HttpVllmDispatcher {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Result<Arc<Self>> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("build reqwest client for HttpVllmDispatcher")?;
        Ok(Arc::new(Self {
            client,
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
        }))
    }
}

impl PrefillRequestDispatcher for HttpVllmDispatcher {
    fn dispatch(&self, request: PrefillRequest) -> BoxFuture<'_, Result<DispatchOutcome>> {
        Box::pin(async move {
            let url = format!("{}/v1/completions", self.base_url);
            let transfer_params = kvbm_disagg_protocol::TransferParams::remote_prefill(
                request.remote_prefill_params(),
            );
            let body = json!({
                "model": self.model,
                "prompt": null,
                "max_tokens": 1,
                "prompt_token_ids": request.token_ids,
                "kv_transfer_params": transfer_params,
            });

            let resp = match self
                .client
                .post(&url)
                .json(&body)
                .send()
                .await
            {
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::PrefillRequest;
    use kvbm_disagg_protocol::DISAGG_PROTOCOL_VERSION;
    use velo_common::InstanceId;

    fn make_request(id: &str) -> PrefillRequest {
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: id.to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            sequence_hashes: vec![],
            token_ids: vec![1, 2, 3],
            num_computed_tokens: 0,
        }
    }

    #[tokio::test]
    async fn recording_dispatcher_accepts_and_records() {
        let d = RecordingDispatcher::new();
        let req = make_request("req-1");
        let outcome = d.dispatch(req.clone()).await.unwrap();
        assert_eq!(outcome, DispatchOutcome::Accepted);
        assert_eq!(d.len(), 1);
        assert_eq!(d.recorded()[0].request_id, "req-1");
    }

    #[tokio::test]
    async fn recording_dispatcher_preserves_order() {
        let d = RecordingDispatcher::new();
        for n in 0..5 {
            let req = make_request(&format!("req-{n}"));
            d.dispatch(req).await.unwrap();
        }
        let recorded = d.recorded();
        assert_eq!(recorded.len(), 5);
        for (i, r) in recorded.iter().enumerate() {
            assert_eq!(r.request_id, format!("req-{i}"));
        }
    }

    #[tokio::test]
    async fn recording_dispatcher_pop_drains() {
        let d = RecordingDispatcher::new();
        d.dispatch(make_request("a")).await.unwrap();
        d.dispatch(make_request("b")).await.unwrap();
        assert_eq!(d.pop().unwrap().request_id, "a");
        assert_eq!(d.pop().unwrap().request_id, "b");
        assert!(d.pop().is_none());
    }

    // ---- HttpVllmDispatcher ----------------------------------------------

    use axum::{Json, Router, http::StatusCode as AxumStatus, routing::post};
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    /// Spin up an axum server that records every POST body and returns
    /// the configured status. Returns base_url, the recorded-body
    /// container, and the join handle (drop = shutdown on test end).
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
    async fn http_dispatcher_posts_to_vllm_completions() {
        let (base, bodies, count, _server) = spawn_stub_vllm(AxumStatus::OK).await;
        let dispatcher = HttpVllmDispatcher::new(base, "Qwen/Qwen3-0.6B").unwrap();
        let req = make_request("http-test-1");
        let outcome = dispatcher.dispatch(req).await.unwrap();
        assert_eq!(outcome, DispatchOutcome::Accepted);
        assert_eq!(count.load(Ordering::SeqCst), 1);

        let body = bodies.lock();
        assert_eq!(body.len(), 1);
        let payload = &body[0];
        assert_eq!(payload["model"].as_str(), Some("Qwen/Qwen3-0.6B"));
        assert_eq!(payload["max_tokens"].as_u64(), Some(1));
        assert!(
            payload["prompt_token_ids"].is_array(),
            "prompt_token_ids must be present"
        );
        // kv_transfer_params must deserialize into kvbm_disagg_protocol::TransferParams
        // — this is the contract the prefill connector relies on via
        // slot.transfer_params(). Round-trip through the typed deserializer so
        // a future field rename / wrapper-key drift fails this test loudly
        // rather than silently producing TransferParams { remote_prefill: None }.
        let kvtp: kvbm_disagg_protocol::TransferParams =
            serde_json::from_value(payload["kv_transfer_params"].clone())
                .expect("kv_transfer_params must deserialize as TransferParams");
        let remote = kvtp
            .remote_prefill
            .expect("remote_prefill must be Some — this is a CD prefill dispatch");
        assert_eq!(remote.protocol_version, DISAGG_PROTOCOL_VERSION);
        assert_eq!(remote.num_computed_tokens, 0);
    }

    #[tokio::test]
    async fn http_dispatcher_marks_5xx_as_rejected() {
        let (base, _bodies, _count, _server) =
            spawn_stub_vllm(AxumStatus::INTERNAL_SERVER_ERROR).await;
        let dispatcher = HttpVllmDispatcher::new(base, "test-model").unwrap();
        let outcome = dispatcher.dispatch(make_request("err-1")).await.unwrap();
        match outcome {
            DispatchOutcome::Rejected { reason } => {
                assert!(reason.contains("500"), "reason should mention status: {reason}");
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_dispatcher_marks_unreachable_as_rejected() {
        // Bind a listener so we can grab a port, then drop it — POST to
        // that port will get connection-refused.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        let dispatcher = HttpVllmDispatcher::new(format!("http://{}", addr), "x").unwrap();
        let outcome = dispatcher.dispatch(make_request("unreachable")).await.unwrap();
        match outcome {
            DispatchOutcome::Rejected { reason } => {
                assert!(
                    reason.contains("failed") || reason.contains("Connection"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }
}
