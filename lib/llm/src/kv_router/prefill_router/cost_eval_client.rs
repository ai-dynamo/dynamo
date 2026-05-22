// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-plane-backed `CostEvaluator` for the RegressionConditionalPrefill
//! policy's slow path.
//!
//! Holds a `PushRouter<CostEvalRequest, Annotated<CostEvalResponse>>` against
//! the cost-eval sidecar's endpoint (default `<namespace>.cost_eval.evaluate_v1`).
//! Each `evaluate` call does a `client.direct(...).await`, takes the first
//! response off the stream, drains the rest (per `agent_controller.rs`
//! precedent), and returns. Any transport failure or timeout returns
//! `CostEvalResponse::unavailable()` so the policy takes its conservative
//! DISAGG fallback.
//!
//! See `docs/regressionpolicy_implementation.md` Phase 3 for the design
//! context; `lib/llm/src/kv_router/agent_controller.rs:289-322` for the
//! pattern this mirrors.

use std::time::Duration;

use async_trait::async_trait;
use dynamo_kv_router::conditional_prefill::{CostEvalRequest, CostEvalResponse, CostEvaluator};
use dynamo_runtime::component::Client;
use dynamo_runtime::pipeline::{PushRouter, RouterMode, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::StreamExt;

/// Default per-request RPC timeout. v1 picks 5ms based on the expected ~1ms
/// regression cost + transport overhead. Configurable via constructor.
const DEFAULT_RPC_TIMEOUT: Duration = Duration::from_millis(5);

type CostEvalPushRouter = PushRouter<CostEvalRequest, Annotated<CostEvalResponse>>;

/// `CostEvaluator` that ships requests over the dynamo request plane to a
/// remote Python sidecar (`components/src/dynamo/cost_eval/`).
pub struct RequestPlaneCostEvaluator {
    router: CostEvalPushRouter,
    timeout: Duration,
}

impl RequestPlaneCostEvaluator {
    /// Build from an already-resolved request-plane `Client` (the kind you
    /// get from `endpoint.client().await?`). Picks `RouterMode::RoundRobin`
    /// — the sidecar is typically a single instance, but if multiple are
    /// stood up for HA we want to spread load.
    pub async fn from_client(client: Client) -> anyhow::Result<Self> {
        let router = PushRouter::from_client(client, RouterMode::RoundRobin).await?;
        Ok(Self {
            router,
            timeout: DEFAULT_RPC_TIMEOUT,
        })
    }

    /// Override the per-request timeout. Default `DEFAULT_RPC_TIMEOUT` (5ms).
    #[allow(dead_code)] // Reserved for the timeout-sweep test wiring (Phase 6).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Pick an instance for round-robin routing. Returns `None` if there are
    /// no available instances (in which case the caller will fall back to
    /// `unavailable()`).
    fn pick_instance(&self) -> Option<u64> {
        self.router.client.instance_ids_avail().first().copied()
    }

    /// Inner async machinery: select instance, dispatch, await first frame
    /// off the stream within the timeout, drain remainder. Returns the
    /// inner `CostEvalResponse` on success; any error path → `Ok(None)`
    /// and the public `evaluate` converts to `unavailable()`.
    async fn evaluate_inner(&self, request: CostEvalRequest) -> Option<CostEvalResponse> {
        let instance_id = self.pick_instance()?;
        let send = self.router.direct(SingleIn::new(request), instance_id);
        let mut stream = match tokio::time::timeout(self.timeout, send).await {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                tracing::warn!(error = %e, "cost-eval RPC failed");
                return None;
            }
            Err(_) => {
                tracing::warn!(
                    timeout_ms = self.timeout.as_millis() as u64,
                    "cost-eval RPC timed out"
                );
                return None;
            }
        };
        let first = stream.next().await;
        // Drain remainder to avoid trailing-publish errors on the sidecar.
        // Matches the pattern in `agent_controller.rs::send_session_request`.
        while stream.next().await.is_some() {}
        let annotated = first?;
        annotated.data
    }
}

#[async_trait]
impl CostEvaluator for RequestPlaneCostEvaluator {
    async fn evaluate(&self, request: CostEvalRequest) -> CostEvalResponse {
        self.evaluate_inner(request)
            .await
            .unwrap_or_else(CostEvalResponse::unavailable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> CostEvalRequest {
        CostEvalRequest {
            request_id: "test-1".into(),
            prompt_tokens: 1024,
            agg_kv_hit_rate: 0.3,
            disagg_kv_hit_rate: 0.1,
        }
    }

    fn sample_response() -> CostEvalResponse {
        CostEvalResponse {
            agg_ttft_ms: Some(12.5),
            disagg_ttft_ms: Some(80.0),
            agg_warm: true,
            disagg_warm: true,
        }
    }

    /// Wire-shape round-trip through serde_json (the format the dynamo
    /// request plane marshals with). Guards against accidental field-name
    /// drift between Rust and the Python Pydantic models.
    #[test]
    fn cost_eval_request_serde_round_trip() {
        let req = sample_request();
        let json = serde_json::to_string(&req).expect("serialize");
        let back: CostEvalRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(req, back);
        // Sanity: field names match the Python wire.py — if these change,
        // the Python Pydantic model must be updated in lockstep.
        assert!(json.contains("\"request_id\""));
        assert!(json.contains("\"prompt_tokens\""));
        assert!(json.contains("\"agg_kv_hit_rate\""));
        assert!(json.contains("\"disagg_kv_hit_rate\""));
    }

    #[test]
    fn cost_eval_response_serde_round_trip() {
        let resp = sample_response();
        let json = serde_json::to_string(&resp).expect("serialize");
        let back: CostEvalResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(resp, back);
        assert!(json.contains("\"agg_ttft_ms\""));
        assert!(json.contains("\"disagg_ttft_ms\""));
        assert!(json.contains("\"agg_warm\""));
        assert!(json.contains("\"disagg_warm\""));
    }

    #[test]
    fn cost_eval_response_unavailable_round_trips() {
        let u = CostEvalResponse::unavailable();
        let json = serde_json::to_string(&u).expect("serialize");
        let back: CostEvalResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(u, back);
        assert!(!u.agg_warm);
        assert!(!u.disagg_warm);
        assert!(u.agg_ttft_ms.is_none());
        assert!(u.disagg_ttft_ms.is_none());
    }
}
