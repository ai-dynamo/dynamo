// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Feature-owned wire protocol for the prefill router.
//!
//! All paths are **relative** — the server nests them under
//! `/v1/features/{ROUTE_PREFIX}` (see
//! [`FeatureManager::route_prefix`](crate::features::FeatureManager::route_prefix)).
//! Nothing here lives in the central [`crate::protocol::paths`]; the feature
//! owns its whole namespace.

use kvbm_protocols::disagg::TransferParams;
use serde::{Deserialize, Serialize};
use velo_ext::InstanceId;

/// URL segment the server nests this feature's routers under
/// (`/v1/features/prefill-router/...`).
pub const ROUTE_PREFIX: &str = "prefill-router";

/// Velo unary handler name the hub's [`super::execution::VeloExecutionBackend`]
/// calls on each registered velo target. The bindings crate registers the
/// matching handler from `PrefillRouterHandler::new`.
pub const PREFILL_DISPATCH_HANDLER: &str = "kvbm.prefill_router.dispatch";

/// Wire payload sent by the hub-side velo execution backend to a registered
/// prefill worker. NOT the raw [`crate::protocol::PrefillRequest`] because the
/// hub pre-wraps `kv_transfer_params` here, so the receiving worker can hand
/// it directly to `sampling_params.extra_args` without re-deriving — the
/// HTTP backend already does this in its `/v1/completions` body and the velo
/// path stays symmetric.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefillDispatchRequest {
    /// Original request id, propagated for logging / tracing.
    pub request_id: String,
    /// Token ids the worker should run a prefill over (1 sampled token).
    pub token_ids: Vec<u32>,
    /// Remote-prefill metadata the prefill connector consumes via
    /// `sampling_params.extra_args["kv_transfer_params"]`.
    pub kv_transfer_params: TransferParams,
}

/// Wire response a prefill worker returns. `ok=true` means the worker
/// accepted and ran the prefill end-to-end; on failure `error` carries the
/// formatted reason.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefillDispatchResponse {
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Relative route paths (mounted under `/v1/features/prefill-router`).
pub mod paths {
    /// `GET /targets` — registered prefill targets + their advertised
    /// execution backend.
    pub const TARGETS: &str = "/targets";

    /// `GET /counters` — per-worker `inflight` and `load_net_new`
    /// snapshot (debug/observability).
    pub const COUNTERS: &str = "/counters";

    /// `POST /calibrate/:instance_id?force=bool` — forward the body
    /// (a `CalibrationRequest`) to the named worker's velo calibrate
    /// handler and return the `CalibrationResponse` it produces.
    pub const CALIBRATE: &str = "/calibrate/{instance_id}";
}

/// HTTP frontend endpoint a prefill worker advertises so the hub can POST
/// `/v1/completions` requests to it directly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VllmHttpEndpoint {
    /// Base URL of the vLLM frontend (e.g. `http://10.0.0.5:8000`).
    /// Trailing slashes are stripped by the dispatcher before appending
    /// paths.
    pub base_url: String,
    /// Model name to pass in dispatched POST bodies. Must match the vLLM
    /// process's `--model` argument.
    pub model: String,
}

/// How a prefill worker can be reached by the router.
///
/// Asymmetry between variants: for [`Http`](Self::Http) the transport target is
/// `base_url` and the hub's `InstanceId` is used only for tracking/logging.
/// For [`Velo`](Self::Velo) the advertised `velo_ext::InstanceId` *is* the
/// addressing key — there is no out-of-band URL.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PrefillBackendAdvertisement {
    /// POST `/v1/completions` against an HTTP frontend.
    Http(VllmHttpEndpoint),
    /// Send a velo unary call to a worker-side handler keyed by the
    /// advertised `InstanceId`.
    Velo {
        /// Worker velo `InstanceId` the unary call targets.
        instance_id: InstanceId,
    },
}

impl PrefillBackendAdvertisement {
    /// Stable label for logging / debug output.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Http(_) => "http",
            Self::Velo { .. } => "velo",
        }
    }
}

/// Configuration payload for [`crate::protocol::Feature::PrefillRouter`].
///
/// Sent by a Prefill worker at registration to tell the hub how to reach it.
/// Decode workers do not declare this feature.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefillRouterConfig {
    /// Transport the hub should use to dispatch prefill requests to this
    /// worker.
    pub backend: PrefillBackendAdvertisement,
}

/// One entry in the `GET /targets` response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefillTargetSummary {
    /// The worker's hub registration `InstanceId`.
    pub instance_id: InstanceId,
    /// Backend label (`"http"` or `"velo"`).
    pub backend: String,
    /// Verbatim advertisement so an operator can see the actual transport
    /// target (URL for HTTP, velo `InstanceId` for velo).
    pub advertisement: PrefillBackendAdvertisement,
}

/// Response body for `GET /v1/features/prefill-router/targets`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct TargetsResponse {
    /// Registered prefill targets, in deterministic order
    /// (by stringified `InstanceId`) for stable diffs.
    pub targets: Vec<PrefillTargetSummary>,
}

/// One entry in the `GET /counters` response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerCountersSnapshot {
    /// Target the counters belong to.
    pub instance_id: InstanceId,
    /// Backend label (`"http"` or `"velo"`).
    pub backend: String,
    /// Current in-flight request count.
    pub inflight: u32,
    /// Current sum of `net_new` tokens across in-flight requests.
    pub load_net_new: u64,
}

/// Response body for `GET /v1/features/prefill-router/counters`.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub struct CountersResponse {
    /// Per-worker counter snapshots in the same order as [`TargetsResponse`].
    pub workers: Vec<WorkerCountersSnapshot>,
    /// Number of fleet-semaphore permits currently available.
    pub available_permits: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_advertisement_round_trips() {
        let adv = PrefillBackendAdvertisement::Http(VllmHttpEndpoint {
            base_url: "http://10.0.0.5:8000".to_string(),
            model: "Qwen/Qwen3-0.6B".to_string(),
        });
        let json = serde_json::to_string(&adv).unwrap();
        assert!(json.contains("\"kind\":\"http\""));
        let back: PrefillBackendAdvertisement = serde_json::from_str(&json).unwrap();
        assert_eq!(back, adv);
        assert_eq!(back.label(), "http");
    }

    #[test]
    fn velo_advertisement_round_trips() {
        let id = InstanceId::new_v4();
        let adv = PrefillBackendAdvertisement::Velo { instance_id: id };
        let json = serde_json::to_string(&adv).unwrap();
        assert!(json.contains("\"kind\":\"velo\""));
        let back: PrefillBackendAdvertisement = serde_json::from_str(&json).unwrap();
        assert_eq!(back, adv);
        assert_eq!(back.label(), "velo");
    }

    #[test]
    fn prefill_router_config_round_trips() {
        let cfg = PrefillRouterConfig {
            backend: PrefillBackendAdvertisement::Http(VllmHttpEndpoint {
                base_url: "http://x:8000".into(),
                model: "m".into(),
            }),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let back: PrefillRouterConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn prefill_dispatch_request_round_trips() {
        use kvbm_protocols::disagg::{
            DISAGG_PROTOCOL_VERSION, KvHashingRequestEnvelope, RemotePrefillParams, TransferParams,
        };
        let params = RemotePrefillParams {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            num_provided_tokens: 0,
            request: KvHashingRequestEnvelope::default(),
            expected_hash_digest: None,
        };
        let req = PrefillDispatchRequest {
            request_id: "r1".to_string(),
            token_ids: vec![1, 2, 3],
            kv_transfer_params: TransferParams::remote_prefill(params),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: PrefillDispatchRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, req);
    }

    #[test]
    fn prefill_dispatch_response_round_trips() {
        for (ok, error) in [(true, None), (false, Some("nope".to_string()))] {
            let resp = PrefillDispatchResponse {
                ok,
                error: error.clone(),
            };
            let json = serde_json::to_string(&resp).unwrap();
            let back: PrefillDispatchResponse = serde_json::from_str(&json).unwrap();
            assert_eq!(back, resp);
        }
    }
}
