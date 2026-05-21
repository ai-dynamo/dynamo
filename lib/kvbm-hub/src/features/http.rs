// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared HTTP plumbing for feature managers that proxy hub control routes to
//! per-leader velo handlers.
//!
//! Both [`ControlPlaneManager`](crate::features::control_plane::ControlPlaneManager)
//! and [`P2pManager`](crate::features::p2p::P2pManager) bridge HTTP →
//! [`LeaderControlClient`] the same way: verify velo + registry are attached,
//! verify the target instance is registered, build the client, and map
//! `Result<T, ControlError>` → an HTTP status + JSON body. These helpers are
//! the single source of truth for that plumbing so the two managers can't
//! drift.

use std::sync::{Arc, OnceLock};

use axum::http::{HeaderMap, HeaderValue, StatusCode, header::CONTENT_TYPE};
use axum::response::{IntoResponse, Response};
use kvbm_protocols::control::{ControlError, LeaderControlClient};
use serde::Serialize;
use velo_ext::InstanceId;

use crate::protocol::{ErrorBody, ErrorCode};
use crate::registry::PeerRegistry;

/// Build a [`LeaderControlClient`] for `instance_id` after validating velo +
/// registry attachment and that the instance is currently registered.
///
/// `velo` / `registry` are the manager's own `OnceLock`s, populated during
/// [`FeatureManager::attach`](crate::features::FeatureManager::attach).
///
/// `Err` is boxed-as-`Response` because [`axum::response::Response`] is
/// ~120 bytes and would otherwise blow up the `result_large_err` clippy lint
/// on the happy path — the `#[allow]` lives here so both call sites inherit it.
#[allow(clippy::result_large_err)]
pub(crate) fn leader_client(
    velo: &OnceLock<Arc<velo::Velo>>,
    registry: &OnceLock<Arc<dyn PeerRegistry>>,
    instance_id: InstanceId,
) -> Result<LeaderControlClient, Response> {
    let Some(velo) = velo.get() else {
        return Err(service_unavailable("hub has no velo transport configured"));
    };
    // The hub self-registers in its own registry (so leaders can discover it),
    // which means its own velo id passes the `contains` check below. A control
    // RPC to ourselves would route through velo to a peer that is never in the
    // hub's own messenger peer table — "Peer <id> not registered" → 500 — and a
    // client that learned `hub_velo_id` from its registration response can spam
    // that on every poll. The hub has no leader control handlers anyway. Reject
    // the self id up front for every control handler that funnels through here.
    if instance_id == velo.instance_id() {
        return Err(error_response(
            StatusCode::BAD_REQUEST,
            "cannot issue leader control to the hub itself",
        ));
    }
    let Some(registry) = registry.get() else {
        return Err(service_unavailable("registry not attached"));
    };
    if !registry.contains(instance_id) {
        return Err(error_response(
            StatusCode::NOT_FOUND,
            "instance not registered",
        ));
    }
    Ok(LeaderControlClient::new(
        velo.messenger().clone(),
        instance_id,
    ))
}

pub(crate) fn ok_response<T: Serialize>(value: &T) -> Response {
    json_response(
        StatusCode::OK,
        serde_json::to_value(value).unwrap_or(serde_json::json!({})),
    )
}

pub(crate) fn control_error_response(
    instance_id: InstanceId,
    handler: &str,
    err: ControlError,
) -> Response {
    let status =
        StatusCode::from_u16(err.http_status()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    tracing::info!(
        instance = %instance_id, %handler, kind = err.kind(), %status,
        "control_plane: leader returned control error"
    );
    json_response(
        status,
        serde_json::json!({
            "error": err.to_string(),
            "kind": err.kind(),
        }),
    )
}

pub(crate) fn service_unavailable(msg: &str) -> Response {
    error_response(StatusCode::SERVICE_UNAVAILABLE, msg)
}

pub(crate) fn error_response(status: StatusCode, msg: &str) -> Response {
    let code = match status {
        StatusCode::NOT_FOUND => ErrorCode::NotFound,
        StatusCode::BAD_REQUEST => ErrorCode::BadRequest,
        StatusCode::SERVICE_UNAVAILABLE => ErrorCode::Internal,
        _ => ErrorCode::Internal,
    };
    let body = ErrorBody {
        code,
        message: msg.to_owned(),
    };
    json_response(
        status,
        serde_json::to_value(body)
            .unwrap_or_else(|_| serde_json::json!({ "code": "internal", "message": msg })),
    )
}

pub(crate) fn json_response(status: StatusCode, value: serde_json::Value) -> Response {
    let body = serde_json::to_vec(&value).unwrap_or_else(|_| b"{}".to_vec());
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    (status, headers, axum::body::Bytes::from(body)).into_response()
}
