// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Axum middleware gating inference routes behind JWT auth.
//!
//! Only layered onto `inference_router` when `DYN_AUTH_ENABLED=true` (see
//! `service_v2.rs`) — system routes (health/live/metrics/models) never pass
//! through this middleware, and it adds zero overhead when auth is disabled
//! since the layer itself isn't added to the router in that case.

use std::sync::Arc;

use axum::extract::Request;
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::auth;
use crate::http::service::{metadata, service_v2};

/// Request header carrying the JWT-verified org UUID down to handlers, via
/// the same `x-dynamo-meta-*` → `Context::metadata()` propagation path
/// already used for other request metadata.
fn org_uuid_header_name() -> String {
    format!("{}org-uuid", metadata::metadata_header_prefix())
}

pub async fn auth_middleware(
    axum::extract::State(state): axum::extract::State<Arc<service_v2::State>>,
    mut request: Request,
    next: Next,
) -> Response {
    let auth_header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string();

    let auth_config = state.auth_config();
    match auth::authenticate(&auth_header, &auth_config.secret_keys, &auth_config.valid_orgs) {
        Ok(ctx) => {
            let header_name = org_uuid_header_name();
            let headers = request.headers_mut();
            // Strip any client-supplied value first — the verified org_uuid
            // must be the only one `extract_metadata_from_http` ever sees.
            headers.remove(header_name.as_str());
            if let Ok(value) = HeaderValue::from_str(&ctx.org_uuid) {
                headers.insert(
                    axum::http::HeaderName::from_bytes(header_name.as_bytes())
                        .expect("metadata header prefix + 'org-uuid' is always a valid header name"),
                    value,
                );
            }
            next.run(request).await
        }
        Err(err) => err.into_response(),
    }
}
