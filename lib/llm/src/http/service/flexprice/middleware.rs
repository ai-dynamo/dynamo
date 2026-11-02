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
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::auth::{self, AuthError};
use super::balance::BalanceStatus;
use crate::http::service::service_v2;

/// The JWT-verified org UUID. Inserted into the request's typed extension map
/// by [`auth_middleware`] on successful auth; billed handlers read it back
/// out via `Option<axum::extract::Extension<OrgUuid>>` — `None` whenever auth
/// is disabled or this particular request never passed through the
/// middleware, so handlers stay a no-op-safe shell in that case.
#[derive(Debug, Clone)]
pub struct OrgUuid(pub String);

/// The JWT-verified user UUID (the `user_uuid` claim). Inserted alongside
/// [`OrgUuid`] so billed handlers can attribute usage events to the actual
/// user, not just the org. Same `None`-when-unauthenticated semantics.
#[derive(Debug, Clone)]
pub struct UserUuid(pub String);

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
    let ctx = match auth::authenticate(
        &auth_header,
        &auth_config.secret_keys,
        &auth_config.valid_orgs,
    ) {
        Ok(ctx) => ctx,
        Err(err) => return err.into_response(),
    };

    // Temporary: remove when AI gateway ships
    // Asked after the org allowlist, so requests rejected there never cost a revoke lookup.
    if let Some(checker) = state.revoke_checker()
        && checker.revoked(&ctx.token_uuid).await
    {
        tracing::warn!(org = %ctx.org_uuid, token_uuid = %ctx.token_uuid, "Unauthorized access: revoked token");
        return AuthError {
            status: StatusCode::UNAUTHORIZED,
            message: "token revoked".to_string(),
        }
        .into_response();
    }

    // Wallet balance gate: prepaid orgs below the configured minimum balance
    // (negative by default) are blocked; postpaid orgs bypass this entirely.
    // Only runs when FlexPrice billing is enabled — no wallet, no gate.
    // `check` never awaits a live FlexPrice call (see balance.rs) — it's
    // cache-only-or-fail-open, so this never adds network latency here.
    if let Some(checker) = state.flexprice_balance_checker() {
        match checker.check(&ctx.org_uuid).await {
            BalanceStatus::Ok => {}
            BalanceStatus::Suspended => {
                return AuthError {
                    status: StatusCode::PAYMENT_REQUIRED,
                    message: "Your account has been suspended. Please contact support to re-activate your account.".to_string(),
                }
                .into_response();
            }
            BalanceStatus::InsufficientBalance => {
                return AuthError {
                    status: StatusCode::PAYMENT_REQUIRED,
                    message: "You have exhausted your wallet balance. Please add credits to resume using.".to_string(),
                }
                .into_response();
            }
        }
    }

    // Request extensions are a server-internal typed map, never populated
    // from client input, so there's no spoofing vector to guard against here
    // (unlike a header-based propagation scheme).
    request.extensions_mut().insert(OrgUuid(ctx.org_uuid));
    request.extensions_mut().insert(UserUuid(ctx.user_uuid));
    next.run(request).await
}
