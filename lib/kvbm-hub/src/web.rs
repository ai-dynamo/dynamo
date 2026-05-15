// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Embedded single-page UI served from the hub's control port.
//!
//! Each asset under `lib/kvbm-hub/web/` is baked into the binary via
//! [`include_str!`] / [`include_bytes!`]; the matching route writes the
//! bytes back with the correct `Content-Type`. Same-origin against the
//! hub's control listener — no CORS, no extra port, no build step.
//!
//! The UI consumes hub HTTP endpoints (`GET /v1/instances`,
//! `GET /v1/instances/{id}/modules`, `GET /v1/instances/{id}/describe`,
//! the `POST /v1/instances/{id}/control/<module>/<handler>` namespace)
//! purely from the browser — Alpine.js holds the entire state machine.

use axum::Router;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header::CONTENT_TYPE};
use axum::response::IntoResponse;
use axum::routing::get;

// ----- Static asset blobs -----

const INDEX_HTML: &str = include_str!("../web/index.html");
const APP_JS: &str = include_str!("../web/app.js");
const STYLE_CSS: &str = include_str!("../web/style.css");
const TOKENS_CSS: &str = include_str!("../web/tokens.css");
const ALPINE_JS: &str = include_str!("../web/vendor/alpine.min.js");

// ----- Router -----

/// Routes for the embedded UI. Mount onto the hub's control router.
pub fn ui_router() -> Router {
    Router::new()
        .route("/", get(serve_index))
        .route("/app.js", get(serve_app_js))
        .route("/style.css", get(serve_style_css))
        .route("/tokens.css", get(serve_tokens_css))
        .route("/vendor/alpine.min.js", get(serve_alpine_js))
}

// ----- Handlers -----

async fn serve_index() -> impl IntoResponse {
    text_response(INDEX_HTML, "text/html; charset=utf-8")
}

async fn serve_app_js() -> impl IntoResponse {
    text_response(APP_JS, "application/javascript; charset=utf-8")
}

async fn serve_style_css() -> impl IntoResponse {
    text_response(STYLE_CSS, "text/css; charset=utf-8")
}

async fn serve_tokens_css() -> impl IntoResponse {
    text_response(TOKENS_CSS, "text/css; charset=utf-8")
}

async fn serve_alpine_js() -> impl IntoResponse {
    text_response(ALPINE_JS, "application/javascript; charset=utf-8")
}

fn text_response(
    body: &'static str,
    content_type: &'static str,
) -> (StatusCode, HeaderMap, &'static str) {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    // Cache embedded assets aggressively — they change only with a hub
    // rebuild, and the browser will get fresh bytes on the next restart.
    headers.insert(
        axum::http::header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=300"),
    );
    (StatusCode::OK, headers, body)
}
