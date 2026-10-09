// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::{RouteDoc, metrics::ErrorType};
use super::*;
use axum::{
    Router,
    body::Bytes,
    extract::{State, rejection::BytesRejection},
    http::HeaderMap,
    routing::post,
};
use dynamo_decisions::{Route, parse_request};

const BODY_LIMIT_BYTES: usize = 4 * 1024 * 1024;

pub fn router(state: Arc<service_v2::State>, path: Option<String>) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or_else(|| DEFAULT_PATH.to_string());
    (
        vec![
            RouteDoc::new(axum::http::Method::POST, &path).with_documentation_path(DEFAULT_PATH),
            RouteDoc::new(axum::http::Method::POST, DECISIONS_PATH),
        ],
        Router::new()
            .route(&path, post(handler))
            .route(DECISIONS_PATH, post(decisions_handler))
            .layer(axum::extract::DefaultBodyLimit::max(BODY_LIMIT_BYTES))
            .with_state(state),
    )
}

async fn handler(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    handle_http(state, headers, body, Route::SystemOne).await
}

async fn decisions_handler(
    State(state): State<Arc<service_v2::State>>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    handle_http(state, headers, body, Route::Decisions).await
}

async fn handle_http(
    state: Arc<service_v2::State>,
    headers: HeaderMap,
    body: Result<Bytes, BytesRejection>,
    route: Route,
) -> Response {
    let request_id = super::super::openai::get_or_create_request_id(&headers);
    let default_dialect = if route == Route::SystemOne {
        Dialect::Jev
    } else {
        Dialect::OpenAi
    };
    let parsed = match body {
        Ok(body) => parse_request(&body, route).map_err(decision_error),
        Err(rejection) => Err(error(
            default_dialect,
            rejection.status(),
            "request body could not be read",
        )),
    };
    let request = match parsed {
        Ok(request) => request,
        Err(mut response) => {
            attach_request_ids(&mut response, &request_id);
            return response;
        }
    };
    if headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_none_or(|value| {
            !value
                .split(';')
                .next()
                .is_some_and(|mime| mime.trim().eq_ignore_ascii_case("application/json"))
        })
    {
        let mut response = error(
            request.dialect,
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "content-type must be application/json",
        );
        attach_request_ids(&mut response, &request_id);
        return response;
    }
    let requested_model = request.model.as_str();
    let endpoint = metric_endpoint(request.dialect);
    let metric_model = state
        .manager()
        .metric_model_for(requested_model)
        .to_string();
    let mut guard =
        state
            .metrics_clone()
            .create_inflight_guard(&metric_model, endpoint, false, &request_id);
    guard.mark_error(ErrorType::Cancelled);
    let mut response = handle_request(state.clone(), request_id.clone(), request).await;
    attach_request_ids(&mut response, &request_id);
    match response.status().as_u16() {
        status
            if status == StatusCode::TOO_MANY_REQUESTS.as_u16()
                && response
                    .headers()
                    .contains_key(axum::http::header::RETRY_AFTER) =>
        {
            state.metrics_clone().inc_rejection(&metric_model, endpoint);
            guard.mark_error(ErrorType::Overload);
        }
        200..=299 => guard.mark_ok(),
        400 | 413 | 415 | 422 => guard.mark_error(ErrorType::Validation),
        404 => guard.mark_error(ErrorType::NotFound),
        499 => guard.mark_error(ErrorType::Cancelled),
        503 | 504 => guard.mark_error(ErrorType::Unavailable),
        _ => guard.mark_error(ErrorType::Internal),
    }
    response
}
