// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::post,
    Json, Router,
};
use futures::StreamExt;
use std::sync::Arc;

use super::{
    metrics::Endpoint,
    openai::{monitor_for_disconnects, process_event_converter, ErrorResponse},
    service_v2, RouteDoc,
};
use crate::protocols::{
    openai::completions::CompletionResponse, token_completions::DynamoTokenCompletionRequest,
};
use dynamo_runtime::pipeline::Context;

/// Token completion endpoint - goes directly to backend, bypassing OpenAI style preprocessing
#[tracing::instrument(skip_all)]
async fn token_completions(
    State(state): State<Arc<service_v2::State>>,
    Json(request): Json<DynamoTokenCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let model = request.model.clone();
    let request_id = uuid::Uuid::new_v4().to_string();
    let streaming = request.stream.unwrap_or(false);

    let engine = state
        .manager()
        .get_token_completion_engine(&model)
        .map_err(|_| ErrorResponse::model_not_found())?;

    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(&model, Endpoint::Completions, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(&model);

    let request_context = Context::with_id(request, request_id);
    let stream = engine
        .generate(request_context)
        .await
        .map_err(|e| ErrorResponse::from_anyhow(e, "Generation failed"))?;

    // Capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    if streaming {
        let stream = stream.map(move |response| {
            process_event_converter(
                super::openai::EventConverter::from(response),
                &mut response_collector,
            )
        });

        // Add monitor_for_disconnects to send [DONE] event - same as OpenAI completions
        let stream = monitor_for_disconnects(stream.boxed(), ctx, inflight_guard).await;

        let mut sse_stream = Sse::new(stream);
        if let Some(keep_alive) = state.sse_keep_alive() {
            sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
        }

        Ok(sse_stream.into_response())
    } else {
        let response = CompletionResponse::from_annotated_stream(stream.into())
            .await
            .map_err(|e| ErrorResponse::from_anyhow(e, "Failed to process response"))?;

        inflight_guard.mark_ok();
        Ok(Json(response).into_response())
    }
}

pub fn token_completions_router(
    state: Arc<service_v2::State>,
    path: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path.unwrap_or("/v1/experimental/dynamo/completions".to_string());
    let doc = RouteDoc::new(axum::http::Method::POST, &path);
    let router = Router::new()
        .route(&path, post(token_completions))
        .with_state(state);
    (vec![doc], router)
}
