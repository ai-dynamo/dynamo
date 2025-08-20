// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::HashSet,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Error;
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use chrono::format::format;
use dynamo_runtime::{
    pipeline::{AsyncEngineContextProvider, Context},
    protocols::annotated::AnnotationsProvider,
};
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};

use crate::preprocessor::LLMMetricAnnotation;
use crate::protocols::openai::{
    chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionResponse},
    completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    embeddings::{NvCreateEmbeddingRequest, NvCreateEmbeddingResponse},
    responses::{NvCreateResponse, NvResponse},
};
use crate::request_template::RequestTemplate;
use crate::types::Annotated;

use super::kserve;
// [gluo NOTE] These are common utilities that should be shared between frontends
use crate::http::service::{
    metrics::{Endpoint, ResponseMetricCollector},
    disconnect::{
        create_connection_monitor, monitor_for_disconnects, ConnectionHandle,
    },
};

use tonic::Status;

/// Dynamo Annotation for the request ID
pub const ANNOTATION_REQUEST_ID: &str = "request_id";

// [gluo NOTE] strip down version of lib/llm/src/http/service/openai.rs
// dupliating it here as the original file has coupling with HTTP objects.

/// Get the request ID from a primary source, or lastly create a new one if not present
fn get_or_create_request_id(primary: Option<&str>) -> String {
    // Try to get the request ID from the primary source
    if let Some(primary) = primary {
        if let Ok(uuid) = uuid::Uuid::parse_str(primary) {
            return uuid.to_string();
        }
    }

    // Try to parse the request ID as a UUID, or generate a new one if missing/invalid
    let uuid = uuid::Uuid::new_v4();
    uuid.to_string()
}

/// OpenAI Completions Request Handler
///
/// This method will handle the incoming request for the `/v1/completions endpoint`. The endpoint is a "source"
/// for an [`super::OpenAICompletionsStreamingEngine`] and will return a stream of
/// responses which will be forward to the client.
///
/// Note: For all requests, streaming or non-streaming, we always call the engine with streaming enabled. For
/// non-streaming requests, we will fold the stream into a single response as part of this handler.
pub async fn model_infer_completions(
    state: Arc<kserve::State>,
    request: NvCreateCompletionRequest,
) -> Result<NvCreateCompletionResponse, Status> {
    // create the context for the request
    // [WIP] from request id.
    let request_id = get_or_create_request_id(request.inner.user.as_deref());
    let request = Context::with_id(request, request_id.clone());
    let context = request.context();

    let streaming = request.inner.stream.unwrap_or(false);

    // create the connection handles
    let (mut connection_handle, _) = create_connection_monitor(context.clone()).await;

    // update the request to always stream
    let request = request.map(|mut req| {
        req.inner.stream = Some(true);
        req
    });

    // todo - make the protocols be optional for model name
    // todo - when optional, if none, apply a default
    let model = &request.inner.model;

    // todo - error handling should be more robust
    let engine = state
        .manager()
        .get_completions_engine(model)
        .map_err(|_| Status::not_found("model not found"))?;

    let mut inflight_guard =
        state
            .metrics_clone()
            .create_inflight_guard(model, Endpoint::Completions, streaming);

    let mut response_collector = state.metrics_clone().create_response_collector(model);

    // prepare to process any annotations
    let annotations = request.annotations();

    // issue the generate call on the engine
    let stream = engine
        .generate(request)
        .await
        .map_err(|e| Status::internal(format!("Failed to generate completions: {}", e)))?;

    // capture the context to cancel the stream if the client disconnects
    let ctx = stream.context();

    // [gluo TODO] annotations complicate the stream processing, will revisit if actually needed
    // let annotations = annotations.map_or(Vec::new(), |annotations| {
    //     annotations
    //         .iter()
    //         .filter_map(|annotation| {
    //             if annotation == ANNOTATION_REQUEST_ID {
    //                 Annotated::<NvCreateCompletionResponse>::from_annotation(
    //                     ANNOTATION_REQUEST_ID,
    //                     &request_id,
    //                 )
    //                 .ok()
    //             } else {
    //                 None
    //             }
    //         })
    //         .collect::<Vec<_>>()
    // });

    // // apply any annotations to the front of the stream
    // let stream = stream::iter(annotations).chain(stream);

    if streaming {
        // [gluo FIXME] Can't use 'monitor_for_disconnects' as is, contain SSE logic
        // let stream = monitor_for_disconnects(stream, ctx, inflight_guard, connection_handle);
        // return stream.map(|response| {response});
        unimplemented!("Streaming completions not implemented yet");
    } else {
        // TODO: report ISL/OSL for non-streaming requests
        let response = NvCreateCompletionResponse::from_annotated_stream(stream)
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to fold completions stream for {}: {:?}",
                    request_id,
                    e
                );
                Status::internal("Failed to fold completions stream")
            })?;

        inflight_guard.mark_ok();
        // if we got here, then we will return a response and the potentially long running task has completed successfully
        // without need to be cancelled.
        connection_handle.disarm();
        return Ok(response);
    }
}

// #[tracing::instrument(skip_all)]
// async fn completions(
//     state: Arc<service_v2::State>,
//     request: Context<NvCreateCompletionRequest>,
//     stream_handle: ConnectionHandle,
// ) -> Result<Response, ErrorResponse> {
//     // return a 503 if the service is not ready
//     check_ready(&state)?;

//     // todo - extract distributed tracing id and context id from headers
//     let request_id = request.id().to_string();

//     // todo - decide on default
//     let streaming = request.inner.stream.unwrap_or(false);

//     // update the request to always stream
//     let request = request.map(|mut req| {
//         req.inner.stream = Some(true);
//         req
//     });

//     // todo - make the protocols be optional for model name
//     // todo - when optional, if none, apply a default
//     let model = &request.inner.model;

//     // todo - error handling should be more robust
//     let engine = state
//         .manager()
//         .get_completions_engine(model)
//         .map_err(|_| ErrorMessage::model_not_found())?;

//     let mut inflight_guard =
//         state
//             .metrics_clone()
//             .create_inflight_guard(model, Endpoint::Completions, streaming);

//     let mut response_collector = state.metrics_clone().create_response_collector(model);

//     // prepare to process any annotations
//     let annotations = request.annotations();

//     // issue the generate call on the engine
//     let stream = engine
//         .generate(request)
//         .await
//         .map_err(|e| ErrorMessage::from_anyhow(e, "Failed to generate completions"))?;

//     // capture the context to cancel the stream if the client disconnects
//     let ctx = stream.context();

//     let annotations = annotations.map_or(Vec::new(), |annotations| {
//         annotations
//             .iter()
//             .filter_map(|annotation| {
//                 if annotation == ANNOTATION_REQUEST_ID {
//                     Annotated::<NvCreateCompletionResponse>::from_annotation(
//                         ANNOTATION_REQUEST_ID,
//                         &request_id,
//                     )
//                     .ok()
//                 } else {
//                     None
//                 }
//             })
//             .collect::<Vec<_>>()
//     });

//     // apply any annotations to the front of the stream
//     let stream = stream::iter(annotations).chain(stream);

//     if streaming {
//         let stream = stream.map(move |response| {
//             process_event_converter(EventConverter::from(response), &mut response_collector)
//         });
//         let stream = monitor_for_disconnects(stream, ctx, inflight_guard, stream_handle);

//         let mut sse_stream = Sse::new(stream);

//         if let Some(keep_alive) = state.sse_keep_alive() {
//             sse_stream = sse_stream.keep_alive(KeepAlive::default().interval(keep_alive));
//         }

//         Ok(sse_stream.into_response())
//     } else {
//         // TODO: report ISL/OSL for non-streaming requests
//         let response = NvCreateCompletionResponse::from_annotated_stream(stream)
//             .await
//             .map_err(|e| {
//                 tracing::error!(
//                     "Failed to fold completions stream for {}: {:?}",
//                     request_id,
//                     e
//                 );
//                 ErrorMessage::internal_server_error("Failed to fold completions stream")
//             })?;

//         inflight_guard.mark_ok();
//         Ok(Json(response).into_response())
//     }
// }

/// openai compatible format
/// Example:
/// {
///  "object": "list",
///  "data": [
///    {
///      "id": "model-id-0",
///      "object": "model",
///      "created": 1686935002,
///      "owned_by": "organization-owner"
///    },
///    ]
/// }
async fn list_models_openai(
    State(state): State<Arc<kserve::State>>,
) -> Result<Response, Error> {

    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let mut data = Vec::new();

    let models: HashSet<String> = state.manager().model_display_names();
    for model_name in models {
        data.push(ModelListing {
            id: model_name.clone(),
            object: "object",
            created,                        // Where would this come from? The GGUF?
            owned_by: "nvidia".to_string(), // Get organization from GGUF
        });
    }

    let out = ListModelOpenAI {
        object: "list",
        data,
    };
    Ok(Json(out).into_response())
}

#[derive(Serialize)]
struct ListModelOpenAI {
    object: &'static str, // always "list"
    data: Vec<ModelListing>,
}

#[derive(Serialize)]
struct ModelListing {
    id: String,
    object: &'static str, // always "object"
    created: u64,         //  Seconds since epoch
    owned_by: String,
}

struct EventConverter<T>(Annotated<T>);

impl<T> From<Annotated<T>> for EventConverter<T> {
    fn from(annotated: Annotated<T>) -> Self {
        EventConverter(annotated)
    }
}

fn process_event_converter<T: Serialize>(
    annotated: EventConverter<T>,
    response_collector: &mut ResponseMetricCollector,
) -> Result<Event, axum::Error> {
    let mut annotated = annotated.0;

    // update metrics
    if let Ok(Some(metrics)) = LLMMetricAnnotation::from_annotation(&annotated) {
        response_collector.observe_current_osl(metrics.output_tokens);
        response_collector.observe_response(metrics.input_tokens, metrics.chunk_tokens);

        // Chomp the LLMMetricAnnotation so it's not returned in the response stream
        // TODO: add a flag to control what is returned in the SSE stream
        if annotated.event.as_deref() == Some(crate::preprocessor::ANNOTATION_LLM_METRICS) {
            annotated.event = None;
            annotated.comment = None;
        }
    }

    let mut event = Event::default();

    if let Some(data) = annotated.data {
        event = event.json_data(data)?;
    }

    if let Some(msg) = annotated.event {
        if msg == "error" {
            let msgs = annotated
                .comment
                .unwrap_or_else(|| vec!["unspecified error".to_string()]);
            return Err(axum::Error::new(msgs.join(" -- ")));
        }
        event = event.event(msg);
    }

    if let Some(comments) = annotated.comment {
        for comment in comments {
            event = event.comment(comment);
        }
    }

    Ok(event)
}
