// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy `ExternalProcessor.Process` bidirectional streaming implementation.
//!
//! Port of the Go EPP `StreamingServer.Process` from
//! `gateway-api-inference-extension/pkg/epp/handlers/server.go`.
//!
//! The state machine enforces ordered responses:
//! `RequestHeaders → RequestBody → RequestTrailers → ResponseHeaders → ResponseBody → ResponseTrailers`

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::mpsc;
use tokio_stream::{Stream, StreamExt, wrappers::ReceiverStream};
use tonic::{Request, Response, Status, Streaming};

use crate::envoy_helpers::{self, metadata};
use crate::proto::envoy::service::ext_proc::v3::{
    self as ext_proc,
    external_processor_server::{ExternalProcessor, ExternalProcessorServer},
    processing_request, ProcessingRequest, ProcessingResponse,
};
use crate::proto::envoy::r#type::v3::StatusCode;
use crate::router::Router;

/// State machine phases for the ext_proc stream, matching the Go implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    RequestReceived,
    HeaderRequestResponseComplete,
    BodyRequestResponsesComplete,
    TrailerRequestResponsesComplete,
    ResponseReceived,
    HeaderResponseResponseComplete,
    BodyResponseResponsesComplete,
    TrailerResponseResponsesComplete,
    RequestEvicted,
}

/// Per-request context carried across the lifetime of one HTTP stream.
struct RequestContext {
    state: StreamState,
    target_endpoint: String,
    incoming_model_name: String,
    target_model_name: String,
    request_id: String,
    request_received_at: Option<Instant>,
    request_size: usize,
    response_size: usize,
    response_complete: bool,
    model_server_streaming: bool,

    request_headers: HashMap<String, String>,
    response_headers: HashMap<String, String>,

    req_header_resp: Option<ProcessingResponse>,
    req_body_resp: Vec<ProcessingResponse>,
    req_trailer_resp: Option<ProcessingResponse>,

    resp_header_resp: Option<ProcessingResponse>,
    resp_body_resp: Vec<ProcessingResponse>,
    resp_trailer_resp: Option<ProcessingResponse>,
}

impl RequestContext {
    fn new() -> Self {
        Self {
            state: StreamState::RequestReceived,
            target_endpoint: String::new(),
            incoming_model_name: String::new(),
            target_model_name: String::new(),
            request_id: String::new(),
            request_received_at: None,
            request_size: 0,
            response_size: 0,
            response_complete: false,
            model_server_streaming: false,
            request_headers: HashMap::new(),
            response_headers: HashMap::new(),
            req_header_resp: None,
            req_body_resp: Vec::new(),
            req_trailer_resp: None,
            resp_header_resp: None,
            resp_body_resp: Vec::new(),
            resp_trailer_resp: None,
        }
    }

    /// Advance the state machine and collect responses that are ready to send.
    /// Mirrors Go `updateStateAndSendIfNeeded`.
    fn drain_pending_responses(&mut self) -> Vec<ProcessingResponse> {
        let mut out = Vec::new();

        if self.state == StreamState::RequestEvicted {
            out.push(envoy_helpers::build_eviction_response());
            return out;
        }

        if self.state == StreamState::RequestReceived {
            if let Some(resp) = self.req_header_resp.take() {
                out.push(resp);
                self.state = StreamState::HeaderRequestResponseComplete;
            }
        }

        if self.state == StreamState::HeaderRequestResponseComplete
            && !self.req_body_resp.is_empty()
        {
            out.append(&mut self.req_body_resp);
            self.state = StreamState::BodyRequestResponsesComplete;
        }

        if self.state == StreamState::BodyRequestResponsesComplete {
            if let Some(resp) = self.req_trailer_resp.take() {
                out.push(resp);
                self.state = StreamState::TrailerRequestResponsesComplete;
            }
        }

        if self.state == StreamState::ResponseReceived {
            if let Some(resp) = self.resp_header_resp.take() {
                out.push(resp);
                self.state = StreamState::HeaderResponseResponseComplete;
            }
        }

        if self.state == StreamState::HeaderResponseResponseComplete {
            out.append(&mut self.resp_body_resp);
            if self.response_complete {
                self.state = StreamState::BodyResponseResponsesComplete;
            }
        }

        if self.state == StreamState::BodyResponseResponsesComplete {
            if let Some(resp) = self.resp_trailer_resp.take() {
                out.push(resp);
                self.state = StreamState::TrailerResponseResponsesComplete;
            }
        }

        out
    }
}

/// The ext_proc gRPC server backed by Dynamo's KV-aware router.
pub struct ExtProcServer {
    router: Arc<Router>,
}

impl ExtProcServer {
    pub fn new(router: Arc<Router>) -> Self {
        Self { router }
    }

    /// Create a `tonic` service ready for registration on a gRPC server.
    pub fn into_service(self) -> ExternalProcessorServer<Self> {
        ExternalProcessorServer::new(self)
    }

    /// Handle request headers phase.
    fn handle_request_headers(
        &self,
        ctx: &mut RequestContext,
        hdr: &ext_proc::HttpHeaders,
    ) {
        ctx.request_received_at = Some(Instant::now());

        let headers = hdr.headers.as_ref();

        // If end_of_stream in request headers, this is a body-less request (e.g. GET).
        // We skip routing (no body to parse) and just pass through.
        if hdr.end_of_stream {
            // No body to route — generate a pass-through header response.
            // The target_endpoint will be empty, signaling no routing was done.
            ctx.req_header_resp = Some(envoy_helpers::build_request_header_response(
                &ctx.target_endpoint,
                None,
                &ctx.request_headers,
            ));
            return;
        }

        if let Some(header_map) = headers {
            ctx.request_headers = envoy_helpers::collect_headers(header_map);

            // Extract request ID
            if let Some(id) = envoy_helpers::extract_header_value(
                header_map,
                metadata::REQUEST_ID_HEADER_KEY,
            ) {
                if !id.is_empty() {
                    ctx.request_id = id;
                }
            }
        }

        if ctx.request_id.is_empty() {
            ctx.request_id = uuid::Uuid::new_v4().to_string();
            ctx.request_headers.insert(
                metadata::REQUEST_ID_HEADER_KEY.to_string(),
                ctx.request_id.clone(),
            );
        }
    }

    /// Handle request body phase: parse JSON, tokenize, route.
    async fn handle_request_body(
        &self,
        ctx: &mut RequestContext,
        raw_body: &[u8],
    ) -> Result<(), ExtProcError> {
        ctx.request_size = raw_body.len();

        let body_str = std::str::from_utf8(raw_body)
            .map_err(|e| ExtProcError::bad_request(format!("Invalid UTF-8 in request body: {e}")))?;

        // Tokenize the request using the Dynamo preprocessor
        let tokens = self
            .router
            .tokenize(body_str)
            .map_err(|e| ExtProcError::bad_request(format!("Failed to tokenize request: {e}")))?;

        // Route the decode request
        let (decode_worker, _overlap) = self
            .router
            .route_decode(&tokens, false, None)
            .await
            .map_err(|e| {
                ExtProcError::service_unavailable(format!("Routing failed: {e}"))
            })?;

        // Set the target endpoint. In the real deployment this maps worker_id → pod IP:port.
        // For now, we store the worker_id as the target and let the caller/Envoy config
        // map it via metadata. The Go EPP gets the endpoint from the datastore.
        ctx.target_endpoint = format!("worker-{}", decode_worker.worker_id);
        ctx.target_model_name = ctx.incoming_model_name.clone();

        tracing::info!(
            request_id = %ctx.request_id,
            decode_worker_id = decode_worker.worker_id,
            decode_dp_rank = decode_worker.dp_rank,
            token_count = tokens.len(),
            "Routed request"
        );

        // Register the request for bookkeeping
        if let Err(e) = self
            .router
            .add_request(
                &ctx.request_id,
                &tokens,
                decode_worker.worker_id,
                decode_worker.dp_rank,
            )
            .await
        {
            tracing::warn!(error = %e, "Failed to register request with router");
        }

        ctx.req_header_resp = Some(envoy_helpers::build_request_header_response(
            &ctx.target_endpoint,
            Some(ctx.request_size),
            &ctx.request_headers,
        ));
        ctx.req_body_resp = envoy_helpers::build_request_body_responses(raw_body);

        Ok(())
    }

    /// Handle response headers from the upstream model server.
    fn handle_response_headers(
        &self,
        ctx: &mut RequestContext,
        hdr: &ext_proc::HttpHeaders,
    ) {
        if let Some(header_map) = &hdr.headers {
            for h in &header_map.headers {
                let value = envoy_helpers::get_header_value(h);
                if h.key == "status" && value != "200" {
                    tracing::warn!(status = %value, "Model server returned non-200");
                } else if h.key == "content-type" && value.contains("text/event-stream") {
                    ctx.model_server_streaming = true;
                }
                ctx.response_headers.insert(h.key.clone(), value);
            }
        }

        ctx.state = StreamState::ResponseReceived;
        ctx.resp_header_resp = Some(envoy_helpers::build_response_header_response());
    }

    /// Handle response body from the upstream model server.
    fn handle_response_body(
        &self,
        ctx: &mut RequestContext,
        body: &ext_proc::HttpBody,
    ) {
        let end_of_stream = body.end_of_stream;
        let chunk = &body.body;
        ctx.response_size += chunk.len();

        if ctx.model_server_streaming {
            if end_of_stream {
                ctx.response_complete = true;
            }
            let rewritten =
                envoy_helpers::rewrite_model_name(chunk, &ctx.target_model_name, &ctx.incoming_model_name);
            ctx.resp_body_resp = envoy_helpers::build_response_body_responses(
                &rewritten,
                end_of_stream,
                None,
            );
        } else {
            // Non-streaming: accumulate (handled as single chunk in practice)
            if end_of_stream {
                ctx.response_complete = true;
                let rewritten = envoy_helpers::rewrite_model_name(
                    chunk,
                    &ctx.target_model_name,
                    &ctx.incoming_model_name,
                );
                ctx.resp_body_resp = envoy_helpers::build_response_body_responses(
                    &rewritten,
                    true,
                    None,
                );
            }
        }
    }

    /// Clean up when stream ends.
    async fn cleanup(&self, ctx: &RequestContext) {
        if !ctx.request_id.is_empty() {
            if let Err(e) = self.router.free_request(&ctx.request_id).await {
                tracing::warn!(
                    request_id = %ctx.request_id,
                    error = %e,
                    "Failed to free request on cleanup"
                );
            }
        }
    }
}

#[tonic::async_trait]
impl ExternalProcessor for ExtProcServer {
    type ProcessStream =
        Pin<Box<dyn Stream<Item = Result<ProcessingResponse, Status>> + Send + 'static>>;

    async fn process(
        &self,
        request: Request<Streaming<ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        let mut inbound = request.into_inner();
        let router = self.router.clone();

        // Channel for sending responses back to Envoy
        let (tx, rx) = mpsc::channel::<Result<ProcessingResponse, Status>>(32);
        let output_stream = ReceiverStream::new(rx);

        tokio::spawn(async move {
            let server = ExtProcServer {
                router: router.clone(),
            };
            let mut ctx = RequestContext::new();
            let mut body_buf: Vec<u8> = Vec::new();
            let mut resp_body_buf: Vec<u8> = Vec::new();

            let result: Result<(), Status> = async {
                while let Some(req_result) = inbound.next().await {
                    let req = req_result.map_err(|e| {
                        Status::unknown(format!("Cannot receive stream request: {e}"))
                    })?;

                    match req.request {
                        Some(processing_request::Request::RequestHeaders(ref hdr)) => {
                            tracing::debug!(request_id = %ctx.request_id, "RequestHeaders received");
                            server.handle_request_headers(&mut ctx, hdr);
                        }
                        Some(processing_request::Request::RequestBody(ref body)) => {
                            tracing::debug!(
                                request_id = %ctx.request_id,
                                eos = body.end_of_stream,
                                "RequestBody chunk received"
                            );
                            body_buf.extend_from_slice(&body.body);

                            if body.end_of_stream {
                                let raw_body = std::mem::take(&mut body_buf);
                                if let Err(e) =
                                    server.handle_request_body(&mut ctx, &raw_body).await
                                {
                                    let resp = e.into_processing_response();
                                    let _ = tx.send(Ok(resp)).await;
                                    return Ok(());
                                }
                            }
                        }
                        Some(processing_request::Request::RequestTrailers(_)) => {
                            // Request trailers are currently unused
                        }
                        Some(processing_request::Request::ResponseHeaders(ref hdr)) => {
                            tracing::debug!(request_id = %ctx.request_id, "ResponseHeaders received");
                            server.handle_response_headers(&mut ctx, hdr);
                        }
                        Some(processing_request::Request::ResponseBody(ref body)) => {
                            if ctx.model_server_streaming {
                                server.handle_response_body(&mut ctx, body);
                            } else {
                                resp_body_buf.extend_from_slice(&body.body);
                                if body.end_of_stream {
                                    let full_body = std::mem::take(&mut resp_body_buf);
                                    let synthetic = ext_proc::HttpBody {
                                        body: full_body,
                                        end_of_stream: true,
                                        ..Default::default()
                                    };
                                    server.handle_response_body(&mut ctx, &synthetic);
                                }
                            }
                        }
                        Some(processing_request::Request::ResponseTrailers(_)) => {
                            if !ctx.response_complete {
                                ctx.response_complete = true;
                                if !resp_body_buf.is_empty() {
                                    let full_body = std::mem::take(&mut resp_body_buf);
                                    let synthetic = ext_proc::HttpBody {
                                        body: full_body,
                                        end_of_stream: true,
                                        ..Default::default()
                                    };
                                    server.handle_response_body(&mut ctx, &synthetic);
                                }
                            }
                            ctx.resp_trailer_resp =
                                Some(envoy_helpers::build_response_trailer_response());
                        }
                        None => {
                            tracing::warn!("Received ProcessingRequest with no request variant");
                        }
                    }

                    let responses = ctx.drain_pending_responses();
                    for resp in responses {
                        if tx.send(Ok(resp)).await.is_err() {
                            return Ok(());
                        }
                    }

                    if ctx.state == StreamState::RequestEvicted {
                        break;
                    }
                }

                Ok(())
            }
            .await;

            if let Err(e) = result {
                let _ = tx.send(Err(e)).await;
            }

            server.cleanup(&ctx).await;
        });

        Ok(Response::new(Box::pin(output_stream)))
    }
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors during ext_proc request processing, mapped to Envoy ImmediateResponse.
struct ExtProcError {
    status_code: StatusCode,
    message: String,
}

impl ExtProcError {
    fn bad_request(msg: String) -> Self {
        Self {
            status_code: StatusCode::BadRequest,
            message: msg,
        }
    }

    fn service_unavailable(msg: String) -> Self {
        Self {
            status_code: StatusCode::ServiceUnavailable,
            message: msg,
        }
    }

    fn into_processing_response(self) -> ProcessingResponse {
        envoy_helpers::build_error_response(self.status_code, Some(&self.message))
    }
}

/// Start the ext_proc gRPC server on the given port.
pub async fn run_ext_proc_server(
    router: Arc<Router>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{port}").parse()?;
    let server = ExtProcServer::new(router);

    tracing::info!(%addr, "Starting ext_proc gRPC server");

    tonic::transport::Server::builder()
        .add_service(server.into_service())
        .serve(addr)
        .await?;

    Ok(())
}
