// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Envoy `ExternalProcessor.Process` bidirectional streaming implementation.
//!
//! Mirrors the Go LW-EPP `StreamingServer` from GAIE `pkg/epp-light/server.go`
//! (issue #2834 / PR #2842). The server handles the ext-proc protocol and
//! delegates endpoint selection to an `EndpointPicker` implementation.
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
use crate::picker::{Endpoint, EndpointPicker, PickError, RequestInfo};
use crate::proto::envoy::service::ext_proc::v3::{
    self as ext_proc, ProcessingRequest, ProcessingResponse,
    external_processor_server::{ExternalProcessor, ExternalProcessorServer},
    processing_request,
};
use crate::proto::envoy::r#type::v3::StatusCode;

/// State machine phases for the ext_proc stream, matching the Go LW-EPP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StreamState {
    RequestReceived,
    HeaderRequestResponseComplete,
    BodyRequestResponsesComplete,
    TrailerRequestResponsesComplete,
    ResponseReceived,
    HeaderResponseResponseComplete,
    BodyResponseResponsesComplete,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    request_received_at: Option<Instant>,
    request_size: usize,
    response_size: usize,
    response_complete: bool,
    model_server_streaming: bool,

    request_headers: Vec<(String, String)>,
    request_metadata: HashMap<String, prost_types::Struct>,
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
            request_headers: Vec::new(),
            request_metadata: HashMap::new(),
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
    /// Mirrors Go LW-EPP `sendPendingResponses`.
    fn drain_pending_responses(&mut self) -> Vec<ProcessingResponse> {
        let mut out = Vec::new();

        if self.state == StreamState::RequestEvicted {
            out.push(envoy_helpers::build_eviction_response());
            return out;
        }

        if self.state == StreamState::RequestReceived
            && let Some(resp) = self.req_header_resp.take()
        {
            if let Some(crate::proto::envoy::service::ext_proc::v3::processing_response::Response::RequestHeaders(ref hr)) = resp.response {
                if let Some(ref common) = hr.response {
                    if let Some(ref hm) = common.header_mutation {
                        tracing::info!(
                            set_headers_count = hm.set_headers.len(),
                            clear_route_cache = common.clear_route_cache,
                            has_dynamic_metadata = resp.dynamic_metadata.is_some(),
                            "[WIRE] Sending RequestHeaders response to Envoy"
                        );
                        for h in &hm.set_headers {
                            if let Some(ref hv) = h.header {
                                tracing::info!(
                                    key = %hv.key,
                                    value = %String::from_utf8_lossy(&hv.raw_value),
                                    "[WIRE] set_header"
                                );
                            }
                        }
                    }
                }
            }
            out.push(resp);
            self.state = StreamState::HeaderRequestResponseComplete;
        }

        if self.state == StreamState::HeaderRequestResponseComplete
            && !self.req_body_resp.is_empty()
        {
            tracing::info!(count = self.req_body_resp.len(), "[WIRE] Sending req_body_resp to Envoy");
            out.append(&mut self.req_body_resp);
            self.state = StreamState::BodyRequestResponsesComplete;
        }

        if self.state == StreamState::BodyRequestResponsesComplete
            && let Some(resp) = self.req_trailer_resp.take()
        {
            out.push(resp);
            self.state = StreamState::TrailerRequestResponsesComplete;
        }

        if self.state == StreamState::ResponseReceived
            && let Some(resp) = self.resp_header_resp.take()
        {
            out.push(resp);
            self.state = StreamState::HeaderResponseResponseComplete;
        }

        if self.state == StreamState::HeaderResponseResponseComplete {
            out.append(&mut self.resp_body_resp);
            if self.response_complete {
                self.state = StreamState::BodyResponseResponsesComplete;
            }
        }

        if self.state == StreamState::BodyResponseResponsesComplete
            && let Some(resp) = self.resp_trailer_resp.take()
        {
            out.push(resp);
        }

        out
    }
}

/// The ext_proc gRPC server. Mirrors Go LW-EPP `StreamingServer`.
///
/// Takes an `EndpointPicker` for endpoint selection, decoupling the ext-proc
/// protocol handling from the routing decision — exactly as the Go LW-EPP
/// separates `StreamingServer` from `EndpointPicker`.
pub struct ExtProcServer<P: EndpointPicker> {
    picker: Arc<P>,
    /// Endpoints known to the server. In the Go LW-EPP these come from
    /// the `Datastore` which watches K8s pods. Here they can be populated
    /// from Dynamo's discovery or set externally.
    endpoints: Arc<tokio::sync::RwLock<Vec<Endpoint>>>,
}

impl<P: EndpointPicker> ExtProcServer<P> {
    pub fn new(picker: Arc<P>) -> Self {
        Self {
            picker,
            endpoints: Arc::new(tokio::sync::RwLock::new(Vec::new())),
        }
    }

    /// Set the available endpoints (equivalent to LW-EPP's `Datastore.ListEndpoints()`).
    pub async fn set_endpoints(&self, endpoints: Vec<Endpoint>) {
        *self.endpoints.write().await = endpoints;
    }

    /// Create a `tonic` service ready for registration on a gRPC server.
    pub fn into_service(self) -> ExternalProcessorServer<Self> {
        ExternalProcessorServer::new(self)
    }

    /// Handle request headers phase.
    /// Mirrors Go LW-EPP `handleRequestHeaders` in `server.go`.
    fn handle_request_headers(ctx: &mut RequestContext, hdr: &ext_proc::HttpHeaders) {
        ctx.request_received_at = Some(Instant::now());

        if hdr.end_of_stream {
            // Body-less request (e.g. GET) — will be picked in header-only path
            return;
        }

        if let Some(header_map) = &hdr.headers {
            ctx.request_headers = envoy_helpers::collect_headers(header_map);

            if let Some(id) =
                envoy_helpers::extract_header_value(header_map, metadata::REQUEST_ID_HEADER_KEY)
                && !id.is_empty()
            {
                ctx.request_id = id;
            }
        }

        if ctx.request_id.is_empty() {
            ctx.request_id = uuid::Uuid::new_v4().to_string();
            ctx.request_headers.push((
                metadata::REQUEST_ID_HEADER_KEY.to_string(),
                ctx.request_id.clone(),
            ));
        }
    }

    /// Handle a header-only request (EndOfStream on headers, no body).
    /// Mirrors Go LW-EPP `handleHeaderOnlyRequest`.
    async fn handle_header_only_request(
        picker: &P,
        ctx: &mut RequestContext,
        endpoints: &[Endpoint],
    ) -> Result<(), ExtProcError> {
        let req_info = RequestInfo {
            headers: ctx.request_headers.clone(),
            body: vec![],
            model: String::new(),
            candidate_subset: vec![],
        };

        let result = picker
            .pick(&req_info, endpoints)
            .await
            .map_err(ExtProcError::from_pick_error)?;

        ctx.target_endpoint = result.endpoint.clone();
        ctx.req_header_resp = Some(envoy_helpers::build_request_header_response(
            &result.endpoint,
            None,
            &ctx.request_headers,
        ));
        Ok(())
    }

    /// Handle request body phase: extract model, call picker.
    /// Mirrors Go LW-EPP `handleRequestBody`.
    async fn handle_request_body(
        picker: &P,
        ctx: &mut RequestContext,
        raw_body: &[u8],
        endpoints: &[Endpoint],
    ) -> Result<(), ExtProcError> {
        ctx.request_size = raw_body.len();

        let model = extract_model_from_body(raw_body);
        let candidate_subset = extract_candidate_subset(&ctx.request_metadata);

        let req_info = RequestInfo {
            headers: ctx.request_headers.clone(),
            body: raw_body.to_vec(),
            model: model.clone(),
            candidate_subset,
        };

        let result = picker
            .pick(&req_info, endpoints)
            .await
            .map_err(ExtProcError::from_pick_error)?;

        ctx.target_endpoint = result.endpoint.clone();
        ctx.incoming_model_name = model;
        ctx.target_model_name = ctx.incoming_model_name.clone();

        tracing::info!(
            request_id = %ctx.request_id,
            endpoint = %result.endpoint,
            picker_header_count = result.headers.len(),
            "Request routed"
        );
        for (k, v) in &result.headers {
            tracing::info!(key = %k, value = %v, "[MUTATION] Routing header going into ext_proc set_headers");
        }

        // Only send NEW headers (routing headers from the picker) in the
        // ext_proc header mutation. Do NOT re-send original request headers —
        // they already exist on the request and Envoy rejects mutations that
        // try to set restricted headers (x-envoy-*, x-forwarded-*, pseudo-headers).
        ctx.req_header_resp = Some(envoy_helpers::build_request_header_response(
            &result.endpoint,
            Some(ctx.request_size),
            &result.headers,
        ));

        // Inject nvext.token_data into the request body JSON so the backend
        // skips redundant tokenization. Mirrors Go EPP's setTokenizedPrompt.
        let forwarded_body = if let Some(ref token_ids) = result.token_ids {
            match inject_token_data(raw_body, token_ids) {
                Ok(modified) => {
                    tracing::info!(
                        token_count = token_ids.len(),
                        body_size_before = raw_body.len(),
                        body_size_after = modified.len(),
                        "Injected nvext.token_data into request body"
                    );
                    modified
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to inject token_data, forwarding original body");
                    raw_body.to_vec()
                }
            }
        } else {
            raw_body.to_vec()
        };

        ctx.req_body_resp = envoy_helpers::build_request_body_responses(&forwarded_body);
        tracing::info!(
            has_header_resp = ctx.req_header_resp.is_some(),
            body_resp_count = ctx.req_body_resp.len(),
            "[MUTATION] Responses prepared, waiting for drain"
        );

        Ok(())
    }

    /// Handle response headers from the upstream model server.
    fn handle_response_headers(ctx: &mut RequestContext, hdr: &ext_proc::HttpHeaders) {
        if let Some(header_map) = &hdr.headers {
            for h in &header_map.headers {
                let key = h.key.to_ascii_lowercase();
                let value = envoy_helpers::get_header_value(h);
                if key == "content-type" && value.contains("text/event-stream") {
                    ctx.model_server_streaming = true;
                }
                ctx.response_headers.insert(key, value);
            }
        }

        ctx.state = StreamState::ResponseReceived;
        ctx.resp_header_resp = Some(envoy_helpers::build_response_header_response());
    }

    /// Handle response body from the upstream model server.
    fn handle_response_body(ctx: &mut RequestContext, body: &ext_proc::HttpBody) {
        let end_of_stream = body.end_of_stream;
        let chunk = &body.body;
        ctx.response_size += chunk.len();

        if ctx.model_server_streaming {
            if end_of_stream {
                ctx.response_complete = true;
            }
            let rewritten = envoy_helpers::rewrite_model_name(
                chunk,
                &ctx.target_model_name,
                &ctx.incoming_model_name,
            );
            ctx.resp_body_resp =
                envoy_helpers::build_response_body_responses(&rewritten, end_of_stream, None);
        } else if end_of_stream {
            ctx.response_complete = true;
            let rewritten = envoy_helpers::rewrite_model_name(
                chunk,
                &ctx.target_model_name,
                &ctx.incoming_model_name,
            );
            ctx.resp_body_resp =
                envoy_helpers::build_response_body_responses(&rewritten, true, None);
        }
    }
}

#[tonic::async_trait]
impl<P: EndpointPicker> ExternalProcessor for ExtProcServer<P> {
    type ProcessStream =
        Pin<Box<dyn Stream<Item = Result<ProcessingResponse, Status>> + Send + 'static>>;

    async fn process(
        &self,
        request: Request<Streaming<ProcessingRequest>>,
    ) -> Result<Response<Self::ProcessStream>, Status> {
        let mut inbound = request.into_inner();
        let picker = self.picker.clone();
        let endpoints_lock = self.endpoints.clone();

        let (tx, rx) = mpsc::channel::<Result<ProcessingResponse, Status>>(32);
        let output_stream = ReceiverStream::new(rx);

        tokio::spawn(async move {
            let mut ctx = RequestContext::new();
            let mut body_buf: Vec<u8> = Vec::new();
            let mut resp_body_buf: Vec<u8> = Vec::new();

            let result: Result<(), Status> = async {
                while let Some(req_result) = inbound.next().await {
                    let req = req_result.map_err(|e| {
                        Status::unknown(format!("Cannot receive stream request: {e}"))
                    })?;

                    ctx.request_metadata = envoy_helpers::extract_metadata_values(&req);

                    if let Some(ref pc) = req.protocol_config {
                        tracing::info!(
                            request_body_mode = pc.request_body_mode,
                            response_body_mode = pc.response_body_mode,
                            send_body_without_waiting = pc.send_body_without_waiting_for_header_response,
                            "[PROTOCOL] ProtocolConfiguration from Envoy"
                        );
                    }

                    match req.request {
                        Some(processing_request::Request::RequestHeaders(ref hdr)) => {
                            tracing::info!(eos = hdr.end_of_stream, "[MSG-ORDER] Received RequestHeaders from Envoy");
                            ExtProcServer::<P>::handle_request_headers(&mut ctx, hdr);

                            if hdr.end_of_stream {
                                let endpoints = endpoints_lock.read().await;
                                if let Err(e) = ExtProcServer::handle_header_only_request(
                                    &*picker, &mut ctx, &endpoints,
                                )
                                .await
                                {
                                    let resp = e.into_processing_response();
                                    let _ = tx.send(Ok(resp)).await;
                                    return Ok(());
                                }
                            }
                        }
                        Some(processing_request::Request::RequestBody(ref body)) => {
                            tracing::info!(eos = body.end_of_stream, body_len = body.body.len(), "[MSG-ORDER] Received RequestBody from Envoy");
                            body_buf.extend_from_slice(&body.body);

                            if body.end_of_stream {
                                let raw_body = std::mem::take(&mut body_buf);
                                let endpoints = endpoints_lock.read().await;
                                if let Err(e) = ExtProcServer::handle_request_body(
                                    &*picker, &mut ctx, &raw_body, &endpoints,
                                )
                                .await
                                {
                                    let resp = e.into_processing_response();
                                    let _ = tx.send(Ok(resp)).await;
                                    return Ok(());
                                }
                            }
                        }
                        Some(processing_request::Request::RequestTrailers(_)) => {}
                        Some(processing_request::Request::ResponseHeaders(ref hdr)) => {
                            ExtProcServer::<P>::handle_response_headers(&mut ctx, hdr);
                        }
                        Some(processing_request::Request::ResponseBody(ref body)) => {
                            if ctx.model_server_streaming {
                                ExtProcServer::<P>::handle_response_body(&mut ctx, body);
                            } else {
                                resp_body_buf.extend_from_slice(&body.body);
                                if body.end_of_stream {
                                    let full_body = std::mem::take(&mut resp_body_buf);
                                    let synthetic = ext_proc::HttpBody {
                                        body: full_body,
                                        end_of_stream: true,
                                    };
                                    ExtProcServer::<P>::handle_response_body(&mut ctx, &synthetic);
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
                                    };
                                    ExtProcServer::<P>::handle_response_body(&mut ctx, &synthetic);
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
        });

        Ok(Response::new(Box::pin(output_stream)))
    }
}

// ---------------------------------------------------------------------------
// Request helpers (mirrors Go LW-EPP request.go)
// ---------------------------------------------------------------------------

/// Inject pre-computed token IDs into the request body JSON as
/// `nvext.token_data`. This lets the backend skip redundant tokenization.
/// Mirrors Go EPP's `setTokenizedPrompt` in `shared.go`.
fn inject_token_data(body: &[u8], token_ids: &[u32]) -> anyhow::Result<Vec<u8>> {
    let mut parsed: serde_json::Value = serde_json::from_slice(body)?;

    let obj = parsed
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("body is not a JSON object"))?;

    let nvext = obj
        .entry("nvext")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));

    let nvext_obj = nvext
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("nvext is not a JSON object"))?;

    nvext_obj.insert(
        "token_data".to_string(),
        serde_json::Value::Array(
            token_ids
                .iter()
                .map(|&t| serde_json::Value::Number(serde_json::Number::from(t)))
                .collect(),
        ),
    );

    Ok(serde_json::to_vec(&parsed)?)
}

/// Extract the "model" field from a JSON request body.
/// Mirrors Go LW-EPP `extractModelFromBody`.
fn extract_model_from_body(body: &[u8]) -> String {
    #[derive(serde::Deserialize)]
    struct ModelField {
        model: Option<String>,
    }

    serde_json::from_slice::<ModelField>(body)
        .ok()
        .and_then(|m| m.model)
        .unwrap_or_default()
}

/// Extract the candidate endpoint subset from ext-proc request metadata.
/// Mirrors Go LW-EPP `extractCandidateSubset`.
fn extract_candidate_subset(
    request_metadata: &HashMap<String, prost_types::Struct>,
) -> Vec<String> {
    let ns = match request_metadata.get(metadata::SUBSET_FILTER_NAMESPACE) {
        Some(s) => s,
        None => return vec![],
    };

    let subset_val = match ns.fields.get(metadata::SUBSET_FILTER_KEY) {
        Some(v) => v,
        None => return vec![],
    };

    if let Some(prost_types::value::Kind::StringValue(s)) = &subset_val.kind {
        if s.is_empty() {
            return vec![];
        }
        return s.split(',').map(|s| s.to_string()).collect();
    }

    if let Some(prost_types::value::Kind::ListValue(list)) = &subset_val.kind {
        return list
            .values
            .iter()
            .filter_map(|v| {
                if let Some(prost_types::value::Kind::StringValue(s)) = &v.kind {
                    Some(s.clone())
                } else {
                    None
                }
            })
            .collect();
    }

    vec![]
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

struct ExtProcError {
    status_code: StatusCode,
    message: String,
}

impl ExtProcError {
    fn from_pick_error(e: PickError) -> Self {
        match e {
            PickError::NoEndpoints => Self {
                status_code: StatusCode::ServiceUnavailable,
                message: e.to_string(),
            },
            PickError::RoutingFailed(msg) => Self {
                status_code: StatusCode::ServiceUnavailable,
                message: msg,
            },
            PickError::TokenizationFailed(msg) => Self {
                status_code: StatusCode::BadRequest,
                message: msg,
            },
        }
    }

    fn into_processing_response(self) -> ProcessingResponse {
        envoy_helpers::build_error_response(self.status_code, Some(&self.message))
    }
}

/// Start the ext_proc gRPC server on the given port.
pub async fn run_ext_proc_server<P: EndpointPicker>(
    picker: Arc<P>,
    port: u16,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = format!("0.0.0.0:{port}").parse()?;
    let server = ExtProcServer::new(picker);

    tracing::info!(%addr, "Starting ext_proc gRPC server");

    tonic::transport::Server::builder()
        .add_service(server.into_service())
        .serve(addr)
        .await?;

    Ok(())
}
