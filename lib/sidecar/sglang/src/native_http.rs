// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque transport for SGLang's native streaming `/generate` API.

use std::{io, time::Duration};

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, FirstTokenNotifier, GenerateContext, LLMEngineOutput,
    PreprocessedRequest, PromptTokensDetails, usage,
};
use dynamo_sidecar_common::{GrpcEndpoint, HttpEndpoint};
use futures::{StreamExt, TryStreamExt, future::BoxFuture, stream::BoxStream};
use reqwest::{Response, StatusCode, header};
use serde::Deserialize;
use serde_json::{Map, Value};
use tokio::time::Instant;
use tokio_util::{
    codec::{FramedRead, LinesCodec, LinesCodecError},
    io::StreamReader,
    sync::CancellationToken,
};

use crate::client::{self, AbortRpc};
use crate::discovery::EngineInfo;
use crate::protocol;

const PAYLOAD_KEY: &str = "sglang_tito";
/// SGLang's default `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`, so a healthy engine ends the wait first.
const DECODE_DRAIN_TIMEOUT: Duration = Duration::from_secs(300);
/// `dynamo_llm::protocols::response_merge::PREFILL_OUTPUT_ANNOTATION`.
const PREFILL_OUTPUT_ANNOTATION: &str = "prefill_output";

pub(crate) struct NativeRequest {
    body: Value,
    is_prefill: bool,
    is_decode: bool,
    prefill_handoff: Option<Value>,
    /// Return prefill's final native response for the frontend to join.
    keep_prefill_output: bool,
}

/// Rebuild the installed SGLang version's request from the opaque frontend
/// envelope, replacing only fields owned by Dynamo routing.
pub(crate) fn request(
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<Option<NativeRequest>, DynamoError> {
    let Some(payload) = request
        .extra_args
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|extra| extra.get(PAYLOAD_KEY))
    else {
        return Ok(None);
    };
    let mut body = payload
        .as_object()
        .cloned()
        .ok_or_else(|| client::invalid_arg("extra_args.sglang_tito must be a JSON object"))?;
    if request.token_ids.is_empty() || request.prompt_embeds.is_some() {
        return Err(client::invalid_arg(
            "native SGLang Generate requires token input",
        ));
    }

    body.insert("input_ids".into(), serde_json::json!(request.token_ids));
    body.insert("rid".into(), Value::String(request_id.to_string()));
    body.insert("stream".into(), Value::Bool(true));

    let routing = request.routing.as_ref();
    if let Some(priority) = routing.and_then(|routing| routing.priority) {
        body.insert("priority".into(), Value::from(priority));
    } else {
        body.remove("priority");
    }

    if mode.is_prefill() {
        let sampling = body
            .entry("sampling_params")
            .or_insert_with(|| Value::Object(Map::new()));
        if sampling.is_null() {
            *sampling = Value::Object(Map::new());
        }
        let sampling = sampling
            .as_object_mut()
            .ok_or_else(|| client::invalid_arg("sampling_params must be an object"))?;
        sampling.insert("n".into(), Value::from(1));
        sampling.insert("max_new_tokens".into(), Value::from(1));
        sampling.remove("min_new_tokens");
    }

    let disaggregated =
        protocol::resolve_disaggregated_params(request, mode, bootstrap_host, bootstrap_port)?;
    if let Some(params) = disaggregated.as_ref() {
        body.insert(
            "bootstrap_host".into(),
            Value::String(params.bootstrap_host.clone()),
        );
        body.insert("bootstrap_port".into(), Value::from(params.bootstrap_port));
        body.insert("bootstrap_room".into(), Value::from(params.bootstrap_room));
    } else {
        body.remove("bootstrap_host");
        body.remove("bootstrap_port");
        body.remove("bootstrap_room");
    }

    // The engine serves one DP rank, and `generate` refuses a request routed
    // to another, so a body's own placement is dropped.
    body.remove("routed_dp_rank");
    if let Some(lora_path) = routing.and_then(|routing| routing.lora_name.as_ref()) {
        body.insert("lora_path".into(), Value::String(lora_path.clone()));
    }

    let prefill_handoff = if mode.is_prefill() {
        disaggregated
            .as_ref()
            .map(protocol::disaggregated_params_to_json)
    } else {
        None
    };
    Ok(Some(NativeRequest {
        body: Value::Object(body),
        is_prefill: mode.is_prefill(),
        is_decode: mode.is_decode(),
        prefill_handoff,
        keep_prefill_output: request
            .annotations
            .iter()
            .any(|a| a == PREFILL_OUTPUT_ANNOTATION),
    }))
}

/// SGLang's native `/generate`, served by its Rust server beside the gRPC
/// service. The server has no HTTP `/abort_request`, so requests are aborted
/// through the gRPC Abort RPC, and it reports a failed request as one
/// `{"error": {message, code}}` frame.
#[derive(Clone)]
pub(crate) struct NativeHttp {
    client: reqwest::Client,
    endpoint: HttpEndpoint,
    abort: AbortRpc,
    max_event_bytes: usize,
    abort_timeout: Duration,
    decode_drain_timeout: Duration,
}

// Closing the SSE connection does not reliably stop SGLang scheduler work.
struct AbortGuard<'a> {
    native_http: &'a NativeHttp,
    request_id: Option<String>,
    runtime: Option<tokio::runtime::Handle>,
    opening: Option<BoxFuture<'static, Result<Response, DynamoError>>>,
    stream: BoxStream<'static, Result<String, LinesCodecError>>,
    needs_first_token: bool,
    first_token: Option<FirstTokenNotifier>,
}

impl Drop for AbortGuard<'_> {
    fn drop(&mut self) {
        let Some(request_id) = self.request_id.take() else {
            return;
        };
        let Some(runtime) = self.runtime.as_ref() else {
            return;
        };
        let native_http = self.native_http.clone();
        let mut opening = self.opening.take();
        let mut stream = std::mem::replace(&mut self.stream, Box::pin(futures::stream::empty()));
        let needs_first_token = self.needs_first_token;
        let first_token = self.first_token.clone();
        runtime.spawn(async move {
            // Decode must finish receiving KV before abort can release its destination pages.
            if needs_first_token {
                let drain = async {
                    if let Some(pending) = opening.as_mut() {
                        let response = pending.await;
                        opening = None;
                        match response {
                            Ok(response) => stream = response_stream(response, native_http.max_event_bytes),
                            Err(error) => {
                                tracing::debug!(%request_id, %error, "SGLang native decode ended before first output");
                                return false;
                            }
                        }
                    }
                    loop {
                        let response = match next_response(&mut stream, native_http.max_event_bytes).await {
                            Ok(response) => response,
                            Err(error) => {
                                tracing::debug!(%request_id, %error, "SGLang native decode ended before first output");
                                return false;
                            }
                        };
                        let has_output = response_has_output(&response);
                        if has_output && let Some(first_token) = &first_token {
                            first_token.notify();
                        }
                        if response_is_terminal(&response) {
                            return false;
                        }
                        if has_output {
                            return true;
                        }
                    }
                };
                match tokio::time::timeout(native_http.decode_drain_timeout, drain).await {
                    Ok(true) => {}
                    Ok(false) => return,
                    Err(_) => tracing::warn!(
                        %request_id,
                        timeout_secs = native_http.decode_drain_timeout.as_secs(),
                        "SGLang native decode produced no output before the drain deadline; forcing abort"
                    ),
                }
            }
            native_http
                .abort
                .abort(&request_id, native_http.abort_timeout)
                .await;
            // Disconnecting first removes the request state needed by SGLang's abort handler.
            drop(opening);
            drop(stream);
        });
    }
}

fn build_http_client(connect_timeout: Duration) -> Result<reqwest::Client, DynamoError> {
    reqwest::Client::builder()
        .connect_timeout(connect_timeout)
        // Retire idle connections before SGLang's default 5s keep-alive closes them.
        .pool_idle_timeout(Duration::from_secs(2))
        .build()
        .map_err(|error| {
            client::invalid_arg(format!("could not configure SGLang HTTP client: {error}"))
        })
}

impl NativeHttp {
    pub(crate) fn discover(
        grpc_endpoint: &GrpcEndpoint,
        discovery: &EngineInfo,
        abort: AbortRpc,
        connect_timeout: Duration,
        max_event_bytes: usize,
    ) -> Result<Option<Self>, DynamoError> {
        let port = match &discovery.http_port {
            None => return Ok(None),
            Some(port) => port.clone()?,
        };
        if !discovery.incremental_streaming_output {
            tracing::warn!(
                port,
                "SGLang native HTTP generation is disabled because incremental streaming output is not enabled"
            );
            return Ok(None);
        }
        let endpoint = HttpEndpoint::from_grpc(grpc_endpoint, port).map_err(|error| {
            client::protocol_error(format!("invalid SGLang HTTP endpoint: {error}"))
        })?;
        let client = build_http_client(connect_timeout)?;
        Ok(Some(Self {
            client,
            endpoint,
            abort,
            max_event_bytes,
            abort_timeout: connect_timeout,
            decode_drain_timeout: DECODE_DRAIN_TIMEOUT,
        }))
    }

    pub(crate) async fn await_ready(
        &self,
        deadline: Instant,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        let endpoint = self.endpoint.with_path("/health");
        loop {
            let response =
                tokio::time::timeout_at(deadline, self.client.get(endpoint.clone()).send()).await;
            let failure = match response {
                Ok(Ok(response)) if response.status().is_success() => return Ok(()),
                Ok(Ok(response)) => {
                    let status = response.status();
                    if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
                        return Err(authentication_error("/health", status));
                    }
                    if status.is_client_error() {
                        return Err(client::protocol_error(format!(
                            "SGLang HTTP readiness probe returned HTTP {status}"
                        )));
                    }
                    format!("HTTP {status}")
                }
                Ok(Err(error)) => error.to_string(),
                Err(_) => {
                    return Err(client::connection_timeout(format!(
                        "SGLang HTTP readiness probe at {endpoint} exceeded the startup deadline"
                    )));
                }
            };

            if Instant::now() >= deadline {
                return Err(client::cannot_connect(format!(
                    "SGLang HTTP endpoint {endpoint} did not become ready: {failure}"
                )));
            }
            tokio::time::sleep_until((Instant::now() + retry_interval).min(deadline)).await;
        }
    }

    async fn open(self, body: Value) -> Result<Response, DynamoError> {
        let response = self
            .client
            .post(self.endpoint.with_path("/generate"))
            .header(header::ACCEPT, "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(request_error)?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let detail = response
            .text()
            .await
            .unwrap_or_else(|error| format!("could not read error response: {error}"));
        Err(response_error(status, detail))
    }

    pub(crate) fn generate(
        self,
        request: NativeRequest,
        ctx: GenerateContext,
        cancel: CancellationToken,
    ) -> BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
        Box::pin(async_stream::stream! {
            let is_prefill = request.is_prefill;
            let keep_prefill_output = request.keep_prefill_output;
            let mut prefill_handoff = request.prefill_handoff;
            let stopped_ctx = ctx.inner_arc();
            let mut request_cancellation = Box::pin(async move { stopped_ctx.stopped().await });
            let mut shutdown_cancellation = Box::pin(async move { cancel.cancelled().await });
            let mut abort_guard = AbortGuard {
                native_http: &self,
                request_id: Some(ctx.id().to_string()),
                runtime: tokio::runtime::Handle::try_current().ok(),
                opening: Some(Box::pin(self.clone().open(request.body))),
                stream: Box::pin(futures::stream::empty()),
                needs_first_token: request.is_decode,
                first_token: ctx.first_token_notifier().cloned(),
            };
            tracing::debug!(request_id = %ctx.id(), endpoint = %self.endpoint.with_path("/generate"), "sending native request to SGLang HTTP");
            let opened = tokio::select! {
                biased;
                _ = &mut request_cancellation => None,
                _ = &mut shutdown_cancellation => None,
                response = abort_guard.opening.as_mut().unwrap() => Some(response),
            };
            let Some(response) = opened else {
                drop(abort_guard);
                yield Err(client::cancelled(format!(
                    "SGLang native request {} was cancelled",
                    ctx.id()
                )));
                return;
            };
            abort_guard.opening = None;
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };
            abort_guard.stream = response_stream(response, self.max_event_bytes);
            if is_prefill {
                let Some(handoff) = prefill_handoff.take() else {
                    yield Err(client::protocol_error(
                        "SGLang native prefill request is missing disaggregated params",
                    ));
                    return;
                };
                // Publish the handoff only after the HTTP transport returns
                // successful response headers so decode can rendezvous while
                // prefill runs.
                yield Ok(LLMEngineOutput {
                    disaggregated_params: Some(handoff),
                    ..Default::default()
                });
            }

            let mut first_output_seen = false;
            loop {
                let selected = tokio::select! {
                    biased;
                    _ = &mut request_cancellation => None,
                    _ = &mut shutdown_cancellation => None,
                    response = next_response(&mut abort_guard.stream, self.max_event_bytes) => Some(response),
                };
                let Some(response) = selected else {
                    drop(abort_guard);
                    yield Err(client::cancelled(format!(
                        "SGLang native request {} was cancelled",
                        ctx.id()
                    )));
                    return;
                };
                let response = match response {
                    Ok(response) => response,
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                };
                // SGLang ends a failed request, prefill included, with one
                // coded error frame, so no abort follows. The native path
                // reports cancellation as a Cancelled error, so a 499 is one
                // whether or not this request was cancelled.
                if let Some(error) = response.get("error") {
                    abort_guard.request_id = None;
                    yield Err(coded_error(error));
                    return;
                }
                // Prefill's raw HTTP payload is forwarded only on request.
                // Surface backend failures before that payload is discarded.
                if is_prefill
                    && let Some(finish) = response.pointer("/meta_info/finish_reason")
                    && let Some(kind @ ("abort" | "error" | "cancelled")) =
                        finish.get("type").and_then(Value::as_str)
                {
                    abort_guard.request_id = None;
                    yield Err(protocol::terminal_failure(kind, finish));
                    return;
                }
                let has_output = response_has_output(&response);
                // The router counts decode output from token_ids; clients get the
                // opaque response. Prefill reports none, as on the gRPC path.
                let token_ids = if is_prefill {
                    Vec::new()
                } else {
                    match output_ids(&response) {
                        Ok(token_ids) => token_ids,
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    }
                };
                let (mut output, terminal) =
                    output(response, is_prefill, &mut prefill_handoff);
                output.token_ids = token_ids;
                if !first_output_seen && has_output && (!is_prefill || terminal) {
                    abort_guard.needs_first_token = false;
                    ctx.notify_first_token();
                    first_output_seen = true;
                }
                if is_prefill {
                    if !terminal {
                        continue;
                    }
                    if !keep_prefill_output {
                        output.engine_data = None;
                    }
                }
                if terminal {
                    abort_guard.request_id = None;
                }
                yield Ok(output);
                if terminal {
                    return;
                }
            }
        })
    }
}

/// An error frame's body, classified like a gRPC Generate error item.
fn coded_error(error: &Value) -> DynamoError {
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("SGLang reported an error without a message");
    let code = error
        .get("code")
        .and_then(Value::as_u64)
        .and_then(|code| u32::try_from(code).ok())
        .unwrap_or(0);
    engine_error(message, code)
}

/// A failure SGLang reported with `message` and HTTP status `code`.
fn engine_error(message: &str, code: u32) -> DynamoError {
    protocol::coded_error("/generate", message, code)
}

/// A `/generate` failure answered with an HTTP error status instead of a
/// stream: a body SGLang cannot parse or decode, or a header override it
/// rejects. It answers these unary even for a streaming request, with the
/// same `{"error": {message, code}}` object as its error frames, so they map
/// like one, keeping the HTTP status as the code. Any other body maps by the
/// status alone.
fn response_error(status: StatusCode, detail: String) -> DynamoError {
    if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
        return authentication_error("/generate", status);
    }
    let body = serde_json::from_str::<Value>(&detail).ok();
    match body
        .as_ref()
        .and_then(|body| body.pointer("/error/message"))
        .and_then(Value::as_str)
    {
        Some(message) => engine_error(message, u32::from(status.as_u16())),
        None => status_error(status, detail),
    }
}

fn output_ids(response: &Value) -> Result<Vec<u32>, DynamoError> {
    response.get("output_ids").map_or(Ok(Vec::new()), |ids| {
        Vec::<u32>::deserialize(ids)
            .map_err(|error| client::protocol_error(format!("invalid SGLang output_ids: {error}")))
    })
}

fn response_stream(
    response: Response,
    max_event_bytes: usize,
) -> BoxStream<'static, Result<String, LinesCodecError>> {
    let bytes = response.bytes_stream().map_err(io::Error::other);
    let reader = StreamReader::new(bytes);
    Box::pin(FramedRead::new(
        reader,
        LinesCodec::new_with_max_length(max_event_bytes),
    ))
}

async fn next_response(
    stream: &mut BoxStream<'static, Result<String, LinesCodecError>>,
    max_event_bytes: usize,
) -> Result<Value, DynamoError> {
    loop {
        let line = stream
            .next()
            .await
            .ok_or_else(|| {
                client::protocol_error("SGLang /generate closed before a terminal response")
            })?
            .map_err(|error| match error {
                LinesCodecError::MaxLineLengthExceeded => client::protocol_error(format!(
                    "SGLang /generate event exceeded {max_event_bytes} bytes; raise --native-http-max-event-bytes (DYN_SGLANG_NATIVE_HTTP_MAX_EVENT_BYTES)"
                )),
                error => client::protocol_error(format!("invalid SGLang /generate stream: {error}")),
            })?;
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.strip_prefix(' ').unwrap_or(data);
        if data.is_empty() {
            continue;
        }
        if data == "[DONE]" {
            return Err(client::protocol_error(
                "SGLang /generate finished without a terminal response",
            ));
        }
        return serde_json::from_str(data).map_err(|error| {
            client::protocol_error(format!("SGLang /generate returned invalid JSON: {error}"))
        });
    }
}

fn response_has_output(response: &Value) -> bool {
    [response.get("output_ids"), response.get("text")]
        .into_iter()
        .flatten()
        .any(|value| match value {
            Value::Array(values) => !values.is_empty(),
            Value::String(value) => !value.is_empty(),
            _ => false,
        })
}

fn response_is_terminal(response: &Value) -> bool {
    response.get("error").is_some()
        || response
            .pointer("/meta_info/finish_reason")
            .is_some_and(|reason| !reason.is_null())
}

/// Error frames fail the stream before they reach here. Prefill's terminal
/// carries no typed finish details.
fn output(
    response: Value,
    is_prefill: bool,
    prefill_handoff: &mut Option<Value>,
) -> (LLMEngineOutput, bool) {
    let finish = response
        .pointer("/meta_info/finish_reason")
        .filter(|reason| !reason.is_null());
    let finished = finish.is_some();
    let mut output = match finish {
        Some(_) if is_prefill => LLMEngineOutput::stop(),
        Some(finish) => decode_terminal(&response["meta_info"], finish),
        None => LLMEngineOutput::default(),
    };
    let mut engine_data = Map::new();
    engine_data.insert("sglang_response".to_string(), response);
    output.engine_data = Some(Value::Object(engine_data));
    if finished {
        output.disaggregated_params = prefill_handoff.take();
    }
    (output, finished)
}

fn decode_terminal(meta: &Value, finish: &Value) -> LLMEngineOutput {
    let mut output = match finish.get("type").and_then(Value::as_str) {
        Some("length") => LLMEngineOutput::length(),
        // SGLang also aborts ordinary cancellations, without error metadata.
        Some("abort")
            if ["status_code", "err_type"]
                .iter()
                .any(|key| finish.get(key).is_some_and(|value| !value.is_null())) =>
        {
            LLMEngineOutput::error(
                finish
                    .get("message")
                    .and_then(Value::as_str)
                    .filter(|message| !message.is_empty())
                    .unwrap_or("SGLang aborted the request with an error")
                    .to_string(),
            )
        }
        Some("abort") => LLMEngineOutput::cancelled(),
        _ => LLMEngineOutput::stop(),
    };
    let count = |key| {
        meta.get(key)
            .and_then(Value::as_u64)
            .and_then(|count| u32::try_from(count).ok())
    };
    if let (Some(prompt_tokens), Some(completion_tokens)) =
        (count("prompt_tokens"), count("completion_tokens"))
    {
        let mut completion_usage = usage(prompt_tokens, completion_tokens);
        completion_usage.prompt_tokens_details = count("cached_tokens")
            .filter(|&cached_tokens| cached_tokens > 0)
            .map(|cached_tokens| PromptTokensDetails {
                cached_tokens: Some(cached_tokens),
                ..Default::default()
            });
        output.completion_usage = Some(completion_usage);
    }
    output
}

fn request_error(error: reqwest::Error) -> DynamoError {
    if error.is_timeout() {
        client::connection_timeout(format!("SGLang /generate HTTP request timed out: {error}"))
    } else if error.is_connect() {
        client::cannot_connect(format!("could not connect to SGLang /generate: {error}"))
    } else {
        client::protocol_error(format!("SGLang /generate HTTP request failed: {error}"))
    }
}

fn authentication_error(operation: &str, status: StatusCode) -> DynamoError {
    client::protocol_error(format!(
        "SGLang HTTP {operation} returned HTTP {status}; the sidecar does not have backend authentication configured"
    ))
}

fn status_error(status: StatusCode, detail: String) -> DynamoError {
    let message = format!("SGLang /generate returned HTTP {status}: {detail}");
    if status.is_client_error() {
        client::invalid_arg(message)
    } else if matches!(status.as_u16(), 502..=504) {
        client::cannot_connect(message)
    } else {
        client::protocol_error(message)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use dynamo_backend_common::engine::RoutingHints;
    use dynamo_backend_common::{
        BackendError, DisaggregationMode, ErrorType, FinishReason, GenerateContext, OutputOptions,
        PreprocessedRequest, SamplingOptions, StopConditions,
    };
    use dynamo_sidecar_common::{GrpcEndpoint, HttpEndpoint};
    use futures::StreamExt;
    use reqwest::StatusCode;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::watch;
    use tokio_util::sync::CancellationToken;

    use super::{
        NativeHttp, NativeRequest, authentication_error, build_http_client, output, request,
        response_error, response_has_output,
    };
    use crate::client::{AbortRpc, client_from_channel};
    use crate::discovery::{EngineInfo, engine_info, parse_discovery};
    use crate::proto as pb;

    const MAX_EVENT_BYTES: usize = crate::args::DEFAULT_NATIVE_HTTP_MAX_EVENT_BYTES.get();

    fn canonical_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions::default())
            .build()
            .unwrap()
    }

    /// Discovery of an engine that reports HTTP `port`.
    fn discovery(port: u32) -> EngineInfo {
        let server_info = pb::GetServerInfoResponse {
            deployment_info: Some(pb::DeploymentInfo {
                nnodes: 1,
                dp_size: 1,
                dcp_size: 1,
                ..Default::default()
            }),
            incremental_streaming_output: true,
            port,
            ..Default::default()
        };
        let model_info = pb::GetModelInfoResponse {
            model_path: "model".to_string(),
            ..Default::default()
        };
        engine_info(parse_discovery(model_info, server_info, Vec::new()).unwrap()).unwrap()
    }

    /// Native HTTP on `port`; aborting through it fails without a gRPC server.
    fn native_http(port: u16) -> NativeHttp {
        let grpc = GrpcEndpoint::parse("127.0.0.1:30001", "test").unwrap();
        NativeHttp {
            client: build_http_client(Duration::from_secs(1)).unwrap(),
            endpoint: HttpEndpoint::from_grpc(&grpc, port).unwrap(),
            abort: AbortRpc::unreachable(),
            max_event_bytes: MAX_EVENT_BYTES,
            abort_timeout: Duration::from_secs(1),
            decode_drain_timeout: Duration::from_secs(1),
        }
    }

    fn native_request(is_prefill: bool) -> NativeRequest {
        NativeRequest {
            body: json!({"input_ids": [1], "stream": true}),
            is_prefill,
            is_decode: false,
            prefill_handoff: is_prefill.then(
                || json!({"bootstrap_host": "prefill", "bootstrap_port": 1, "bootstrap_room": 7}),
            ),
            keep_prefill_output: false,
        }
    }

    async fn serve_once(body: String, status: &str) -> (u16, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let status = status.to_string();
        let task = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
        });
        (port, task)
    }

    #[test]
    fn prefill_rewrites_minimum_generation() {
        let mut canonical = canonical_request();
        canonical.extra_args = Some(json!({
            "sglang_tito": {
                "sampling_params": {
                    "min_new_tokens": 2,
                    "max_new_tokens": 16
                }
            }
        }));

        let native = request(
            &canonical,
            "request-id",
            DisaggregationMode::Prefill,
            Some("prefill"),
            Some(5000),
        )
        .unwrap()
        .unwrap();
        assert_eq!(native.body["sampling_params"]["max_new_tokens"], 1);
        assert_eq!(native.body["sampling_params"]["n"], 1);
        assert!(
            native.body["sampling_params"]
                .get("min_new_tokens")
                .is_none()
        );
    }

    #[test]
    fn native_request_preserves_engine_fields_and_overrides_routing_fields() {
        let mut canonical = canonical_request();
        let payload = json!({
            "input_ids": [99], "rid": "stale", "stream": false,
            "priority": 99, "routed_dp_rank": 99, "lora_path": "old-adapter",
            "bootstrap_host": "stale", "bootstrap_port": 1, "bootstrap_room": 2,
            "sampling_params": {"temperature": 0.7, "seed": 42, "max_new_tokens": 9},
            "image_data": ["opaque-image"], "custom_engine_option": {"enabled": true}
        });
        canonical.extra_args = Some(json!({"sglang_tito": payload}));
        canonical.routing = Some(RoutingHints {
            priority: Some(-5),
            dp_rank: Some(0),
            lora_name: Some("selected-adapter".to_string()),
            ..Default::default()
        });

        let native = request(
            &canonical,
            "actual-rid",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(native.body["input_ids"], json!([1, 2, 3]));
        assert_eq!(native.body["rid"], "actual-rid");
        assert_eq!(native.body["stream"], true);
        assert_eq!(native.body["priority"], -5);
        assert!(native.body.get("routed_dp_rank").is_none());
        assert_eq!(native.body["lora_path"], "selected-adapter");
        for key in ["sampling_params", "image_data", "custom_engine_option"] {
            assert_eq!(native.body[key], payload[key], "{key}");
        }
        for key in ["bootstrap_host", "bootstrap_port", "bootstrap_room"] {
            assert!(native.body.get(key).is_none(), "{key}");
        }
        assert!(!native.is_prefill);
        assert!(native.prefill_handoff.is_none());
        assert_eq!(canonical.extra_args.unwrap()["sglang_tito"], payload);

        canonical = canonical_request();
        canonical.extra_args = Some(json!({"sglang_tito": payload}));
        let native = request(
            &canonical,
            "rid",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert!(native.body.get("priority").is_none());
        assert!(native.body.get("routed_dp_rank").is_none());
        assert_eq!(native.body["lora_path"], "old-adapter");
    }

    #[test]
    fn native_envelope_is_optional_and_requires_an_object_with_token_input() {
        let mut canonical = canonical_request();
        assert!(
            request(
                &canonical,
                "rid",
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .unwrap()
            .is_none()
        );
        for payload in [json!(null), json!([]), json!("invalid")] {
            canonical.extra_args = Some(json!({"sglang_tito": payload}));
            let error = request(
                &canonical,
                "rid",
                DisaggregationMode::Aggregated,
                None,
                None,
            )
            .err()
            .unwrap();
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::InvalidArgument)
            );
        }
        canonical.extra_args = Some(json!({"sglang_tito": {}}));
        canonical.token_ids = Vec::new().into();
        let error = request(
            &canonical,
            "rid",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .err()
        .unwrap();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn native_prefill_normalizes_sampling_and_attaches_the_same_handoff() {
        let mut canonical = canonical_request();
        for payload in [json!({}), json!({"sampling_params": null})] {
            canonical.extra_args = Some(json!({"sglang_tito": payload}));
            let native = request(
                &canonical,
                "rid",
                DisaggregationMode::Prefill,
                Some("prefill"),
                Some(5000),
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                native.body["sampling_params"],
                json!({"n": 1, "max_new_tokens": 1})
            );
            assert!(native.is_prefill);
            let handoff = native.prefill_handoff.unwrap();
            assert_eq!(handoff["bootstrap_host"], "prefill");
            assert_eq!(handoff["bootstrap_port"], 5000);
            assert!(handoff["bootstrap_room"].as_u64().unwrap() <= i64::MAX as u64);
            for key in ["bootstrap_host", "bootstrap_port", "bootstrap_room"] {
                assert_eq!(native.body[key], handoff[key], "{key}");
            }
        }
        canonical.extra_args = Some(json!({"sglang_tito": {"sampling_params": []}}));
        let error = request(
            &canonical,
            "rid",
            DisaggregationMode::Prefill,
            Some("prefill"),
            Some(5000),
        )
        .err()
        .unwrap();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn native_output_moves_expert_payload_into_envelope() {
        let response = json!({
            "meta_info": {"routed_experts": "A".repeat(65536)},
            "unknown_field": [1, 2, 3]
        });
        let experts = response["meta_info"]["routed_experts"].as_str().unwrap();
        let original_ptr = experts.as_ptr();
        let (converted, finished) = output(response, false, &mut None);
        assert!(!finished);
        let response = &converted.engine_data.unwrap()["sglang_response"];
        assert_eq!(
            response["meta_info"]["routed_experts"]
                .as_str()
                .unwrap()
                .as_ptr(),
            original_ptr
        );
        assert_eq!(response["unknown_field"], json!([1, 2, 3]));
    }

    #[test]
    fn native_output_preserves_the_envelope_and_marks_terminals() {
        let chunk = json!({"output_ids": [10], "meta_info": {"finish_reason": null}});
        let (converted, is_terminal) = output(chunk.clone(), false, &mut None);
        assert!(!is_terminal);
        assert_eq!(
            converted.engine_data,
            Some(json!({"sglang_response": chunk}))
        );
        assert!(converted.finish_reason.is_none());
        assert!(converted.disaggregated_params.is_none());

        let terminal = json!({"meta_info": {"finish_reason": {"type": "length"}}});
        let (converted, is_terminal) = output(terminal.clone(), false, &mut None);
        assert!(is_terminal);
        assert_eq!(
            converted.engine_data,
            Some(json!({"sglang_response": terminal}))
        );
        assert_eq!(converted.finish_reason, Some(FinishReason::Length));
        assert!(converted.disaggregated_params.is_none());
    }

    #[test]
    fn native_decode_terminal_reports_typed_finish_and_usage() {
        for (finish, expected) in [
            (json!({"type": "stop", "matched": 2}), FinishReason::Stop),
            (json!({"type": "length"}), FinishReason::Length),
            (json!({"type": "unknown"}), FinishReason::Stop),
            (
                json!({"type": "abort", "message": "x"}),
                FinishReason::Cancelled,
            ),
            (
                json!({"type": "abort", "message": "bad", "status_code": 400}),
                FinishReason::Error("bad".into()),
            ),
            (
                json!({"type": "abort", "message": "", "err_type": "E"}),
                FinishReason::Error("SGLang aborted the request with an error".into()),
            ),
        ] {
            let response = json!({"meta_info": {
                "finish_reason": finish, "prompt_tokens": 4, "completion_tokens": 3,
                "cached_tokens": 2,
            }});
            let (converted, is_terminal) = output(response, false, &mut None);
            assert!(is_terminal, "{finish}");
            assert_eq!(converted.finish_reason, Some(expected), "{finish}");
            assert!(converted.stop_reason.is_none(), "{finish}");
            let usage = converted.completion_usage.unwrap();
            assert_eq!((usage.prompt_tokens, usage.completion_tokens), (4, 3));
            assert_eq!(usage.prompt_tokens_details.unwrap().cached_tokens, Some(2));
        }

        let response = json!({"meta_info": {
            "finish_reason": {"type": "stop"}, "prompt_tokens": 4, "cached_tokens": 0,
        }});
        assert!(
            output(response, false, &mut None)
                .0
                .completion_usage
                .is_none()
        );

        // Prefill terminals keep the untyped stop.
        let response = json!({"meta_info": {
            "finish_reason": {"type": "length"}, "prompt_tokens": 4, "completion_tokens": 1,
        }});
        let (converted, is_terminal) = output(response, true, &mut None);
        assert!(is_terminal);
        assert_eq!(converted.finish_reason, Some(FinishReason::Stop));
        assert!(converted.completion_usage.is_none());
    }

    // AbortRpc's lazy channel needs a runtime.
    #[tokio::test]
    async fn native_http_discovery_checks_port_and_preserves_endpoint_host() {
        let grpc = GrpcEndpoint::parse("http://engine.example:30001", "test").unwrap();
        let discover = |info: &EngineInfo| {
            NativeHttp::discover(
                &grpc,
                info,
                AbortRpc::unreachable(),
                Duration::from_secs(1),
                MAX_EVENT_BYTES,
            )
        };
        // Zero is proto3's unset port: the server serves no native HTTP.
        assert!(discover(&discovery(0)).unwrap().is_none());
        let error = discover(&discovery(65_536)).err().unwrap();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::Unknown)
        );
        let native = discover(&discovery(30000)).unwrap().unwrap();
        assert_eq!(
            native.endpoint.with_path("/generate").as_str(),
            "http://engine.example:30000/generate"
        );
    }

    #[tokio::test]
    async fn readiness_probe_accepts_healthy_http_endpoint() {
        let (port, server) = serve_once(String::new(), "200 OK").await;
        native_http(port)
            .await_ready(
                tokio::time::Instant::now() + Duration::from_secs(1),
                Duration::from_millis(10),
            )
            .await
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_stream_notifies_first_output_and_accepts_sse_fields() {
        let body = concat!(
            "event: message\n",
            "data:{\"output_ids\":[101],\"meta_info\":{\"finish_reason\":null}}\n\n",
            "retry: 1000\n",
            "data: {\"output_ids\":[102],\"meta_info\":{\"finish_reason\":{\"type\":\"stop\",\"matched\":102},",
            "\"prompt_tokens\":1,\"completion_tokens\":2}}\n\n"
        )
        .to_string();
        let (port, server) = serve_once(body, "200 OK").await;
        let (first_token, first_token_seen) = watch::channel(false);
        let ctx = GenerateContext::new(
            dynamo_backend_common::testing::mock_context(),
            Some(first_token),
        );
        let mut stream = native_http(port).generate(
            NativeRequest {
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: false,
                is_decode: false,
                prefill_handoff: None,
                keep_prefill_output: false,
            },
            ctx,
            CancellationToken::new(),
        );

        assert_eq!(stream.next().await.unwrap().unwrap().token_ids, [101]);
        assert!(*first_token_seen.borrow());
        let terminal = stream.next().await.unwrap().unwrap();
        assert_eq!(terminal.token_ids, [102]);
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        assert!(terminal.stop_reason.is_none());
        assert_eq!(terminal.completion_usage.unwrap().total_tokens, 3);
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn frames_without_output_ids_carry_no_tokens() {
        // SGLang omits output_ids while a frame has no tokens.
        let body = concat!(
            "data: {\"text\":\"a\",\"meta_info\":{\"finish_reason\":null}}\n\n",
            "data: {\"text\":\"\",\"meta_info\":{\"finish_reason\":{\"type\":\"length\"},",
            "\"prompt_tokens\":1,\"completion_tokens\":1}}\n\n",
            "data: [DONE]\n\n"
        );
        let (port, server) = serve_once(body.to_string(), "200 OK").await;
        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let outputs: Vec<_> = native_http(port)
            .generate(native_request(false), ctx, CancellationToken::new())
            .collect()
            .await;
        let outputs = outputs.into_iter().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(outputs.len(), 2);
        assert!(outputs.iter().all(|output| output.token_ids.is_empty()));
        assert_eq!(outputs[1].finish_reason, Some(FinishReason::Length));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn error_frames_fail_like_grpc_error_items() {
        let token = "data: {\"text\":\"a\",\"meta_info\":{\"finish_reason\":null},\"output_ids\":[101]}\n\n";
        for (code, expected) in [
            (400, BackendError::InvalidArgument),
            (499, BackendError::Cancelled),
            (503, BackendError::Unknown),
        ] {
            let failure = format!(
                "data: {{\"error\":{{\"message\":\"engine failure\",\"code\":{code}}}}}\n\ndata: [DONE]\n\n"
            );
            // Prefill must fail fast instead of waiting for a terminal frame.
            for is_prefill in [false, true] {
                let (port, server) = serve_once(format!("{token}{failure}"), "200 OK").await;
                let ctx =
                    GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
                let mut stream = native_http(port).generate(
                    native_request(is_prefill),
                    ctx,
                    CancellationToken::new(),
                );
                if is_prefill {
                    let handoff = stream.next().await.unwrap().unwrap();
                    assert!(handoff.disaggregated_params.is_some());
                } else {
                    assert_eq!(stream.next().await.unwrap().unwrap().token_ids, [101]);
                }
                let error = stream.next().await.unwrap().unwrap_err();
                assert_eq!(error.error_type(), ErrorType::Backend(expected), "{code}");
                assert!(
                    error
                        .to_string()
                        .contains(&format!("/generate: engine failure (code {code})")),
                    "{error}"
                );
                assert!(stream.next().await.is_none());
                server.await.unwrap();
            }
        }
    }

    /// SGLang answers a body it cannot parse or decode, or a header override
    /// it rejects, with an HTTP error status and its error object instead of a
    /// stream, even for a streaming request. That maps like an error frame,
    /// with the HTTP status as the code. A body that is not SGLang's error
    /// object keeps the private status error.
    #[tokio::test]
    async fn http_rejections_fail_like_error_frames() {
        let message = "invalid type: string \"x\", expected f32";
        for (status, expected) in [
            ("400 Bad Request", BackendError::InvalidArgument),
            ("503 Service Unavailable", BackendError::Unknown),
        ] {
            let code: u16 = status[..3].parse().unwrap();
            let rejection = json!({"error": {"message": message, "code": code}}).to_string();
            let (port, server) = serve_once(rejection, status).await;
            let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
            let mut stream =
                native_http(port).generate(native_request(false), ctx, CancellationToken::new());
            let error = stream.next().await.unwrap().unwrap_err();
            assert_eq!(error.error_type(), ErrorType::Backend(expected), "{status}");
            assert!(
                error
                    .to_string()
                    .contains(&format!("/generate: {message} (code {code})")),
                "{error}"
            );
            assert!(stream.next().await.is_none());
            server.await.unwrap();
        }

        // A 502..=504 body that is not SGLang's, such as a proxy's, stays a
        // transport failure.
        for (body, status, expected) in [
            (
                "bad request",
                "400 Bad Request",
                BackendError::InvalidArgument,
            ),
            (
                "upstream unavailable",
                "503 Service Unavailable",
                BackendError::CannotConnect,
            ),
        ] {
            let (port, server) = serve_once(body.to_string(), status).await;
            let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
            let mut stream =
                native_http(port).generate(native_request(false), ctx, CancellationToken::new());
            let error = stream.next().await.unwrap().unwrap_err();
            assert_eq!(error.error_type(), ErrorType::Backend(expected), "{status}");
            assert_eq!(
                error.message(),
                format!("SGLang /generate returned HTTP {status}: {body}")
            );
            assert_eq!(error.public_message(), None);
            assert!(stream.next().await.is_none());
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn native_stream_rejects_invalid_output_ids() {
        let body = "data: {\"output_ids\":[-1],\"meta_info\":{\"finish_reason\":null}}\n\n";
        let (port, server) = serve_once(body.to_string(), "200 OK").await;
        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let mut stream = native_http(port).generate(
            NativeRequest {
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: false,
                is_decode: false,
                prefill_handoff: None,
                keep_prefill_output: false,
            },
            ctx,
            CancellationToken::new(),
        );

        let error = stream.next().await.unwrap().unwrap_err();
        assert!(
            error.to_string().contains("invalid SGLang output_ids"),
            "{error}"
        );
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn native_stream_rejects_events_over_configured_limit() {
        let body = "data: {\"output_ids\":[101],\"meta_info\":{\"finish_reason\":null}}\n\n";
        let (port, server) = serve_once(body.to_string(), "200 OK").await;
        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let native = NativeHttp {
            max_event_bytes: 16,
            ..native_http(port)
        };
        let mut stream = native.generate(
            NativeRequest {
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: false,
                is_decode: false,
                prefill_handoff: None,
                keep_prefill_output: false,
            },
            ctx,
            CancellationToken::new(),
        );

        let error = stream.next().await.unwrap().unwrap_err();
        assert!(
            error.to_string().contains("--native-http-max-event-bytes"),
            "unexpected error: {error}"
        );
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn prefill_stream_rejects_backend_error_before_bootstrap() {
        let (port, server) = serve_once("rejected".to_string(), "500 Internal Server Error").await;
        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let mut stream = native_http(port).generate(
            NativeRequest {
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: true,
                is_decode: false,
                prefill_handoff: Some(json!({
                    "bootstrap_host": "prefill",
                    "bootstrap_port": 5000,
                    "bootstrap_room": 7
                })),
                keep_prefill_output: false,
            },
            ctx,
            CancellationToken::new(),
        );

        let error = stream.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("HTTP 500"));
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn prefill_stream_reports_failures_after_success_headers() {
        for (body, expected) in [
            ("", "closed before a terminal response"),
            ("data: broken-json\n\n", "invalid JSON"),
            (
                "data: {\"meta_info\":{\"finish_reason\":{\"type\":\"abort\",\"message\":\"prefill rejected\",\"status_code\":400}}}\n\n",
                "prefill rejected",
            ),
        ] {
            let (port, server) = serve_once(body.to_string(), "200 OK").await;
            let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
            let mut stream = native_http(port).generate(
                NativeRequest {
                    body: json!({"input_ids": [1], "stream": true}),
                    is_prefill: true,
                    is_decode: false,
                    prefill_handoff: Some(json!({
                        "bootstrap_host": "prefill", "bootstrap_port": 1, "bootstrap_room": 7,
                    })),
                    keep_prefill_output: false,
                },
                ctx,
                CancellationToken::new(),
            );
            let handoff = stream.next().await.unwrap().unwrap();
            assert!(handoff.disaggregated_params.is_some());
            assert!(handoff.finish_reason.is_none());
            let error = stream.next().await.unwrap().unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
            assert!(stream.next().await.is_none());
            server.await.unwrap();
        }
    }

    #[tokio::test]
    async fn annotated_prefill_keeps_its_final_native_response() {
        let mut canonical = canonical_request();
        canonical.extra_args = Some(json!({"sglang_tito": {}}));
        canonical.annotations = vec![super::PREFILL_OUTPUT_ANNOTATION.to_string()];
        let mode = DisaggregationMode::Prefill;
        let native = request(&canonical, "rid", mode, Some("prefill"), Some(5000));
        let native = native.unwrap().unwrap();
        let response =
            json!({"output_ids": [101], "meta_info": {"finish_reason": {"type": "length"}}});
        let (port, server) = serve_once(format!("data: {response}\n\n"), "200 OK").await;
        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let mut stream = native_http(port).generate(native, ctx, CancellationToken::new());
        assert!(
            stream
                .next()
                .await
                .unwrap()
                .unwrap()
                .disaggregated_params
                .is_some()
        );
        let terminal = stream.next().await.unwrap().unwrap();
        assert_eq!(
            terminal.engine_data,
            Some(json!({"sglang_response": response}))
        );
        assert!(terminal.token_ids.is_empty());
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn prefill_handoff_waits_for_backend_response_headers() {
        let body =
            "data: {\"output_ids\":[101],\"meta_info\":{\"finish_reason\":{\"type\":\"length\"}}}\n\n"
                .to_string();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let (release_headers_tx, release_headers_rx) = tokio::sync::oneshot::channel();
        let (release_body_tx, release_body_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = socket.read(&mut request).await.unwrap();
            request_seen_tx.send(()).unwrap();
            release_headers_rx.await.unwrap();
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            release_body_rx.await.unwrap();
            socket.write_all(body.as_bytes()).await.unwrap();
        });

        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let mut stream = native_http(port).generate(
            NativeRequest {
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: true,
                is_decode: false,
                prefill_handoff: Some(json!({
                    "bootstrap_host": "prefill",
                    "bootstrap_port": 5000,
                    "bootstrap_room": 7
                })),
                keep_prefill_output: false,
            },
            ctx,
            CancellationToken::new(),
        );

        let mut first_output = Box::pin(stream.next());
        tokio::select! {
            result = request_seen_rx => result.unwrap(),
            output = &mut first_output => panic!(
                "prefill handoff arrived before the backend saw the request: {output:?}"
            ),
        }
        assert!(
            tokio::time::timeout(Duration::from_millis(25), &mut first_output)
                .await
                .is_err(),
            "sending the request alone must not release the prefill handoff"
        );

        release_headers_tx.send(()).unwrap();
        let handoff = tokio::time::timeout(Duration::from_secs(1), first_output)
            .await
            .expect("handoff did not arrive after successful response headers")
            .unwrap()
            .unwrap();
        assert_eq!(handoff.disaggregated_params.unwrap()["bootstrap_room"], 7);

        release_body_tx.send(()).unwrap();
        let terminal = stream.next().await.unwrap().unwrap();
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        assert!(terminal.disaggregated_params.is_none());
        assert!(terminal.engine_data.is_none());
        assert!(terminal.token_ids.is_empty());
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn cancellation_is_a_typed_stream_error() {
        let context = dynamo_backend_common::testing::mock_context();
        context.stop_generating();
        let ctx = GenerateContext::new(context, None);
        let mut stream = native_http(30000).generate(
            NativeRequest {
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: false,
                is_decode: false,
                prefill_handoff: None,
                keep_prefill_output: false,
            },
            ctx,
            CancellationToken::new(),
        );

        let error = stream.next().await.unwrap().unwrap_err();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::Cancelled)
        );
        assert!(stream.next().await.is_none());
    }

    /// Records the rid of every Abort RPC; every other RPC is unimplemented.
    struct AbortRecorder(tokio::sync::mpsc::UnboundedSender<String>);

    #[tonic::async_trait]
    impl pb::sglang_service_server::SglangService for AbortRecorder {
        type GenerateStream =
            futures::stream::BoxStream<'static, Result<pb::GenerateStreamItem, tonic::Status>>;

        async fn generate(
            &self,
            _: tonic::Request<pb::GenerateRequest>,
        ) -> Result<tonic::Response<Self::GenerateStream>, tonic::Status> {
            Err(tonic::Status::unimplemented("Generate"))
        }

        async fn health_check(
            &self,
            _: tonic::Request<pb::HealthCheckRequest>,
        ) -> Result<tonic::Response<pb::HealthCheckResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("HealthCheck"))
        }

        async fn get_model_info(
            &self,
            _: tonic::Request<pb::GetModelInfoRequest>,
        ) -> Result<tonic::Response<pb::GetModelInfoResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("GetModelInfo"))
        }

        async fn get_server_info(
            &self,
            _: tonic::Request<pb::GetServerInfoRequest>,
        ) -> Result<tonic::Response<pb::GetServerInfoResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("GetServerInfo"))
        }

        async fn list_models(
            &self,
            _: tonic::Request<pb::ListModelsRequest>,
        ) -> Result<tonic::Response<pb::ListModelsResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("ListModels"))
        }

        async fn detokenize(
            &self,
            _: tonic::Request<pb::DetokenizeRequest>,
        ) -> Result<tonic::Response<pb::DetokenizeResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("Detokenize"))
        }

        async fn abort(
            &self,
            request: tonic::Request<pb::AbortRequest>,
        ) -> Result<tonic::Response<pb::AbortResponse>, tonic::Status> {
            let _ = self.0.send(request.into_inner().rid);
            Ok(tonic::Response::new(pb::AbortResponse { success: true }))
        }

        async fn get_hi_cache_storage(
            &self,
            _: tonic::Request<pb::GetHiCacheStorageRequest>,
        ) -> Result<tonic::Response<pb::HiCacheStatus>, tonic::Status> {
            Err(tonic::Status::unimplemented("GetHiCacheStorage"))
        }

        async fn attach_hi_cache_storage(
            &self,
            _: tonic::Request<pb::AttachHiCacheStorageRequest>,
        ) -> Result<tonic::Response<pb::HiCacheStorageResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("AttachHiCacheStorage"))
        }

        async fn detach_hi_cache_storage(
            &self,
            _: tonic::Request<pb::DetachHiCacheStorageRequest>,
        ) -> Result<tonic::Response<pb::HiCacheStorageResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("DetachHiCacheStorage"))
        }

        async fn clear_hi_cache_storage(
            &self,
            _: tonic::Request<pb::ClearHiCacheStorageRequest>,
        ) -> Result<tonic::Response<pb::HiCacheStorageResponse>, tonic::Status> {
            Err(tonic::Status::unimplemented("ClearHiCacheStorage"))
        }
    }

    /// An Abort RPC client of a server that records each aborted rid.
    async fn abort_recorder() -> (AbortRpc, tokio::sync::mpsc::UnboundedReceiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (aborts, aborted) = tokio::sync::mpsc::unbounded_channel();
        let incoming = async_stream::stream! {
            loop {
                yield listener.accept().await.map(|(socket, _)| socket);
            }
        };
        tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(pb::sglang_service_server::SglangServiceServer::new(
                    AbortRecorder(aborts),
                ))
                .serve_with_incoming(incoming),
        );
        let channel = tonic::transport::Endpoint::from_shared(format!("http://{address}"))
            .unwrap()
            .connect_lazy();
        (AbortRpc::new(client_from_channel(channel)), aborted)
    }

    #[tokio::test]
    async fn dropped_decode_forces_abort_after_drain_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            // Never answer /generate, like a hung prefill or KV transfer.
            let (mut generate, _) = listener.accept().await.unwrap();
            let _ = generate.read(&mut [0_u8; 4096]).await.unwrap();
            request_seen_tx.send(()).unwrap();
            let _ = release_rx.await;
        });
        let (abort, mut aborted) = abort_recorder().await;

        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let request_id = ctx.id().to_string();
        let native = NativeHttp {
            abort,
            decode_drain_timeout: Duration::from_millis(50),
            ..native_http(port)
        };
        let mut stream = native.generate(
            NativeRequest {
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: false,
                is_decode: true,
                prefill_handoff: None,
                keep_prefill_output: false,
            },
            ctx,
            CancellationToken::new(),
        );
        tokio::select! {
            result = request_seen_rx => result.unwrap(),
            output = stream.next() => panic!("decode returned before the backend responded: {output:?}"),
        }
        drop(stream);

        // SGLang serves no HTTP /abort_request; the Abort RPC stops the request.
        let rid = tokio::time::timeout(Duration::from_secs(5), aborted.recv())
            .await
            .expect("abort was not sent after the drain deadline")
            .unwrap();
        assert_eq!(rid, request_id);
        release_tx.send(()).unwrap();
        server.await.unwrap();
    }

    #[test]
    fn authentication_failures_are_not_client_input_errors() {
        for error in [
            authentication_error("/health", StatusCode::UNAUTHORIZED),
            response_error(StatusCode::FORBIDDEN, "denied".to_string()),
            response_error(
                StatusCode::UNAUTHORIZED,
                json!({"error": {"message": "denied", "code": 401}}).to_string(),
            ),
        ] {
            assert_eq!(
                error.error_type(),
                ErrorType::Backend(BackendError::Unknown)
            );
        }
        assert_eq!(
            response_error(StatusCode::BAD_REQUEST, "bad request".to_string()).error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
    }

    #[test]
    fn detects_native_output_fields() {
        assert!(response_has_output(&json!({"output_ids": [1]})));
        assert!(response_has_output(&json!({"text": "a"})));
        assert!(!response_has_output(&json!({"output_ids": [], "text": ""})));
    }
}
