// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming `/generate` transport for native pass-through and multimodal requests.

use std::{collections::HashMap, io, time::Duration};

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, LLMEngineOutput, PreprocessedRequest,
};
use dynamo_sidecar_common::{GrpcEndpoint, HttpEndpoint};
use futures::{StreamExt, TryStreamExt, stream::BoxStream};
use reqwest::{Response, StatusCode, header};
use serde_json::{Map, Value};
use tokio::time::Instant;
use tokio_util::{
    codec::{FramedRead, LinesCodec},
    io::StreamReader,
    sync::CancellationToken,
};

use crate::{client, client::Discovery, multimodal, protocol};

const PAYLOAD_KEY: &str = "sglang_tito";
const MAX_EVENT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct NativeRequest {
    body: Value,
    is_prefill: bool,
    prefill_handoff: Option<Value>,
    token_output: bool,
    return_tokens_as_ids: bool,
}

/// Lower a multimodal request or rebuild an opaque native frontend envelope,
/// replacing fields owned by Dynamo routing.
pub(crate) fn request(
    request: &PreprocessedRequest,
    request_id: &str,
    mode: DisaggregationMode,
    bootstrap_host: Option<&str>,
    bootstrap_port: Option<u16>,
) -> Result<Option<NativeRequest>, DynamoError> {
    let payload = request
        .extra_args
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|extra| extra.get(PAYLOAD_KEY));
    let token_output = payload.is_none();
    let mut body = match payload {
        Some(payload) => payload
            .as_object()
            .cloned()
            .ok_or_else(|| client::invalid_arg("extra_args.sglang_tito must be a JSON object"))?,
        None => match multimodal::request_body(
            request,
            request_id,
            mode,
            bootstrap_host,
            bootstrap_port,
        )? {
            Some(body) => body,
            None => return Ok(None),
        },
    };
    if request.token_ids.is_empty() || request.prompt_embeds.is_some() {
        return Err(client::invalid_arg(
            "native SGLang Generate requires token input",
        ));
    }

    // Native EPD processors may require the rendered text when rebuilding
    // the expanded multimodal prompt. The frontend already supplies it;
    // preserve explicit token-input requests instead of re-rendering them.
    let rendered_prompt = request.extra_args.as_ref().and_then(|extra| {
        if token_output && extra.pointer("/nvext/token_in").and_then(Value::as_bool) != Some(true) {
            extra.get("formatted_prompt").and_then(Value::as_str)
        } else {
            None
        }
    });
    if let Some(prompt) = rendered_prompt {
        body.insert("text".into(), Value::String(prompt.to_string()));
    } else {
        body.insert("input_ids".into(), serde_json::json!(request.token_ids));
    }
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

    let mut trace_headers = HashMap::new();
    dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);
    if !trace_headers.is_empty() {
        body.insert(
            "external_trace_header".into(),
            serde_json::to_value(trace_headers).expect("string map is serializable"),
        );
    }
    if let Some(dp_rank) = protocol::routed_dp_rank(request, mode) {
        body.insert("routed_dp_rank".into(), Value::from(dp_rank));
    } else {
        body.remove("routed_dp_rank");
    }
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
        prefill_handoff,
        token_output,
        return_tokens_as_ids: request
            .output_options
            .return_tokens_as_token_ids
            .unwrap_or(false),
    }))
}

#[derive(Clone)]
pub(crate) struct NativeHttp {
    client: reqwest::Client,
    endpoint: HttpEndpoint,
}

impl NativeHttp {
    pub(crate) fn discover(
        grpc_endpoint: &GrpcEndpoint,
        discovery: &Discovery,
        connect_timeout: Duration,
    ) -> Result<Option<Self>, DynamoError> {
        let Some(raw_port) = discovery.server_info.get("port") else {
            return Ok(None);
        };
        let port = client::json_u64(&discovery.server_info, "port")
            .and_then(|port| u16::try_from(port).ok())
            .filter(|port| *port != 0)
            .ok_or_else(|| {
                client::protocol_error(format!(
                    "SGLang GetServerInfo.port must be in 1..=65535, got {raw_port}"
                ))
            })?;
        if discovery
            .server_info
            .get("incremental_streaming_output")
            .and_then(Value::as_bool)
            != Some(true)
        {
            tracing::warn!(
                port,
                "SGLang native HTTP generation is disabled because incremental streaming output is not enabled"
            );
            return Ok(None);
        }
        let endpoint = HttpEndpoint::from_grpc(grpc_endpoint, port).map_err(|error| {
            client::protocol_error(format!("invalid SGLang HTTP endpoint: {error}"))
        })?;
        let client = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|error| {
                client::invalid_arg(format!("could not configure SGLang HTTP client: {error}"))
            })?;
        Ok(Some(Self { client, endpoint }))
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

    async fn open(&self, body: &Value) -> Result<Response, DynamoError> {
        let response = self
            .client
            .post(self.endpoint.with_path("/generate"))
            .header(header::ACCEPT, "text/event-stream")
            .json(body)
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
            let mut prefill_handoff = request.prefill_handoff;
            tracing::debug!(request_id = %ctx.id(), endpoint = %self.endpoint.with_path("/generate"), "sending native request to SGLang HTTP");
            let opened = tokio::select! {
                biased;
                _ = ctx.stopped() => None,
                _ = cancel.cancelled() => None,
                response = self.open(&request.body) => Some(response),
            };
            let Some(response) = opened else {
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

            let bytes = response.bytes_stream().map_err(io::Error::other);
            let reader = StreamReader::new(bytes);
            let mut lines = FramedRead::new(reader, LinesCodec::new_with_max_length(MAX_EVENT_BYTES));
            let mut first_output_seen = false;
            loop {
                let selected = tokio::select! {
                    biased;
                    _ = ctx.stopped() => None,
                    _ = cancel.cancelled() => None,
                    line = lines.next() => Some(line),
                };
                let Some(line) = selected else {
                    yield Err(client::cancelled(format!(
                        "SGLang native request {} was cancelled",
                        ctx.id()
                    )));
                    return;
                };
                let line = match line {
                    Some(Ok(line)) => line,
                    Some(Err(error)) => {
                        yield Err(client::protocol_error(format!(
                            "invalid SGLang /generate stream: {error}"
                        )));
                        return;
                    }
                    None => {
                        yield Err(client::protocol_error(
                            "SGLang /generate closed before a terminal response",
                        ));
                        return;
                    }
                };
                if line.is_empty() {
                    continue;
                }
                let Some(data) = line.strip_prefix("data:") else {
                    // SSE comments and fields such as event, id, and retry do not
                    // carry the SGLang response payload.
                    continue;
                };
                let data = data.strip_prefix(' ').unwrap_or(data);
                if data.is_empty() {
                    continue;
                }
                if data == "[DONE]" {
                    yield Err(client::protocol_error(
                        "SGLang /generate finished without a terminal response",
                    ));
                    return;
                }
                let response: Value = match serde_json::from_str(data) {
                    Ok(response) => response,
                    Err(error) => {
                        yield Err(client::protocol_error(format!(
                            "SGLang /generate returned invalid JSON: {error}"
                        )));
                        return;
                    }
                };
                // Prefill's raw HTTP payload is not forwarded to the caller.
                // Surface backend failures before that payload is discarded.
                if is_prefill
                    && let Some(finish) = response.pointer("/meta_info/finish_reason")
                    && let Some(kind @ ("abort" | "error" | "cancelled")) =
                        finish.get("type").and_then(Value::as_str)
                {
                    yield Err(protocol::terminal_failure(kind, finish));
                    return;
                }
                let has_output = response_has_output(&response);
                let (mut output, terminal) = if request.token_output {
                    match multimodal::output(response, is_prefill, request.return_tokens_as_ids) {
                        Ok(output) => output,
                        Err(error) => {
                            yield Err(error);
                            return;
                        }
                    }
                } else {
                    output(response, &mut prefill_handoff)
                };
                if !first_output_seen && has_output && (!is_prefill || terminal) {
                    ctx.notify_first_token();
                    first_output_seen = true;
                }
                if is_prefill {
                    if !terminal {
                        continue;
                    }
                    output.engine_data = None;
                }
                yield Ok(output);
                if terminal {
                    return;
                }
            }
        })
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

fn output(response: Value, prefill_handoff: &mut Option<Value>) -> (LLMEngineOutput, bool) {
    let error = response.get("error");
    let finished = error.is_some()
        || response
            .pointer("/meta_info/finish_reason")
            .is_some_and(|reason| !reason.is_null());
    let mut output = match error {
        Some(error) => LLMEngineOutput::error(
            error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("SGLang generation failed")
                .to_string(),
        ),
        None if finished => LLMEngineOutput::stop(),
        None => LLMEngineOutput::default(),
    };
    output.engine_data = Some(serde_json::json!({"sglang_response": response}));
    if finished {
        output.disaggregated_params = prefill_handoff.take();
    }
    (output, finished)
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

fn response_error(status: StatusCode, detail: String) -> DynamoError {
    if matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN) {
        return authentication_error("/generate", status);
    }
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
        NativeHttp, NativeRequest, authentication_error, request, response_error,
        response_has_output,
    };
    use crate::client::Discovery;

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

    fn discovery(server_info: serde_json::Value) -> Discovery {
        Discovery {
            model_path: "model".to_string(),
            tokenizer_path: "tokenizer".to_string(),
            served_model_name: None,
            max_model_len: None,
            model_info: json!({}),
            server_info,
        }
    }

    fn native_http(port: u16) -> NativeHttp {
        let grpc = GrpcEndpoint::parse("127.0.0.1:30001", "test").unwrap();
        NativeHttp {
            client: reqwest::Client::new(),
            endpoint: HttpEndpoint::from_grpc(&grpc, port).unwrap(),
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
    fn prefill_rewrites_rank_and_minimum_generation() {
        let mut canonical = canonical_request();
        canonical.routing = Some(RoutingHints {
            dp_rank: Some(7),
            prefill_dp_rank: Some(3),
            ..Default::default()
        });
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
        assert_eq!(native.body["routed_dp_rank"], 3);
        assert_eq!(native.body["sampling_params"]["max_new_tokens"], 1);
        assert!(
            native.body["sampling_params"]
                .get("min_new_tokens")
                .is_none()
        );
    }

    #[test]
    fn multimodal_uses_rendered_prompt_without_overriding_explicit_token_input() {
        use dynamo_backend_common::MultimodalData;
        let mut canonical = canonical_request();
        canonical.multi_modal_data = Some(std::collections::HashMap::from([(
            "image_url".into(),
            vec![MultimodalData::RawUrl("https://example.com/a.png".into())],
        )]));
        canonical.extra_args = Some(json!({"formatted_prompt": "<image>Describe."}));
        let native = request(
            &canonical,
            "req",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(native.body["text"], "<image>Describe.");
        assert!(native.body.get("input_ids").is_none());
        canonical.extra_args.as_mut().unwrap()["nvext"] = json!({"token_in": true});
        let native = request(
            &canonical,
            "req",
            DisaggregationMode::Aggregated,
            None,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(native.body["input_ids"], json!([1, 2, 3]));
        assert!(native.body.get("text").is_none());
    }

    #[tokio::test]
    async fn multimodal_http_stream_returns_token_deltas_and_prefill_handoff() {
        use dynamo_backend_common::{FinishReason, MultimodalData};
        let body = concat!(
            "data: {\"output_ids\":[101],\"meta_info\":{\"prompt_tokens\":256,\"completion_tokens\":1,\"finish_reason\":null}}\n\n",
            "data: {\"output_ids\":[102],\"meta_info\":{\"prompt_tokens\":256,\"completion_tokens\":2,\"finish_reason\":{\"type\":\"length\"}}}\n\n"
        );
        for mode in [DisaggregationMode::Aggregated, DisaggregationMode::Prefill] {
            let mut canonical = canonical_request();
            canonical.multi_modal_data = Some(std::collections::HashMap::from([(
                "image_url".into(),
                vec![MultimodalData::RawUrl("data:image/png;base64,YQ==".into())],
            )]));
            let native = request(&canonical, "req", mode, Some("p"), Some(8998))
                .unwrap()
                .unwrap();
            assert_eq!(
                native.body["image_data"],
                json!(["data:image/png;base64,YQ=="])
            );
            assert_eq!(native.body["input_ids"], json!([1, 2, 3]));
            assert!(native.token_output);
            let (port, server) = serve_once(body.into(), "200 OK").await;
            let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
            let mut stream = native_http(port).generate(native, ctx, CancellationToken::new());
            let first = stream.next().await.unwrap().unwrap();
            if mode.is_prefill() {
                let handoff = first.disaggregated_params.unwrap();
                assert_eq!(handoff["bootstrap_host"], "p");
                assert_eq!(handoff["bootstrap_port"], 8998);
                assert!(first.token_ids.is_empty());
            } else {
                assert_eq!(first.token_ids, [101]);
            }
            let last = stream.next().await.unwrap().unwrap();
            assert_eq!(last.finish_reason, Some(FinishReason::Length));
            assert_eq!(
                last.token_ids,
                if mode.is_prefill() { vec![] } else { vec![102] }
            );
            assert!(last.engine_data.is_none());
            assert!(stream.next().await.is_none());
            server.await.unwrap();
        }
    }

    #[test]
    fn discovery_requires_incremental_streaming() {
        let grpc = GrpcEndpoint::parse("127.0.0.1:30001", "test").unwrap();
        assert!(
            NativeHttp::discover(
                &grpc,
                &discovery(json!({"port": 30000})),
                Duration::from_secs(1),
            )
            .unwrap()
            .is_none()
        );
        assert!(
            NativeHttp::discover(
                &grpc,
                &discovery(json!({
                    "port": 30000,
                    "incremental_streaming_output": true
                })),
                Duration::from_secs(1),
            )
            .unwrap()
            .is_some()
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
            "data: {\"output_ids\":[102],\"meta_info\":{\"finish_reason\":{\"type\":\"stop\"}}}\n\n"
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
                token_output: false,
                return_tokens_as_ids: false,
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: false,
                prefill_handoff: None,
            },
            ctx,
            CancellationToken::new(),
        );

        assert!(stream.next().await.unwrap().is_ok());
        assert!(*first_token_seen.borrow());
        assert!(stream.next().await.unwrap().is_ok());
        assert!(stream.next().await.is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn prefill_stream_rejects_backend_error_before_bootstrap() {
        let (port, server) = serve_once("rejected".to_string(), "500 Internal Server Error").await;
        let ctx = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
        let mut stream = native_http(port).generate(
            NativeRequest {
                token_output: false,
                return_tokens_as_ids: false,
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: true,
                prefill_handoff: Some(json!({
                    "bootstrap_host": "prefill",
                    "bootstrap_port": 5000,
                    "bootstrap_room": 7
                })),
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
                    token_output: false,
                    return_tokens_as_ids: false,
                    body: json!({"input_ids": [1], "stream": true}),
                    is_prefill: true,
                    prefill_handoff: Some(json!({
                        "bootstrap_host": "prefill", "bootstrap_port": 1, "bootstrap_room": 7,
                    })),
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
                token_output: false,
                return_tokens_as_ids: false,
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: true,
                prefill_handoff: Some(json!({
                    "bootstrap_host": "prefill",
                    "bootstrap_port": 5000,
                    "bootstrap_room": 7
                })),
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
                token_output: false,
                return_tokens_as_ids: false,
                body: json!({"input_ids": [1], "stream": true}),
                is_prefill: false,
                prefill_handoff: None,
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

    #[test]
    fn authentication_failures_are_not_client_input_errors() {
        for error in [
            authentication_error("/health", StatusCode::UNAUTHORIZED),
            response_error(StatusCode::FORBIDDEN, "denied".to_string()),
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
