// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque transport for SGLang's native streaming `/generate` API.

use std::{collections::HashMap, io, time::Duration};

use dynamo_backend_common::{
    DisaggregationMode, DynamoError, GenerateContext, LLMEngineOutput, PreprocessedRequest,
};
use dynamo_sidecar_common::GrpcEndpoint;
use futures::{StreamExt, TryStreamExt, stream::BoxStream};
use reqwest::{Response, Url, header};
use serde_json::{Map, Value};
use tokio_util::{
    codec::{FramedRead, LinesCodec},
    io::StreamReader,
    sync::CancellationToken,
};

use crate::{client, client::Discovery, protocol};

const PAYLOAD_KEY: &str = "sglang_tito";
const MAX_EVENT_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct NativeRequest {
    body: Value,
    prefill_handoff: Option<Value>,
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
    body.entry("rid")
        .or_insert_with(|| Value::String(request_id.to_string()));
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
    }

    let mut trace_headers = HashMap::new();
    dynamo_runtime::logging::inject_trace_headers_into_map(&mut trace_headers);
    if !trace_headers.is_empty() {
        body.insert(
            "external_trace_header".into(),
            serde_json::to_value(trace_headers).expect("string map is serializable"),
        );
    }
    if let Some(dp_rank) = routing.and_then(|routing| routing.dp_rank) {
        body.insert("routed_dp_rank".into(), Value::from(dp_rank));
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
        prefill_handoff,
    }))
}

#[derive(Clone)]
pub(crate) struct NativeHttp {
    client: reqwest::Client,
    endpoint: Url,
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
        let endpoint = Url::parse(&format!(
            "http://{}:{port}/generate",
            grpc_endpoint.authority_host()
        ))
        .map_err(|error| {
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

    async fn open(&self, body: &Value) -> Result<Response, DynamoError> {
        let response = self
            .client
            .post(self.endpoint.clone())
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
        let message = format!("SGLang /generate returned HTTP {status}: {detail}");
        if status.is_client_error() {
            Err(client::invalid_arg(message))
        } else if matches!(status.as_u16(), 502..=504) {
            Err(client::cannot_connect(message))
        } else {
            Err(client::protocol_error(message))
        }
    }

    pub(crate) fn generate(
        self,
        request: NativeRequest,
        ctx: GenerateContext,
        cancel: CancellationToken,
    ) -> BoxStream<'static, Result<LLMEngineOutput, DynamoError>> {
        Box::pin(async_stream::stream! {
            tracing::debug!(request_id = %ctx.id(), endpoint = %self.endpoint, "sending native request to SGLang HTTP");
            let response = tokio::select! {
                biased;
                _ = ctx.stopped() => return,
                _ = cancel.cancelled() => return,
                response = self.open(&request.body) => response,
            };
            let response = match response {
                Ok(response) => response,
                Err(error) => {
                    yield Err(error);
                    return;
                }
            };

            let bytes = response.bytes_stream().map_err(io::Error::other);
            let reader = StreamReader::new(bytes);
            let mut lines = FramedRead::new(reader, LinesCodec::new_with_max_length(MAX_EVENT_BYTES));
            let mut prefill_handoff = request.prefill_handoff;
            loop {
                let line = tokio::select! {
                    biased;
                    _ = ctx.stopped() => return,
                    _ = cancel.cancelled() => return,
                    line = lines.next() => line,
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
                let Some(data) = line.strip_prefix("data: ") else {
                    yield Err(client::protocol_error(format!(
                        "SGLang /generate returned an unexpected SSE line: {line}"
                    )));
                    return;
                };
                if data == "[DONE]" {
                    yield Err(client::protocol_error(
                        "SGLang /generate finished without a terminal response",
                    ));
                    return;
                }
                let response = match serde_json::from_str(data) {
                    Ok(response) => response,
                    Err(error) => {
                        yield Err(client::protocol_error(format!(
                            "SGLang /generate returned invalid JSON: {error}"
                        )));
                        return;
                    }
                };
                let (output, terminal) = output(response, &mut prefill_handoff);
                yield Ok(output);
                if terminal {
                    return;
                }
            }
        })
    }
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
