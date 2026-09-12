// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, Request, Response, StatusCode, header};
use eventsource_stream::{EventStreamError, Eventsource};
use futures::{Stream, StreamExt, stream};
use reqwest::{Client, Url};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_util::sync::{CancellationToken, WaitForCancellationFutureOwned};
use tokio_util::task::TaskTracker;
use uuid::Uuid;

use crate::error::SidecarError;
use crate::metadata::PrefillEndpoint;
use crate::proxy::{strip_proxy_headers, target_url};
use crate::server::PdAdapter;

const SUPPORTED_VERSION: &str = "0.5.19";
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
const MAX_SERVER_INFO_BYTES: usize = 1024 * 1024;
const ABORT_TIMEOUT: Duration = Duration::from_secs(5);

/// OpenAI HTTP P/D adapter for the SGLang 0.5.19 release protocol.
pub struct SglangPdAdapter {
    client: Client,
    decode_engine_url: Url,
    cleanup: TaskTracker,
}

impl SglangPdAdapter {
    pub fn new(
        decode_engine_url: Url,
        connect_timeout: Duration,
        read_timeout: Duration,
    ) -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: Client::builder()
                .no_proxy()
                .redirect(reqwest::redirect::Policy::none())
                .connect_timeout(connect_timeout)
                .read_timeout(read_timeout)
                .build()?,
            decode_engine_url,
            cleanup: TaskTracker::new(),
        })
    }

    async fn dispatch(
        &self,
        request: Request<Body>,
        prefill_endpoint: PrefillEndpoint,
        cancellation: CancellationToken,
    ) -> Result<Response<Body>, SidecarError> {
        let (parts, body) = request.into_parts();
        let bytes = read_request(body).await?;
        let mut payload: Value = serde_json::from_slice(&bytes).map_err(|_| {
            SidecarError::adapter(
                StatusCode::BAD_REQUEST,
                "invalid_pd_request",
                "P/D requests must contain a JSON object",
            )
        })?;
        let object = payload.as_object_mut().ok_or_else(|| {
            SidecarError::adapter(
                StatusCode::BAD_REQUEST,
                "invalid_pd_request",
                "P/D requests must contain a JSON object",
            )
        })?;
        if object.get("n").is_some_and(|n| n.as_u64() != Some(1)) {
            return Err(SidecarError::adapter(
                StatusCode::BAD_REQUEST,
                "unsupported_pd_sampling",
                "SGLang P/D requests currently support only n=1",
            ));
        }
        let streaming = match object.get("stream") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(value)) => *value,
            _ => {
                return Err(SidecarError::adapter(
                    StatusCode::BAD_REQUEST,
                    "invalid_pd_request",
                    "stream must be a boolean",
                ));
            }
        };

        let prefill_url = Url::parse(&format!("http://{prefill_endpoint}"))
            .map_err(|_| protocol_error("Selected prefill endpoint is not a valid HTTP URL"))?;
        let mut headers = parts.headers;
        strip_proxy_headers(&mut headers);
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );
        // Discovery and cancellation only need backend credentials, not
        // inference-specific headers or a client-controlled request identifier.
        let mut control_headers = HeaderMap::new();
        if let Some(authorization) = headers.get(header::AUTHORIZATION) {
            control_headers.insert(header::AUTHORIZATION, authorization.clone());
        }
        let (prefill_info, _) = tokio::try_join!(
            self.server_info(&prefill_url, &control_headers, "prefill"),
            self.server_info(&self.decode_engine_url, &control_headers, "decode"),
        )?;
        let bootstrap_port = prefill_info
            .disaggregation_bootstrap_port
            .filter(|port| *port != 0)
            .ok_or_else(|| {
                protocol_error("Prefill engine did not advertise a valid bootstrap port")
            })?;
        let host = prefill_endpoint.authority().host();
        let host = host
            .strip_prefix('[')
            .and_then(|host| host.strip_suffix(']'))
            .unwrap_or(host);
        let rid = Uuid::new_v4().simple().to_string();
        let room = (Uuid::new_v4().as_u128() & i64::MAX as u128) as u64;
        object.insert("bootstrap_host".into(), json!(host));
        object.insert("bootstrap_port".into(), json!(bootstrap_port));
        object.insert("bootstrap_room".into(), json!(room));
        object.insert("rid".into(), json!(rid));
        let bytes = Bytes::from(serde_json::to_vec(&payload).expect("JSON values serialize"));

        let prefill_guard = self.abort_guard(&prefill_url, &control_headers, &rid, "prefill");
        let decode_guard =
            self.abort_guard(&self.decode_engine_url, &control_headers, &rid, "decode");
        // Both requests are polled together. Prefill completion, not its HTTP
        // headers, is the gate for exposing any decode response to Gateway.
        let prefill = async {
            let mut guard = prefill_guard;
            let response = self
                .client
                .post(target_url(&prefill_url, &parts.uri))
                .headers(headers.clone())
                .body(bytes.clone())
                .send()
                .await
                .map_err(|error| upstream_error(error, "prefill"))?;
            require_inference_success(&response)?;
            let bytes = read_limited(response, MAX_BODY_BYTES, "prefill").await?;
            validate_prefill(bytes, streaming).await?;
            guard.complete();
            Ok::<_, SidecarError>(())
        };
        let decode = async {
            let guard = decode_guard;
            let response = self
                .client
                .post(target_url(&self.decode_engine_url, &parts.uri))
                .headers(headers.clone())
                .body(bytes.clone())
                .send()
                .await
                .map_err(|error| upstream_error(error, "decode"))?;
            require_inference_success(&response)?;
            Ok::<_, SidecarError>((response, guard))
        };
        tokio::pin!(prefill, decode);
        let mut prefill_done = false;
        let (decode_response, mut guard) = tokio::select! {
            result = &mut prefill => {
                result?;
                prefill_done = true;
                decode.await?
            }
            result = &mut decode => result?,
        };
        let status = decode_response.status();
        let mut headers = decode_response.headers().clone();
        strip_proxy_headers(&mut headers);
        let mut decode_bytes: DecodeBytes = Box::pin(decode_response.bytes_stream());
        let mut prefix = Vec::new();
        if !prefill_done {
            // Decode can fail after its 200 headers, while prefill is still
            // transferring KV. Observe both bodies until prefill completes.
            tokio::select! {
                result = &mut prefill => result?,
                result = prefetch_decode(&mut decode_bytes, &mut prefix, streaming) => {
                    result?;
                    guard.complete();
                    prefill.await?;
                }
            }
        }
        let prefix = stream::iter((!prefix.is_empty()).then(|| Ok(Bytes::from(prefix))));
        let body = Body::from_stream(DecodeStream {
            inner: Some(Box::pin(prefix.chain(decode_bytes))),
            guard: Some(guard),
            cancelled: Box::pin(cancellation.cancelled_owned()),
        });
        let mut response = Response::new(body);
        *response.status_mut() = status;
        *response.headers_mut() = headers;
        Ok(response)
    }

    async fn server_info(
        &self,
        base_url: &Url,
        headers: &HeaderMap,
        role: &'static str,
    ) -> Result<ServerInfo, SidecarError> {
        let response = self
            .client
            .get(target_url(base_url, &"/server_info".parse().unwrap()))
            .headers(headers.clone())
            .send()
            .await
            .map_err(|error| upstream_error(error, role))?;
        require_success(&response, role)?;
        let bytes = read_limited(response, MAX_SERVER_INFO_BYTES, role).await?;
        let info: ServerInfo = serde_json::from_slice(&bytes)
            .map_err(|_| protocol_error("Invalid SGLang /server_info response"))?;
        if info.version != SUPPORTED_VERSION || info.disaggregation_mode != role {
            return Err(protocol_error(&format!(
                "The {role} engine must run SGLang {SUPPORTED_VERSION} in {role} mode"
            )));
        }
        Ok(info)
    }

    fn abort_guard(
        &self,
        base_url: &Url,
        headers: &HeaderMap,
        rid: &str,
        role: &'static str,
    ) -> AbortOnDrop {
        AbortOnDrop {
            pending: true,
            client: self.client.clone(),
            cleanup: self.cleanup.clone(),
            url: target_url(base_url, &"/abort_request".parse().unwrap()),
            headers: headers.clone(),
            rid: rid.to_owned(),
            role,
        }
    }
}

#[async_trait]
impl PdAdapter for SglangPdAdapter {
    async fn execute(
        &self,
        request: Request<Body>,
        prefill_endpoint: PrefillEndpoint,
        cancellation: CancellationToken,
    ) -> Result<Response<Body>, SidecarError> {
        tokio::select! {
            result = self.dispatch(request, prefill_endpoint, cancellation.clone()) => result,
            () = cancellation.cancelled() => Err(SidecarError::Cancelled),
        }
    }

    async fn shutdown(&self) {
        self.cleanup.close();
        self.cleanup.wait().await;
    }
}

#[derive(Deserialize)]
struct ServerInfo {
    version: String,
    disaggregation_mode: String,
    disaggregation_bootstrap_port: Option<u16>,
}

fn protocol_error(message: &str) -> SidecarError {
    SidecarError::adapter(StatusCode::BAD_GATEWAY, "sglang_pd_protocol_error", message)
}

fn upstream_error(error: reqwest::Error, role: &str) -> SidecarError {
    let status = if error.is_timeout() {
        StatusCode::GATEWAY_TIMEOUT
    } else {
        StatusCode::BAD_GATEWAY
    };
    SidecarError::adapter(
        status,
        "sglang_pd_upstream_error",
        format!("The {role} engine request failed"),
    )
}

fn require_success(response: &reqwest::Response, role: &str) -> Result<(), SidecarError> {
    if !response.status().is_success() {
        return Err(SidecarError::adapter(
            StatusCode::BAD_GATEWAY,
            "sglang_pd_upstream_error",
            format!("The {role} engine returned HTTP {}", response.status()),
        ));
    }
    Ok(())
}

fn require_inference_success(response: &reqwest::Response) -> Result<(), SidecarError> {
    let status = response.status();
    if status.is_client_error() || status.is_server_error() {
        return Err(SidecarError::EngineRejected {
            status,
            retry_after: response.headers().get(header::RETRY_AFTER).cloned(),
        });
    }
    require_success(response, "P/D")
}

async fn read_limited(
    mut response: reqwest::Response,
    limit: usize,
    role: &str,
) -> Result<Bytes, SidecarError> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| upstream_error(error, role))?
    {
        if chunk.len() > limit.saturating_sub(bytes.len()) {
            return Err(protocol_error(&format!(
                "The {role} engine response exceeded its size limit"
            )));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(bytes))
}

async fn validate_prefill(bytes: Bytes, streaming: bool) -> Result<(), SidecarError> {
    if !streaming {
        return if validate_completion_json(&bytes, "prefill")? {
            Ok(())
        } else {
            Err(protocol_error("Prefill response has no completed choice"))
        };
    }
    validate_event_stream(stream::iter([Ok(bytes)]), "prefill").await
}

async fn prefetch_decode(
    source: &mut DecodeBytes,
    prefix: &mut Vec<u8>,
    streaming: bool,
) -> Result<(), SidecarError> {
    // The parser borrows the source. On prefill completion it is dropped, but
    // all bytes it consumed (including partial events/UTF-8) remain in prefix.
    let mut recorded = source.map(|chunk| {
        let chunk = chunk.map_err(|error| upstream_error(error, "decode"))?;
        if chunk.len() > MAX_BODY_BYTES.saturating_sub(prefix.len()) {
            return Err(protocol_error(
                "Decode prefix exceeded 32 MiB before prefill completed",
            ));
        }
        prefix.extend_from_slice(&chunk);
        Ok(chunk)
    });
    if streaming {
        validate_event_stream(recorded, "decode").await
    } else {
        while let Some(chunk) = recorded.next().await {
            chunk?;
        }
        if !validate_completion_json(prefix, "decode")? {
            return Err(protocol_error("Decode response has no completed choice"));
        }
        Ok(())
    }
}

async fn validate_event_stream<S>(source: S, role: &str) -> Result<(), SidecarError>
where
    S: Stream<Item = Result<Bytes, SidecarError>> + Unpin,
{
    // eventsource-stream 0.2.3 slices a leading BOM at byte 1 and panics.
    // SGLang emits no BOM; reject it before the parser sees its third byte,
    // including when the UTF-8 sequence is split across HTTP chunks.
    let mut first_bytes: Vec<u8> = Vec::new();
    let checked = source.map(|chunk| {
        let chunk = chunk?;
        first_bytes.extend(chunk.iter().take(3 - first_bytes.len()));
        if first_bytes == [0xef, 0xbb, 0xbf] {
            return Err(protocol_error(
                "SGLang event streams must not start with a UTF-8 BOM",
            ));
        }
        Ok(chunk)
    });
    let mut events = checked.eventsource();
    let mut done = false;
    let mut finished = false;
    while let Some(event) = events.next().await {
        let event = event.map_err(|error| match error {
            EventStreamError::Transport(error) => error,
            _ => protocol_error(&format!("Invalid {role} event stream")),
        })?;
        if done {
            return Err(protocol_error(&format!(
                "The {role} engine sent data after [DONE]"
            )));
        }
        if event.data == "[DONE]" {
            done = true;
        } else {
            finished |= validate_completion_json(event.data.as_bytes(), role)?;
        }
    }
    if !done || !finished {
        return Err(protocol_error(&format!(
            "The {role} event stream ended without a completed choice and [DONE]"
        )));
    }
    Ok(())
}

fn validate_completion_json(bytes: &[u8], role: &str) -> Result<bool, SidecarError> {
    let value: Value = serde_json::from_slice(bytes)
        .map_err(|_| protocol_error(&format!("Invalid {role} JSON response")))?;
    if !value.is_object()
        || value.get("object").and_then(Value::as_str) == Some("error")
        || value.get("error").is_some_and(|error| !error.is_null())
        || value
            .get("choices")
            .and_then(Value::as_array)
            .is_some_and(|choices| {
                choices.iter().any(|choice| {
                    choice.get("finish_reason").and_then(Value::as_str) == Some("abort")
                })
            })
    {
        return Err(protocol_error(&format!(
            "The {role} engine reported an error"
        )));
    }
    Ok(value
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("finish_reason")
                    .and_then(Value::as_str)
                    .is_some_and(|reason| {
                        matches!(
                            reason,
                            "stop" | "length" | "tool_calls" | "function_call" | "content_filter"
                        )
                    })
            })
        }))
}

async fn read_request(body: Body) -> Result<Bytes, SidecarError> {
    let mut chunks = body.into_data_stream();
    let mut bytes = Vec::new();
    while let Some(chunk) = chunks.next().await {
        let chunk = chunk.map_err(|_| {
            SidecarError::adapter(
                StatusCode::BAD_REQUEST,
                "invalid_pd_request",
                "Cannot read P/D request body",
            )
        })?;
        if chunk.len() > MAX_BODY_BYTES.saturating_sub(bytes.len()) {
            return Err(SidecarError::adapter(
                StatusCode::PAYLOAD_TOO_LARGE,
                "invalid_pd_request",
                "P/D request exceeds the 32 MiB limit",
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(Bytes::from(bytes))
}

// Each guard owns one leg, including a leg whose HTTP send future is dropped.
// Abort is best effort: SGLang ignores unknown rids, so closing the original
// connection is also required; a successful abort HTTP status is not an ack
// that model execution has stopped.
struct AbortOnDrop {
    pending: bool,
    client: Client,
    cleanup: TaskTracker,
    url: Url,
    headers: HeaderMap,
    rid: String,
    role: &'static str,
}

impl AbortOnDrop {
    fn complete(&mut self) {
        self.pending = false;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if !self.pending {
            return;
        }
        let request = self
            .client
            .post(self.url.clone())
            .headers(self.headers.clone())
            .json(&json!({"rid": self.rid, "abort_all": false}))
            .timeout(ABORT_TIMEOUT);
        let role = self.role;
        self.cleanup.spawn(async move {
            match request.send().await {
                Ok(response) if response.status().is_success() => {}
                Ok(response) => {
                    tracing::warn!(role, status = %response.status(), "SGLang abort was rejected")
                }
                Err(_) => tracing::warn!(role, "SGLang abort request failed"),
            }
        });
    }
}

type DecodeBytes = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

struct DecodeStream {
    // Drop the connection before scheduling the best-effort abort.
    inner: Option<DecodeBytes>,
    guard: Option<AbortOnDrop>,
    cancelled: Pin<Box<WaitForCancellationFutureOwned>>,
}

impl Stream for DecodeStream {
    type Item = Result<Bytes, reqwest::Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if std::future::Future::poll(self.cancelled.as_mut(), cx).is_ready() {
            self.inner.take();
            self.guard.take();
            return Poll::Ready(None);
        }
        let Some(inner) = self.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                if let Some(mut guard) = self.guard.take() {
                    guard.complete();
                }
                self.inner.take();
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(error))) => {
                self.inner.take();
                self.guard.take();
                Poll::Ready(Some(Err(error)))
            }
            result => result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    #[tokio::test]
    async fn accepts_complete_prefill_json_and_sse() {
        validate_prefill(
            Bytes::from_static(br#"{"choices":[{"finish_reason":"length"}]}"#),
            false,
        )
        .await
        .unwrap();
        validate_prefill(
            Bytes::from_static(b": comment\r\ndata: {\"choices\":[{\"finish_reason\":\"length\"}]}\r\n\r\ndata: [DONE]\r\n\r\n"),
            true,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn rejects_prefill_errors_and_incomplete_streams() {
        for (body, streaming) in [
            ("", false),
            ("null", false),
            ("[]", false),
            ("{}", false),
            ("data: [DONE]\n\n", true),
            (r#"{"object":"error","message":"failed"}"#, false),
            (r#"{"error":{"message":"failed"}}"#, false),
            (r#"{"choices":[{"finish_reason":"abort"}]}"#, false),
            (
                "data: {\"error\":{\"message\":\"failed\"}}\n\ndata: [DONE]\n\n",
                true,
            ),
            ("data: {\"choices\":[]}\n\n", true),
            ("data: not-json\n\ndata: [DONE]\n\n", true),
            ("data: [DONE]\n\ndata: {\"choices\":[]}\n\n", true),
        ] {
            assert!(
                validate_prefill(Bytes::from(body), streaming)
                    .await
                    .is_err(),
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn request_body_limit_and_read_errors_are_client_errors() {
        let chunk = Bytes::from(vec![b' '; 1024 * 1024]);
        let body = Body::from_stream(stream::iter(
            (0..33).map(move |_| Ok::<_, std::io::Error>(chunk.clone())),
        ));
        assert_eq!(
            read_request(body)
                .await
                .unwrap_err()
                .into_response()
                .status(),
            StatusCode::PAYLOAD_TOO_LARGE,
        );
        let body = Body::from_stream(stream::iter([Err::<Bytes, _>(std::io::Error::other(
            "client disconnected",
        ))]));
        assert_eq!(
            read_request(body)
                .await
                .unwrap_err()
                .into_response()
                .status(),
            StatusCode::BAD_REQUEST,
        );
    }

    #[tokio::test]
    async fn rejects_bom_without_panicking_even_across_chunks() {
        for chunks in [
            vec![Bytes::from_static(b"\xef\xbb\xbfdata: [DONE]\n\n")],
            vec![
                Bytes::from_static(b"\xef"),
                Bytes::from_static(b"\xbb"),
                Bytes::from_static(b"\xbfdata: [DONE]\n\n"),
            ],
        ] {
            assert!(
                validate_event_stream(stream::iter(chunks.into_iter().map(Ok)), "decode")
                    .await
                    .is_err()
            );
        }
    }
}
