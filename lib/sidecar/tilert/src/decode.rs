// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HTTP client for TileRT's decode server.
//!
//! The server takes one request at a time. `/pd/decode` claims the KV the
//! prefill pushed for `rid` and answers with newline-delimited JSON: token
//! batches `{"t":[...]}`, then one `{"done":true,...}` or `{"error":...}`.

use std::time::Duration;

use dynamo_backend_common::{BackendError, DynamoError};
use dynamo_sidecar_common::{HttpEndpoint, cannot_connect, connection_timeout, invalid_argument};
use futures::{Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use tokio::time::Instant;
use tokio_util::codec::{FramedRead, LinesCodec, LinesCodecError};
use tokio_util::io::StreamReader;

use crate::error::{backend_error, is_kind};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(5);
/// How long the server may take to answer `/pd/decode` beyond its own wait
/// for the KV.
const RESPONSE_SLACK: Duration = Duration::from_secs(10);
/// How long a release waits for KV that is still landing. KV that arrives
/// later is discarded by the server, which remembers the id a claim gave up on.
const RELEASE_WAIT: Duration = Duration::from_secs(1);
const MAX_LINE_BYTES: usize = 1 << 20;

#[derive(Serialize, Clone, Debug)]
pub(crate) struct DecodeBody {
    pub rid: String,
    pub first_token_id: u32,
    pub max_tokens: u32,
    pub sampling: serde_json::Map<String, serde_json::Value>,
    pub timeout_s: f64,
}

#[derive(Serialize)]
struct DecodeRequest<'a> {
    #[serde(flatten)]
    body: &'a DecodeBody,
    stream: bool,
}

impl DecodeBody {
    fn response_deadline(&self) -> Duration {
        Duration::from_secs_f64(self.timeout_s.max(0.0)) + RESPONSE_SLACK
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeFinish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum DecodeEvent {
    Tokens(Vec<u32>),
    Done {
        finish: DecodeFinish,
        declared_tokens: Option<u64>,
        phases: Option<serde_json::Value>,
    },
}

/// What the decode server's model profile can honour beyond plain
/// temperature sampling.
#[derive(Deserialize, Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct DecodeCapabilities {
    #[serde(default)]
    pub penalties: bool,
    #[serde(default)]
    pub ignore_eos: bool,
}

#[derive(Deserialize)]
struct CapabilitiesReply {
    capabilities: DecodeCapabilities,
}

#[derive(Deserialize)]
struct RawLine {
    t: Option<Vec<u32>>,
    done: Option<bool>,
    n: Option<u64>,
    finish_reason: Option<String>,
    phase_ms: Option<serde_json::Value>,
    error: Option<serde_json::Value>,
    error_type: Option<String>,
}

#[derive(Deserialize)]
struct BlockingReply {
    token_ids: Vec<u32>,
    #[serde(default)]
    phase_ms: Option<serde_json::Value>,
}

pub(crate) fn parse_line(line: &str) -> Result<Option<DecodeEvent>, DynamoError> {
    let line = line.trim();
    if line.is_empty() {
        return Ok(None);
    }
    let raw: RawLine = serde_json::from_str(line).map_err(|error| {
        backend_error(
            BackendError::Unknown,
            format!("TileRT decode sent a line that is not JSON ({error}): {line:.200}"),
        )
    })?;
    if let Some(error) = raw.error {
        return Err(node_error(raw.error_type.as_deref(), &error));
    }
    if let Some(tokens) = raw.t {
        return Ok(Some(DecodeEvent::Tokens(tokens)));
    }
    if raw.done == Some(true) {
        let finish = match raw.finish_reason.as_deref() {
            Some("length") => DecodeFinish::Length,
            Some("cancelled") => DecodeFinish::Cancelled,
            Some("stop") | None => DecodeFinish::Stop,
            Some(other) => {
                tracing::warn!(
                    finish_reason = other,
                    "unknown TileRT finish reason; reporting stop"
                );
                DecodeFinish::Stop
            }
        };
        return Ok(Some(DecodeEvent::Done {
            finish,
            declared_tokens: raw.n,
            phases: raw.phase_ms,
        }));
    }
    Ok(None)
}

fn node_error(error_type: Option<&str>, error: &serde_json::Value) -> DynamoError {
    let message = match error {
        serde_json::Value::String(message) => message.clone(),
        other => other.to_string(),
    };
    let message = format!("TileRT decode error: {message}");
    match error_type {
        Some("invalid_parameter" | "capability_unavailable") => invalid_argument(message),
        Some("request_cancelled") => backend_error(BackendError::Cancelled, message),
        _ => backend_error(BackendError::Unknown, message),
    }
}

fn status_error(status: reqwest::StatusCode, body: &str) -> DynamoError {
    let message = format!("TileRT decode answered HTTP {status}: {body:.500}");
    match status.as_u16() {
        // 501 is the server's answer to a parameter this model's decode cannot honour.
        400 | 501 => invalid_argument(message),
        // The KV for this rid never arrived.
        504 => connection_timeout(message),
        _ => backend_error(BackendError::Unknown, message),
    }
}

fn transport_error(what: &str, error: reqwest::Error) -> DynamoError {
    let message = format!("TileRT decode {what} failed: {error}");
    if error.is_connect() {
        cannot_connect(message)
    } else if error.is_timeout() {
        backend_error(BackendError::ResponseTimeout, message)
    } else {
        backend_error(BackendError::Disconnected, message)
    }
}

#[derive(Clone)]
pub(crate) struct DecodeClient {
    http: reqwest::Client,
    base: HttpEndpoint,
}

impl DecodeClient {
    pub(crate) fn new(base: HttpEndpoint) -> Result<Self, DynamoError> {
        let http = reqwest::Client::builder()
            // Cluster proxies must not intercept node-to-node calls.
            .no_proxy()
            .tcp_nodelay(true)
            .connect_timeout(CONTROL_TIMEOUT)
            .build()
            .map_err(|error| {
                backend_error(
                    BackendError::Unknown,
                    format!("cannot build the TileRT decode HTTP client: {error}"),
                )
            })?;
        Ok(Self { http, base })
    }

    pub(crate) async fn wait_healthy(
        &self,
        deadline: Instant,
        retry_interval: Duration,
    ) -> Result<(), DynamoError> {
        loop {
            let last = match self
                .http
                .get(self.base.with_path("/health"))
                .timeout(CONTROL_TIMEOUT)
                .send()
                .await
            {
                Ok(response) if response.status().is_success() => return Ok(()),
                Ok(response) => format!("HTTP {}", response.status()),
                Err(error) => error.to_string(),
            };
            if Instant::now() + retry_interval >= deadline {
                return Err(cannot_connect(format!(
                    "TileRT decode at {} did not become healthy before the startup deadline: {last}",
                    self.base
                )));
            }
            tokio::time::sleep(retry_interval).await;
        }
    }

    pub(crate) async fn capabilities(&self) -> Result<DecodeCapabilities, DynamoError> {
        let path = "/capabilities";
        let response = self
            .http
            .get(self.base.with_path(path))
            .timeout(CONTROL_TIMEOUT)
            .send()
            .await
            .map_err(|error| transport_error(path, error))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(status_error(status, &body));
        }
        let reply: CapabilitiesReply = response
            .json()
            .await
            .map_err(|error| transport_error(path, error))?;
        Ok(reply.capabilities)
    }

    /// Stops a decode the server has already started. A request it has not
    /// claimed yet has nothing to cancel and answers 404.
    pub(crate) async fn cancel(&self, rid: &str) {
        let result = self
            .http
            .post(self.base.with_path("/pd/cancel"))
            .timeout(CONTROL_TIMEOUT)
            .json(&serde_json::json!({ "rid": rid }))
            .send()
            .await;
        match result {
            Ok(response) if response.status().is_success() => {}
            Ok(response) => {
                tracing::debug!(rid, status = %response.status(), "TileRT /pd/cancel found nothing to cancel")
            }
            Err(error) => tracing::warn!(rid, %error, "TileRT /pd/cancel failed"),
        }
    }

    /// Frees the KV of a request that will not be decoded, and returns whether
    /// there was any. The server holds unclaimed KV for two minutes and
    /// refuses every other request's KV meanwhile, so a one-token decode
    /// claims it and lets it go.
    pub(crate) async fn release(&self, rid: &str) -> bool {
        let released = self
            .decode_blocking(DecodeBody {
                rid: rid.to_string(),
                first_token_id: 0,
                max_tokens: 1,
                sampling: serde_json::Map::new(),
                timeout_s: RELEASE_WAIT.as_secs_f64(),
            })
            .await;
        match released {
            Ok(_) => {
                tracing::debug!(rid, "released the KV of a request that was not decoded");
                true
            }
            // Nothing had landed; the server now discards this rid's KV on arrival.
            Err(error) if is_kind(&error, BackendError::ConnectionTimeout) => false,
            Err(error) => {
                tracing::warn!(rid, %error, "could not release the KV of a request that was not decoded");
                false
            }
        }
    }

    /// Polls `/health` until the server misses `failures` checks in a row,
    /// and returns why. One slow answer under load is not a lost server.
    pub(crate) async fn watch_health(&self, interval: Duration, failures: u32) -> String {
        let mut missed = 0;
        loop {
            tokio::time::sleep(interval).await;
            let check = self
                .http
                .get(self.base.with_path("/health"))
                .timeout(CONTROL_TIMEOUT)
                .send()
                .await;
            let last = match check {
                Ok(response) if response.status().is_success() => {
                    missed = 0;
                    continue;
                }
                Ok(response) => format!("HTTP {}", response.status()),
                Err(error) => error.to_string(),
            };
            missed += 1;
            if missed >= failures {
                return format!("{missed} health checks in a row failed: {last}");
            }
        }
    }

    async fn post_decode(
        &self,
        body: &DecodeBody,
        stream: bool,
    ) -> Result<reqwest::Response, DynamoError> {
        let send = self
            .http
            .post(self.base.with_path("/pd/decode"))
            .json(&DecodeRequest { body, stream })
            .send();
        let response = tokio::time::timeout(body.response_deadline(), send)
            .await
            .map_err(|_| {
                backend_error(
                    BackendError::ResponseTimeout,
                    format!(
                        "TileRT decode did not answer /pd/decode within {:?}",
                        body.response_deadline()
                    ),
                )
            })?
            .map_err(|error| transport_error("/pd/decode", error))?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(status_error(status, &text));
        }
        Ok(response)
    }

    pub(crate) async fn decode_blocking(
        &self,
        body: DecodeBody,
    ) -> Result<(Vec<u32>, Option<serde_json::Value>), DynamoError> {
        let response = self.post_decode(&body, false).await?;
        let reply: BlockingReply = tokio::time::timeout(RESPONSE_SLACK, response.json())
            .await
            .map_err(|_| {
                backend_error(
                    BackendError::ResponseTimeout,
                    "TileRT decode stalled while sending a non-streaming reply",
                )
            })?
            .map_err(|error| transport_error("/pd/decode", error))?;
        Ok((reply.token_ids, reply.phase_ms))
    }

    pub(crate) async fn decode_stream(
        &self,
        body: DecodeBody,
    ) -> Result<impl Stream<Item = Result<DecodeEvent, DynamoError>> + Send + 'static, DynamoError>
    {
        let response = self.post_decode(&body, true).await?;
        let bytes = response.bytes_stream().map_err(std::io::Error::other);
        let mut lines = Box::pin(FramedRead::new(
            StreamReader::new(bytes),
            LinesCodec::new_with_max_length(MAX_LINE_BYTES),
        ));
        Ok(async_stream::stream! {
            while let Some(line) = lines.next().await {
                let event = match line {
                    Ok(line) => parse_line(&line),
                    Err(LinesCodecError::MaxLineLengthExceeded) => Err(backend_error(
                        BackendError::Unknown,
                        format!("TileRT decode sent a line longer than {MAX_LINE_BYTES} bytes"),
                    )),
                    Err(LinesCodecError::Io(error)) => Err(backend_error(
                        BackendError::Disconnected,
                        format!("TileRT decode stream failed: {error}"),
                    )),
                };
                match event {
                    Ok(Some(event)) => yield Ok(event),
                    Ok(None) => {}
                    Err(error) => {
                        yield Err(error);
                        return;
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_token_and_done_lines() {
        assert_eq!(
            parse_line(r#"{"t":[5,6,7]}"#).unwrap(),
            Some(DecodeEvent::Tokens(vec![5, 6, 7]))
        );
        assert_eq!(parse_line("  ").unwrap(), None);
        let done = parse_line(
            r#"{"done":true,"n":3,"seq_len":10,"finish_reason":"length","phase_ms":{"decode":1.5}}"#,
        )
        .unwrap();
        assert_eq!(
            done,
            Some(DecodeEvent::Done {
                finish: DecodeFinish::Length,
                declared_tokens: Some(3),
                phases: Some(serde_json::json!({"decode": 1.5})),
            })
        );
        for (line, expected) in [
            (
                r#"{"done":true,"finish_reason":"stop"}"#,
                DecodeFinish::Stop,
            ),
            (r#"{"done":true}"#, DecodeFinish::Stop),
            (
                r#"{"done":true,"finish_reason":"cancelled"}"#,
                DecodeFinish::Cancelled,
            ),
        ] {
            let Some(DecodeEvent::Done { finish, .. }) = parse_line(line).unwrap() else {
                panic!("{line} is a terminal message");
            };
            assert_eq!(finish, expected, "{line}");
        }
    }

    /// The error kind decides the HTTP status the client sees and whether the
    /// request may be retried on another worker.
    #[test]
    fn decode_failures_map_to_error_kinds() {
        for (line, kind) in [
            (
                r#"{"error":"top_p < 1 is not supported","error_type":"invalid_parameter"}"#,
                BackendError::InvalidArgument,
            ),
            (
                r#"{"error":"no penalty pre-pass","error_type":"capability_unavailable"}"#,
                BackendError::InvalidArgument,
            ),
            (r#"{"error":"decode stalled"}"#, BackendError::Unknown),
            (r#"{"error":{"code":7}}"#, BackendError::Unknown),
            ("not json", BackendError::Unknown),
        ] {
            let error = parse_line(line).unwrap_err();
            assert!(is_kind(&error, kind), "{line}: {error}");
        }
        for (status, kind) in [
            (400, BackendError::InvalidArgument),
            (501, BackendError::InvalidArgument),
            (504, BackendError::ConnectionTimeout),
            (500, BackendError::Unknown),
        ] {
            let status = reqwest::StatusCode::from_u16(status).unwrap();
            let error = status_error(status, "{}");
            assert!(is_kind(&error, kind), "{status}: {error}");
        }
    }
}
