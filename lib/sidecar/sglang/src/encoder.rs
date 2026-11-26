// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The Dynamo encode worker for SGLang's native HTTP encoder, and the
//! encoder result its language workers forward to SGLang.
//!
//! The encode worker sends the encoder one `/encode` request per modality,
//! under request IDs `{request_id}_local_part_{index}`. `/encode` answers only
//! after the embedding reached a language worker, so the encode terminal is
//! yielded while those requests are still open. The language worker then
//! registers its receive address with the encoder under the same request ID.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use dynamo_backend_common::{
    BackendError, DynamoError, EngineConfig, ErrorType, GenerateContext, LLMEngineOutput,
    PreprocessedRequest,
};
use dynamo_llm::kv_router::encoder_router::ENCODER_RELEASE;
use dynamo_llm::local_model::runtime_config::ENCODER_HANDOFF_CONTRACT;
use dynamo_sidecar_common::{GrpcEndpoint, GrpcTransportConfig, HttpEndpoint};
use futures::stream::BoxStream;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;
use tokio::task::AbortHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::client;
use crate::protocol;

/// How long an encode may fail before its terminal is yielded. Covers an
/// unreachable encoder and a rejected request, which fail at once.
const ENCODE_FAILURE_GRACE: Duration = Duration::from_millis(200);

/// Above the native encoder's own 180 s limit, so its error arrives first.
const ENCODE_TIMEOUT: Duration = Duration::from_secs(200);

/// `/health` runs a dummy encode on an idle encoder, so it is polled less
/// often than a language worker's metadata RPC.
const HEALTH_POLL_INTERVAL: Duration = Duration::from_secs(5);

/// The contract every SGLang encode worker and every language worker that
/// routes to one advertises.
const CONTRACT: &str = "native-encode-v1";

/// Mark `config` as part of the external-encoder deployment.
pub(crate) fn advertise(config: &mut EngineConfig) {
    config.runtime_data.insert(
        ENCODER_HANDOFF_CONTRACT.into(),
        serde_json::json!({ "sglang": CONTRACT }),
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Modality {
    Image,
    Video,
    Audio,
}

impl Modality {
    /// The order in which a request's parts are numbered.
    const ALL: [Self; 3] = [Self::Image, Self::Video, Self::Audio];

    fn media_key(self) -> &'static str {
        match self {
            Self::Image => "image_url",
            Self::Video => "video_url",
            Self::Audio => "audio_url",
        }
    }

    /// The native encoder's `modality` value.
    fn native(self) -> &'static str {
        match self {
            Self::Image => "IMAGE",
            Self::Video => "VIDEO",
            Self::Audio => "AUDIO",
        }
    }
}

/// One modality of a request: one native encode request.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Part {
    modality: Modality,
    items: usize,
}

/// The encode terminal's `encoder_result`.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    /// Where the language worker reaches the encoder, as `http://host:port`.
    encoder_url: String,
    /// The ID the parts were encoded under, which SGLang must get as `rid`.
    request_id: String,
    /// In part-index order.
    parts: Vec<Part>,
}

impl Descriptor {
    fn parse(result: &Value) -> Result<Self, DynamoError> {
        let descriptor = Self::deserialize(result)
            .map_err(|e| client::invalid_arg(format!("invalid SGLang encoder_result: {e}")))?;
        let mut seen = [false; 3];
        let distinct = descriptor.parts.iter().all(|part| {
            part.items > 0 && !std::mem::replace(&mut seen[part.modality as usize], true)
        });
        if !distinct
            || descriptor.parts.is_empty()
            || descriptor.request_id.is_empty()
            || !descriptor.encoder_url.starts_with("http://")
        {
            return Err(client::invalid_request(
                "invalid SGLang encoder_result request, encoder, or parts",
            ));
        }
        Ok(descriptor)
    }

    /// Item counts as `[images, videos, audios]`.
    fn counts(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for part in &self.parts {
            counts[part.modality as usize] = part.items;
        }
        counts
    }
}

/// SGLang's `encoder_handoff` for the encoder result of a request.
pub(crate) fn handoff(result: &Value) -> Result<Value, DynamoError> {
    let descriptor = Descriptor::parse(result)?;
    Ok(serde_json::json!({
        "encoder_url": descriptor.encoder_url,
        "parts": descriptor.parts,
    }))
}

/// Count original source items as `[images, videos, audios]`. Each source
/// list preserves its own placeholder order.
fn media_counts(request: &PreprocessedRequest) -> Result<[usize; 3], DynamoError> {
    let Some(media) = &request.multi_modal_data else {
        return Ok([0; 3]);
    };
    if media.iter().any(|(key, items)| {
        !Modality::ALL.iter().any(|m| m.media_key() == key.as_str()) && !items.is_empty()
    }) {
        return Err(client::invalid_request(
            "SGLang encoders require image, video, or audio URLs",
        ));
    }
    Ok(Modality::ALL.map(|m| media.get(m.media_key()).map_or(0, Vec::len)))
}

/// Check what a language worker received for `request_id`. `required` is
/// `--route-to-encoder`: SGLang then takes media only through an encoder.
pub(crate) fn validate_consumer(
    request: &PreprocessedRequest,
    request_id: &str,
    required: bool,
) -> Result<(), DynamoError> {
    if let Some(result) = &request.encoder_result {
        if !required {
            return Err(client::invalid_request(
                "this SGLang worker consumes encoder_result only with --route-to-encoder",
            ));
        }
        let descriptor = Descriptor::parse(result)?;
        if descriptor.request_id != request_id {
            return Err(client::invalid_request(
                "encoder_result was produced for another request",
            ));
        }
        // SGLang counts placeholders in the token IDs. When Dynamo retains
        // the sources, also check each modality's count.
        if request.multi_modal_data.is_some() && media_counts(request)? != descriptor.counts() {
            return Err(client::invalid_request(
                "encoder_result does not match the request's media counts",
            ));
        }
    } else if required
        && (request
            .multi_modal_data
            .iter()
            .flat_map(|m| m.values())
            .any(|v| !v.is_empty())
            || request
                .extra_args
                .as_ref()
                .and_then(|v| v.get(dynamo_sglang_generate::PAYLOAD_KEY))
                .is_some_and(|body| {
                    ["image_data", "video_data", "audio_data"]
                        .iter()
                        .any(|key| {
                            body.get(*key).is_some_and(|v| {
                                !v.is_null() && v.as_array().is_none_or(|v| !v.is_empty())
                            })
                        })
                }))
    {
        // A caller that bypasses the encoder router still needs a result.
        media_counts(request)?;
        for source in request
            .multi_modal_data
            .iter()
            .flat_map(|m| m.values())
            .flatten()
        {
            protocol::media_source(source)?;
        }
        return Err(client::cannot_connect(
            "multimodal request requires a successful Dynamo encoder_result",
        ));
    }
    Ok(())
}

/// The native encoder's `/encode` body.
#[derive(Debug, Serialize)]
struct EncodeRequest {
    mm_items: Vec<String>,
    num_parts: usize,
    part_idx: usize,
    req_id: String,
    modality: &'static str,
    // The language worker registers its own receive address, so these stay null.
    prefill_host: Option<String>,
    embedding_port: Option<u16>,
}

/// One encode request per modality the request carries, numbered in
/// [`Modality::ALL`] order, and the parts they announce.
fn encode_requests(
    request: PreprocessedRequest,
    request_id: &str,
) -> Result<(Vec<Part>, Vec<EncodeRequest>), DynamoError> {
    let counts = media_counts(&request)?;
    if counts == [0; 3] || request.encoder_result.is_some() {
        return Err(client::invalid_request(
            "SGLang encode requires unencoded media",
        ));
    }
    protocol::validate_request(&request)?;
    if request.media_io_kwargs.is_some() {
        return Err(client::invalid_request(
            "the SGLang encoder cannot apply media_io_kwargs",
        ));
    }
    let num_parts = counts.iter().filter(|count| **count > 0).count();
    let mut media = request.multi_modal_data.unwrap_or_default();
    let mut parts = Vec::with_capacity(num_parts);
    let mut requests = Vec::with_capacity(num_parts);
    for modality in Modality::ALL {
        let sources = media.remove(modality.media_key()).unwrap_or_default();
        if sources.is_empty() {
            continue;
        }
        let mm_items = sources
            .into_iter()
            .map(protocol::into_media_source)
            .collect::<Result<Vec<_>, _>>()?;
        let part_idx = requests.len();
        parts.push(Part {
            modality,
            items: mm_items.len(),
        });
        requests.push(EncodeRequest {
            mm_items,
            num_parts,
            part_idx,
            req_id: format!("{request_id}_local_part_{part_idx}"),
            modality: modality.native(),
            prefill_host: None,
            embedding_port: None,
        });
    }
    Ok((parts, requests))
}

/// `http://host:port` at which language workers reach the encoder.
fn advertised_url(raw: &str) -> Result<String, DynamoError> {
    let endpoint = GrpcEndpoint::parse(raw, "--encoder-advertise-endpoint")?;
    if !endpoint.as_str().starts_with("http://") {
        return Err(client::invalid_arg(
            "an encoder must advertise a TCP endpoint reachable by its language workers",
        ));
    }
    if matches!(endpoint.authority_host(), "0.0.0.0" | "[::]" | "*") {
        return Err(client::invalid_arg(
            "cannot advertise a wildcard encoder address",
        ));
    }
    let port = url::Url::parse(endpoint.as_str())
        .ok()
        .and_then(|u| u.port_or_known_default())
        .filter(|port| *port != 0)
        .ok_or_else(|| client::invalid_arg("invalid encoder port"))?;
    // GrpcEndpoint normalizes the default HTTP port away. SGLang matches the
    // URL against its --encoder-urls as text, so the port stays explicit.
    Ok(format!("http://{}:{port}", endpoint.authority_host()))
}

/// Each running encode's token and task, by request ID.
type Inflight = Arc<Mutex<HashMap<String, (u64, AbortHandle)>>>;

/// Each critical section is one map operation, so a poisoned lock is still
/// safe to use.
fn lock(inflight: &Inflight) -> MutexGuard<'_, HashMap<String, (u64, AbortHandle)>> {
    inflight.lock().unwrap_or_else(|e| e.into_inner())
}

/// Removes a running encode's entry when its task ends or is aborted.
struct Entry {
    inflight: Inflight,
    request_id: String,
    token: u64,
}

impl Drop for Entry {
    fn drop(&mut self) {
        let mut inflight = lock(&self.inflight);
        // A later encode of the same request ID owns the entry.
        if inflight
            .get(&self.request_id)
            .is_some_and(|(token, _)| *token == self.token)
        {
            inflight.remove(&self.request_id);
        }
    }
}

/// Aborts an encode whose terminal was never yielded.
struct AbortUnlessYielded(Option<AbortHandle>);

impl AbortUnlessYielded {
    fn abort(&mut self) {
        if let Some(task) = self.0.take() {
            task.abort();
        }
    }
}

impl Drop for AbortUnlessYielded {
    fn drop(&mut self) {
        self.abort();
    }
}

/// What the wait for an early encode failure saw.
enum Early {
    Stopped,
    /// The encode ended; `Err` when its task was aborted.
    Ended(Result<Result<(), DynamoError>, oneshot::error::RecvError>),
    Running,
}

pub(crate) struct Encoder {
    model_path: String,
    served_model_name: Option<String>,
    endpoint: HttpEndpoint,
    /// `encoder_url` of every descriptor.
    advertised: String,
    http: reqwest::Client,
    inflight: Inflight,
    next_token: AtomicU64,
}

impl Encoder {
    /// `endpoint` is the encoder's HTTP listener. `advertised` defaults to it.
    pub(crate) fn new(
        endpoint: &GrpcEndpoint,
        advertised: Option<&str>,
        model_path: String,
        served_model_name: Option<String>,
        connect_timeout: Duration,
    ) -> Result<Self, DynamoError> {
        if !endpoint.as_str().starts_with("http://") {
            return Err(client::invalid_arg(
                "an encode worker reaches SGLang's encoder over HTTP at --grpc-endpoint host:port",
            ));
        }
        let http = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .build()
            .map_err(|error| {
                client::invalid_arg(format!(
                    "could not configure the SGLang encoder HTTP client: {error}"
                ))
            })?;
        Ok(Self {
            model_path,
            served_model_name,
            advertised: advertised_url(advertised.unwrap_or(endpoint.as_str()))?,
            endpoint: HttpEndpoint::parse(endpoint.as_str(), "--grpc-endpoint")?,
            http,
            inflight: Inflight::default(),
            next_token: AtomicU64::new(0),
        })
    }

    /// Wait until the encoder is healthy, then describe the encode worker.
    pub(crate) async fn start(
        &self,
        transport: &GrpcTransportConfig,
    ) -> Result<EngineConfig, DynamoError> {
        let deadline = Instant::now() + transport.startup_deadline;
        let health = self.endpoint.with_path("/health");
        loop {
            let response =
                tokio::time::timeout_at(deadline, self.http.get(health.clone()).send()).await;
            let failure = match response {
                Ok(Ok(response)) if response.status().is_success() => break,
                Ok(Ok(response)) => format!("HTTP {}", response.status()),
                Ok(Err(error)) => error.to_string(),
                Err(_) => {
                    return Err(client::connection_timeout(format!(
                        "SGLang encoder {health} did not answer within the startup deadline"
                    )));
                }
            };
            if Instant::now() >= deadline {
                return Err(client::cannot_connect(format!(
                    "SGLang encoder {health} did not become healthy: {failure}"
                )));
            }
            tokio::time::sleep_until((Instant::now() + transport.retry_interval).min(deadline))
                .await;
        }
        let mut config = EngineConfig {
            model: self.model_path.clone(),
            served_model_name: self.served_model_name.clone(),
            llm: Some(Default::default()),
            ..Default::default()
        };
        advertise(&mut config);
        Ok(config)
    }

    /// Polls `/health` and resolves with the reason once the encoder cannot be
    /// reached, or once [`client::SERVING_POLL_FAILURES`] polls in a row fail
    /// otherwise or time out.
    pub(crate) fn watch_serving(
        &self,
        attempt_timeout: Duration,
    ) -> impl Future<Output = String> + Send + 'static {
        let http = self.http.clone();
        let health = self.endpoint.with_path("/health");
        async move {
            let poll_timeout = attempt_timeout.min(client::SERVING_POLL_TIMEOUT);
            let mut failures = 0;
            loop {
                tokio::time::sleep(HEALTH_POLL_INTERVAL).await;
                let poll = http.get(health.clone()).send();
                let failure = match tokio::time::timeout(poll_timeout, poll).await {
                    Ok(Ok(response)) if response.status().is_success() => {
                        failures = 0;
                        continue;
                    }
                    Ok(Ok(response)) => format!("/health returned HTTP {}", response.status()),
                    Ok(Err(error)) if error.is_connect() => return format!("/health: {error}"),
                    Ok(Err(error)) => format!("/health: {error}"),
                    Err(_) => format!("/health did not answer in {poll_timeout:?}"),
                };
                failures += 1;
                tracing::debug!(failures, %failure, "SGLang encoder health poll failed");
                if failures == client::SERVING_POLL_FAILURES {
                    return format!("{failure}; {failures} polls in a row failed");
                }
            }
        }
    }

    /// Start the request's encodes and yield their descriptor, or handle the
    /// release of an earlier one.
    pub(crate) fn encode(
        &self,
        request: PreprocessedRequest,
        ctx: GenerateContext,
        stop: CancellationToken,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        if let Some(original_id) = ctx.metadata().get(ENCODER_RELEASE) {
            return self.release(&request, original_id);
        }
        let request_id = ctx.id().to_string();
        let (parts, requests) = encode_requests(request, &request_id)?;
        let descriptor = Descriptor {
            encoder_url: self.advertised.clone(),
            request_id: request_id.clone(),
            parts,
        };
        let Value::Object(result) = serde_json::to_value(&descriptor)
            .map_err(|error| client::protocol_error(error.to_string()))?
        else {
            unreachable!("the descriptor is a struct")
        };
        let http = self.http.clone();
        let url = self.endpoint.with_path("/encode");
        let inflight = self.inflight.clone();
        let token = self.next_token.fetch_add(1, Ordering::Relaxed);
        Ok(Box::pin(async_stream::try_stream! {
            if stop.is_cancelled() || ctx.is_stopped() {
                yield LLMEngineOutput::cancelled();
                return;
            }
            let (task, mut ended) =
                spawn_encode(&inflight, token, &request_id, http, url, requests);
            let mut task = AbortUnlessYielded(Some(task));
            let early = tokio::select! {
                biased;
                _ = stop.cancelled() => Early::Stopped,
                _ = ctx.stopped() => Early::Stopped,
                result = &mut ended => Early::Ended(result),
                _ = tokio::time::sleep(ENCODE_FAILURE_GRACE) => Early::Running,
            };
            let outcome = match early {
                Early::Stopped | Early::Ended(Err(_)) => None,
                Early::Ended(Ok(result)) => Some(result),
                Early::Running => {
                    // A failure may land between the grace expiring and this close.
                    ended.close();
                    Some(ended.try_recv().unwrap_or(Ok(())))
                }
            };
            let Some(outcome) = outcome else {
                task.abort();
                yield LLMEngineOutput::cancelled();
                return;
            };
            outcome?;
            if stop.is_cancelled() || ctx.is_stopped() {
                task.abort();
                yield LLMEngineOutput::cancelled();
                return;
            }
            tracing::debug!(request_id = %request_id, "native SGLang encoder is encoding");
            // The frontend releases the encode from here on.
            task.0 = None;
            let mut terminal = LLMEngineOutput::encode_terminal(result);
            terminal.extra_args = Some(Box::new(serde_json::json!({ (ENCODER_RELEASE): true })));
            yield terminal;
        }))
    }

    /// Abort the encode a cancelled or rejected request left running.
    fn release(
        &self,
        request: &PreprocessedRequest,
        original_id: &str,
    ) -> Result<BoxStream<'static, Result<LLMEngineOutput, DynamoError>>, DynamoError> {
        let descriptor = request
            .encoder_result
            .as_ref()
            .and_then(|result| Descriptor::parse(result).ok())
            .ok_or_else(|| client::invalid_request("encoder release requires an encoder result"))?;
        // Only an encode this worker started can be released here.
        if original_id.is_empty()
            || descriptor.request_id != original_id
            || descriptor.encoder_url != self.advertised
            || media_counts(request)? != [0; 3]
            || !request.token_ids.is_empty()
        {
            return Err(client::invalid_request(
                "encoder release does not match its original encode",
            ));
        }
        let running = lock(&self.inflight).remove(original_id);
        if let Some((_, task)) = running {
            task.abort();
        }
        // Release uses the encode endpoint, whose terminal result must be an object.
        let acknowledgement: Result<_, DynamoError> =
            Ok(LLMEngineOutput::encode_terminal(Default::default()));
        Ok(Box::pin(futures::stream::iter([acknowledgement])))
    }
}

/// Run the encode in a task of its own and record it. The receiver gets the
/// outcome while it still listens; a later failure is logged.
fn spawn_encode(
    inflight: &Inflight,
    token: u64,
    request_id: &str,
    http: reqwest::Client,
    url: url::Url,
    requests: Vec<EncodeRequest>,
) -> (AbortHandle, oneshot::Receiver<Result<(), DynamoError>>) {
    let (ended_tx, ended_rx) = oneshot::channel();
    let entry = Entry {
        inflight: inflight.clone(),
        request_id: request_id.to_string(),
        token,
    };
    // Held across the spawn, so that the task removes its entry after the insert.
    let mut running = lock(inflight);
    let task = tokio::spawn(async move {
        // Every part at once. The first failure drops the other requests.
        let posts = requests
            .into_iter()
            .map(|request| post_part(http.clone(), url.clone(), request));
        let result = futures::future::try_join_all(posts).await.map(|_| ());
        let request_id = entry.request_id.clone();
        drop(entry);
        if let Err(Err(error)) = ended_tx.send(result) {
            tracing::warn!(%request_id, %error, "SGLang encoder failed after its encode terminal");
        }
    })
    .abort_handle();
    let earlier = running.insert(request_id.to_string(), (token, task.clone()));
    drop(running);
    // The encoder keys its work by request ID, so a repeat replaces the earlier encode.
    if let Some((_, earlier)) = earlier {
        earlier.abort();
    }
    (task, ended_rx)
}

async fn post_part(
    http: reqwest::Client,
    url: url::Url,
    request: EncodeRequest,
) -> Result<(), DynamoError> {
    let response = http
        .post(url)
        .timeout(ENCODE_TIMEOUT)
        .json(&request)
        .send()
        .await
        .map_err(request_error)?;
    let status = response.status();
    if status.is_success() {
        return Ok(());
    }
    let detail = response
        .text()
        .await
        .unwrap_or_else(|error| format!("could not read error response: {error}"));
    Err(status_error(status, &detail))
}

fn request_error(error: reqwest::Error) -> DynamoError {
    if error.is_timeout() {
        client::connection_timeout(format!("SGLang encoder /encode timed out: {error}"))
    } else if error.is_connect() {
        client::cannot_connect(format!("could not connect to the SGLang encoder: {error}"))
    } else {
        client::protocol_error(format!("SGLang encoder /encode request failed: {error}"))
    }
}

/// An `/encode` error status. The encoder answers `{"status": "error",
/// "message": ...}`; a 4xx names media or a modality it cannot encode.
fn status_error(status: StatusCode, detail: &str) -> DynamoError {
    let body = serde_json::from_str::<Value>(detail).ok();
    let reason = body
        .as_ref()
        .and_then(|body| body.get("message"))
        .and_then(Value::as_str)
        .unwrap_or(detail);
    let message = format!("SGLang encoder /encode returned HTTP {status}: {reason}");
    if status.is_client_error() {
        // The encoder's text may name internal addresses, so it stays private.
        DynamoError::builder()
            .error_type(ErrorType::Backend(BackendError::InvalidArgument))
            .message(message)
            .public_message("the SGLang encoder rejected the request's media")
            .build()
    } else if status == StatusCode::GATEWAY_TIMEOUT {
        client::connection_timeout(message)
    } else if matches!(status.as_u16(), 502 | 503) {
        client::cannot_connect(message)
    } else {
        client::protocol_error(message)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use dynamo_backend_common::testing::mock_context;
    use dynamo_backend_common::{
        FinishReason, MultimodalData, OutputOptions, SamplingOptions, StopConditions,
    };
    use futures::StreamExt;
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    const IMAGE: &str = "data:image/png;base64,AAAA";

    fn request(media: &[(&str, &[&str])]) -> PreprocessedRequest {
        let mut request = PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1, 2, 3])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions::default())
            .build()
            .unwrap();
        request.multi_modal_data = Some(
            media
                .iter()
                .map(|(key, sources)| {
                    let items = sources
                        .iter()
                        .map(|source| MultimodalData::RawUrl(source.to_string()))
                        .collect();
                    (key.to_string(), items)
                })
                .collect(),
        );
        request
    }

    fn descriptor(request_id: &str, parts: Value) -> Value {
        json!({"encoder_url": "http://e:30000", "request_id": request_id, "parts": parts})
    }

    #[test]
    fn parts_are_numbered_image_video_audio_under_the_request_id() {
        let clip = "https://example.test/a.mp4";
        let sound = "https://example.test/a.wav";
        let mixed = request(&[
            ("audio_url", &[sound]),
            ("image_url", &[IMAGE, "https://example.test/b.png"]),
            ("video_url", &[clip]),
        ]);
        let (parts, requests) = encode_requests(mixed, "req").unwrap();
        assert_eq!(
            serde_json::to_value(&parts).unwrap(),
            json!([
                {"modality": "image", "items": 2},
                {"modality": "video", "items": 1},
                {"modality": "audio", "items": 1}
            ])
        );
        assert_eq!(
            serde_json::to_value(&requests).unwrap(),
            json!([
                {"mm_items": [IMAGE, "https://example.test/b.png"], "num_parts": 3,
                 "part_idx": 0, "req_id": "req_local_part_0", "modality": "IMAGE",
                 "prefill_host": null, "embedding_port": null},
                {"mm_items": [clip], "num_parts": 3, "part_idx": 1,
                 "req_id": "req_local_part_1", "modality": "VIDEO",
                 "prefill_host": null, "embedding_port": null},
                {"mm_items": [sound], "num_parts": 3, "part_idx": 2,
                 "req_id": "req_local_part_2", "modality": "AUDIO",
                 "prefill_host": null, "embedding_port": null}
            ])
        );

        // A modality the request lacks takes no part index.
        let audio = request(&[("image_url", &[]), ("audio_url", &[sound])]);
        let (parts, requests) = encode_requests(audio, "req").unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!((requests[0].num_parts, requests[0].part_idx), (1, 0));
        assert_eq!(requests[0].req_id, "req_local_part_0");
        assert_eq!(requests[0].modality, "AUDIO");
    }

    #[test]
    fn encode_refuses_what_the_native_encoder_cannot_take() {
        assert!(encode_requests(request(&[]), "req").is_err());
        assert!(encode_requests(request(&[("image_url", &["file:///a.png"])]), "req").is_err());
        assert!(encode_requests(request(&[("chemistry", &[IMAGE])]), "req").is_err());
        let mut encoded = request(&[("image_url", &[IMAGE])]);
        encoded.encoder_result = Some(json!({}));
        assert!(encode_requests(encoded, "req").is_err());
    }

    #[test]
    fn consumer_checks_the_descriptor_against_its_own_request() {
        let mut req = request(&[("image_url", &[IMAGE, IMAGE]), ("audio_url", &[IMAGE])]);

        // A media request that bypassed the encoder is unavailable, not invalid.
        let missing = validate_consumer(&req, "public", true).unwrap_err();
        assert_eq!(missing.class(), ErrorType::Unavailable);
        validate_consumer(&req, "public", false).unwrap();

        let parts = json!([{"modality": "image", "items": 2}, {"modality": "audio", "items": 1}]);
        req.encoder_result = Some(descriptor("public", parts.clone()));
        validate_consumer(&req, "public", true).unwrap();
        assert!(validate_consumer(&req, "other", true).is_err());
        // Without --route-to-encoder SGLang would ignore the handoff.
        assert!(validate_consumer(&req, "public", false).is_err());

        for parts in [
            json!([]),
            json!([{"modality": "image", "items": 2}]),
            json!([{"modality": "image", "items": 3}, {"modality": "audio", "items": 1}]),
            json!([{"modality": "image", "items": 2}, {"modality": "video", "items": 1}]),
            json!([{"modality": "image", "items": 2}, {"modality": "image", "items": 1}]),
            json!([{"modality": "image", "items": 0}, {"modality": "audio", "items": 1}]),
            json!([{"modality": "text", "items": 2}]),
        ] {
            req.encoder_result = Some(descriptor("public", parts.clone()));
            assert!(validate_consumer(&req, "public", true).is_err(), "{parts}");
        }
        let mut unknown = descriptor("public", parts.clone());
        unknown["signature"] = json!("ab");
        req.encoder_result = Some(unknown);
        assert!(validate_consumer(&req, "public", true).is_err());

        // A request that kept no sources is checked by SGLang's placeholder count.
        req.multi_modal_data = None;
        req.encoder_result = Some(descriptor("public", parts));
        validate_consumer(&req, "public", true).unwrap();

        // Text requests need no encoder.
        validate_consumer(&request(&[]), "public", true).unwrap();
        // Native /generate envelopes may carry media without canonical media slots.
        let mut native = request(&[]);
        native.multi_modal_data = None;
        native.extra_args = Some(json!({
            (dynamo_sglang_generate::PAYLOAD_KEY): {"image_data": IMAGE}
        }));
        assert!(validate_consumer(&native, "public", true).is_err());
    }

    #[test]
    fn sglang_gets_the_encoder_and_parts_without_the_request_id() {
        let parts = json!([{"modality": "video", "items": 1}, {"modality": "image", "items": 2}]);
        assert_eq!(
            handoff(&descriptor("public", parts.clone())).unwrap(),
            json!({"encoder_url": "http://e:30000", "parts": parts})
        );
        assert!(handoff(&json!({"encoder_url": "http://e:30000"})).is_err());
    }

    #[test]
    fn workers_advertise_one_contract_and_a_reachable_encoder() {
        let mut config = EngineConfig::default();
        advertise(&mut config);
        assert_eq!(
            config.runtime_data[ENCODER_HANDOFF_CONTRACT],
            json!({"sglang": "native-encode-v1"})
        );

        assert_eq!(advertised_url("e:30000").unwrap(), "http://e:30000");
        assert_eq!(advertised_url("http://e").unwrap(), "http://e:80");
        assert_eq!(advertised_url("[::1]:9").unwrap(), "http://[::1]:9");
        for unreachable in ["0.0.0.0:30000", "[::]:30000", "unix:///run/e.sock"] {
            assert!(advertised_url(unreachable).is_err(), "{unreachable}");
        }
    }

    /// An encoder that answers every request with `reply`, or never.
    async fn fake_encoder(reply: Option<String>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        tokio::spawn(async move {
            loop {
                let (mut socket, _) = listener.accept().await.unwrap();
                let reply = reply.clone();
                tokio::spawn(async move {
                    let mut received = vec![0; 65536];
                    let _ = socket.read(&mut received).await;
                    match reply {
                        Some(reply) => {
                            let _ = socket.write_all(reply.as_bytes()).await;
                        }
                        None => std::future::pending::<()>().await,
                    }
                });
            }
        });
        address
    }

    fn encoder_at(address: &str) -> Encoder {
        let endpoint = GrpcEndpoint::parse(address, "--grpc-endpoint").unwrap();
        Encoder::new(
            &endpoint,
            None,
            "model".into(),
            None,
            Duration::from_secs(5),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn an_immediate_encoder_rejection_fails_the_encode() {
        let body = r#"{"status":"error","message":"unsupported modality: VIDEO"}"#;
        let reply = format!(
            "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let encoder = encoder_at(&fake_encoder(Some(reply)).await);
        let ctx = GenerateContext::new(mock_context(), None);
        let mut stream = encoder
            .encode(
                request(&[("image_url", &[IMAGE])]),
                ctx,
                CancellationToken::new(),
            )
            .unwrap();
        let error = stream.next().await.unwrap().err().unwrap();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::InvalidArgument)
        );
        assert!(error.message().contains("unsupported modality: VIDEO"));
        assert!(lock(&encoder.inflight).is_empty());
    }

    #[tokio::test]
    async fn an_unreachable_encoder_fails_the_encode_as_unavailable() {
        // Bound, then closed: nothing listens on the port.
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap().to_string();
        drop(listener);
        let encoder = encoder_at(&address);
        let ctx = GenerateContext::new(mock_context(), None);
        let mut stream = encoder
            .encode(
                request(&[("image_url", &[IMAGE])]),
                ctx,
                CancellationToken::new(),
            )
            .unwrap();
        let error = stream.next().await.unwrap().err().unwrap();
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::CannotConnect)
        );
    }

    #[tokio::test]
    async fn a_running_encode_yields_its_descriptor_and_release_aborts_it() {
        let address = fake_encoder(None).await;
        let encoder = encoder_at(&address);
        let ctx = GenerateContext::new(mock_context(), None);
        let request_id = ctx.id().to_string();
        let mut stream = encoder
            .encode(
                request(&[("image_url", &[IMAGE])]),
                ctx,
                CancellationToken::new(),
            )
            .unwrap();
        let terminal = stream.next().await.unwrap().unwrap();
        assert_eq!(terminal.finish_reason, Some(FinishReason::Stop));
        let result = *terminal.encoder_result.unwrap();
        assert_eq!(
            result,
            json!({
                "encoder_url": format!("http://{address}"),
                "request_id": request_id,
                "parts": [{"modality": "image", "items": 1}]
            })
        );
        assert_eq!(terminal.extra_args.unwrap()[ENCODER_RELEASE], true);
        assert!(stream.next().await.is_none());
        // The encode outlives its stream until the encoder answers.
        drop(stream);
        assert!(lock(&encoder.inflight).contains_key(&request_id));

        // The frontend's cleanup call: no tokens, no media, the original ID in metadata.
        let release = |result: Value, original_id: &str| {
            let cleanup = PreprocessedRequest::builder()
                .model("model".to_string())
                .token_ids(Vec::new())
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .encoder_result(Some(result))
                .build()
                .unwrap();
            let metadata = BTreeMap::from([(ENCODER_RELEASE.to_string(), original_id.to_string())]);
            let ctx = GenerateContext::with_metadata(mock_context(), None, metadata);
            encoder.encode(cleanup, ctx, CancellationToken::new())
        };
        // A release names its own encoder and request.
        assert!(release(result.clone(), "other").is_err());
        let mut foreign = result.clone();
        foreign["encoder_url"] = json!("http://other:1");
        assert!(release(foreign, &request_id).is_err());
        assert!(lock(&encoder.inflight).contains_key(&request_id));

        let mut acknowledged = release(result, &request_id).unwrap();
        let ack = acknowledged.next().await.unwrap().unwrap();
        assert_eq!(ack.finish_reason, Some(FinishReason::Stop));
        assert!(lock(&encoder.inflight).is_empty());
    }

    #[tokio::test]
    async fn a_request_stopped_before_its_terminal_aborts_the_encode() {
        let encoder = encoder_at(&fake_encoder(None).await);
        let stop = CancellationToken::new();
        let ctx = GenerateContext::new(mock_context(), None);
        let mut stream = encoder
            .encode(request(&[("image_url", &[IMAGE])]), ctx, stop.clone())
            .unwrap();
        // Stop while the encode waits out its failure grace.
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            stop.cancel();
        });
        let cancelled = stream.next().await.unwrap().unwrap();
        assert_eq!(cancelled.finish_reason, Some(FinishReason::Cancelled));
        drop(stream);
        // The aborted task removes its entry when the runtime drops it.
        for _ in 0..100 {
            if lock(&encoder.inflight).is_empty() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("the stopped encode is still running");
    }
}
