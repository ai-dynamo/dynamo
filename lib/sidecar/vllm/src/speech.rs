// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    num::NonZeroUsize,
    path::PathBuf,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{Engine, prelude::BASE64_STANDARD};
use dynamo_backend_common::{
    DynamoError, EngineConfig, GenerateContext, HEALTH_CHECK_KEY, RawEngine, RuntimeConfig,
};
use dynamo_llm::preprocessor::media::MediaFetcher;
use dynamo_realtime::{Client, Limits, Protocol};
use dynamo_sidecar_common::{EngineBootstrapResult, engine_shutdown, invalid_argument};
use futures::{StreamExt, stream::BoxStream};
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use url::Url;

use crate::media::{MediaArgs, registration, to_dynamo};

const DEFAULT_MAX_RESPONSE_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_REFERENCE_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_REFERENCE_TIMEOUT_SECS: u64 = 30;

#[derive(clap::Args)]
pub(crate) struct SpeechArgs {
    #[command(flatten)]
    media: MediaArgs,
    /// Maximum generated audio bytes in one response.
    #[arg(long, default_value_t = NonZeroUsize::new(DEFAULT_MAX_RESPONSE_BYTES).unwrap())]
    max_response_bytes: NonZeroUsize,
    /// Maximum downloaded or decoded reference-audio bytes.
    #[arg(long, default_value_t = NonZeroUsize::new(DEFAULT_MAX_REFERENCE_BYTES).unwrap())]
    tts_ref_audio_max_bytes: NonZeroUsize,
    /// Reference-audio fetch timeout (seconds).
    #[arg(long, default_value_t = DEFAULT_REFERENCE_TIMEOUT_SECS, value_parser = clap::value_parser!(u64).range(1..))]
    tts_ref_audio_timeout_secs: u64,
    /// Shared directory for data_source=url responses.
    #[arg(long, requires = "media_output_http_url")]
    media_output_dir: Option<PathBuf>,
    /// HTTP base URL exposing --media-output-dir.
    #[arg(long, requires = "media_output_dir")]
    media_output_http_url: Option<Url>,
}

struct OutputStore {
    directory: PathBuf,
    base_url: Url,
}

pub(crate) struct SpeechEngine {
    client: Client,
    model: String,
    max_response_bytes: usize,
    max_reference_bytes: usize,
    reference_policy: MediaFetcher,
    reference_client: reqwest::Client,
    output_store: Option<Arc<OutputStore>>,
}

impl SpeechEngine {
    pub(crate) fn bootstrap(
        args: SpeechArgs,
    ) -> Result<
        (
            RuntimeConfig,
            impl Future<Output = EngineBootstrapResult<Self>>,
        ),
        DynamoError,
    > {
        args.media.validate("audios")?;
        if let Some(url) = &args.media_output_http_url
            && (!matches!(url.scheme(), "http" | "https")
                || url.host().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some())
        {
            return Err(invalid_argument(
                "media output URL must be an HTTP base URL without credentials, query, or fragment",
            ));
        }
        let config = args.media.worker_config();
        let client = args.media.client(Limits::for_protocol(Protocol::Omni))?;
        let mut reference_policy = MediaFetcher::from_env();
        let reference_timeout = Duration::from_secs(args.tts_ref_audio_timeout_secs);
        if std::time::Instant::now()
            .checked_add(reference_timeout)
            .is_none()
        {
            return Err(invalid_argument("reference-audio timeout is too large"));
        }
        reference_policy.timeout = Some(reference_timeout);
        let reference_client = if reference_policy.allow_private_ips {
            reference_policy.build_http_client()
        } else {
            reference_policy.build_https_client()
        }
        .map_err(|error| invalid_argument(format!("reference-audio HTTP client: {error}")))?;
        Ok((config.runtime.clone(), async move {
            args.media.wait_ready(&client, None).await?;
            let output_store = match (args.media_output_dir, args.media_output_http_url) {
                (Some(directory), Some(base_url)) => {
                    tokio::fs::create_dir_all(&directory)
                        .await
                        .map_err(|error| {
                            invalid_argument(format!("media output directory: {error}"))
                        })?;
                    Some(Arc::new(OutputStore {
                        directory,
                        base_url,
                    }))
                }
                _ => None,
            };
            Ok((
                Self {
                    client,
                    model: args.media.model,
                    max_response_bytes: args.max_response_bytes.get(),
                    max_reference_bytes: args.tts_ref_audio_max_bytes.get(),
                    reference_policy,
                    reference_client,
                    output_store,
                },
                config,
            ))
        }))
    }
}

#[async_trait::async_trait]
impl RawEngine for SpeechEngine {
    async fn start(&self, _: u64) -> Result<EngineConfig, DynamoError> {
        Ok(registration(&self.model))
    }

    async fn generate(
        &self,
        request: Value,
        ctx: GenerateContext,
    ) -> Result<BoxStream<'static, Result<Value, DynamoError>>, DynamoError> {
        let client = self.client.clone();
        if request.get(HEALTH_CHECK_KEY).and_then(Value::as_bool) == Some(true) {
            return Ok(Box::pin(futures::stream::once(async move {
                client.health().await.map_err(to_dynamo)?;
                Ok(json!({"healthy": true}))
            })));
        }
        let PreparedRequest {
            mut body,
            codec,
            streaming,
            url_delivery,
        } = prepare(request)?;
        if url_delivery && self.output_store.is_none() {
            return Err(invalid_argument(
                "data_source=url requires --media-output-dir and --media-output-http-url",
            ));
        }
        let policy = self.reference_policy.clone();
        let reference_client = self.reference_client.clone();
        let max_reference_bytes = self.max_reference_bytes;
        let reference_timeout = policy
            .timeout
            .unwrap_or(Duration::from_secs(DEFAULT_REFERENCE_TIMEOUT_SECS));
        let max_response_bytes = self.max_response_bytes;
        let store = self.output_store.clone();
        let model = self.model.clone();
        let id = format!("audio_{}", uuid::Uuid::new_v4().simple());
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| engine_shutdown("system clock precedes Unix epoch"))?
            .as_secs();
        // Returning the stream before any network I/O lets runtime cancellation
        // interrupt reference fetches and the upstream request's headers, too.
        Ok(Box::pin(async_stream::try_stream! {
            let work = async {
                for field in ["ref_audio", "ref_audio_2"] {
                    if let Some(reference) = body.get(field).filter(|value| !value.is_null()) {
                        let reference = reference.as_str().ok_or_else(|| {
                            invalid_argument("reference audio must be a URL or data URI string")
                        })?;
                        body[field] = tokio::time::timeout(
                            reference_timeout,
                            resolve_reference(reference, &policy, &reference_client, max_reference_bytes),
                        ).await.map_err(|_| {
                            dynamo_sidecar_common::connection_timeout("reference-audio fetch timed out")
                        })??.into();
                    }
                }
                client.speech(body).await.map_err(to_dynamo)
            };
            let result = tokio::select! {
                _ = ctx.stopped() => None,
                _ = ctx.killed() => None,
                result = work => Some(result),
            };
            if let Some(result) = result {
                let mut upstream = result?;
                let mut total = 0usize;
                let mut accumulated = Vec::new();
                loop {
                    let chunk = tokio::select! {
                        _ = ctx.stopped() => break,
                        _ = ctx.killed() => break,
                        chunk = upstream.next() => chunk,
                    };
                    match chunk {
                        Some(chunk) => {
                            let chunk = chunk.map_err(to_dynamo)?;
                            if chunk.len() > max_response_bytes.saturating_sub(total) {
                                Err(engine_shutdown("generated audio exceeds the configured byte limit"))?;
                            }
                            total += chunk.len();
                            if streaming {
                                if !chunk.is_empty() {
                                    yield response(&id, &model, created, "in_progress", json!([{
                                        "output_format": codec, "b64_json": BASE64_STANDARD.encode(chunk)
                                    }]));
                                }
                            } else {
                                accumulated.extend_from_slice(&chunk);
                            }
                        }
                        None => {
                            if total == 0 {
                                Err(engine_shutdown("upstream returned empty audio"))?;
                            }
                            let data = if streaming {
                                json!([])
                            } else if url_delivery {
                                let store = store.as_ref().expect("validated output storage");
                                let url = write_output(store, &accumulated, &codec).await?;
                                json!([{"output_format": codec, "url": url}])
                            } else {
                                json!([{"output_format": codec, "b64_json": BASE64_STANDARD.encode(&accumulated)}])
                            };
                            yield response(&id, &model, created, "completed", data);
                            break;
                        }
                    }
                }
            }
        }))
    }

    async fn health_check_payload(&self) -> Result<Option<Value>, DynamoError> {
        Ok(Some(json!({HEALTH_CHECK_KEY: true})))
    }

    async fn cleanup(&self) -> Result<(), DynamoError> {
        self.client.shutdown().await;
        Ok(())
    }
}

struct PreparedRequest {
    body: Value,
    codec: String,
    streaming: bool,
    url_delivery: bool,
}

fn prepare(mut request: Value) -> Result<PreparedRequest, DynamoError> {
    let body = request
        .as_object_mut()
        .ok_or_else(|| invalid_argument("speech request must be an object"))?;
    let delivery = body.remove("data_source").unwrap_or(Value::Null);
    let url_delivery = delivery == "url";
    if !delivery.is_null() && !url_delivery && delivery != "b64_json" {
        return Err(invalid_argument("unsupported audio data_source"));
    }
    let nvext = body.remove("nvext").unwrap_or(Value::Null);
    // Keep the worker-boundary passthrough policy aligned with
    // dynamo.common.protocols.sanitize_media_passthrough. Typed fields win.
    if let Some(extra) = body.remove("extra_args")
        && let Some(knobs) = extra["media_passthrough"].as_object()
    {
        validate_passthrough(&extra["media_passthrough"])?;
        for (key, value) in knobs {
            body.entry(key).or_insert_with(|| value.clone());
        }
    }
    // vLLM-Omni reads sampling knobs from extra_params; unknown top-level
    // fields would otherwise be silently discarded by its speech schema.
    let mut sampling = match body.remove("extra_params") {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(params)) => params,
        Some(_) => return Err(invalid_argument("extra_params must be an object")),
    };
    validate_sampling(&sampling)?;
    for key in ["temperature", "top_p", "top_k"] {
        if let Some(value) = body.remove(key) {
            sampling.insert(key.into(), value);
        }
    }
    if !nvext["cfg_scale"].is_null() {
        sampling.insert("cfg_scale".into(), nvext["cfg_scale"].clone());
    }
    validate_sampling(&sampling)?;
    if !sampling.is_empty() {
        body.insert("extra_params".into(), Value::Object(sampling));
    }
    let codec = body
        .get("response_format")
        .and_then(Value::as_str)
        .unwrap_or("wav")
        .to_ascii_lowercase();
    if !matches!(
        codec.as_str(),
        "wav" | "pcm" | "mp3" | "flac" | "aac" | "opus"
    ) {
        return Err(invalid_argument("unsupported speech response_format"));
    }
    // N-2 compatibility: v1.4 frontends require one aggregated item.
    // TODO(v1.7): Remove the capability check when v1.4 leaves the window.
    let streaming = nvext["frontend_accepts_audio_chunks"] == true
        && !url_delivery
        && matches!(codec.as_str(), "pcm" | "wav")
        && body
            .get("speed")
            .is_none_or(|value| value.is_null() || value.as_f64() == Some(1.0));
    // vLLM-Omni's stream=true selects SSE; Dynamo needs raw codec bytes.
    body.insert("stream".into(), false.into());
    body.insert(
        "stream_format".into(),
        if streaming {
            "audio".into()
        } else {
            Value::Null
        },
    );
    body.insert("response_format".into(), codec.clone().into());
    Ok(PreparedRequest {
        body: request,
        codec,
        streaming,
        url_delivery,
    })
}

fn validate_sampling(sampling: &serde_json::Map<String, Value>) -> Result<(), DynamoError> {
    // Upstream model-specific extra_params may fetch media (for example,
    // IndexTTS2 emo_audio). Keep this map restricted to scalar sampling knobs;
    // reference media must use ref_audio/ref_audio_2 and our deployment policy.
    for (key, value) in sampling {
        if !matches!(
            key.as_str(),
            "temperature" | "top_p" | "top_k" | "cfg_scale"
        ) || value.is_object()
            || value.is_array()
        {
            return Err(invalid_argument(
                "extra_params only supports scalar temperature, top_p, top_k, and cfg_scale",
            ));
        }
    }
    Ok(())
}

fn validate_passthrough(value: &Value) -> Result<(), DynamoError> {
    const UNSAFE: &[&str] = &[
        "path",
        "checkpoint",
        "ckpt",
        "repo",
        "hub",
        "url",
        "uri",
        "local_dir",
        "guardrail",
        "safety",
        "nsfw",
        "moderation",
    ];
    match value {
        Value::Object(fields) => {
            for (key, value) in fields {
                let key = key.to_ascii_lowercase();
                if matches!(key.as_str(), "extra_args" | "nvext" | "data_source")
                    || key.starts_with('_')
                    || UNSAFE.iter().any(|marker| key.contains(marker))
                {
                    return Err(invalid_argument(
                        "media passthrough cannot set load locations or deployment policy",
                    ));
                }
                // Nested extra_params and reference aliases must not bypass
                // the policy applied to top-level extension fields.
                validate_passthrough(value)?;
            }
        }
        Value::Array(items) => {
            for item in items {
                validate_passthrough(item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn response(id: &str, model: &str, created: u64, status: &str, data: Value) -> Value {
    json!({
        "id": id, "object": "audio.speech", "model": model, "created": created,
        "status": status, "progress": if status == "completed" { 100 } else { 0 }, "data": data,
    })
}

async fn resolve_reference(
    reference: &str,
    policy: &MediaFetcher,
    client: &reqwest::Client,
    limit: usize,
) -> Result<String, DynamoError> {
    if reference.starts_with("data:") {
        let (metadata, encoded) = reference
            .split_once(',')
            .ok_or_else(|| invalid_argument("invalid reference-audio data URI"))?;
        if !metadata.ends_with(";base64")
            || encoded.len().div_ceil(4).saturating_mul(3) > limit.saturating_add(2)
        {
            return Err(invalid_argument(
                "reference audio must be bounded base64 data",
            ));
        }
        let bytes = BASE64_STANDARD
            .decode(encoded)
            .map_err(|_| invalid_argument("invalid reference-audio base64"))?;
        if bytes.len() > limit || bytes.is_empty() {
            return Err(invalid_argument(
                "reference audio is empty or exceeds the byte limit",
            ));
        }
        return Ok(format!(
            "data:application/octet-stream;base64,{}",
            BASE64_STANDARD.encode(bytes)
        ));
    }
    let url = Url::parse(reference)
        .map_err(|_| invalid_argument("ref_audio must be an HTTP(S) URL or base64 data URI"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(invalid_argument(
            "ref_audio must be an HTTP(S) URL or base64 data URI",
        ));
    }
    if url.scheme() == "http" && !policy.allow_private_ips {
        return Err(invalid_argument(
            "HTTP reference audio requires DYN_MM_ALLOW_INTERNAL=1",
        ));
    }
    policy
        .check_if_url_allowed_with_dns(&url)
        .await
        .map_err(|_| invalid_argument("reference-audio URL was rejected by media policy"))?;
    let response = client
        .get(url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|_| invalid_argument("could not fetch reference audio"))?;
    if response
        .content_length()
        .is_some_and(|size| size > limit as u64)
    {
        return Err(invalid_argument("reference audio exceeds the byte limit"));
    }
    let mut stream = response.bytes_stream();
    let mut audio = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|_| invalid_argument("could not read reference audio"))?;
        if chunk.len() > limit.saturating_sub(audio.len()) {
            return Err(invalid_argument("reference audio exceeds the byte limit"));
        }
        audio.extend_from_slice(&chunk);
    }
    if audio.is_empty() {
        return Err(invalid_argument("reference audio is empty"));
    }
    Ok(format!(
        "data:application/octet-stream;base64,{}",
        BASE64_STANDARD.encode(audio)
    ))
}

async fn write_output(
    store: &OutputStore,
    audio: &[u8],
    codec: &str,
) -> Result<String, DynamoError> {
    let filename = format!("{}.{}", uuid::Uuid::new_v4().simple(), codec);
    let temporary = tempfile::NamedTempFile::new_in(&store.directory)
        .map_err(|error| engine_shutdown(format!("create audio output: {error}")))?;
    let mut file = tokio::fs::File::from_std(
        temporary
            .reopen()
            .map_err(|error| engine_shutdown(format!("open audio output: {error}")))?,
    );
    file.write_all(audio)
        .await
        .map_err(|error| engine_shutdown(format!("write audio output: {error}")))?;
    file.flush()
        .await
        .map_err(|error| engine_shutdown(format!("flush audio output: {error}")))?;
    drop(file);
    temporary
        .persist_noclobber(store.directory.join(&filename))
        .map_err(|error| engine_shutdown(format!("publish audio output: {error}")))?;
    let mut url = store.base_url.clone();
    url.set_path(&format!(
        "{}/{}",
        url.path().trim_end_matches('/'),
        filename
    ));
    Ok(url.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        body::Body,
        http::{Response, StatusCode},
        routing::post,
    };
    use dynamo_runtime::pipeline::{AsyncEngineContextProvider, Context};
    use futures::TryStreamExt;

    #[test]
    fn speech_preserves_typed_options_and_legacy_frontend_aggregation() {
        let request = json!({
            "input": "hello", "voice": "speaker", "task_type": "Base",
            "language": "English", "instructions": "quietly", "ref_text": "sample",
            "max_new_tokens": 123, "response_format": "wav",
            "extra_args": {"media_passthrough": {"temperature": 0.5, "voice": "overridden"}},
        });
        let legacy = prepare(request.clone()).unwrap();
        assert!(!legacy.streaming);
        assert!(legacy.body["stream_format"].is_null());
        assert_eq!(legacy.body["extra_params"]["temperature"], 0.5);
        assert_eq!(legacy.body["voice"], "speaker");
        assert_eq!(legacy.body["max_new_tokens"], 123);
        let mut capable = request;
        capable["nvext"] = json!({"frontend_accepts_audio_chunks": true});
        assert!(prepare(capable.clone()).unwrap().streaming);
        capable["speed"] = json!(1.5);
        assert!(!prepare(capable).unwrap().streaming);
        assert!(
            prepare(json!({"input": "hello", "extra_args": {
                "media_passthrough": {"checkpoint_path": "/tmp/model"}
            }}))
            .is_err()
        );
        assert!(
            prepare(json!({"input": "hello", "extra_args": {
                "media_passthrough": {"extra_params": {"checkpoint_path": "/tmp/model"}}
            }}))
            .is_err()
        );
    }

    #[test]
    fn speech_extra_params_reject_media_aliases_unknown_keys_and_nested_values() {
        for params in [
            json!({"emo_audio": "http://127.0.0.1/reference.wav"}),
            json!({"ref_audio": "data:audio/wav;base64,AAAA"}),
            json!({"unknown": 1}),
            json!({"temperature": {"emo_audio": "file:///tmp/reference.wav"}}),
            json!({"top_k": [1, 2]}),
        ] {
            for request in [
                json!({"input": "hello", "extra_params": params}),
                json!({"input": "hello", "extra_args": {
                    "media_passthrough": {"extra_params": params}
                }}),
            ] {
                let error = prepare(request).err().unwrap();
                assert_eq!(
                    error.error_type(),
                    dynamo_backend_common::ErrorType::Backend(
                        dynamo_backend_common::BackendError::InvalidArgument
                    )
                );
            }
        }
        assert!(
            prepare(json!({"input": "hello", "extra_args": {
                "media_passthrough": {"temperature": {"nested": 1}}
            }}))
            .is_err()
        );
    }

    #[test]
    fn speech_typed_cfg_scale_wins_and_qwen_fields_remain_top_level() {
        for cfg_scale in [0.0, 1.5] {
            let prepared = prepare(json!({
                "input": "hello", "voice": "vivian", "task_type": "CustomVoice",
                "language": "English", "instructions": "quietly", "ref_text": "sample",
                "ref_audio": "data:audio/wav;base64,AAAA", "max_new_tokens": 123,
                "nvext": {"cfg_scale": cfg_scale},
                "extra_args": {"media_passthrough": {
                    "ref_audio_2": "data:audio/wav;base64,AQID",
                    "temperature": 0.25,
                    "extra_params": {"temperature": 0.5, "top_p": 0.8, "top_k": 12, "cfg_scale": 8}
                }}
            }))
            .unwrap();
            assert_eq!(
                prepared.body["extra_params"],
                json!({"temperature": 0.25, "top_p": 0.8, "top_k": 12, "cfg_scale": cfg_scale})
            );
            assert_eq!(prepared.body["voice"], "vivian");
            assert_eq!(prepared.body["task_type"], "CustomVoice");
            assert_eq!(prepared.body["language"], "English");
            assert_eq!(prepared.body["instructions"], "quietly");
            assert_eq!(prepared.body["ref_text"], "sample");
            assert_eq!(prepared.body["ref_audio"], "data:audio/wav;base64,AAAA");
            assert_eq!(prepared.body["ref_audio_2"], "data:audio/wav;base64,AQID");
            assert_eq!(prepared.body["max_new_tokens"], 123);
        }
        assert_eq!(
            prepare(json!({"input": "hello", "extra_params": {"cfg_scale": 8}}))
                .unwrap()
                .body["extra_params"]["cfg_scale"],
            8
        );
    }

    async fn engine() -> (SpeechEngine, tokio::task::JoinHandle<()>) {
        let app = Router::new().route(
            "/v1/audio/speech",
            post(|Json(body): Json<Value>| async move {
                if body["input"] == "bad" {
                    return Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .body(Body::empty())
                        .unwrap();
                }
                if body["input"] == "partial" {
                    return Response::builder()
                        .header("content-type", "audio/wav")
                        .body(Body::from_stream(async_stream::stream! {
                            yield Ok::<_, std::io::Error>(b"partial".to_vec());
                            tokio::task::yield_now().await;
                            yield Err(std::io::Error::other("upstream generation failed"));
                        }))
                        .unwrap();
                }
                assert_eq!(body["model"], "test-model");
                assert!(body.get("nvext").is_none());
                assert!(body.get("data_source").is_none());
                Response::builder()
                    .header("content-type", "audio/wav")
                    .body(Body::from_stream(futures::stream::iter([
                        Ok::<_, std::io::Error>(b"first".to_vec()),
                        Ok(b"second".to_vec()),
                    ])))
                    .unwrap()
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap())
            .parse()
            .unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let policy = MediaFetcher::from_env();
        (
            SpeechEngine {
                client: Client::new(
                    url,
                    "test-model".into(),
                    Limits::for_protocol(Protocol::Omni),
                    None,
                )
                .unwrap(),
                model: "test-model".into(),
                max_response_bytes: 1024,
                max_reference_bytes: 1024,
                reference_client: policy.build_http_client().unwrap(),
                reference_policy: policy,
                output_store: None,
            },
            task,
        )
    }

    fn context() -> GenerateContext {
        GenerateContext::new(Context::new(()).context(), None)
    }

    #[tokio::test]
    async fn speech_streaming_and_legacy_delivery_preserve_upstream_bytes() {
        let (engine, server) = engine().await;
        for capable in [false, true] {
            let output: Vec<_> = engine
                .generate(
                    json!({
                        "input": "hello", "response_format": "wav",
                        "nvext": {"frontend_accepts_audio_chunks": capable}
                    }),
                    context(),
                )
                .await
                .unwrap()
                .try_collect()
                .await
                .unwrap();
            if !capable {
                assert_eq!(output.len(), 1);
            }
            assert_eq!(output.last().unwrap()["status"], "completed");
            let mut audio = Vec::new();
            for response in output {
                serde_json::from_value::<
                    dynamo_llm::protocols::openai::audios::NvAudioSpeechResponse,
                >(response.clone())
                .unwrap();
                for data in response["data"].as_array().unwrap() {
                    audio.extend(
                        BASE64_STANDARD
                            .decode(data["b64_json"].as_str().unwrap())
                            .unwrap(),
                    );
                }
            }
            assert_eq!(audio, b"firstsecond");
        }
        let output = engine
            .generate(json!({"input": "bad"}), context())
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await;
        assert_eq!(
            output.unwrap_err().error_type(),
            dynamo_backend_common::ErrorType::Backend(
                dynamo_backend_common::BackendError::InvalidArgument
            )
        );
        let output: Vec<_> = engine
            .generate(
                json!({
                    "input": "partial", "nvext": {"frontend_accepts_audio_chunks": true},
                }),
                context(),
            )
            .await
            .unwrap()
            .collect()
            .await;
        assert!(output.last().unwrap().is_err());
        assert!(!output.iter().any(|item| {
            item.as_ref()
                .is_ok_and(|value| value["status"] == "completed")
        }));
        engine.cleanup().await.unwrap();
        server.abort();
    }

    #[tokio::test]
    async fn url_delivery_uses_configured_storage_and_reference_policy_cannot_load_files() {
        let (mut engine, server) = engine().await;
        assert!(
            engine
                .generate(json!({"input": "hello", "data_source": "url"}), context())
                .await
                .is_err()
        );
        let directory = tempfile::tempdir().unwrap();
        engine.output_store = Some(Arc::new(OutputStore {
            directory: directory.path().into(),
            base_url: "https://example.com/audio/".parse().unwrap(),
        }));
        let output: Vec<_> = engine
            .generate(json!({"input": "hello", "data_source": "url"}), context())
            .await
            .unwrap()
            .try_collect()
            .await
            .unwrap();
        let url: Url = output[0]["data"][0]["url"]
            .as_str()
            .unwrap()
            .parse()
            .unwrap();
        let filename = url.path_segments().unwrap().next_back().unwrap();
        assert_eq!(
            tokio::fs::read(directory.path().join(filename))
                .await
                .unwrap(),
            b"firstsecond"
        );
        assert!(
            resolve_reference(
                "file:///etc/passwd",
                &engine.reference_policy,
                &engine.reference_client,
                1024
            )
            .await
            .is_err()
        );
        let rejected = engine
            .generate(
                json!({"input": "hello", "extra_args": {
                    "media_passthrough": {"ref_audio_2": "file:///etc/passwd"}
                }}),
                context(),
            )
            .await
            .unwrap()
            .try_collect::<Vec<_>>()
            .await;
        assert!(rejected.is_err());
        assert!(
            resolve_reference(
                "data:audio/wav;base64,AAAA",
                &engine.reference_policy,
                &engine.reference_client,
                1
            )
            .await
            .is_err()
        );
        engine.cleanup().await.unwrap();
        server.abort();
    }
}
