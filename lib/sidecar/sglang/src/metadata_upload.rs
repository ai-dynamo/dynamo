// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Out-of-band storage for SGLang response metadata.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use dynamo_backend_common::{DynamoError, PreprocessedRequest};
use lru::LruCache;
use opendal::layers::{RetryLayer, TimeoutLayer};
use opendal::{Buffer, Operator};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{Mutex, OnceCell};

use crate::client;
use crate::protocol::{ExtractedLogprobs, engine_data_from_meta, extract_logprobs};

mod config;
pub(crate) use config::MetadataUploadArgs;
use config::OperatorPolicy;

const OUTPUT_PATH: &str = "choice_0.msgpack.zst";

pub(crate) struct MetadataUploadService {
    operators: Arc<OperatorCache>,
}

pub(crate) enum MetadataDelivery {
    Inline,
    Upload(MetadataUploader),
}

pub(crate) struct MetadataUploader {
    primary: Operator,
    fallback: Option<String>,
    operators: Arc<OperatorCache>,
}

pub(crate) struct OperatorCache {
    operators: Mutex<LruCache<String, Arc<OnceCell<Operator>>>>,
    policy: OperatorPolicy,
    #[cfg(test)]
    build_count: std::sync::atomic::AtomicUsize,
}

#[derive(Serialize)]
struct MetadataPayload<'a> {
    schema_version: u8,
    metadata: &'a Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct MetadataUploadConfig {
    url: String,
    fallback_url: Option<String>,
}

impl OperatorCache {
    fn new(policy: OperatorPolicy) -> Self {
        // Ensure built-in services are registered when this crate is linked statically.
        opendal::install_default();
        Self {
            operators: Mutex::new(LruCache::new(policy.capacity)),
            policy,
            #[cfg(test)]
            build_count: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    async fn get(&self, raw: String, field: &'static str) -> Result<Operator, DynamoError> {
        let url = normalize_url(raw, field)?;
        let cell = {
            let mut operators = self.operators.lock().await;
            if let Some(cell) = operators.get(&url) {
                cell.clone()
            } else {
                let cell = Arc::new(OnceCell::new());
                operators.put(url.clone(), cell.clone());
                cell
            }
        };
        let policy = self.policy;
        let operator = cell
            .get_or_try_init(|| async {
                #[cfg(test)]
                self.build_count
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let build_url = url.clone();
                tokio::task::spawn_blocking(move || build_operator(&build_url, policy))
                    .await
                    .map_err(|error| {
                        client::protocol_error(format!(
                            "metadata upload setup task failed: {error}"
                        ))
                    })?
            })
            .await?;
        Ok(operator.clone())
    }

    #[cfg(test)]
    async fn insert(&self, url: &str, operator: Operator) {
        let cell = OnceCell::new();
        cell.set(operator).expect("new OnceCell must be empty");
        self.operators
            .lock()
            .await
            .put(url.to_string(), Arc::new(cell));
    }
}

impl MetadataUploadService {
    pub(crate) fn new(args: MetadataUploadArgs) -> Result<Self, DynamoError> {
        Ok(Self {
            operators: Arc::new(OperatorCache::new(args.try_into()?)),
        })
    }

    pub(crate) async fn delivery_for(
        &self,
        request: &PreprocessedRequest,
    ) -> Result<MetadataDelivery, DynamoError> {
        let Some(config) = request
            .extra_args
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|extra| extra.get("nvext"))
            .and_then(Value::as_object)
            .and_then(|nvext| nvext.get("metadata_upload"))
        else {
            return Ok(MetadataDelivery::Inline);
        };
        if config.is_null() {
            return Ok(MetadataDelivery::Inline);
        }
        let config = MetadataUploadConfig::deserialize(config).map_err(|error| {
            client::invalid_arg(format!("invalid extra_args.nvext.metadata_upload: {error}"))
        })?;
        let primary = self.operators.get(config.url, "url").await?;
        let fallback = config
            .fallback_url
            .map(|url| normalize_url(url, "fallback_url"))
            .transpose()?;
        Ok(MetadataDelivery::Upload(MetadataUploader {
            primary,
            fallback,
            operators: self.operators.clone(),
        }))
    }
}

impl MetadataDelivery {
    pub(crate) fn response_fields(
        &self,
        meta: &HashMap<String, String>,
        return_tokens_as_ids: bool,
        terminal: bool,
    ) -> Result<(ExtractedLogprobs, Option<Value>), DynamoError> {
        match self {
            Self::Inline => Ok((
                extract_logprobs(meta, return_tokens_as_ids)?,
                engine_data_from_meta(meta, terminal)?,
            )),
            Self::Upload(_) => Ok(((None, None), None)),
        }
    }

    pub(crate) fn uploader(&self) -> Option<&MetadataUploader> {
        match self {
            Self::Inline => None,
            Self::Upload(uploader) => Some(uploader),
        }
    }
}

impl MetadataUploader {
    pub(crate) async fn upload(&self, metadata: Value) -> Result<(), DynamoError> {
        if metadata.as_object().is_some_and(|value| value.is_empty()) {
            return Ok(());
        }
        let compressed: Buffer = tokio::task::spawn_blocking(move || encode(metadata))
            .await
            .map_err(|error| {
                client::protocol_error(format!("SGLang metadata encoding task failed: {error}"))
            })??
            .into();
        match write(&self.primary, compressed.clone()).await {
            Ok(()) => Ok(()),
            Err(primary_error) => {
                let Some(fallback_url) = self.fallback.as_ref() else {
                    return Err(primary_error);
                };
                tracing::warn!(error = %primary_error, "primary metadata upload failed; attempting fallback");
                let fallback = self
                    .operators
                    .get(fallback_url.clone(), "fallback_url")
                    .await?;
                write(&fallback, compressed).await
            }
        }
    }
}

fn normalize_url(raw: String, field: &str) -> Result<String, DynamoError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(client::invalid_arg(format!(
            "extra_args.nvext.metadata_upload.{field} must not be empty"
        )));
    }
    Ok(raw.to_string())
}

fn build_operator(raw: &str, policy: OperatorPolicy) -> Result<Operator, DynamoError> {
    Operator::from_uri(raw)
        .map(|operator| configure_operator(operator, policy))
        .map_err(|error| {
            client::invalid_arg(format!(
                "could not configure `{}` metadata upload destination: {error}",
                raw.split_once(':')
                    .map(|(scheme, _)| scheme)
                    .unwrap_or("unknown")
            ))
        })
}

fn configure_operator(operator: Operator, policy: OperatorPolicy) -> Operator {
    let retry = RetryLayer::new()
        .with_factor(policy.retry_factor)
        .with_min_delay(policy.retry_min_delay)
        .with_max_delay(policy.retry_max_delay)
        .with_max_times(policy.retry_max_times);
    let retry = if policy.retry_jitter {
        retry.with_jitter()
    } else {
        retry
    };
    let timeout = TimeoutLayer::new()
        .with_timeout(policy.timeout)
        .with_io_timeout(policy.io_timeout);
    // Timeout must be inside RetryLayer so a timeout cannot drop a stateful
    // retry future before OpenDAL restores its body state.
    operator.layer(timeout).layer(retry)
}

async fn write(operator: &Operator, data: Buffer) -> Result<(), DynamoError> {
    operator
        .write(OUTPUT_PATH, data)
        .await
        .map(|_| ())
        .map_err(|error| upload_error(operator, error))
}

fn upload_error(operator: &Operator, error: opendal::Error) -> DynamoError {
    client::protocol_error(format!(
        "could not upload SGLang metadata to {}: {error}",
        operator.info().root()
    ))
}

fn encode(metadata: Value) -> Result<Vec<u8>, DynamoError> {
    let payload = MetadataPayload {
        schema_version: 1,
        metadata: &metadata,
    };
    let msgpack = rmp_serde::to_vec_named(&payload).map_err(|error| {
        client::protocol_error(format!("could not serialize SGLang metadata: {error}"))
    })?;
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 0).map_err(|error| {
        client::protocol_error(format!(
            "could not initialize SGLang metadata compression: {error}"
        ))
    })?;
    // Python's ZstdDecompressor.decompress(), used by the existing fsspec
    // reader, requires the uncompressed size in the frame header.
    encoder
        .set_pledged_src_size(Some(msgpack.len() as u64))
        .map_err(|error| {
            client::protocol_error(format!(
                "could not configure SGLang metadata compression: {error}"
            ))
        })?;
    encoder.include_contentsize(true).map_err(|error| {
        client::protocol_error(format!(
            "could not configure SGLang metadata compression: {error}"
        ))
    })?;
    encoder.write_all(&msgpack).map_err(|error| {
        client::protocol_error(format!("could not compress SGLang metadata: {error}"))
    })?;
    encoder.finish().map_err(|error| {
        client::protocol_error(format!(
            "could not finish SGLang metadata compression: {error}"
        ))
    })
}

pub(crate) fn grpc_metadata(meta: &HashMap<String, String>) -> Value {
    Value::Object(
        meta.iter()
            .map(|(key, raw)| {
                let value =
                    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.clone()));
                (key.clone(), value)
            })
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::num::NonZeroUsize;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_backend_common::{OutputOptions, SamplingOptions, StopConditions};
    use serde_json::{Value, json};

    use super::config::OperatorPolicy;
    use super::{MetadataUploadService, OUTPUT_PATH, OperatorCache, grpc_metadata};

    fn policy(capacity: usize) -> OperatorPolicy {
        OperatorPolicy {
            capacity: NonZeroUsize::new(capacity).unwrap(),
            timeout: Duration::from_secs(10),
            io_timeout: Duration::from_secs(10),
            retry_max_times: 3,
            retry_min_delay: Duration::from_millis(1),
            retry_max_delay: Duration::from_millis(10),
            retry_factor: 2.0,
            retry_jitter: false,
        }
    }

    fn operator_cache(capacity: usize) -> Arc<OperatorCache> {
        Arc::new(OperatorCache::new(policy(capacity)))
    }

    fn service(capacity: usize) -> MetadataUploadService {
        MetadataUploadService {
            operators: operator_cache(capacity),
        }
    }

    fn fs_uri(path: &Path) -> String {
        format!("fs://{}", path.display())
    }

    fn request(extra_args: Value) -> dynamo_backend_common::PreprocessedRequest {
        dynamo_backend_common::PreprocessedRequest::builder()
            .model("model".to_string())
            .token_ids(vec![1])
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .stop_conditions(StopConditions::default())
            .extra_args(Some(extra_args))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn request_configuration_is_optional_and_strict() {
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": "fs:///tmp/metadata"}}
        }));
        assert!(
            service(2)
                .delivery_for(&configured)
                .await
                .unwrap()
                .uploader()
                .is_some()
        );

        let unconfigured = request(json!({"nvext": {}}));
        assert!(
            service(2)
                .delivery_for(&unconfigured)
                .await
                .unwrap()
                .uploader()
                .is_none()
        );

        let invalid = request(json!({
            "nvext": {"metadata_upload": {"url": "fs:///tmp", "format": "json"}}
        }));
        assert!(service(2).delivery_for(&invalid).await.is_err());
    }

    #[tokio::test]
    async fn uploads_python_compatible_msgpack_zstd_payload() {
        let directory = tempfile::tempdir().unwrap();
        let url = fs_uri(directory.path());
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": url}}
        }));
        let delivery = service(2).delivery_for(&configured).await.unwrap();
        let uploader = delivery.uploader().unwrap();
        uploader
            .upload(grpc_metadata(&std::collections::HashMap::from([
                ("id".into(), "sglang-1".into()),
                ("finish_reason".into(), r#"{"type":"stop"}"#.into()),
                ("output_token_logprobs".into(), r#"[[-0.1,101,"a"]]"#.into()),
            ])))
            .await
            .unwrap();

        let compressed = std::fs::read(directory.path().join("choice_0.msgpack.zst")).unwrap();
        assert!(
            zstd::zstd_safe::get_frame_content_size(&compressed)
                .unwrap()
                .is_some(),
            "the frame must advertise its content size for Python compatibility"
        );
        let msgpack = zstd::stream::decode_all(Cursor::new(compressed)).unwrap();
        let payload: Value = rmp_serde::from_slice(&msgpack).unwrap();
        assert_eq!(payload["schema_version"], 1);
        assert_eq!(payload["metadata"]["id"], "sglang-1");
        assert_eq!(payload["metadata"]["output_token_logprobs"][0][1], 101);
    }

    #[tokio::test]
    async fn uses_fallback_only_after_primary_failure() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("primary-is-a-file");
        std::fs::write(&primary, b"primary sentinel").unwrap();
        let fallback = directory.path().join("fallback");
        let configured = request(json!({
            "nvext": {"metadata_upload": {
                "url": fs_uri(&primary),
                "fallback_url": fs_uri(&fallback)
            }}
        }));
        let delivery = service(2).delivery_for(&configured).await.unwrap();
        let uploader = delivery.uploader().unwrap();

        uploader.upload(json!({"id": "sglang-2"})).await.unwrap();

        assert_eq!(std::fs::read(primary).unwrap(), b"primary sentinel");
        assert!(fallback.join(OUTPUT_PATH).is_file());
    }

    #[tokio::test]
    async fn accepts_a_registered_custom_scheme() {
        opendal::OperatorRegistry::get().register::<opendal::services::Memory>("custom-metadata");
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": "custom-metadata://rollout"}}
        }));

        assert!(
            service(2)
                .delivery_for(&configured)
                .await
                .unwrap()
                .uploader()
                .is_some()
        );
    }

    #[tokio::test]
    async fn operator_cache_evicts_least_recently_used_url() {
        let directory = tempfile::tempdir().unwrap();
        let first = fs_uri(&directory.path().join("first"));
        let second = fs_uri(&directory.path().join("second"));
        let third = fs_uri(&directory.path().join("third"));
        let cache = operator_cache(2);

        cache.get(first.clone(), "url").await.unwrap();
        cache.get(second.clone(), "url").await.unwrap();
        cache.get(first.clone(), "url").await.unwrap();
        cache.get(third.clone(), "url").await.unwrap();

        let operators = cache.operators.lock().await;
        assert!(operators.peek(&first).is_some());
        assert!(operators.peek(&second).is_none());
        assert!(operators.peek(&third).is_some());
    }
}

#[cfg(test)]
#[path = "metadata_upload/resilience_tests.rs"]
mod resilience_tests;
