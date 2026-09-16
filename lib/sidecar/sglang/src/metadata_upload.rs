// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Out-of-band storage for SGLang response metadata.

use std::collections::HashMap;
use std::io::Write;
use std::time::Duration;

use dynamo_backend_common::{DynamoError, PreprocessedRequest};
use opendal::{Buffer, Operator};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::client;

const OUTPUT_PATH: &str = "choice_0.msgpack.zst";
const WRITE_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) struct MetadataUploader {
    primary: Operator,
    fallback: Option<String>,
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

impl MetadataUploader {
    pub(crate) async fn from_request(
        request: &PreprocessedRequest,
        enabled: bool,
    ) -> Result<Option<Self>, DynamoError> {
        if !enabled {
            return Ok(None);
        }
        let Some(config) = request
            .extra_args
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|extra| extra.get("nvext"))
            .and_then(Value::as_object)
            .and_then(|nvext| nvext.get("metadata_upload"))
        else {
            return Ok(None);
        };
        if config.is_null() {
            return Ok(None);
        }
        let config = MetadataUploadConfig::deserialize(config).map_err(|error| {
            client::invalid_arg(format!("invalid extra_args.nvext.metadata_upload: {error}"))
        })?;
        // Ensure built-in services are registered when this crate is linked statically.
        opendal::install_default();
        let primary = operator_from_url(config.url, "url").await?;
        let fallback = config
            .fallback_url
            .map(|url| normalize_url(url, "fallback_url"))
            .transpose()?;
        Ok(Some(Self { primary, fallback }))
    }

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
                let fallback = operator_from_url(fallback_url.clone(), "fallback_url").await?;
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

async fn operator_from_url(raw: String, field: &'static str) -> Result<Operator, DynamoError> {
    tokio::task::spawn_blocking(move || {
        let raw = normalize_url(raw, field)?;
        Operator::from_uri(raw.as_str()).map_err(|error| {
            client::invalid_arg(format!(
                "could not configure `{}` metadata upload destination: {error}",
                raw.split_once(':')
                    .map(|(scheme, _)| scheme)
                    .unwrap_or("unknown")
            ))
        })
    })
    .await
    .map_err(|error| {
        client::protocol_error(format!("metadata upload setup task failed: {error}"))
    })?
}

async fn write(operator: &Operator, data: Buffer) -> Result<(), DynamoError> {
    match tokio::time::timeout(WRITE_TIMEOUT, operator.write(OUTPUT_PATH, data)).await {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(error)) => Err(upload_error(operator, error)),
        Err(_) => Err(client::protocol_error(format!(
            "SGLang metadata upload to {} timed out after {WRITE_TIMEOUT:?}",
            operator.info().root()
        ))),
    }
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
    use std::path::Path;

    use dynamo_backend_common::{OutputOptions, SamplingOptions, StopConditions};
    use serde_json::{Value, json};

    use super::{MetadataUploader, OUTPUT_PATH, grpc_metadata};

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
    async fn metadata_upload_is_rl_gated_and_strict() {
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": "fs:///tmp/metadata"}}
        }));
        assert!(
            MetadataUploader::from_request(&configured, false)
                .await
                .unwrap()
                .is_none()
        );
        assert!(
            MetadataUploader::from_request(&configured, true)
                .await
                .unwrap()
                .is_some()
        );

        let invalid = request(json!({
            "nvext": {"metadata_upload": {"url": "fs:///tmp", "format": "json"}}
        }));
        assert!(
            MetadataUploader::from_request(&invalid, true)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn uploads_python_compatible_msgpack_zstd_payload() {
        let directory = tempfile::tempdir().unwrap();
        let url = fs_uri(directory.path());
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": url}}
        }));
        let uploader = MetadataUploader::from_request(&configured, true)
            .await
            .unwrap()
            .unwrap();
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
        let uploader = MetadataUploader::from_request(&configured, true)
            .await
            .unwrap()
            .unwrap();

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
            MetadataUploader::from_request(&configured, true)
                .await
                .unwrap()
                .is_some()
        );
    }
}
