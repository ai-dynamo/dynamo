// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Out-of-band storage for SGLang response metadata.

use std::collections::HashMap;
use std::io::Write;

use dynamo_backend_common::{DynamoError, PreprocessedRequest};
use opendal::{Buffer, Operator};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::client;

const OUTPUT_PATH: &str = "choice_0.msgpack.zst";

/// The SGLang sidecar supports one output choice, so one uploader owns the
/// primary and optional fallback destinations for that choice.
#[derive(Clone)]
pub(crate) struct MetadataUploader {
    primary: Operator,
    fallback: Option<Operator>,
}

#[derive(Serialize)]
struct MetadataPayload<'a> {
    schema_version: u8,
    metadata: &'a Value,
}

impl MetadataUploader {
    pub(crate) fn from_request(
        request: &PreprocessedRequest,
        enabled: bool,
    ) -> Result<Option<Self>, DynamoError> {
        if !enabled {
            return Ok(None);
        }
        let Some(settings) = request
            .extra_args
            .as_ref()
            .and_then(Value::as_object)
            .and_then(|extra| extra.get("nvext"))
            .and_then(Value::as_object)
            .and_then(|nvext| nvext.get("metadata_upload"))
        else {
            return Ok(None);
        };
        if settings.is_null() {
            return Ok(None);
        }
        let settings = settings.as_object().ok_or_else(|| {
            client::invalid_arg("extra_args.nvext.metadata_upload must be an object")
        })?;
        if let Some(unexpected) = settings
            .keys()
            .find(|key| !matches!(key.as_str(), "url" | "fallback_url"))
        {
            return Err(client::invalid_arg(format!(
                "extra_args.nvext.metadata_upload.{unexpected} is not supported"
            )));
        }
        let url = settings
            .get("url")
            .ok_or_else(|| client::invalid_arg("extra_args.nvext.metadata_upload.url is required"))?
            .as_str()
            .ok_or_else(|| {
                client::invalid_arg("extra_args.nvext.metadata_upload.url must be a string")
            })?;
        // The sidecar is also linked into a Python static library, where
        // OpenDAL's constructor-based automatic registration is not reliable.
        opendal::install_default();
        let primary = operator_from_url(url, "url")?;
        let fallback = settings
            .get("fallback_url")
            .filter(|value| !value.is_null())
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    client::invalid_arg(
                        "extra_args.nvext.metadata_upload.fallback_url must be a string",
                    )
                })
            })
            .transpose()?
            .map(|url| operator_from_url(url, "fallback_url"))
            .transpose()?;
        Ok(Some(Self { primary, fallback }))
    }

    pub(crate) async fn upload(&self, metadata: Value) -> Result<(), DynamoError> {
        if metadata.as_object().is_some_and(Map::is_empty) {
            return Ok(());
        }
        let compressed: Buffer = tokio::task::spawn_blocking(move || encode(metadata))
            .await
            .map_err(|error| {
                client::protocol_error(format!("SGLang metadata encoding task failed: {error}"))
            })??
            .into();
        match self.primary.write(OUTPUT_PATH, compressed.clone()).await {
            Ok(_) => Ok(()),
            Err(primary_error) => {
                let Some(fallback) = self.fallback.as_ref() else {
                    return Err(upload_error(&self.primary, primary_error));
                };
                tracing::warn!("primary metadata upload failed; attempting configured fallback");
                fallback
                    .write(OUTPUT_PATH, compressed)
                    .await
                    .map_err(|error| upload_error(fallback, error))?;
                Ok(())
            }
        }
    }
}

fn operator_from_url(raw: &str, field: &str) -> Result<Operator, DynamoError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(client::invalid_arg(format!(
            "extra_args.nvext.metadata_upload.{field} must not be empty"
        )));
    }
    let uri = normalize_fsspec_uri(raw, field)?;
    Operator::from_uri(uri.as_str()).map_err(|error| {
        client::invalid_arg(format!(
            "could not configure `{}` metadata upload destination: {error}",
            uri.split_once(':')
                .map(|(scheme, _)| scheme)
                .unwrap_or("unknown")
        ))
    })
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

/// OpenDAL uses canonical service names in URIs. Preserve the fsspec spellings
/// accepted by the Python implementation at the public request boundary.
fn normalize_fsspec_uri(raw: &str, field: &str) -> Result<String, DynamoError> {
    let uri = url::Url::parse(raw).map_err(|error| {
        client::invalid_arg(format!(
            "extra_args.nvext.metadata_upload.{field} is not a valid URL: {error}"
        ))
    })?;
    let canonical_scheme = match uri.scheme() {
        "file" => "fs",
        "gs" => "gcs",
        "az" => "azblob",
        scheme => scheme,
    };
    Ok(format!("{canonical_scheme}{}", &raw[uri.scheme().len()..]))
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

    use dynamo_backend_common::{OutputOptions, SamplingOptions, StopConditions};
    use serde_json::{Value, json};

    use super::{MetadataUploader, OUTPUT_PATH, grpc_metadata, normalize_fsspec_uri};

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

    #[test]
    fn normalizes_fsspec_scheme_aliases() {
        assert_eq!(
            normalize_fsspec_uri("file:///tmp/metadata", "url")
                .unwrap()
                .as_str(),
            "fs:///tmp/metadata"
        );
        assert_eq!(
            normalize_fsspec_uri("gs://bucket/metadata", "url")
                .unwrap()
                .as_str(),
            "gcs://bucket/metadata"
        );
        assert_eq!(
            normalize_fsspec_uri("az://container/metadata", "url")
                .unwrap()
                .as_str(),
            "azblob://container/metadata"
        );
    }

    #[test]
    fn metadata_upload_is_rl_gated_and_strict() {
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": "file:///tmp/metadata"}}
        }));
        assert!(
            MetadataUploader::from_request(&configured, false)
                .unwrap()
                .is_none()
        );
        assert!(
            MetadataUploader::from_request(&configured, true)
                .unwrap()
                .is_some()
        );

        let invalid = request(json!({
            "nvext": {"metadata_upload": {"url": "file:///tmp", "format": "json"}}
        }));
        assert!(MetadataUploader::from_request(&invalid, true).is_err());
    }

    #[test]
    fn grpc_metadata_decodes_json_values_and_preserves_plain_strings() {
        let metadata = grpc_metadata(&std::collections::HashMap::from([
            ("finish_reason".into(), r#"{"type":"stop"}"#.into()),
            ("legacy".into(), "plain".into()),
        ]));
        assert_eq!(metadata["finish_reason"]["type"], "stop");
        assert_eq!(metadata["legacy"], "plain");
    }

    #[tokio::test]
    async fn uploads_python_compatible_msgpack_zstd_payload() {
        let directory = tempfile::tempdir().unwrap();
        let url = url::Url::from_directory_path(directory.path())
            .unwrap()
            .to_string();
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": url}}
        }));
        let uploader = MetadataUploader::from_request(&configured, true)
            .unwrap()
            .unwrap();
        uploader
            .upload(json!({
                "id": "sglang-1",
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[-0.1, 101, "a"]]
            }))
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
                "url": url::Url::from_file_path(&primary).unwrap(),
                "fallback_url": url::Url::from_directory_path(&fallback).unwrap()
            }}
        }));
        let uploader = MetadataUploader::from_request(&configured, true)
            .unwrap()
            .unwrap();

        uploader.upload(json!({"id": "sglang-2"})).await.unwrap();

        assert_eq!(std::fs::read(primary).unwrap(), b"primary sentinel");
        assert!(fallback.join(OUTPUT_PATH).is_file());
    }

    #[tokio::test]
    async fn does_not_write_fallback_when_primary_succeeds() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("primary");
        let fallback = directory.path().join("fallback");
        let configured = request(json!({
            "nvext": {"metadata_upload": {
                "url": url::Url::from_directory_path(&primary).unwrap(),
                "fallback_url": url::Url::from_directory_path(&fallback).unwrap()
            }}
        }));
        let uploader = MetadataUploader::from_request(&configured, true)
            .unwrap()
            .unwrap();

        uploader.upload(json!({"id": "sglang-3"})).await.unwrap();

        assert!(primary.join(OUTPUT_PATH).is_file());
        assert!(!fallback.join(OUTPUT_PATH).exists());
    }

    #[tokio::test]
    async fn returns_fallback_error_when_both_destinations_fail() {
        let directory = tempfile::tempdir().unwrap();
        let primary = directory.path().join("primary-is-a-file");
        let fallback = directory.path().join("fallback-is-a-file");
        std::fs::write(&primary, b"primary sentinel").unwrap();
        std::fs::write(&fallback, b"fallback sentinel").unwrap();
        let configured = request(json!({
            "nvext": {"metadata_upload": {
                "url": url::Url::from_file_path(&primary).unwrap(),
                "fallback_url": url::Url::from_file_path(&fallback).unwrap()
            }}
        }));
        let uploader = MetadataUploader::from_request(&configured, true)
            .unwrap()
            .unwrap();

        let error = uploader
            .upload(json!({"id": "sglang-4"}))
            .await
            .unwrap_err();

        assert!(error.to_string().contains("fallback-is-a-file"));
        assert_eq!(std::fs::read(primary).unwrap(), b"primary sentinel");
        assert_eq!(std::fs::read(fallback).unwrap(), b"fallback sentinel");
    }

    #[test]
    fn accepts_a_registered_custom_scheme() {
        opendal::OperatorRegistry::get().register::<opendal::services::Memory>("custom-metadata");
        let configured = request(json!({
            "nvext": {"metadata_upload": {"url": "custom-metadata://rollout"}}
        }));

        assert!(
            MetadataUploader::from_request(&configured, true)
                .unwrap()
                .is_some()
        );
    }
}
