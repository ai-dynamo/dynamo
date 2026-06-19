// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! S3 implementation of [`SegmentSink`](super::jsonl_gz::SegmentSink).
//!
//! Each rotated segment is buffered in memory across one or more
//! `append_to_segment` calls, then uploaded to S3 as a single object on
//! `close_segment`. The object body is a concatenation of self-contained
//! gzip members, so it can be downloaded and `gunzip`'d as a normal
//! `.jsonl.gz` file.
//!
//! Object keys:
//! ```text
//! {prefix}/[{deployment}/]YYYY/MM/DD/HH/{instance}-{startup}-{seq:06}.jsonl.gz
//! ```
//! `instance` is resolved at construction from the supplied override, or
//! from the `POD_NAME` env var (Kubernetes downward API), or from
//! `gethostname`, in that order. `startup` is an 8-character random hex
//! generated once per process so a pod that gets restarted with the same
//! name (StatefulSet) does not overwrite its predecessor's segments.
//!
//! Failure policy: rely on the SDK's default retry behavior (3 attempts,
//! exponential backoff with jitter). On terminal failure we log and drop
//! the segment — never propagate, so a single bad upload cannot kill the
//! writer task.

use std::collections::HashMap;
use std::sync::Mutex;

use anyhow::{Context as _, Result, anyhow};
use async_trait::async_trait;
use aws_sdk_s3::Client;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::ServerSideEncryption;
use chrono::{DateTime, Datelike, Timelike, Utc};

use super::jsonl_gz::SegmentSink;

/// Identity strings baked into S3 object keys.
#[derive(Clone, Debug)]
pub struct S3SegmentIdentity {
    pub instance: String,
    pub startup: String,
    pub deployment: Option<String>,
}

impl S3SegmentIdentity {
    /// Resolve instance and startup ids using the standard fallback
    /// chain: explicit override → POD_NAME → hostname → "unknown".
    pub fn resolve(instance_override: Option<String>, deployment: Option<String>) -> Self {
        let instance = instance_override
            .filter(|s| !s.trim().is_empty())
            .or_else(|| {
                std::env::var("POD_NAME")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            })
            .or_else(|| {
                std::env::var("HOSTNAME")
                    .ok()
                    .filter(|s| !s.trim().is_empty())
            })
            .or_else(|| {
                std::fs::read_to_string("/etc/hostname")
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .unwrap_or_else(|| "unknown".to_string());
        let startup = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
        Self {
            instance,
            startup,
            deployment,
        }
    }
}

/// Server-side encryption settings for the S3 audit sink.
#[derive(Clone, Debug, Default)]
pub struct S3SinkEncryption {
    pub sse: Option<String>,
    pub kms_key_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct S3SegmentSinkConfig {
    pub bucket: String,
    pub prefix: String,
    pub identity: S3SegmentIdentity,
    pub encryption: S3SinkEncryption,
}

pub struct S3SegmentSink {
    client: Client,
    config: S3SegmentSinkConfig,
    /// Per-seq accumulator. `append_to_segment` extends the entry; the
    /// matching `close_segment` removes it and runs the upload.
    segments: Mutex<HashMap<u64, Vec<u8>>>,
}

impl S3SegmentSink {
    pub fn new(client: Client, config: S3SegmentSinkConfig) -> Self {
        Self {
            client,
            config,
            segments: Mutex::new(HashMap::new()),
        }
    }

    /// Construct a default-credentials S3 client honoring an optional
    /// region and endpoint URL. Callers should reuse one client across
    /// many sinks where possible, but each audit sink only needs one,
    /// so this is a reasonable per-sink helper.
    pub async fn build_client(
        region: Option<String>,
        endpoint_url: Option<String>,
    ) -> Result<Client> {
        let mut loader = aws_config::defaults(aws_config::BehaviorVersion::latest());
        if let Some(region) = region {
            loader = loader.region(aws_sdk_s3::config::Region::new(region));
        }
        let sdk_config = loader.load().await;

        let mut s3_config_builder = aws_sdk_s3::config::Builder::from(&sdk_config);
        if let Some(endpoint) = endpoint_url.as_deref() {
            s3_config_builder = s3_config_builder.endpoint_url(endpoint);
            // LocalStack and MinIO require path-style addressing.
            s3_config_builder = s3_config_builder.force_path_style(true);
        }
        Ok(Client::from_conf(s3_config_builder.build()))
    }

    fn build_key(&self, seq: u64) -> String {
        format_object_key(
            &self.config.prefix,
            self.config.identity.deployment.as_deref(),
            Utc::now(),
            &self.config.identity.instance,
            &self.config.identity.startup,
            seq,
        )
    }

    async fn upload(&self, key: &str, body: Vec<u8>) -> Result<()> {
        let body_len = body.len();
        let mut put = self
            .client
            .put_object()
            .bucket(&self.config.bucket)
            .key(key)
            .body(ByteStream::from(body))
            .content_type("application/x-ndjson")
            .content_encoding("gzip");

        if let Some(sse) = self.config.encryption.sse.as_deref() {
            match sse {
                "AES256" => put = put.server_side_encryption(ServerSideEncryption::Aes256),
                "aws:kms" => {
                    put = put.server_side_encryption(ServerSideEncryption::AwsKms);
                    if let Some(kms_key) = self.config.encryption.kms_key_id.as_deref() {
                        put = put.ssekms_key_id(kms_key);
                    }
                }
                other => {
                    tracing::warn!(
                        sse = other,
                        "audit s3: unknown SSE mode, falling back to bucket default"
                    );
                }
            }
        }

        put.send()
            .await
            .with_context(|| format!("audit s3 put_object key={key} size={body_len}"))?;
        tracing::debug!(key, body_len, "audit s3 segment uploaded");
        Ok(())
    }
}

#[async_trait]
impl SegmentSink for S3SegmentSink {
    async fn append_to_segment(&self, seq: u64, gz_bytes: Vec<u8>) -> Result<()> {
        let mut segments = self
            .segments
            .lock()
            .map_err(|_| anyhow!("audit s3: lock poisoned"))?;
        segments
            .entry(seq)
            .or_default()
            .extend_from_slice(&gz_bytes);
        Ok(())
    }

    async fn close_segment(&self, seq: u64) -> Result<()> {
        let body = {
            let mut segments = self
                .segments
                .lock()
                .map_err(|_| anyhow!("audit s3: lock poisoned"))?;
            segments.remove(&seq)
        };
        let Some(body) = body else {
            // close_segment may fire on an empty segment (e.g. shutdown
            // before any record arrives). That's fine — nothing to ship.
            return Ok(());
        };
        if body.is_empty() {
            return Ok(());
        }
        let body_len = body.len();
        let key = self.build_key(seq);
        let start = std::time::Instant::now();
        let result = self.upload(&key, body).await;
        let elapsed = start.elapsed().as_secs_f64();

        use crate::audit::metrics::{
            AUDIT_S3_SEGMENT_SIZE_BYTES, AUDIT_S3_SEGMENTS_TOTAL, AUDIT_S3_UPLOAD_DURATION_SECONDS,
        };
        AUDIT_S3_UPLOAD_DURATION_SECONDS
            .with_label_values(&[] as &[&str])
            .observe(elapsed);
        AUDIT_S3_SEGMENT_SIZE_BYTES
            .with_label_values(&[] as &[&str])
            .observe(body_len as f64);

        match result {
            Ok(()) => {
                AUDIT_S3_SEGMENTS_TOTAL.with_label_values(&["ok"]).inc();
            }
            Err(err) => {
                AUDIT_S3_SEGMENTS_TOTAL.with_label_values(&["failed"]).inc();
                tracing::warn!(
                    err = format!("{err:#}"),
                    key,
                    seq,
                    "audit s3 upload failed; dropping segment"
                );
            }
        }
        Ok(())
    }
}

/// Build an Athena-friendly object key. Pure function so we can unit-test
/// the format independent of AWS.
fn format_object_key(
    prefix: &str,
    deployment: Option<&str>,
    when: DateTime<Utc>,
    instance: &str,
    startup: &str,
    seq: u64,
) -> String {
    let prefix = prefix.trim_end_matches('/');
    let dgd = match deployment {
        Some(d) if !d.is_empty() => format!("{d}/"),
        _ => String::new(),
    };
    let (year, month, day, hour) = (when.year(), when.month(), when.day(), when.hour());
    format!(
        "{prefix}/{dgd}{year:04}/{month:02}/{day:02}/{hour:02}/{instance}-{startup}-{seq:06}.jsonl.gz"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn fixed_when() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 17, 14, 0, 0).unwrap()
    }

    #[test]
    fn key_includes_deployment_when_set() {
        let key = format_object_key(
            "dynamo-audit",
            Some("dgd-prod"),
            fixed_when(),
            "frontend-7c9d-x4kqp",
            "3f7a2b9c",
            42,
        );
        assert_eq!(
            key,
            "dynamo-audit/dgd-prod/2026/06/17/14/frontend-7c9d-x4kqp-3f7a2b9c-000042.jsonl.gz"
        );
    }

    #[test]
    fn key_omits_deployment_segment_when_unset() {
        let key = format_object_key("dynamo-audit", None, fixed_when(), "host", "abcd1234", 0);
        assert_eq!(
            key,
            "dynamo-audit/2026/06/17/14/host-abcd1234-000000.jsonl.gz"
        );
    }

    #[test]
    fn key_treats_empty_deployment_as_unset() {
        let key = format_object_key(
            "dynamo-audit",
            Some(""),
            fixed_when(),
            "host",
            "abcd1234",
            0,
        );
        assert_eq!(
            key,
            "dynamo-audit/2026/06/17/14/host-abcd1234-000000.jsonl.gz"
        );
    }

    #[test]
    fn prefix_trailing_slash_is_normalized() {
        let key = format_object_key("dynamo-audit/", None, fixed_when(), "host", "abcd1234", 0);
        assert_eq!(
            key,
            "dynamo-audit/2026/06/17/14/host-abcd1234-000000.jsonl.gz"
        );
    }

    #[test]
    #[serial_test::serial]
    fn identity_falls_back_to_pod_name_when_no_override() {
        let prev = std::env::var("POD_NAME").ok();
        unsafe {
            std::env::set_var("POD_NAME", "frontend-abc-x");
        }
        let id = S3SegmentIdentity::resolve(None, None);
        assert_eq!(id.instance, "frontend-abc-x");
        unsafe {
            match prev {
                Some(v) => std::env::set_var("POD_NAME", v),
                None => std::env::remove_var("POD_NAME"),
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn identity_explicit_override_beats_pod_name() {
        let prev = std::env::var("POD_NAME").ok();
        unsafe {
            std::env::set_var("POD_NAME", "from-pod");
        }
        let id = S3SegmentIdentity::resolve(Some("explicit-id".into()), Some("dgd".into()));
        assert_eq!(id.instance, "explicit-id");
        assert_eq!(id.deployment.as_deref(), Some("dgd"));
        assert_eq!(id.startup.len(), 8);
        unsafe {
            match prev {
                Some(v) => std::env::set_var("POD_NAME", v),
                None => std::env::remove_var("POD_NAME"),
            }
        }
    }
}
