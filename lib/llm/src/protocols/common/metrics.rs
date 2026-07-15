// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_runtime::protocols::annotated::Annotated;

/// Omit count fields from the serialized annotation when zero.
fn is_zero(value: &usize) -> bool {
    *value == 0
}

pub const ANNOTATION_LLM_METRICS: &str = "llm_metrics";

/// Marks the payload-only usage chunk. It carries the same `LLMMetricAnnotation`
/// payload as `ANNOTATION_LLM_METRICS` but on a dedicated tag so the client-path
/// `EventConverter` can strip it entirely, including data. It exists solely to
/// carry `usage` to the payload `DeltaAggregator` and is never sent to the client.
pub const ANNOTATION_PAYLOAD_USAGE: &str = "payload_usage";

/// Why a request's analytic image-token count was withheld. Emitted as the
/// `reason` label on `image_tokens_skipped_total` so operators can see *why* a
/// model has no image-token data instead of a silently-missing series. Bounded
/// set (never a URL / config value) to keep label cardinality fixed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTokenSkipReason {
    /// Model family/config not verified against the backend tokenization
    /// (e.g. the crate's analytic count would use values that don't match the
    /// model's real processor config).
    UnverifiedFamily,
    /// Not every image's dimensions resolved, so a summed count would be a
    /// plausible-but-wrong partial total.
    PartialResolution,
    /// Request carried `mm_processor_kwargs`, which the backend applies but the
    /// analytic counter cannot see — the estimate can't be trusted.
    RequestOverride,
}

impl ImageTokenSkipReason {
    /// Stable snake_case label for the `reason` metric dimension.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UnverifiedFamily => "unverified_family",
            Self::PartialResolution => "partial_resolution",
            Self::RequestOverride => "request_override",
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LLMMetricAnnotation {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub chunk_tokens: usize,
    pub cached_tokens: Option<usize>,
    /// Number of `image_url` content parts in the request (0 for text-only).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub image_count: usize,
    /// Number of `video_url` content parts in the request (0 for text-only).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub video_count: usize,
    /// Number of `audio_url` content parts in the request (0 for text-only).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub audio_count: usize,
    /// Analytic vision-token count for the request's images, set by the Rust
    /// frontend only when it can be trusted as a usage figure (see the guard in
    /// `preprocessor::resolve_image_token_usage`). `None` = not applicable
    /// (text-only), guard-skipped (see `image_tokens_skip_reason`), or a frontend
    /// path that doesn't compute it (Python processors).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens: Option<usize>,
    /// Why `image_tokens` was withheld for an image-bearing request. `Some` only
    /// on the Rust path that attempted the count; drives
    /// `image_tokens_skipped_total{reason}`. Never set for text-only requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_tokens_skip_reason: Option<ImageTokenSkipReason>,
    /// Prefill worker ID (for TTFT attribution in disaggregated mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,
    /// Prefill worker DP rank
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,
    /// Prefill worker type ("prefill" or "decode") for Prometheus metric labeling.
    /// Stored at routing time to avoid expensive MDC lookup when updating TTFT metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_type: Option<String>,
    /// Decode worker ID (for ITL attribution in disaggregated mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,
    /// Decode worker DP rank
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_dp_rank: Option<u32>,
    /// Decode worker type ("prefill" or "decode") for Prometheus metric labeling.
    /// Stored at routing time to avoid expensive MDC lookup when updating ITL metrics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenize_latency: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detokenize_total_latency: Option<Duration>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detokenize_count: Option<u64>,
}

impl LLMMetricAnnotation {
    /// Convert this metrics struct to an Annotated event.
    pub fn to_annotation<T>(&self) -> Result<Annotated<T>, serde_json::Error> {
        Annotated::from_annotation(ANNOTATION_LLM_METRICS, self)
    }

    /// Extract LLM metrics from an Annotated event, if present.
    pub fn from_annotation<T>(
        annotation: &Annotated<T>,
    ) -> Result<Option<LLMMetricAnnotation>, Box<dyn std::error::Error>> {
        // Metrics ride on an event-tagged annotation: `ANNOTATION_LLM_METRICS`
        // for per-chunk metrics (kept on the client stream as content, comment
        // stripped) and `ANNOTATION_PAYLOAD_USAGE` for the payload-only final usage
        // chunk. Both carry the serialized `LLMMetricAnnotation` as their comment.
        let Some(event) = annotation.event.as_deref() else {
            return Ok(None);
        };
        if event != ANNOTATION_LLM_METRICS && event != ANNOTATION_PAYLOAD_USAGE {
            return Ok(None);
        }
        let comments = annotation
            .comment
            .as_ref()
            .ok_or("missing comments block")?;
        if comments.len() != 1 {
            return Err("malformed comments block - expected exactly 1 comment".into());
        }
        let metrics: LLMMetricAnnotation = serde_json::from_str(&comments[0])?;
        Ok(Some(metrics))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The metric `reason` label (`as_str`) must match the serde snake_case wire
    /// form for every variant, so a future rename can't silently diverge the two.
    #[test]
    fn image_token_skip_reason_label_matches_serde() {
        for reason in [
            ImageTokenSkipReason::UnverifiedFamily,
            ImageTokenSkipReason::PartialResolution,
            ImageTokenSkipReason::RequestOverride,
        ] {
            let serde_form = serde_json::to_string(&reason).unwrap();
            // to_string wraps in quotes: "unverified_family"
            assert_eq!(serde_form, format!("\"{}\"", reason.as_str()));
        }
    }
}
