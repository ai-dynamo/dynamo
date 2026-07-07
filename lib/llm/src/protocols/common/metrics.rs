// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_runtime::protocols::annotated::Annotated;

pub const ANNOTATION_LLM_METRICS: &str = "llm_metrics";

/// Marks the payload-only usage chunk. It carries the same `LLMMetricAnnotation`
/// payload as `ANNOTATION_LLM_METRICS` but on a dedicated tag so the client-path
/// `EventConverter` can strip it entirely, including data. It exists solely to
/// carry `usage` to the payload `DeltaAggregator` and is never sent to the client.
pub const ANNOTATION_PAYLOAD_USAGE: &str = "payload_usage";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct LLMMetricAnnotation {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub chunk_tokens: usize,
    pub cached_tokens: Option<usize>,
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
