// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::replay::{
    OfflineRuntimeEvidence, PerRequestRecord, ReplayCaptureOptions, ReplayReport,
    TraceDistributionStats,
};

pub const CANONICAL_SCHEMA_VERSION: &str = "dynamo.offline-replay.v1";
pub const CANONICAL_RESULT_EXCLUSIONS: [&str; 4] = [
    "/summary/wall_time_ms",
    "/summary/processed_tokens_per_s",
    "/summary/processed_output_tokens_per_s",
    "/planner/html_report_path",
];

/// Capture coverage included in a canonical report.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct CanonicalReplayCoverage {
    pub capture_per_request: bool,
    pub capture_planner_details: bool,
    pub capture_canonical_evidence: bool,
    pub per_request_records: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pressure: Option<crate::replay::PressureEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_ingest: Option<crate::replay::KvIngestEvidence>,
}

impl CanonicalReplayCoverage {
    pub fn from_report(report: &ReplayReport, options: ReplayCaptureOptions) -> Self {
        Self {
            capture_per_request: options.effective_per_request() || !report.per_request.is_empty(),
            capture_planner_details: options.capture_lifecycle_evidence,
            capture_canonical_evidence: options.capture_canonical_evidence,
            per_request_records: report.per_request.len(),
            pressure: report.runtime_evidence.pressure.clone(),
            kv_ingest: report.runtime_evidence.kv_ingest.clone(),
        }
    }
}

/// Canonical, byte-stable representation of one offline replay result.
///
/// Dynamo owns backend/router/planner identity. It supplies that identity as a
/// JSON object, keeping this crate independent of Dynamo configuration types.
#[derive(Debug, Serialize)]
pub struct CanonicalReplayRecord {
    pub coverage: Value,
    pub metadata: Value,
    pub per_request: Value,
    pub planner: Value,
    pub summary: Value,
}

impl CanonicalReplayRecord {
    pub fn build(
        report: &ReplayReport,
        mut metadata: Value,
        coverage: &CanonicalReplayCoverage,
        mut planner: Value,
    ) -> Result<Self> {
        validate_report_finite(report)?;
        validate_json_finite("/metadata", &metadata)?;
        validate_json_finite("/coverage", &serde_json::to_value(coverage)?)?;
        validate_json_finite("/planner", &planner)?;

        let metadata_object = metadata
            .as_object_mut()
            .context("canonical replay metadata must be an object")?;
        metadata_object.insert(
            "schema_version".to_string(),
            Value::String(CANONICAL_SCHEMA_VERSION.to_string()),
        );
        metadata_object.insert(
            "result_exclusions".to_string(),
            serde_json::to_value(CANONICAL_RESULT_EXCLUSIONS)?,
        );
        if let Some(planner_metadata) = planner.get("metadata").and_then(Value::as_object).cloned()
        {
            metadata_object.insert("planner".to_string(), Value::Object(planner_metadata));
        }

        let mut summary = serde_json::to_value(report)?;
        let summary_object = summary
            .as_object_mut()
            .context("serialized replay summary must be an object")?;
        summary_object.remove("wall_time_ms");
        summary_object.remove("processed_tokens_per_s");
        summary_object.remove("processed_output_tokens_per_s");

        let mut per_request = serde_json::to_value(&report.per_request)?;
        let records = per_request
            .as_array_mut()
            .context("serialized per-request replay details must be an array")?;
        records.sort_unstable_by(|left, right| {
            let left_uuid = left.get("uuid").and_then(Value::as_str).unwrap_or_default();
            let right_uuid = right
                .get("uuid")
                .and_then(Value::as_str)
                .unwrap_or_default();
            left_uuid.cmp(right_uuid)
        });

        if let Some(planner_object) = planner.as_object_mut() {
            planner_object.remove("html_report_path");
        }

        Ok(Self {
            coverage: canonicalize_json(serde_json::to_value(coverage)?),
            metadata: canonicalize_json(metadata),
            per_request: canonicalize_json(per_request),
            planner: canonicalize_json(planner),
            summary: canonicalize_json(summary),
        })
    }

    pub fn into_json_line(self) -> Result<Vec<u8>> {
        let mut line =
            serde_json::to_vec(&self).context("failed to serialize canonical replay JSON")?;
        line.push(b'\n');
        Ok(line)
    }
}

fn validate_report_finite(report: &ReplayReport) -> Result<()> {
    let throughput = &report.throughput;
    for (path, value) in [
        ("/summary/duration_ms", throughput.duration_ms),
        ("/summary/wall_time_ms", throughput.wall_time_ms),
        (
            "/summary/request_throughput_rps",
            throughput.request_throughput_rps,
        ),
        (
            "/summary/input_throughput_tok_s",
            throughput.input_throughput_tok_s,
        ),
        (
            "/summary/output_throughput_tok_s",
            throughput.output_throughput_tok_s,
        ),
        (
            "/summary/total_throughput_tok_s",
            throughput.total_throughput_tok_s,
        ),
        (
            "/summary/prefill_worker_seconds",
            throughput.prefill_worker_seconds,
        ),
        (
            "/summary/decode_worker_seconds",
            throughput.decode_worker_seconds,
        ),
        ("/summary/gpu_hours", throughput.gpu_hours),
        (
            "/summary/prefix_cache_reused_ratio",
            report.prefix_cache_reused_ratio,
        ),
        (
            "/summary/first_admission_prefix_cache_reused_ratio",
            report.first_admission_prefix_cache_reused_ratio,
        ),
    ] {
        ensure_finite(path, value)?;
    }

    validate_distribution("/summary/ttft", &report.latency.ttft)?;
    validate_distribution("/summary/ttst", &report.latency.ttst)?;
    validate_distribution("/summary/tpot", &report.latency.tpot)?;
    validate_distribution("/summary/itl", &report.latency.itl.distribution)?;
    ensure_finite("/summary/max_itl_ms", report.latency.itl.max_ms)?;
    validate_distribution("/summary/e2e", &report.latency.e2e)?;
    validate_distribution(
        "/summary/output_token_throughput_per_user",
        &report.latency.output_token_throughput_per_user,
    )?;
    if let Some(goodput) = &report.goodput {
        ensure_finite(
            "/summary/goodput_request_throughput_rps",
            goodput.request_throughput_rps,
        )?;
        ensure_finite(
            "/summary/goodput_output_throughput_tok_s",
            goodput.output_throughput_tok_s,
        )?;
    }
    for record in &report.per_request {
        validate_per_request_finite(record)?;
    }
    validate_runtime_evidence_finite(&report.runtime_evidence)
}

fn validate_distribution(path: &str, stats: &TraceDistributionStats) -> Result<()> {
    for (field, value) in [
        ("mean_ms", stats.mean_ms),
        ("min_ms", stats.min_ms),
        ("max_ms", stats.max_ms),
        ("median_ms", stats.median_ms),
        ("p75_ms", stats.p75_ms),
        ("p90_ms", stats.p90_ms),
        ("p95_ms", stats.p95_ms),
        ("p99_ms", stats.p99_ms),
        ("std_ms", stats.std_ms),
    ] {
        ensure_finite(&format!("{path}/{field}"), value)?;
    }
    Ok(())
}

fn validate_per_request_finite(record: &PerRequestRecord) -> Result<()> {
    let path = format!("/per_request/{}", record.uuid);
    for (field, value) in [
        ("arrival_time_ms", Some(record.arrival_time_ms)),
        ("first_admit_ms", record.first_admit_ms),
        ("terminal_time_ms", Some(record.terminal_time_ms)),
        ("first_token_ms", record.first_token_ms),
        ("last_token_ms", record.last_token_ms),
        ("ttft_ms", record.ttft_ms),
        ("ttst_ms", record.ttst_ms),
        ("e2e_latency_ms", record.e2e_latency_ms),
        ("itl_ms", record.itl_ms),
        ("prefill_admit_ms", record.prefill_admit_ms),
        ("source_held_ms", record.source_held_ms),
        ("destination_reserved_ms", record.destination_reserved_ms),
        ("destination_activated_ms", record.destination_activated_ms),
        ("decode_admit_ms", record.decode_admit_ms),
        ("source_released_ms", record.source_released_ms),
    ] {
        if let Some(value) = value {
            ensure_finite(&format!("{path}/{field}"), value)?;
        }
    }
    for (index, route) in record.routing_history.iter().enumerate() {
        for (field, value) in [
            ("queue_entered_at_ms", route.queue_entered_at_ms),
            ("released_at_ms", route.released_at_ms),
            ("queue_wait_ms", route.queue_wait_ms),
        ] {
            if let Some(value) = value {
                ensure_finite(&format!("{path}/routing_history/{index}/{field}"), value)?;
            }
        }
    }
    for (index, admission) in record.admission_history.iter().enumerate() {
        ensure_finite(
            &format!("{path}/admission_history/{index}/at_ms"),
            admission.at_ms,
        )?;
    }
    Ok(())
}

fn validate_runtime_evidence_finite(evidence: &OfflineRuntimeEvidence) -> Result<()> {
    for operation in &evidence.lifecycle_operations {
        ensure_finite(
            &format!(
                "/runtime_evidence/lifecycle_operations/{}/at_ms",
                operation.operation_ordinal
            ),
            operation.at_ms,
        )?;
    }
    if let Some(pressure) = &evidence.pressure {
        for record in &pressure.records {
            let path = format!(
                "/runtime_evidence/pressure/records/{}",
                record.pressure_ordinal
            );
            ensure_finite(&format!("{path}/at_ms"), record.at_ms)?;
            if let Some(at_ms) = record.readmitted_at_ms {
                ensure_finite(&format!("{path}/readmitted_at_ms"), at_ms)?;
            }
        }
    }
    if let Some(kv_ingest) = &evidence.kv_ingest {
        for (boundary, stats) in &kv_ingest.boundaries {
            let path = format!("/runtime_evidence/kv_ingest/boundaries/{boundary}");
            ensure_finite(&format!("{path}/first_at_ms"), stats.first_at_ms)?;
            ensure_finite(&format!("{path}/last_at_ms"), stats.last_at_ms)?;
        }
    }
    Ok(())
}

fn validate_json_finite(path: &str, value: &Value) -> Result<()> {
    match value {
        Value::Array(values) => {
            for (index, value) in values.iter().enumerate() {
                validate_json_finite(&format!("{path}/{index}"), value)?;
            }
        }
        Value::Object(values) => {
            for (key, value) in values {
                validate_json_finite(&format!("{path}/{}", escape_json_pointer(key)), value)?;
            }
        }
        Value::Number(number) => {
            if let Some(value) = number.as_f64() {
                ensure_finite(path, value)?;
            }
        }
        Value::Null | Value::Bool(_) | Value::String(_) => {}
    }
    Ok(())
}

fn escape_json_pointer(value: &str) -> String {
    value.replace('~', "~0").replace('/', "~1")
}

fn ensure_finite(path: &str, value: f64) -> Result<()> {
    anyhow::ensure!(
        value.is_finite(),
        "canonical replay rejects non-finite number at {path}"
    );
    Ok(())
}

pub fn canonicalize_json(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(canonicalize_json).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, canonicalize_json(value)))
                .collect::<BTreeMap<_, _>>()
                .into_iter()
                .collect::<Map<_, _>>(),
        ),
        scalar => scalar,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::replay::{ReplayCaptureOptions, TraceCollector};

    use super::{CanonicalReplayCoverage, CanonicalReplayRecord, canonicalize_json};

    #[test]
    fn canonicalize_json_sorts_object_keys_but_preserves_array_order() {
        assert_eq!(
            serde_json::to_string(&canonicalize_json(json!({
                "z": [{"b": 1, "a": 2}, 3],
                "a": 4,
            })))
            .unwrap(),
            r#"{"a":4,"z":[{"a":2,"b":1},3]}"#
        );
    }

    #[test]
    fn canonical_record_sorts_roots_and_injects_schema() {
        let report = TraceCollector::default().finish();
        let coverage =
            CanonicalReplayCoverage::from_report(&report, ReplayCaptureOptions::default());
        let record =
            CanonicalReplayRecord::build(&report, json!({"z": 1, "a": 2}), &coverage, Value::Null)
                .unwrap();
        assert_eq!(
            record.metadata["schema_version"],
            "dynamo.offline-replay.v1"
        );
        let line = String::from_utf8(record.into_json_line().unwrap()).unwrap();
        assert!(line.starts_with(r#"{"coverage":"#));
    }

    #[test]
    fn canonical_record_rejects_non_finite_report_values() {
        let mut report = TraceCollector::default().finish();
        report.throughput.wall_time_ms = f64::NAN;
        let coverage =
            CanonicalReplayCoverage::from_report(&report, ReplayCaptureOptions::default());
        let error =
            CanonicalReplayRecord::build(&report, json!({}), &coverage, Value::Null).unwrap_err();
        assert!(error.to_string().contains("/summary/wall_time_ms"));
    }
}
