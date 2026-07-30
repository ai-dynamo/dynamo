// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::common::protocols::MockEngineArgs;
use crate::replay::{
    PerRequestRecord, ReplayArgsMode, TraceDistributionStats, TraceSimulationReport,
};

use super::evidence::{KvIngestEvidence, PressureEvidence};

pub const CANONICAL_SCHEMA_VERSION: &str = "dynamo.offline-replay.v1";
pub const CANONICAL_RESULT_EXCLUSIONS: [&str; 4] = [
    "/summary/wall_time_ms",
    "/summary/processed_tokens_per_s",
    "/summary/processed_output_tokens_per_s",
    "/planner/html_report_path",
];

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalReplayTopology {
    Aggregated,
    Disaggregated,
}

pub fn canonical_topology(mode: ReplayArgsMode) -> CanonicalReplayTopology {
    match mode {
        ReplayArgsMode::Aggregated => CanonicalReplayTopology::Aggregated,
        ReplayArgsMode::Disagg => CanonicalReplayTopology::Disaggregated,
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalAicIdentity {
    pub backend: Option<String>,
    pub system: Option<String>,
    pub backend_version: Option<String>,
    pub tp_size: Option<usize>,
    pub model: Option<String>,
    pub moe_tp_size: Option<usize>,
    pub moe_ep_size: Option<usize>,
    pub attention_dp_size: Option<usize>,
    pub gemm_dtype: Option<String>,
    pub moe_dtype: Option<String>,
    pub fmha_dtype: Option<String>,
    pub kv_cache_dtype: Option<String>,
    pub comm_dtype: Option<String>,
    pub nextn: Option<usize>,
    pub nextn_accept_rates: Option<String>,
}

impl CanonicalAicIdentity {
    fn from_engine_args(args: &MockEngineArgs) -> Self {
        let enabled = args.aic_backend.is_some();
        Self {
            backend: args.aic_backend.clone(),
            system: enabled.then(|| {
                args.aic_system
                    .clone()
                    .unwrap_or_else(|| "h200_sxm".to_string())
            }),
            backend_version: args.aic_backend_version.clone(),
            tp_size: enabled.then(|| args.aic_tp_size.unwrap_or(1)),
            model: args.aic_model_path.clone(),
            moe_tp_size: args.aic_moe_tp_size,
            moe_ep_size: args.aic_moe_ep_size,
            attention_dp_size: args.aic_attention_dp_size,
            gemm_dtype: args.aic_gemm_dtype.clone(),
            moe_dtype: args.aic_moe_dtype.clone(),
            fmha_dtype: args.aic_fmha_dtype.clone(),
            kv_cache_dtype: args.aic_kv_cache_dtype.clone(),
            comm_dtype: args.aic_comm_dtype.clone(),
            nextn: args.aic_nextn,
            nextn_accept_rates: args.aic_nextn_accept_rates.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalWorkloadMetadata {
    Trace {
        format: String,
        block_size: Option<usize>,
        digest: String,
    },
    Synthetic {
        block_size: usize,
        digest: String,
        spec: CanonicalSyntheticSpec,
    },
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalSyntheticSpec {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub request_count: usize,
    pub replay_concurrency: Option<isize>,
    pub request_rate: Option<f64>,
    pub arrival_interval_ms: Option<f64>,
    pub arrival_seed: u64,
    pub turns_per_session: usize,
    pub shared_prefix_ratio: f64,
    pub num_prefix_groups: usize,
    pub inter_turn_delay_ms: f64,
    pub output_seed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalExecutionMetadata {
    pub topology: CanonicalReplayTopology,
    pub num_workers: usize,
    pub num_prefill_workers: usize,
    pub num_decode_workers: usize,
    pub replay_concurrency: Option<isize>,
    pub arrival_speedup_ratio: f64,
    pub max_sim_time_ms: Option<f64>,
    pub aic_prefill_load_estimator: Option<CanonicalAicIdentity>,
    pub aic_performance_model_implementation: Option<CanonicalAicImplementation>,
    pub aic_prefill_load_estimator_implementation: Option<CanonicalAicImplementation>,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalAicImplementation {
    CompiledRust,
    PythonCompatibility,
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalEngineConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aggregated: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode: Option<Value>,
}

impl CanonicalEngineConfig {
    pub fn aggregated(args: &MockEngineArgs) -> Result<Self> {
        Ok(Self {
            aggregated: Some(canonical_engine_pool_metadata(args)?),
            prefill: None,
            decode: None,
        })
    }

    pub fn disaggregated(prefill: &MockEngineArgs, decode: &MockEngineArgs) -> Result<Self> {
        Ok(Self {
            aggregated: None,
            prefill: Some(canonical_engine_pool_metadata(prefill)?),
            decode: Some(canonical_engine_pool_metadata(decode)?),
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalSlaMetadata {
    pub ttft_ms: Option<f64>,
    pub itl_ms: Option<f64>,
    pub e2e_ms: Option<f64>,
}

pub fn canonical_engine_pool_metadata(args: &MockEngineArgs) -> Result<Value> {
    validate_mock_engine_args_finite(args)?;
    anyhow::ensure!(
        args.planner_profile_data.is_none(),
        "canonical replay does not support planner_profile_data"
    );
    anyhow::ensure!(
        args.response_replay_trace_path.is_none(),
        "canonical replay does not support response_replay_trace_path"
    );
    anyhow::ensure!(
        args.aic_backend.is_none() || args.aic_backend_version.is_some(),
        "canonical AIC replay requires a resolved backend version"
    );
    let mut metadata = serde_json::to_value(args)?;
    let metadata = metadata
        .as_object_mut()
        .context("serialized replay engine configuration must be an object")?;
    metadata.insert(
        "performance_model".to_string(),
        json!({
            "kind": if args.aic_backend.is_some() {
                "aic_callback"
            } else if args.planner_profile_data.is_some() {
                "planner_profile"
            } else {
                "builtin_polynomial"
            },
            "aic": CanonicalAicIdentity::from_engine_args(args),
        }),
    );
    Ok(Value::Object(metadata.clone()))
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalDeterminismMetadata {
    pub request_ids: String,
    pub selection: String,
    pub seed: u64,
    pub candidate_order: [String; 2],
}

impl CanonicalDeterminismMetadata {
    pub fn canonical_v1() -> Self {
        Self {
            request_ids: "ordinal_u128_v1".to_string(),
            selection: "default_worker_selector_seeded_v1".to_string(),
            seed: 0xd1a0_5eed,
            candidate_order: ["worker_id".to_string(), "dp_rank".to_string()],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalSemanticFeatures {
    pub canonical_replay: bool,
    pub mocker_kvbm_offload: bool,
    pub aic_forward_pass: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalReplayMetadata {
    pub replay_bench: bool,
    pub byte_identity_scope: String,
    pub workload: CanonicalWorkloadMetadata,
    pub execution: CanonicalExecutionMetadata,
    pub engine_config: CanonicalEngineConfig,
    pub router: super::extensions::kv_router::CanonicalRouterMetadata,
    pub sla: CanonicalSlaMetadata,
    pub determinism: CanonicalDeterminismMetadata,
    pub semantic_features: CanonicalSemanticFeatures,
}

#[derive(Clone, Debug, Serialize)]
pub struct CanonicalReplayCoverage {
    pub capture_per_request: bool,
    pub capture_planner_details: bool,
    pub capture_canonical_evidence: bool,
    pub per_request_records: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pressure: Option<PressureEvidence>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_ingest: Option<KvIngestEvidence>,
}

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
        report: &TraceSimulationReport,
        metadata: &CanonicalReplayMetadata,
        coverage: &CanonicalReplayCoverage,
        mut planner: Value,
    ) -> Result<Self> {
        validate_report_finite(report)?;
        validate_metadata_finite(metadata)?;
        let mut metadata = serde_json::to_value(metadata)?;
        {
            let metadata = metadata
                .as_object_mut()
                .context("canonical replay metadata must be an object")?;
            metadata.insert(
                "schema_version".to_string(),
                Value::String(CANONICAL_SCHEMA_VERSION.to_string()),
            );
            metadata.insert(
                "result_exclusions".to_string(),
                serde_json::to_value(CANONICAL_RESULT_EXCLUSIONS)?,
            );
            if let Some(planner_metadata) =
                planner.get("metadata").and_then(Value::as_object).cloned()
            {
                metadata.insert("planner".to_string(), Value::Object(planner_metadata));
            }
        }

        let mut summary = serde_json::to_value(report)?;
        {
            let summary = summary
                .as_object_mut()
                .context("serialized replay summary must be an object")?;
            summary.remove("wall_time_ms");
            summary.remove("processed_tokens_per_s");
            summary.remove("processed_output_tokens_per_s");
        }

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

        if let Some(planner) = planner.as_object_mut() {
            planner.remove("html_report_path");
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

fn validate_metadata_finite(metadata: &CanonicalReplayMetadata) -> Result<()> {
    if let CanonicalWorkloadMetadata::Synthetic { spec, .. } = &metadata.workload {
        for (path, value) in [
            ("/metadata/workload/spec/request_rate", spec.request_rate),
            (
                "/metadata/workload/spec/arrival_interval_ms",
                spec.arrival_interval_ms,
            ),
            (
                "/metadata/workload/spec/shared_prefix_ratio",
                Some(spec.shared_prefix_ratio),
            ),
            (
                "/metadata/workload/spec/inter_turn_delay_ms",
                Some(spec.inter_turn_delay_ms),
            ),
        ] {
            if let Some(value) = value {
                ensure_finite(path, value)?;
            }
        }
    }
    ensure_finite(
        "/metadata/execution/arrival_speedup_ratio",
        metadata.execution.arrival_speedup_ratio,
    )?;
    if let Some(value) = metadata.execution.max_sim_time_ms {
        ensure_finite("/metadata/execution/max_sim_time_ms", value)?;
    }
    for (path, value) in [
        ("/metadata/sla/ttft_ms", metadata.sla.ttft_ms),
        ("/metadata/sla/itl_ms", metadata.sla.itl_ms),
        ("/metadata/sla/e2e_ms", metadata.sla.e2e_ms),
    ] {
        if let Some(value) = value {
            ensure_finite(path, value)?;
        }
    }
    if let Some(config) = &metadata.router.config {
        super::extensions::kv_router::validate_canonical_router_config(config)?;
    }
    Ok(())
}

fn validate_mock_engine_args_finite(args: &MockEngineArgs) -> Result<()> {
    for (field, value) in [
        ("speedup_ratio", Some(args.speedup_ratio)),
        ("decode_speedup_ratio", Some(args.decode_speedup_ratio)),
        ("startup_time", args.startup_time),
        ("gpu_memory_utilization", args.gpu_memory_utilization),
        ("mem_fraction_static", args.mem_fraction_static),
        ("free_gpu_memory_fraction", args.free_gpu_memory_fraction),
        ("kv_transfer_bandwidth", args.kv_transfer_bandwidth),
        ("bandwidth_g1_to_g2_gbps", args.bandwidth_g1_to_g2_gbps),
        ("bandwidth_g2_to_g1_gbps", args.bandwidth_g2_to_g1_gbps),
        ("bandwidth_g2_to_g3_gbps", args.bandwidth_g2_to_g3_gbps),
        ("bandwidth_g3_to_g2_gbps", args.bandwidth_g3_to_g2_gbps),
        ("bandwidth_g2_to_g4_gbps", args.bandwidth_g2_to_g4_gbps),
        ("bandwidth_g4_to_g2_gbps", args.bandwidth_g4_to_g2_gbps),
    ] {
        if let Some(value) = value {
            ensure_finite(&format!("/metadata/engine_config/{field}"), value)?;
        }
    }
    if let Some(reasoning) = &args.reasoning {
        ensure_finite(
            "/metadata/engine_config/reasoning/thinking_ratio",
            reasoning.thinking_ratio,
        )?;
    }
    if let Some(value) = args
        .sglang
        .as_ref()
        .and_then(|sglang| sglang.schedule_conservativeness)
    {
        ensure_finite(
            "/metadata/engine_config/sglang/schedule_conservativeness",
            value,
        )?;
    }
    Ok(())
}

fn validate_report_finite(report: &TraceSimulationReport) -> Result<()> {
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
    Ok(())
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
    use serde_json::json;

    use crate::common::protocols::MockEngineArgs;
    use crate::replay::{ReplayArgsMode, ReplayRouterMode, TraceCollector};

    use super::{
        CanonicalDeterminismMetadata, CanonicalEngineConfig, CanonicalExecutionMetadata,
        CanonicalReplayCoverage, CanonicalReplayMetadata, CanonicalReplayRecord,
        CanonicalSemanticFeatures, CanonicalSlaMetadata, CanonicalWorkloadMetadata,
        canonical_engine_pool_metadata, canonical_topology, canonicalize_json,
    };
    use crate::replay::offline::extensions::kv_router::{
        ReplayKvRouterConfig, canonical_router_metadata,
    };

    fn metadata() -> CanonicalReplayMetadata {
        CanonicalReplayMetadata {
            replay_bench: true,
            byte_identity_scope: "same_target_toolchain_semantic_features".to_string(),
            workload: CanonicalWorkloadMetadata::Trace {
                format: "mooncake".to_string(),
                block_size: Some(512),
                digest: "digest".to_string(),
            },
            execution: CanonicalExecutionMetadata {
                topology: canonical_topology(ReplayArgsMode::Aggregated),
                num_workers: 1,
                num_prefill_workers: 1,
                num_decode_workers: 1,
                replay_concurrency: None,
                arrival_speedup_ratio: 1.0,
                max_sim_time_ms: None,
                aic_prefill_load_estimator: None,
                aic_performance_model_implementation: None,
                aic_prefill_load_estimator_implementation: None,
            },
            engine_config: CanonicalEngineConfig {
                aggregated: None,
                prefill: None,
                decode: None,
            },
            router: canonical_router_metadata(ReplayRouterMode::RoundRobin, None).unwrap(),
            sla: CanonicalSlaMetadata {
                ttft_ms: None,
                itl_ms: None,
                e2e_ms: None,
            },
            determinism: CanonicalDeterminismMetadata::canonical_v1(),
            semantic_features: CanonicalSemanticFeatures {
                canonical_replay: true,
                mocker_kvbm_offload: false,
                aic_forward_pass: false,
            },
        }
    }

    fn coverage() -> CanonicalReplayCoverage {
        CanonicalReplayCoverage {
            capture_per_request: true,
            capture_planner_details: false,
            capture_canonical_evidence: true,
            per_request_records: 0,
            pressure: None,
            kv_ingest: None,
        }
    }

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
    fn canonical_engine_identity_rejects_path_dependent_behavior_inputs() {
        let mut args = MockEngineArgs {
            planner_profile_data: Some("profile.npz".into()),
            ..Default::default()
        };
        assert!(
            canonical_engine_pool_metadata(&args)
                .unwrap_err()
                .to_string()
                .contains("planner_profile_data")
        );

        args.planner_profile_data = None;
        args.response_replay_trace_path = Some("responses.jsonl".into());
        assert!(
            canonical_engine_pool_metadata(&args)
                .unwrap_err()
                .to_string()
                .contains("response_replay_trace_path")
        );
    }

    #[test]
    fn canonical_engine_identity_uses_effective_aic_defaults() {
        let unresolved = MockEngineArgs {
            aic_backend: Some("vllm".to_string()),
            aic_model_path: Some("model".to_string()),
            ..Default::default()
        };
        assert!(
            canonical_engine_pool_metadata(&unresolved)
                .unwrap_err()
                .to_string()
                .contains("resolved backend version")
        );
        let args = MockEngineArgs {
            aic_backend_version: Some("0.19.0".to_string()),
            ..unresolved
        };
        let metadata = canonical_engine_pool_metadata(&args).unwrap();
        let aic = &metadata["performance_model"]["aic"];
        assert_eq!(aic["system"], "h200_sxm");
        assert_eq!(aic["tp_size"], 1);
        assert_eq!(aic["backend_version"], "0.19.0");
    }

    #[test]
    fn canonical_engine_identity_rejects_non_finite_configuration() {
        let args = MockEngineArgs {
            speedup_ratio: f64::INFINITY,
            ..Default::default()
        };

        let error = canonical_engine_pool_metadata(&args).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("/metadata/engine_config/speedup_ratio")
        );
    }

    #[test]
    fn canonical_router_identity_rejects_non_finite_configuration() {
        let config = ReplayKvRouterConfig {
            router_ttl_secs: f64::INFINITY,
            ..Default::default()
        };

        let error =
            canonical_router_metadata(ReplayRouterMode::KvRouter, Some(&config)).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("/metadata/router/config/router_ttl_secs")
        );
    }

    #[test]
    fn canonical_record_rejects_non_finite_report_values() {
        let mut report = TraceCollector::default().finish();
        report.throughput.wall_time_ms = f64::NAN;
        let metadata = metadata();
        let coverage = coverage();

        let error =
            CanonicalReplayRecord::build(&report, &metadata, &coverage, serde_json::Value::Null)
                .unwrap_err();

        assert!(error.to_string().contains("/summary/wall_time_ms"));
    }

    #[test]
    fn canonical_record_rejects_non_finite_execution_metadata() {
        let report = TraceCollector::default().finish();
        let mut metadata = metadata();
        metadata.execution.arrival_speedup_ratio = f64::NAN;
        let coverage = coverage();

        let error =
            CanonicalReplayRecord::build(&report, &metadata, &coverage, serde_json::Value::Null)
                .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("/metadata/execution/arrival_speedup_ratio")
        );
    }

    #[test]
    fn canonical_record_sorts_root_keys() {
        let metadata = metadata();
        let coverage = coverage();
        let record = CanonicalReplayRecord::build(
            &TraceCollector::default().finish(),
            &metadata,
            &coverage,
            serde_json::Value::Null,
        )
        .unwrap();
        let line = String::from_utf8(record.into_json_line().unwrap()).unwrap();
        assert!(line.starts_with(r#"{"coverage":"#));
    }
}
