// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Versioned semantic configuration shared by the frontend and live replay.

use std::sync::OnceLock;

use anyhow::{Context, ensure};
use dynamo_kv_router::config::KvRouterConfig;
use serde::{Deserialize, Serialize};

use crate::{entrypoint::RouterConfig, worker_type::WorkerType};

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RetuneContract {
    version: u32,
    pub config_hash: String,
    pub frontend_component_name: String,
    pub admin_port: u16,
    pub trace_storage: TraceStorage,
    pub router: RouterProfile,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TraceStorage {
    pub claim_name: String,
    pub mount_path: String,
    pub segments_dir: String,
    pub windows_dir: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RouterProfile {
    pub profile: String,
    pub overlap_score_credit: f64,
    pub prefill_load_scale: f64,
    pub temperature: f64,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RetuneAcknowledgment<'a> {
    pub config_hash: &'a str,
}

impl RetuneContract {
    pub fn from_json(value: &str) -> anyhow::Result<Self> {
        let contract: Self = serde_json::from_str(value).context("invalid Retune contract")?;
        ensure!(contract.version == 1, "unsupported Retune contract version");
        ensure!(
            contract.config_hash.len() == 64
                && contract
                    .config_hash
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit()),
            "Retune configHash must be a SHA-256 digest"
        );
        ensure!(
            !contract.frontend_component_name.is_empty(),
            "Retune frontend component is empty"
        );
        ensure!(contract.admin_port != 0, "Retune admin port is zero");
        let mount = std::path::Path::new(&contract.trace_storage.mount_path);
        ensure!(
            mount.is_absolute()
                && !mount
                    .components()
                    .any(|component| component == std::path::Component::ParentDir)
                && mount
                    .components()
                    .collect::<std::path::PathBuf>()
                    .as_os_str()
                    == mount.as_os_str(),
            "Retune mountPath must be a normalized absolute path"
        );
        ensure!(
            !contract.trace_storage.claim_name.is_empty()
                && contract.trace_storage.segments_dir == "segments"
                && contract.trace_storage.windows_dir == "windows",
            "unsupported Retune trace layout"
        );
        contract.router.config()?;
        Ok(contract)
    }

    pub fn validate_frontend(&self, config: &RouterConfig) -> anyhow::Result<()> {
        ensure!(
            config.router_mode.is_kv_routing(),
            "Retune requires KV routing"
        );
        ensure!(
            config.session_affinity_ttl_secs.is_none()
                && !config.load_threshold_config.is_configured(),
            "Retune kv-v1 does not support session affinity or busy thresholds"
        );
        self.router.validate(&config.kv_router_config, None)
    }

    pub fn validate_capture(
        &self,
        policy: &crate::request_trace::config::RequestTracePolicy,
    ) -> anyhow::Result<()> {
        use crate::request_trace::config::{RequestTraceFileFormat, RequestTraceSinkKind};
        let segments = std::path::Path::new(&self.trace_storage.mount_path)
            .join(&self.trace_storage.segments_dir);
        ensure!(
            policy.enabled
                && policy.emit_request_end_records()
                && policy.sinks.contains(&RequestTraceSinkKind::File)
                && policy.file_format == RequestTraceFileFormat::JsonlGz
                && policy.file_path.as_ref().is_some_and(|path| {
                    let path = std::path::Path::new(path);
                    path.starts_with(&segments)
                        && path != segments
                        && !path
                            .components()
                            .any(|component| component == std::path::Component::ParentDir)
                }),
            "Retune requires request_end capture to gzip JSONL files under {}",
            segments.display()
        );
        Ok(())
    }

    pub fn validate_admin(
        &self,
        enabled: bool,
        address: std::net::SocketAddr,
        tls: bool,
    ) -> anyhow::Result<()> {
        ensure!(
            enabled && address.port() == self.admin_port && !address.ip().is_loopback() && !tls,
            "Retune requires its admin listener on a reachable Pod address, the contracted port and without TLS"
        );
        Ok(())
    }
}

impl RouterProfile {
    pub fn config(&self) -> anyhow::Result<KvRouterConfig> {
        profile_config(
            &self.profile,
            self.overlap_score_credit,
            self.prefill_load_scale,
            self.temperature,
        )
    }

    pub fn validate(
        &self,
        actual: &KvRouterConfig,
        role: Option<WorkerType>,
    ) -> anyhow::Result<()> {
        let mut actual = actual.clone();
        actual.apply_policy_config().map_err(anyhow::Error::msg)?;
        if let Some(policy) = actual.policy_config_cache.get() {
            ensure!(
                !policy.has_routing_profiles() && policy.request_classifier().is_none(),
                "Retune kv-v1 does not support custom queue profiles or request classifiers"
            );
        }
        for role in role.into_iter().chain(
            role.is_none()
                .then_some([
                    WorkerType::Aggregated,
                    WorkerType::Prefill,
                    WorkerType::Decode,
                ])
                .into_iter()
                .flatten(),
        ) {
            ensure!(
                actual
                    .selected_worker_selection_policy_instance_for(role)?
                    .is_none(),
                "Retune kv-v1 requires the builtin default worker-selection policy"
            );
        }
        actual.router_policy_config = None;
        actual.policy_config_cache = OnceLock::new();

        // Process-local fields are omitted by KvRouterConfig's wire serialization.
        ensure!(
            actual.router_approximate_cache_policy == "ttl".parse().map_err(anyhow::Error::msg)?
                && actual
                    .router_prefill_policy
                    .as_deref()
                    .is_none_or(|value| value == "default")
                && actual
                    .router_decode_policy
                    .as_deref()
                    .is_none_or(|value| value == "default"),
            "Retune kv-v1 conflicts with process-local router settings"
        );
        actual.router_prefill_policy = None;
        actual.router_decode_policy = None;
        let mut expected = self.config()?;
        if role == Some(WorkerType::Prefill) {
            expected.router_track_active_blocks = false;
            expected.router_track_output_blocks = false;
        }
        ensure!(
            serde_json::to_value(&actual)? == serde_json::to_value(&expected)?,
            "router configuration conflicts with the Retune kv-v1 profile"
        );
        Ok(())
    }
}

/// Immutable v1 settings. New KvRouterConfig fields must make an explicit versioning decision.
pub fn profile_config(
    profile: &str,
    overlap_score_credit: f64,
    prefill_load_scale: f64,
    temperature: f64,
) -> anyhow::Result<KvRouterConfig> {
    ensure!(
        profile == "kv-v1",
        "unsupported Retune router profile {profile}"
    );
    let config = KvRouterConfig {
        overlap_score_credit,
        overlap_score_credit_decay: 0.0,
        prefill_load_scale,
        decode_active_request_weight: 0.0,
        host_cache_hit_weight: 0.75,
        disk_cache_hit_weight: 0.25,
        router_temperature: temperature,
        use_kv_events: true,
        router_replica_sync: true,
        router_track_active_blocks: true,
        router_track_output_blocks: false,
        router_assume_kv_reuse: true,
        router_track_prefill_tokens: true,
        router_tracking_hash: "public-xxh3-v1".parse().map_err(anyhow::Error::msg)?,
        router_tracking_key_file: None,
        router_tracking_key_id: None,
        router_prefill_load_model: "none".parse().map_err(anyhow::Error::msg)?,
        router_ttl_secs: 120.0,
        router_approximate_cache_policy: "ttl".parse().map_err(anyhow::Error::msg)?,
        router_queue_threshold: None,
        router_policy_config: None,
        router_prefill_policy: None,
        router_decode_policy: None,
        policy_model_name: None,
        policy_config_cache: OnceLock::new(),
        router_event_threads: 4,
        skip_initial_worker_wait: false,
        router_queue_policy: "fcfs".parse().map_err(anyhow::Error::msg)?,
        use_remote_indexer: false,
        serve_indexer: false,
        enable_session_prefix_index: false,
        shared_cache_multiplier: None,
        shared_cache_type: "none".parse().map_err(anyhow::Error::msg)?,
        router_predicted_ttl_secs: None,
        conditional_disagg_enabled: false,
        conditional_disagg_policy: "isl_bounding".parse().map_err(anyhow::Error::msg)?,
        conditional_disagg_eff_isl_threshold: 2048,
        conditional_disagg_eff_isl_ratio_threshold: 0.7,
        conditional_disagg_prefill_busy_threshold: None,
        conditional_disagg_decode_busy_threshold: None,
    };
    config.validate().map_err(anyhow::Error::msg)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incompatible_capture_and_admin_configuration_is_rejected() {
        use crate::request_trace::config::{
            RequestTraceFileFormat, RequestTraceRecordKind, RequestTraceSinkKind,
        };
        let contract = RetuneContract {
            version: 1,
            config_hash: "a".repeat(64),
            frontend_component_name: "gateway".into(),
            admin_port: 8002,
            trace_storage: TraceStorage {
                claim_name: "traces".into(),
                mount_path: "/trace".into(),
                segments_dir: "segments".into(),
                windows_dir: "windows".into(),
            },
            router: RouterProfile {
                profile: "kv-v1".into(),
                overlap_score_credit: 1.0,
                prefill_load_scale: 1.0,
                temperature: 0.0,
            },
        };
        let mut policy = crate::request_trace::config::RequestTracePolicy {
            enabled: true,
            records: vec![RequestTraceRecordKind::RequestEnd],
            sinks: vec![RequestTraceSinkKind::File],
            file_format: RequestTraceFileFormat::JsonlGz,
            file_path: Some("/trace/segments/deployment/pod/req".into()),
            capacity: 1,
            file_buffer_bytes: 1024,
            file_flush_interval_ms: 1000,
            file_roll_bytes: 1024,
            file_roll_lines: None,
            nats_subject: String::new(),
            otel_max_payload_bytes: 1024,
            http_header_capture_list: Vec::new(),
            tool_events_zmq_endpoint: None,
            tool_events_zmq_topic: None,
            s3_bucket: None,
            s3_region: None,
            s3_prefix: None,
            s3_roll_uncompressed_bytes: 1024,
            s3_flush_interval_ms: 1000,
            router_selection_top_k: 0,
        };
        contract.validate_capture(&policy).unwrap();
        policy.file_format = RequestTraceFileFormat::Jsonl;
        assert!(contract.validate_capture(&policy).is_err());
        policy.file_format = RequestTraceFileFormat::JsonlGz;
        policy.file_path = Some("/trace/segments/../wrong/req".into());
        assert!(contract.validate_capture(&policy).is_err());
        contract
            .validate_admin(true, "0.0.0.0:8002".parse().unwrap(), false)
            .unwrap();
        assert!(
            contract
                .validate_admin(true, "127.0.0.1:8002".parse().unwrap(), false)
                .is_err()
        );
        assert!(
            contract
                .validate_admin(true, "0.0.0.0:8003".parse().unwrap(), false)
                .is_err()
        );
        assert!(
            contract
                .validate_admin(true, "0.0.0.0:8002".parse().unwrap(), true)
                .is_err()
        );
    }

    #[test]
    fn kv_v1_conformance() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/data/retune-kv-v1.json")).unwrap();
        let config = profile_config("kv-v1", 1.0, 1.0, 0.0).unwrap();
        assert_eq!(
            serde_json::to_value(&config).unwrap(),
            fixture["router_config"]
        );
        for role in [
            WorkerType::Aggregated,
            WorkerType::Prefill,
            WorkerType::Decode,
        ] {
            let parameters = dynamo_router_policies::startup_policy_parameters(
                &dynamo_worker_selection::host::SelectionConfig::from(&config),
                role,
            );
            assert_eq!(
                serde_json::json!(parameters),
                fixture["policy_parameters"][role.as_str()]
            );
        }
        assert_eq!(config.router_approximate_cache_policy.to_string(), "ttl");
        assert!(config.router_prefill_policy.is_none() && config.router_decode_policy.is_none());
    }

    #[test]
    fn profile_versions_and_values_are_explicit() {
        assert!(profile_config("future", 1.0, 1.0, 0.0).is_err());
        assert!(profile_config("kv-v1", f64::NAN, 1.0, 0.0).is_err());
        let profile = RouterProfile {
            profile: "kv-v1".into(),
            overlap_score_credit: 1.0,
            prefill_load_scale: 1.0,
            temperature: 0.0,
        };
        let mut config = profile.config().unwrap();
        profile.validate(&config, None).unwrap();
        config.router_replica_sync = false;
        assert!(profile.validate(&config, None).is_err());
        config.router_replica_sync = true;
        config.router_track_active_blocks = false;
        profile
            .validate(&config, Some(WorkerType::Prefill))
            .unwrap();
        assert!(
            profile
                .validate(&config, Some(WorkerType::Aggregated))
                .is_err()
        );
    }
}
