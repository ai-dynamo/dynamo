// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::replay::{ReplayError, ReplayResult, SlaThresholds};

pub const CURRENT_REPLAY_SPEC_VERSION: u32 = 1;

/// Serializable input to one replay execution.
///
/// Provider descriptors are data only. A runner resolves them to concrete
/// placement/scaling implementations before constructing [`crate::replay::Replayer`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplaySpec {
    #[serde(default = "default_spec_version")]
    pub version: u32,
    pub topology: ReplayTopology,
    #[serde(default)]
    pub engine: Value,
    #[serde(default)]
    pub adapters: ReplayAdapters,
    /// Soft virtual-time cutoff. Events at the cutoff are processed; replay
    /// stops before the first event after it and leaves in-flight requests
    /// non-terminal in the report.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sim_time_ms: Option<f64>,
    /// Optional source-side in-flight cap. Requests whose authored arrival is
    /// ready remain outside the simulated system until an earlier request is
    /// terminal, matching closed-loop and replay-concurrency workloads.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_in_flight: Option<usize>,
    /// Whether the report should retain one record per arrived request.
    ///
    /// This defaults to `true` to preserve reports produced by version-1
    /// execution specs written before this control was added.
    #[serde(
        default = "default_record_per_request",
        skip_serializing_if = "is_true"
    )]
    pub record_per_request: bool,
    /// Optional latency targets used to calculate goodput.
    #[serde(default, skip_serializing_if = "SlaThresholds::is_unset")]
    pub sla: SlaThresholds,
    pub requests: Vec<ReplayRequest>,
}

impl ReplaySpec {
    pub fn validate(&self) -> ReplayResult<()> {
        if self.version != CURRENT_REPLAY_SPEC_VERSION {
            return Err(ReplayError::InvalidSpec(format!(
                "unsupported version {}; expected {}",
                self.version, CURRENT_REPLAY_SPEC_VERSION
            )));
        }
        self.topology.validate()?;
        if let Some(max_sim_time_ms) = self.max_sim_time_ms {
            validate_time("max_sim_time_ms", max_sim_time_ms)?;
        }
        if self.max_in_flight == Some(0) {
            return Err(ReplayError::InvalidSpec(
                "max_in_flight must be positive".to_string(),
            ));
        }
        self.sla.validate()?;

        let mut ids = BTreeSet::new();
        for request in &self.requests {
            request.validate()?;
            if !ids.insert(request.id.clone()) {
                return Err(ReplayError::InvalidSpec(format!(
                    "duplicate request id {:?}",
                    request.id
                )));
            }
        }
        Ok(())
    }
}

fn default_spec_version() -> u32 {
    CURRENT_REPLAY_SPEC_VERSION
}

fn default_record_per_request() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ReplayTopology {
    Aggregated {
        workers: WorkerPoolSpec,
    },
    Disaggregated {
        prefill: WorkerPoolSpec,
        decode: WorkerPoolSpec,
        /// Transfer latency used only when the selected engine cannot derive
        /// one from KV bytes and bandwidth.
        #[serde(default)]
        handoff_latency_ms: f64,
    },
}

impl ReplayTopology {
    pub fn aggregated(workers: usize) -> Self {
        Self::Aggregated {
            workers: WorkerPoolSpec {
                initial_workers: workers,
                ..WorkerPoolSpec::default()
            },
        }
    }

    pub fn validate(&self) -> ReplayResult<()> {
        match self {
            Self::Aggregated { workers } => workers.validate("aggregated"),
            Self::Disaggregated {
                prefill,
                decode,
                handoff_latency_ms,
            } => {
                prefill.validate("prefill")?;
                decode.validate("decode")?;
                validate_time("handoff_latency_ms", *handoff_latency_ms)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkerPoolSpec {
    pub initial_workers: usize,
    #[serde(default)]
    pub startup_delay_ms: f64,
}

impl Default for WorkerPoolSpec {
    fn default() -> Self {
        Self {
            initial_workers: 1,
            startup_delay_ms: 0.0,
        }
    }
}

impl WorkerPoolSpec {
    fn validate(&self, name: &str) -> ReplayResult<()> {
        if self.initial_workers == 0 {
            return Err(ReplayError::InvalidSpec(format!(
                "{name} pool must start with at least one worker"
            )));
        }
        validate_time(&format!("{name} startup_delay_ms"), self.startup_delay_ms)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub id: String,
    pub arrival_time_ms: f64,
    pub input_tokens: usize,
    /// Materialized prompt content for KV-aware replay. Length-only engine
    /// replay may omit it; adapters must never invent tokens when it is needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_ids: Option<Vec<u32>>,
    pub output_tokens: usize,
    /// Optional exact generated-token plan. When present, its length controls
    /// native generation while `output_tokens` remains the authored maximum.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_ids: Option<Vec<u32>>,
    /// Optional attention-DP rank selected by the workload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Zero-based turn index within `session_id`, when the workload has one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_index: Option<usize>,
    /// Provider-neutral request metadata resolved by the selected runner.
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl ReplayRequest {
    pub fn validate(&self) -> ReplayResult<()> {
        if self.id.is_empty() {
            return Err(ReplayError::InvalidSpec(
                "request id must not be empty".to_string(),
            ));
        }
        validate_time("request arrival_time_ms", self.arrival_time_ms)?;
        if let Some(input_token_ids) = &self.input_token_ids
            && input_token_ids.len() != self.input_tokens
        {
            return Err(ReplayError::InvalidSpec(format!(
                "request {:?} declares {} input tokens but materializes {} token IDs",
                self.id,
                self.input_tokens,
                input_token_ids.len()
            )));
        }
        self.routing_metadata()?;
        Ok(())
    }

    /// Resolve the small routing subset of provider-neutral metadata while
    /// leaving the authored metadata value intact for reporting and other
    /// adapters.
    pub fn routing_metadata(&self) -> ReplayResult<ReplayRoutingMetadata> {
        if self.metadata.is_null() {
            return Ok(ReplayRoutingMetadata::default());
        }
        serde_json::from_value(self.metadata.clone()).map_err(|error| {
            ReplayError::InvalidSpec(format!(
                "request {:?} has invalid routing metadata: {error}",
                self.id
            ))
        })
    }
}

/// Provider-neutral routing controls recognized during ReplaySpec lowering.
/// Unknown metadata keys remain available in [`ReplayRequest::metadata`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayRoutingMetadata {
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub strict_priority: u32,
    #[serde(default)]
    pub policy_class: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayAdapters {
    #[serde(default = "ProviderSpec::round_robin")]
    pub placement: ProviderSpec,
    #[serde(default = "ProviderSpec::no_scaling")]
    pub scaling: ProviderSpec,
}

impl Default for ReplayAdapters {
    fn default() -> Self {
        Self {
            placement: ProviderSpec::round_robin(),
            scaling: ProviderSpec::no_scaling(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderSpec {
    pub provider: String,
    #[serde(default)]
    pub config: Value,
}

impl ProviderSpec {
    pub fn round_robin() -> Self {
        Self {
            provider: "round_robin".to_string(),
            config: Value::Null,
        }
    }

    pub fn no_scaling() -> Self {
        Self {
            provider: "none".to_string(),
            config: Value::Null,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerStage {
    Aggregated,
    Prefill,
    Decode,
}

fn validate_time(name: &str, value: f64) -> ReplayResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(ReplayError::InvalidSpec(format!(
            "{name} must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}
