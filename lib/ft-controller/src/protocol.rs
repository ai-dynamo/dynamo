// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire types of vLLM's fault-tolerance HTTP API (`/v1/fault_tolerance/*`,
//! vLLM 0.34.0), plus the two OpenAI routes used to wake an idle engine.
//!
//! The readers are tolerant: unknown fields are ignored, and a status string
//! this crate does not know is kept as [`EngineStatus::Unknown`], which the
//! decision core never mistakes for healthy or unhealthy.

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::decision::StoreAddr;

/// The status schema this controller understands.
pub const STATUS_SCHEMA_VERSION: u64 = 1;

/// Body of `GET {base}/v1/fault_tolerance/status`.
#[derive(Clone, Debug, Deserialize)]
pub struct StatusResponse {
    pub schema_version: Option<u64>,
    pub total_engines: Option<u64>,
    #[serde(default)]
    pub engines: Vec<EngineEntry>,
}

/// One engine of a status response. `fault_info` and `mask` are not read.
#[derive(Clone, Debug, Deserialize)]
pub struct EngineEntry {
    /// The engine index. In vLLM 0.34.0 this is the global data-parallel
    /// rank, which a scale_down never renumbers.
    pub id: Option<Value>,
    pub status: String,
    pub ft_state: Option<String>,
    pub last_ft_request_id: Option<String>,
    pub ft_error: Option<String>,
}

/// An engine's health as its front end reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EngineStatus {
    Healthy,
    Unhealthy,
    Diagnosing,
    Dead,
    /// A status this controller does not know; never treated as healthy.
    Unknown(String),
}

impl EngineStatus {
    pub fn parse(raw: &str) -> Self {
        match raw.to_ascii_lowercase().as_str() {
            "healthy" => Self::Healthy,
            "unhealthy" => Self::Unhealthy,
            "diagnosing" => Self::Diagnosing,
            "dead" => Self::Dead,
            _ => Self::Unknown(raw.to_string()),
        }
    }
}

impl fmt::Display for EngineStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => f.write_str("healthy"),
            Self::Unhealthy => f.write_str("unhealthy"),
            Self::Diagnosing => f.write_str("diagnosing"),
            Self::Dead => f.write_str("dead"),
            Self::Unknown(raw) => write!(f, "unknown status {raw:?}"),
        }
    }
}

/// The front end's view of the last recovery order it handled.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FtState {
    Recovering,
    Failed,
    Unknown(String),
}

impl FtState {
    pub fn parse(raw: &str) -> Self {
        match raw.to_ascii_lowercase().as_str() {
            "recovering" => Self::Recovering,
            "failed" => Self::Failed,
            _ => Self::Unknown(raw.to_string()),
        }
    }
}

/// What one rank's front end says about its engine.
///
/// In vLLM 0.34.0, `ft_state`, `last_ft_request_id` and `ft_error` are kept
/// by the front end only. Every status the engine pushes (for example
/// `healthy` after a successful round) replaces the front end's entry, so
/// they disappear again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineReport {
    /// The engine id the front end reports, when it is a non-negative
    /// integer (or a string holding one).
    pub id: Option<u64>,
    pub status: EngineStatus,
    pub ft_state: Option<FtState>,
    pub last_ft_request_id: Option<String>,
    pub ft_error: Option<String>,
}

impl EngineReport {
    /// A healthy engine with no recovery history.
    pub fn healthy() -> Self {
        Self::with_status(EngineStatus::Healthy)
    }

    pub fn with_status(status: EngineStatus) -> Self {
        Self {
            id: None,
            status,
            ft_state: None,
            last_ft_request_id: None,
            ft_error: None,
        }
    }

    /// Healthy, and not in the middle of a recovery.
    pub fn is_settled_healthy(&self) -> bool {
        self.status == EngineStatus::Healthy && self.ft_state != Some(FtState::Recovering)
    }
}

impl StatusResponse {
    /// The single engine this front end manages. With external load balancing
    /// (the only mode this controller supports) a front end manages exactly one.
    pub fn into_report(self) -> Result<EngineReport, String> {
        if let Some(version) = self.schema_version
            && version != STATUS_SCHEMA_VERSION
        {
            return Err(format!(
                "status schema_version {version} is not supported (expected {STATUS_SCHEMA_VERSION})"
            ));
        }
        let mut engines = self.engines;
        if engines.len() != 1 {
            return Err(format!(
                "front end reports {} engines (total_engines {:?}); expected exactly one (external load balancing)",
                engines.len(),
                self.total_engines
            ));
        }
        let Some(engine) = engines.pop() else {
            return Err("front end reports no engine".to_string());
        };
        Ok(EngineReport {
            id: engine_id(engine.id.as_ref()),
            status: EngineStatus::parse(&engine.status),
            ft_state: engine.ft_state.as_deref().map(FtState::parse),
            last_ft_request_id: engine.last_ft_request_id,
            ft_error: engine.ft_error,
        })
    }
}

/// An engine id as an integer, if it is one.
fn engine_id(id: Option<&Value>) -> Option<u64> {
    match id? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse().ok(),
        _ => None,
    }
}

/// A recovery instruction of `POST {base}/v1/fault_tolerance/apply`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Instruction {
    Retry,
    ScaleDown,
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Retry => f.write_str("retry"),
            Self::ScaleDown => f.write_str("scale_down"),
        }
    }
}

/// Body of `POST {base}/v1/fault_tolerance/apply`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ApplyRequest {
    pub instruction: Instruction,
    pub params: Value,
    pub request_id: String,
}

/// The `params` of one round. `removed` must be empty for a retry. `store`,
/// when set, carries the address the survivors rebuild the coordination
/// store at before they check in (vLLM 0.34.0 rebuilds it only in a
/// scale_down that removes the store's host, and ignores it otherwise).
/// `retry_abort` adds `{"abort": true}` to a retry; vLLM 0.34.0 ignores it.
pub fn round_params(
    instruction: Instruction,
    removed: &[u32],
    store: Option<&StoreAddr>,
    retry_abort: bool,
) -> Value {
    let mut params = Map::new();
    match instruction {
        Instruction::Retry => {
            if retry_abort {
                params.insert("abort".to_string(), Value::Bool(true));
            }
        }
        Instruction::ScaleDown => {
            params.insert("removed_dp_ranks".to_string(), json!(removed));
        }
    }
    if let Some(store) = store {
        params.insert("dp_master_ip".to_string(), json!(store.host));
        params.insert("dp_store_port".to_string(), json!(store.port));
    }
    Value::Object(params)
}

/// Body of `GET {base}/v1/models`.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelList {
    #[serde(default)]
    pub data: Vec<ModelEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ModelEntry {
    pub id: String,
}

/// The one-token completion that makes an idle engine run a step.
pub fn wake_request(model: &str) -> Value {
    json!({"model": model, "prompt": "ping", "max_tokens": 1})
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: Value) -> Result<EngineReport, String> {
        serde_json::from_value::<StatusResponse>(body)
            .map_err(|e| e.to_string())?
            .into_report()
    }

    #[test]
    fn reads_the_single_engine_of_a_front_end() {
        let report = parse(json!({
            "schema_version": 1,
            "total_engines": 1,
            "engines": [{
                "id": 3,
                "status": "unhealthy",
                "fault_info": {"type": "nixl"},
                "mask": [1, 1, 0, 1],
                "ft_state": "failed",
                "last_ft_request_id": "w-e1-r1-0000abcd",
                "ft_error": "check-in timed out",
            }],
        }))
        .unwrap();
        assert_eq!(report.id, Some(3));
        assert_eq!(report.status, EngineStatus::Unhealthy);
        assert_eq!(report.ft_state, Some(FtState::Failed));
        assert_eq!(
            report.last_ft_request_id.as_deref(),
            Some("w-e1-r1-0000abcd")
        );
        assert_eq!(report.ft_error.as_deref(), Some("check-in timed out"));
    }

    #[test]
    fn unknown_values_are_kept_not_guessed() {
        let report = parse(json!({
            "schema_version": 1,
            "engines": [{"id": 0, "status": "rebooting", "ft_state": "paused"}],
        }))
        .unwrap();
        assert_eq!(
            report.status,
            EngineStatus::Unknown("rebooting".to_string())
        );
        assert_eq!(
            report.ft_state,
            Some(FtState::Unknown("paused".to_string()))
        );
        assert!(!report.is_settled_healthy());
    }

    #[test]
    fn engine_ids_are_read_when_they_are_integers() {
        let id = |id: Value| {
            parse(json!({"schema_version": 1, "engines": [{"id": id, "status": "healthy"}]}))
                .unwrap()
                .id
        };
        assert_eq!(id(json!(0)), Some(0));
        assert_eq!(id(json!("7")), Some(7));
        assert_eq!(id(json!(-1)), None);
        assert_eq!(id(json!(1.5)), None);
        assert_eq!(id(json!("engine-1")), None);
        assert_eq!(id(Value::Null), None);
        let missing = parse(json!({"schema_version": 1, "engines": [{"status": "healthy"}]}));
        assert_eq!(missing.unwrap().id, None);
    }

    #[test]
    fn rejects_other_schemas_and_engine_counts() {
        let err = parse(json!({"schema_version": 2, "engines": [{"status": "healthy"}]}));
        assert!(err.unwrap_err().contains("schema_version 2"));
        let err = parse(json!({
            "schema_version": 1,
            "engines": [{"status": "healthy"}, {"status": "healthy"}],
        }));
        assert!(err.unwrap_err().contains("2 engines"));
        let err = parse(json!({"schema_version": 1, "engines": []}));
        assert!(err.unwrap_err().contains("0 engines"));
    }

    #[test]
    fn round_params_follow_the_instruction() {
        assert_eq!(
            round_params(Instruction::Retry, &[], None, false),
            json!({})
        );
        assert_eq!(
            round_params(Instruction::Retry, &[], None, true),
            json!({"abort": true})
        );
        assert_eq!(
            round_params(Instruction::ScaleDown, &[1, 3], None, false),
            json!({"removed_dp_ranks": [1, 3]})
        );
        let store = StoreAddr {
            host: "10.0.0.6".to_string(),
            port: 29651,
        };
        assert_eq!(
            round_params(Instruction::ScaleDown, &[0], Some(&store), false),
            json!({"removed_dp_ranks": [0], "dp_master_ip": "10.0.0.6", "dp_store_port": 29651})
        );
        assert_eq!(
            round_params(Instruction::Retry, &[], Some(&store), false),
            json!({"dp_master_ip": "10.0.0.6", "dp_store_port": 29651})
        );
    }

    #[test]
    fn instructions_use_the_wire_names() {
        let order = ApplyRequest {
            instruction: Instruction::ScaleDown,
            params: json!({"removed_dp_ranks": [2]}),
            request_id: "w-e1-r1-00000001".to_string(),
        };
        assert_eq!(
            serde_json::to_value(&order).unwrap(),
            json!({
                "instruction": "scale_down",
                "params": {"removed_dp_ranks": [2]},
                "request_id": "w-e1-r1-00000001",
            })
        );
        assert_eq!(Instruction::Retry.to_string(), "retry");
    }
}
