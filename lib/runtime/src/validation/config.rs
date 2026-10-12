// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing for `DYN_WORKER_VALIDATION_GATES`.
//!
//! Every misconfiguration is an error: a worker that cannot tell what it is
//! supposed to check must not start unchecked.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::config::environment_names::runtime::validation as env_validation;

/// How one gate is checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateCheck {
    /// Run this command in the worker container. Exit status 0 is a pass.
    Exec(Vec<String>),
}

/// One configured gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateSpec {
    pub name: String,
    pub check: GateCheck,
    /// The gate fails if it has no verdict by then.
    pub timeout: Duration,
}

/// Parsed validation configuration. Only exists when gates are configured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationConfig {
    /// Gates in name order.
    pub gates: Vec<GateSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGate {
    exec: Vec<String>,
    timeout_secs: u64,
}

impl ValidationConfig {
    /// Read the configuration from the environment. `Ok(None)` when no gates
    /// are configured, which keeps today's behaviour unchanged.
    pub fn from_env() -> Result<Option<Self>> {
        Self::parse(std::env::var(env_validation::DYN_WORKER_VALIDATION_GATES).ok())
    }

    pub fn parse(gates: Option<String>) -> Result<Option<Self>> {
        let Some(gates) = gates.filter(|value| !value.trim().is_empty()) else {
            return Ok(None);
        };
        let raw: BTreeMap<String, RawGate> = serde_json::from_str(&gates).with_context(|| {
            format!(
                "{} must be a JSON object of gate name to \
                 {{\"exec\": [...], \"timeout_secs\": N}}",
                env_validation::DYN_WORKER_VALIDATION_GATES
            )
        })?;
        if raw.is_empty() {
            bail!(
                "{} must configure at least one gate",
                env_validation::DYN_WORKER_VALIDATION_GATES
            );
        }
        let gates = raw
            .into_iter()
            .map(|(name, gate)| {
                if gate.exec.first().is_none_or(|program| program.is_empty()) {
                    bail!("validation gate {name:?}: \"exec\" must be a non-empty command");
                }
                if gate.timeout_secs == 0 {
                    bail!("validation gate {name:?}: \"timeout_secs\" must be greater than zero");
                }
                Ok(GateSpec {
                    name,
                    check: GateCheck::Exec(gate.exec),
                    timeout: Duration::from_secs(gate.timeout_secs),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Some(ValidationConfig { gates }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(gates: &str) -> Result<Option<ValidationConfig>> {
        ValidationConfig::parse(Some(gates.to_string()))
    }

    #[test]
    fn parses_gates_in_name_order() {
        assert_eq!(ValidationConfig::parse(None).unwrap(), None);
        let config = parse(
            r#"{"smoke": {"exec": ["python", "/checks/smoke.py"], "timeout_secs": 900},
                "basic": {"exec": ["true"], "timeout_secs": 60}}"#,
        )
        .unwrap()
        .unwrap();
        let names: Vec<_> = config.gates.iter().map(|gate| gate.name.as_str()).collect();
        assert_eq!(names, ["basic", "smoke"]);
        assert_eq!(
            config.gates[1].check,
            GateCheck::Exec(vec!["python".into(), "/checks/smoke.py".into()])
        );
        assert_eq!(config.gates[1].timeout, Duration::from_secs(900));
    }

    #[test]
    fn rejects_every_misconfiguration() {
        for (gates, expected) in [
            ("not json", "must be a JSON object"),
            ("{}", "at least one gate"),
            (r#"{"a": {"timeout_secs": 1}}"#, "must be a JSON object"),
            (r#"{"a": {"exec": ["true"]}}"#, "must be a JSON object"),
            (
                r#"{"a": {"exec": [], "timeout_secs": 1}}"#,
                "non-empty command",
            ),
            (
                r#"{"a": {"exec": ["true"], "timeout_secs": 0}}"#,
                "greater than zero",
            ),
            (
                r#"{"a": {"exec": ["true"], "timeout_secs": 1, "retries": 3}}"#,
                "must be a JSON object",
            ),
        ] {
            let error = format!("{:#}", parse(gates).unwrap_err());
            assert!(
                error.contains(expected),
                "{gates}: expected {expected:?} in {error:?}"
            );
        }
    }
}
