// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing for `DYN_WORKER_VALIDATION_GATES` and `DYN_WORKER_VALIDATION_HOLD`.
//!
//! Every misconfiguration is an error: a worker that cannot tell what it is
//! supposed to check must not start unchecked.

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

use crate::config::environment_names::runtime::validation as env_validation;

/// What a held worker keeps back until every gate passes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HoldMode {
    /// Hold the model card's discovery write as well as `/health`. No card
    /// means no routing under any router.
    #[default]
    Card,
    /// Publish the card at once and hold only `/health`. The deployment's own
    /// routing must keep client traffic off the worker.
    Health,
}

/// How one gate is checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GateCheck {
    /// Run this command in the worker container. Exit status 0 is a pass.
    Exec(Vec<String>),
    /// POST to a check runner in the same pod, such as a sidecar with its
    /// own image. A `200` is a pass; any other status is a fail.
    Url(reqwest::Url),
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
    pub hold: HoldMode,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawGate {
    exec: Option<Vec<String>>,
    url: Option<String>,
    timeout_secs: u64,
}

impl ValidationConfig {
    /// Read the configuration from the environment. `Ok(None)` when no gates
    /// are configured, which keeps today's behaviour unchanged.
    pub fn from_env() -> Result<Option<Self>> {
        Self::parse(
            std::env::var(env_validation::DYN_WORKER_VALIDATION_GATES).ok(),
            std::env::var(env_validation::DYN_WORKER_VALIDATION_HOLD).ok(),
        )
    }

    pub fn parse(gates: Option<String>, hold: Option<String>) -> Result<Option<Self>> {
        let gates = gates.filter(|value| !value.trim().is_empty());
        let hold = match (hold.as_deref().map(str::trim), &gates) {
            (None | Some(""), _) => HoldMode::Card,
            (Some(_), None) => bail!(
                "{} is set but {} is not",
                env_validation::DYN_WORKER_VALIDATION_HOLD,
                env_validation::DYN_WORKER_VALIDATION_GATES
            ),
            (Some("card"), _) => HoldMode::Card,
            (Some("health"), _) => HoldMode::Health,
            (Some(other), _) => bail!(
                "{} must be \"card\" or \"health\", got {other:?}",
                env_validation::DYN_WORKER_VALIDATION_HOLD
            ),
        };
        let Some(gates) = gates else {
            return Ok(None);
        };
        let raw: BTreeMap<String, RawGate> = serde_json::from_str(&gates).with_context(|| {
            format!(
                "{} must be a JSON object of gate name to \
                 {{\"exec\": [...] or \"url\": \"...\", \"timeout_secs\": N}}",
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
                let check = match (gate.exec, gate.url) {
                    (Some(argv), None)
                        if argv.first().is_some_and(|program| !program.is_empty()) =>
                    {
                        GateCheck::Exec(argv)
                    }
                    (None, Some(url)) => GateCheck::Url(
                        loopback_url(&url).with_context(|| format!("validation gate {name:?}"))?,
                    ),
                    _ => bail!(
                        "validation gate {name:?}: set exactly one of a non-empty \"exec\" \
                         command or a \"url\""
                    ),
                };
                if gate.timeout_secs == 0 {
                    bail!("validation gate {name:?}: \"timeout_secs\" must be greater than zero");
                }
                Ok(GateSpec {
                    name,
                    check,
                    timeout: Duration::from_secs(gate.timeout_secs),
                })
            })
            .collect::<Result<_>>()?;
        Ok(Some(ValidationConfig { gates, hold }))
    }
}

/// `url` gates may only reach a check runner in the same pod.
fn loopback_url(url: &str) -> Result<reqwest::Url> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("invalid \"url\" {url:?}"))?;
    let loopback = match parsed.host() {
        Some(url::Host::Domain(domain)) => domain == "localhost",
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    };
    if parsed.scheme() != "http" || !loopback {
        bail!("\"url\" must be http on loopback (127.0.0.1, ::1 or localhost), got {url:?}");
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(gates: &str) -> Result<Option<ValidationConfig>> {
        ValidationConfig::parse(Some(gates.to_string()), None)
    }

    #[test]
    fn parses_gates_in_name_order() {
        assert_eq!(ValidationConfig::parse(None, None).unwrap(), None);
        let config = parse(
            r#"{"smoke": {"exec": ["python", "/checks/smoke.py"], "timeout_secs": 900},
                "basic": {"url": "http://127.0.0.1:9100/validate", "timeout_secs": 60}}"#,
        )
        .unwrap()
        .unwrap();
        let names: Vec<_> = config.gates.iter().map(|gate| gate.name.as_str()).collect();
        assert_eq!(names, ["basic", "smoke"]);
        assert_eq!(
            config.gates[0].check,
            GateCheck::Url("http://127.0.0.1:9100/validate".parse().unwrap())
        );
        assert_eq!(
            config.gates[1].check,
            GateCheck::Exec(vec!["python".into(), "/checks/smoke.py".into()])
        );
        assert_eq!(config.gates[1].timeout, Duration::from_secs(900));
        assert_eq!(config.hold, HoldMode::Card);
        let health = ValidationConfig::parse(
            Some(r#"{"a": {"exec": ["true"], "timeout_secs": 1}}"#.into()),
            Some("health".into()),
        );
        assert_eq!(health.unwrap().unwrap().hold, HoldMode::Health);
    }

    #[test]
    fn rejects_every_misconfiguration() {
        for (gates, expected) in [
            ("not json", "must be a JSON object"),
            ("{}", "at least one gate"),
            (r#"{"a": {"timeout_secs": 1}}"#, "exactly one"),
            (
                r#"{"a": {"exec": ["true"], "url": "http://127.0.0.1/", "timeout_secs": 1}}"#,
                "exactly one",
            ),
            (
                r#"{"a": {"url": "http://10.0.0.1:9100/", "timeout_secs": 1}}"#,
                "on loopback",
            ),
            (r#"{"a": {"exec": ["true"]}}"#, "must be a JSON object"),
            (r#"{"a": {"exec": [], "timeout_secs": 1}}"#, "exactly one"),
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
                "{gates:?}: expected {expected:?} in {error:?}"
            );
        }
        for (gates, hold, expected) in [
            (None, "card", "is set but"),
            (
                Some(r#"{"a": {"exec": ["true"], "timeout_secs": 1}}"#),
                "both",
                "must be \"card\"",
            ),
        ] {
            let error = ValidationConfig::parse(gates.map(Into::into), Some(hold.into()));
            let error = format!("{:#}", error.unwrap_err());
            assert!(
                error.contains(expected),
                "{gates:?}: expected {expected:?} in {error:?}"
            );
        }
    }
}
