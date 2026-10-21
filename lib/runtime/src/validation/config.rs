// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing for `DYN_PUBLISH_GATE` and `DYN_READY_GATE`.
//!
//! Every misconfiguration is an error: a worker that cannot tell what it is
//! supposed to check must not start unchecked.

use anyhow::{Context, Result, bail};

use crate::config::environment_names::runtime::validation as env_validation;

/// The release point a gate holds. `/health` is held until every gate passes,
/// whichever it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gate {
    /// Holds the model card's discovery write: no frontend routes to the
    /// worker under any router while these run.
    Publish,
    /// Called once the card is published; holds only `/health`. Under etcd
    /// discovery the worker is routable while these run.
    Ready,
}

impl Gate {
    /// Name in the `/health` body and the check request.
    pub fn as_str(self) -> &'static str {
        match self {
            Gate::Publish => "publish",
            Gate::Ready => "ready",
        }
    }

    fn env_name(self) -> &'static str {
        match self {
            Gate::Publish => env_validation::DYN_PUBLISH_GATE,
            Gate::Ready => env_validation::DYN_READY_GATE,
        }
    }
}

/// One check URL at one release point.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateSpec {
    pub gate: Gate,
    /// A check runner in the same pod. A `200` is a pass; any other status
    /// is a fail.
    pub url: reqwest::Url,
}

/// Parsed validation configuration. Only exists when a gate is configured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationConfig {
    /// Publish gates, then ready gates, each in configured order.
    pub gates: Vec<GateSpec>,
}

impl ValidationConfig {
    /// Read the configuration from the environment. `Ok(None)` when no gate
    /// is configured, which keeps today's behaviour unchanged.
    pub fn from_env() -> Result<Option<Self>> {
        Self::parse(
            std::env::var(env_validation::DYN_PUBLISH_GATE).ok(),
            std::env::var(env_validation::DYN_READY_GATE).ok(),
        )
    }

    pub fn parse(publish: Option<String>, ready: Option<String>) -> Result<Option<Self>> {
        let mut gates = urls(Gate::Publish, publish)?;
        gates.extend(urls(Gate::Ready, ready)?);
        Ok((!gates.is_empty()).then_some(ValidationConfig { gates }))
    }

    /// True when the model card is held until the publish gates pass, which
    /// is whenever there are any.
    pub fn holds_card(&self) -> bool {
        self.gates.iter().any(|spec| spec.gate == Gate::Publish)
    }
}

/// Parse one comma-separated URL list. Unset or blank means none.
fn urls(gate: Gate, list: Option<String>) -> Result<Vec<GateSpec>> {
    let Some(list) = list.filter(|value| !value.trim().is_empty()) else {
        return Ok(Vec::new());
    };
    let mut specs: Vec<GateSpec> = Vec::new();
    for url in list.split(',').map(str::trim) {
        let url = loopback_url(url).with_context(|| {
            format!(
                "{} must be a comma-separated list of check URLs",
                gate.env_name()
            )
        })?;
        if specs.iter().any(|spec| spec.url == url) {
            bail!("{} lists {url} twice", gate.env_name());
        }
        specs.push(GateSpec { gate, url });
    }
    Ok(specs)
}

/// Gates may only reach a check runner in the same pod.
fn loopback_url(url: &str) -> Result<reqwest::Url> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("invalid URL {url:?}"))?;
    let loopback = match parsed.host() {
        Some(url::Host::Domain(domain)) => domain == "localhost",
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        None => false,
    };
    if parsed.scheme() != "http" || !loopback {
        bail!("URL must be http on loopback (127.0.0.1, ::1 or localhost), got {url:?}");
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(publish: &str, ready: &str) -> Result<Option<ValidationConfig>> {
        ValidationConfig::parse(Some(publish.to_string()), Some(ready.to_string()))
    }

    #[test]
    fn parses_both_lists_in_order() {
        assert_eq!(ValidationConfig::parse(None, None).unwrap(), None);
        assert_eq!(parse(" ", "").unwrap(), None);
        let config = parse(
            "http://127.0.0.1:9100/checks/smoke, http://localhost:9100/checks/tools",
            "http://[::1]:9100/checks/warm",
        )
        .unwrap()
        .unwrap();
        let gates: Vec<_> = config
            .gates
            .iter()
            .map(|spec| (spec.gate, spec.url.as_str()))
            .collect();
        assert_eq!(
            gates,
            [
                (Gate::Publish, "http://127.0.0.1:9100/checks/smoke"),
                (Gate::Publish, "http://localhost:9100/checks/tools"),
                (Gate::Ready, "http://[::1]:9100/checks/warm"),
            ]
        );
        assert!(config.holds_card());
    }

    #[test]
    fn only_publish_gates_hold_the_card() {
        let config = parse("", "http://127.0.0.1:9100/warm").unwrap().unwrap();
        assert!(!config.holds_card());
    }

    #[test]
    fn rejects_every_misconfiguration() {
        for (publish, ready, expected) in [
            ("not a url", "", "DYN_PUBLISH_GATE"),
            ("http://10.0.0.1:9100/", "", "on loopback"),
            ("", "https://127.0.0.1:9100/", "on loopback"),
            ("", "http://example.com/", "DYN_READY_GATE"),
            (
                "http://127.0.0.1:9100/a,,http://127.0.0.1:9100/b",
                "",
                "invalid URL",
            ),
            (
                "http://127.0.0.1:9100/a,http://127.0.0.1:9100/a",
                "",
                "twice",
            ),
        ] {
            let error = format!("{:#}", parse(publish, ready).unwrap_err());
            assert!(
                error.contains(expected),
                "{publish:?} {ready:?}: expected {expected:?} in {error:?}"
            );
        }
    }
}
