// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parsing for `DYN_PUBLISH_GATE`.
//!
//! Every misconfiguration is an error: a worker that cannot tell what it is
//! supposed to check must not start unchecked.

use anyhow::{Context, Result, bail};

use crate::config::environment_names::runtime::validation as env_validation;

/// Parsed validation configuration. Only exists when a gate is configured.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidationConfig {
    /// Check URLs on a runner in the same pod, in configured order. A `200`
    /// is a pass; any other status is a fail.
    pub urls: Vec<reqwest::Url>,
}

impl ValidationConfig {
    /// Read the configuration from the environment. `Ok(None)` when no gate
    /// is configured, which keeps today's behaviour unchanged.
    pub fn from_env() -> Result<Option<Self>> {
        Self::parse(std::env::var(env_validation::DYN_PUBLISH_GATE).ok())
    }

    /// Parse one comma-separated URL list. Unset or blank means no gate.
    pub fn parse(list: Option<String>) -> Result<Option<Self>> {
        let Some(list) = list.filter(|value| !value.trim().is_empty()) else {
            return Ok(None);
        };
        let mut urls: Vec<reqwest::Url> = Vec::new();
        for url in list.split(',').map(str::trim) {
            let url = loopback_url(url).with_context(|| {
                format!(
                    "{} must be a comma-separated list of check URLs",
                    env_validation::DYN_PUBLISH_GATE
                )
            })?;
            if urls.contains(&url) {
                bail!("{} lists {url} twice", env_validation::DYN_PUBLISH_GATE);
            }
            urls.push(url);
        }
        Ok(Some(ValidationConfig { urls }))
    }
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

    fn parse(list: &str) -> Result<Option<ValidationConfig>> {
        ValidationConfig::parse(Some(list.to_string()))
    }

    #[test]
    fn parses_the_list_in_order() {
        assert_eq!(ValidationConfig::parse(None).unwrap(), None);
        assert_eq!(parse(" ").unwrap(), None);
        let config = parse(
            "http://127.0.0.1:9100/checks/smoke, http://localhost:9100/checks/tools,http://[::1]:9100/checks/soak",
        )
        .unwrap()
        .unwrap();
        let urls: Vec<_> = config.urls.iter().map(reqwest::Url::as_str).collect();
        assert_eq!(
            urls,
            [
                "http://127.0.0.1:9100/checks/smoke",
                "http://localhost:9100/checks/tools",
                "http://[::1]:9100/checks/soak",
            ]
        );
    }

    #[test]
    fn rejects_every_misconfiguration() {
        for (list, expected) in [
            ("not a url", "DYN_PUBLISH_GATE"),
            ("http://10.0.0.1:9100/", "on loopback"),
            ("https://127.0.0.1:9100/", "on loopback"),
            ("http://example.com/", "on loopback"),
            (
                "http://127.0.0.1:9100/a,,http://127.0.0.1:9100/b",
                "invalid URL",
            ),
            ("http://127.0.0.1:9100/a,http://127.0.0.1:9100/a", "twice"),
        ] {
            let error = format!("{:#}", parse(list).unwrap_err());
            assert!(
                error.contains(expected),
                "{list:?}: expected {expected:?} in {error:?}"
            );
        }
    }
}
