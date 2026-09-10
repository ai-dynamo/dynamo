// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::str::FromStr;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Url;

const SIDECAR_PORT_ENV: &str = "DYN_SIDECAR_PORT";
const DECODE_ENGINE_PORT_ENV: &str = "DYN_DECODE_ENGINE_PORT";
const CONNECT_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_CONNECT_TIMEOUT_MS";
const READ_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_READ_TIMEOUT_MS";
const DRAIN_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_DRAIN_TIMEOUT_MS";
const PD_BACKEND_ENV: &str = "DYN_SIDECAR_PD_BACKEND";

const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_READ_TIMEOUT_MS: u64 = 300_000;
const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PdBackend {
    #[default]
    Unavailable,
    Sglang,
}

impl FromStr for PdBackend {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self> {
        match raw {
            "unavailable" => Ok(Self::Unavailable),
            "sglang" => Ok(Self::Sglang),
            _ => bail!("{PD_BACKEND_ENV} must be unavailable or sglang, got {raw:?}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: SocketAddr,
    pub decode_engine_url: Url,
    /// Backend protocol used when EPP selects a prefill endpoint.
    pub pd_backend: PdBackend,
    /// Maximum time allowed to establish a connection to the decode engine.
    pub connect_timeout: Duration,
    /// Maximum idle time between reads from a streaming decode response.
    pub read_timeout: Duration,
    /// Maximum time to drain active requests before forcing their streams closed.
    pub drain_timeout: Duration,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let sidecar_port = port_from_env(SIDECAR_PORT_ENV, 8000)?;
        let decode_engine_port = port_from_env(DECODE_ENGINE_PORT_ENV, 8001)?;
        Ok(Self {
            listen_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), sidecar_port),
            decode_engine_url: Url::parse(&format!("http://localhost:{decode_engine_port}"))
                .context("failed to construct local decode-engine URL")?,
            pd_backend: pd_backend_from_value(std::env::var_os(PD_BACKEND_ENV))?,
            connect_timeout: duration_from_env(CONNECT_TIMEOUT_MS_ENV, DEFAULT_CONNECT_TIMEOUT_MS)?,
            read_timeout: duration_from_env(READ_TIMEOUT_MS_ENV, DEFAULT_READ_TIMEOUT_MS)?,
            drain_timeout: duration_from_env(DRAIN_TIMEOUT_MS_ENV, DEFAULT_DRAIN_TIMEOUT_MS)?,
        })
    }
}

fn pd_backend_from_value(value: Option<OsString>) -> Result<PdBackend> {
    let Some(raw) = value else {
        return Ok(PdBackend::default());
    };
    raw.into_string()
        .map_err(|_| anyhow::anyhow!("{PD_BACKEND_ENV} must be valid UTF-8"))?
        .parse()
}

fn duration_from_env(name: &str, default_ms: u64) -> Result<Duration> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(Duration::from_millis(default_ms));
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))?;
    let milliseconds: u64 = raw
        .parse()
        .with_context(|| format!("{name} must be a valid duration in milliseconds"))?;
    if milliseconds == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(Duration::from_millis(milliseconds))
}

fn port_from_env(name: &str, default: u16) -> Result<u16> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(default);
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))?;
    let port: u16 = raw
        .parse()
        .with_context(|| format!("{name} must be a valid TCP port"))?;
    if port == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(port)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pd_backend_defaults_to_unavailable() {
        assert_eq!(pd_backend_from_value(None).unwrap(), PdBackend::Unavailable);
    }

    #[test]
    fn pd_backend_requires_an_explicit_supported_name() {
        assert_eq!(
            pd_backend_from_value(Some("sglang".into())).unwrap(),
            PdBackend::Sglang
        );
        assert_eq!(
            pd_backend_from_value(Some("unavailable".into())).unwrap(),
            PdBackend::Unavailable
        );
        for raw in ["", "SGLang", "vllm", " sglang", "sglang "] {
            assert!(
                pd_backend_from_value(Some(raw.into()))
                    .unwrap_err()
                    .to_string()
                    .contains(PD_BACKEND_ENV)
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn pd_backend_rejects_non_utf8_values() {
        use std::os::unix::ffi::OsStringExt;

        let error = pd_backend_from_value(Some(OsString::from_vec(vec![0xff]))).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("{PD_BACKEND_ENV} must be valid UTF-8")
        );
    }
}
