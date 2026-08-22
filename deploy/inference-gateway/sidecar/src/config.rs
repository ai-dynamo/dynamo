// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use reqwest::Url;

const SIDECAR_PORT_ENV: &str = "DYN_SIDECAR_PORT";
const DECODE_ENGINE_PORT_ENV: &str = "DYN_DECODE_ENGINE_PORT";
const CONNECT_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_CONNECT_TIMEOUT_MS";
const READ_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_READ_TIMEOUT_MS";

const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_READ_TIMEOUT_MS: u64 = 300_000;

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: SocketAddr,
    pub decode_engine_url: Url,
    /// Maximum time allowed to establish a connection to the decode engine.
    pub connect_timeout: Duration,
    /// Maximum idle time between reads from a streaming decode response.
    pub read_timeout: Duration,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let sidecar_port = port_from_env(SIDECAR_PORT_ENV, 8000)?;
        let decode_engine_port = port_from_env(DECODE_ENGINE_PORT_ENV, 8001)?;
        Ok(Self {
            listen_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), sidecar_port),
            decode_engine_url: Url::parse(&format!("http://localhost:{decode_engine_port}"))
                .context("failed to construct local decode-engine URL")?,
            connect_timeout: duration_from_env(CONNECT_TIMEOUT_MS_ENV, DEFAULT_CONNECT_TIMEOUT_MS)?,
            read_timeout: duration_from_env(READ_TIMEOUT_MS_ENV, DEFAULT_READ_TIMEOUT_MS)?,
        })
    }
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
