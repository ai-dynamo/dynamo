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
const DRAIN_TIMEOUT_MS_ENV: &str = "DYN_SIDECAR_DRAIN_TIMEOUT_MS";
const MODEL_NAME_ENV: &str = "DYN_MODEL_NAME";

const DEFAULT_CONNECT_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_READ_TIMEOUT_MS: u64 = 300_000;
const DEFAULT_DRAIN_TIMEOUT_MS: u64 = 30_000;

/// Which disaggregated P/D adapter the sidecar runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterMode {
    /// No adapter. A request carrying EPP P/D metadata fails with
    /// `pd_adapter_unavailable`, which is the historical behaviour and the
    /// default, so opting in is an explicit deployment decision.
    None,
    /// Raw-vLLM NIXL pull handoff adapter.
    VllmNixl,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub listen_addr: SocketAddr,
    pub decode_engine_url: Url,
    /// Maximum time allowed to establish a connection to the decode engine.
    pub connect_timeout: Duration,
    /// Maximum idle time between reads from a streaming decode response.
    pub read_timeout: Duration,
    /// Maximum time to drain active requests before forcing their streams closed.
    pub drain_timeout: Duration,
    /// Which P/D adapter to construct.
    pub adapter_mode: AdapterMode,
    /// Assumed vLLM NIXL protocol revision. Asserted at startup so a
    /// deployment cannot silently run a protocol the fixtures do not cover.
    pub protocol_version: String,
    /// Maximum accepted request body size, in bytes.
    pub max_request_bytes: usize,
    /// Maximum time allowed to read a whole request body.
    pub client_body_timeout: Duration,
    /// Maximum accepted prefill response size, in bytes.
    pub max_prefill_response_bytes: usize,
    /// Total time allowed for the prefill leg, across every chunk.
    pub prefill_deadline: Duration,
    /// Model both workers serve. Recorded at startup for diagnosis.
    pub model: String,
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
            drain_timeout: duration_from_env(DRAIN_TIMEOUT_MS_ENV, DEFAULT_DRAIN_TIMEOUT_MS)?,
            adapter_mode: adapter_mode_from_env()?,
            protocol_version: optional_env(crate::vllm_nixl::PROTOCOL_VERSION_ENV)?
                .unwrap_or_else(|| crate::vllm_nixl::SUPPORTED_PROTOCOL_VERSION.to_string()),
            client_body_timeout: duration_from_env(
                crate::vllm_nixl::CLIENT_BODY_TIMEOUT_MS_ENV,
                crate::vllm_nixl::DEFAULT_CLIENT_BODY_TIMEOUT.as_millis() as u64,
            )?,
            max_request_bytes: byte_limit_from_env(
                crate::vllm_nixl::MAX_REQUEST_BYTES_ENV,
                crate::vllm_nixl::DEFAULT_MAX_REQUEST_BYTES,
            )?,
            prefill_deadline: duration_from_env(
                crate::vllm_nixl::PREFILL_DEADLINE_MS_ENV,
                crate::vllm_nixl::DEFAULT_PREFILL_DEADLINE.as_millis() as u64,
            )?,
            max_prefill_response_bytes: byte_limit_from_env(
                crate::vllm_nixl::MAX_PREFILL_RESPONSE_BYTES_ENV,
                crate::vllm_nixl::DEFAULT_MAX_PREFILL_RESPONSE_BYTES,
            )?,
            model: optional_env(MODEL_NAME_ENV)?.unwrap_or_default(),
        })
    }
}

/// Read an optional environment value, treating a missing one as absent but a
/// non-UTF-8 one as a configuration error rather than silently falling back to
/// a default.
fn optional_env(name: &str) -> Result<Option<String>> {
    match std::env::var_os(name) {
        None => Ok(None),
        Some(raw) => raw
            .into_string()
            .map(Some)
            .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8")),
    }
}

fn adapter_mode_from_env() -> Result<AdapterMode> {
    let Some(raw) = std::env::var_os(crate::vllm_nixl::ADAPTER_ENV) else {
        return Ok(AdapterMode::None);
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{} must be valid UTF-8", crate::vllm_nixl::ADAPTER_ENV))?;
    match raw.trim() {
        "none" => Ok(AdapterMode::None),
        "vllm_nixl" => Ok(AdapterMode::VllmNixl),
        other => bail!(
            "{} must be one of none, vllm_nixl (got {other:?})",
            crate::vllm_nixl::ADAPTER_ENV
        ),
    }
}

fn byte_limit_from_env(name: &str, default: usize) -> Result<usize> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(default);
    };
    let raw = raw
        .into_string()
        .map_err(|_| anyhow::anyhow!("{name} must be valid UTF-8"))?;
    let limit: usize = raw
        .parse()
        .with_context(|| format!("{name} must be a byte count"))?;
    if limit == 0 {
        bail!("{name} must be greater than zero");
    }
    Ok(limit)
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
