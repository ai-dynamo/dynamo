// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang's versioned, engine-owned managed-sidecar launch contract.

use std::collections::HashSet;
use std::str::FromStr;

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

pub(crate) const WORKER_GROUP_KEY: &str = "sglang_worker_group_id";
pub(crate) const KV_CONFIG_KEY: &str = "sglang_sidecar_kv_events";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SidecarMode {
    Full,
    Telemetry,
}

/// Matches sglang.srt.entrypoints.sidecar_context.SidecarContext, version 1.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarContext {
    pub version: u32,
    pub mode: SidecarMode,
    pub node_rank: u32,
    pub nnodes: u32,
    pub dp_size: u32,
    pub dist_init_addr: Option<String>,
    pub kv_event_sources: Vec<KvEventSource>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KvEventSource {
    pub dp_rank: u32,
    pub endpoint: String,
    pub topic: String,
    pub block_size: u32,
    /// Accepted for wire compatibility; the existing Dynamo relay is live-only.
    pub replay_endpoint: Option<String>,
}

impl FromStr for SidecarContext {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let context: Self = serde_json::from_str(raw)
            .map_err(|error| format!("invalid SGLANG_SIDECAR_CONTEXT: {error}"))?;
        context.validate().map_err(|error| error.to_string())?;
        Ok(context)
    }
}

impl SidecarContext {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.version == 1,
            "unsupported SGLANG_SIDECAR_CONTEXT version {}",
            self.version
        );
        ensure!(
            self.nnodes > 0 && self.node_rank < self.nnodes,
            "invalid sidecar node topology"
        );
        ensure!(self.dp_size > 0, "sidecar dp_size must be positive");
        ensure!(
            (self.mode == SidecarMode::Full) == (self.node_rank == 0),
            "full mode requires node_rank=0; telemetry mode requires a follower node"
        );
        ensure!(
            self.mode != SidecarMode::Telemetry || !self.kv_event_sources.is_empty(),
            "telemetry mode requires at least one local KV source"
        );
        ensure!(
            self.nnodes == 1
                || self
                    .dist_init_addr
                    .as_ref()
                    .is_some_and(|addr| !addr.trim().is_empty()),
            "multinode sidecars require dist_init_addr for leader matching"
        );
        let mut ranks = HashSet::new();
        let mut endpoints = HashSet::new();
        for source in &self.kv_event_sources {
            ensure!(
                source.dp_rank < self.dp_size,
                "KV source rank {} is outside dp_size {}",
                source.dp_rank,
                self.dp_size
            );
            ensure!(
                ranks.insert(source.dp_rank),
                "duplicate local KV source rank {}",
                source.dp_rank
            );
            ensure!(
                endpoints.insert(&source.endpoint),
                "duplicate local KV source endpoint {}",
                source.endpoint
            );
            ensure!(
                source.block_size > 0,
                "KV source block_size must be positive"
            );
            validate_endpoint(&source.endpoint)?;
            if let Some(endpoint) = &source.replay_endpoint {
                validate_endpoint(endpoint)?;
            }
        }
        Ok(())
    }

    pub(crate) fn validate_registration(
        &self,
        dp_size: u32,
        block_size: Option<u32>,
    ) -> Result<()> {
        ensure!(
            self.dp_size == dp_size,
            "sidecar context dp_size does not match leader registration"
        );
        for source in &self.kv_event_sources {
            ensure!(
                Some(source.block_size) == block_size,
                "KV source block_size does not match leader registration"
            );
        }
        Ok(())
    }

    pub(crate) fn worker_group_id(&self) -> Result<Option<String>> {
        if self.nnodes == 1 {
            return Ok(None);
        }
        let raw = self
            .dist_init_addr
            .as_deref()
            .context("missing dist_init_addr")?
            .trim();
        let url = if raw.contains("://") {
            raw.to_owned()
        } else {
            format!("tcp://{raw}")
        };
        let address = url::Url::parse(&url).context("invalid dist_init_addr")?;
        ensure!(
            address.scheme() == "tcp",
            "dist_init_addr must be a TCP address"
        );
        validate_endpoint(&url)?;
        // Use the in-process group-key format: resolve the shared rendezvous
        // address, not a per-node source or gRPC address.
        let resolved = address.socket_addrs(|| None)?;
        let address = resolved
            .first()
            .context("dist_init_addr resolved to no addresses")?;
        Ok(Some(format!("dist_init:tcp://{address}")))
    }
}

fn validate_endpoint(endpoint: &str) -> Result<()> {
    if let Some(path) = endpoint.strip_prefix("ipc://") {
        ensure!(
            (path.starts_with('/') || path.starts_with('@'))
                && path.len() > 1
                && !path.contains('\0'),
            "IPC source must have an absolute or abstract socket path"
        );
        return Ok(());
    }
    let url = url::Url::parse(endpoint).context("invalid KV source endpoint")?;
    let host = url
        .host_str()
        .context("KV source endpoint requires a host")?;
    ensure!(
        url.scheme() == "tcp" && url.port().is_some_and(|port| port > 0),
        "KV source must use tcp://HOST:PORT or ipc://PATH"
    );
    ensure!(
        !matches!(host, "*" | "0.0.0.0" | "[::]" | "::"),
        "KV source endpoint must be dialable, not a wildcard bind address"
    );
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !matches!(url.path(), "" | "/")
    {
        bail!("KV source TCP endpoint must contain only host and port");
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use serde_json::{Value, json};

    pub(crate) fn context_json(mode: &str) -> Value {
        json!({"version":1,"mode":mode,"node_rank":if mode == "full" {0} else {1},
            "nnodes":2,"dp_size":8,"dist_init_addr":"tcp://127.0.0.1:2345",
            "kv_event_sources":[{"dp_rank":4,"endpoint":"tcp://127.0.0.1:5561",
                "topic":"","block_size":64,"replay_endpoint":null}]})
    }

    #[test]
    fn accepts_upstream_contract_and_global_not_local_rank() {
        let context: SidecarContext = context_json("telemetry").to_string().parse().unwrap();
        assert_eq!(context.kv_event_sources[0].dp_rank, 4);
        assert_eq!(
            context.worker_group_id().unwrap().as_deref(),
            Some("dist_init:tcp://127.0.0.1:2345")
        );
        context.validate_registration(8, Some(64)).unwrap();
        assert!(context.validate_registration(4, Some(64)).is_err());
        assert!(context.validate_registration(8, Some(32)).is_err());
    }

    #[test]
    fn validates_context_before_runtime_or_grpc_startup() {
        for (field, value) in [
            ("version", json!(2)),
            ("mode", json!("unknown")),
            ("node_rank", json!(0)),
            ("node_rank", json!(2)),
            ("dp_size", json!(0)),
            ("dist_init_addr", Value::Null),
            ("kv_event_sources", json!([])),
        ] {
            let mut raw = context_json("telemetry");
            raw[field] = value;
            assert!(raw.to_string().parse::<SidecarContext>().is_err(), "{raw}");
        }
    }

    #[test]
    fn rejects_conflicting_or_undialable_sources() {
        for (field, value) in [
            ("dp_rank", json!(8)),
            ("dp_rank", json!(-1)),
            ("block_size", json!(0)),
            ("endpoint", json!("tcp://*:5561")),
            ("endpoint", json!("inproc://events")),
            ("endpoint", json!("ipc://relative")),
        ] {
            let mut raw = context_json("telemetry");
            raw["kv_event_sources"][0][field] = value;
            assert!(raw.to_string().parse::<SidecarContext>().is_err(), "{raw}");
        }
        let mut raw = context_json("telemetry");
        let source = raw["kv_event_sources"][0].clone();
        raw["kv_event_sources"].as_array_mut().unwrap().push(source);
        assert!(raw.to_string().parse::<SidecarContext>().is_err());
        raw["kv_event_sources"][1]["dp_rank"] = json!(5);
        assert!(raw.to_string().parse::<SidecarContext>().is_err());
    }

    #[test]
    fn leader_may_have_no_local_publishers() {
        let mut raw = context_json("full");
        raw["kv_event_sources"] = json!([]);
        let context: SidecarContext = raw.to_string().parse().unwrap();
        context.validate_registration(8, Some(64)).unwrap();
        assert!(context.kv_event_sources.is_empty());
    }

    #[test]
    fn ipv6_and_ipc_are_supported() {
        let mut raw = context_json("telemetry");
        raw["dist_init_addr"] = json!("tcp://[::1]:2345");
        raw["kv_event_sources"][0]["endpoint"] = json!("ipc:///tmp/sglang-kv-test.sock");
        let context: SidecarContext = raw.to_string().parse().unwrap();
        assert_eq!(
            context.worker_group_id().unwrap().as_deref(),
            Some("dist_init:tcp://[::1]:2345")
        );
    }
}
