// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in worker-group routing advertisement, consumed by the frontend.
//! No mode means no advertisement, not an implicit round-robin override.

use std::fmt::Display;
use std::str::FromStr;

use dynamo_llm::discovery::LoadThresholdConfig;
use dynamo_llm::entrypoint::RouterConfig;
use dynamo_llm::session_affinity::{MAX_SESSION_AFFINITY_TTL_SECS, SessionAffinityMode};
use dynamo_runtime::pipeline::RouterMode;

/// The worker routing options used by SGLang sidecar deployments. These affect
/// only this worker group's model card, never the engine's execution settings.
#[derive(clap::Args, Clone, Debug, Default)]
pub struct WorkerRouterArgs {
    /// Advertise this group's routing policy instead of inheriting the frontend's.
    #[arg(long, value_parser = parse_router_mode)]
    pub router_mode: Option<RouterMode>,

    /// Synchronize KV-router state across frontend replicas for this group.
    #[arg(long, requires = "router_mode", num_args = 0..=1,
          require_equals = true, default_missing_value = "true")]
    pub router_replica_sync: Option<bool>,

    /// Enable session affinity for this group with an idle TTL in seconds.
    #[arg(long, requires = "router_mode",
          value_parser = clap::value_parser!(u64).range(1..=MAX_SESSION_AFFINITY_TTL_SECS))]
    pub router_session_affinity_ttl_secs: Option<u64>,

    /// Session affinity policy: hard (default) or soft.
    #[arg(long, requires = "router_mode")]
    pub router_session_affinity_mode: Option<SessionAffinityMode>,

    /// Decode-block utilization threshold (0..1), or None to disable.
    #[arg(long, requires = "router_mode")]
    active_decode_blocks_threshold: Option<Nullable<f64>>,

    /// Prefill token-count threshold, or None to disable.
    #[arg(long, requires = "router_mode")]
    active_prefill_tokens_threshold: Option<Nullable<u64>>,

    /// Nonnegative prefill token fraction threshold, or None to disable.
    #[arg(long, requires = "router_mode")]
    active_prefill_tokens_threshold_frac: Option<Nullable<f64>>,
}

impl WorkerRouterArgs {
    pub fn router_config(&self) -> Result<Option<RouterConfig>, String> {
        let Some(router_mode) = self.router_mode else {
            return Ok(None);
        };
        let thresholds = LoadThresholdConfig {
            active_decode_blocks_threshold: self
                .active_decode_blocks_threshold
                .as_ref()
                .and_then(|v| v.0),
            active_prefill_tokens_threshold: self
                .active_prefill_tokens_threshold
                .as_ref()
                .and_then(|v| v.0),
            active_prefill_tokens_threshold_frac: self
                .active_prefill_tokens_threshold_frac
                .as_ref()
                .and_then(|v| v.0),
        };
        thresholds.validate()?;
        let mut config = RouterConfig {
            router_mode,
            load_threshold_config: thresholds,
            session_affinity_ttl_secs: self.router_session_affinity_ttl_secs,
            session_affinity_mode: self.router_session_affinity_mode.unwrap_or_default(),
            ..Default::default()
        };
        config.kv_router_config.router_replica_sync = self.router_replica_sync.unwrap_or(false);
        Ok(Some(config))
    }
}

// Match the Python worker's CLI spellings, which differ from RouterMode's
// serde names (notably `kv`).
fn parse_router_mode(value: &str) -> Result<RouterMode, String> {
    match value {
        "round-robin" => Ok(RouterMode::RoundRobin),
        "random" => Ok(RouterMode::Random),
        "kv" => Ok(RouterMode::KV),
        "direct" => Ok(RouterMode::Direct),
        "least-loaded" => Ok(RouterMode::LeastLoaded),
        "device-aware-weighted" => Ok(RouterMode::DeviceAwareWeighted),
        _ => Err(format!("unknown router mode: {value}")),
    }
}

/// Preserve Python argparse's explicit `None` threshold values.
#[derive(Clone, Debug)]
struct Nullable<T>(Option<T>);

impl<T: FromStr> FromStr for Nullable<T>
where
    T::Err: Display,
{
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case("none") {
            Ok(Self(None))
        } else {
            value
                .parse()
                .map(|v| Self(Some(v)))
                .map_err(|e: T::Err| e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        routing: WorkerRouterArgs,
    }

    #[test]
    fn absent_mode_inherits_frontend() {
        let args = Cli::try_parse_from(["worker"]).unwrap();
        assert!(args.routing.router_config().unwrap().is_none());
    }

    #[test]
    fn accepts_prefill_recipe_and_explicit_none_thresholds() {
        let args = Cli::try_parse_from([
            "worker",
            "--router-mode",
            "kv",
            "--router-replica-sync",
            "--router-session-affinity-ttl-secs",
            "3600",
            "--active-decode-blocks-threshold",
            "None",
            "--active-prefill-tokens-threshold",
            "None",
            "--active-prefill-tokens-threshold-frac",
            "None",
        ])
        .unwrap();
        let config = args.routing.router_config().unwrap().unwrap();
        assert_eq!(config.router_mode, RouterMode::KV);
        assert!(config.kv_router_config.router_replica_sync);
        assert_eq!(config.session_affinity_ttl_secs, Some(3600));
        assert_eq!(config.session_affinity_mode, SessionAffinityMode::Hard);
        assert!(!config.load_threshold_config.is_configured());
    }
}
