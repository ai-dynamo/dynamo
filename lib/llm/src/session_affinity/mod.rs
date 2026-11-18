// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::{env::VarError, num::NonZeroUsize, time::Duration};

use dynamo_runtime::{
    component::Client,
    config::environment_names::router::{
        DYN_ROUTER_SESSION_AFFINITY_ACTIVE_SECS, DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES,
        DYN_ROUTER_SESSION_AFFINITY_PLACEMENT, DYN_ROUTER_SESSION_AFFINITY_REBALANCE_DELTA,
    },
    pipeline::Error,
};
use dynamo_session_affinity::host::{MAX_SESSION_AFFINITY_ENTRIES, SessionAffinityConfig};

#[cfg(test)]
pub(crate) use coordinator::to_table;
pub use coordinator::{
    AffinityCoordinator, AffinityTarget, PlacementSettings, SessionPlacement, explicit_target,
};
pub(crate) use coordinator::{WorkerLoads, affinity_id, from_table, invalid_argument};
pub(crate) use dynamo_session_affinity::host::{Hold, subagent_group_affinity_id};
pub use dynamo_session_affinity::host::{MAX_SESSION_AFFINITY_TTL_SECS, SessionAffinityMode};

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

/// The config every frontend affinity table is built from, with its entry
/// limit read from `DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES`.
pub(crate) fn affinity_config(
    ttl: Duration,
    mode: SessionAffinityMode,
) -> Result<SessionAffinityConfig, Error> {
    let mut config = SessionAffinityConfig::new(ttl).with_mode(mode);
    config.max_entries = max_entries_from(std::env::var(DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES))?;
    Ok(config)
}

/// Default activity window for counting live sessions (`--router-session-affinity-active-secs`).
pub(crate) const DEFAULT_SESSION_ACTIVE_SECS: u64 = 60;

/// Placement settings from `DYN_ROUTER_SESSION_AFFINITY_PLACEMENT`,
/// `DYN_ROUTER_SESSION_AFFINITY_REBALANCE_DELTA` and `DYN_ROUTER_SESSION_AFFINITY_ACTIVE_SECS`.
pub(crate) fn placement_settings_from_env() -> Result<PlacementSettings, Error> {
    placement_settings_from(|name| std::env::var(name))
}

fn placement_settings_from(
    var: impl Fn(&str) -> Result<String, VarError>,
) -> Result<PlacementSettings, Error> {
    let placement = match env_value(&var, DYN_ROUTER_SESSION_AFFINITY_PLACEMENT)? {
        Some(value) => value.parse::<SessionPlacement>().map_err(|message| {
            anyhow::anyhow!("{DYN_ROUTER_SESSION_AFFINITY_PLACEMENT}: {message}")
        })?,
        None => SessionPlacement::default(),
    };
    let rebalance_delta = env_count::<usize>(&var, DYN_ROUTER_SESSION_AFFINITY_REBALANCE_DELTA)?
        .and_then(NonZeroUsize::new);
    let active_secs = env_count::<u64>(&var, DYN_ROUTER_SESSION_AFFINITY_ACTIVE_SECS)?
        .unwrap_or(DEFAULT_SESSION_ACTIVE_SECS);
    Ok(PlacementSettings {
        placement,
        rebalance_delta,
        active_within: (active_secs > 0).then(|| Duration::from_secs(active_secs)),
    })
}

fn env_value(
    var: &impl Fn(&str) -> Result<String, VarError>,
    name: &str,
) -> Result<Option<String>, Error> {
    match var(name) {
        Ok(value) => Ok(Some(value.trim().to_string())),
        Err(VarError::NotPresent) => Ok(None),
        Err(VarError::NotUnicode(_)) => anyhow::bail!("{name} must be valid unicode"),
    }
}

fn env_count<T: std::str::FromStr>(
    var: &impl Fn(&str) -> Result<String, VarError>,
    name: &str,
) -> Result<Option<T>, Error> {
    env_value(var, name)?
        .map(|value| {
            value.parse::<T>().map_err(|_| {
                anyhow::anyhow!("{name} must be a non-negative integer, got {value:?}")
            })
        })
        .transpose()
}

fn max_entries_from(raw: Result<String, VarError>) -> Result<usize, Error> {
    match raw {
        Ok(value) => value.parse::<NonZeroUsize>().map(NonZeroUsize::get).map_err(|_| {
            anyhow::anyhow!(
                "{DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES} must be a positive integer, got {value:?}"
            )
        }),
        Err(VarError::NotPresent) => Ok(MAX_SESSION_AFFINITY_ENTRIES),
        Err(VarError::NotUnicode(_)) => {
            anyhow::bail!("{DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES} must be valid unicode")
        }
    }
}

pub(crate) async fn create_affinity_coordinator(
    ttl: Option<Duration>,
    mode: SessionAffinityMode,
    client: Client,
) -> Result<Option<AffinityCoordinator>, Error> {
    let Some(ttl) = ttl else {
        return Ok(None);
    };
    let coordinator = AffinityCoordinator::new(ttl, mode)?;
    coordinator.enable_replica_sync(client).await?;
    Ok(Some(coordinator))
}

#[cfg(test)]
mod tests;
