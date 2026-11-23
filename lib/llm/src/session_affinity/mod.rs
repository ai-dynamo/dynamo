// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::{env::VarError, num::NonZeroUsize, time::Duration};

use dynamo_runtime::{
    component::Client, config::environment_names::router::DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES,
    pipeline::Error,
};
use dynamo_session_affinity::host::{MAX_SESSION_AFFINITY_ENTRIES, SessionAffinityConfig};

#[cfg(test)]
pub(crate) use coordinator::to_table;
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};
pub(crate) use coordinator::{affinity_id, from_table, invalid_argument};
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
