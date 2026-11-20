// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::{env::VarError, num::NonZeroUsize};

pub(crate) use dynamo_kv_router::plugins::worker_selection::SessionAffinitySpec;
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

/// The config every frontend affinity table is built from. Its entry limit is the role's
/// `max_entries`, else `DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES`.
pub(crate) fn affinity_config(spec: SessionAffinitySpec) -> Result<SessionAffinityConfig, Error> {
    let mut config = SessionAffinityConfig::new(spec.ttl).with_mode(spec.mode);
    config.max_entries = max_entries_from(
        spec.max_entries,
        std::env::var(DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES),
    )?;
    Ok(config)
}

fn max_entries_from(
    role: Option<NonZeroUsize>,
    env: Result<String, VarError>,
) -> Result<usize, Error> {
    if let Some(max_entries) = role {
        return Ok(max_entries.get());
    }
    match env {
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
    spec: Option<SessionAffinitySpec>,
    client: Client,
) -> Result<Option<AffinityCoordinator>, Error> {
    let Some(spec) = spec else {
        return Ok(None);
    };
    let coordinator = AffinityCoordinator::with_spec(spec)?;
    coordinator.enable_replica_sync(client).await?;
    Ok(Some(coordinator))
}

#[cfg(test)]
mod tests;
