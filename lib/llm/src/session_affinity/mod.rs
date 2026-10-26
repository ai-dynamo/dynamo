// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::{env, num::NonZeroUsize, time::Duration};

use anyhow::Context;
use dynamo_kv_router::services::selection::affinity::SessionAffinityConfig;
use dynamo_runtime::{
    component::Client, config::environment_names::router::DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES,
    pipeline::Error,
};

#[cfg(test)]
pub(crate) use coordinator::to_table;
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};
pub(crate) use coordinator::{affinity_id, from_table, invalid_argument};
pub(crate) use dynamo_kv_router::services::selection::affinity::{
    Hold, subagent_group_affinity_id,
};
pub use dynamo_kv_router::services::selection::affinity::{
    MAX_SESSION_AFFINITY_TTL_SECS, SessionAffinityMode,
};

pub(crate) fn config_from_env(
    ttl: Duration,
    mode: SessionAffinityMode,
) -> Result<SessionAffinityConfig, Error> {
    let mut config = SessionAffinityConfig::new(ttl).with_mode(mode);
    match env::var(DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES) {
        Ok(value) => {
            config.max_entries = value
                .parse::<NonZeroUsize>()
                .with_context(|| {
                    format!("{DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES} must be a positive integer")
                })?
                .get();
        }
        Err(env::VarError::NotPresent) => {}
        Err(error) => return Err(error).context(DYN_ROUTER_SESSION_AFFINITY_MAX_ENTRIES),
    }
    Ok(config)
}

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

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
