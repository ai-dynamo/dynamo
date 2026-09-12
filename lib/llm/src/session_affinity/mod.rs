// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::time::Duration;

use dynamo_runtime::{component::Client, pipeline::Error};

#[cfg(test)]
pub(crate) use coordinator::to_table;
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};
pub(crate) use coordinator::{affinity_id, from_table, invalid_argument};
pub(crate) use dynamo_kv_router::services::selection::affinity::Hold;
pub use dynamo_kv_router::services::selection::affinity::{
    MAX_SESSION_AFFINITY_TTL_SECS, SessionAffinityMode,
};

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
