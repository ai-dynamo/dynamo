// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod push_router;
mod replica_sync;

use std::time::Duration;

use dynamo_runtime::{component::Client, pipeline::Error};

pub(crate) use coordinator::{AffinityAcquire, affinity_id};
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};
pub use push_router::SessionAffinityPushRouter;

pub const MAX_SESSION_AFFINITY_TTL_SECS: u64 = 31_536_000;
pub const MAX_SESSION_AFFINITY_ENTRIES: usize = 65_536;
pub const MAX_SESSION_AFFINITY_ID_BYTES: usize = 256;

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

pub(crate) async fn create_affinity_coordinator(
    ttl: Option<Duration>,
    client: Client,
    replica_sync: bool,
) -> Result<Option<AffinityCoordinator>, Error> {
    let Some(ttl) = ttl else {
        return Ok(None);
    };
    let coordinator = AffinityCoordinator::new(ttl)?;
    if replica_sync {
        coordinator.enable_replica_sync(client).await?;
    }
    Ok(Some(coordinator))
}

#[cfg(test)]
mod tests;
