// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod push_router;
mod replica_sync;

use std::time::Duration;

use dynamo_runtime::{component::Client, pipeline::Error};
use serde::{Deserialize, Serialize};

pub(crate) use coordinator::{AffinityAcquire, affinity_id};
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};
pub use push_router::SessionAffinityPushRouter;

pub const MAX_SESSION_AFFINITY_TTL_SECS: u64 = 31_536_000;
pub const MAX_SESSION_AFFINITY_ENTRIES: usize = 65_536;
pub const MAX_SESSION_AFFINITY_ID_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionAffinityGrouping {
    Parent,
    Root,
}

impl std::str::FromStr for SessionAffinityGrouping {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "parent" => Ok(Self::Parent),
            "root" => Ok(Self::Root),
            _ => Err("session affinity grouping must be 'parent' or 'root'".to_string()),
        }
    }
}

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

pub(crate) async fn create_affinity_coordinator(
    ttl: Option<Duration>,
    grouping: Option<SessionAffinityGrouping>,
    client: Client,
) -> Result<Option<AffinityCoordinator>, Error> {
    let Some(ttl) = ttl else {
        if grouping.is_some() {
            return Err(coordinator::invalid_argument(
                "session affinity grouping requires session affinity TTL",
            ));
        }
        return Ok(None);
    };
    let coordinator = AffinityCoordinator::new_with_grouping(ttl, grouping)?;
    coordinator.enable_replica_sync(client).await?;
    Ok(Some(coordinator))
}

#[cfg(test)]
mod tests;
