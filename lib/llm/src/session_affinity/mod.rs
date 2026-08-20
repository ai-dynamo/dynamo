// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod lifecycle;
mod push_router;
mod replica_sync;
mod state;

use std::{str::FromStr, time::Duration};

use dynamo_runtime::{
    component::Client,
    error::{ErrorType, match_error_chain},
    pipeline::Error,
};
use serde::{Deserialize, Serialize};

pub(crate) use coordinator::affinity_id;
pub use coordinator::{AffinityCoordinator, explicit_target};
pub(crate) use lifecycle::AffinityAcquire;
pub use push_router::SessionAffinityPushRouter;

pub type AffinityTarget = dynamo_runtime::pipeline::RouteTarget;

pub(crate) fn is_cancelled(error: &Error) -> bool {
    match_error_chain(error.as_ref(), &[ErrorType::Cancelled], &[])
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionAffinityMode {
    #[default]
    Hard,
    Soft,
}

impl FromStr for SessionAffinityMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "hard" => Ok(Self::Hard),
            "soft" => Ok(Self::Soft),
            _ => Err(format!(
                "invalid session affinity mode {value:?}; expected 'hard' or 'soft'"
            )),
        }
    }
}

pub const MAX_SESSION_AFFINITY_TTL_SECS: u64 = 31_536_000;
pub const MAX_SESSION_AFFINITY_ENTRIES: usize = 65_536;
pub const MAX_SESSION_AFFINITY_ID_BYTES: usize = 256;

pub type LlmResponse =
    crate::types::Annotated<crate::protocols::common::llm_backend::LLMEngineOutput>;

pub(crate) async fn create_affinity_coordinator(
    ttl: Option<Duration>,
    client: Client,
) -> Result<Option<AffinityCoordinator>, Error> {
    let Some(ttl) = ttl else {
        return Ok(None);
    };
    let coordinator = AffinityCoordinator::new(ttl)?;
    coordinator.enable_replica_sync(client).await?;
    Ok(Some(coordinator))
}

#[cfg(test)]
mod tests;
