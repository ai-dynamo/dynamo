// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::{str::FromStr, time::Duration};

use dynamo_runtime::{component::Client, pipeline::Error};
use serde::{Deserialize, Serialize};

pub(crate) use coordinator::{AffinityAcquire, affinity_id, invalid_argument};
pub use coordinator::{AffinityCoordinator, AffinityTarget, explicit_target};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionAffinityMode {
    #[default]
    Hard,
    Soft,
}

impl SessionAffinityMode {
    pub fn is_soft(self) -> bool {
        matches!(self, Self::Soft)
    }
}

/// Which id a request's session binding is keyed on.
///
/// Frontend-only: it never appears on a model card, so it cannot break an older frontend's
/// card parsing the way a new `SessionAffinityMode` wire value would.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionAffinityBinding {
    #[default]
    Session,
    /// A subagent binds under its parent's group rather than its own session.
    ParentGroup,
}

impl FromStr for SessionAffinityBinding {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "session" => Ok(Self::Session),
            "parent-group" => Ok(Self::ParentGroup),
            _ => Err(format!(
                "invalid session affinity binding {value:?}; expected 'session' or 'parent-group'"
            )),
        }
    }
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

/// The `\u{1}` prefix cannot appear in an HTTP header value, so no client can claim this key as
/// its own session id; the constant length keeps a long parent id from overflowing the limit.
pub(crate) fn subagent_group_affinity_id(parent_session_id: &str) -> String {
    let digest = blake3::hash(parent_session_id.as_bytes());
    format!("\u{1}sg:{}", digest.to_hex())
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
