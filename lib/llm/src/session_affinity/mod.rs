// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod coordinator;
mod replica_sync;

use std::{sync::LazyLock, time::Duration};

use dynamo_runtime::{
    component::Client, config::environment_names::router::DYN_ROUTER_SESSION_AFFINITY_BINDING,
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

/// Which id a request's session binding is keyed on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionAffinityBinding {
    /// Default. A request that carries a parent session id binds under its parent's group, so
    /// the subagents of one parent share one target.
    #[default]
    ParentGroup,
    /// Every request binds on its own session id, so subagents are placed independently.
    Session,
}

impl std::str::FromStr for SessionAffinityBinding {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim() {
            "parent-group" => Ok(Self::ParentGroup),
            "session" => Ok(Self::Session),
            _ => Err(format!(
                "invalid session affinity binding {value:?}; expected 'parent-group' or 'session'"
            )),
        }
    }
}

impl SessionAffinityBinding {
    /// The process-wide binding from `DYN_ROUTER_SESSION_AFFINITY_BINDING`, read once.
    pub(crate) fn from_env() -> Self {
        static BINDING: LazyLock<SessionAffinityBinding> =
            LazyLock::new(|| SessionAffinityBinding::from_lookup(|key| std::env::var(key).ok()));
        *BINDING
    }

    fn from_lookup(get_env: impl Fn(&str) -> Option<String>) -> Self {
        let Some(value) = get_env(DYN_ROUTER_SESSION_AFFINITY_BINDING) else {
            return Self::default();
        };
        value.parse::<Self>().unwrap_or_else(|error| {
            tracing::warn!(%error, "ignoring {DYN_ROUTER_SESSION_AFFINITY_BINDING}");
            Self::default()
        })
    }
}

#[cfg(test)]
mod tests;
