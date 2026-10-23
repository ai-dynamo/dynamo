// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What worker selection reads from a request.

use std::sync::Arc;

use dynamo_kv_router::{RouterConfigOverride, protocols::BlockExtraInfo};
use dynamo_runtime::pipeline::{AsyncEngineContext, AsyncEngineContextProvider, Error, SingleIn};

use crate::{
    preprocessor::PreprocessedRequest,
    protocols::{
        TokenIdType,
        common::{
            extensions::{AgentContext, SessionAffinityId},
            preprocessor::{MigrationState, RoutingHints},
        },
    },
    session_affinity::affinity_id,
};

/// The part of a request that worker selection reads.
///
/// Selection and route planning take `&dyn RouteRequest`, so a host can plan a
/// route for its own request type. Dispatch still takes the frontend's
/// [`PreprocessedRequest`].
pub trait RouteRequest: Send + Sync {
    /// Identifies the request and cancels a selection still waiting in a queue.
    fn engine_context(&self) -> Arc<dyn AsyncEngineContext>;

    /// Prompt tokens, with per-block multimodal info when the prompt carries media.
    fn routing_tokens(&self) -> (&[TokenIdType], Option<&[Option<BlockExtraInfo>]>);

    /// Caller hints: worker pins, priority, LoRA, cache namespace, constraints
    /// and allowed workers.
    fn routing_hints(&self) -> Option<&RoutingHints> {
        None
    }

    fn router_config_override(&self) -> Option<&RouterConfigOverride> {
        None
    }

    /// The scheduling policy class the request queues under.
    fn policy_class(&self) -> Option<&str> {
        None
    }

    /// Agent session metadata, used for session-aware scoring and subagent
    /// affinity groups.
    fn agent_context(&self) -> Option<&AgentContext> {
        None
    }

    /// The session-affinity id, or an error when the request carries a malformed one.
    fn session_affinity_id(&self) -> Result<Option<Arc<SessionAffinityId>>, Error> {
        Ok(None)
    }

    /// Frontend retry state: workers a migration excludes and a parked
    /// classifier lifecycle.
    fn migration_state(&self) -> Option<&MigrationState> {
        None
    }

    /// Whether remote prefill staged KV that only this request's decode leg can release.
    fn staged_kv_cleanup(&self) -> bool {
        false
    }
}

/// A route request made only of an id and prompt tokens, for tests.
#[cfg(test)]
pub(crate) struct TestRouteRequest {
    context: Arc<dyn AsyncEngineContext>,
    token_ids: Vec<TokenIdType>,
}

#[cfg(test)]
impl TestRouteRequest {
    pub(crate) fn new(id: &str, token_ids: &[TokenIdType]) -> Self {
        Self {
            context: Arc::new(dynamo_runtime::pipeline::context::Controller::new(
                id.to_string(),
            )),
            token_ids: token_ids.to_vec(),
        }
    }
}

#[cfg(test)]
impl RouteRequest for TestRouteRequest {
    fn engine_context(&self) -> Arc<dyn AsyncEngineContext> {
        Arc::clone(&self.context)
    }

    fn routing_tokens(&self) -> (&[TokenIdType], Option<&[Option<BlockExtraInfo>]>) {
        (&self.token_ids, None)
    }
}

impl RouteRequest for SingleIn<PreprocessedRequest> {
    fn engine_context(&self) -> Arc<dyn AsyncEngineContext> {
        self.context()
    }

    fn routing_tokens(&self) -> (&[TokenIdType], Option<&[Option<BlockExtraInfo>]>) {
        self.content().block_mm_routing_info()
    }

    fn routing_hints(&self) -> Option<&RoutingHints> {
        self.content().routing.as_ref()
    }

    fn router_config_override(&self) -> Option<&RouterConfigOverride> {
        self.content().router_config_override.as_ref()
    }

    fn policy_class(&self) -> Option<&str> {
        self.metadata().get("policy-class").map(String::as_str)
    }

    fn agent_context(&self) -> Option<&AgentContext> {
        self.content().agent_context.as_ref()
    }

    fn session_affinity_id(&self) -> Result<Option<Arc<SessionAffinityId>>, Error> {
        affinity_id(self)
    }

    fn migration_state(&self) -> Option<&MigrationState> {
        self.content().migration_state.as_ref()
    }

    fn staged_kv_cleanup(&self) -> bool {
        self.content().staged_kv_cleanup
    }
}
