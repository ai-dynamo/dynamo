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
/// Planning a route ([`RoutingHost::plan_route`](super::RoutingHost::plan_route),
/// [`StageRouter::schedule`](super::stage_router::StageRouter::schedule)) takes
/// `&dyn RouteRequest`, so a host that never builds the frontend's
/// [`PreprocessedRequest`], such as the inference-gateway EPP, selects through
/// the same path. Dispatch still takes a `PreprocessedRequest`.
///
/// Every input has a default meaning "not supplied". When you add a selection
/// input, add a method here, implement it for `SingleIn<PreprocessedRequest>`,
/// and decide whether the EPP must supply it too.
pub trait RouteRequest: Send + Sync {
    /// Identifies the request, and stops a selection still waiting in a queue
    /// when the request is cancelled. A booking is kept under this id.
    fn engine_context(&self) -> Arc<dyn AsyncEngineContext>;

    /// Prompt tokens, with per-block multimodal info when the prompt carries media.
    fn routing_tokens(&self) -> (&[TokenIdType], Option<&[Option<BlockExtraInfo>]>);

    /// Caller hints: worker pins, LoRA, cache namespace, queue priorities,
    /// expected output length, allowed workers and routing constraints.
    fn routing_hints(&self) -> Option<&RoutingHints> {
        None
    }

    /// Per-request overrides of the KV scoring configuration.
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

    /// Frontend retry state: the workers a migration excludes, and a parked
    /// classifier lifecycle.
    fn migration_state(&self) -> Option<&MigrationState> {
        None
    }

    /// Whether remote prefill staged KV that only this request's decode leg can release.
    fn staged_kv_cleanup(&self) -> bool {
        false
    }

    /// Whether the caller only asks which worker would be chosen. The
    /// built-in plane then reads the session-affinity binding without holding
    /// it. A KV plan books either way.
    fn is_query_only(&self) -> bool {
        false
    }

    /// The frontend request, for the one input only it carries: the
    /// multimodal cache keys device-aware routing reads. `None` for a host
    /// that has no `PreprocessedRequest`; device-aware routing then weighs
    /// device types without cache hits.
    fn preprocessed(&self) -> Option<&PreprocessedRequest> {
        None
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

    fn is_query_only(&self) -> bool {
        self.get_annotation_value("query_instance_id").is_some()
    }

    fn preprocessed(&self) -> Option<&PreprocessedRequest> {
        Some(self.content())
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
