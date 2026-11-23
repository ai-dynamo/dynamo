// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The fields of a routed request that do not depend on how its prompt was rendered.
//!
//! A chat processor turns a request into tokens: Dynamo's own preprocessor, or a Python one
//! the frontend was given. The fields here come from the model card, the request and the
//! request's context instead, so every processor's requests get them from this one place.

use std::sync::Arc;
use std::time::Duration;

use dynamo_renderer::OAIChatLikeRequest;
use dynamo_runtime::pipeline::Context;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;

use super::LORA_NAME_CONTEXT_KEY;
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::extensions::{
    AGENT_CONTEXT_CONTEXT_KEY, AgentContext, AgentHints, NvExtProvider,
    SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId, UPSTREAM_ELAPSED_CONTEXT_KEY,
    request_cache_salt, routing_constraints_to_kv,
};
use crate::protocols::common::preprocessor::{
    PreprocessedRequest, PreprocessedRequestBuilder, RoutingHints,
};
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

/// What every request to one model carries, read from its model card.
///
/// It needs neither a tokenizer nor a chat template, so a frontend whose chat processor
/// renders prompts itself can build one.
#[derive(Debug, Clone)]
pub struct RequestEnvelope {
    mdcsum: String,
    lora_name: Option<String>,
}

impl RequestEnvelope {
    pub fn from_card(card: &ModelDeploymentCard) -> Self {
        Self {
            mdcsum: card.mdcsum().to_string(),
            lora_name: card.lora.as_ref().map(|lora| lora.name.clone()),
        }
    }

    pub(crate) fn mdcsum(&self) -> &str {
        &self.mdcsum
    }

    /// Write the routing and bookkeeping fields of `request` onto `builder`.
    ///
    /// `lora_name_override` names the adapter when the model card does not.
    pub(crate) fn apply_request_facts<R>(
        &self,
        request: &R,
        lora_name_override: Option<String>,
        builder: &mut PreprocessedRequestBuilder,
    ) where
        R: OAIChatLikeRequest + AnnotationsProvider + NvExtProvider,
    {
        builder.annotations(request.annotations().unwrap_or_default());
        builder.mdc_sum(Some(self.mdcsum.clone()));
        let lora_name = self.lora_name.clone().or(lora_name_override);
        let cache_namespace = request_cache_salt(request).map(str::to_owned);

        if let Some(nvext) = request.nvext() {
            let hints = nvext.agent_hints.as_ref();
            let (priority_jump, strict_priority, priority) = routing_priorities(hints);
            builder.request_timestamp_ms(nvext.request_timestamp_ms);
            builder.routing(Some(RoutingHints {
                backend_instance_id: nvext.backend_instance_id,
                prefill_worker_id: nvext.prefill_worker_id,
                decode_worker_id: nvext.decode_worker_id,
                dp_rank: nvext.dp_rank,
                prefill_dp_rank: nvext.prefill_dp_rank,
                expected_output_tokens: hints.and_then(|h| h.osl),
                priority_jump,
                strict_priority,
                priority,
                lora_name,
                base_model_name: None,
                lora_source_uri: None,
                lora_resolution_version: None,
                cache_namespace,
                allowed_worker_ids: None,
                routing_constraints: nvext
                    .routing_constraints
                    .clone()
                    .map(routing_constraints_to_kv),
            }));
        } else if lora_name.is_some() || cache_namespace.is_some() {
            // LoRA and a legacy top-level `cache_salt` route without nvext too.
            builder.routing(Some(RoutingHints {
                lora_name,
                cache_namespace,
                ..Default::default()
            }));
        }

        // e.g. `use_audio_in_video`
        builder.mm_processor_kwargs(request.mm_processor_kwargs().cloned());
    }

    /// Start a chat request whose processor renders the prompt itself.
    pub fn begin_chat(&self, request: &Context<NvCreateChatCompletionRequest>) -> RequestScope {
        RequestScope::from_context(request)
    }
}

fn routing_priorities(hints: Option<&AgentHints>) -> (Option<f64>, Option<u32>, Option<i32>) {
    let priority_jump = hints.and_then(|h| {
        h.priority
            .map(|priority| priority.max(0) as f64)
            .or(h.latency_sensitivity)
    });
    let strict_priority = hints.and_then(|h| h.strict_priority);
    let priority = hints.and_then(|h| h.priority);
    (priority_jump, strict_priority, priority)
}

/// What the HTTP layer put on one request's context, read when the request arrives.
#[derive(Debug, Clone, Default)]
pub struct RequestScope {
    agent_context: Option<Arc<AgentContext>>,
    session_affinity: Option<Arc<SessionAffinityId>>,
    upstream_elapsed: Option<Duration>,
    lora_name: Option<Arc<String>>,
}

impl RequestScope {
    pub fn from_context<T: Send + Sync + 'static>(context: &Context<T>) -> Self {
        Self {
            agent_context: context
                .get_optional(AGENT_CONTEXT_CONTEXT_KEY)
                .ok()
                .flatten(),
            session_affinity: context
                .get_optional(SESSION_AFFINITY_CONTEXT_KEY)
                .ok()
                .flatten(),
            upstream_elapsed: context
                .get_optional::<Duration>(UPSTREAM_ELAPSED_CONTEXT_KEY)
                .ok()
                .flatten()
                .map(|elapsed| *elapsed),
            lora_name: context.get_optional(LORA_NAME_CONTEXT_KEY).ok().flatten(),
        }
    }

    /// The LoRA adapter the request was resolved to, when discovery named one.
    pub(crate) fn lora_name(&self) -> Option<String> {
        self.lora_name.as_deref().cloned()
    }

    /// Start `tracker`'s arrival age from the time a gateway reports it held the request, so a
    /// backend that orders admission by arrival counts the wait in front of this process too.
    pub fn apply_upstream_elapsed(&self, tracker: &RequestTracker) {
        if let Some(upstream_elapsed) = self.upstream_elapsed {
            tracker.set_upstream_elapsed(upstream_elapsed);
        }
    }

    /// Give `request` the fields its processor left unset. A value the processor set is kept.
    pub fn complete_request(&self, request: &mut PreprocessedRequest) {
        // No `..`: a new field does not compile until it is given an owner here.
        let PreprocessedRequest {
            // Filled here when unset.
            agent_context,
            image_cache_scope,

            // The processor's: its rendering, tokens and sampling decide these.
            model: _,
            token_ids: _,
            prompt_embeds: _,
            multi_modal_data: _,
            multi_modal_uuids: _,
            multi_modal_order: _,
            mm_routing_info: _,
            stop_conditions: _,
            sampling_options: _,
            output_options: _,
            eos_token_ids: _,
            mdc_sum: _,
            annotations: _,
            routing: _,
            extra_args: _,
            require_reasoning: _,
            router: _,
            mm_processor_kwargs: _,
            media_io_kwargs: _,
            request_timestamp_ms: _,

            // Set after this point, by the router and the operators around it.
            migration_state: _,
            staged_kv_cleanup: _,
            router_config_override: _,
            prefill_result: _,
            encoder_result: _,
            migration_link: _,
            jail_seed: _,
            bootstrap_info: _,
            kv_hint: _,
            elapsed_since_arrival_ms: _,
            tracker: _,
            is_probe: _,
        } = request;

        if agent_context.is_none() {
            *agent_context = self.agent_context.as_deref().cloned();
        }
        // Workers key their image caches by the session the request is routed under.
        if image_cache_scope.is_none() {
            *image_cache_scope = self
                .session_affinity
                .as_deref()
                .map(|session| session.as_str().to_owned());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

    fn rendered_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    fn agent_context(session_id: &str) -> AgentContext {
        AgentContext {
            session_id: session_id.to_string(),
            parent_session_id: None,
            session_final: None,
            agent_headers: Default::default(),
            input_trigger: None,
        }
    }

    #[test]
    fn routing_priorities_keep_strict_tier_independent() {
        let hints = AgentHints {
            priority: Some(-3),
            strict_priority: Some(7),
            ..Default::default()
        };

        assert_eq!(
            routing_priorities(Some(&hints)),
            (Some(0.0), Some(7), Some(-3))
        );
        assert_eq!(routing_priorities(None), (None, None, None));
    }

    /// A model card with no tokenizer or chat template is enough to start a chat request, and
    /// the request its processor rendered gets the session the HTTP layer saw.
    #[test]
    fn chat_request_is_completed_without_a_tokenizer() {
        let envelope = RequestEnvelope::from_card(&ModelDeploymentCard::with_name_only("m"));

        let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .unwrap();
        let mut request = Context::new(request);
        request.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context("from-headers"));
        request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("routing-session"),
        );
        let scope = envelope.begin_chat(&request);

        let mut rendered = rendered_request();
        scope.complete_request(&mut rendered);
        assert_eq!(
            rendered.agent_context.as_ref().unwrap().session_id,
            "from-headers"
        );
        assert_eq!(
            rendered.image_cache_scope.as_deref(),
            Some("routing-session")
        );

        // What the processor set stays.
        let mut rendered = rendered_request();
        rendered.agent_context = Some(agent_context("from-processor"));
        rendered.image_cache_scope = Some("processor-scope".to_string());
        scope.complete_request(&mut rendered);
        assert_eq!(
            rendered.agent_context.as_ref().unwrap().session_id,
            "from-processor"
        );
        assert_eq!(
            rendered.image_cache_scope.as_deref(),
            Some("processor-scope")
        );
    }
}
