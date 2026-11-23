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

use super::{LORA_NAME_CONTEXT_KEY, OpenAIPreprocessor};
use crate::local_model::runtime_config::{TOKEN_BUDGET_RUNTIME_KEY, TokenBudget};
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::extensions::{
    AGENT_CONTEXT_CONTEXT_KEY, AgentContext, AgentHints, NvExtProvider, RouterParams,
    SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId, UPSTREAM_ELAPSED_CONTEXT_KEY,
    request_cache_salt, routing_constraints_to_kv,
};
use crate::protocols::common::preprocessor::{
    PreprocessedRequest, PreprocessedRequestBuilder, RoutingHints,
};
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::common::{SamplingOptions, StopConditions, StopConditionsProvider};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::sampling_scalars;

type JsonObject = serde_json::Map<String, serde_json::Value>;

/// What every request to one model carries, read from its model card.
///
/// It needs neither a tokenizer nor a chat template, so a frontend whose chat processor
/// renders prompts itself can build one.
#[derive(Debug, Clone)]
pub struct RequestEnvelope {
    mdcsum: String,
    lora_name: Option<String>,
    token_budget: Option<TokenBudget>,
}

impl RequestEnvelope {
    pub fn from_card(card: &ModelDeploymentCard) -> Self {
        let token_budget = card
            .runtime_config
            .get_engine_specific::<TokenBudget>(TOKEN_BUDGET_RUNTIME_KEY)
            .unwrap_or_else(|error| {
                tracing::warn!(
                    %error,
                    key = TOKEN_BUDGET_RUNTIME_KEY,
                    "Ignoring invalid runtime metadata; token overflow handling will be delegated to the backend"
                );
                None
            });
        Self {
            mdcsum: card.mdcsum().to_string(),
            lora_name: card.lora.as_ref().map(|lora| lora.name.clone()),
            token_budget,
        }
    }

    pub(crate) fn mdcsum(&self) -> &str {
        &self.mdcsum
    }

    /// The engine-published request-token admission policy.
    pub(crate) fn token_budget(&self) -> Option<TokenBudget> {
        self.token_budget
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
        builder.routing(routing_hints(request, lora_name));
        builder.request_timestamp_ms(request.nvext().and_then(|nvext| nvext.request_timestamp_ms));
        // e.g. `use_audio_in_video`
        builder.mm_processor_kwargs(request.mm_processor_kwargs().cloned());
    }

    /// Start a chat request whose processor renders the prompt itself.
    ///
    /// The scope keeps what the request and its context say about routing, sampling and
    /// limits, for [`RequestScope::complete_request`] to give the request the processor builds.
    pub fn begin_chat(&self, request: &Context<NvCreateChatCompletionRequest>) -> RequestScope {
        let mut scope = RequestScope::from_context(request);
        let chat: &NvCreateChatCompletionRequest = request;
        let lora_name = self.lora_name.clone().or_else(|| scope.lora_name());
        scope.request = Some(Arc::new(RequestFacts {
            mdc_sum: self.mdcsum.clone(),
            routing: routing_hints(chat, lora_name),
            router: chat.nvext().and_then(|nvext| nvext.router.clone()),
            request_timestamp_ms: chat.nvext().and_then(|nvext| nvext.request_timestamp_ms),
            // A request these reject is left to the processor's own validation.
            stop_conditions: chat.extract_stop_conditions().ok(),
            sampling_options: sampling_scalars(chat).ok(),
            // `token_in` says the tokens came from the client; only the processor knows.
            nvext_args: nvext_passthrough_args(chat).map(|args| {
                args.into_iter()
                    .filter(|(key, _)| key != "token_in")
                    .collect()
            }),
            sampling_args: sampling_passthrough_args(chat),
            token_budget: self.token_budget,
        }));
        scope
    }
}

/// Where the router should send a request, from its `nvext` and `lora_name`.
fn routing_hints<R: NvExtProvider>(request: &R, lora_name: Option<String>) -> Option<RoutingHints> {
    let cache_namespace = request_cache_salt(request).map(str::to_owned);
    let Some(nvext) = request.nvext() else {
        // LoRA and a legacy top-level `cache_salt` route without nvext too.
        return (lora_name.is_some() || cache_namespace.is_some()).then(|| RoutingHints {
            lora_name,
            cache_namespace,
            ..Default::default()
        });
    };
    let hints = nvext.agent_hints.as_ref();
    let (priority_jump, strict_priority, priority) = routing_priorities(hints);
    Some(RoutingHints {
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
    })
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

/// The `nvext` fields a worker reads from `extra_args.nvext`.
pub(super) fn nvext_passthrough_args<R: NvExtProvider>(request: &R) -> Option<JsonObject> {
    let mut nvext_passthrough = serde_json::Map::new();

    if let Some(nvext) = request.nvext() {
        if let Some(ref fields) = nvext.extra_fields {
            nvext_passthrough.insert("extra_fields".to_string(), serde_json::json!(fields));
        }
        if let Some(ref metadata_upload) = nvext.metadata_upload {
            nvext_passthrough.insert(
                "metadata_upload".to_string(),
                serde_json::json!(metadata_upload),
            );
        }
        if nvext.token_data.is_some() {
            nvext_passthrough.insert("token_in".to_string(), serde_json::Value::Bool(true));
        }
    }

    if let Some(salt) = request_cache_salt(request) {
        nvext_passthrough.insert("cache_salt".to_string(), serde_json::json!(salt));
    }

    (!nvext_passthrough.is_empty()).then_some(nvext_passthrough)
}

/// The engine sampling fields a worker reads from `extra_args.sampling_options`.
pub(super) fn sampling_passthrough_args<R: NvExtProvider>(request: &R) -> Option<JsonObject> {
    let mut sampling_passthrough = serde_json::Map::new();

    if let Some(fields) = request.unsupported_fields() {
        for key in [
            "detokenize",
            "allowed_token_ids",
            "bad_words_token_ids",
            "logprob_token_ids",
            "vllm_xargs",
        ] {
            if let Some(value) = fields.get(key) {
                sampling_passthrough.insert(key.to_string(), value.clone());
            }
        }
    }

    (!sampling_passthrough.is_empty()).then_some(sampling_passthrough)
}

/// What one request says, apart from its prompt, about the request to route.
#[derive(Debug)]
struct RequestFacts {
    mdc_sum: String,
    routing: Option<RoutingHints>,
    router: Option<RouterParams>,
    request_timestamp_ms: Option<f64>,
    stop_conditions: Option<StopConditions>,
    sampling_options: Option<SamplingOptions>,
    nvext_args: Option<JsonObject>,
    sampling_args: Option<JsonObject>,
    token_budget: Option<TokenBudget>,
}

/// One request's engine-agnostic facts, read when the request arrives.
#[derive(Debug, Clone, Default)]
pub struct RequestScope {
    agent_context: Option<Arc<AgentContext>>,
    session_affinity: Option<Arc<SessionAffinityId>>,
    upstream_elapsed: Option<Duration>,
    lora_name: Option<Arc<String>>,
    /// Set by [`RequestEnvelope::begin_chat`]. Dynamo's preprocessor writes these fields on
    /// the request builder itself.
    request: Option<Arc<RequestFacts>>,
}

/// Take `value` where the processor left `slot` unset.
fn fill<T: Clone>(slot: &mut Option<T>, value: &Option<T>) {
    if slot.is_none() {
        slot.clone_from(value);
    }
}

/// Fill the listed fields of `$dst` from `$src`. The pattern has no `..`, so a field added to
/// the struct does not compile until it is listed, as filled or after `keep`.
macro_rules! fill_unset {
    ($dst:expr, $src:expr, $ty:ident { $($field:ident),* $(,)? } keep { $($kept:ident),* $(,)? }) => {{
        let $ty { $($field,)* $($kept: _,)* } = $src;
        $( fill(&mut $dst.$field, $field); )*
    }};
}

impl RequestScope {
    /// What the HTTP layer put on the request's context.
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
            request: None,
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
            mdc_sum,
            routing,
            router,
            request_timestamp_ms,
            // Filled field by field, below.
            stop_conditions,
            sampling_options,
            extra_args,

            // The processor's: its rendering, tokens and parsers decide these.
            model: _,
            token_ids: _,
            prompt_embeds: _,
            multi_modal_data: _,
            multi_modal_uuids: _,
            multi_modal_order: _,
            mm_routing_info: _,
            output_options: _,
            eos_token_ids: _,
            annotations: _,
            require_reasoning: _,
            mm_processor_kwargs: _,
            media_io_kwargs: _,

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

        let Some(facts) = self.request.as_deref() else {
            return;
        };
        if mdc_sum.is_none() {
            *mdc_sum = Some(facts.mdc_sum.clone());
        }
        fill(router, &facts.router);
        fill(request_timestamp_ms, &facts.request_timestamp_ms);

        match (routing.as_mut(), &facts.routing) {
            (None, hints) => routing.clone_from(hints),
            (Some(routing), Some(hints)) => fill_unset!(
                routing,
                hints,
                RoutingHints {
                    backend_instance_id,
                    prefill_worker_id,
                    decode_worker_id,
                    dp_rank,
                    prefill_dp_rank,
                    expected_output_tokens,
                    lora_name,
                    base_model_name,
                    lora_source_uri,
                    lora_resolution_version,
                    cache_namespace,
                    priority_jump,
                    strict_priority,
                    priority,
                    allowed_worker_ids,
                    routing_constraints,
                } keep {}
            ),
            (Some(_), None) => {}
        }

        // Stop strings and stop tokens are matched where the output is detokenized, so they
        // stay with the processor.
        if let Some(stop) = &facts.stop_conditions {
            fill_unset!(
                stop_conditions,
                stop,
                StopConditions {
                    max_tokens,
                    min_tokens,
                    ignore_eos,
                    max_thinking_tokens,
                } keep {
                    stop,
                    stop_token_ids,
                    stop_token_ids_visible,
                    stop_token_ids_hidden,
                }
            );
        }
        // A grammar depends on the processor's tool-call and reasoning parsers.
        if let Some(sampling) = &facts.sampling_options {
            fill_unset!(
                sampling_options,
                sampling,
                SamplingOptions {
                    n,
                    best_of,
                    presence_penalty,
                    frequency_penalty,
                    repetition_penalty,
                    temperature,
                    top_p,
                    top_k,
                    min_p,
                    seed,
                    logit_bias,
                } keep {
                    use_beam_search,
                    length_penalty,
                    include_stop_str_in_output,
                    guided_decoding,
                }
            );
        }

        fill_extra_args(extra_args, "nvext", facts.nvext_args.as_ref());
        fill_extra_args(extra_args, "sampling_options", facts.sampling_args.as_ref());
    }

    /// Apply the engine-published token budget to the prompt the processor rendered and the
    /// output length the client asked for.
    pub fn validate_token_budget(&self, request: &PreprocessedRequest) -> anyhow::Result<()> {
        let Some(facts) = self.request.as_deref() else {
            return Ok(());
        };
        if request.prompt_embeds.is_some() {
            return Ok(());
        }
        let exact_prompt_len = OpenAIPreprocessor::exact_prompt_len(
            request
                .mm_routing_info
                .as_ref()
                .map(|mm| mm.expanded_prompt_len),
            request.multi_modal_data.as_ref(),
            request.token_ids.len(),
        );
        let Some(prompt_len) = exact_prompt_len else {
            return Ok(());
        };
        // The client's limit, not the processor's: a processor may have replaced an omitted
        // one with the room it computed was left.
        let max_tokens = facts
            .stop_conditions
            .as_ref()
            .and_then(|stop| stop.max_tokens);
        OpenAIPreprocessor::validate_requested_token_budget(
            prompt_len,
            max_tokens,
            facts.token_budget.as_ref(),
        )
    }
}

/// Add the entries of `args` that `extra_args[group]` does not have.
fn fill_extra_args(
    extra_args: &mut Option<serde_json::Value>,
    group: &str,
    args: Option<&JsonObject>,
) {
    let Some(args) = args else {
        return;
    };
    let extra_args = extra_args.get_or_insert_with(|| serde_json::json!({}));
    let Some(extra_args) = extra_args.as_object_mut() else {
        return;
    };
    let entry = extra_args
        .entry(group)
        .or_insert_with(|| serde_json::json!({}));
    let Some(entry) = entry.as_object_mut() else {
        return;
    };
    for (key, value) in args {
        if !entry.contains_key(key) {
            entry.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::OutputOptions;

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

    fn chat_request(body: serde_json::Value) -> Context<NvCreateChatCompletionRequest> {
        let mut body = body;
        body["model"] = "m".into();
        body["messages"] = serde_json::json!([{"role": "user", "content": "hi"}]);
        Context::new(serde_json::from_value(body).unwrap())
    }

    fn envelope(token_budget: Option<TokenBudget>) -> RequestEnvelope {
        let mut envelope = RequestEnvelope::from_card(&ModelDeploymentCard::with_name_only("m"));
        envelope.token_budget = token_budget;
        envelope
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

    /// A model card with no tokenizer or chat template is enough to complete the request a
    /// chat processor rendered, from the request and its context.
    #[test]
    fn rendered_request_gets_what_the_processor_left_unset() {
        let mut request = chat_request(serde_json::json!({
            "seed": 7,
            "logit_bias": {"5": 1.5},
            "max_tokens": 64,
            "response_format": {"type": "json_object"},
            // Conflicting grammars are the processor's to reject.
            "guided_regex": "a+",
            "guided_choice": ["x"],
            "logprobs": true,
            "stop": ["END"],
            "nvext": {
                "backend_instance_id": 42,
                "agent_hints": {"priority": 3, "osl": 128},
                "cache_salt": "tenant-a",
                "extra_fields": ["worker_id"],
                "annotations": ["query_instance_id"],
            },
        }));
        request.insert(
            AGENT_CONTEXT_CONTEXT_KEY,
            AgentContext {
                session_id: "from-headers".to_string(),
                parent_session_id: None,
                session_final: None,
                agent_headers: Default::default(),
                input_trigger: None,
            },
        );
        request.insert(
            SESSION_AFFINITY_CONTEXT_KEY,
            SessionAffinityId::new("routing-session"),
        );
        let scope = envelope(None).begin_chat(&request);

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
        assert!(rendered.mdc_sum.is_some());
        let routing = rendered.routing.as_ref().unwrap();
        assert_eq!(routing.backend_instance_id, Some(42));
        assert_eq!(routing.priority, Some(3));
        assert_eq!(routing.expected_output_tokens, Some(128));
        assert_eq!(routing.cache_namespace.as_deref(), Some("tenant-a"));
        assert_eq!(rendered.sampling_options.seed, Some(7));
        assert_eq!(
            rendered.sampling_options.logit_bias.as_ref().unwrap()[&5],
            1.5
        );
        assert_eq!(rendered.stop_conditions.max_tokens, Some(64));
        assert_eq!(
            rendered.extra_args.as_ref().unwrap()["nvext"],
            serde_json::json!({"extra_fields": ["worker_id"], "cache_salt": "tenant-a"})
        );

        // The processor's own: a grammar, logprobs, stop strings and annotations are never
        // filled, even though the request has them.
        assert!(rendered.sampling_options.guided_decoding.is_none());
        assert!(rendered.output_options.logprobs.is_none());
        assert!(rendered.stop_conditions.stop.is_none());
        assert!(rendered.annotations.is_empty());
    }

    #[test]
    fn values_the_processor_set_are_kept() {
        let scope = envelope(None).begin_chat(&chat_request(serde_json::json!({
            "seed": 7,
            "temperature": 0.2,
            "max_tokens": 64,
            "nvext": {"backend_instance_id": 42, "cache_salt": "tenant-a"},
        })));

        let mut rendered = rendered_request();
        rendered.sampling_options.temperature = Some(1.0);
        rendered.stop_conditions.max_tokens = Some(900);
        rendered.routing = Some(RoutingHints {
            backend_instance_id: Some(7),
            ..Default::default()
        });
        rendered.extra_args = Some(serde_json::json!({
            "nvext": {"cache_salt": "from-processor"},
            "reasoning_ended": true,
        }));
        scope.complete_request(&mut rendered);

        assert_eq!(rendered.sampling_options.temperature, Some(1.0));
        assert_eq!(rendered.sampling_options.seed, Some(7));
        assert_eq!(rendered.stop_conditions.max_tokens, Some(900));
        let routing = rendered.routing.as_ref().unwrap();
        assert_eq!(routing.backend_instance_id, Some(7));
        assert_eq!(routing.cache_namespace.as_deref(), Some("tenant-a"));
        assert_eq!(
            rendered.extra_args.unwrap(),
            serde_json::json!({
                "nvext": {"cache_salt": "from-processor"},
                "reasoning_ended": true,
            })
        );
    }

    /// The budget is checked against the client's `max_tokens`. A processor that replaced an
    /// omitted one with the room left in the context must not trip it.
    #[test]
    fn token_budget_uses_the_clients_output_limit() {
        let budget = TokenBudget {
            combined_limit: 10,
            reject_prompt_overflow: true,
            reject_total_overflow: true,
        };
        let envelope = envelope(Some(budget));
        let mut rendered = rendered_request();
        rendered.stop_conditions.max_tokens = Some(1_000);

        let omitted = envelope.begin_chat(&chat_request(serde_json::json!({})));
        omitted.validate_token_budget(&rendered).unwrap();

        let too_long = envelope.begin_chat(&chat_request(serde_json::json!({"max_tokens": 8})));
        let error = too_long.validate_token_budget(&rendered).unwrap_err();
        assert!(error.to_string().contains("asks for 8 output tokens"));
    }
}
