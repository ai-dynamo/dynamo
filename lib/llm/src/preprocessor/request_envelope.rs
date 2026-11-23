// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The fields of a routed request that do not depend on how its prompt was rendered.
//!
//! A chat processor turns a request into tokens: Dynamo's own preprocessor, or a Python one
//! the frontend was given. The fields here come from the model card, the request and the
//! request's context instead, so every processor's requests get them from this one place.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use dynamo_runtime::pipeline::Context;

use super::{LORA_NAME_CONTEXT_KEY, OpenAIPreprocessor};
use crate::local_model::runtime_config::{TOKEN_BUDGET_RUNTIME_KEY, TokenBudget};
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::extensions::{
    AGENT_CONTEXT_CONTEXT_KEY, AgentContext, AgentHints, NvExtProvider, RouterParams,
    SESSION_AFFINITY_CONTEXT_KEY, SessionAffinityId, UPSTREAM_ELAPSED_CONTEXT_KEY,
    request_cache_salt, routing_constraints_to_kv,
};
use crate::protocols::common::llm_backend::LLMEngineOutput;
use crate::protocols::common::preprocessor::{PreprocessedRequest, RoutingHints};
use crate::protocols::common::timing::RequestTracker;
use crate::protocols::common::{
    SamplingOptions, SamplingOptionsProvider, StopConditions, StopConditionsProvider,
};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

type JsonObject = serde_json::Map<String, serde_json::Value>;

/// What every request to one model carries, read from its model card.
///
/// It needs neither a tokenizer nor a chat template, so a frontend whose chat processor
/// renders prompts itself can build one.
#[derive(Debug, Clone)]
pub struct RequestEnvelope {
    pub(super) mdcsum: String,
    pub(super) lora_name: Option<String>,
    /// The engine-published request-token admission policy.
    pub(super) token_budget: Option<TokenBudget>,
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
            // The grammar is the processor's, and a schema can be large, so it is not kept
            // for the life of the request.
            sampling_options: chat.extract_sampling_options().ok().map(|sampling| {
                SamplingOptions {
                    guided_decoding: None,
                    ..sampling
                }
            }),
            // `token_in` says the tokens came from the client; only the processor knows.
            nvext_args: nvext_passthrough_args(chat)
                .map(|args| {
                    args.into_iter()
                        .filter(|(key, _)| key != "token_in")
                        .collect::<JsonObject>()
                })
                .filter(|args| !args.is_empty()),
            sampling_args: sampling_passthrough_args(chat),
            wants_prompt_token_ids: chat
                .nvext()
                .and_then(|nvext| nvext.extra_fields.as_ref())
                .is_some_and(|fields| fields.iter().any(|field| field == "prompt_token_ids")),
        }));
        scope
    }

    /// Apply the engine-published token budget to the request as it will reach the worker.
    pub fn validate_token_budget(&self, request: &PreprocessedRequest) -> anyhow::Result<()> {
        OpenAIPreprocessor::validate_preprocessed_token_budget(request, self.token_budget.as_ref())
            .map(drop)
    }
}

/// Where the router should send a request, from its `nvext` and `lora_name`.
pub(super) fn routing_hints<R: NvExtProvider>(
    request: &R,
    lora_name: Option<String>,
) -> Option<RoutingHints> {
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

/// The part of a [`RequestScope`] read from the chat request itself.
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
    wants_prompt_token_ids: bool,
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
    /// The tracker the response is built from, for the routed request to share.
    tracker: Option<Arc<RequestTracker>>,
    /// Length of the prompt the processor rendered; zero until it routes a request.
    prompt_tokens: Arc<AtomicU32>,
}

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
            tracker: None,
            prompt_tokens: Arc::default(),
        }
    }

    /// Have the routed request record its timing and worker on `tracker`, the one the
    /// response reads them from.
    pub fn share_tracker(&mut self, tracker: Arc<RequestTracker>) {
        self.apply_upstream_elapsed(Some(&tracker));
        self.tracker = Some(tracker);
    }

    /// Length of the prompt the processor routed, once it has.
    pub fn prompt_tokens(&self) -> Option<u32> {
        match self.prompt_tokens.load(Ordering::Relaxed) {
            0 => None,
            prompt_tokens => Some(prompt_tokens),
        }
    }

    /// Note the prompt the processor rendered, for the response's usage and, when the client
    /// asked for them, its `prompt_token_ids`.
    pub fn record_prompt(&self, request: &PreprocessedRequest) {
        let prompt_tokens = u32::try_from(request.token_ids.len()).unwrap_or(u32::MAX);
        self.prompt_tokens.store(prompt_tokens, Ordering::Relaxed);
        if let Some(tracker) = &self.tracker
            && self
                .request
                .as_deref()
                .is_some_and(|facts| facts.wants_prompt_token_ids)
        {
            tracker.set_prompt_token_ids(request.token_ids.to_vec());
        }
    }

    /// Take what the router reports about a request from an engine output before a chat
    /// processor sees it. A standalone router sends its timing and worker this way.
    pub fn observe_engine_output(&self, output: &LLMEngineOutput) {
        if let Some(tracker) = &self.tracker
            && let Some(routing_data) = output.routing_data.as_deref()
        {
            tracker.ingest_routing_data(routing_data.clone());
        }
    }

    /// The LoRA adapter the request was resolved to, when discovery named one.
    pub(crate) fn lora_name(&self) -> Option<String> {
        self.lora_name.as_deref().cloned()
    }

    /// Start `tracker`'s arrival age from the time a gateway reports it held the request, so a
    /// backend that orders admission by arrival counts the wait in front of this process too.
    pub fn apply_upstream_elapsed(&self, tracker: Option<&RequestTracker>) {
        if let Some(tracker) = tracker
            && let Some(upstream_elapsed) = self.upstream_elapsed
        {
            tracker.set_upstream_elapsed(upstream_elapsed);
        }
    }

    /// Give `request` the fields its processor left unset. A value the processor set is kept.
    /// Every field is named below, so a new one does not compile until it has an owner.
    pub fn complete_request(&self, request: &mut PreprocessedRequest) {
        let PreprocessedRequest {
            // Filled here when unset.
            agent_context,
            image_cache_scope,
            mdc_sum,
            routing,
            router,
            request_timestamp_ms,
            tracker,
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
            is_probe: _,
        } = request;

        if agent_context.is_none() {
            *agent_context = self.agent_context.as_deref().cloned();
        }
        fill(tracker, &self.tracker);
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
        if let Some(client) = &facts.router {
            let router = router.get_or_insert_default();
            fill_unset!(router, client, RouterParams { ttft_target, itl_target } keep {});
        }
        fill(request_timestamp_ms, &facts.request_timestamp_ms);

        // The router and the worker must use one cache salt, so a salt the processor set in
        // either place goes to both. The client's fills both below when it set neither.
        let processor_salt = routing
            .as_ref()
            .and_then(|routing| routing.cache_namespace.clone())
            .or_else(|| {
                let salt = extra_args.as_ref()?.get("nvext")?.get("cache_salt")?;
                salt.as_str().map(str::to_owned)
            });
        if let Some(salt) = processor_salt {
            routing.get_or_insert_default().cache_namespace = Some(salt.clone());
            if let Some(nvext) = extra_args_group(extra_args, "nvext") {
                nvext.insert("cache_salt".to_string(), salt.into());
            }
        }

        if let Some(hints) = &facts.routing {
            let routing = routing.get_or_insert_default();
            fill_unset!(
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
            );
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
        // A grammar depends on the processor's tool-call and reasoning parsers. An engine
        // turns `top_k` and `min_p` off for greedy sampling and from its generation config,
        // so an unset one is the processor's decision, not a gap.
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
                    seed,
                    logit_bias,
                } keep {
                    top_k,
                    min_p,
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
}

/// `extra_args[group]` as an object, created when it is absent or null.
fn extra_args_group<'a>(
    extra_args: &'a mut Option<serde_json::Value>,
    group: &str,
) -> Option<&'a mut JsonObject> {
    let entry = extra_args
        .get_or_insert_with(|| serde_json::json!({}))
        .as_object_mut()?
        .entry(group)
        .or_insert(serde_json::Value::Null);
    if entry.is_null() {
        *entry = serde_json::json!({});
    }
    entry.as_object_mut()
}

/// Add the entries of `args` that `extra_args[group]` does not have, or has as null.
fn fill_extra_args(
    extra_args: &mut Option<serde_json::Value>,
    group: &str,
    args: Option<&JsonObject>,
) {
    let Some(args) = args else {
        return;
    };
    let Some(entry) = extra_args_group(extra_args, group) else {
        return;
    };
    for (key, value) in args {
        if entry.get(key).is_none_or(serde_json::Value::is_null) {
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
    /// chat processor rendered.
    #[test]
    fn rendered_request_gets_what_the_processor_left_unset() {
        let request = chat_request(serde_json::json!({
            "seed": 7,
            "logit_bias": {"5": 1.5},
            "max_tokens": 64,
            "min_p": 0.05,
            "response_format": {"type": "json_object"},
            "logprobs": true,
            "stop": ["END"],
            "allowed_token_ids": [1, 2],
            "nvext": {
                "backend_instance_id": 42,
                "agent_hints": {"priority": 3, "osl": 128},
                "cache_salt": "tenant-a",
                "extra_fields": ["worker_id"],
                "annotations": ["query_instance_id"],
            },
        }));
        let scope = envelope(None).begin_chat(&request);
        // The grammar is never filled, so the scope does not keep the request's schema.
        let facts = scope.request.as_deref().unwrap();
        let sampling = facts.sampling_options.as_ref().unwrap();
        assert!(sampling.guided_decoding.is_none());

        let mut rendered = rendered_request();
        // A null entry is unset, like a missing one.
        rendered.extra_args = Some(serde_json::json!({
            "sampling_options": {"allowed_token_ids": null},
        }));
        scope.complete_request(&mut rendered);

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
            rendered.extra_args.unwrap(),
            serde_json::json!({
                "nvext": {"extra_fields": ["worker_id"], "cache_salt": "tenant-a"},
                "sampling_options": {"allowed_token_ids": [1, 2]},
            })
        );

        // The processor's own are never filled, even though the request has them: a
        // grammar, logprobs, stop strings, annotations, and the `min_p` an engine may
        // have turned off.
        assert!(rendered.sampling_options.guided_decoding.is_none());
        assert!(rendered.sampling_options.min_p.is_none());
        assert!(rendered.output_options.logprobs.is_none());
        assert!(rendered.stop_conditions.stop.is_none());
        assert!(rendered.annotations.is_empty());
    }

    #[test]
    fn values_the_processor_set_are_kept() {
        let agent_context = |session_id: &str| -> AgentContext {
            serde_json::from_value(serde_json::json!({ "session_id": session_id })).unwrap()
        };
        let mut request = chat_request(serde_json::json!({
            "seed": 7,
            "temperature": 0.2,
            "max_tokens": 64,
            "nvext": {
                "backend_instance_id": 42,
                "cache_salt": "tenant-a",
                "router": {"ttft_target": 95.0, "itl_target": 12.0},
            },
        }));
        request.insert(AGENT_CONTEXT_CONTEXT_KEY, agent_context("from-headers"));
        let scope = envelope(None).begin_chat(&request);

        let mut rendered = rendered_request();
        rendered.agent_context = Some(agent_context("from-processor"));
        rendered.router = Some(RouterParams {
            ttft_target: Some(50.0),
            itl_target: None,
        });
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

        assert_eq!(
            rendered.agent_context,
            Some(agent_context("from-processor"))
        );
        assert_eq!(rendered.sampling_options.temperature, Some(1.0));
        assert_eq!(rendered.sampling_options.seed, Some(7));
        assert_eq!(rendered.stop_conditions.max_tokens, Some(900));
        // Each latency target is the processor's when it set one, else the client's.
        assert_eq!(
            rendered.router,
            Some(RouterParams {
                ttft_target: Some(50.0),
                itl_target: Some(12.0),
            })
        );
        let routing = rendered.routing.as_ref().unwrap();
        assert_eq!(routing.backend_instance_id, Some(7));
        // The router keys on the salt the processor gave the worker.
        assert_eq!(routing.cache_namespace.as_deref(), Some("from-processor"));
        assert_eq!(
            rendered.extra_args.unwrap(),
            serde_json::json!({
                "nvext": {"cache_salt": "from-processor"},
                "reasoning_ended": true,
            })
        );
    }

    /// The budget is checked against the output limit the worker will get: the
    /// processor's, or the client's where the processor left it unset.
    #[test]
    fn token_budget_checks_the_completed_request() {
        let budget = TokenBudget {
            combined_limit: 10,
            reject_prompt_overflow: true,
            reject_total_overflow: true,
        };
        let envelope = envelope(Some(budget));
        let check = |client: serde_json::Value, processor_max_tokens: Option<u32>| {
            let scope = envelope.begin_chat(&chat_request(client));
            let mut rendered = rendered_request();
            rendered.stop_conditions.max_tokens = processor_max_tokens;
            scope.complete_request(&mut rendered);
            envelope.validate_token_budget(&rendered)
        };

        check(serde_json::json!({}), Some(7)).unwrap();
        let error = check(serde_json::json!({}), Some(1_000)).unwrap_err();
        assert!(error.to_string().contains("asks for 1000 output tokens"));
        let error = check(serde_json::json!({"max_tokens": 8}), None).unwrap_err();
        assert!(error.to_string().contains("asks for 8 output tokens"));
    }
}
