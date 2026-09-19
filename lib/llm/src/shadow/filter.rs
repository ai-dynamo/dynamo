// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Named filters for shadow taps.
//!
//! Filters run on `&PreprocessedRequest` before anything is copied. A tap never
//! clones a request and then strips it: a multimodal payload can be megabytes,
//! and the copy happens on the request-serving path.

use std::sync::Arc;

use anyhow::{Result, bail};
use bitflags::bitflags;

use crate::protocols::common::preprocessor::{PreprocessedRequest, RoutingHints};

pub const STRIP_WORKER_PINS: &str = "strip-worker-pins";
pub const KEEP_WORKER_PINS: &str = "keep-worker-pins";
pub const TOKENS_ONLY: &str = "tokens-only";
pub const STRIP_MULTIMODAL: &str = "strip-multimodal";
pub const STRIP_EMBEDS: &str = "strip-embeds";
pub const STRIP_EXTRA_ARGS: &str = "strip-extra-args";
pub const STRIP_AGENT_CONTEXT: &str = "strip-agent-context";
pub const SKIP_MULTIMODAL: &str = "skip-multimodal";
pub const MAX_TOKENS_PREFIX: &str = "max-tokens=";

bitflags! {
    /// The fields a tap leaves out of its copy.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Projection: u16 {
        const MULTIMODAL = 1 << 0;
        const EMBEDS = 1 << 1;
        const EXTRA_ARGS = 1 << 2;
        const AGENT_CONTEXT = 1 << 3;
        const WORKER_PINS = 1 << 4;
        /// Everything in `routing`, plus `router`, `kv_hint`,
        /// `router_config_override`, and `mdc_sum`.
        const ROUTING = 1 << 5;
        const TOKENS_ONLY = Self::MULTIMODAL.bits()
            | Self::EMBEDS.bits()
            | Self::EXTRA_ARGS.bits()
            | Self::AGENT_CONTEXT.bits()
            | Self::WORKER_PINS.bits()
            | Self::ROUTING.bits();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Select {
    SkipMultimodal,
    MaxTokens(usize),
    Models(Vec<String>),
}

impl Select {
    fn accepts(&self, request: &PreprocessedRequest) -> bool {
        match self {
            Select::SkipMultimodal => {
                request.multi_modal_data.is_none() && request.prompt_embeds.is_none()
            }
            Select::MaxTokens(limit) => request.token_ids.len() <= *limit,
            Select::Models(models) => models.iter().any(|model| model == &request.model),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterSet {
    pub projection: Projection,
    pub selects: Vec<Select>,
    /// The filters in effect, defaults included. A consumer reads this from
    /// the envelope to learn which fields are missing.
    pub names: Arc<[String]>,
}

impl FilterSet {
    /// Resolve filter names from the config. An unknown name is an error so
    /// that a typo cannot silently mirror more data than intended.
    pub fn resolve(filters: &[String], models: &[String]) -> Result<Self> {
        let mut projection = Projection::WORKER_PINS;
        let mut keep_worker_pins = false;
        let mut selects = Vec::new();

        for filter in filters {
            match filter.as_str() {
                TOKENS_ONLY => projection |= Projection::TOKENS_ONLY,
                STRIP_MULTIMODAL => projection |= Projection::MULTIMODAL,
                STRIP_EMBEDS => projection |= Projection::EMBEDS,
                STRIP_EXTRA_ARGS => projection |= Projection::EXTRA_ARGS,
                STRIP_AGENT_CONTEXT => projection |= Projection::AGENT_CONTEXT,
                STRIP_WORKER_PINS => {}
                KEEP_WORKER_PINS => keep_worker_pins = true,
                SKIP_MULTIMODAL => selects.push(Select::SkipMultimodal),
                other => match other.strip_prefix(MAX_TOKENS_PREFIX) {
                    Some(limit) => {
                        selects.push(Select::MaxTokens(limit.parse().map_err(|error| {
                            anyhow::anyhow!("shadow tap filter `{other}`: {error}")
                        })?))
                    }
                    None => bail!("unknown shadow tap filter `{other}`"),
                },
            }
        }

        if keep_worker_pins {
            if projection.contains(Projection::ROUTING) {
                bail!(
                    "`{KEEP_WORKER_PINS}` conflicts with `{TOKENS_ONLY}`, which drops all routing"
                );
            }
            projection.remove(Projection::WORKER_PINS);
        }
        if !models.is_empty() {
            selects.push(Select::Models(models.to_vec()));
        }

        let mut names: Vec<String> = filters
            .iter()
            .filter(|name| name.as_str() != STRIP_WORKER_PINS)
            .cloned()
            .collect();
        if projection.contains(Projection::WORKER_PINS) {
            names.push(STRIP_WORKER_PINS.to_string());
        }

        Ok(Self {
            projection,
            selects,
            names: names.into(),
        })
    }

    /// Health-check probes are synthetic and are never mirrored.
    pub fn accepts(&self, request: &PreprocessedRequest) -> bool {
        !request.is_probe && self.selects.iter().all(|select| select.accepts(request))
    }
}

/// Copy the fields `projection` keeps. Fields it leaves out are never cloned.
///
/// Both the source and the result are written without a `..` rest pattern, so a
/// new `PreprocessedRequest` field does not compile until someone decides how a
/// shadow tap treats it.
pub fn project(request: &PreprocessedRequest, projection: Projection) -> PreprocessedRequest {
    let PreprocessedRequest {
        model,
        migration_state: _,
        staged_kv_cleanup: _,
        token_ids,
        prompt_embeds,
        multi_modal_data,
        multi_modal_uuids,
        mm_routing_info,
        stop_conditions,
        sampling_options,
        output_options,
        eos_token_ids,
        mdc_sum,
        annotations,
        routing,
        router_config_override,
        prefill_result: _,
        encoder_result: _,
        migration_link: _,
        jail_seed: _,
        bootstrap_info: _,
        extra_args,
        kv_hint,
        require_reasoning,
        router,
        agent_context,
        mm_processor_kwargs,
        media_io_kwargs,
        request_timestamp_ms,
        tracker: _,
        is_probe: _,
    } = request;

    let keep = |flag: Projection| !projection.contains(flag);
    let multimodal = keep(Projection::MULTIMODAL);
    let agent = keep(Projection::AGENT_CONTEXT);
    let route = keep(Projection::ROUTING);

    PreprocessedRequest {
        model: model.clone(),
        token_ids: Arc::clone(token_ids),
        stop_conditions: stop_conditions.clone(),
        sampling_options: sampling_options.clone(),
        output_options: output_options.clone(),
        eos_token_ids: eos_token_ids.clone(),
        require_reasoning: *require_reasoning,
        request_timestamp_ms: *request_timestamp_ms,

        prompt_embeds: keep(Projection::EMBEDS)
            .then(|| prompt_embeds.clone())
            .flatten(),
        multi_modal_data: multimodal.then(|| multi_modal_data.clone()).flatten(),
        multi_modal_uuids: multimodal.then(|| multi_modal_uuids.clone()).flatten(),
        mm_routing_info: multimodal.then(|| mm_routing_info.clone()).flatten(),
        mm_processor_kwargs: multimodal.then(|| mm_processor_kwargs.clone()).flatten(),
        media_io_kwargs: multimodal.then(|| media_io_kwargs.clone()).flatten(),
        extra_args: keep(Projection::EXTRA_ARGS)
            .then(|| extra_args.clone())
            .flatten(),
        annotations: if agent {
            annotations.clone()
        } else {
            Vec::new()
        },
        agent_context: agent.then(|| agent_context.clone()).flatten(),

        routing: route
            .then(|| {
                routing
                    .as_ref()
                    .map(|hints| project_routing(hints, keep(Projection::WORKER_PINS)))
            })
            .flatten(),
        router_config_override: route.then(|| router_config_override.clone()).flatten(),
        kv_hint: route.then(|| kv_hint.clone()).flatten(),
        router: route.then(|| router.clone()).flatten(),
        mdc_sum: route.then(|| mdc_sum.clone()).flatten(),

        // State of the primary deployment's workers and traces. A shadow has
        // its own, so these are never mirrored.
        prefill_result: None,
        encoder_result: None,
        bootstrap_info: None,
        migration_link: None,
        // Frontend-only, never serialized.
        migration_state: None,
        staged_kv_cleanup: false,
        jail_seed: None,
        tracker: None,
        is_probe: false,
    }
}

/// Worker pins name workers of the primary deployment. A shadow router has no
/// such workers, so a mirrored pin would fail or misroute there. The other
/// hints describe the workload and are kept.
fn project_routing(hints: &RoutingHints, keep_worker_pins: bool) -> RoutingHints {
    let RoutingHints {
        backend_instance_id,
        prefill_worker_id,
        decode_worker_id,
        dp_rank,
        prefill_dp_rank,
        expected_output_tokens,
        lora_name,
        cache_namespace,
        priority_jump,
        strict_priority,
        priority,
        allowed_worker_ids,
        routing_constraints,
    } = hints;

    RoutingHints {
        backend_instance_id: backend_instance_id.filter(|_| keep_worker_pins),
        prefill_worker_id: prefill_worker_id.filter(|_| keep_worker_pins),
        decode_worker_id: decode_worker_id.filter(|_| keep_worker_pins),
        dp_rank: dp_rank.filter(|_| keep_worker_pins),
        prefill_dp_rank: prefill_dp_rank.filter(|_| keep_worker_pins),
        allowed_worker_ids: keep_worker_pins
            .then(|| allowed_worker_ids.clone())
            .flatten(),
        expected_output_tokens: *expected_output_tokens,
        lora_name: lora_name.clone(),
        cache_namespace: cache_namespace.clone(),
        priority_jump: *priority_jump,
        strict_priority: *strict_priority,
        priority: *priority,
        routing_constraints: routing_constraints.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::common::{SamplingOptions, StopConditions};

    pub(crate) fn full_request() -> PreprocessedRequest {
        let mut request = PreprocessedRequest::builder()
            .model("m".to_string())
            .token_ids(vec![1, 2, 3, 4])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(Default::default())
            .build()
            .unwrap();
        request.prompt_embeds = Some("embeds".to_string());
        request.extra_args = Some(serde_json::json!({"k": "v"}));
        request.mm_processor_kwargs = Some(serde_json::json!({"fps": 1}));
        request.annotations = vec!["a".to_string()];
        request.mdc_sum = Some("sum".to_string());
        request.bootstrap_info = None;
        request.encoder_result = Some(serde_json::json!({"x": 1}));
        request.jail_seed = Some("seed".to_string());
        request.routing = Some(RoutingHints {
            backend_instance_id: Some(7),
            decode_worker_id: Some(8),
            dp_rank: Some(1),
            allowed_worker_ids: Some([7u64].into_iter().collect()),
            lora_name: Some("lora".to_string()),
            priority: Some(3),
            ..Default::default()
        });
        request
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|name| name.to_string()).collect()
    }

    #[test]
    fn default_strips_worker_pins_and_keeps_workload_hints() {
        let filters = FilterSet::resolve(&[], &[]).unwrap();
        assert_eq!(&*filters.names, &[STRIP_WORKER_PINS.to_string()]);

        let source = full_request();
        let copy = project(&source, filters.projection);
        let routing = copy.routing.unwrap();
        assert_eq!(routing.backend_instance_id, None);
        assert_eq!(routing.decode_worker_id, None);
        assert_eq!(routing.dp_rank, None);
        assert_eq!(routing.allowed_worker_ids, None);
        assert_eq!(routing.lora_name.as_deref(), Some("lora"));
        assert_eq!(routing.priority, Some(3));
        assert_eq!(copy.prompt_embeds.as_deref(), Some("embeds"));
        assert_eq!(copy.extra_args, source.extra_args);
        assert_eq!(copy.mdc_sum.as_deref(), Some("sum"));
    }

    #[test]
    fn keep_worker_pins_copies_pins() {
        let filters = FilterSet::resolve(&names(&[KEEP_WORKER_PINS]), &[]).unwrap();
        let copy = project(&full_request(), filters.projection);
        let routing = copy.routing.unwrap();
        assert_eq!(routing.backend_instance_id, Some(7));
        assert!(routing.allowed_worker_ids.is_some());
        assert!(!filters.names.contains(&STRIP_WORKER_PINS.to_string()));
    }

    #[test]
    fn tokens_only_keeps_the_prompt_and_generation_settings() {
        let filters = FilterSet::resolve(&names(&[TOKENS_ONLY]), &[]).unwrap();
        let source = full_request();
        let copy = project(&source, filters.projection);
        assert!(Arc::ptr_eq(&copy.token_ids, &source.token_ids));
        assert_eq!(copy.model, "m");
        assert!(copy.routing.is_none());
        assert!(copy.prompt_embeds.is_none());
        assert!(copy.extra_args.is_none());
        assert!(copy.mm_processor_kwargs.is_none());
        assert!(copy.annotations.is_empty());
        assert!(copy.mdc_sum.is_none());
    }

    #[test]
    fn primary_deployment_state_is_never_copied() {
        let copy = project(&full_request(), Projection::empty());
        assert!(copy.encoder_result.is_none());
        assert!(copy.jail_seed.is_none());
        assert!(copy.migration_link.is_none());
    }

    #[test]
    fn each_strip_filter_leaves_other_fields_alone() {
        let source = full_request();
        let embeds = FilterSet::resolve(&names(&[STRIP_EMBEDS]), &[]).unwrap();
        let copy = project(&source, embeds.projection);
        assert!(copy.prompt_embeds.is_none());
        assert!(copy.extra_args.is_some());

        let extra = FilterSet::resolve(&names(&[STRIP_EXTRA_ARGS]), &[]).unwrap();
        let copy = project(&source, extra.projection);
        assert!(copy.extra_args.is_none());
        assert!(copy.prompt_embeds.is_some());

        let agent = FilterSet::resolve(&names(&[STRIP_AGENT_CONTEXT]), &[]).unwrap();
        assert!(project(&source, agent.projection).annotations.is_empty());

        let multimodal = FilterSet::resolve(&names(&[STRIP_MULTIMODAL]), &[]).unwrap();
        let copy = project(&source, multimodal.projection);
        assert!(copy.mm_processor_kwargs.is_none());
        assert!(copy.prompt_embeds.is_some());
    }

    #[test]
    fn unknown_filter_is_an_error() {
        let error = FilterSet::resolve(&names(&["strip-everything"]), &[]).unwrap_err();
        assert!(error.to_string().contains("strip-everything"));
        assert!(FilterSet::resolve(&names(&["max-tokens=abc"]), &[]).is_err());
        assert!(FilterSet::resolve(&names(&[TOKENS_ONLY, KEEP_WORKER_PINS]), &[]).is_err());
    }

    #[test]
    fn select_filters_reject_requests() {
        let source = full_request();
        let skip = FilterSet::resolve(&names(&[SKIP_MULTIMODAL]), &[]).unwrap();
        assert!(!skip.accepts(&source), "request carries prompt embeds");

        let short = FilterSet::resolve(&names(&["max-tokens=3"]), &[]).unwrap();
        assert!(!short.accepts(&source));
        let long = FilterSet::resolve(&names(&["max-tokens=4"]), &[]).unwrap();
        assert!(long.accepts(&source));

        let other_model = FilterSet::resolve(&[], &names(&["other"])).unwrap();
        assert!(!other_model.accepts(&source));
        let this_model = FilterSet::resolve(&[], &names(&["m", "other"])).unwrap();
        assert!(this_model.accepts(&source));

        let mut probe = full_request();
        probe.is_probe = true;
        assert!(!FilterSet::resolve(&[], &[]).unwrap().accepts(&probe));
    }
}
