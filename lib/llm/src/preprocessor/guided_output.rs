// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::{Value, json};

use super::{OpenAIPreprocessor, PromptReasoningPrefill};
use crate::protocols::{
    common::{
        guided_output::{
            GUIDED_OUTPUT_REASONING_RUNTIME_KEY, GuidedOutputCapability, GuidedOutputPlan,
            GuidedReasoningMode,
        },
        invalid_argument_error,
        llm_backend::PreprocessedRequest,
    },
    openai::{
        GuidedToolConstraint, chat_completions::NvCreateChatCompletionRequest,
        common_ext::CommonExtProvider, validate,
    },
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FinalAnswerConstraint {
    Json(Value),
    Choice(Vec<String>),
    Regex(String),
    Grammar(String),
}

impl FinalAnswerConstraint {
    pub(crate) fn from_request(request: &NvCreateChatCompletionRequest) -> Option<Self> {
        request
            .get_guided_json()
            .map(Self::Json)
            .or_else(|| request.common.guided_regex.clone().map(Self::Regex))
            .or_else(|| request.common.guided_grammar.clone().map(Self::Grammar))
            .or_else(|| {
                request
                    .common
                    .guided_choice
                    .clone()
                    .filter(|choices| !choices.is_empty())
                    .map(Self::Choice)
            })
    }
}

pub(crate) fn normalize_guided_json(
    request: &mut NvCreateChatCompletionRequest,
) -> anyhow::Result<()> {
    validate::validate_response_format_conflicts(&request.inner.response_format, &request.common)
        .map_err(|error| invalid_argument_error(error.to_string()))?;
    let count = usize::from(request.common.guided_json.is_some())
        + usize::from(request.common.guided_regex.is_some())
        + usize::from(request.common.guided_grammar.is_some())
        + usize::from(
            request
                .common
                .guided_choice
                .as_ref()
                .is_some_and(|v| !v.is_empty()),
        );
    if count > 1 {
        return Err(invalid_argument_error(
            "Only one final-answer guided constraint may be supplied",
        ));
    }
    let Some(schema) = request.common.guided_json.as_ref() else {
        return Ok(());
    };
    let schema = if let Some(encoded) = schema.as_str() {
        serde_json::from_str::<Value>(encoded).map_err(|error| {
            invalid_argument_error(format!(
                "guided_json is not an encoded JSON schema: {error}"
            ))
        })?
    } else {
        schema.clone()
    };
    if !schema.is_object() {
        return Err(invalid_argument_error(
            "guided_json must be a JSON schema object or an encoded schema object",
        ));
    }
    request.inner.response_format = Some(serde_json::from_value(json!({
        "type": "json_schema",
        "json_schema": {"name": "guided_json", "schema": schema}
    }))?);
    request.common.guided_json = None;
    Ok(())
}

impl OpenAIPreprocessor {
    pub(crate) fn resolve_guided_output_plan(
        &self,
        request: &NvCreateChatCompletionRequest,
        common: &mut PreprocessedRequest,
        prompt: &PromptReasoningPrefill,
        installed: &GuidedToolConstraint,
    ) -> anyhow::Result<Option<GuidedOutputPlan>> {
        let Some(constraint) = FinalAnswerConstraint::from_request(request) else {
            return Ok(None);
        };
        // A forced tool call uses its own argument constraint, just as response_format does.
        if matches!(constraint, FinalAnswerConstraint::Json(_))
            && matches!(
                request.inner.tool_choice,
                Some(
                    dynamo_protocols::types::ChatCompletionToolChoiceOption::Required
                        | dynamo_protocols::types::ChatCompletionToolChoiceOption::Named(_)
                )
            )
        {
            return Ok(None);
        }
        let parser = self.runtime_config.reasoning_parser.as_deref();
        let family = crate::protocols::openai::chat_completions::unified_parser::configured_family(
            self.tool_call_parser.as_deref(),
            parser,
        );
        let reasoning = !prompt.response
            && match prompt.reasoning_closed_by_prompt {
                Some(closed) => !closed,
                None => {
                    prompt.legacy
                        || prompt.unified
                        || Self::reasoning_is_mandatory(family)
                        || Self::sglang_effective_reasoning_enabled(
                            parser.or(family),
                            request.chat_template_args.as_ref(),
                        )
                }
            };
        if reasoning
            && Self::wants_reasoning_as_content_when_empty(request.chat_template_args.as_ref())
        {
            return Err(invalid_argument_error(
                "force_nonempty_content=true cannot be combined with a guided answer constraint and enabled reasoning",
            ));
        }
        if matches!(constraint, FinalAnswerConstraint::Json(_))
            && (installed.uses_structural_tag() || installed.frames_structured_answer())
        {
            return Ok(Some(GuidedOutputPlan::NativeJson));
        }
        let capability = self
            .runtime_config
            .get_engine_specific::<GuidedOutputCapability>(GUIDED_OUTPUT_REASONING_RUNTIME_KEY)
            .map_err(|error| {
                invalid_argument_error(format!("invalid guided-output worker capability: {error}"))
            })?;
        let capability = capability.filter(|capability| capability.version == 1);
        let unsupported = || {
            invalid_argument_error(concat!(
                "The selected workers cannot preserve this guided-answer reasoning policy. ",
                "Use workers publishing guided_output_reasoning version 1 with a supported native ",
                "--reasoning-parser and request-local reasoning controls, or disable thinking if the model supports it. ",
                "Implicit-boundary and non-JSON native-channel grammars are not supported."
            ))
        };
        if reasoning
            && !capability.as_ref().is_some_and(|capability| {
                capability.mode == GuidedReasoningMode::ExplicitGate
                    && capability.request_controls
                    && capability.reasoner.is_some()
                    && !capability.end_token_ids.is_empty()
            })
        {
            return Err(unsupported());
        }
        if !reasoning
            && capability.as_ref().is_some_and(|capability| {
                capability.reasoner.is_some()
                    && capability.mode != GuidedReasoningMode::Immediate
                    && !capability.request_controls
            })
        {
            return Err(unsupported());
        }
        if !reasoning && capability.is_none() {
            // Legacy positive gate flags are not proof of named request-local controls.
            for key in [
                "tool_call_structural_tag_excludes_reasoning",
                "tool_call_structural_tag_reasoning_gate",
            ] {
                if self
                    .runtime_config
                    .get_engine_specific::<bool>(key)?
                    .unwrap_or(false)
                {
                    return Err(unsupported());
                }
            }
        }
        common.require_reasoning = reasoning;
        let extra = common.extra_args.get_or_insert_with(|| json!({}));
        let extra = extra
            .as_object_mut()
            .ok_or_else(|| invalid_argument_error("extra_args must be an object"))?;
        extra.insert("reasoning_ended".into(), json!(!reasoning));
        extra.insert(
            "reasoning_parser_kwargs".into(),
            json!({
                "chat_template_kwargs": request.chat_template_args.clone().unwrap_or_default()
            }),
        );
        let plan = if reasoning {
            GuidedOutputPlan::EngineGated(capability.expect("validated above"))
        } else {
            GuidedOutputPlan::Direct
        };
        common.guided_output_plan = Some(plan.clone());
        Ok(Some(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(fields: Value) -> NvCreateChatCompletionRequest {
        let mut request = json!({"model":"test", "messages":[{"role":"user","content":"test"}]});
        request
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(request).unwrap()
    }

    #[test]
    fn legacy_schema_alias_preserves_each_json_root_and_modifiers() {
        for root in [
            "object", "array", "integer", "number", "string", "boolean", "null",
        ] {
            let schema = json!({"type": root});
            for input in [schema.clone(), Value::String(schema.to_string())] {
                let mut request =
                    request(json!({"guided_json":input,"guided_whitespace_pattern":" *"}));
                normalize_guided_json(&mut request).unwrap();
                assert!(request.common.guided_json.is_none());
                assert_eq!(request.get_guided_json(), Some(schema.clone()));
                assert_eq!(
                    request.common.guided_whitespace_pattern.as_deref(),
                    Some(" *")
                );
                let format = serde_json::to_value(request.inner.response_format).unwrap();
                assert_eq!(format["json_schema"]["name"], "guided_json");
                assert_ne!(format["json_schema"]["strict"], true);
            }
        }
    }

    #[test]
    fn raw_constraint_conflicts_and_invalid_encoded_schemas_are_rejected() {
        for fields in [
            json!({"guided_json":{}, "guided_choice":["yes"]}),
            json!({"guided_json":{}, "response_format":{"type":"json_object"}}),
            json!({"guided_json":"not JSON"}),
            json!({"guided_json":"[]"}),
            json!({"guided_json":true}),
        ] {
            assert!(
                normalize_guided_json(&mut request(fields.clone())).is_err(),
                "{fields}"
            );
        }
    }

    #[tokio::test]
    async fn guided_output_json_aliases_render_and_install_identically() {
        let processor = processor(Some(gate()));
        for root in ["object", "array", "number", "string", "boolean", "null"] {
            for thinking in [None, Some(false), Some(true)] {
                let schema = json!({"type":root});
                let mut snapshots = Vec::new();
                for fields in [
                    json!({"guided_json":schema}),
                    json!({"guided_json":schema.to_string()}),
                    json!({"response_format":{"type":"json_schema","json_schema":{"name":"guided_json","schema":schema}}}),
                ] {
                    let mut request = request(fields);
                    if let Some(thinking) = thinking {
                        request.chat_template_args = Some(std::collections::HashMap::from([(
                            "enable_thinking".into(),
                            json!(thinking),
                        )]));
                    }
                    normalize_guided_json(&mut request).unwrap();
                    let from_client =
                        OpenAIPreprocessor::request_has_client_thinking_control(&request);
                    processor.apply_default_thinking_mode(&mut request);
                    OpenAIPreprocessor::normalize_thinking_arg_with_source(
                        &mut request,
                        Some("qwen3"),
                        None,
                        from_client,
                    );
                    let (mut common, _, prefill) =
                        processor.preprocess_request(&request, None).await.unwrap();
                    let installed = processor
                        .apply_tool_choice_guided_decoding(&request, &mut common, prefill.clone())
                        .unwrap();
                    let plan = processor
                        .resolve_guided_output_plan(&request, &mut common, &prefill, &installed)
                        .unwrap();
                    snapshots.push((serde_json::to_value(common).unwrap(), installed, plan));
                }
                assert_eq!(snapshots[0], snapshots[1], "{root} encoded alias");
                assert_eq!(snapshots[0], snapshots[2], "{root} response_format alias");
            }
        }
    }

    fn processor(capability: Option<GuidedOutputCapability>) -> std::sync::Arc<OpenAIPreprocessor> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut card = crate::model_card::ModelDeploymentCard::load_from_disk(path, None).unwrap();
        card.runtime_config.reasoning_parser = Some("qwen3".into());
        if let Some(capability) = capability {
            card.runtime_config
                .set_engine_specific(GUIDED_OUTPUT_REASONING_RUNTIME_KEY, capability)
                .unwrap();
        }
        OpenAIPreprocessor::new(card).unwrap()
    }

    fn gate() -> GuidedOutputCapability {
        GuidedOutputCapability {
            version: 1,
            engine: "vllm".into(),
            mode: GuidedReasoningMode::ExplicitGate,
            request_controls: true,
            reasoner: Some("deepseek_r1".into()),
            start_token_ids: vec![4],
            end_token_ids: vec![5],
            start_text: "<think>".into(),
            end_text: "</think>".into(),
        }
    }

    #[test]
    fn all_formats_resolve_the_same_reasoning_policy_and_worker_controls() {
        let processor = processor(Some(gate()));
        for fields in [
            json!({"guided_json":{"type":"integer"}}),
            json!({"response_format":{"type":"json_schema","json_schema":{"name":"guided_json","schema":{"type":"integer"}}}}),
            json!({"guided_choice":["yes","no"]}),
            json!({"guided_regex":"yes|no"}),
            json!({"guided_grammar":"root ::= \"yes\" | \"no\""}),
        ] {
            for thinking in [false, true] {
                let mut request = request(fields.clone());
                request.chat_template_args = Some(std::collections::HashMap::from([(
                    "enable_thinking".into(),
                    json!(thinking),
                )]));
                normalize_guided_json(&mut request).unwrap();
                let mut common = processor
                    .builder_with_lora(&request, None)
                    .unwrap()
                    .token_ids(vec![])
                    .build()
                    .unwrap();
                let plan = processor
                    .resolve_guided_output_plan(
                        &request,
                        &mut common,
                        &false.into(),
                        &GuidedToolConstraint::None,
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(
                    matches!(plan, GuidedOutputPlan::EngineGated(_)),
                    thinking,
                    "{fields}"
                );
                assert_eq!(common.require_reasoning, thinking);
                assert_eq!(
                    common.extra_args.as_ref().unwrap()["reasoning_ended"],
                    !thinking
                );
                assert_eq!(common.guided_output_plan, Some(plan));
            }
        }
    }

    #[test]
    fn guided_output_preserves_existing_native_json_envelopes() {
        let processor = processor(None);
        for fields in [
            json!({"guided_json":{"type":"array"}}),
            json!({"response_format":{"type":"json_schema","json_schema":{"name":"guided_json","schema":{"type":"array"}}}}),
        ] {
            let mut request = request(fields);
            normalize_guided_json(&mut request).unwrap();
            let mut common = processor
                .builder_with_lora(&request, None)
                .unwrap()
                .token_ids(vec![])
                .build()
                .unwrap();
            for installed in [
                GuidedToolConstraint::StructuralTag,
                GuidedToolConstraint::BeamTurnGrammar {
                    frames_answer: true,
                },
            ] {
                assert_eq!(
                    processor
                        .resolve_guided_output_plan(&request, &mut common, &true.into(), &installed)
                        .unwrap(),
                    Some(GuidedOutputPlan::NativeJson)
                );
                assert!(common.guided_output_plan.is_none());
            }
        }
    }

    #[test]
    fn guided_output_rejects_mandatory_beam_reasoning_without_an_explicit_gate() {
        for tool_only in [false, true] {
            let mut processor = processor(None);
            let inner = std::sync::Arc::get_mut(&mut processor).unwrap();
            inner.tool_call_parser = Some("beam".into());
            inner.runtime_config.reasoning_parser = (!tool_only).then(|| "beam".into());
            for fields in [
                json!({"guided_choice":["yes"]}),
                json!({"guided_regex":"yes"}),
                json!({"guided_grammar":"root ::= \"yes\""}),
            ] {
                let request = request(fields);
                let mut common = processor
                    .builder_with_lora(&request, None)
                    .unwrap()
                    .token_ids(vec![])
                    .build()
                    .unwrap();
                for prompt in [
                    "<|start|>assistant",
                    "<|start|>assistant<|channel|>analysis<|message|>",
                ] {
                    let prefill = OpenAIPreprocessor::prompt_reasoning_prefill_at_boundary(
                        Some("beam"),
                        (!tool_only).then_some("beam"),
                        Some(prompt),
                        None,
                    );
                    let error = processor
                        .resolve_guided_output_plan(
                            &request,
                            &mut common,
                            &prefill,
                            &GuidedToolConstraint::None,
                        )
                        .unwrap_err();
                    assert!(error.to_string().contains("cannot preserve"), "{error}");
                }
            }
        }
    }

    #[test]
    fn missing_gate_and_force_nonempty_conflicts_fail_before_generation() {
        for capability in [
            None,
            Some(GuidedOutputCapability {
                mode: GuidedReasoningMode::Unsupported,
                ..gate()
            }),
        ] {
            let processor = processor(capability);
            let request = request(json!({"guided_choice":["yes","no"]}));
            let mut common = processor
                .builder_with_lora(&request, None)
                .unwrap()
                .token_ids(vec![])
                .build()
                .unwrap();
            let error = processor
                .resolve_guided_output_plan(
                    &request,
                    &mut common,
                    &false.into(),
                    &GuidedToolConstraint::None,
                )
                .unwrap_err();
            assert!(error.to_string().contains("cannot preserve"));
        }
        let processor = processor(Some(gate()));
        let request = request(json!({"guided_choice":["yes","no"],
            "chat_template_kwargs":{"force_nonempty_content":true}}));
        let mut common = processor
            .builder_with_lora(&request, None)
            .unwrap()
            .token_ids(vec![])
            .build()
            .unwrap();
        let error = processor
            .resolve_guided_output_plan(
                &request,
                &mut common,
                &false.into(),
                &GuidedToolConstraint::None,
            )
            .unwrap_err();
        assert!(error.to_string().contains("force_nonempty_content"));
    }
}
