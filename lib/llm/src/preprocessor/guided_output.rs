// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_protocols::types::ResponseFormat;
use serde_json::{Value, json};

use super::{OpenAIPreprocessor, PromptReasoningPrefill};
use crate::protocols::{
    common::{
        guided_output::{
            GUIDED_OUTPUT_COHORT_CONFLICT_RUNTIME_KEY, GUIDED_OUTPUT_REASONING_RUNTIME_KEY,
            GuidedOutputCapability, GuidedOutputPlan, GuidedReasoningMode,
        },
        invalid_argument_error,
        llm_backend::PreprocessedRequest,
    },
    openai::{GuidedToolConstraint, chat_completions::NvCreateChatCompletionRequest, validate},
};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FinalAnswerConstraint {
    Json,
    Choice,
    Regex,
    Grammar,
}

impl FinalAnswerConstraint {
    pub(crate) fn from_request(request: &NvCreateChatCompletionRequest) -> Option<Self> {
        let json = request.common.guided_json.is_some()
            || match request.inner.response_format.as_ref() {
                Some(ResponseFormat::JsonObject) => true,
                Some(ResponseFormat::JsonSchema { json_schema }) => !json_schema.schema.is_null(),
                _ => false,
            };
        json.then_some(Self::Json)
            .or_else(|| request.common.guided_regex.as_ref().map(|_| Self::Regex))
            .or_else(|| {
                request
                    .common
                    .guided_grammar
                    .as_ref()
                    .map(|_| Self::Grammar)
            })
            .or_else(|| {
                request
                    .common
                    .guided_choice
                    .as_ref()
                    .filter(|choices| !choices.is_empty())
                    .map(|_| Self::Choice)
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
        if matches!(constraint, FinalAnswerConstraint::Json)
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
        let capability = self
            .runtime_config
            .runtime_data
            .get(GUIDED_OUTPUT_REASONING_RUNTIME_KEY)
            .and_then(|value| {
                if value.get("version").and_then(Value::as_u64) != Some(1) {
                    tracing::warn!("Unavailable guided-output worker capability version");
                    return None;
                }
                match serde_json::from_value::<GuidedOutputCapability>(value.clone()) {
                    Ok(capability) => Some(capability),
                    Err(error) => {
                        tracing::warn!(%error, "Unavailable guided-output worker capability");
                        None
                    }
                }
            });
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
                        || (parser.is_none()
                            && capability
                                .as_ref()
                                .is_some_and(|value| value.reasoner.is_some())
                            && Self::normalize_thinking_aliases(
                                request.chat_template_args.as_ref(),
                                capability
                                    .as_ref()
                                    .and_then(|value| value.reasoner.as_deref()),
                            ) != Some(false))
                        || Self::known_reasoning_enabled(
                            parser,
                            request.chat_template_args.as_ref(),
                        )
                        .unwrap_or_else(|| {
                            parser.is_some()
                                && Self::normalize_thinking_aliases(
                                    request.chat_template_args.as_ref(),
                                    parser,
                                )
                                .unwrap_or(true)
                        })
                }
            };
        if reasoning
            && Self::wants_reasoning_as_content_when_empty(request.chat_template_args.as_ref())
        {
            return Err(invalid_argument_error(
                "force_nonempty_content=true cannot be combined with a guided answer constraint and enabled reasoning",
            ));
        }
        let native_json = matches!(constraint, FinalAnswerConstraint::Json)
            && (installed.uses_structural_tag() || installed.frames_structured_answer());
        if native_json && !common.native_json_requires_reasoning_gate {
            return Ok(Some(GuidedOutputPlan::NativeJson));
        }
        if self
            .runtime_config
            .runtime_data
            .get(GUIDED_OUTPUT_COHORT_CONFLICT_RUNTIME_KEY)
            == Some(&Value::Bool(true))
        {
            return Err(invalid_argument_error(
                "The selected workers cannot preserve this guided-answer reasoning policy because their guided_output_reasoning capabilities disagree. Configure matching workers before retrying.",
            ));
        }
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
                match self.runtime_config.runtime_data.get(key) {
                    None | Some(Value::Bool(false)) => {}
                    Some(Value::Bool(true)) => return Err(unsupported()),
                    Some(_) => {
                        tracing::warn!(key, "Unverified legacy guided-output reasoning metadata");
                        return Err(unsupported());
                    }
                }
            }
        }
        common.require_reasoning = reasoning;
        let extra = common.extra_args.get_or_insert_with(|| json!({}));
        let extra = extra
            .as_object_mut()
            .ok_or_else(|| invalid_argument_error("extra_args must be an object"))?;
        extra.insert("reasoning_ended".into(), json!(!reasoning));
        if parser.is_some()
            || capability
                .as_ref()
                .is_some_and(|value| value.reasoner.is_some())
        {
            let kwargs = extra
                .entry("reasoning_parser_kwargs")
                .or_insert_with(|| json!({}));
            let kwargs = kwargs.as_object_mut().ok_or_else(|| {
                invalid_argument_error("reasoning_parser_kwargs must be an object")
            })?;
            let chat = kwargs
                .entry("chat_template_kwargs")
                .or_insert_with(|| json!({}));
            let chat = chat.as_object_mut().ok_or_else(|| {
                invalid_argument_error(
                    "reasoning_parser_kwargs.chat_template_kwargs must be an object",
                )
            })?;
            if let Some(args) = &request.chat_template_args {
                chat.extend(args.iter().map(|(key, value)| (key.clone(), value.clone())));
            }
            if matches!(parser, Some("glm45" | "glm47"))
                && prompt.reasoning_closed_by_prompt.is_some()
            {
                chat.insert("thinking".into(), json!(reasoning));
                chat.insert("enable_thinking".into(), json!(reasoning));
            }
        }
        if native_json {
            // Tool/answer envelopes keep their parser after the engine gate is verified.
            return Ok(Some(GuidedOutputPlan::NativeJson));
        }
        let plan = if reasoning {
            GuidedOutputPlan::EngineGated(capability.expect("validated above"))
        } else {
            GuidedOutputPlan::Direct
        };
        // Projected answers bypass the parser that consumes model EOS markup.
        // Explicit no_stop_trim retains its normal behavior in the decoder.
        common.output_options.retain_model_stop_token_text = None;
        common.guided_output_plan = Some(plan.clone());
        Ok(Some(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::common_ext::CommonExtProvider;

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

    #[tokio::test]
    async fn guided_output_json_aliases_keep_the_same_tool_loop_contract() {
        let processor = processor(Some(gate()));
        let tools = json!([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]);
        let user =
            json!({"role":"user","content":"Get Paris weather, then return the result as JSON."});
        let schema = json!({"type":"object","properties":{"weather":{"type":"string"}},"required":["weather"]});
        for tool_turn in [true, false] {
            let mut messages = vec![user.clone()];
            if !tool_turn {
                messages.extend([
                    json!({"role":"assistant","tool_calls":[{"id":"weather_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}),
                    json!({"role":"tool","tool_call_id":"weather_1","content":"sunny"}),
                ]);
            }
            let mut snapshots = Vec::new();
            for format in [
                json!({"guided_json":schema}),
                json!({"response_format":{"type":"json_schema","json_schema":{"name":"guided_json","schema":schema}}}),
            ] {
                let mut fields = json!({"tools":tools,"messages":messages,"tool_choice":if tool_turn {"required"} else {"auto"}});
                fields
                    .as_object_mut()
                    .unwrap()
                    .extend(format.as_object().unwrap().clone());
                let mut request = request(fields);
                normalize_guided_json(&mut request).unwrap();
                let (mut common, _, prefill) =
                    processor.preprocess_request(&request, None).await.unwrap();
                let installed = processor
                    .apply_tool_choice_guided_decoding(&request, &mut common, prefill.clone())
                    .unwrap();
                let plan = processor
                    .resolve_guided_output_plan(&request, &mut common, &prefill, &installed)
                    .unwrap();
                assert_eq!(plan.is_none(), tool_turn);
                assert_eq!(installed.installs_guided_json(), tool_turn);
                snapshots.push((serde_json::to_value(common).unwrap(), installed, plan));
            }
            assert_eq!(snapshots[0], snapshots[1], "tool turn: {tool_turn}");
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
    fn guided_output_tool_only_deployments_keep_direct_answers() {
        let immediate = GuidedOutputCapability {
            mode: GuidedReasoningMode::Immediate,
            reasoner: None,
            ..gate()
        };
        for tool in ["qwen3_coder", "kimi_k2"] {
            for capability in [None, Some(immediate.clone())] {
                let mut processor = processor(capability);
                let inner = std::sync::Arc::get_mut(&mut processor).unwrap();
                inner.tool_call_parser = Some(tool.into());
                inner.runtime_config.reasoning_parser = None;
                for fields in [
                    json!({"guided_json":{"type":"object"}}),
                    json!({"response_format":{"type":"json_schema","json_schema":{"name":"answer","schema":{"type":"object"}}}}),
                    json!({"guided_choice":["yes","no"]}),
                    json!({"guided_regex":"yes|no"}),
                    json!({"guided_grammar":"root ::= \"yes\" | \"no\""}),
                ] {
                    let mut request = request(fields);
                    normalize_guided_json(&mut request).unwrap();
                    let mut common = processor
                        .builder_with_lora(&request, None)
                        .unwrap()
                        .token_ids(vec![])
                        .build()
                        .unwrap();
                    let result = processor.resolve_guided_output_plan(
                        &request,
                        &mut common,
                        &false.into(),
                        &GuidedToolConstraint::None,
                    );
                    assert_eq!(
                        result.unwrap_or_else(|error| panic!("{tool}: {error}")),
                        Some(GuidedOutputPlan::Direct),
                        "{tool}"
                    );
                    assert!(!common.require_reasoning);
                    assert_eq!(common.extra_args.as_ref().unwrap()["reasoning_ended"], true);
                    assert!(
                        common
                            .extra_args
                            .as_ref()
                            .unwrap()
                            .get("reasoning_parser_kwargs")
                            .is_none()
                    );
                }
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

    #[test]
    fn guided_output_unknown_metadata_and_reasoner_defaults_are_conservative() {
        let mut processor = processor(Some(gate()));
        let mut additive = serde_json::to_value(gate()).unwrap();
        additive["future_field"] = json!(true);
        for (metadata, available) in [
            (additive, true),
            (json!({"version":2,"mode":"future"}), false),
            (json!({"version":1,"mode":"future","engine":"vllm"}), false),
            (
                json!({"version":1,"mode":"explicit_gate","engine":7}),
                false,
            ),
            (json!("malformed capability"), false),
        ] {
            std::sync::Arc::get_mut(&mut processor)
                .unwrap()
                .runtime_config
                .runtime_data
                .insert(GUIDED_OUTPUT_REASONING_RUNTIME_KEY.into(), metadata);
            for thinking in [false, true] {
                let mut request = request(json!({"guided_choice":["yes"]}));
                request.chat_template_args = Some(std::collections::HashMap::from([(
                    "enable_thinking".into(),
                    json!(thinking),
                )]));
                let mut common = processor
                    .builder_with_lora(&request, None)
                    .unwrap()
                    .token_ids(vec![])
                    .build()
                    .unwrap();
                let plan = processor.resolve_guided_output_plan(
                    &request,
                    &mut common,
                    &false.into(),
                    &GuidedToolConstraint::None,
                );
                assert_eq!(plan.is_ok(), !thinking || available);
            }
        }
        let inner = std::sync::Arc::get_mut(&mut processor).unwrap();
        inner
            .runtime_config
            .runtime_data
            .remove(GUIDED_OUTPUT_REASONING_RUNTIME_KEY);
        inner.runtime_config.reasoning_parser = Some("glm47".into());
        let request = request(json!({"guided_choice":["yes"]}));
        let mut common = processor
            .builder_with_lora(&request, None)
            .unwrap()
            .token_ids(vec![])
            .build()
            .unwrap();
        assert!(
            processor
                .resolve_guided_output_plan(
                    &request,
                    &mut common,
                    &false.into(),
                    &GuidedToolConstraint::None
                )
                .is_err(),
            "an unknown default must not silently suppress reasoning"
        );
    }

    #[test]
    fn guided_output_honors_engine_only_reasoning_and_direct_controls() {
        for tool in [None, Some("qwen3_coder"), Some("kimi_k2")] {
            for mode in [
                GuidedReasoningMode::ExplicitGate,
                GuidedReasoningMode::Unsupported,
                GuidedReasoningMode::Immediate,
            ] {
                let mut capability = gate();
                capability.mode = mode;
                let mut processor = processor(Some(capability));
                let inner = std::sync::Arc::get_mut(&mut processor).unwrap();
                inner.runtime_config.reasoning_parser = None;
                inner.tool_call_parser = tool.map(str::to_owned);
                for thinking in [None, Some(true), Some(false)] {
                    let mut request = request(json!({"guided_choice":["yes"]}));
                    request.chat_template_args = thinking.map(|value| {
                        std::collections::HashMap::from([("enable_thinking".into(), json!(value))])
                    });
                    let mut common = processor
                        .builder_with_lora(&request, None)
                        .unwrap()
                        .token_ids(vec![])
                        .build()
                        .unwrap();
                    let result = processor.resolve_guided_output_plan(
                        &request,
                        &mut common,
                        &false.into(),
                        &GuidedToolConstraint::None,
                    );
                    if thinking == Some(false) {
                        assert_eq!(result.unwrap(), Some(GuidedOutputPlan::Direct));
                        assert!(!common.require_reasoning);
                        assert_eq!(common.extra_args.as_ref().unwrap()["reasoning_ended"], true);
                    } else if mode == GuidedReasoningMode::ExplicitGate {
                        assert!(matches!(
                            result.unwrap(),
                            Some(GuidedOutputPlan::EngineGated(_))
                        ));
                        assert!(common.require_reasoning);
                        assert_eq!(
                            common.extra_args.as_ref().unwrap()["reasoning_ended"],
                            false
                        );
                    } else {
                        assert!(result.unwrap_err().to_string().contains("cannot preserve"));
                    }
                }
            }
        }
    }

    #[test]
    fn guided_output_malformed_legacy_gate_metadata_has_a_configuration_error() {
        let mut processor = processor(None);
        for key in [
            "tool_call_structural_tag_excludes_reasoning",
            "tool_call_structural_tag_reasoning_gate",
        ] {
            std::sync::Arc::get_mut(&mut processor)
                .unwrap()
                .runtime_config
                .runtime_data
                .insert(key.into(), json!("malformed"));
            let mut request = request(json!({"guided_regex":"yes"}));
            request.chat_template_args = Some(std::collections::HashMap::from([(
                "enable_thinking".into(),
                json!(false),
            )]));
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
            assert!(error.to_string().contains("cannot preserve"), "{error}");
            std::sync::Arc::get_mut(&mut processor)
                .unwrap()
                .runtime_config
                .runtime_data
                .remove(key);
        }
    }

    #[tokio::test]
    async fn guided_output_native_composites_validate_the_actual_engine_gate() {
        if crate::test_utils::run_isolated(
            concat!(
                module_path!(),
                "::guided_output_native_composites_validate_the_actual_engine_gate"
            ),
            &[(
                dynamo_runtime::config::environment_names::llm::DYN_PARSER_VERSION,
                "2",
            )],
        ) {
            return;
        }
        for (reasoner, tool, capability) in [
            ("minimax_m3", "minimax_m3", None),
            (
                "minimax_m3",
                "minimax_m3",
                Some(GuidedReasoningMode::Unsupported),
            ),
            (
                "qwen3",
                "qwen3_coder",
                Some(GuidedReasoningMode::ExplicitGate),
            ),
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
            let mut card =
                crate::model_card::ModelDeploymentCard::load_from_disk(path, None).unwrap();
            card.runtime_config.reasoning_parser = Some(reasoner.into());
            card.runtime_config.tool_call_parser = Some(tool.into());
            card.runtime_config.structural_tag =
                Some(crate::local_model::runtime_config::StructuralTagConfig {
                    allow_tool_calls_with_structured_output: true,
                    ..Default::default()
                });
            card.runtime_config
                .set_engine_specific("tool_call_structural_tag_excludes_reasoning", true)
                .unwrap();
            if let Some(mode) = capability {
                let mut value = gate();
                value.mode = mode;
                card.runtime_config
                    .set_engine_specific(GUIDED_OUTPUT_REASONING_RUNTIME_KEY, value)
                    .unwrap();
            }
            let processor = OpenAIPreprocessor::new(card).unwrap();
            for format in [
                json!({"guided_json":{"type":"object"}}),
                json!({"response_format":{"type":"json_schema","json_schema":{"name":"guided_json","schema":{"type":"object"}}}}),
            ] {
                let mut fields = json!({"chat_template_kwargs":{"thinking_mode":"enabled","enable_thinking":true},"tool_choice":"auto","tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"type":"string","enum":["Paris"]}}}}}]});
                fields
                    .as_object_mut()
                    .unwrap()
                    .extend(format.as_object().unwrap().clone());
                let mut request = request(fields);
                normalize_guided_json(&mut request).unwrap();
                let mut common = processor
                    .builder_with_lora(&request, None)
                    .unwrap()
                    .token_ids(vec![])
                    .build()
                    .unwrap();
                let prefill: PromptReasoningPrefill = true.into();
                let installed = processor
                    .apply_tool_choice_guided_decoding(&request, &mut common, prefill.clone())
                    .unwrap();
                assert!(installed.uses_structural_tag());
                assert!(common.native_json_requires_reasoning_gate);
                let result = processor.resolve_guided_output_plan(
                    &request,
                    &mut common,
                    &prefill,
                    &installed,
                );
                if capability == Some(GuidedReasoningMode::ExplicitGate) {
                    assert_eq!(result.unwrap(), Some(GuidedOutputPlan::NativeJson));
                    assert!(common.guided_output_plan.is_none());
                    assert!(common.require_reasoning);
                    assert_eq!(
                        common.extra_args.as_ref().unwrap()["reasoning_ended"],
                        false
                    );
                } else {
                    assert!(result.unwrap_err().to_string().contains("cannot preserve"));
                }
            }
        }
    }

    #[test]
    fn guided_output_merges_effective_glm_kwargs_and_omits_unconfigured_kwargs() {
        let mut processor = processor(Some(gate()));
        std::sync::Arc::get_mut(&mut processor)
            .unwrap()
            .runtime_config
            .reasoning_parser = Some("glm47".into());
        let mut request = request(json!({"guided_regex":"yes"}));
        request.chat_template_args = Some(std::collections::HashMap::from([
            ("thinking".into(), json!(true)),
            ("other".into(), json!("kept")),
        ]));
        let mut common = processor
            .builder_with_lora(&request, None)
            .unwrap()
            .token_ids(vec![])
            .build()
            .unwrap();
        common.extra_args = Some(
            json!({"reasoning_parser_kwargs":{"extension":42,"chat_template_kwargs":{"thinking":false,"enable_thinking":false,"existing":"kept"}}}),
        );
        let mut prefill: PromptReasoningPrefill = false.into();
        prefill.reasoning_closed_by_prompt = Some(true);
        assert_eq!(
            processor
                .resolve_guided_output_plan(
                    &request,
                    &mut common,
                    &prefill,
                    &GuidedToolConstraint::None
                )
                .unwrap(),
            Some(GuidedOutputPlan::Direct)
        );
        let kwargs = &common.extra_args.as_ref().unwrap()["reasoning_parser_kwargs"];
        assert_eq!(kwargs["extension"], 42);
        assert_eq!(kwargs["chat_template_kwargs"]["thinking"], false);
        assert_eq!(kwargs["chat_template_kwargs"]["enable_thinking"], false);
        assert_eq!(kwargs["chat_template_kwargs"]["existing"], "kept");
        assert_eq!(kwargs["chat_template_kwargs"]["other"], "kept");

        let inner = std::sync::Arc::get_mut(&mut processor).unwrap();
        inner.runtime_config.reasoning_parser = None;
        inner
            .runtime_config
            .runtime_data
            .remove(GUIDED_OUTPUT_REASONING_RUNTIME_KEY);
        let mut common = processor
            .builder_with_lora(&request, None)
            .unwrap()
            .token_ids(vec![])
            .build()
            .unwrap();
        processor
            .resolve_guided_output_plan(
                &request,
                &mut common,
                &false.into(),
                &GuidedToolConstraint::None,
            )
            .unwrap();
        assert!(
            common
                .extra_args
                .as_ref()
                .unwrap()
                .get("reasoning_parser_kwargs")
                .is_none()
        );
    }

    #[tokio::test]
    async fn guided_output_beam_final_prefill_consumes_model_eos_without_a_parser() {
        use crate::protocols::{
            Annotated,
            common::llm_backend::LLMEngineOutput,
            openai::{DeltaGeneratorExt, ParsingOptions, chat_completions::DeltaAggregator},
        };
        use dynamo_runtime::pipeline::{
            AsyncEngine, AsyncEngineContextProvider, ManyOut, Operator, ResponseStream, SingleIn,
            async_trait,
        };
        use futures::StreamExt;
        use std::sync::{Arc, atomic::AtomicBool};

        struct Tokens(Vec<u32>);
        #[async_trait]
        impl
            AsyncEngine<
                SingleIn<PreprocessedRequest>,
                ManyOut<Annotated<LLMEngineOutput>>,
                anyhow::Error,
            > for Tokens
        {
            async fn generate(
                &self,
                request: SingleIn<PreprocessedRequest>,
            ) -> anyhow::Result<ManyOut<Annotated<LLMEngineOutput>>> {
                let output = LLMEngineOutput {
                    token_ids: self.0.clone(),
                    finish_reason: Some(crate::protocols::common::FinishReason::EoS),
                    ..Default::default()
                };
                Ok(ResponseStream::new(
                    Box::pin(futures::stream::iter([Annotated::from_data(output)])),
                    request.context(),
                ))
            }
        }

        for no_stop_trim in [false, true] {
            let mut immediate = gate();
            immediate.mode = GuidedReasoningMode::Immediate;
            immediate.reasoner = None;
            let mut processor = processor(Some(immediate));
            let inner = Arc::get_mut(&mut processor).unwrap();
            inner.beam_route = true;
            inner.tool_call_parser = Some("beam".into());
            inner.runtime_config.tool_call_parser = Some("beam".into());
            inner.runtime_config.reasoning_parser = Some("beam".into());
            // The sample card's reduced tokenizer has no supplier EOS IDs.
            // Install a tiny real decoder with an explicit Beam model EOS.
            let hf: tokenizers::Tokenizer = serde_json::from_value(json!({
                "version":"1.0","truncation":null,"padding":null,
                "added_tokens":[{"id":1,"content":"<|return|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],
                "normalizer":null,"pre_tokenizer":null,"post_processor":null,
                "decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"}]},
                "model":{"type":"BPE","vocab":{"yes":0,"<|return|>":1},"merges":[],"byte_fallback":true}
            })).unwrap();
            inner.tokenizer = Arc::new(crate::tokenizers::HuggingFaceTokenizer::from_tokenizer(hf));
            let request = request(json!({
                "messages":[{"role":"user","content":"answer yes"},{"role":"assistant","reasoning_content":"already thought","content":""}],
                "continue_final_message":true,"add_generation_prompt":false,
                "guided_choice":["yes"],"skip_special_tokens":false,
            }));
            let mut common = processor
                .builder_with_lora(&request, None)
                .unwrap()
                .token_ids(vec![])
                .build()
                .unwrap();
            common.eos_token_ids = vec![1];
            common.stop_conditions.stop_token_ids_hidden = Some(vec![1]);
            common.output_options.no_stop_trim = Some(no_stop_trim);
            assert_eq!(
                common.output_options.retain_model_stop_token_text,
                Some(true)
            );
            let mut prefill = OpenAIPreprocessor::prompt_reasoning_prefill_for_parsers(
                Some("beam"),
                Some("beam"),
                Some("<|start|>assistant<|channel|>final<|message|>"),
            );
            let plan = processor
                .resolve_guided_output_plan(
                    &request,
                    &mut common,
                    &prefill,
                    &GuidedToolConstraint::None,
                )
                .unwrap();
            assert_eq!(plan, Some(GuidedOutputPlan::Direct));
            prefill.guided_output_plan = plan;
            let eos = 1;
            let eos_text: String = processor.tokenizer.decode(&[eos], false).unwrap().into();
            assert!(!eos_text.is_empty());
            let mut ids = vec![0];
            ids.push(eos);
            let backend = crate::backend::Backend::from_tokenizer(
                crate::tokenizers::Tokenizer::from(processor.tokenizer.clone()),
            );
            let mut stream = Operator::generate(
                backend.as_ref(),
                SingleIn::new(common),
                Arc::new(Tokens(ids)),
            )
            .await
            .unwrap();
            let mut generator = request.response_generator("beam-guided".into());
            let mut chunks = Vec::new();
            while let Some(chunk) = stream.next().await {
                chunks.push(Annotated::from_data(
                    generator
                        .choice_from_postprocessor(chunk.data.unwrap())
                        .unwrap(),
                ));
            }
            chunks.push(Annotated::from_data(generator.create_usage_chunk()));
            let parsed = processor
                .postprocessor_parsing_stream(
                    futures::stream::iter(chunks),
                    &request,
                    prefill,
                    false,
                )
                .unwrap();
            let mut options = ParsingOptions::new(Some("beam".into()), Some("beam".into()));
            options.guided_answer_projected = Some(Arc::new(AtomicBool::new(true)));
            let response = DeltaAggregator::apply(parsed, options).await.unwrap();
            let expected = if no_stop_trim {
                format!("yes{eos_text}")
            } else {
                "yes".into()
            };
            assert_eq!(
                response.inner.choices[0].message.content,
                Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                    expected
                ))
            );
            assert!(
                response.inner.choices[0]
                    .message
                    .reasoning_content
                    .is_none()
            );
            assert_eq!(
                response
                    .inner
                    .usage
                    .unwrap()
                    .completion_tokens_details
                    .unwrap()
                    .reasoning_tokens,
                Some(0)
            );
        }
    }
}
