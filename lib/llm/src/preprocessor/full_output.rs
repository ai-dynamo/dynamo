// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Proof of concept: one grammar owns reasoning and the final output.

use crate::protocols::common::GuidedDecodingOptions;
use serde_json::{Value, json};

/// The rendered prompt determines whether the model must emit the opener.
#[derive(Clone, Copy, Debug)]
pub(super) enum ReasoningPhase {
    Disabled,
    Generated {
        begin: &'static str,
        end: &'static str,
    },
    PromptOpened {
        end: &'static str,
    },
}

/// Convert the final-output constraint without changing its response format.
/// Tool markup remains tool markup; a JSON argument object remains bare JSON.
pub(super) fn apply_full_output_grammar(
    guided: &mut GuidedDecodingOptions,
    phase: ReasoningPhase,
) -> Result<bool, String> {
    let constraint_count = usize::from(guided.json.is_some())
        + usize::from(guided.structural_tag.is_some())
        + usize::from(guided.regex.is_some())
        + usize::from(guided.choice.is_some())
        + usize::from(guided.grammar.is_some());
    if constraint_count == 0 {
        return Ok(false);
    }
    if constraint_count != 1 {
        return Err("full-output grammar requires exactly one final-output constraint".into());
    }
    if guided.grammar.is_some() || guided.whitespace_pattern.is_some() {
        return Err(
            "full-output grammar POC does not support guided_grammar or guided_whitespace_pattern"
                .into(),
        );
    }
    if guided
        .backend
        .as_deref()
        .is_some_and(|backend| backend != "xgrammar")
    {
        return Err("full-output grammar POC requires the xgrammar backend".into());
    }

    let final_format = if let Some(schema) = &guided.json {
        json!({"type": "json_schema", "json_schema": schema, "style": "json"})
    } else if let Some(tag) = &guided.structural_tag {
        if tag.get("type").and_then(Value::as_str) != Some("structural_tag") {
            return Err("full-output grammar requires a structural_tag with a format node".into());
        }
        tag.get("format")
            .filter(|format| format.is_object())
            .cloned()
            .ok_or("full-output grammar requires a structural_tag with a format node")?
    } else if let Some(regex) = &guided.regex {
        json!({"type": "regex", "pattern": regex})
    } else {
        let choices = guided.choice.as_ref().expect("one supported constraint");
        if choices.is_empty() {
            return Err("full-output grammar requires a nonempty guided_choice".into());
        }
        json!({"type": "or", "elements": choices.iter().map(|choice| {
            json!({"type": "const_string", "value": choice})
        }).collect::<Vec<_>>()})
    };

    let format = match phase {
        ReasoningPhase::Disabled => final_format,
        ReasoningPhase::Generated { end, .. } | ReasoningPhase::PromptOpened { end, .. } => {
            let begin = match phase {
                ReasoningPhase::Generated { begin, .. } => begin,
                _ => "",
            };
            json!({"type": "sequence", "elements": [
                {"type": "tag", "begin": begin,
                 "content": {"type": "any_text", "excludes": []}, "end": end},
                {"type": "regex", "pattern": "\\s*"},
                final_format
            ]})
        }
    };

    guided.structural_tag = Some(json!({"type": "structural_tag", "format": format}));
    guided.json = None;
    guided.regex = None;
    guided.choice = None;
    Ok(true)
}

use super::{OpenAIPreprocessor, PreprocessedRequest, PromptReasoningPrefill};
use crate::local_model::runtime_config::VLLM_INFERENCE_V1_GENERATE_CAPABILITY;
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use dynamo_runtime::error::{DynamoError, ErrorType};

pub(super) fn reasoning_markers(parser: Option<&str>) -> Option<(&'static str, &'static str)> {
    match parser {
        Some("qwen3" | "minimax_m2" | "deepseek_v41") => Some(("<think>", "</think>")),
        Some("minimax_m3" | "minimax-m3") => Some(("<mm:think>", "</mm:think>")),
        Some("kimi_k3" | "kimi-k3") => Some(("<|open|>think<|sep|>", "<|close|>think<|sep|>")),
        _ => None,
    }
}

impl OpenAIPreprocessor {
    pub(super) fn uses_full_output_grammar(&self) -> bool {
        self.runtime_config
            .runtime_flag_enabled(VLLM_INFERENCE_V1_GENERATE_CAPABILITY)
            && self.runtime_config.reasoning_parser.is_some()
    }

    pub(super) fn apply_reasoning_to_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_prefill: PromptReasoningPrefill,
    ) -> Result<(), DynamoError> {
        if !self.uses_full_output_grammar() {
            return Ok(());
        }
        let Some(guided) = common_request.sampling_options.guided_decoding.as_mut() else {
            return Ok(());
        };
        let invalid_argument = |message: String| {
            DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .message(message)
                .build()
        };
        // Explicit profiles keep this POC from guessing markers for other families.
        let parser = self.runtime_config.reasoning_parser.as_deref();
        let (begin, end) = reasoning_markers(parser).ok_or_else(|| {
            invalid_argument(format!(
                "full-output grammar POC does not support reasoning parser {parser:?}"
            ))
        })?;
        let disabled =
            dynamo_renderer::thinking_bool_from_args(request.chat_template_args.as_ref())
                == Some(false)
                || Self::is_reasoning_disabled_by_request(
                    self.runtime_config.reasoning_parser.as_deref(),
                    request.chat_template_args.as_ref(),
                );
        let phase = if disabled || prompt_prefill.closed {
            ReasoningPhase::Disabled
        } else if prompt_prefill.unified {
            ReasoningPhase::PromptOpened { end }
        } else {
            ReasoningPhase::Generated { begin, end }
        };
        if apply_full_output_grammar(guided, phase).map_err(invalid_argument)? {
            // The backend starts this grammar at token one. It needs no gate state.
            common_request.require_reasoning = false;
            if let Some(extra_args) = common_request
                .extra_args
                .as_mut()
                .and_then(Value::as_object_mut)
            {
                extra_args.remove("reasoning_parser_kwargs");
                extra_args.remove("reasoning_ended");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_output_prompt_state_follows_rendered_markers() {
        for parser in [
            "qwen3",
            "minimax_m2",
            "minimax_m3",
            "kimi_k3",
            "deepseek_v41",
        ] {
            let (begin, end) = reasoning_markers(Some(parser)).unwrap();
            for (suffix, open, closed) in [
                (begin, true, false),
                (end, false, true),
                ("assistant", false, false),
            ] {
                let prompt = format!("user\nassistant\n{suffix}\n\n");
                let state = OpenAIPreprocessor::prompt_reasoning_prefill_for_parsers(
                    None,
                    Some(parser),
                    Some(&prompt),
                );
                assert_eq!(state.unified, open, "{parser} {suffix}");
                assert_eq!(state.closed, closed, "{parser} {suffix}");
            }
        }
    }

    #[test]
    fn full_output_schema_keeps_the_schema_and_removes_json_guidance() {
        let schema = json!({"type": "object", "properties": {"answer": {"type": "integer"}},
            "required": ["answer"], "additionalProperties": false});
        let mut guided = GuidedDecodingOptions {
            json: Some(schema.clone()),
            ..Default::default()
        };
        apply_full_output_grammar(
            &mut guided,
            ReasoningPhase::Generated {
                begin: "<think>",
                end: "</think>",
            },
        )
        .unwrap();
        assert!(guided.json.is_none());
        let tag = guided.structural_tag.unwrap();
        assert_eq!(tag["format"]["elements"][2]["json_schema"], schema);
        assert_eq!(tag["format"]["elements"][0]["begin"], "<think>");
    }

    #[test]
    fn full_output_prompt_open_does_not_repeat_the_opener() {
        let mut guided = GuidedDecodingOptions {
            json: Some(json!({"type": "object"})),
            ..Default::default()
        };
        apply_full_output_grammar(
            &mut guided,
            ReasoningPhase::PromptOpened { end: "</think>" },
        )
        .unwrap();
        assert_eq!(
            guided.structural_tag.unwrap()["format"]["elements"][0]["begin"],
            ""
        );
    }

    #[test]
    fn full_output_disabled_keeps_only_the_final_constraint() {
        let mut guided = GuidedDecodingOptions {
            json: Some(json!({"type": "object"})),
            ..Default::default()
        };
        apply_full_output_grammar(&mut guided, ReasoningPhase::Disabled).unwrap();
        assert_eq!(
            guided.structural_tag.unwrap()["format"]["type"],
            "json_schema"
        );
    }

    #[test]
    fn full_output_preserves_native_tool_markup() {
        let final_format = json!({"type": "tag", "begin": "<tool_call>",
            "content": {"type": "json_schema", "json_schema": {"type": "object"}}, "end": "</tool_call>"});
        let mut guided = GuidedDecodingOptions {
            structural_tag: Some(json!({"type": "structural_tag", "format": final_format})),
            ..Default::default()
        };
        apply_full_output_grammar(
            &mut guided,
            ReasoningPhase::PromptOpened { end: "</think>" },
        )
        .unwrap();
        assert_eq!(
            guided.structural_tag.unwrap()["format"]["elements"][2],
            final_format
        );
    }

    #[test]
    fn full_output_rejects_conflicting_constraints_without_mutation() {
        let mut guided = GuidedDecodingOptions {
            json: Some(json!({"type": "object"})),
            regex: Some(".*".into()),
            ..Default::default()
        };
        let original = serde_json::to_value(&guided).unwrap();
        assert!(apply_full_output_grammar(&mut guided, ReasoningPhase::Disabled).is_err());
        assert_eq!(serde_json::to_value(&guided).unwrap(), original);
    }
}
