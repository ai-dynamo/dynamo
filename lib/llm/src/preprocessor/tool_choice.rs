// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

use crate::local_model::runtime_config::SGLANG_EBNF_FULL_ASSISTANT_CAPABILITY;
use crate::preprocessor::structural_tag::requires_native_tool_call_format;
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::tools::{
    ToolChoiceGuidance, get_tool_choice_guidance_from_tools, validate_openai_tool_choice,
};
use crate::protocols::openai::{GuidedToolConstraint, validate};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, ResponseFormat,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use serde_json::Value;
use sha2::{Digest, Sha256};

fn invalid_argument(message: impl Into<String>, public_message: Option<&str>) -> DynamoError {
    let mut error = DynamoError::builder()
        .error_type(ErrorType::InvalidArgument)
        .message(message);
    if let Some(public_message) = public_message {
        error = error.public_message(public_message);
    }
    error.build()
}

impl OpenAIPreprocessor {
    /// Whether this request permits model output to be interpreted as tool calls.
    ///
    /// A configured parser describes the model's wire format; it does not grant
    /// every request permission to return tool calls. Permission depends only on
    /// whether the request supplies effective tools and whether `tool_choice`
    /// forbids them. Effective tools include both the OpenAI top-level list and
    /// Kimi-style dynamic declarations carried by system messages.
    /// Assistant-output constraints apply to assistant content and do not revoke
    /// an `auto` request's ability to choose a tool call.
    pub(crate) fn tool_call_parsing_enabled(request: &NvCreateChatCompletionRequest) -> bool {
        if !request.inner.has_effective_tools() {
            return false;
        }

        match request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto)
        {
            ChatCompletionToolChoiceOption::None => false,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_) => {
                true
            }
            ChatCompletionToolChoiceOption::Auto => true,
        }
    }

    /// Apply guided decoding for OpenAI tool-choice requests.
    ///
    /// Structural tags are preferred when enabled and supported by the configured
    /// tool-call parser. When disabled or unavailable, ordinary forced choices
    /// retain the legacy JSON-schema fallback. Kimi's marker-delimited forced
    /// formats cannot be represented by that fallback and remain on their
    /// prompt/parser best-effort path instead.
    pub(super) fn apply_tool_choice_guided_decoding(
        &self,
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
        prompt_injected_reasoning: bool,
    ) -> Result<GuidedToolConstraint, DynamoError> {
        let tool_choice = request
            .inner
            .tool_choice
            .as_ref()
            .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
        let tools = effective_tools(&request.inner)?;
        let is_forced_tool_choice = matches!(
            tool_choice,
            ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
        );
        let has_explicit_guided_decoding = has_explicit_guided_decoding(request);
        let has_response_format_constraint = has_response_format_constraint(request);
        let parser_tool_choice = convert_tool_choice(tool_choice);

        self.validate_structured_output_composition(request, tool_choice, tools.as_ref())?;

        if is_forced_tool_choice && has_explicit_guided_decoding {
            return Err(invalid_argument(
                concat!(
                    "guided decoding cannot be used in the same request as ",
                    "tool_choice=\"required\" or a named tool_choice.",
                ),
                None,
            ));
        }

        // Explicit guided decoding constrains assistant content and cannot be
        // composed with structural-tag tool calls.
        if !is_forced_tool_choice && has_explicit_guided_decoding {
            return Ok(GuidedToolConstraint::None);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        // A full-assistant grammar covers reasoning, prose, and tool-call XML,
        // so it cannot compose with a separate assistant response-format schema.
        if !has_response_format_constraint
            && self.apply_full_assistant_ebnf(
                tool_choice,
                &convert_tools(tools.as_ref()),
                request,
                common_request,
            )?
        {
            return Ok(GuidedToolConstraint::None);
        }

        if self.apply_tool_choice_structural_tag(
            &parser_tool_choice,
            &convert_tools(tools.as_ref()),
            request.inner.parallel_tool_calls,
            prompt_injected_reasoning,
            request.inner.response_format.as_ref(),
            common_request,
        )? {
            return Ok(GuidedToolConstraint::StructuralTag);
        }

        // If the structural-tag layer did not compose this response format with
        // auto tool calls, it remains the sole assistant-output constraint.
        if !is_forced_tool_choice && has_response_format_constraint {
            return Ok(GuidedToolConstraint::None);
        }

        let requires_native_format =
            requires_native_tool_call_format(self.tool_call_parser.as_deref(), &parser_tool_choice)
                || requires_native_tool_call_format(
                    self.runtime_config.reasoning_parser.as_deref(),
                    &parser_tool_choice,
                );
        if is_forced_tool_choice && requires_native_format {
            tracing::debug!(
                tool_choice = ?parser_tool_choice,
                "Structural tag was not applied and the parser uses a native tool-call format; using prompt/parser compatibility fallback"
            );
            return Ok(GuidedToolConstraint::None);
        }

        match get_tool_choice_guidance_from_tools(
            Some(tool_choice),
            Some(tools.as_ref()),
            request.inner.parallel_tool_calls,
        ) {
            Ok(Some(guidance)) => {
                let gd = common_request
                    .sampling_options
                    .guided_decoding
                    .get_or_insert_default();
                match guidance {
                    ToolChoiceGuidance::Json(schema) => gd.json = Some(schema),
                    ToolChoiceGuidance::Regex(regex) => gd.regex = Some(regex),
                }

                // Report the parser constraint implied by the installed grammar,
                // not merely the tool_choice that requested it. Both JSON and
                // regex guidance still use the guided-JSON tool-output parser.
                return Ok(installed_json_constraint(tool_choice));
            }
            Ok(None) => {}
            Err(err) => {
                return Err(invalid_argument(err.to_string(), None));
            }
        }

        // Auto/None requests can reach here when neither structural tags nor a
        // tool-choice JSON fallback were needed.
        Ok(GuidedToolConstraint::None)
    }

    fn validate_structured_output_composition(
        &self,
        request: &NvCreateChatCompletionRequest,
        tool_choice: &ChatCompletionToolChoiceOption,
        tools: &[ChatCompletionTool],
    ) -> Result<(), DynamoError> {
        let allow_tool_calls_with_structured_output = self
            .runtime_config
            .structural_tag
            .as_ref()
            .is_some_and(|config| config.allow_tool_calls_with_structured_output);

        if matches!(tool_choice, ChatCompletionToolChoiceOption::Auto)
            && !tools.is_empty()
            && allow_tool_calls_with_structured_output
        {
            validate::validate_response_format_conflicts(
                &request.inner.response_format,
                &request.common,
            )
            .map_err(|error| {
                let message = error.to_string();
                invalid_argument(&message, Some(&message))
            })?;
        }

        Ok(())
    }

    /// The protocol and installation path are parser-agnostic. A parser must
    /// provide a grammar for its complete native assistant format before this
    /// path is engaged.
    fn apply_full_assistant_ebnf(
        &self,
        tool_choice: &ChatCompletionToolChoiceOption,
        tools: &[ToolDefinition],
        request: &NvCreateChatCompletionRequest,
        common_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        let tools_are_non_strict = tools.iter().all(|tool| !tool.strict.unwrap_or(false));
        if tools.is_empty()
            || !tools_are_non_strict
            || !self
                .runtime_config
                .supports_runtime_capability(SGLANG_EBNF_FULL_ASSISTANT_CAPABILITY)
        {
            return Ok(false);
        }
        let functions = full_assistant_tools(tool_choice, tools);
        if functions.is_empty() {
            return Ok(false);
        }

        let thinking_mode = request
            .chat_template_args
            .as_ref()
            .and_then(|args| args.get("enable_thinking"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let Some(grammar) = generate_full_assistant_grammar(
            self.tool_call_parser.as_deref(),
            &functions,
            thinking_mode,
            request.inner.parallel_tool_calls.unwrap_or(true),
            matches!(
                tool_choice,
                ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
            ),
        ) else {
            return Ok(false);
        };

        let gd = common_request
            .sampling_options
            .guided_decoding
            .get_or_insert_default();
        gd.grammar = Some(grammar);
        gd.ebnf_full_assistant = true;
        Ok(true)
    }
}

fn full_assistant_tools<'a>(
    tool_choice: &ChatCompletionToolChoiceOption,
    tools: &'a [ToolDefinition],
) -> Vec<&'a ToolDefinition> {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => Vec::new(),
        ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::Required => {
            tools.iter().collect()
        }
        ChatCompletionToolChoiceOption::Named(named) => tools
            .iter()
            .filter(|tool| tool.name == named.function.name)
            .collect(),
    }
}

fn generate_full_assistant_grammar(
    parser_name: Option<&str>,
    functions: &[&ToolDefinition],
    enable_thinking: bool,
    parallel_tool_calls: bool,
    required: bool,
) -> Option<String> {
    match parser_name {
        Some("glm47") => Some(generate_glm47_full_assistant_grammar(
            functions,
            enable_thinking,
            parallel_tool_calls,
            required,
        )),
        _ => None,
    }
}

fn generate_glm47_full_assistant_grammar(
    functions: &[&ToolDefinition],
    enable_thinking: bool,
    parallel_tool_calls: bool,
    required: bool,
) -> String {
    const BEGIN_TOOL_CALL: &str = "<tool_call>";
    const END_TOOL_CALL: &str = "</tool_call>";
    const BEGIN_KEY: &str = "<arg_key>";
    const END_KEY: &str = "</arg_key>";
    const BEGIN_VALUE: &str = "<arg_value>";
    const END_VALUE: &str = "</arg_value>";
    const END_THINKING: &str = "</think>";
    const ASSISTANT_TOKEN: &str = "<|assistant|>";

    let mut lines = vec![
        "root ::= assistant_turn".to_string(),
        "assistant_turn ::= thinking_block text_block tool_call_blocks".to_string(),
    ];
    let thinking_exclusions = [
        BEGIN_TOOL_CALL,
        END_TOOL_CALL,
        BEGIN_KEY,
        END_KEY,
        BEGIN_VALUE,
        END_VALUE,
        END_THINKING,
    ];
    if enable_thinking {
        lines.push(format!(
            "thinking_block ::= thinking_block_content \"{END_THINKING}\""
        ));
        lines.extend(any_string_exclude(
            "thinking_block_content",
            &thinking_exclusions,
        ));
    } else {
        lines.push("thinking_block ::= \"\"".to_string());
    }

    lines.extend(any_string_exclude(
        "text_without_special_tokens",
        &[
            "<think>",
            END_THINKING,
            BEGIN_TOOL_CALL,
            END_TOOL_CALL,
            BEGIN_KEY,
            END_KEY,
            BEGIN_VALUE,
            END_VALUE,
            ASSISTANT_TOKEN,
        ],
    ));
    lines.push("text_block ::= (  text_without_special_tokens )?".to_string());

    if functions.is_empty() {
        lines.push("tool_call_blocks ::= \"\"".to_string());
    } else {
        let repetition = if parallel_tool_calls {
            if required { "+" } else { "*" }
        } else if required {
            ""
        } else {
            "?"
        };
        lines.push(format!(
            "tool_call_blocks ::= (  tool_call_unit ){repetition}"
        ));
        lines.push(format!(
            "tool_call_unit ::= \"{BEGIN_TOOL_CALL}\" single_tool_call \"{END_TOOL_CALL}\""
        ));

        let alternatives = functions
            .iter()
            .enumerate()
            .map(|(index, tool)| {
                format!("call_{}", glm_hash_name(&format!("{}{index}", tool.name)))
            })
            .collect::<Vec<_>>()
            .join(" | ");
        lines.push(format!("single_tool_call ::= {alternatives}"));

        for (index, tool) in functions.iter().enumerate() {
            let name_hash = glm_hash_name(&format!("{}{index}", tool.name));
            let params = tool.parameters.as_ref().unwrap_or(&Value::Null);
            let mut properties = schema_properties(params);
            if !has_complete_properties(params) {
                properties.clear();
            }
            let uses_composition = ["allOf", "anyOf", "oneOf"]
                .iter()
                .any(|keyword| params.as_object().is_some_and(|p| p.contains_key(*keyword)));

            let pairs = properties
                .iter()
                .map(|(name, schema)| {
                    let value_rule = if uses_composition {
                        "text_without_special_tokens".to_string()
                    } else {
                        value_rule(schema)
                    };
                    format!(
                        "\"{BEGIN_KEY}{}{END_KEY}\"  \"{BEGIN_VALUE}\" ({value_rule}) \"{END_VALUE}\"",
                        ebnf_string_inner(name)
                    )
                })
                .collect::<Vec<_>>();

            let arguments_rule = if pairs.is_empty() {
                format!(
                    "( \"{BEGIN_KEY}\" text_without_special_tokens \"{END_KEY}\"  \"{BEGIN_VALUE}\" text_without_special_tokens \"{END_VALUE}\"  )*"
                )
            } else {
                let choices = pairs.join(" | ");
                format!("( ( {choices} ) (  ( {choices} ) )* )?")
            };

            lines.push(format!(
                "call_{name_hash} ::= \"{}\"  ( arguments_{name_hash}  )?",
                ebnf_string_inner(&tool.name)
            ));
            lines.push(format!("arguments_{name_hash} ::= {arguments_rule}"));
        }

        lines.extend([
            r#"basic_string ::= (([\"] basic_string_1 [\"]))"#,
            r#"basic_string_1 ::= "" | [^"\\\x00-\x1F] basic_string_1 | "\\" escape basic_string_1"#,
            r#"escape ::= ["\\//bfnrt] | "u" [A-Fa-f0-9]{4}"#,
            r#"basic_integer ::= "-"? ("0" | [1-9] [0-9]*) ".0"?"#,
            r#"basic_number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?"#,
            r#"basic_array ::= "[" ("" | ws basic_any (ws "," ws basic_any)*) ws "]""#,
            r#"basic_object ::= "{" ("" | ws basic_string ws ":" ws basic_any ( ws "," ws basic_string ws ":" ws basic_any)*) ws "}""#,
            r#"ws ::= [ \n\t]*"#,
            r#"basic_any ::= basic_number | basic_string | basic_boolean | basic_null | basic_array | basic_object"#,
            r#"basic_boolean ::= "true" | "false""#,
            r#"basic_null ::= "null""#,
        ]
        .into_iter()
        .map(String::from));
    }

    lines.join("\n")
}

fn glm_hash_name(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    let digest = format!("{digest:x}");
    digest[..16].to_string()
}

fn ebnf_string_inner(value: &str) -> String {
    let mut output = String::new();
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{8}' => output.push_str("\\b"),
            '\u{c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if (character as u32) < 0x20 => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output
}

fn value_rule(schema: &Value) -> String {
    let Some(schema) = schema.as_object() else {
        return "text_without_special_tokens".to_string();
    };
    if let Some(values) = schema.get("enum").and_then(Value::as_array) {
        let formatted = values
            .iter()
            .map(|value| {
                let json_value = match value {
                    Value::String(string) => ebnf_string_inner(string),
                    other => ebnf_string_inner(
                        &serde_json::to_string(other).expect("serde_json values always serialize"),
                    ),
                };
                format!("\"{json_value}\"")
            })
            .collect::<Vec<_>>();
        if formatted.is_empty() {
            return "text_without_special_tokens".to_string();
        }
        let rule = formatted.join(" | ");
        return if formatted.len() > 1 {
            format!("({rule})")
        } else {
            rule
        };
    }

    let Some(schema_type) = schema.get("type") else {
        return "text_without_special_tokens".to_string();
    };
    let mapped = |value: &Value| match value.as_str() {
        Some("string") => "text_without_special_tokens",
        Some("number" | "integer") => "basic_number",
        Some("boolean") => "basic_boolean",
        Some("null") => "basic_null",
        Some("array") => "basic_array",
        Some("object") => "basic_object",
        _ => "text_without_special_tokens",
    };
    match schema_type {
        Value::Array(types) => {
            let rules = types.iter().map(mapped).collect::<Vec<_>>();
            if rules.is_empty() {
                "text_without_special_tokens".to_string()
            } else {
                rules.join(" | ")
            }
        }
        value => mapped(value).to_string(),
    }
}

fn schema_properties(schema: &Value) -> Vec<(String, Value)> {
    let Some(object) = schema.as_object() else {
        return Vec::new();
    };
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        return properties
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
    }

    let mut merged = Vec::new();
    for keyword in ["anyOf", "oneOf", "allOf"] {
        let Some(branches) = object.get(keyword).and_then(Value::as_array) else {
            continue;
        };
        for branch in branches {
            for (key, value) in schema_properties(branch) {
                if !merged.iter().any(|(existing, _)| existing == &key) {
                    merged.push((key, value));
                }
            }
        }
    }
    merged
}

fn has_complete_properties(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return schema == &Value::Bool(false);
    };
    if [
        "$ref",
        "$dynamicRef",
        "patternProperties",
        "dependentSchemas",
        "if",
        "then",
        "else",
    ]
    .iter()
    .any(|keyword| object.contains_key(*keyword))
        || ["additionalProperties", "unevaluatedProperties"]
            .iter()
            .any(|keyword| {
                object.get(*keyword).unwrap_or(&Value::Bool(false)) != &Value::Bool(false)
            })
    {
        return false;
    }

    let branches = ["allOf", "anyOf", "oneOf"]
        .iter()
        .flat_map(|keyword| {
            object
                .get(*keyword)
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
        })
        .collect::<Vec<_>>();
    if let Some(properties) = object.get("properties").and_then(Value::as_object) {
        return branches.is_empty()
            && (!properties.is_empty()
                || object.get("additionalProperties") == Some(&Value::Bool(false)));
    }
    if !branches.is_empty() {
        return branches
            .iter()
            .all(|branch| has_complete_properties(branch));
    }
    object.get("additionalProperties") == Some(&Value::Bool(false))
}

#[derive(Debug, Default)]
struct GlmTrieNode {
    children: BTreeMap<char, usize>,
    failure: usize,
    is_end: bool,
}

fn any_string_exclude(rule_name: &str, negative_strings: &[&str]) -> Vec<String> {
    let mut patterns = negative_strings
        .iter()
        .filter(|pattern| !pattern.is_empty())
        .copied()
        .collect::<Vec<_>>();
    patterns.sort_unstable();
    patterns.dedup();
    if patterns.is_empty() {
        return vec![format!("{rule_name} ::= [^]*")];
    }

    let hash_prefix = glm_hash_name(&patterns.join("\0"));
    let mut nodes = vec![GlmTrieNode::default()];
    for pattern in &patterns {
        let mut node = 0;
        for character in pattern.chars() {
            let next = nodes[node].children.get(&character).copied();
            node = match next {
                Some(next) => next,
                None => {
                    nodes.push(GlmTrieNode::default());
                    let next = nodes.len() - 1;
                    nodes[node].children.insert(character, next);
                    next
                }
            };
        }
        nodes[node].is_end = true;
    }

    let mut queue = std::collections::VecDeque::new();
    let root_children = nodes[0].children.values().copied().collect::<Vec<_>>();
    for child in root_children {
        nodes[child].failure = 0;
        queue.push_back(child);
    }
    while let Some(current) = queue.pop_front() {
        let children = nodes[current]
            .children
            .iter()
            .map(|(character, child)| (*character, *child))
            .collect::<Vec<_>>();
        for (character, child) in children {
            queue.push_back(child);
            let mut failure = nodes[current].failure;
            while failure != 0 && !nodes[failure].children.contains_key(&character) {
                failure = nodes[failure].failure;
            }
            nodes[child].failure = nodes[failure]
                .children
                .get(&character)
                .copied()
                .filter(|target| *target != child)
                .unwrap_or(0);
            let failure_is_end = nodes[nodes[child].failure].is_end;
            nodes[child].is_end = nodes[child].is_end || failure_is_end;
        }
    }

    let mut pattern_chars = BTreeSet::new();
    for pattern in &patterns {
        pattern_chars.extend(pattern.chars());
    }
    let state_name = |state: usize| format!("s_{hash_prefix}_{state}");
    let mut rules = vec![format!("{rule_name} ::= {}", state_name(0))];

    for state in 0..nodes.len() {
        if nodes[state].is_end {
            continue;
        }
        let mut excluded_chars = Vec::new();
        let mut transitions = BTreeMap::new();
        for character in &pattern_chars {
            let mut current = state;
            loop {
                if let Some(target) = nodes[current].children.get(character) {
                    if nodes[*target].is_end {
                        excluded_chars.push(*character);
                    } else {
                        transitions
                            .entry(*target)
                            .or_insert_with(Vec::new)
                            .push(*character);
                    }
                    break;
                }
                if current == 0 {
                    if nodes[0].is_end {
                        excluded_chars.push(*character);
                    } else {
                        transitions
                            .entry(0)
                            .or_insert_with(Vec::new)
                            .push(*character);
                    }
                    break;
                }
                current = nodes[current].failure;
            }
        }

        let mut explicit_chars = BTreeSet::from_iter(excluded_chars.iter().copied());
        for chars in transitions.values() {
            explicit_chars.extend(chars.iter().copied());
        }
        let mut alternatives = Vec::new();
        if explicit_chars.is_empty() {
            alternatives.push(format!("[^] {}", state_name(0)));
        } else {
            let class = explicit_chars
                .iter()
                .map(|c| escape_char_class(*c))
                .collect::<String>();
            alternatives.push(format!("[^{class}] {}", state_name(0)));
        }
        for (target, chars) in &transitions {
            for character in chars {
                alternatives.push(format!(
                    "\"{}\" {}",
                    escape_string_literal(*character),
                    state_name(*target)
                ));
            }
        }
        alternatives.push("\"\"".to_string());
        rules.push(format!(
            "{} ::= {}",
            state_name(state),
            alternatives.join(" | ")
        ));
    }
    rules
}

fn escape_char_class(character: char) -> String {
    match character {
        '\\' | ']' | '^' | '-' => format!("\\{character}"),
        '\n' => "\\n".to_string(),
        '\t' => "\\t".to_string(),
        '\r' => "\\r".to_string(),
        character if (character as u32) < 32 || (character as u32) > 126 => {
            format!("\\x{:02X}", character as u32)
        }
        character => character.to_string(),
    }
}

fn escape_string_literal(character: char) -> String {
    match character {
        '"' => "\\\"".to_string(),
        '\\' => "\\\\".to_string(),
        '\n' => "\\n".to_string(),
        '\t' => "\\t".to_string(),
        '\r' => "\\r".to_string(),
        character if (character as u32) < 32 || (character as u32) > 126 => {
            format!("\\x{:02X}", character as u32)
        }
        character => character.to_string(),
    }
}

fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

fn has_response_format_constraint(request: &NvCreateChatCompletionRequest) -> bool {
    request
        .inner
        .response_format
        .as_ref()
        .is_some_and(|format| !matches!(format, ResponseFormat::Text))
}

pub(crate) fn convert_tool_choice(tool_choice: &ChatCompletionToolChoiceOption) -> ToolChoice {
    match tool_choice {
        ChatCompletionToolChoiceOption::None => ToolChoice::None,
        ChatCompletionToolChoiceOption::Auto => ToolChoice::Auto,
        ChatCompletionToolChoiceOption::Required => ToolChoice::Required,
        ChatCompletionToolChoiceOption::Named(named) => {
            ToolChoice::Named(named.function.name.clone())
        }
    }
}

pub(crate) fn convert_tools(tools: &[ChatCompletionTool]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.function.name.clone(),
            parameters: tool.function.parameters.clone(),
            strict: tool.function.strict,
        })
        .collect()
}

/// Normalize and validate all tools visible to the model without rewriting the request.
///
/// Top-level OpenAI tools remain first. Kimi-style tools declared on system
/// messages follow in message order and may use either the wrapped OpenAI form
/// or Kimi's bare function-schema form. This view is used only by validation,
/// parser, and guided-decoding policy; dynamic declarations stay at their
/// original message positions for prompt rendering and KV-cache correctness.
pub(crate) fn effective_tools(
    request: &CreateChatCompletionRequest,
) -> Result<Cow<'_, [ChatCompletionTool]>, DynamoError> {
    validate::validated_effective_tools(request)
        .map_err(|error| invalid_argument(error.to_string(), None))
}

pub(crate) fn effective_tool_definitions(
    request: &CreateChatCompletionRequest,
) -> Result<Vec<ToolDefinition>, DynamoError> {
    effective_tools(request).map(|tools| convert_tools(tools.as_ref()))
}

/// The guided-tool constraint a request implies, given what the structural-tag stage
/// already decided.
///
/// Direct postprocessor callers and remote frontends use this compatibility helper
/// when they did not run request preprocessing. The worker streaming path carries the
/// enum returned by `apply_tool_choice_guided_decoding`; topology-B aggregation may
/// reconstruct a structural-tag request as guided JSON, so its complete-output parser
/// also checks the generated wire shape.
pub(crate) fn guided_tool_constraint(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
) -> Result<GuidedToolConstraint, DynamoError> {
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    let tools = effective_tools(&request.inner)?;
    guided_tool_constraint_with_effective_tools(
        request,
        tool_call_parser,
        reasoning_parser,
        false,
        tools.as_ref(),
    )
}

/// Derive the guided-tool constraint from an effective tool set the caller has
/// already normalized and validated through [`effective_tools`].
pub(crate) fn guided_tool_constraint_with_effective_tools(
    request: &NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
    uses_structural_tag: bool,
    tools: &[ChatCompletionTool],
) -> Result<GuidedToolConstraint, DynamoError> {
    if uses_structural_tag {
        return Ok(GuidedToolConstraint::StructuralTag);
    }
    let tool_choice = request
        .inner
        .tool_choice
        .as_ref()
        .unwrap_or(&ChatCompletionToolChoiceOption::Auto);
    validate_openai_tool_choice(Some(tool_choice), Some(tools))
        .map_err(|error| invalid_argument(error.to_string(), None))?;
    let is_forced_tool_choice = matches!(
        tool_choice,
        ChatCompletionToolChoiceOption::Required | ChatCompletionToolChoiceOption::Named(_)
    );
    // Only a forced choice installs a tool-level JSON schema. Auto/None leave any
    // JSON constraint to `response_format`, which governs assistant CONTENT.
    if !is_forced_tool_choice {
        return Ok(GuidedToolConstraint::None);
    }
    // Forced + explicit guided decoding is rejected at request time, so reaching here
    // with both means the request never ran.
    if has_explicit_guided_decoding(request) {
        return Ok(GuidedToolConstraint::None);
    }
    let parser_tool_choice = convert_tool_choice(tool_choice);
    // Kimi forced requests use marker-delimited native formats that generic JSON
    // guidance cannot represent. Keep this compatibility path aligned with
    // request preprocessing for either configured parser slot.
    if requires_native_tool_call_format(tool_call_parser, &parser_tool_choice)
        || requires_native_tool_call_format(reasoning_parser, &parser_tool_choice)
    {
        return Ok(GuidedToolConstraint::None);
    }
    // Validate the forced choice against the actual tools the same way
    // `apply_tool_choice_guided_decoding` does, instead of blindly installing a
    // constraint for a `tool_choice` that names a tool absent from `tools` (or an
    // empty `tools` list under `tool_choice: "required"`).
    match get_tool_choice_guidance_from_tools(
        Some(tool_choice),
        Some(tools),
        request.inner.parallel_tool_calls,
    ) {
        Ok(Some(_)) => Ok(installed_json_constraint(tool_choice)),
        Ok(None) => Ok(GuidedToolConstraint::None),
        Err(e) => Err(invalid_argument(e.to_string(), None)),
    }
}

/// Map a forced `tool_choice` onto the guided-JSON parser constraint for its grammar.
///
/// Only reachable once `get_tool_choice_guidance_from_tools` produced either JSON or
/// regex guidance. It returns `None` for `Auto`/`None`, so those arms are unreachable
/// in practice and report [`GuidedToolConstraint::None`] rather than guessing.
fn installed_json_constraint(tool_choice: &ChatCompletionToolChoiceOption) -> GuidedToolConstraint {
    match tool_choice {
        ChatCompletionToolChoiceOption::Named(named) => GuidedToolConstraint::GuidedJsonNamed {
            tool_name: named.function.name.clone(),
        },
        ChatCompletionToolChoiceOption::Required => GuidedToolConstraint::GuidedJsonRequired,
        ChatCompletionToolChoiceOption::Auto | ChatCompletionToolChoiceOption::None => {
            GuidedToolConstraint::None
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use super::*;
    use crate::local_model::runtime_config::StructuralTagMode;
    use crate::model_card::ModelDeploymentCard;
    use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};
    use dynamo_protocols::types::{ChatCompletionNamedToolChoice, FunctionName};
    use serde_json::{Value, json};

    fn request(extra: Value) -> NvCreateChatCompletionRequest {
        let mut value = json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "test"}]
        });
        value
            .as_object_mut()
            .expect("base request is an object")
            .extend(extra.as_object().expect("extra is an object").clone());
        serde_json::from_value(value).expect("request must deserialize")
    }

    fn tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }])
    }

    fn preprocessor(parser: &str, mode: StructuralTagMode) -> Arc<OpenAIPreprocessor> {
        preprocessor_with_capability(parser, mode, true)
    }

    fn preprocessor_with_capability(
        parser: &str,
        mode: StructuralTagMode,
        full_assistant_supported: bool,
    ) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = (mode == StructuralTagMode::On)
            .then(crate::local_model::runtime_config::StructuralTagConfig::default);
        mdc.runtime_config.tool_call_parser = Some(parser.to_string());
        if full_assistant_supported {
            mdc.runtime_config.runtime_data.insert(
                SGLANG_EBNF_FULL_ASSISTANT_CAPABILITY.to_string(),
                json!(true),
            );
        }
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn preprocessed_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test-model".to_string())
            .token_ids(Vec::new())
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap()
    }

    #[test]
    fn tool_call_parsing_requires_tools() {
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({})
        )));
        assert!(!OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({"tool_choice": "required"})
        )));
    }

    #[test]
    fn tool_call_parsing_honors_each_tool_choice() {
        for (tool_choice, expected) in [
            (json!(null), true),
            (json!("none"), false),
            (json!("auto"), true),
            (json!("required"), true),
            (
                json!({"type": "function", "function": {"name": "get_weather"}}),
                true,
            ),
        ] {
            let mut extra = json!({"tools": tools()});
            if !tool_choice.is_null() {
                extra["tool_choice"] = tool_choice;
            }
            assert_eq!(
                OpenAIPreprocessor::tool_call_parsing_enabled(&request(extra)),
                expected,
            );
        }
    }

    #[test]
    fn glm47_auto_installs_full_assistant_ebnf_guidance() {
        let preprocessor = preprocessor("glm47", StructuralTagMode::Off);
        let chat_request = request(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "auto"
        }));
        let mut common_request = preprocessed_request();

        let constraint = preprocessor
            .apply_tool_choice_guided_decoding(&chat_request, &mut common_request, false)
            .unwrap();
        let guided = common_request
            .sampling_options
            .guided_decoding
            .expect("GLM-4.7 installs full-assistant EBNF");

        assert_eq!(constraint, GuidedToolConstraint::None);
        assert!(guided.ebnf_full_assistant);
        assert!(
            guided
                .grammar
                .as_deref()
                .is_some_and(|grammar| grammar.contains("\"get_current_time\"")
                    && grammar.contains("\"<tool_call>\""))
        );
        assert!(guided.structural_tag.is_none());
        guided.validate().unwrap();

        let strict_request = request(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "strict": true,
                    "parameters": {
                        "type": "object",
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": "auto"
        }));
        let mut strict_common_request = preprocessed_request();
        preprocessor
            .apply_tool_choice_guided_decoding(&strict_request, &mut strict_common_request, false)
            .unwrap();
        assert!(
            strict_common_request
                .sampling_options
                .guided_decoding
                .is_none()
        );
    }

    #[test]
    fn glm47_full_assistant_grammar_uses_only_current_request_tools() {
        let tools = [
            ToolDefinition {
                name: "get_current_time".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}}
                })),
                strict: Some(false),
            },
            ToolDefinition {
                name: "lookup_country".to_string(),
                parameters: Some(json!({"type": "object"})),
                strict: Some(false),
            },
        ];

        let auto = full_assistant_tools(&ChatCompletionToolChoiceOption::Auto, &tools)
            .into_iter()
            .collect::<Vec<_>>();
        let auto_grammar = generate_glm47_full_assistant_grammar(&auto, true, false, false);
        let parallel_grammar = generate_glm47_full_assistant_grammar(&auto, true, true, false);
        assert!(auto_grammar.contains("\"get_current_time\""));
        assert!(auto_grammar.contains("\"lookup_country\""));
        assert!(auto_grammar.contains("thinking_block_content \"</think>\""));
        assert!(parallel_grammar.contains("tool_call_blocks ::= (  tool_call_unit )*"));
        assert!(auto_grammar.contains("tool_call_blocks ::= (  tool_call_unit )?"));

        let named = full_assistant_tools(&named("get_current_time"), &tools)
            .into_iter()
            .collect::<Vec<_>>();
        let named_grammar = generate_glm47_full_assistant_grammar(&named, false, false, true);
        assert!(named_grammar.contains("\"get_current_time\""));
        assert!(!named_grammar.contains("\"lookup_country\""));
        assert!(named_grammar.contains("thinking_block ::= \"\""));
        assert!(named_grammar.contains("tool_call_blocks ::= (  tool_call_unit )"));
    }

    #[test]
    fn glm47_full_assistant_ebnf_requires_effective_tools() {
        let preprocessor = preprocessor("glm47", StructuralTagMode::Off);

        for extra in [
            json!({"tool_choice": "auto"}),
            json!({"tools": tools(), "tool_choice": "none"}),
        ] {
            let chat_request = request(extra.clone());
            let mut common_request = preprocessed_request();

            preprocessor
                .apply_tool_choice_guided_decoding(&chat_request, &mut common_request, false)
                .unwrap();

            assert!(
                common_request.sampling_options.guided_decoding.is_none(),
                "full-assistant EBNF must not be installed without effective tools: {extra:?}"
            );
        }
    }

    #[test]
    fn non_string_enum_values_use_json_literals() {
        let schema = json!({
            "enum": [
                "text",
                7,
                ["text"],
                {"kind": "text"}
            ]
        });

        let rule = value_rule(&schema);

        assert!(rule.contains("\"text\""));
        assert!(rule.contains("\"7\""));
        assert!(rule.contains("\"[\\\"text\\\"]\""));
        assert!(rule.contains("\"{\\\"kind\\\":\\\"text\\\"}\""));
    }

    #[test]
    fn full_assistant_ebnf_requires_a_known_parser_dialect() {
        let preprocessor = preprocessor_with_capability("hermes", StructuralTagMode::Off, true);
        let chat_request = request(json!({"tools": tools(), "tool_choice": "auto"}));
        let mut common_request = preprocessed_request();

        preprocessor
            .apply_tool_choice_guided_decoding(&chat_request, &mut common_request, false)
            .unwrap();

        assert!(
            common_request.sampling_options.guided_decoding.is_none(),
            "a parser without a registered full-assistant grammar must not receive the GLM grammar"
        );
    }

    #[test]
    fn full_assistant_ebnf_requires_worker_capability() {
        let preprocessor = preprocessor_with_capability("glm47", StructuralTagMode::Off, false);
        let chat_request = request(json!({"tools": tools(), "tool_choice": "auto"}));
        let mut common_request = preprocessed_request();

        preprocessor
            .apply_tool_choice_guided_decoding(&chat_request, &mut common_request, false)
            .unwrap();

        assert!(
            common_request.sampling_options.guided_decoding.is_none(),
            "an unsupported worker must not receive a full-assistant grammar"
        );
    }

    #[test]
    fn effective_tools_borrows_top_level_tools_without_dynamic_declarations() {
        let request = request(json!({"tools": tools()}));
        let top_level = request.inner.tools.as_deref().unwrap();

        let effective = effective_tools(&request.inner).expect("top-level tools must validate");

        assert!(matches!(&effective, Cow::Borrowed(_)));
        assert!(std::ptr::eq(effective.as_ptr(), top_level.as_ptr()));
    }

    #[test]
    fn dynamic_system_tools_enable_parsing_and_honor_tool_choice_none() {
        for (tool_choice, expected) in [(None, true), (Some(json!("none")), false)] {
            let mut extra = json!({
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                        "tools": [{"name": "lookup", "parameters": {"type": "object"}}]
                    },
                    {"role": "user", "content": "look it up"}
                ]
            });
            if let Some(tool_choice) = tool_choice {
                extra["tool_choice"] = tool_choice;
            }
            assert_eq!(
                OpenAIPreprocessor::tool_call_parsing_enabled(&request(extra)),
                expected,
            );
        }
    }

    #[test]
    fn effective_tools_preserve_order_and_normalize_wrapped_and_bare_shapes() {
        let request = request(json!({
            "tools": [{
                "type": "function",
                "function": {"name": "static_tool", "parameters": {"type": "object"}}
            }],
            "messages": [
                {"role": "user", "content": "start"},
                {
                    "role": "system",
                    "content": "",
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "wrapped_dynamic",
                            "parameters": {
                                "type": "object",
                                "properties": {"x": {"type": "integer"}},
                                "required": ["x"],
                                "additionalProperties": false
                            },
                            "strict": true
                        }
                    }]
                },
                {
                    "role": "system",
                    "content": "",
                    "tools": [{
                        "name": "bare_dynamic",
                        "description": "bare form",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": false,
                        "vendor_hint": "kept only in the original message"
                    }]
                },
                {"role": "user", "content": "continue"}
            ]
        }));

        let normalized = effective_tools(&request.inner).expect("dynamic tools must normalize");
        assert!(matches!(&normalized, Cow::Owned(_)));
        assert_eq!(
            normalized
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<Vec<_>>(),
            ["static_tool", "wrapped_dynamic", "bare_dynamic"]
        );
        assert_eq!(normalized[1].function.strict, Some(true));
        assert_eq!(normalized[2].function.strict, Some(false));
        assert_eq!(
            normalized[2].function.description.as_deref(),
            Some("bare form")
        );
        let original = serde_json::to_value(&request.inner.messages).unwrap();
        assert_eq!(
            original[2]["tools"][0]["vendor_hint"], "kept only in the original message",
            "normalization must not rewrite dynamic message declarations"
        );
    }

    #[test]
    fn effective_tools_reject_invalid_dynamic_metadata() {
        for (field, value) in [
            ("strict", json!("yes")),
            ("parameters", json!([])),
            ("description", json!(7)),
        ] {
            let mut tool = json!({"name": "lookup"});
            tool[field] = value;
            let request = request(json!({
                "messages": [
                    {"role": "system", "content": "", "tools": [tool]},
                    {"role": "user", "content": "go"}
                ]
            }));
            let error = effective_tools(&request.inner).expect_err("invalid field must fail");
            assert!(error.to_string().contains(field), "{field}: {error}");
        }

        let request = request(json!({
            "messages": [
                {"role": "system", "content": "", "tools": [{"name": "bad name"}]},
                {"role": "user", "content": "go"}
            ]
        }));
        let error = effective_tools(&request.inner).expect_err("invalid name must fail");
        assert_eq!(error.error_type(), ErrorType::InvalidArgument);
        assert!(error.to_string().contains("has an invalid name"));
    }

    #[test]
    fn assistant_constraints_do_not_revoke_auto_tool_permission() {
        let constraints = [
            json!({"response_format": {"type": "json_object"}}),
            json!({
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "event",
                        "schema": {"type": "object"}
                    }
                }
            }),
            json!({"guided_json": {"type": "object"}}),
            json!({"guided_regex": "[a-z]+"}),
            json!({"guided_choice": ["a", "b"]}),
            json!({"guided_grammar": "root ::= 'a'"}),
        ];

        for constraint in constraints {
            let mut auto = json!({"tools": tools(), "tool_choice": "auto"});
            auto.as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                auto
            )));

            let mut required = json!({"tools": tools(), "tool_choice": "required"});
            required
                .as_object_mut()
                .unwrap()
                .extend(constraint.as_object().unwrap().clone());
            assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
                required
            )));
        }
    }

    #[test]
    fn text_response_format_does_not_disable_auto_tool_parsing() {
        assert!(OpenAIPreprocessor::tool_call_parsing_enabled(&request(
            json!({
                "tools": tools(),
                "tool_choice": "auto",
                "response_format": {"type": "text"}
            })
        )));
    }

    fn named(name: &str) -> ChatCompletionToolChoiceOption {
        ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
            r#type: dynamo_protocols::types::ChatCompletionToolType::Function,
            function: FunctionName {
                name: name.to_string(),
            },
        })
    }

    #[test]
    fn only_structural_tag_reports_a_structural_tag() {
        assert!(GuidedToolConstraint::StructuralTag.uses_structural_tag());
        assert!(!GuidedToolConstraint::None.uses_structural_tag());
        assert!(!GuidedToolConstraint::GuidedJsonRequired.uses_structural_tag());
        assert!(
            !GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
            .uses_structural_tag()
        );
    }

    #[test]
    fn required_reports_the_array_shape() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Required),
            GuidedToolConstraint::GuidedJsonRequired
        );
    }

    #[test]
    fn named_carries_the_tool_name_from_the_request() {
        assert_eq!(
            installed_json_constraint(&named("get_weather")),
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_weather".to_string(),
            }
        );
    }

    #[test]
    fn named_closed_zero_arg_tool_keeps_the_named_parser_constraint() {
        let request = request(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_server_time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_server_time"}
            }
        }));

        assert_eq!(
            guided_tool_constraint(&request, None, None, false).expect("constraint is valid"),
            GuidedToolConstraint::GuidedJsonNamed {
                tool_name: "get_server_time".to_string(),
            }
        );
    }

    #[test]
    fn unforced_choices_install_no_tool_constraint() {
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::Auto),
            GuidedToolConstraint::None
        );
        assert_eq!(
            installed_json_constraint(&ChatCompletionToolChoiceOption::None),
            GuidedToolConstraint::None
        );
    }

    #[test]
    fn native_format_forced_choices_install_no_compatibility_json_constraint() {
        let requests = [
            request(json!({
                "tools": tools(),
                "tool_choice": "required"
            })),
            request(json!({
                "tools": tools(),
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"}
                }
            })),
        ];

        for request in &requests {
            for (tool_call_parser, reasoning_parser) in [
                (Some("kimi_k2"), None),
                (None, Some("kimi_k2")),
                (Some("kimi_k3"), None),
                (None, Some("kimi-k3")),
            ] {
                assert_eq!(
                    guided_tool_constraint(request, tool_call_parser, reasoning_parser, false,)
                        .expect("native-format choice is valid"),
                    GuidedToolConstraint::None,
                );
            }
        }
    }

    #[test]
    fn kimi_k3_required_without_tools_is_rejected_before_the_xtml_return() {
        for extra in [
            json!({"tool_choice": "required"}),
            json!({"tool_choice": "required", "tools": []}),
        ] {
            let request = request(extra);
            let result = guided_tool_constraint(&request, Some("kimi_k3"), None, false);
            assert!(
                result.is_err(),
                "Kimi K3 required must reject both missing and empty tools"
            );
        }
    }

    #[test]
    fn structural_tag_off_preserves_kimi_k2_required_native_tag() {
        let request = request(json!({
            "tools": tools(),
            "tool_choice": "required"
        }));
        let preprocessor = preprocessor("kimi_k2", StructuralTagMode::Off);
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();

        assert_eq!(applied, GuidedToolConstraint::StructuralTag);
        assert!(
            common_request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.structural_tag.as_ref())
                .is_some()
        );
    }

    #[test]
    fn structural_tag_off_preserves_kimi_k3_named_native_tag() {
        let request = request(json!({
            "tools": tools(),
            "tool_choice": {
                "type": "function",
                "function": {"name": "get_weather"}
            }
        }));
        let preprocessor = preprocessor("kimi_k3", StructuralTagMode::Off);
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();

        assert_eq!(applied, GuidedToolConstraint::StructuralTag);
        assert!(
            common_request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.structural_tag.as_ref())
                .is_some()
        );
    }

    #[test]
    fn structural_tag_off_preserves_generic_forced_tool_fallback() {
        let request = request(json!({
            "tools": tools(),
            "tool_choice": "required"
        }));
        let preprocessor = preprocessor("hermes", StructuralTagMode::Off);
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .unwrap();

        assert_eq!(applied, GuidedToolConstraint::GuidedJsonRequired);
        assert!(
            common_request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.json.as_ref())
                .is_some()
        );
    }

    #[test]
    fn kimi_k3_builder_error_uses_native_fallback() {
        let request = request(json!({
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "$id": "urn:example:scoped",
                                "$defs": {
                                    "allowed": {"type": "string", "enum": ["safe"]}
                                },
                                "allOf": [{"$ref": "#/$defs/allowed"}]
                            }
                        },
                        "required": ["value"]
                    }
                }
            }],
            "tool_choice": "auto"
        }));
        let preprocessor = preprocessor("kimi_k3", StructuralTagMode::On);
        let tools = convert_tools(request.inner.tools.as_deref().unwrap());
        let ctx = dynamo_parsers::tool_calling::ToolCallFormatBuildContext {
            tool_choice: &ToolChoice::Auto,
            tools: &tools,
            parallel_tool_calls: request.inner.parallel_tool_calls,
            schema_mode: preprocessor.runtime_config.structural_tag_schema,
            starts_in_reasoning: false,
        };
        // A scoped string reference cannot be safely rendered as raw XTML.
        // Verify an actual builder error, rather than a policy skip or Ok(None).
        let error = dynamo_parsers::tool_calling::StructuralTagBuilder::KimiK3
            .build_tool_call_format(&ctx)
            .expect_err("scoped string reference must reach the builder-error fallback");
        let message = format!("{error:#}");
        assert!(
            message.contains("value") && message.contains("string"),
            "{message}"
        );

        let mut common_request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
            .expect("builder errors must allow the request to continue without a structural tag");

        assert_eq!(applied, GuidedToolConstraint::None);
        assert!(common_request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn kimi_k3_forced_choices_accept_dynamic_system_tools() {
        for tool_choice in [
            json!("required"),
            json!({"type": "function", "function": {"name": "lookup"}}),
        ] {
            let request = request(json!({
                "messages": [
                    {
                        "role": "system",
                        "content": "",
                        "tools": [{"name": "lookup", "parameters": {"type": "object"}}]
                    },
                    {"role": "user", "content": "go"}
                ],
                "tool_choice": tool_choice
            }));
            assert_eq!(
                guided_tool_constraint(&request, Some("kimi_k3"), None, false)
                    .expect("dynamic forced choice must validate"),
                GuidedToolConstraint::None,
                "K3 forced choices use native XTML rather than guided JSON"
            );
        }
    }
}
