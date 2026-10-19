// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tool-choice guided decoding policy for OpenAI chat requests.

use std::borrow::Cow;

use crate::preprocessor::structural_tag::{is_beam_parser, requires_native_tool_call_format};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest, PromptReasoningPrefill};
use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::common_ext::CommonExtProvider;
use crate::protocols::openai::tools::{
    ToolChoiceGuidance, get_tool_choice_guidance_from_tools, validate_openai_tool_choice,
};
use crate::protocols::openai::{GuidedToolConstraint, validate};

use dynamo_parsers::tool_calling::{ToolChoice, ToolDefinition};
use dynamo_protocols::types::{
    ChatCompletionTool, ChatCompletionToolChoiceOption, CreateChatCompletionRequest, ResponseFormat,
};
use dynamo_renderer::OAIChatLikeRequest;
use dynamo_runtime::error::{DynamoError, ErrorType};

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
        prompt_prefill: impl Into<PromptReasoningPrefill>,
    ) -> Result<GuidedToolConstraint, DynamoError> {
        let prompt_prefill = prompt_prefill.into();
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

        // Beam: the parsers crate's turn grammar holds the whole turn to the response
        // template over the Harmony wire — a non-empty analysis channel, then calls
        // whose arguments keep their schema types (unheld, Beam writes integers as
        // `N.0`) or the answer — so forced choices are enforced and the response keeps
        // its reasoning. Dynamo only decides when it applies (`beam_turn_grammar_constraint`)
        // and from which channel the rendered prompt leaves open.
        let beam_constraint = if self.is_beam() {
            beam_turn_grammar_constraint(request)
        } else {
            GuidedToolConstraint::None
        };

        // Explicit guided decoding constrains assistant content and cannot be
        // composed with structural-tag tool calls.
        if !is_forced_tool_choice && has_explicit_guided_decoding {
            return Ok(GuidedToolConstraint::None);
        }

        // Beam's `response_format` is held inside the final channel behind the
        // analysis (beside the calls under automatic choice), the way the model
        // builder's serving layer derives it, instead of a bare JSON document from
        // the first token that loses the reasoning and the channel structure.
        if !is_forced_tool_choice && beam_constraint.frames_structured_answer() {
            if let Some(schema) = request.get_guided_json()
                && let Some(structural_tag) = beam_turn_structural_tag(
                    tools.as_ref(),
                    &parser_tool_choice,
                    request.inner.parallel_tool_calls,
                    prompt_prefill.beam_channel,
                    Some(&schema),
                )
            {
                install_structural_tag(common_request, structural_tag);
                return Ok(beam_constraint);
            }
            return Ok(GuidedToolConstraint::None);
        }

        if is_forced_tool_choice
            && has_response_format_constraint
            && let Some(gd) = common_request.sampling_options.guided_decoding.as_mut()
        {
            // OpenAI `response_format` applies to assistant content, not tool calls.
            gd.json = None;
        }

        if self.apply_tool_choice_structural_tag(
            &parser_tool_choice,
            &convert_tools(tools.as_ref()),
            request.inner.parallel_tool_calls,
            prompt_prefill.unified,
            request.inner.response_format.as_ref(),
            common_request,
        )? {
            return Ok(GuidedToolConstraint::StructuralTag);
        }

        if matches!(
            beam_constraint,
            GuidedToolConstraint::BeamTurnGrammar { .. }
        ) {
            if let Some(structural_tag) = beam_turn_structural_tag(
                tools.as_ref(),
                &parser_tool_choice,
                request.inner.parallel_tool_calls,
                prompt_prefill.beam_channel,
                None,
            ) {
                install_structural_tag(common_request, structural_tag);
                return Ok(beam_constraint);
            }
            if is_forced_tool_choice {
                tracing::debug!(
                    tool_choice = ?parser_tool_choice,
                    "Beam grammar could not hold this forced choice; using prompt/parser compatibility fallback"
                );
                return Ok(GuidedToolConstraint::None);
            }
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
}

/// Whether top-level `guided_*` fields install a grammar; mirrors
/// `GuidedDecodingOptions::from_optional`.
pub(super) fn has_explicit_guided_decoding(request: &NvCreateChatCompletionRequest) -> bool {
    request.common.guided_json.is_some()
        || request.common.guided_regex.is_some()
        || request
            .common
            .guided_choice
            .as_ref()
            .is_some_and(|v| !v.is_empty())
        || request.common.guided_grammar.is_some()
}

impl OpenAIPreprocessor {
    fn is_beam(&self) -> bool {
        is_beam_parser(self.tool_call_parser.as_deref())
            || is_beam_parser(self.runtime_config.reasoning_parser.as_deref())
    }

    /// Whether the Beam turn grammar holds this request's `response_format` inside the
    /// final channel behind the analysis (automatic or `none` tool choice, no explicit
    /// guided decoding, grammar enabled) instead of a bare JSON document.
    pub(crate) fn beam_frames_structured_response<R: OAIChatLikeRequest + CommonExtProvider>(
        &self,
        request: &R,
    ) -> bool {
        self.is_beam() && beam_turn_grammar_constraint(request).frames_structured_answer()
    }

    /// Whether Beam's output is a bare `response_format` document the engine's grammar
    /// owns in full, with no channel structure and so no end-of-turn marker for the
    /// parser to see. A forced choice ignores `response_format` and generates native
    /// calls; a framed `response_format` keeps the channel structure.
    pub(crate) fn beam_output_is_bare_document<R: OAIChatLikeRequest + CommonExtProvider>(
        &self,
        request: &R,
    ) -> bool {
        Self::has_structured_response_format(request)
            && !Self::has_guided_tool_choice(request)
            && !self.beam_frames_structured_response(request)
    }
}

/// Beam route only: a tool declared `strict: true` whose schema lies outside
/// OpenAI's strict subset (every property required, `additionalProperties: false`, no
/// composition keywords — enforced upstream since #15153 with an HTTP 400) is served as
/// non-strict with a warning instead of rejected. The Beam turn grammar holds every
/// typeable tool regardless of `strict`, so nothing is lost, and clients that stamp
/// `strict: true` on every tool (inspect_ai's OpenAI-compatible provider does) keep
/// working. Other routes keep upstream's rejection. Returns the number of tools relaxed.
pub(crate) fn relax_noncompliant_strict_tools_for_beam(
    request: &mut NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> usize {
    if !(is_beam_parser(tool_call_parser) || is_beam_parser(reasoning_parser)) {
        return 0;
    }
    let Some(tools) = request.inner.tools.as_mut() else {
        return 0;
    };
    let mut relaxed: Vec<String> = Vec::new();
    let mut first_violation = None;
    for tool in tools.iter_mut() {
        if tool.function.strict != Some(true) {
            continue;
        }
        if let Err(error) =
            crate::protocols::openai::strict_schema::validate_strict_function(&tool.function)
        {
            // The name is unvalidated request data at this point; bound what is logged.
            relaxed.push(tool.function.name.chars().take(64).collect());
            first_violation.get_or_insert_with(|| error.to_string());
            tool.function.strict = None;
        }
    }
    if !relaxed.is_empty() {
        // One line per request: a client that marks every tool strict would otherwise
        // log once per tool per request.
        tracing::warn!(
            relaxed = relaxed.len(),
            tools = %relaxed.iter().take(5).cloned().collect::<Vec<_>>().join(", "),
            first_violation = %first_violation.unwrap_or_default(),
            "Beam route: tool(s) declared `strict` with a schema outside OpenAI's strict subset; serving them as non-strict (the turn grammar holds every typeable tool regardless)"
        );
    }
    relaxed.len()
}

/// Beam route only: a `builtin_tools` entry given as a declaration object (the legacy
/// form; the 1.0 template takes names) that the request does not also declare in
/// `tools` is added to `tools`, so the prompt, the turn grammar and the parser all see
/// one tool set. Entries that are names, or objects already declared, are left to the
/// renderer, which validates them. Returns how many tools were added.
pub(crate) fn declare_legacy_builtin_tools_for_beam(
    request: &mut NvCreateChatCompletionRequest,
    tool_call_parser: Option<&str>,
    reasoning_parser: Option<&str>,
) -> usize {
    if !(is_beam_parser(tool_call_parser) || is_beam_parser(reasoning_parser)) {
        return 0;
    }
    let Some(entries) = request
        .chat_template_args
        .as_ref()
        .and_then(|args| args.get("builtin_tools"))
        .and_then(serde_json::Value::as_array)
    else {
        return 0;
    };
    let mut added: Vec<ChatCompletionTool> = Vec::new();
    for entry in entries.iter().filter(|entry| entry.is_object()) {
        let declaration = if entry.get("function").is_some() {
            entry.clone()
        } else {
            serde_json::json!({"type": "function", "function": entry})
        };
        // A malformed declaration is left for the renderer's error message.
        let Ok(tool) = serde_json::from_value::<ChatCompletionTool>(declaration) else {
            continue;
        };
        let declared = request
            .inner
            .tools
            .iter()
            .flatten()
            .chain(added.iter())
            .any(|existing| existing.function.name == tool.function.name);
        if !declared {
            added.push(tool);
        }
    }
    let count = added.len();
    if count > 0 {
        request
            .inner
            .tools
            .get_or_insert_with(Vec::new)
            .extend(added);
    }
    count
}

fn install_structural_tag(
    common_request: &mut PreprocessedRequest,
    structural_tag: serde_json::Value,
) {
    let gd = common_request
        .sampling_options
        .guided_decoding
        .get_or_insert_default();
    gd.json = None;
    gd.regex = None;
    gd.structural_tag = Some(structural_tag);
}

/// The constraint the Beam turn grammar implies for a request, decided from request
/// facts alone so a reconstructing frontend reports what preprocessing installed:
/// every turn is held unless `DYN_BEAM_TURN_GRAMMAR=0` opts automatic choice out or an
/// explicit guided-decoding field constrains the answer itself; a `response_format` on
/// an unforced choice is framed inside the final channel. (A forced choice with
/// explicit guided decoding is rejected at request time.)
/// Plain `ignore_eos` requests skip the turn grammar to avoid grammar-driven early stops.
fn beam_turn_grammar_constraint<R: OAIChatLikeRequest + CommonExtProvider>(
    request: &R,
) -> GuidedToolConstraint {
    let explicit = has_explicit_guided_decoding_ext(request);
    match (
        OpenAIPreprocessor::has_guided_tool_choice(request),
        explicit,
    ) {
        (true, true) => GuidedToolConstraint::None,
        (true, false) => GuidedToolConstraint::BeamTurnGrammar {
            frames_answer: false,
        },
        (false, _) if explicit || !beam_turn_grammar_enabled() => GuidedToolConstraint::None,
        (false, _)
            if request.common_ext().and_then(|common| common.ignore_eos) == Some(true)
                && !OpenAIPreprocessor::has_structured_response_format(request)
                && (request
                    .tool_choice()
                    .is_some_and(|choice| choice.as_str() == Some("none"))
                    || !OpenAIPreprocessor::request_has_effective_tools(request)) =>
        {
            GuidedToolConstraint::None
        }
        (false, _) => GuidedToolConstraint::BeamTurnGrammar {
            frames_answer: OpenAIPreprocessor::has_structured_response_format(request),
        },
    }
}

/// Environment variable that opts a deployment out of the per-turn Beam grammar for
/// automatic tool choice (`0`, `false`, `off` or `no`). Forced tool choices keep theirs:
/// without a grammar, `required` and named choices are not enforced on Beam's wire.
pub(crate) const BEAM_TURN_GRAMMAR_ENV: &str = "DYN_BEAM_TURN_GRAMMAR";

fn beam_turn_grammar_enabled() -> bool {
    !dynamo_runtime::config::env_is_falsey(BEAM_TURN_GRAMMAR_ENV)
}

/// The Beam turn grammar from the parsers crate's family registry, built for the channel
/// the rendered prompt leaves open (`None` = the bare header slot) and, with
/// `answer_schema`, holding the answer in the final channel to that JSON schema. `None`
/// where the grammar derives nothing (see
/// `dynamo_parsers_v2::structural_tag::builders::beam`).
fn beam_turn_structural_tag(
    tools: &[ChatCompletionTool],
    tool_choice: &ToolChoice,
    parallel_tool_calls: Option<bool>,
    prompt_channel: Option<dynamo_parsers_v2::UnifiedParserStartingState>,
    answer_schema: Option<&serde_json::Value>,
) -> Option<serde_json::Value> {
    use dynamo_parsers_v2::UnifiedParserStartingState as State;
    use dynamo_parsers_v2::structural_tag::{
        PromptChannel, StructuralTagContext, StructuralTagSchemaMode, StructuralTagToolChoice,
    };
    let builder = dynamo_parsers_v2::structural_tag_builder_for_family(
        crate::protocols::openai::chat_completions::unified_parser::BEAM_UNIFIED_FAMILY,
    )?;
    let v2_tools: Vec<dynamo_parsers_v2::Tool> = tools
        .iter()
        .map(|tool| dynamo_parsers_v2::Tool {
            name: tool.function.name.clone(),
            description: None,
            parameters: tool
                .function
                .parameters
                .clone()
                .unwrap_or(serde_json::Value::Null),
            strict: tool.function.strict,
        })
        .collect();
    let context = StructuralTagContext {
        tool_choice: match tool_choice {
            ToolChoice::None => StructuralTagToolChoice::None,
            ToolChoice::Auto => StructuralTagToolChoice::Auto,
            ToolChoice::Required => StructuralTagToolChoice::Required,
            ToolChoice::Named(name) => StructuralTagToolChoice::Named(name.as_str()),
        },
        tools: &v2_tools,
        parallel_tool_calls,
        schema_mode: StructuralTagSchemaMode::Auto,
        structured_output_schema: answer_schema,
        starts_in_reasoning: false,
    };
    let channel = match prompt_channel {
        Some(State::Reasoning) => PromptChannel::OpenReasoning,
        Some(State::Response) => PromptChannel::OpenAnswer,
        Some(State::None) | None => PromptChannel::MessageStart,
    };
    match builder.build_turn(&context, channel) {
        Ok(tag) => tag,
        Err(error) => {
            tracing::warn!(%error, "Beam turn grammar failed to build; leaving the turn unconstrained");
            None
        }
    }
}

/// [`has_explicit_guided_decoding`] for any request type carrying the common extension
/// (the Beam turn-grammar decision runs on chat-like requests, not only the OpenAI
/// chat request); `get_guided_json` is not consulted because it also derives a schema
/// from `response_format`.
fn has_explicit_guided_decoding_ext<R: CommonExtProvider>(request: &R) -> bool {
    request
        .common_ext()
        .is_some_and(|common| common.guided_json.is_some() || common.guided_grammar.is_some())
        || request.get_guided_regex().is_some()
        || request
            .get_guided_choice()
            .is_some_and(|choices| !choices.is_empty())
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
    // Beam holds every turn with its grammar (`apply_tool_choice_guided_decoding`); a
    // reconstructing frontend reports the same so parsing frames the arguments and a
    // `response_format` answer the way that grammar shaped them.
    if is_beam_parser(tool_call_parser) || is_beam_parser(reasoning_parser) {
        return Ok(beam_turn_grammar_constraint(request));
    }
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
    // Kimi and Beam forced requests use native wire formats that generic JSON
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

    fn preprocessor_for(parser: &str) -> Arc<OpenAIPreprocessor> {
        preprocessor(parser, StructuralTagMode::Off)
    }

    fn preprocessor(parser: &str, mode: StructuralTagMode) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = (mode == StructuralTagMode::On)
            .then(crate::local_model::runtime_config::StructuralTagConfig::default);
        mdc.runtime_config.tool_call_parser = Some(parser.to_string());
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
    fn beam_forced_choices_install_the_harmony_wire_structural_tag_and_no_json_schema() {
        // Single tool, `required`, parallel calls allowed by default: the turn grammar the
        // parsers crate derives for the same tools (asserted there against the model
        // builder's own fixtures) is what the policy installs, and nothing else.
        let first = request(json!({"tools": tools(), "tool_choice": "required"}));
        let blueprint = beam_turn_structural_tag(
            effective_tools(&first.inner).unwrap().as_ref(),
            &ToolChoice::Required,
            None,
            None,
            None,
        )
        .expect("the parsers crate derives the Beam turn grammar");
        let preprocessor = preprocessor("beam", StructuralTagMode::Off);
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&first, &mut common_request, false)
            .unwrap();

        assert_eq!(
            applied,
            GuidedToolConstraint::BeamTurnGrammar {
                frames_answer: false
            }
        );
        let guided = common_request
            .sampling_options
            .guided_decoding
            .as_ref()
            .expect("beam forced choice installs guided decoding");
        assert!(guided.json.is_none() && guided.regex.is_none());
        assert_eq!(
            guided.structural_tag.as_ref().expect("structural tag"),
            &blueprint
        );

        // A named choice among several tools forces that tool only; `required` offers each
        // tool, and `parallel_tool_calls: false` drops the repeated-call group.
        let mut tools = tools();
        tools.as_array_mut().unwrap().push(json!({
            "type": "function",
            "function": {"name": "get-time", "parameters": {"type": "object", "properties": {}}}
        }));
        for (tool_choice, expected, unexpected) in [
            (
                json!({"type": "function", "function": {"name": "get-time"}}),
                vec!["to=get-time<"],
                vec!["to=get_weather<"],
            ),
            (
                json!("required"),
                vec!["to=get_weather<", "to=get-time<"],
                vec![],
            ),
        ] {
            let forced = request(json!({
                "tools": tools,
                "tool_choice": tool_choice,
                "parallel_tool_calls": false
            }));
            let mut common_request = preprocessed_request();
            preprocessor
                .apply_tool_choice_guided_decoding(&forced, &mut common_request, false)
                .unwrap();
            let tag = common_request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.structural_tag.as_ref())
                .expect("structural tag")
                .to_string();
            for needle in expected {
                assert!(tag.contains(needle), "{tag}");
            }
            for needle in unexpected {
                assert!(!tag.contains(needle), "{tag}");
            }
            assert!(
                !tag.contains(
                    r#""type":"star","content":{"type":"sequence","elements":[{"type":"or""#
                ),
                "{tag}"
            );
        }
    }

    fn installed_structural_tag(common_request: &PreprocessedRequest) -> Option<&Value> {
        common_request
            .sampling_options
            .guided_decoding
            .as_ref()
            .and_then(|guided| guided.structural_tag.as_ref())
    }

    #[test]
    fn beam_ignore_eos_skips_only_the_plain_chat_turn_grammar() {
        for mode in [StructuralTagMode::Off, StructuralTagMode::On] {
            let preprocessor = preprocessor("beam", mode);
            for extra in [
                json!({}),
                json!({"tools": tools(), "tool_choice": "none"}),
                json!({"tool_choice": "auto", "response_format": {"type": "text"}}),
            ] {
                for ignore_eos in [None, Some(false), Some(true)] {
                    let mut request = request(extra.clone());
                    request.common.ignore_eos = ignore_eos;
                    let mut common_request = preprocessed_request();
                    let applied = preprocessor
                        .apply_tool_choice_guided_decoding(&request, &mut common_request, false)
                        .unwrap();
                    let expected = if ignore_eos == Some(true) {
                        GuidedToolConstraint::None
                    } else {
                        GuidedToolConstraint::BeamTurnGrammar {
                            frames_answer: false,
                        }
                    };
                    assert_eq!(applied, expected, "{mode:?} {extra} {ignore_eos:?}");
                    assert_eq!(
                        installed_structural_tag(&common_request).is_some(),
                        ignore_eos != Some(true)
                    );
                    // Reconstructed response policy must match preprocessing with either parser slot.
                    for parsers in [(Some("beam"), None), (None, Some("beam"))] {
                        assert_eq!(
                            guided_tool_constraint(&request, parsers.0, parsers.1, false).unwrap(),
                            expected,
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn beam_ignore_eos_preserves_tool_and_output_constraints() {
        let preprocessor = preprocessor_for("beam");
        for extra in [
            json!({"tools": tools()}),
            json!({"messages": [
                {"role": "system", "content": "", "tools": [{"name": "lookup", "parameters": {"type": "object"}}]},
                {"role": "user", "content": "hi"}
            ]}),
            json!({"response_format": {"type": "json_object"}}),
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}}}),
        ] {
            let mut request = request(extra.clone());
            let mut normal = preprocessed_request();
            let expected = preprocessor
                .apply_tool_choice_guided_decoding(&request, &mut normal, false)
                .unwrap();
            assert!(installed_structural_tag(&normal).is_some(), "{extra}");
            request.common.ignore_eos = Some(true);
            let mut ignored = preprocessed_request();
            let applied = preprocessor
                .apply_tool_choice_guided_decoding(&request, &mut ignored, false)
                .unwrap();
            assert_eq!(applied, expected, "{extra}");
            assert_eq!(
                installed_structural_tag(&ignored),
                installed_structural_tag(&normal),
                "{extra}"
            );
        }
    }

    #[test]
    fn beam_automatic_choice_installs_the_turn_grammar_and_keeps_the_parser_route() {
        let preprocessor = preprocessor("beam", StructuralTagMode::Off);
        let auto = request(json!({"tools": tools()}));
        let blueprint = beam_turn_structural_tag(
            effective_tools(&auto.inner).unwrap().as_ref(),
            &ToolChoice::Auto,
            None,
            None,
            None,
        )
        .expect("the parsers crate derives the Beam turn grammar");
        let mut common_request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&auto, &mut common_request, false)
            .unwrap();

        assert_eq!(
            applied,
            GuidedToolConstraint::BeamTurnGrammar {
                frames_answer: false
            }
        );
        let guided = common_request
            .sampling_options
            .guided_decoding
            .as_ref()
            .expect("automatic choice installs the turn grammar");
        assert!(guided.json.is_none() && guided.regex.is_none());
        assert_eq!(
            guided.structural_tag.as_ref().expect("structural tag"),
            &blueprint
        );

        // Plain chat and `tool_choice: none` hold the reasoning, then the answer.
        let answer_only = beam_turn_structural_tag(&[], &ToolChoice::Auto, None, None, None)
            .expect("plain chat is held to analysis then the answer");
        for extra in [json!({}), json!({"tools": tools(), "tool_choice": "none"})] {
            let mut common_request = preprocessed_request();
            preprocessor
                .apply_tool_choice_guided_decoding(&request(extra), &mut common_request, false)
                .unwrap();
            assert_eq!(
                installed_structural_tag(&common_request),
                Some(&answer_only)
            );
        }

        // An explicit guided-decoding field constrains the answer instead of the turn.
        let mut constrained = request(json!({"tools": tools()}));
        constrained.common.guided_regex = Some("[a-z]+".to_string());
        let mut common_request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&constrained, &mut common_request, false)
            .unwrap();
        assert_eq!(applied, GuidedToolConstraint::None);
        assert!(installed_structural_tag(&common_request).is_none());

        // Other parsers see no Beam grammar.
        let hermes = preprocessor_for("hermes");
        let mut common_request = preprocessed_request();
        hermes
            .apply_tool_choice_guided_decoding(
                &request(json!({"tools": tools()})),
                &mut common_request,
                false,
            )
            .unwrap();
        assert!(installed_structural_tag(&common_request).is_none());
    }

    #[test]
    fn beam_response_format_is_held_in_the_final_channel_behind_the_analysis() {
        let preprocessor = preprocessor("beam", StructuralTagMode::Off);
        let schema = json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"]
        });
        let format = json!({
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema}
        });

        // Automatic choice: the calls stay available beside the framed answer, and the
        // bare JSON constraint `response_format` arrived as is replaced.
        let auto = request(json!({"tools": tools(), "response_format": format}));
        assert!(preprocessor.beam_frames_structured_response(&auto));
        assert!(!preprocessor.beam_output_is_bare_document(&auto));
        let blueprint = beam_turn_structural_tag(
            effective_tools(&auto.inner).unwrap().as_ref(),
            &ToolChoice::Auto,
            None,
            None,
            Some(&schema),
        )
        .expect("the parsers crate derives the framed answer");
        let mut common_request = preprocessed_request();
        common_request
            .sampling_options
            .guided_decoding
            .get_or_insert_default()
            .json = Some(schema.clone());
        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&auto, &mut common_request, false)
            .unwrap();
        assert_eq!(
            applied,
            GuidedToolConstraint::BeamTurnGrammar {
                frames_answer: true
            }
        );
        let guided = common_request
            .sampling_options
            .guided_decoding
            .as_ref()
            .expect("the framed answer is a structural tag");
        assert!(guided.json.is_none() && guided.regex.is_none());
        assert_eq!(guided.structural_tag.as_ref().expect("tag"), &blueprint);
        let tag = blueprint.to_string();
        assert!(
            tag.contains("<|channel|>final<|message|>")
                && tag.contains(r#""type":"json_schema""#)
                && tag.contains(r#""ok""#)
                && tag.contains("to=get_weather<"),
            "{tag}"
        );

        // `json_object` is any object; `tool_choice: none` and no tools frame the
        // answer only.
        for extra in [
            json!({"response_format": {"type": "json_object"}}),
            json!({
                "tools": tools(),
                "tool_choice": "none",
                "response_format": {"type": "json_object"}
            }),
        ] {
            let answer_only = request(extra);
            let mut common_request = preprocessed_request();
            let applied = preprocessor
                .apply_tool_choice_guided_decoding(&answer_only, &mut common_request, false)
                .unwrap();
            assert_eq!(
                applied,
                GuidedToolConstraint::BeamTurnGrammar {
                    frames_answer: true
                }
            );
            let tag = installed_structural_tag(&common_request)
                .expect("framed answer")
                .to_string();
            assert!(
                tag.contains(r#""json_schema":{"type":"object"}"#)
                    && !tag.contains("to=get_weather<"),
                "{tag}"
            );
        }

        // A forced choice ignores `response_format` (it governs content, not calls) and
        // still generates native calls that end on a marker.
        let forced = request(json!({
            "tools": tools(),
            "tool_choice": "required",
            "response_format": {"type": "json_object"}
        }));
        assert!(!preprocessor.beam_frames_structured_response(&forced));
        assert!(!preprocessor.beam_output_is_bare_document(&forced));
        let mut common_request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&forced, &mut common_request, false)
            .unwrap();
        assert_eq!(
            applied,
            GuidedToolConstraint::BeamTurnGrammar {
                frames_answer: false
            }
        );
        let tag = installed_structural_tag(&common_request)
            .expect("forced calls")
            .to_string();
        assert!(
            !tag.contains("json_schema") && tag.contains("to=get_weather<"),
            "{tag}"
        );

        // Explicit guided decoding stays a bare constraint from the first token.
        let mut explicit = request(json!({
            "tools": tools(),
            "response_format": {"type": "json_object"}
        }));
        explicit.common.guided_json = Some(schema.clone());
        assert!(!preprocessor.beam_frames_structured_response(&explicit));
        assert!(preprocessor.beam_output_is_bare_document(&explicit));
        let mut common_request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_guided_decoding(&explicit, &mut common_request, false)
            .unwrap();
        assert_eq!(applied, GuidedToolConstraint::None);
        assert!(installed_structural_tag(&common_request).is_none());

        // Other parsers leave `response_format` to the engine.
        let hermes = preprocessor_for("hermes");
        assert!(!hermes.beam_frames_structured_response(&auto));
        let mut common_request = preprocessed_request();
        let applied = hermes
            .apply_tool_choice_guided_decoding(&auto, &mut common_request, false)
            .unwrap();
        assert_eq!(applied, GuidedToolConstraint::None);
        assert!(installed_structural_tag(&common_request).is_none());

        // A reconstructing frontend reports what preprocessing installed.
        for (request, parsers, expected) in [
            (
                &auto,
                (Some("beam"), None),
                GuidedToolConstraint::BeamTurnGrammar {
                    frames_answer: true,
                },
            ),
            (
                &forced,
                (Some("beam"), Some("beam")),
                GuidedToolConstraint::BeamTurnGrammar {
                    frames_answer: false,
                },
            ),
            (&explicit, (None, Some("beam")), GuidedToolConstraint::None),
            (&auto, (Some("hermes"), None), GuidedToolConstraint::None),
        ] {
            assert_eq!(
                guided_tool_constraint(request, parsers.0, parsers.1, false).unwrap(),
                expected,
                "{parsers:?}"
            );
        }
    }

    #[test]
    fn beam_route_declares_legacy_builtin_declarations_as_tools() {
        let weather = json!({"type": "function", "function": {
            "name": "weather", "parameters": {"type": "object", "properties": {}}}});
        let mut beam = request(json!({"tools": [weather.clone()]}));
        beam.chat_template_args = Some(std::collections::HashMap::from([(
            "builtin_tools".to_string(),
            json!([
                "weather",
                {"name": "bash", "parameters": {"type": "object", "properties": {
                    "cmd": {"type": "string"}}}},
                {"type": "function", "function": {"name": "weather"}},
                {"type": "function", "function": {"name": "python",
                    "parameters": {"type": "object", "properties": {}}}}
            ]),
        )]));

        // Names and already-declared objects add nothing; undeclared objects join `tools`
        // in OpenAI form, once each.
        assert_eq!(
            declare_legacy_builtin_tools_for_beam(&mut beam, Some("beam"), None),
            2
        );
        let names: Vec<&str> = beam
            .inner
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .map(|tool| tool.function.name.as_str())
            .collect();
        assert_eq!(names, vec!["weather", "bash", "python"]);
        assert_eq!(
            declare_legacy_builtin_tools_for_beam(&mut beam, Some("beam"), None),
            0
        );

        // Other routes do not read `builtin_tools`.
        let mut hermes = request(json!({"tools": [weather]}));
        hermes.chat_template_args = beam.chat_template_args.clone();
        assert_eq!(
            declare_legacy_builtin_tools_for_beam(&mut hermes, Some("hermes"), None),
            0
        );
        assert_eq!(hermes.inner.tools.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn beam_route_serves_noncompliant_strict_tools_as_non_strict() {
        use crate::engines::ValidateRequest;
        // BFCL-style: optional properties, so `required` does not list every property.
        let loose = json!({"type": "function", "function": {
            "name": "triangle_properties_get", "strict": true,
            "parameters": {"type": "object", "properties": {
                "side1": {"type": "integer"}, "get_area": {"type": "boolean"}},
                "required": ["side1"], "additionalProperties": false}}});
        let tight = json!({"type": "function", "function": {
            "name": "ping", "strict": true,
            "parameters": {"type": "object", "properties": {"x": {"type": "integer"}},
                "required": ["x"], "additionalProperties": false}}});

        // Upstream's contract: the strict declaration makes the loose schema a 400.
        let mut beam = request(json!({"tools": [loose.clone(), tight.clone()]}));
        assert!(ValidateRequest::validate(&beam).is_err());

        // Beam route: the loose tool is served as non-strict, the compliant one keeps strict.
        assert_eq!(
            relax_noncompliant_strict_tools_for_beam(&mut beam, Some("beam"), Some("beam")),
            1
        );
        let tools = beam.inner.tools.as_ref().unwrap();
        assert_eq!(tools[0].function.strict, None);
        assert_eq!(tools[1].function.strict, Some(true));
        assert!(ValidateRequest::validate(&beam).is_ok());
        assert_eq!(
            relax_noncompliant_strict_tools_for_beam(&mut beam, Some("beam"), None),
            0
        );

        // Every other route keeps the rejection.
        let mut hermes = request(json!({"tools": [loose]}));
        assert_eq!(
            relax_noncompliant_strict_tools_for_beam(&mut hermes, Some("hermes"), None),
            0
        );
        assert_eq!(
            hermes.inner.tools.as_ref().unwrap()[0].function.strict,
            Some(true)
        );
        assert!(ValidateRequest::validate(&hermes).is_err());
    }

    #[test]
    fn beam_open_final_channel_holds_only_the_answer_and_forces_no_call() {
        let preprocessor = preprocessor("beam", StructuralTagMode::Off);
        let open_answer = PromptReasoningPrefill {
            legacy: false,
            unified: false,
            prefix: None,
            response: true,
            beam_channel: Some(dynamo_parsers_v2::UnifiedParserStartingState::Response),
        };

        let mut common_request = preprocessed_request();
        preprocessor
            .apply_tool_choice_guided_decoding(
                &request(json!({"tools": tools()})),
                &mut common_request,
                open_answer,
            )
            .unwrap();
        let tag = installed_structural_tag(&common_request).expect("answer rest");
        assert_eq!(tag["format"]["elements"][0]["type"], "any_text");
        assert!(!tag.to_string().contains("to=get_weather"));

        let mut common_request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_guided_decoding(
                &request(json!({"tools": tools(), "tool_choice": "required"})),
                &mut common_request,
                open_answer,
            )
            .unwrap();
        assert_eq!(applied, GuidedToolConstraint::None);
        assert!(installed_structural_tag(&common_request).is_none());
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
