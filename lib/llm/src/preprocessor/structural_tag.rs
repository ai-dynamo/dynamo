// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Structural tag policy for chat tool-call guided decoding.

mod v1;
mod v2;

use std::borrow::Cow;

use crate::local_model::runtime_config::{
    ModelRuntimeConfig, StructuralTagConfig, StructuralTagReasoningBoundary, StructuralTagScope,
    TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
    TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY,
};
use crate::preprocessor::{OpenAIPreprocessor, PreprocessedRequest};
use crate::protocols::openai::tools::{ToolChoiceValidation, validate_tool_choice_against_names};

use dynamo_parsers::tool_calling::{StructuralTagSchemaMode, ToolChoice, ToolDefinition};
use dynamo_protocols::types::ResponseFormat;
#[cfg(test)]
use dynamo_runtime::config::environment_names::llm as env_llm;
use dynamo_runtime::error::{DynamoError, ErrorType};

struct StructuralTagBuildRequest<'a> {
    tool_choice: &'a ToolChoice,
    tools: &'a [ToolDefinition],
    parallel_tool_calls: Option<bool>,
    schema_mode: StructuralTagSchemaMode,
    exclude_special_tokens: Option<bool>,
    reasoning_boundary: ResolvedReasoningBoundary,
    tool_arguments_any_order: bool,
    starts_in_reasoning: bool,
    structured_output_schema: Option<&'a serde_json::Value>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResolvedReasoningBoundary {
    StructuralTag,
    Backend,
}

fn resolve_reasoning_boundary(
    configured: StructuralTagReasoningBoundary,
    backend_excludes_reasoning: bool,
) -> Result<ResolvedReasoningBoundary, &'static str> {
    match (configured, backend_excludes_reasoning) {
        (StructuralTagReasoningBoundary::Auto, false)
        | (StructuralTagReasoningBoundary::StructuralTag, false) => {
            Ok(ResolvedReasoningBoundary::StructuralTag)
        }
        (StructuralTagReasoningBoundary::Auto, true)
        | (StructuralTagReasoningBoundary::Backend, _) => Ok(ResolvedReasoningBoundary::Backend),
        (StructuralTagReasoningBoundary::StructuralTag, true) => Err(
            "structural_tag.reasoning_boundary=structural_tag conflicts with the backend's reasoning-aware guided-decoding policy; use 'auto' or 'backend'",
        ),
    }
}

#[derive(Clone, Copy)]
pub(crate) enum SelectedStructuralTagBuilder {
    V1(v1::StructuralTagBuilder),
    V2(v2::StructuralTagBuilder),
}

impl SelectedStructuralTagBuilder {
    fn for_parser(parser_name: &str, reasoning_parser: Option<&str>) -> Option<Self> {
        if let Some((_, builder)) = Self::for_unified_parser(Some(parser_name), reasoning_parser) {
            return Some(builder);
        }

        v1::StructuralTagBuilder::for_parser(parser_name).map(Self::V1)
    }

    pub(super) fn for_unified_parser(
        tool_call_parser: Option<&str>,
        reasoning_parser: Option<&str>,
    ) -> Option<(&'static str, Self)> {
        let family = crate::protocols::openai::chat_completions::unified_parser::selected_family(
            tool_call_parser,
            reasoning_parser,
        )?;
        v2::StructuralTagBuilder::for_parser(family).map(|builder| (family, Self::V2(builder)))
    }

    fn build(
        self,
        request: &StructuralTagBuildRequest<'_>,
    ) -> anyhow::Result<Option<serde_json::Value>> {
        match self {
            Self::V1(builder) => builder.build(request),
            Self::V2(builder) => builder.build(request),
        }
    }
}

/// Validate a forced `tool_choice` against the request's actual `tools` list.
///
/// `Required` with no tools, and a `Named` choice for a tool absent from `tools`,
/// have nothing valid to constrain against. The rule itself lives in
/// `validate_tool_choice_against_names`, which is also used by the OpenAI-wire
/// schema builder; this adapter only maps the parser-facing types and error.
fn validate_forced_tool_choice(
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
) -> Result<(), DynamoError> {
    let tool_choice = match tool_choice {
        ToolChoice::Required => ToolChoiceValidation::Required,
        ToolChoice::Named(name) => ToolChoiceValidation::Named(name),
        ToolChoice::None | ToolChoice::Auto => ToolChoiceValidation::Unforced,
    };
    validate_tool_choice_against_names(tool_choice, tools.iter().map(|tool| tool.name.as_str()))
        .map_err(|error| {
            DynamoError::builder()
                .error_type(ErrorType::InvalidArgument)
                .message(error.to_string())
                .build()
        })
}

fn is_kimi_k3_parser(parser_name: Option<&str>) -> bool {
    parser_name.is_some_and(|parser| matches!(parser, "kimi_k3" | "kimi-k3"))
}

// Unlike K3, the parser registry exposes only the canonical `kimi_k2` spelling.
fn is_kimi_k2_parser(parser_name: Option<&str>) -> bool {
    parser_name == Some("kimi_k2")
}

// Beam tool calls live on the Harmony wire: an analysis channel first, then the
// `<|start|>assistant to=NAME<|channel|>analysis <|constrain|>xml<|message|>` call
// header with an `rfl:parameter` body. The generic forced-tool JSON grammar replaces
// that whole wire with a bare JSON object, so the model emits no analysis channel and
// the response carries no `reasoning_content`; the Beam chat template then rejects the
// follow-up request whose history replays that tool-call turn without its reasoning.
// Forced choices therefore stay on the parser path; the preprocessor installs a regex
// over the native wire instead (see `tool_choice.rs`), which keeps the analysis channel
// and still forces the call header of an allowed tool.
pub(crate) fn is_beam_parser(parser_name: Option<&str>) -> bool {
    parser_name == Some("beam")
}

fn is_muse_spark_parser(parser_name: Option<&str>) -> bool {
    matches!(parser_name, Some("muse_spark_open" | "muse_spark"))
}

// Spark's unified parser can be selected through either parser slot. Its
// forced-tool grammar must use that same family when only reasoning is set.
fn structural_tag_parser<'a>(tool: Option<&'a str>, reasoning: Option<&'a str>) -> Option<&'a str> {
    tool.or_else(|| reasoning.filter(|parser| is_muse_spark_parser(Some(parser))))
}

fn requires_intrinsic_structural_tag(parser_name: Option<&str>, tool_choice: &ToolChoice) -> bool {
    // K2/Muse Spark forced calls and K3 named calls cannot use the generic JSON-schema
    // fallback because both families emit native, marker-delimited formats.
    // Treat their structural tags as part of implementing these standard OpenAI
    // request shapes, not as an operator opt-in. K3 required remains on its
    // intentional prompt-level XTML path.
    (is_kimi_k2_parser(parser_name)
        && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)))
        || (is_kimi_k3_parser(parser_name) && matches!(tool_choice, ToolChoice::Named(_)))
        || (is_muse_spark_parser(parser_name)
            && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)))
}

pub(super) fn requires_native_tool_call_format(
    parser_name: Option<&str>,
    tool_choice: &ToolChoice,
) -> bool {
    // The generic forced-tool JSON grammar describes a different wire format
    // from Kimi's native marker-delimited calls, Beam's Harmony calls and Muse Spark's
    // calls. Required/named choices use best-effort prompting when no structural-tag
    // builder is available. This also covers K3 required, which uses its prompt/parser
    // path when structural guidance is off.
    let forced = matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_));
    forced
        && (is_kimi_k2_parser(parser_name)
            || is_kimi_k3_parser(parser_name)
            || is_beam_parser(parser_name)
            || is_muse_spark_parser(parser_name))
}

fn should_skip_tool_call_ban(exclude_tools_when_none: bool, tool_choice: &ToolChoice) -> bool {
    exclude_tools_when_none && matches!(tool_choice, ToolChoice::None)
}

/// Name the first regex construct in `pattern` that XGrammar cannot compile in
/// any tag format. `pattern` is ECMA-262, as JSON Schema specifies. Keep in step
/// with `unsupported_regex_construct` in the Python frontend.
///
/// Runs per request, so it scans bytes: every syntax character is ASCII.
fn unsupported_regex_construct(pattern: &str) -> Option<&'static str> {
    let bytes = pattern.as_bytes();
    let at = |i: usize| bytes.get(i).copied();
    let mut i = 0;
    let mut in_class = false;
    // Whether the previous token can take a quantifier.
    let mut can_repeat = false;
    // After a quantifier: 1 while a lazy `?` may follow, 2 once it has.
    let mut quantified = 0;
    while let Some(c) = at(i) {
        let following = at(i + 1);
        if in_class {
            match c {
                b'\\' => i += 1,
                b']' => in_class = false,
                _ => {}
            }
            i += 1;
            continue;
        }
        if matches!(c, b'*' | b'+' | b'?' | b'{') {
            if c == b'?' && quantified == 1 {
                quantified = 2;
                i += 1;
                continue;
            }
            if quantified != 0 {
                return Some("consecutive quantifiers");
            }
            if c == b'{' {
                match repetition_end(bytes, i) {
                    Some(end) => i = end,
                    None => return Some("malformed repetition"),
                }
            }
            if !can_repeat {
                return Some("quantifier without an element");
            }
            can_repeat = false;
            quantified = 1;
            i += 1;
            continue;
        }
        quantified = 0;
        can_repeat = true;
        match c {
            b'\\' => {
                match following {
                    Some(b'1'..=b'9' | b'k') => return Some("backreference"),
                    Some(b'b' | b'B') => return Some("word boundary"),
                    Some(b'p' | b'P') => return Some("Unicode property escape"),
                    _ => {}
                }
                // `\u{...}` is one escape; its braces are not a repetition.
                i = unicode_escape_end(bytes, i).unwrap_or(i + 1);
            }
            b'[' => {
                if following == Some(b']') {
                    return Some("empty character class");
                }
                in_class = true;
                if following == Some(b'^') {
                    i += 1;
                }
            }
            b'(' => {
                can_repeat = false;
                if following == Some(b'?') {
                    match (at(i + 2), at(i + 3)) {
                        (Some(b'<'), Some(b'=' | b'!')) => return Some("lookbehind"),
                        (Some(b'=' | b'!'), _) => return Some("lookahead"),
                        // `(?:` opens a group; `(?<` continues with a name.
                        (Some(b':'), _) => i += 2,
                        (Some(b'<'), _) => i += 1,
                        _ => return Some("group modifier"),
                    }
                }
            }
            b'|' | b'^' | b'$' => can_repeat = false,
            _ => {}
        }
        i += 1;
    }
    None
}

/// XGrammar rejects repetition counts above this.
const MAX_REPETITION: u64 = 1_000_000_000_000_000;

/// Index of the `}` closing a well-formed `{n}`, `{n,}`, or `{n,m}` at `start`.
fn repetition_end(bytes: &[u8], start: usize) -> Option<usize> {
    let number = |from: usize| {
        let digits = bytes.get(from..).unwrap_or_default();
        let len = digits.iter().take_while(|c| c.is_ascii_digit()).count();
        // Over 16 digits is always above the limit; up to 16 cannot overflow.
        let value = match len {
            0 => Some(None),
            1..=16 => std::str::from_utf8(&digits[..len])
                .ok()
                .and_then(|digits| digits.parse::<u64>().ok())
                .filter(|&count| count <= MAX_REPETITION)
                .map(Some),
            _ => None,
        };
        value.map(|value| (from + len, value))
    };
    let (end, min) = number(start + 1)?;
    let min = min?;
    match bytes.get(end) {
        Some(b'}') => Some(end),
        Some(b',') => {
            let (end, max) = number(end + 1)?;
            (bytes.get(end) == Some(&b'}') && max.is_none_or(|max| max >= min)).then_some(end)
        }
        _ => None,
    }
}

/// Index of the `}` closing a `\u{hex}` escape at `start`.
fn unicode_escape_end(bytes: &[u8], start: usize) -> Option<usize> {
    if bytes.get(start + 1..start + 3) != Some(b"u{".as_slice()) {
        return None;
    }
    let hex = &bytes[start + 3..];
    let len = hex.iter().take_while(|b| b.is_ascii_hexdigit()).count();
    (len > 0 && hex.get(len) == Some(&b'}')).then_some(start + 3 + len)
}

/// Instance data, not schemas.
const INSTANCE_KEYWORDS: [&str; 4] = ["const", "default", "enum", "examples"];
/// Searched by value only, so a property named `default` is still checked.
const SUBSCHEMA_MAPS: [&str; 6] = [
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
    "dependencies",
];

/// Name the first unsupported regex in `pattern` values or `patternProperties` keys.
fn unsupported_schema_construct(schema: &serde_json::Value) -> Option<&'static str> {
    match schema {
        serde_json::Value::Array(items) => items.iter().find_map(unsupported_schema_construct),
        serde_json::Value::Object(map) => {
            let own = map
                .get("pattern")
                .and_then(serde_json::Value::as_str)
                .and_then(unsupported_regex_construct)
                .or_else(|| {
                    map.get("patternProperties")
                        .and_then(serde_json::Value::as_object)
                        .and_then(|patterns| {
                            patterns
                                .keys()
                                .find_map(|pattern| unsupported_regex_construct(pattern))
                        })
                });
            if own.is_some() {
                return own;
            }
            map.iter()
                .find_map(|(key, value)| match (key.as_str(), value) {
                    (key, _) if INSTANCE_KEYWORDS.contains(&key) => None,
                    (key, serde_json::Value::Object(subschemas))
                        if SUBSCHEMA_MAPS.contains(&key) =>
                    {
                        subschemas.values().find_map(unsupported_schema_construct)
                    }
                    _ => unsupported_schema_construct(value),
                })
        }
        _ => None,
    }
}

/// Remove `pattern` values XGrammar cannot compile, in place. Fails on an
/// unsupported `patternProperties` key, since removing it would change which
/// property names are allowed.
fn strip_unsupported_patterns(schema: &mut serde_json::Value) -> Result<(), &'static str> {
    match schema {
        serde_json::Value::Array(items) => {
            items.iter_mut().try_for_each(strip_unsupported_patterns)
        }
        serde_json::Value::Object(map) => {
            if let Some(construct) = map
                .get("patternProperties")
                .and_then(serde_json::Value::as_object)
                .and_then(|patterns| {
                    patterns
                        .keys()
                        .find_map(|pattern| unsupported_regex_construct(pattern))
                })
            {
                return Err(construct);
            }
            if map
                .get("pattern")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|pattern| unsupported_regex_construct(pattern).is_some())
            {
                map.remove("pattern");
            }
            map.iter_mut()
                .try_for_each(|(key, value)| match (key.as_str(), value) {
                    (key, _) if INSTANCE_KEYWORDS.contains(&key) => Ok(()),
                    (key, serde_json::Value::Object(subschemas))
                        if SUBSCHEMA_MAPS.contains(&key) =>
                    {
                        subschemas
                            .values_mut()
                            .try_for_each(strip_unsupported_patterns)
                    }
                    (_, value) => strip_unsupported_patterns(value),
                })
        }
        _ => Ok(()),
    }
}

/// Leave regexes XGrammar cannot compile out of schemas Dynamo enforces on its
/// own: omitted `strict` under schema mode `auto`, which OpenAI treats as
/// non-strict. Otherwise one bad `pattern` fails the whole request. Explicit
/// `strict` and schema mode `strict` are left for the backend to reject.
fn adapt_unsupported_auto_schemas<'a>(
    parser_name: &str,
    tools: &'a [ToolDefinition],
    schema_mode: StructuralTagSchemaMode,
) -> Cow<'a, [ToolDefinition]> {
    if !matches!(schema_mode, StructuralTagSchemaMode::Auto) {
        return Cow::Borrowed(tools);
    }
    let unsupported = |tool: &ToolDefinition| {
        tool.strict
            .is_none()
            .then(|| {
                tool.parameters
                    .as_ref()
                    .and_then(unsupported_schema_construct)
            })
            .flatten()
    };
    if !tools.iter().any(|tool| unsupported(tool).is_some()) {
        return Cow::Borrowed(tools);
    }
    Cow::Owned(
        tools
            .iter()
            .map(|tool| {
                let mut parameters = tool.parameters.clone();
                let mut strict = tool.strict;
                if let Some(construct) = unsupported(tool) {
                    match parameters.as_mut().map(strip_unsupported_patterns) {
                        Some(Err(construct)) => {
                            tracing::debug!(
                                parser = parser_name,
                                tool = %tool.name,
                                construct,
                                "patternProperties uses a regex the grammar backend cannot \
                                 compile; not enforcing the tool's arguments"
                            );
                            parameters = tool.parameters.clone();
                            strict = Some(false);
                        }
                        _ => tracing::debug!(
                            parser = parser_name,
                            tool = %tool.name,
                            construct,
                            "Leaving a pattern the grammar backend cannot compile out of \
                             the tool's argument grammar"
                        ),
                    }
                }
                ToolDefinition {
                    name: tool.name.clone(),
                    parameters,
                    strict,
                }
            })
            .collect(),
    )
}

pub(super) fn uses_gpt_oss_required_array(
    (parser_name, reasoning_parser): (Option<&str>, Option<&str>),
    tool_choice: &ToolChoice,
    parallel_tool_calls: Option<bool>,
) -> bool {
    crate::protocols::openai::chat_completions::unified_parser::selected_family(
        parser_name,
        reasoning_parser,
    ) == Some("gpt_oss")
        && matches!(tool_choice, ToolChoice::Required)
        && parallel_tool_calls != Some(false)
}

/// Whether a request uses a native tool-call structural tag, bundled with proof
/// the parser registry can actually build one. Required GPT-OSS arrays keep their
/// guided-JSON parser mode even when their JSON payload has a structural envelope.
///
/// This is the single owner of "is a structural tag applicable and available" —
/// covering model-family/tool-choice intrinsic eligibility, the configured
/// structural-tag policy, tools-with-structured-output composition, the
/// `tool_choice=None` ban-tag exclusion, and real parser-registry builder
/// availability. A caller can only reach `Required` with a registered builder
/// selected for the active parser generation — it cannot ask for the eligibility
/// half of this decision without also getting the registry-availability proof in
/// the same value. Both the real
/// preprocessing path
/// (`apply_tool_choice_structural_tag`) and the HTTP-layer reconstruction
/// (`http::service::apply_request_tool_call_parsing_options`) must consult this
/// function; neither may re-derive eligibility, mode, scope, or registry
/// availability independently.
pub(crate) enum StructuralTagDecision {
    Required(SelectedStructuralTagBuilder),
    NotApplicable,
}

impl StructuralTagDecision {
    pub(crate) fn is_required(&self) -> bool {
        matches!(self, Self::Required(_))
    }
}

/// Decide whether a structural tag applies to this request. See
/// [`StructuralTagDecision`] for why this is the single shared owner.
///
/// Returns `Err` when `tool_choice` is a forced choice (`Required` or `Named`)
/// that is not valid against the request's `tools` — e.g. `required` with no
/// tools, or a named tool absent from `tools`. Both the real preprocessing path
/// (`apply_tool_choice_structural_tag`) and the HTTP-layer reconstruction
/// (`http::service::apply_request_tool_call_parsing_options`) must propagate
/// this error rather than falling back to `NotApplicable`, or an invalid forced
/// choice would silently install a structural tag anyway.
pub(crate) fn structural_tag_decision(
    (parser_name, reasoning_parser): (Option<&str>, Option<&str>),
    tool_choice: &ToolChoice,
    tools: &[ToolDefinition],
    parallel_tool_calls: Option<bool>,
    config: Option<&StructuralTagConfig>,
    has_structured_output: bool,
    exclude_tools_when_tool_choice_none: bool,
) -> Result<StructuralTagDecision, DynamoError> {
    // Validate before any mode or family gate. Kimi K3 required requests use a
    // prompt-level XTML path, but they still need an actual tool to require.
    validate_forced_tool_choice(tool_choice, tools)?;
    let parser_name = structural_tag_parser(parser_name, reasoning_parser);

    // A native Harmony handoff stops vLLM after the first call. Required parallel
    // calls instead share one guided JSON array, parsed as GuidedJsonRequired.
    if uses_gpt_oss_required_array(
        (parser_name, reasoning_parser),
        tool_choice,
        parallel_tool_calls,
    ) {
        return Ok(StructuralTagDecision::NotApplicable);
    }

    if config.is_none() && !requires_intrinsic_structural_tag(parser_name, tool_choice) {
        return Ok(StructuralTagDecision::NotApplicable);
    }

    let requires_tool_calls_with_structured_output = if has_structured_output {
        match tool_choice {
            ToolChoice::Auto => {
                if config.is_some_and(|config| config.allow_tool_calls_with_structured_output)
                    && !tools.is_empty()
                {
                    true
                } else {
                    return Ok(StructuralTagDecision::NotApplicable);
                }
            }
            ToolChoice::None => return Ok(StructuralTagDecision::NotApplicable),
            ToolChoice::Required | ToolChoice::Named(_) => false,
        }
    } else {
        false
    };

    if should_skip_tool_call_ban(exclude_tools_when_tool_choice_none, tool_choice) {
        // The prompt formatter already omits tools for this request. Avoid
        // sending a redundant AnyTokens structural tag: vLLM cannot
        // validate token-string exclusions without tokenizer metadata.
        return Ok(StructuralTagDecision::NotApplicable);
    }

    let Some(parser_name) = parser_name else {
        tracing::debug!(
            "Structural tag is enabled but --dyn-tool-call-parser is not set; \
             structural tags will not be applied"
        );
        return Ok(StructuralTagDecision::NotApplicable);
    };

    let Some(builder) = SelectedStructuralTagBuilder::for_parser(parser_name, reasoning_parser)
    else {
        return Ok(StructuralTagDecision::NotApplicable);
    };

    if matches!(tool_choice, ToolChoice::None) {
        if tools.is_empty() {
            return Ok(StructuralTagDecision::NotApplicable);
        }
        return Ok(StructuralTagDecision::Required(builder));
    }

    if requires_tool_calls_with_structured_output {
        return Ok(StructuralTagDecision::Required(builder));
    }

    let scope = config.map_or_else(Default::default, |config| config.scope);

    if !OpenAIPreprocessor::should_apply_tool_call_format(
        scope,
        tool_choice,
        tools,
        parallel_tool_calls,
    ) {
        return Ok(StructuralTagDecision::NotApplicable);
    }

    Ok(StructuralTagDecision::Required(builder))
}

impl OpenAIPreprocessor {
    pub(super) fn apply_structured_output_tag(
        &self,
        parser_name: &str,
        builder: SelectedStructuralTagBuilder,
        schema: &serde_json::Value,
        starts_in_reasoning: bool,
        request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        let config = self
            .runtime_config
            .structural_tag
            .clone()
            .unwrap_or_default();
        let applied = apply_structural_tag(
            parser_name,
            builder.build(&StructuralTagBuildRequest {
                tool_choice: &ToolChoice::None,
                tools: &[],
                parallel_tool_calls: None,
                schema_mode: config.schema,
                exclude_special_tokens: config.exclude_special_tokens,
                reasoning_boundary: ResolvedReasoningBoundary::StructuralTag,
                tool_arguments_any_order: false,
                starts_in_reasoning,
                structured_output_schema: Some(schema),
            }),
            request,
        )?;
        if applied {
            request
                .sampling_options
                .guided_decoding
                .get_or_insert_default()
                .json = None;
        }
        Ok(applied)
    }

    /// Apply structural tag guided decoding when enabled for this request.
    pub(super) fn apply_tool_choice_structural_tag(
        &self,
        tool_choice: &ToolChoice,
        tools: &[ToolDefinition],
        parallel_tool_calls: Option<bool>,
        prompt_injected_reasoning: bool,
        response_format: Option<&ResponseFormat>,
        preprocessed_request: &mut PreprocessedRequest,
    ) -> Result<bool, DynamoError> {
        let parser_name = structural_tag_parser(
            self.tool_call_parser.as_deref(),
            self.runtime_config.reasoning_parser.as_deref(),
        );
        let explicit_config = self.runtime_config.structural_tag.as_ref();

        let config = explicit_config.cloned().unwrap_or_default();
        let reasoning_boundary = if is_muse_spark_parser(parser_name) {
            // Muse's backend reasoner may not leave reasoning until a tool has
            // already been selected. The native grammar must own the entire
            // generation to enforce required/named choice. Setting require_reasoning
            // false below disables SGLang's per-request reasoning gate.
            ResolvedReasoningBoundary::StructuralTag
        } else if config.reasoning_boundary == StructuralTagReasoningBoundary::Auto
            && preprocessed_request.require_reasoning
            && self.structural_tag_reasoning_metadata(
                TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY,
            )
        {
            // SGLang consumes the prefix only for requests that activate its gate.
            ResolvedReasoningBoundary::Backend
        } else {
            self.structural_tag_reasoning_boundary
        };
        let StructuralTagDecision::Required(builder) = structural_tag_decision(
            (parser_name, self.runtime_config.reasoning_parser.as_deref()),
            tool_choice,
            tools,
            parallel_tool_calls,
            explicit_config,
            response_format.is_some_and(|format| !matches!(format, ResponseFormat::Text)),
            self.runtime_config.exclude_tools_when_tool_choice_none,
        )?
        else {
            return Ok(false);
        };
        let parser_name = parser_name.expect("Required decision implies a parser name");

        let structured_output_schema = if matches!(tool_choice, ToolChoice::Auto)
            && config.allow_tool_calls_with_structured_output
        {
            match response_format {
                Some(ResponseFormat::JsonSchema { json_schema }) => {
                    Some(Cow::Borrowed(&json_schema.schema))
                }
                Some(ResponseFormat::JsonObject) => {
                    Some(Cow::Owned(serde_json::json!({"type": "object"})))
                }
                Some(ResponseFormat::Text) | None => None,
            }
        } else {
            None
        };

        // After the decision: activation still follows the client's own `strict`.
        let tools = adapt_unsupported_auto_schemas(parser_name, tools, config.schema);
        let request = StructuralTagBuildRequest {
            tool_choice,
            tools: &tools,
            parallel_tool_calls,
            schema_mode: config.schema,
            exclude_special_tokens: config.exclude_special_tokens,
            reasoning_boundary,
            tool_arguments_any_order: config.tool_arguments_any_order,
            starts_in_reasoning: prompt_injected_reasoning,
            structured_output_schema: structured_output_schema.as_deref(),
        };

        let built_tag = match builder.build(&request) {
            Err(error)
                if is_muse_spark_parser(Some(parser_name))
                    && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_)) =>
            {
                return Err(DynamoError::builder()
                    .error_type(ErrorType::InvalidArgument)
                    .message(format!("cannot constrain native Muse Spark tools: {error}"))
                    .build());
            }
            Err(error)
                if !(matches!(tool_choice, ToolChoice::None)
                    || (is_muse_spark_parser(Some(parser_name))
                        && matches!(tool_choice, ToolChoice::Required | ToolChoice::Named(_))))
                    && structured_output_schema.is_none() =>
            {
                tracing::warn!(
                    parser = parser_name,
                    %error,
                    "Failed to build structural tag; using compatibility fallback"
                );
                return Ok(false);
            }
            result => result,
        };
        let applied = apply_structural_tag(parser_name, built_tag, preprocessed_request)?;
        if applied && structured_output_schema.is_some() {
            // Request preprocessing materializes `response_format` as guided JSON.
            // The composite structural tag now owns that schema.
            preprocessed_request
                .sampling_options
                .guided_decoding
                .get_or_insert_default()
                .json = None;
        }
        if applied && is_muse_spark_parser(Some(parser_name)) {
            preprocessed_request.require_reasoning = false;
        } else if applied
            && prompt_injected_reasoning
            && config.reasoning_boundary != StructuralTagReasoningBoundary::Auto
        {
            preprocessed_request.require_reasoning =
                reasoning_boundary == ResolvedReasoningBoundary::Backend;
        }
        Ok(applied)
    }

    fn structural_tag_reasoning_metadata(&self, key: &str) -> bool {
        match self.runtime_config.get_engine_specific::<bool>(key) {
            Ok(Some(excludes_reasoning)) => excludes_reasoning,
            Ok(None) => false,
            Err(error) => {
                tracing::warn!(
                    %error,
                    key,
                    "Ignoring invalid structural-tag reasoning metadata; using the compatibility behavior"
                );
                false
            }
        }
    }

    /// Decide whether this request should use a tool-call format tag.
    fn should_apply_tool_call_format(
        scope: StructuralTagScope,
        tool_choice: &ToolChoice,
        tools: &[ToolDefinition],
        parallel_tool_calls: Option<bool>,
    ) -> bool {
        match tool_choice {
            ToolChoice::None => false,
            ToolChoice::Required | ToolChoice::Named(_) => true,
            ToolChoice::Auto => match scope {
                StructuralTagScope::Always => true,
                StructuralTagScope::Auto => {
                    let explicit_single_call = parallel_tool_calls == Some(false);
                    tools.iter().any(|t| t.strict.unwrap_or(false)) || explicit_single_call
                }
            },
        }
    }
}

fn apply_structural_tag(
    parser_name: &str,
    structural_tag: anyhow::Result<Option<serde_json::Value>>,
    request: &mut PreprocessedRequest,
) -> Result<bool, DynamoError> {
    let structural_tag = match structural_tag {
        Ok(Some(tag)) => tag,
        Ok(None) => return Ok(false),
        Err(error) => {
            return Err(DynamoError::builder()
                .error_type(ErrorType::Unknown)
                .message(format!(
                    "failed to build structural_tag for parser '{parser_name}': {error}"
                ))
                .build());
        }
    };

    let guided_decoding = request
        .sampling_options
        .guided_decoding
        .get_or_insert_default();
    guided_decoding.structural_tag = Some(structural_tag);
    Ok(true)
}

pub(super) fn validate_runtime_config(
    runtime_config: &ModelRuntimeConfig,
) -> anyhow::Result<ResolvedReasoningBoundary> {
    let backend_excludes_reasoning = match runtime_config
        .get_engine_specific::<bool>(TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY)
    {
        // GPT-OSS's native reasoner is a Harmony stub, not a delayed grammar gate.
        Ok(_)
            if crate::protocols::openai::chat_completions::unified_parser::selected_family(
                runtime_config.tool_call_parser.as_deref(),
                runtime_config.reasoning_parser.as_deref(),
            ) == Some("gpt_oss") =>
        {
            false
        }
        Ok(value) => value.unwrap_or(false),
        Err(error) if runtime_config.structural_tag.is_none() => {
            tracing::warn!(
                %error,
                key = TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
                "Ignoring invalid structural-tag reasoning metadata"
            );
            false
        }
        Err(error) => return Err(error),
    };
    let configured_boundary = runtime_config
        .structural_tag
        .as_ref()
        .map_or(StructuralTagReasoningBoundary::default(), |config| {
            config.reasoning_boundary
        });
    if is_muse_spark_parser(structural_tag_parser(
        runtime_config.tool_call_parser.as_deref(),
        runtime_config.reasoning_parser.as_deref(),
    )) {
        anyhow::ensure!(
            !backend_excludes_reasoning
                && configured_boundary != StructuralTagReasoningBoundary::Backend,
            "Muse Spark structural tags require grammar-owned reasoning; enable vLLM structured_outputs_config.enable_in_reasoning and use reasoning_boundary=auto or structural_tag"
        );
        anyhow::ensure!(
            !runtime_config
                .structural_tag
                .as_ref()
                .is_some_and(|config| config.tool_arguments_any_order),
            "Muse Spark does not support structural_tag.tool_arguments_any_order"
        );
    }
    let reasoning_boundary =
        resolve_reasoning_boundary(configured_boundary, backend_excludes_reasoning)
            .map_err(anyhow::Error::msg)?;

    let Some(config) = runtime_config.structural_tag.as_ref() else {
        return Ok(reasoning_boundary);
    };

    if config.reasoning_boundary == StructuralTagReasoningBoundary::Backend {
        anyhow::ensure!(
            runtime_config.reasoning_parser.is_some(),
            "structural_tag.reasoning_boundary=backend requires a reasoning parser"
        );
    }

    let v2_only_option = config
        .allow_tool_calls_with_structured_output
        .then_some("allow_tool_calls_with_structured_output")
        .or(config
            .exclude_special_tokens
            .is_some()
            .then_some("exclude_special_tokens"))
        .or(
            (config.reasoning_boundary == StructuralTagReasoningBoundary::Backend)
                .then_some("reasoning_boundary"),
        )
        .or(config
            .tool_arguments_any_order
            .then_some("tool_arguments_any_order"));
    let Some(v2_only_option) = v2_only_option else {
        return Ok(reasoning_boundary);
    };

    anyhow::ensure!(
        crate::protocols::openai::chat_completions::unified_parser::selected_family(
            runtime_config.tool_call_parser.as_deref(),
            runtime_config.reasoning_parser.as_deref(),
        )
        .is_some(),
        "structural_tag.{v2_only_option} requires a compatible V2 parser route",
    );
    let parser_name = runtime_config.tool_call_parser.as_deref().ok_or_else(|| {
        anyhow::anyhow!("structural_tag.{v2_only_option} requires a tool-call parser")
    })?;
    anyhow::ensure!(
        v2::supports_family(parser_name),
        "structural_tag.{v2_only_option} is not supported by parser '{parser_name}'"
    );

    Ok(reasoning_boundary)
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, sync::Arc};

    use crate::{
        local_model::runtime_config::StructuralTagConfig,
        model_card::ModelDeploymentCard,
        protocols::common::{OutputOptions, SamplingOptions, StopConditions},
    };

    use super::*;

    fn structural_tag_preprocessor(exclude_tools_when_none: bool) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = Some(StructuralTagConfig::default());
        mdc.runtime_config.tool_call_parser = Some("qwen3_coder".to_string());
        mdc.runtime_config.exclude_tools_when_tool_choice_none = exclude_tools_when_none;

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

    fn kimi_k2_preprocessor(excludes_reasoning: Option<bool>) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = Some(StructuralTagConfig::default());
        mdc.runtime_config.tool_call_parser = Some("kimi_k2".to_string());
        if let Some(excludes_reasoning) = excludes_reasoning {
            mdc.runtime_config
                .set_engine_specific(
                    TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
                    excludes_reasoning,
                )
                .unwrap();
        }

        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn kimi_k3_preprocessor() -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.tool_call_parser = Some("kimi_k3".to_string());
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    fn kimi_k2_required_format(excludes_reasoning: Option<bool>) -> serde_json::Value {
        let preprocessor = kimi_k2_preprocessor(excludes_reasoning);
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        assert!(
            preprocessor
                .apply_tool_choice_structural_tag(
                    &ToolChoice::Required,
                    &tools,
                    None,
                    true,
                    None,
                    &mut request,
                )
                .unwrap()
        );

        request
            .sampling_options
            .guided_decoding
            .unwrap()
            .structural_tag
            .unwrap()["format"]
            .clone()
    }

    fn kimi_k2_call_section(format: &serde_json::Value) -> &serde_json::Value {
        if format["type"] == "triggered_tags" {
            assert_eq!(
                format["triggers"],
                serde_json::json!(["<|tool_calls_section_begin|>"])
            );
            assert_eq!(format["at_least_one"], true);
            let section = &format["tags"][0];
            assert_eq!(section["begin"], "<|tool_calls_section_begin|>");
            assert_eq!(section["end"], "<|tool_calls_section_end|>");
            &section["content"]
        } else {
            assert_eq!(format["type"], "sequence");
            assert_eq!(format["elements"][0]["type"], "const_string");
            assert_eq!(
                format["elements"][0]["value"],
                "<|tool_calls_section_begin|>"
            );
            assert_eq!(format["elements"][2]["value"], "<|tool_calls_section_end|>");
            &format["elements"][1]
        }
    }

    #[test]
    fn reasoning_metadata_controls_whether_forced_tool_tag_models_reasoning() {
        for policy in [None, Some(false)] {
            let format = kimi_k2_required_format(policy);
            assert_eq!(format["type"], "sequence");
            assert_eq!(format["elements"][0]["type"], "tag");
            assert_eq!(format["elements"][0]["end"], "</think>");
        }

        let format = kimi_k2_required_format(Some(true));
        assert_eq!(kimi_k2_call_section(&format)["type"], "tags_with_separator");
    }

    #[test]
    fn reasoning_boundary_resolution_follows_backend_policy_and_rejects_conflicts() {
        assert_eq!(
            resolve_reasoning_boundary(StructuralTagReasoningBoundary::Auto, false).unwrap(),
            ResolvedReasoningBoundary::StructuralTag
        );
        assert_eq!(
            resolve_reasoning_boundary(StructuralTagReasoningBoundary::Auto, true).unwrap(),
            ResolvedReasoningBoundary::Backend
        );
        assert_eq!(
            resolve_reasoning_boundary(StructuralTagReasoningBoundary::Backend, false).unwrap(),
            ResolvedReasoningBoundary::Backend
        );
        assert!(
            resolve_reasoning_boundary(StructuralTagReasoningBoundary::StructuralTag, true)
                .is_err()
        );
    }

    #[test]
    fn reasoning_boundary_is_resolved_when_the_preprocessor_is_created() {
        assert_eq!(
            kimi_k2_preprocessor(None).structural_tag_reasoning_boundary,
            ResolvedReasoningBoundary::StructuralTag
        );
        assert_eq!(
            kimi_k2_preprocessor(Some(true)).structural_tag_reasoning_boundary,
            ResolvedReasoningBoundary::Backend
        );
    }

    #[test]
    fn conflicting_reasoning_boundary_is_rejected_when_the_preprocessor_is_created() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = Some(StructuralTagConfig {
            reasoning_boundary: StructuralTagReasoningBoundary::StructuralTag,
            ..Default::default()
        });
        mdc.runtime_config
            .set_engine_specific(
                TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
                true,
            )
            .unwrap();

        let error = OpenAIPreprocessor::new(mdc)
            .err()
            .expect("the conflicting boundary must fail during preprocessor initialization");
        assert!(error.to_string().contains(
            "structural_tag.reasoning_boundary=structural_tag conflicts with the backend"
        ));
    }

    #[test]
    fn reasoning_only_spark_forced_choice_installs_native_grammar() {
        for parser in ["muse_spark", "muse_spark_open"] {
            let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
            let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
            mdc.runtime_config.tool_call_parser = None;
            mdc.runtime_config.reasoning_parser = Some(parser.into());
            // Startup must protect the same family selected for forced-call guidance.
            let mut late_config = mdc.runtime_config.clone();
            late_config
                .set_engine_specific(
                    TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
                    true,
                )
                .unwrap();
            assert!(
                validate_runtime_config(&late_config)
                    .unwrap_err()
                    .to_string()
                    .contains("grammar-owned reasoning")
            );
            let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
            for choice in [
                serde_json::json!("required"),
                serde_json::json!({"type":"function","function":{"name":"get_weather"}}),
            ] {
                let chat:crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest=serde_json::from_value(serde_json::json!({
                    "model":"m","messages":[{"role":"user","content":"Weather"}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"],"additionalProperties":false}}}],"tool_choice":choice,"parallel_tool_calls":false
                })).unwrap();
                let mut request = preprocessed_request();
                let constraint = preprocessor
                    .apply_tool_choice_guided_decoding(&chat, &mut request, false)
                    .unwrap();
                assert!(
                    matches!(
                        constraint,
                        crate::protocols::openai::GuidedToolConstraint::StructuralTag
                    ),
                    "{parser}: {constraint:?}"
                );
                let guidance = request.sampling_options.guided_decoding.unwrap();
                assert!(
                    guidance
                        .structural_tag
                        .unwrap()
                        .to_string()
                        .contains("atem:invoke")
                );
                assert!(guidance.json.is_none());
            }
        }
    }

    #[test]
    fn muse_spark_forced_choices_install_native_grammar_without_operator_opt_in() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"], "additionalProperties": false
            })),
            strict: Some(true),
        }];
        for parser in ["muse_spark_open", "muse_spark"] {
            let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
            let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
            mdc.runtime_config.tool_call_parser = Some(parser.to_string());
            let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
            for choice in [
                ToolChoice::Required,
                ToolChoice::Named("get_weather".into()),
            ] {
                let mut request = preprocessed_request();
                assert!(
                    preprocessor
                        .apply_tool_choice_structural_tag(
                            &choice,
                            &tools,
                            Some(false),
                            false,
                            None,
                            &mut request
                        )
                        .unwrap(),
                    "{parser} {choice:?} must be enforced by the native grammar"
                );
                let guidance = request.sampling_options.guided_decoding.as_ref().unwrap();
                assert!(
                    guidance
                        .structural_tag
                        .as_ref()
                        .unwrap()
                        .to_string()
                        .contains("atem:invoke")
                );
                assert!(guidance.json.is_none());
            }
        }
    }

    #[test]
    fn muse_spark_grammar_disables_sglang_gate_and_rejects_late_vllm_gate() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.tool_call_parser = Some("muse_spark_open".into());
        mdc.runtime_config
            .set_engine_specific(TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY, true)
            .unwrap();
        let mut late_config = mdc.runtime_config.clone();
        late_config
            .set_engine_specific(
                TOOL_CALL_STRUCTURAL_TAG_EXCLUDES_REASONING_RUNTIME_KEY,
                true,
            )
            .unwrap();
        assert!(
            validate_runtime_config(&late_config)
                .unwrap_err()
                .to_string()
                .contains("grammar-owned reasoning")
        );
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "ping".into(),
            parameters: Some(serde_json::json!({"type":"object", "properties":{}})),
            strict: Some(true),
        }];
        let mut request = preprocessed_request();
        request.require_reasoning = true;
        assert!(
            preprocessor
                .apply_tool_choice_structural_tag(
                    &ToolChoice::Required,
                    &tools,
                    Some(false),
                    false,
                    None,
                    &mut request
                )
                .unwrap()
        );
        assert!(
            !request.require_reasoning,
            "SGLang must apply the complete ATEM grammar from the first generated token"
        );
    }

    #[test]
    fn muse_spark_unsupported_constraints_do_not_silently_fall_back_to_prompting() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.tool_call_parser = Some("muse_spark_open".into());
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "lookup".into(),
            parameters: Some(
                serde_json::json!({"type":"object", "properties":{"q":{"type":"string", "pattern":"^A", "minLength":4, "contentMediaType":"text/plain"}}}),
            ),
            strict: Some(true),
        }];
        let error = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &tools,
                None,
                false,
                None,
                &mut preprocessed_request(),
            )
            .unwrap_err();
        assert_eq!(error.error_type(), ErrorType::InvalidArgument);
        assert!(
            error
                .to_string()
                .contains("cannot constrain native Muse Spark tools")
        );
    }

    #[test]
    fn named_kimi_k3_is_intrinsic_even_when_global_mode_is_off() {
        let named = ToolChoice::Named("get_weather".to_string());
        assert!(requires_intrinsic_structural_tag(Some("kimi_k3"), &named));
        assert!(requires_intrinsic_structural_tag(Some("kimi-k3"), &named));
    }

    #[test]
    fn forced_kimi_k2_is_intrinsic_even_when_global_mode_is_off() {
        assert!(requires_intrinsic_structural_tag(
            Some("kimi_k2"),
            &ToolChoice::Required
        ));
        assert!(requires_intrinsic_structural_tag(
            Some("kimi_k2"),
            &ToolChoice::Named("get_weather".to_string())
        ));
    }

    // Durable registry-parity property: every parser/tool_choice combination the
    // intrinsic predicate recognizes must have a real, registered
    // `StructuralTagBuilder` in the parser registry. If a future edit registers a
    // new intrinsic-family alias, or de-registers an existing one, without keeping
    // both in lockstep, `structural_tag_decision` would otherwise report `Required`
    // with no way to actually build the tag — this is the exact gap the shared
    // decision owner exists to close. No global-state mutation, no scratch worktree,
    // runs against the real production registry.
    #[test]
    fn every_intrinsic_parser_choice_has_a_registered_structural_tag_builder() {
        let cases: &[(&str, ToolChoice)] = &[
            ("kimi_k2", ToolChoice::Required),
            ("kimi_k2", ToolChoice::Named("get_weather".to_string())),
            ("kimi_k3", ToolChoice::Named("get_weather".to_string())),
            ("kimi-k3", ToolChoice::Named("get_weather".to_string())),
            ("muse_spark", ToolChoice::Required),
            ("muse_spark", ToolChoice::Named("get_weather".to_string())),
            ("muse_spark_open", ToolChoice::Required),
            (
                "muse_spark_open",
                ToolChoice::Named("get_weather".to_string()),
            ),
        ];
        for (parser, choice) in cases {
            assert!(
                requires_intrinsic_structural_tag(Some(parser), choice),
                "test fixture drifted: '{parser}' + {choice:?} is no longer intrinsic"
            );
            assert!(
                SelectedStructuralTagBuilder::for_parser(parser, None).is_some(),
                "'{parser}' is intrinsic per the predicate but the parser registry has \
                 no structural-tag builder for it — the predicate and the registry have \
                 drifted apart"
            );
        }
    }

    #[test]
    fn other_choices_and_parsers_still_follow_the_global_mode() {
        assert!(!requires_intrinsic_structural_tag(
            Some("kimi_k3"),
            &ToolChoice::Required
        ));
        assert!(!requires_intrinsic_structural_tag(
            Some("hermes"),
            &ToolChoice::Named("get_weather".to_string())
        ));
        assert!(!requires_intrinsic_structural_tag(
            Some("kimi_k2"),
            &ToolChoice::Auto
        ));
    }

    #[test]
    fn tools_with_structured_output_require_a_tag_independently_of_scope() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let config = StructuralTagConfig {
            scope: StructuralTagScope::Auto,
            allow_tool_calls_with_structured_output: true,
            ..Default::default()
        };

        let decision = structural_tag_decision(
            (Some("qwen3_coder"), None),
            &ToolChoice::Auto,
            &tools,
            None,
            Some(&config),
            true,
            true,
        )
        .unwrap();

        assert!(decision.is_required());
    }

    #[test]
    fn structured_output_requires_explicit_tool_call_composition_opt_in() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        for scope in [StructuralTagScope::Auto, StructuralTagScope::Always] {
            let config = StructuralTagConfig {
                scope,
                allow_tool_calls_with_structured_output: false,
                ..Default::default()
            };
            let decision = structural_tag_decision(
                (Some("qwen3_coder"), None),
                &ToolChoice::Auto,
                &tools,
                None,
                Some(&config),
                true,
                true,
            )
            .unwrap();

            assert!(!decision.is_required());
        }

        let config = StructuralTagConfig {
            scope: StructuralTagScope::Auto,
            allow_tool_calls_with_structured_output: true,
            ..Default::default()
        };
        let decision = structural_tag_decision(
            (Some("qwen3_coder"), None),
            &ToolChoice::Auto,
            &tools,
            None,
            Some(&config),
            false,
            true,
        )
        .unwrap();

        assert!(!decision.is_required());
    }

    #[test]
    fn structured_output_takes_precedence_over_a_tool_call_ban() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let config = StructuralTagConfig::default();

        let decision = structural_tag_decision(
            (Some("qwen3_coder"), None),
            &ToolChoice::None,
            &tools,
            None,
            Some(&config),
            true,
            false,
        )
        .unwrap();

        assert!(!decision.is_required());
    }

    #[test]
    fn kimi_k2_required_installs_native_tag_when_global_mode_is_off() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.tool_call_parser = Some("kimi_k2".to_string());
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            })),
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(applied);
        let format = &request
            .sampling_options
            .guided_decoding
            .as_ref()
            .unwrap()
            .structural_tag
            .as_ref()
            .unwrap()["format"];
        let section = kimi_k2_call_section(format);
        assert_eq!(section["type"], "tags_with_separator");
        assert_eq!(section["at_least_one"], true);
        assert_eq!(
            section["tags"][0]["begin"],
            "<|tool_call_begin|>functions.get_weather:"
        );
    }

    #[test]
    fn kimi_k3_required_stays_non_structural_when_global_mode_is_off() {
        let preprocessor = kimi_k3_preprocessor();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    // Regression for the CodeRabbit finding on PR #12576 (structural_tag.rs is the
    // real root cause; the finding was filed against http/service.rs lines 66-81):
    // `should_apply_tool_call_format` returned `true` for `Required`/`Named`
    // unconditionally, so `structural_tag_decision` built a structural tag for a
    // forced tool_choice that `get_json_schema_from_tools` would have rejected on
    // the non-structural-tag path. This must be rejected on the REAL preprocessing
    // path (`apply_tool_choice_structural_tag`), not just the HTTP compatibility
    // helper.
    #[test]
    fn kimi_k2_required_with_empty_tools_is_rejected() {
        let preprocessor = kimi_k2_preprocessor(None);
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "kimi_k2 + required with no tools must be rejected, not silently \
                 resolved to a structural tag",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn kimi_k3_required_with_empty_tools_is_rejected_before_the_mode_gate() {
        let preprocessor = kimi_k3_preprocessor();
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .expect_err("required with no tools must be rejected before Kimi K3's XTML path");
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn kimi_k2_named_tool_absent_from_tools_is_rejected() {
        let preprocessor = kimi_k2_preprocessor(None);
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Named("does_not_exist".to_string()),
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "a named tool_choice for a tool absent from `tools` must be rejected, \
                 not silently resolved to a structural tag",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn operator_enabled_qwen3_coder_required_with_empty_tools_is_rejected() {
        let preprocessor = structural_tag_preprocessor(false);
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Required,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "operator-configured structural tags must not let required-with-no-tools \
                 through for a non-Kimi registry-supported parser either",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn operator_enabled_qwen3_coder_named_tool_absent_from_tools_is_rejected() {
        let preprocessor = structural_tag_preprocessor(false);
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];
        let mut request = preprocessed_request();

        let err = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Named("does_not_exist".to_string()),
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .expect_err(
                "operator-enabled structural tags must reject a named tool_choice \
                 that is absent from `tools`",
            );
        assert_eq!(err.error_type(), ErrorType::InvalidArgument);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    // `None`/`Auto` are not forced choices; the new validation must not affect them
    // even when `tools` is empty.
    #[test]
    fn none_and_auto_are_unaffected_by_forced_choice_validation_with_empty_tools() {
        let preprocessor = kimi_k2_preprocessor(None);

        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::None,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();
        assert!(!applied);

        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Auto,
                &[],
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();
        assert!(!applied);
    }

    #[test]
    fn structural_tag_generation_follows_model_pair_in_isolated_processes() {
        const CHILD: &str = "DYNAMO_STRUCTURAL_TAG_PAIR_CHILD";
        let Ok(version) = std::env::var(CHILD) else {
            for version in ["auto", "1", "2"] {
                let output = crate::test_utils::isolated_command(
                    "preprocessor::structural_tag::tests::structural_tag_generation_follows_model_pair_in_isolated_processes",
                ).env(CHILD, version).env(env_llm::DYN_PARSER_VERSION, version).output().unwrap();
                crate::test_utils::assert_isolated_success(&output);
            }
            return;
        };
        for (tool, reasoning) in [
            ("qwen3_coder", Some("qwen3")),
            ("qwen3_coder", None),
            ("glm47", Some("glm45")),
            ("qwen3_coder", Some("deepseek_r1")),
            ("harmony", Some("gpt_oss")),
            ("gpt_oss", None),
            ("harmony", Some("deepseek_r1")),
        ] {
            let compatible = reasoning != Some("deepseek_r1");
            let uses_v2 = version != "1" && compatible;
            assert_eq!(
                uses_gpt_oss_required_array((Some(tool), reasoning), &ToolChoice::Required, None),
                uses_v2 && matches!(tool, "harmony" | "gpt_oss")
            );
            assert!(!uses_gpt_oss_required_array(
                (Some(tool), reasoning),
                &ToolChoice::Required,
                Some(false),
            ));
            assert_eq!(
                matches!(
                    SelectedStructuralTagBuilder::for_parser(tool, reasoning),
                    Some(SelectedStructuralTagBuilder::V2(_))
                ),
                uses_v2
            );
            let config = ModelRuntimeConfig {
                tool_call_parser: Some(tool.into()),
                reasoning_parser: reasoning.map(str::to_string),
                structural_tag: Some(StructuralTagConfig {
                    tool_arguments_any_order: true,
                    ..Default::default()
                }),
                ..Default::default()
            };
            assert_eq!(validate_runtime_config(&config).is_ok(), uses_v2);
        }
    }

    #[test]
    fn tool_choice_none_skips_ban_only_when_prompt_excludes_tools() {
        const CHILD: &str = "DYNAMO_STRUCTURAL_TAG_NONE_TEST_CHILD";
        if std::env::var_os(CHILD).is_none() {
            for enabled in ["1", "2"] {
                let output = std::process::Command::new(std::env::current_exe().unwrap())
                    .args([
                        "--exact",
                        "preprocessor::structural_tag::tests::tool_choice_none_skips_ban_only_when_prompt_excludes_tools",
                        "--nocapture",
                    ])
                    .env(CHILD, enabled)
                    .env(env_llm::DYN_PARSER_VERSION, enabled)
                    .output()
                    .unwrap();
                assert!(
                    output.status.success()
                        && String::from_utf8_lossy(&output.stdout).contains("running 1 test"),
                    "parsers v2={enabled}:\n{}\n{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr),
                );
            }
            return;
        }
        assert_eq!(v2::enabled(), std::env::var(CHILD).unwrap() == "2");

        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        let preprocessor = structural_tag_preprocessor(true);
        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::None,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());

        let preprocessor = structural_tag_preprocessor(false);
        let mut request = preprocessed_request();
        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::None,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(applied);
        let structural_tag = request
            .sampling_options
            .guided_decoding
            .as_ref()
            .and_then(|guided| guided.structural_tag.as_ref())
            .expect("tool-call ban should be installed");
        let expected_content = if matches!(
            SelectedStructuralTagBuilder::for_parser("qwen3_coder", None),
            Some(SelectedStructuralTagBuilder::V2(_))
        ) {
            serde_json::json!({
                "type": "any_text",
                "excludes": ["<tool_call>", "<function="]
            })
        } else {
            serde_json::json!({
                "type": "any_tokens",
                "exclude_tokens": ["<tool_call>"]
            })
        };
        assert_eq!(structural_tag["format"]["content"], expected_content);
    }

    #[test]
    fn kimi_k3_reasoning_gate_avoids_duplicate_prefix_per_request() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })),
            strict: None,
        }];
        for (gate, require_reasoning, prompt_injected_reasoning, choice, expects_prefix) in [
            (Some(true), true, true, ToolChoice::Required, false),
            (None, true, true, ToolChoice::Required, true),
            (Some(false), true, true, ToolChoice::Required, true),
            (Some(true), false, true, ToolChoice::Auto, true),
            (Some(true), true, false, ToolChoice::Required, false),
            (
                Some(true),
                false,
                false,
                ToolChoice::Named("get_weather".to_string()),
                false,
            ),
        ] {
            let mut mdc = ModelDeploymentCard::load_from_disk(&model_path, None).unwrap();
            mdc.runtime_config.structural_tag = Some(StructuralTagConfig {
                scope: StructuralTagScope::Always,
                ..Default::default()
            });
            mdc.runtime_config.tool_call_parser = Some("kimi_k3".to_string());
            if let Some(gate) = gate {
                mdc.runtime_config
                    .set_engine_specific(TOOL_CALL_STRUCTURAL_TAG_REASONING_GATE_RUNTIME_KEY, gate)
                    .unwrap();
            }
            let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
            let mut request = preprocessed_request();
            request.require_reasoning = require_reasoning;
            assert!(
                preprocessor
                    .apply_tool_choice_structural_tag(
                        &choice,
                        &tools,
                        None,
                        prompt_injected_reasoning,
                        None,
                        &mut request,
                    )
                    .unwrap(),
                "gate={gate:?}, require_reasoning={require_reasoning}, choice={choice:?}"
            );
            let tag = request
                .sampling_options
                .guided_decoding
                .unwrap()
                .structural_tag
                .unwrap();
            let has_prefix = tag["format"]["elements"][0]["end"] == "<|close|>think<|sep|>";
            assert_eq!(
                has_prefix, expects_prefix,
                "gate={gate:?}, require_reasoning={require_reasoning}, prompt_injected_reasoning={prompt_injected_reasoning}, choice={choice:?}"
            );
            assert!(tag.to_string().contains("<|open|>tools<|sep|>"));
        }
    }

    #[test]
    fn kimi_forced_choices_are_classified_as_native_format_fallbacks() {
        let named = ToolChoice::Named("get_weather".to_string());
        assert!(requires_native_tool_call_format(
            Some("kimi_k2"),
            &ToolChoice::Required
        ));
        assert!(requires_native_tool_call_format(Some("kimi_k2"), &named));
        assert!(requires_native_tool_call_format(
            Some("kimi_k3"),
            &ToolChoice::Required
        ));
        assert!(requires_native_tool_call_format(Some("kimi_k3"), &named));
        assert!(requires_native_tool_call_format(Some("kimi-k3"), &named));
        assert!(!requires_native_tool_call_format(
            Some("kimi_k3"),
            &ToolChoice::Auto
        ));
        assert!(!requires_native_tool_call_format(
            Some("hermes"),
            &ToolChoice::Named("get_weather".to_string())
        ));
        assert!(!requires_native_tool_call_format(
            Some("kimi_k2"),
            &ToolChoice::Auto
        ));
    }

    #[test]
    fn beam_forced_choices_keep_the_harmony_wire_and_its_analysis_channel() {
        let named = ToolChoice::Named("get_weather".to_string());
        assert!(requires_native_tool_call_format(
            Some("beam"),
            &ToolChoice::Required
        ));
        assert!(requires_native_tool_call_format(Some("beam"), &named));
        assert!(!requires_native_tool_call_format(
            Some("beam"),
            &ToolChoice::Auto
        ));
        assert!(!requires_native_tool_call_format(
            Some("beam"),
            &ToolChoice::None
        ));
    }

    #[test]
    fn kimi_forced_choices_keep_native_tags_when_global_mode_is_off() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            })),
            strict: None,
        }];
        for (parser, choice, marker) in [
            (
                "kimi_k2",
                ToolChoice::Required,
                "<|tool_calls_section_begin|>",
            ),
            (
                "kimi_k2",
                ToolChoice::Named("get_weather".to_string()),
                "<|tool_calls_section_begin|>",
            ),
            (
                "kimi_k3",
                ToolChoice::Named("get_weather".to_string()),
                "<|open|>call",
            ),
            (
                "kimi-k3",
                ToolChoice::Named("get_weather".to_string()),
                "<|open|>call",
            ),
        ] {
            let mut mdc = ModelDeploymentCard::load_from_disk(&model_path, None).unwrap();
            mdc.runtime_config.structural_tag = None;
            mdc.runtime_config.tool_call_parser = Some(parser.to_string());
            let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
            let mut request = preprocessed_request();

            let applied = preprocessor
                .apply_tool_choice_structural_tag(&choice, &tools, None, false, None, &mut request)
                .unwrap();

            assert!(applied, "{parser} + {choice:?} must retain its native tag");
            let tag = request
                .sampling_options
                .guided_decoding
                .as_ref()
                .and_then(|guided| guided.structural_tag.as_ref())
                .expect("forced Kimi choice must install a native structural tag");
            assert!(tag.to_string().contains(marker));
            assert!(tag.to_string().contains("get_weather"));
        }
    }

    #[test]
    fn global_mode_off_is_authoritative_for_kimi_k3_auto() {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = None;
        mdc.runtime_config.tool_call_parser = Some("kimi_k3".to_string());
        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            })),
            // K3 auto must be constrained even when the caller does not opt in
            // to OpenAI strict schema enforcement.
            strict: None,
        }];
        let mut request = preprocessed_request();

        let applied = preprocessor
            .apply_tool_choice_structural_tag(
                &ToolChoice::Auto,
                &tools,
                None,
                false,
                None,
                &mut request,
            )
            .unwrap();

        assert!(!applied);
        assert!(request.sampling_options.guided_decoding.is_none());
    }

    #[test]
    fn always_scope_activates_auto_when_strict_is_omitted() {
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: None,
            strict: None,
        }];

        assert!(OpenAIPreprocessor::should_apply_tool_call_format(
            StructuralTagScope::Always,
            &ToolChoice::Auto,
            &tools,
            None,
        ));
        assert!(!OpenAIPreprocessor::should_apply_tool_call_format(
            StructuralTagScope::Auto,
            &ToolChoice::Auto,
            &tools,
            None,
        ));
    }

    // Each verdict matches XGrammar 0.2.7. The Python frontend test shares this table.
    #[test]
    fn unsupported_regex_constructs_match_the_grammar_compiler() {
        for (pattern, expected) in [
            (r"^(?!\s*$).+", Some("lookahead")),
            (r"^(?=.*\d).+$", Some("lookahead")),
            (r"\\(?=x)", Some("lookahead")),
            (r"(?<=a)b", Some("lookbehind")),
            (r"(?<!a)b", Some("lookbehind")),
            (r"(a)\1", Some("backreference")),
            (r"\8", Some("backreference")),
            (r"(?<n>a)\k<n>", Some("backreference")),
            (r"[^]\1]x", Some("backreference")),
            (r"(?>ab)", Some("group modifier")),
            (r"(?i)abc", Some("group modifier")),
            (r"(?i:a)", Some("group modifier")),
            (r"(?P<year>\d{4})", Some("group modifier")),
            (r"(?#note)a", Some("group modifier")),
            (r"a\b", Some("word boundary")),
            (r"\Ba", Some("word boundary")),
            (r"\p{L}+", Some("Unicode property escape")),
            (r"[]a", Some("empty character class")),
            (r"a++", Some("consecutive quantifiers")),
            (r"a{2}*", Some("consecutive quantifiers")),
            (r"a+??", Some("consecutive quantifiers")),
            (r"a{,3}", Some("malformed repetition")),
            (r"a{x}", Some("malformed repetition")),
            (r"a{ 2}", Some("malformed repetition")),
            (r"a{3,2}", Some("malformed repetition")),
            (r"x{", Some("malformed repetition")),
            (r"{2}a", Some("quantifier without an element")),
            (r"a{1000000000000001}", Some("malformed repetition")),
            (r"a{1,1000000000000001}", Some("malformed repetition")),
            (r"a{99999999999999999}", Some("malformed repetition")),
            (r"é+(?=ø)", Some("lookahead")),
            (r"^[a-z]+$", None),
            (r"(?<year>\d{4})", None),
            (r"(?:ab)+", None),
            (r"\(\?=x\)", None),
            (r"[(?=]+", None),
            (r"[\b\p{L}]", None),
            (r"[^]a", None),
            (r"a+?b*?c??d{2}?", None),
            (r"a{2}b{2,}c{2,3}d{02}", None),
            (r"a{1000000000000000}b{1,1000000000000000}", None),
            (r"\u{1F600}+", None),
            (r"a}", None),
            (r"[ä-ö]+ø{2}\é", None),
            (r"\{2}", None),
            (r"\0\cA\x41\u0041", None),
        ] {
            assert_eq!(unsupported_regex_construct(pattern), expected, "{pattern}");
        }
    }

    #[test]
    fn client_sized_repetition_counts_are_rejected_without_parsing() {
        let pattern = format!("a{{1,{}}}", "9".repeat(5000));
        assert_eq!(
            unsupported_regex_construct(&pattern),
            Some("malformed repetition")
        );
    }

    #[test]
    fn unsupported_schema_constructs_are_found_in_subschemas_only() {
        let lookahead = r"^(?!\s*$).+";
        for schema in [
            serde_json::json!({"properties": {"a": {"items": {"pattern": lookahead}}}}),
            serde_json::json!({"anyOf": [{"type": "integer"}, {"pattern": lookahead}]}),
            serde_json::json!({"patternProperties": {lookahead: {"type": "string"}}}),
            serde_json::json!({"$defs": {"name": {"pattern": lookahead}}}),
            serde_json::json!({"properties": {"default": {"pattern": lookahead}}}),
            serde_json::json!({"dependencies": {"enum": {"pattern": lookahead}}}),
        ] {
            assert_eq!(
                unsupported_schema_construct(&schema),
                Some("lookahead"),
                "{schema}"
            );
        }
        for schema in [
            serde_json::json!({"properties": {"a": {"type": "string", "pattern": "^[a-z]+$"}}}),
            serde_json::json!({"properties": {"pattern": {"type": "string"}}}),
            serde_json::json!({"const": {"pattern": lookahead}}),
            serde_json::json!({"default": {"pattern": lookahead}}),
            serde_json::json!({"enum": [{"pattern": lookahead}]}),
            serde_json::json!({"examples": [{"pattern": lookahead}]}),
        ] {
            assert_eq!(unsupported_schema_construct(&schema), None, "{schema}");
        }
    }

    fn record_value_tool(pattern: &str, strict: Option<bool>) -> ToolDefinition {
        ToolDefinition {
            name: "record_value".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"value": {"type": "string", "pattern": pattern}},
                "required": ["value"],
                "additionalProperties": false
            })),
            strict,
        }
    }

    fn record_value_tool_with_named_properties(pattern: &str) -> ToolDefinition {
        ToolDefinition {
            name: "record_value".to_string(),
            parameters: Some(serde_json::json!({
                "type": "object",
                "patternProperties": {pattern: {"type": "string"}},
                "additionalProperties": false
            })),
            strict: None,
        }
    }

    /// The named `record_value` call in a structural tag: the tag whose `begin`
    /// names the tool.
    fn record_value_call(
        preprocessor: &OpenAIPreprocessor,
        tool: ToolDefinition,
    ) -> serde_json::Value {
        fn call(value: &serde_json::Value) -> Option<&serde_json::Value> {
            match value {
                serde_json::Value::Object(map)
                    if map
                        .get("begin")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|begin| begin.contains("record_value")) =>
                {
                    Some(value)
                }
                serde_json::Value::Object(map) => map.values().find_map(call),
                serde_json::Value::Array(items) => items.iter().find_map(call),
                _ => None,
            }
        }

        let mut request = preprocessed_request();
        assert!(
            preprocessor
                .apply_tool_choice_structural_tag(
                    &ToolChoice::Named("record_value".to_string()),
                    &[tool],
                    None,
                    false,
                    None,
                    &mut request,
                )
                .unwrap()
        );
        let tag = request
            .sampling_options
            .guided_decoding
            .unwrap()
            .structural_tag
            .unwrap();
        call(&tag)
            .unwrap_or_else(|| panic!("no tag begins with the tool name: {tag}"))
            .clone()
    }

    /// The JSON-schema argument grammar of the named `record_value` call.
    fn record_value_arguments(
        preprocessor: &OpenAIPreprocessor,
        tool: ToolDefinition,
    ) -> serde_json::Value {
        fn find<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a serde_json::Value> {
            match value {
                serde_json::Value::Object(map) => map
                    .get(key)
                    .or_else(|| map.values().find_map(|value| find(value, key))),
                serde_json::Value::Array(items) => items.iter().find_map(|value| find(value, key)),
                _ => None,
            }
        }
        let call = record_value_call(preprocessor, tool);
        find(&call, "json_schema")
            .unwrap_or_else(|| panic!("no argument grammar: {call}"))
            .clone()
    }

    #[derive(Debug, PartialEq)]
    enum Arguments {
        /// The client's schema, pattern included.
        Enforced,
        /// The client's schema without the pattern.
        PatternLeftOut,
        /// Any arguments.
        Unconstrained,
    }

    fn arguments_kind(arguments: &serde_json::Value) -> Arguments {
        if arguments == &serde_json::Value::Bool(true) {
            return Arguments::Unconstrained;
        }
        let value = &arguments["properties"]["value"];
        assert_eq!(value["type"], "string", "{arguments}");
        assert_eq!(
            arguments["required"],
            serde_json::json!(["value"]),
            "{arguments}"
        );
        if value.get("pattern").is_some() {
            Arguments::Enforced
        } else {
            Arguments::PatternLeftOut
        }
    }

    fn qwen3_coder_preprocessor(schema: StructuralTagSchemaMode) -> Arc<OpenAIPreprocessor> {
        let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct");
        let mut mdc = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
        mdc.runtime_config.structural_tag = Some(StructuralTagConfig {
            schema,
            ..Default::default()
        });
        mdc.runtime_config.tool_call_parser = Some("qwen3_coder".to_string());
        OpenAIPreprocessor::new(mdc).unwrap()
    }

    #[test]
    fn automatic_argument_enforcement_leaves_out_patterns_the_backend_cannot_compile() {
        let lookahead = r"^(?!\s*$).+";
        let auto = qwen3_coder_preprocessor(StructuralTagSchemaMode::Auto);
        let strict = qwen3_coder_preprocessor(StructuralTagSchemaMode::Strict);
        for (case, preprocessor, tool, expected) in [
            (
                "omitted strict",
                &auto,
                record_value_tool(lookahead, None),
                Arguments::PatternLeftOut,
            ),
            (
                "explicit strict",
                &auto,
                record_value_tool(lookahead, Some(true)),
                Arguments::Enforced,
            ),
            (
                "explicit non-strict",
                &auto,
                record_value_tool(lookahead, Some(false)),
                Arguments::Unconstrained,
            ),
            (
                "global strict mode",
                &strict,
                record_value_tool(lookahead, None),
                Arguments::Enforced,
            ),
            (
                "supported pattern",
                &auto,
                record_value_tool("^[a-z]+$", None),
                Arguments::Enforced,
            ),
        ] {
            let arguments = record_value_arguments(preprocessor, tool);
            assert_eq!(arguments_kind(&arguments), expected, "{case}: {arguments}");
        }
    }

    #[test]
    fn unsupported_property_name_patterns_leave_the_arguments_unconstrained() {
        let arguments = record_value_arguments(
            &qwen3_coder_preprocessor(StructuralTagSchemaMode::Auto),
            record_value_tool_with_named_properties(r"^(?!x)"),
        );
        assert_eq!(arguments_kind(&arguments), Arguments::Unconstrained);
    }

    // The opt-out keeps K3 named tags, and K3 splices each pattern into its own
    // grammar instead of a JSON schema.
    #[test]
    fn kimi_k3_named_tag_leaves_out_patterns_the_backend_cannot_compile_when_global_mode_is_off() {
        // Without anchors, which K3 strips when it splices a pattern in.
        for (pattern, kept) in [(r"(?<=a)b", false), ("[a-z]+", true)] {
            let call = record_value_call(&kimi_k3_preprocessor(), record_value_tool(pattern, None))
                .to_string();
            assert!(
                call.contains(r#"argument key=\"value\" type=\"string\""#),
                "{call}"
            );
            let pattern = serde_json::to_string(pattern).unwrap();
            assert_eq!(
                call.contains(pattern.trim_matches('"')),
                kept,
                "{pattern}: {call}"
            );
        }
    }
}
