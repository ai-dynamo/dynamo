// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model-native scripted responses for the runtime-integrated mocker.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use dashmap::DashMap;
use dynamo_mocker::common::protocols::{
    ModelOutputEncoding, ModelOutputGrammar, ModelOutputProfile, ModelOutputPromptFraming,
};
use dynamo_renderer::RenderedSegment;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::protocols::TokenIdType;
use crate::tokenizers::{EncodeSegment, Tokenizer};

const CATALOG_VERSION: u32 = 1;
const HARMONY_COMPATIBILITY_PROBE: &str =
    "<|start|>assistant<|channel|>final<|message|>ok<|return|>";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CatalogFinishReason {
    #[default]
    Stop,
    Length,
}

#[derive(Debug, Clone)]
pub(super) struct CompiledCatalogCase {
    pub token_ids: Vec<TokenIdType>,
    pub finish_reason: CatalogFinishReason,
    pub chunk_size: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogDocument {
    version: u32,
    cases: Vec<CatalogCase>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CatalogCase {
    id: String,
    #[serde(default)]
    response: Option<SemanticResponse>,
    #[serde(default)]
    raw_output: Option<String>,
    #[serde(default)]
    output_token_ids: Option<Vec<TokenIdType>>,
    #[serde(default)]
    finish_reason: CatalogFinishReason,
    #[serde(default = "default_chunk_size")]
    chunk_size: usize,
}

fn default_chunk_size() -> usize {
    1
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SemanticResponse {
    #[serde(default)]
    reasoning: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<SemanticToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SemanticToolCall {
    name: String,
    #[serde(default = "empty_arguments")]
    arguments: Value,
}

fn empty_arguments() -> Value {
    Value::Object(Map::new())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PromptFraming {
    None,
    ReasoningOpen,
    KimiResponseOpen,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CacheKey {
    case_id: String,
    prompt_framing: PromptFraming,
}

enum NativeOutput {
    Text(String),
    Segments(Vec<RenderedSegment>),
    TokenIds(Vec<TokenIdType>),
}

pub(super) struct ResponseCatalog {
    profile: ModelOutputProfile,
    tokenizer: Tokenizer,
    cases: HashMap<String, CatalogCase>,
    compiled: DashMap<CacheKey, Arc<CompiledCatalogCase>>,
}

impl ResponseCatalog {
    pub fn from_path(
        path: &Path,
        profile: ModelOutputProfile,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read response catalog {}", path.display()))?;
        let document: CatalogDocument = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse response catalog {}", path.display()))?;
        let cases = validate_document(document)
            .with_context(|| format!("invalid response catalog {}", path.display()))?;
        if profile.spec().encoding == ModelOutputEncoding::Harmony {
            encode_harmony_compatible(&tokenizer, HARMONY_COMPATIBILITY_PROBE)
                .context("loaded model tokenizer is incompatible with GPT-OSS Harmony")?;
        }

        Ok(Self {
            profile,
            tokenizer,
            cases,
            compiled: DashMap::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.cases.len()
    }

    pub fn resolve(
        &self,
        case_id: &str,
        prompt_token_ids: &[TokenIdType],
    ) -> Result<Arc<CompiledCatalogCase>> {
        let case = self
            .cases
            .get(case_id)
            .with_context(|| format!("response catalog case {case_id:?} was not found"))?;
        let prompt_framing = if case.response.is_some() {
            self.detect_prompt_framing(prompt_token_ids)?
        } else {
            PromptFraming::None
        };
        let cache_key = CacheKey {
            case_id: case_id.to_string(),
            prompt_framing,
        };
        if let Some(compiled) = self.compiled.get(&cache_key) {
            return Ok(Arc::clone(compiled.value()));
        }

        let compiled = Arc::new(self.compile_case(case, prompt_framing)?);
        let cached = self
            .compiled
            .entry(cache_key)
            .or_insert_with(|| Arc::clone(&compiled));
        Ok(Arc::clone(cached.value()))
    }

    fn detect_prompt_framing(&self, prompt_token_ids: &[TokenIdType]) -> Result<PromptFraming> {
        let tail_start = prompt_token_ids.len().saturating_sub(64);
        let decoded = self
            .tokenizer
            .decode(&prompt_token_ids[tail_start..], false)
            .context("failed to decode prompt tail for response framing")?;
        let tail = decoded.as_str().trim_end();
        Ok(match self.profile.spec().prompt_framing {
            ModelOutputPromptFraming::KimiK3Xtml if tail.ends_with("<|open|>think<|sep|>") => {
                PromptFraming::ReasoningOpen
            }
            ModelOutputPromptFraming::KimiK3Xtml if tail.ends_with("<|open|>response<|sep|>") => {
                PromptFraming::KimiResponseOpen
            }
            ModelOutputPromptFraming::ThinkTag if tail.ends_with("<think>") => {
                PromptFraming::ReasoningOpen
            }
            _ => PromptFraming::None,
        })
    }

    fn compile_case(
        &self,
        case: &CatalogCase,
        prompt_framing: PromptFraming,
    ) -> Result<CompiledCatalogCase> {
        let native = if let Some(response) = case.response.as_ref() {
            render_semantic(self.profile, response, prompt_framing)?
        } else if let Some(raw_output) = case.raw_output.as_ref() {
            if self.profile.spec().encoding == ModelOutputEncoding::SegmentedSpecialTokens {
                NativeOutput::Segments(vec![RenderedSegment::new(raw_output.clone(), true)])
            } else {
                NativeOutput::Text(raw_output.clone())
            }
        } else if let Some(token_ids) = case.output_token_ids.as_ref() {
            NativeOutput::TokenIds(token_ids.clone())
        } else {
            unreachable!("catalog validation requires one payload")
        };

        let token_ids = match (native, self.profile.spec().encoding) {
            (NativeOutput::Text(text), ModelOutputEncoding::Harmony) => {
                encode_harmony_compatible(&self.tokenizer, &text)?
            }
            (NativeOutput::Text(text), ModelOutputEncoding::TokenizerText) => self
                .tokenizer
                .encode(&text)
                .context("failed to encode scripted response")?
                .token_ids()
                .to_vec(),
            (NativeOutput::Segments(segments), ModelOutputEncoding::TokenizerText) => {
                encode_text_segments(&self.tokenizer, self.profile, prompt_framing, segments)?
            }
            (NativeOutput::Segments(segments), ModelOutputEncoding::SegmentedSpecialTokens) => {
                encode_special_segments(&self.tokenizer, self.profile, prompt_framing, segments)?
            }
            (NativeOutput::TokenIds(token_ids), _) => token_ids,
            _ => unreachable!("profile registry and native renderer encoding disagree"),
        };
        ensure!(
            !token_ids.is_empty(),
            "response catalog case {:?} compiled to no tokens",
            case.id
        );

        Ok(CompiledCatalogCase {
            token_ids,
            finish_reason: case.finish_reason,
            chunk_size: case.chunk_size,
        })
    }
}

fn framing_prefix(
    profile: ModelOutputProfile,
    prompt_framing: PromptFraming,
) -> Vec<RenderedSegment> {
    match (profile.spec().prompt_framing, prompt_framing) {
        (ModelOutputPromptFraming::ThinkTag, PromptFraming::ReasoningOpen) => {
            vec![RenderedSegment::new("<think>", true)]
        }
        (ModelOutputPromptFraming::KimiK3Xtml, PromptFraming::ReasoningOpen) => vec![
            RenderedSegment::new("<|open|>", true),
            RenderedSegment::new("think", false),
            RenderedSegment::new("<|sep|>", true),
        ],
        (ModelOutputPromptFraming::KimiK3Xtml, PromptFraming::KimiResponseOpen) => vec![
            RenderedSegment::new("<|open|>", true),
            RenderedSegment::new("response", false),
            RenderedSegment::new("<|sep|>", true),
        ],
        _ => Vec::new(),
    }
}

fn strip_framing_prefix(
    mut full_token_ids: Vec<TokenIdType>,
    prefix_token_ids: &[TokenIdType],
) -> Result<Vec<TokenIdType>> {
    ensure!(
        full_token_ids.starts_with(prefix_token_ids),
        "prompt-framing prefix does not align with scripted response tokenization"
    );
    full_token_ids.drain(..prefix_token_ids.len());
    Ok(full_token_ids)
}

fn encode_text_segments(
    tokenizer: &Tokenizer,
    profile: ModelOutputProfile,
    prompt_framing: PromptFraming,
    segments: Vec<RenderedSegment>,
) -> Result<Vec<TokenIdType>> {
    let prefix: String = framing_prefix(profile, prompt_framing)
        .into_iter()
        .map(|segment| segment.text)
        .collect();
    let text: String = segments.into_iter().map(|segment| segment.text).collect();
    if prefix.is_empty() {
        return Ok(tokenizer
            .encode(&text)
            .context("failed to encode scripted response")?
            .token_ids()
            .to_vec());
    }

    let prefix_token_ids = tokenizer
        .encode(&prefix)
        .context("failed to encode scripted response framing prefix")?
        .token_ids()
        .to_vec();
    let full_token_ids = tokenizer
        .encode(&(prefix + &text))
        .context("failed to encode prompt-framed scripted response")?
        .token_ids()
        .to_vec();
    strip_framing_prefix(full_token_ids, &prefix_token_ids)
}

fn encode_special_segments(
    tokenizer: &Tokenizer,
    profile: ModelOutputProfile,
    prompt_framing: PromptFraming,
    segments: Vec<RenderedSegment>,
) -> Result<Vec<TokenIdType>> {
    let prefix = framing_prefix(profile, prompt_framing);
    let mut full_segments = prefix.clone();
    full_segments.extend(segments);
    let full_token_ids = tokenizer
        .encode_segments(&as_encode_segments(&full_segments))
        .context("failed to encode segmented scripted response")?
        .token_ids()
        .to_vec();
    if prefix.is_empty() {
        return Ok(full_token_ids);
    }

    let prefix_token_ids = tokenizer
        .encode_segments(&as_encode_segments(&prefix))
        .context("failed to encode segmented response framing prefix")?
        .token_ids()
        .to_vec();
    strip_framing_prefix(full_token_ids, &prefix_token_ids)
}

fn as_encode_segments(segments: &[RenderedSegment]) -> Vec<EncodeSegment<'_>> {
    segments
        .iter()
        .map(|segment| EncodeSegment::new(&segment.text, segment.allow_special))
        .collect()
}

fn encode_harmony_compatible(tokenizer: &Tokenizer, text: &str) -> Result<Vec<TokenIdType>> {
    let token_ids = dynamo_parsers_v2::encode_harmony(text)
        .context("failed to encode GPT-OSS Harmony response")?;
    let harmony_decoded = dynamo_parsers_v2::decode_harmony(&token_ids)
        .context("failed to validate GPT-OSS Harmony response tokens")?;
    ensure!(
        harmony_decoded == text,
        "Harmony encoder did not round-trip the scripted GPT-OSS response"
    );
    let decoded = tokenizer
        .decode(&token_ids, false)
        .context("model tokenizer could not decode Harmony token IDs")?;
    ensure!(
        decoded.as_str() == harmony_decoded,
        "model tokenizer is incompatible with GPT-OSS Harmony encoding"
    );
    Ok(token_ids)
}

fn validate_document(document: CatalogDocument) -> Result<HashMap<String, CatalogCase>> {
    ensure!(
        document.version == CATALOG_VERSION,
        "catalog uses version {}, expected {}",
        document.version,
        CATALOG_VERSION
    );
    ensure!(!document.cases.is_empty(), "catalog has no cases");

    let mut cases = HashMap::with_capacity(document.cases.len());
    for case in document.cases {
        case.validate()
            .with_context(|| format!("invalid case {:?}", case.id))?;
        let case_id = case.id.clone();
        ensure!(
            cases.insert(case_id.clone(), case).is_none(),
            "catalog contains duplicate case id {case_id:?}"
        );
    }
    Ok(cases)
}

impl CatalogCase {
    fn validate(&self) -> Result<()> {
        ensure!(!self.id.trim().is_empty(), "case id must not be empty");
        ensure!(self.chunk_size > 0, "chunk_size must be greater than zero");
        let payload_count = self.response.is_some() as usize
            + self.raw_output.is_some() as usize
            + self.output_token_ids.is_some() as usize;
        ensure!(
            payload_count == 1,
            "exactly one of response, raw_output, or output_token_ids must be set"
        );
        if let Some(raw_output) = self.raw_output.as_ref() {
            ensure!(!raw_output.is_empty(), "raw_output must not be empty");
        }
        if let Some(token_ids) = self.output_token_ids.as_ref() {
            ensure!(!token_ids.is_empty(), "output_token_ids must not be empty");
        }
        if let Some(response) = self.response.as_ref() {
            response.validate()?;
        }
        Ok(())
    }
}

impl SemanticResponse {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.reasoning
                .as_ref()
                .is_some_and(|value| !value.is_empty())
                || self.content.as_ref().is_some_and(|value| !value.is_empty())
                || !self.tool_calls.is_empty(),
            "semantic response must contain reasoning, content, or tool_calls"
        );
        for call in &self.tool_calls {
            ensure!(
                !call.name.trim().is_empty(),
                "tool call name must not be empty"
            );
            ensure!(
                call.arguments.is_object(),
                "tool call {:?} arguments must be a JSON object",
                call.name
            );
        }
        Ok(())
    }
}

fn render_semantic(
    profile: ModelOutputProfile,
    response: &SemanticResponse,
    prompt_framing: PromptFraming,
) -> Result<NativeOutput> {
    Ok(match profile.spec().grammar {
        ModelOutputGrammar::KimiK3Xtml => {
            NativeOutput::Segments(render_kimi_k3(response, prompt_framing)?)
        }
        ModelOutputGrammar::DeepseekV4Dsml => NativeOutput::Segments(render_deepseek_v4(
            response,
            prompt_framing == PromptFraming::ReasoningOpen,
        )?),
        ModelOutputGrammar::Qwen35Xml => NativeOutput::Segments(render_qwen35(
            response,
            prompt_framing == PromptFraming::ReasoningOpen,
        )?),
        ModelOutputGrammar::Glm52Xml => NativeOutput::Segments(render_glm52(
            response,
            prompt_framing == PromptFraming::ReasoningOpen,
        )?),
        ModelOutputGrammar::GptOssHarmony => NativeOutput::Text(render_gpt_oss(response)?),
    })
}

fn native_control(output: &mut Vec<RenderedSegment>, text: impl Into<String>) {
    output.push(RenderedSegment::new(text, true));
}

fn native_text(output: &mut Vec<RenderedSegment>, text: impl Into<String>) {
    let text = text.into();
    if !text.is_empty() {
        output.push(RenderedSegment::new(text, false));
    }
}

fn render_think_prefix(
    output: &mut Vec<RenderedSegment>,
    reasoning: Option<&str>,
    prompt_open: bool,
) {
    if reasoning.is_none() && !prompt_open {
        return;
    }
    if !prompt_open {
        native_control(output, "<think>");
    }
    if let Some(reasoning) = reasoning {
        native_text(output, reasoning);
    }
    native_control(output, "</think>");
}

fn arguments(call: &SemanticToolCall) -> Result<&Map<String, Value>> {
    call.arguments
        .as_object()
        .with_context(|| format!("tool call {:?} arguments must be an object", call.name))
}

fn scalar_or_json(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        _ => serde_json::to_string(value).context("failed to serialize tool argument"),
    }
}

fn validate_markup_name(name: &str, field: &str) -> Result<()> {
    ensure!(
        !name.contains('"') && !name.contains('<') && !name.contains('>'),
        "{field} {name:?} contains an unsupported markup delimiter"
    );
    Ok(())
}

fn render_deepseek_v4(
    response: &SemanticResponse,
    prompt_open: bool,
) -> Result<Vec<RenderedSegment>> {
    let mut output = Vec::new();
    render_think_prefix(&mut output, response.reasoning.as_deref(), prompt_open);
    if let Some(content) = response.content.as_ref() {
        native_text(&mut output, content);
    }
    if response.tool_calls.is_empty() {
        return Ok(output);
    }
    native_control(&mut output, "<｜DSML｜tool_calls>\n");
    for call in &response.tool_calls {
        validate_markup_name(&call.name, "tool name")?;
        native_control(&mut output, "<｜DSML｜invoke name=\"");
        native_text(&mut output, &call.name);
        native_control(&mut output, "\">\n");
        for (name, value) in arguments(call)? {
            validate_markup_name(name, "argument name")?;
            let is_string = value.is_string();
            let value = scalar_or_json(value)?;
            native_control(&mut output, "<｜DSML｜parameter name=\"");
            native_text(&mut output, name);
            native_control(&mut output, format!("\" string=\"{is_string}\">"));
            native_text(&mut output, value);
            native_control(&mut output, "</｜DSML｜parameter>\n");
        }
        native_control(&mut output, "</｜DSML｜invoke>\n");
    }
    native_control(&mut output, "</｜DSML｜tool_calls>");
    Ok(output)
}

fn render_qwen35(response: &SemanticResponse, prompt_open: bool) -> Result<Vec<RenderedSegment>> {
    let mut output = Vec::new();
    render_think_prefix(&mut output, response.reasoning.as_deref(), prompt_open);
    if let Some(content) = response.content.as_ref() {
        native_text(&mut output, content);
    }
    for call in &response.tool_calls {
        validate_markup_name(&call.name, "tool name")?;
        native_control(&mut output, "<tool_call>\n<function=");
        native_text(&mut output, &call.name);
        native_control(&mut output, ">\n");
        for (name, value) in arguments(call)? {
            validate_markup_name(name, "argument name")?;
            let value = scalar_or_json(value)?;
            native_control(&mut output, "<parameter=");
            native_text(&mut output, name);
            native_control(&mut output, ">\n");
            native_text(&mut output, value);
            native_control(&mut output, "\n</parameter>\n");
        }
        native_control(&mut output, "</function>\n</tool_call>");
    }
    Ok(output)
}

fn render_glm52(response: &SemanticResponse, prompt_open: bool) -> Result<Vec<RenderedSegment>> {
    let mut output = Vec::new();
    render_think_prefix(&mut output, response.reasoning.as_deref(), prompt_open);
    if let Some(content) = response.content.as_ref() {
        native_text(&mut output, content);
    }
    for call in &response.tool_calls {
        validate_markup_name(&call.name, "tool name")?;
        native_control(&mut output, "<tool_call>");
        native_text(&mut output, &call.name);
        for (name, value) in arguments(call)? {
            validate_markup_name(name, "argument name")?;
            let value = scalar_or_json(value)?;
            native_control(&mut output, "<arg_key>");
            native_text(&mut output, name);
            native_control(&mut output, "</arg_key><arg_value>");
            native_text(&mut output, value);
            native_control(&mut output, "</arg_value>");
        }
        native_control(&mut output, "</tool_call>");
    }
    Ok(output)
}

fn render_gpt_oss(response: &SemanticResponse) -> Result<String> {
    let mut output = String::new();
    let mut needs_assistant_start = false;
    if let Some(reasoning) = response.reasoning.as_ref() {
        write!(output, "<|channel|>analysis<|message|>{reasoning}<|end|>")?;
        needs_assistant_start = true;
    }
    for call in &response.tool_calls {
        validate_markup_name(&call.name, "tool name")?;
        let arguments = serde_json::to_string(arguments(call)?)?;
        if needs_assistant_start {
            output.push_str("<|start|>assistant");
        }
        write!(
            output,
            "<|channel|>commentary to=functions.{} <|constrain|>json<|message|>{arguments}<|call|>",
            call.name
        )?;
        needs_assistant_start = true;
    }
    if let Some(content) = response.content.as_ref() {
        if needs_assistant_start {
            output.push_str("<|start|>assistant");
        }
        write!(output, "<|channel|>final<|message|>{content}<|return|>")?;
    }
    Ok(output)
}

fn kimi_control(segments: &mut Vec<RenderedSegment>, text: &str) {
    segments.push(RenderedSegment::new(text, true));
}

fn kimi_text(segments: &mut Vec<RenderedSegment>, text: impl Into<String>) {
    let text = text.into();
    if !text.is_empty() {
        segments.push(RenderedSegment::new(text, false));
    }
}

fn kimi_open(segments: &mut Vec<RenderedSegment>, tag: &str, attrs: &[(&str, String)]) {
    kimi_control(segments, "<|open|>");
    kimi_text(segments, tag);
    for (name, value) in attrs {
        kimi_text(segments, format!(" {name}=\""));
        kimi_text(segments, value.replace('&', "&amp;").replace('"', "&quot;"));
        kimi_text(segments, "\"");
    }
    kimi_control(segments, "<|sep|>");
}

fn kimi_close(segments: &mut Vec<RenderedSegment>, tag: &str) {
    kimi_control(segments, "<|close|>");
    kimi_text(segments, tag);
    kimi_control(segments, "<|sep|>");
}

fn kimi_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn render_kimi_k3(
    response: &SemanticResponse,
    prompt_framing: PromptFraming,
) -> Result<Vec<RenderedSegment>> {
    // Grammar source is recorded in MODEL_OUTPUT_PROFILE_SPECS.
    ensure!(
        prompt_framing != PromptFraming::KimiResponseOpen || response.reasoning.is_none(),
        "Kimi K3 prompt already opened the response channel; semantic reasoning cannot precede it"
    );
    let mut segments = Vec::new();
    if prompt_framing == PromptFraming::ReasoningOpen || response.reasoning.is_some() {
        if prompt_framing != PromptFraming::ReasoningOpen {
            kimi_open(&mut segments, "think", &[]);
        }
        if let Some(reasoning) = response.reasoning.as_ref() {
            kimi_text(&mut segments, reasoning);
        }
        kimi_close(&mut segments, "think");
    }

    if prompt_framing != PromptFraming::KimiResponseOpen {
        kimi_open(&mut segments, "response", &[]);
    }
    if let Some(content) = response.content.as_ref() {
        kimi_text(&mut segments, content);
    }
    kimi_close(&mut segments, "response");

    if !response.tool_calls.is_empty() {
        kimi_open(&mut segments, "tools", &[]);
        for (index, call) in response.tool_calls.iter().enumerate() {
            validate_markup_name(&call.name, "tool name")?;
            kimi_open(
                &mut segments,
                "call",
                &[
                    ("tool", call.name.clone()),
                    ("index", (index + 1).to_string()),
                ],
            );
            for (name, value) in arguments(call)? {
                kimi_open(
                    &mut segments,
                    "argument",
                    &[
                        ("key", name.clone()),
                        ("type", kimi_type(value).to_string()),
                    ],
                );
                kimi_text(&mut segments, scalar_or_json(value)?);
                kimi_close(&mut segments, "argument");
            }
            kimi_close(&mut segments, "call");
        }
        kimi_close(&mut segments, "tools");
    }
    kimi_close(&mut segments, "message");
    kimi_control(&mut segments, "<|end_of_msg|>");
    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_parsers::reasoning::{ReasoningParser, ReasoningParserType};
    use dynamo_parsers::tool_calling::ToolDefinition;
    use serde_json::json;

    fn response() -> SemanticResponse {
        SemanticResponse {
            reasoning: Some("check weather".to_string()),
            content: None,
            tool_calls: vec![SemanticToolCall {
                name: "get_weather".to_string(),
                arguments: json!({"city": "Seattle", "days": 2}),
            }],
        }
    }

    fn native_text(output: NativeOutput) -> String {
        match output {
            NativeOutput::Text(text) => text,
            NativeOutput::Segments(segments) => {
                segments.into_iter().map(|segment| segment.text).collect()
            }
            NativeOutput::TokenIds(_) => panic!("semantic response must render text"),
        }
    }

    fn validate_catalog(value: Value) -> Result<HashMap<String, CatalogCase>> {
        let document: CatalogDocument = serde_json::from_value(value)?;
        validate_document(document)
    }

    #[test]
    fn catalog_defaults_and_payload_union_are_strict() {
        let cases = validate_catalog(json!({
            "version": 1,
            "cases": [{"id": "tokens", "output_token_ids": [1, 2, 3]}]
        }))
        .unwrap();
        let case = &cases["tokens"];
        assert_eq!(case.finish_reason, CatalogFinishReason::Stop);
        assert_eq!(case.chunk_size, 1);

        for invalid in [
            json!({"version": 1, "cases": [{"id": "missing"}]}),
            json!({
                "version": 1,
                "cases": [{
                    "id": "multiple",
                    "raw_output": "hello",
                    "output_token_ids": [1]
                }]
            }),
            json!({
                "version": 1,
                "cases": [{"id": "chunk", "raw_output": "hello", "chunk_size": 0}]
            }),
        ] {
            assert!(validate_catalog(invalid).is_err());
        }
    }

    #[test]
    fn catalog_rejects_version_duplicates_and_invalid_semantics() {
        let duplicate = json!({
            "version": 1,
            "cases": [
                {"id": "same", "raw_output": "one"},
                {"id": "same", "raw_output": "two"}
            ]
        });
        assert!(
            validate_catalog(duplicate)
                .unwrap_err()
                .to_string()
                .contains("duplicate")
        );
        assert!(
            validate_catalog(json!({
                "version": 2,
                "cases": [{"id": "case", "raw_output": "text"}]
            }))
            .unwrap_err()
            .to_string()
            .contains("expected 1")
        );
        assert!(
            validate_catalog(json!({
                "version": 1,
                "cases": [{
                    "id": "bad-args",
                    "response": {
                        "tool_calls": [{"name": "tool", "arguments": [1, 2]}]
                    }
                }]
            }))
            .is_err()
        );
    }

    #[test]
    fn profile_registry_maps_runtime_parsers() {
        let profiles = [
            (ModelOutputProfile::KimiK3, "kimi_k3", "kimi_k3"),
            (ModelOutputProfile::DeepseekV4, "deepseek_v4", "deepseek_v4"),
            (ModelOutputProfile::Qwen35, "qwen3_coder", "qwen3"),
            (ModelOutputProfile::Glm52, "glm47", "glm45"),
            (ModelOutputProfile::GptOss, "harmony", "gpt_oss"),
        ];
        for (profile, tool_parser, reasoning_parser) in profiles {
            assert_eq!(profile.tool_call_parser(), tool_parser);
            assert_eq!(profile.reasoning_parser(), reasoning_parser);
        }
    }

    #[test]
    fn semantic_profiles_render_native_grammars() {
        let response = response();
        let expected = [
            (
                ModelOutputProfile::KimiK3,
                "<|open|>think<|sep|>check weather<|close|>think<|sep|><|open|>response<|sep|><|close|>response<|sep|><|open|>tools<|sep|><|open|>call tool=\"get_weather\" index=\"1\"<|sep|><|open|>argument key=\"city\" type=\"string\"<|sep|>Seattle<|close|>argument<|sep|><|open|>argument key=\"days\" type=\"number\"<|sep|>2<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|><|end_of_msg|>",
            ),
            (
                ModelOutputProfile::DeepseekV4,
                "<think>check weather</think><｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Seattle</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">2</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            ),
            (
                ModelOutputProfile::Qwen35,
                "<think>check weather</think><tool_call>\n<function=get_weather>\n<parameter=city>\nSeattle\n</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                ModelOutputProfile::Glm52,
                "<think>check weather</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Seattle</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>",
            ),
            (
                ModelOutputProfile::GptOss,
                "<|channel|>analysis<|message|>check weather<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Seattle\",\"days\":2}<|call|>",
            ),
        ];
        for (profile, expected) in expected {
            let actual =
                native_text(render_semantic(profile, &response, PromptFraming::None).unwrap());
            assert_eq!(actual, expected, "profile={profile}");
        }
    }

    #[test]
    fn semantic_rendering_preserves_special_token_trust_boundaries() {
        let response = SemanticResponse {
            reasoning: Some("literal <think> text".to_string()),
            content: Some("literal <tool_call> and <|open|> text".to_string()),
            tool_calls: vec![SemanticToolCall {
                name: "get_weather".to_string(),
                arguments: json!({"city": "<|open|>Seattle"}),
            }],
        };

        for profile in [
            ModelOutputProfile::KimiK3,
            ModelOutputProfile::DeepseekV4,
            ModelOutputProfile::Qwen35,
            ModelOutputProfile::Glm52,
        ] {
            let NativeOutput::Segments(segments) =
                render_semantic(profile, &response, PromptFraming::None).unwrap()
            else {
                panic!("profile {profile} must use segmented encoding");
            };
            assert!(segments.iter().any(|segment| segment.allow_special));
            assert!(segments.iter().any(|segment| {
                !segment.allow_special
                    && (segment.text.contains("literal <tool_call>")
                        || segment.text.contains("<|open|>Seattle"))
            }));
        }
    }

    #[test]
    fn prompt_open_reasoning_is_not_duplicated() {
        let response = response();
        let deepseek = native_text(
            render_semantic(
                ModelOutputProfile::DeepseekV4,
                &response,
                PromptFraming::ReasoningOpen,
            )
            .unwrap(),
        );
        assert!(deepseek.starts_with("check weather</think>"));
        assert!(!deepseek.starts_with("<think>"));

        let content_only = SemanticResponse {
            reasoning: None,
            content: Some("done".to_string()),
            tool_calls: Vec::new(),
        };
        let kimi = native_text(
            render_semantic(
                ModelOutputProfile::KimiK3,
                &content_only,
                PromptFraming::KimiResponseOpen,
            )
            .unwrap(),
        );
        assert!(kimi.starts_with("done<|close|>response<|sep|>"));
        assert_eq!(kimi.matches("<|open|>response<|sep|>").count(), 0);
        assert!(
            render_semantic(
                ModelOutputProfile::KimiK3,
                &response,
                PromptFraming::KimiResponseOpen,
            )
            .is_err()
        );
    }

    #[test]
    fn prompt_open_encoding_preserves_the_first_reasoning_token() {
        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
        ))
        .unwrap();
        let catalog_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            catalog_file.path(),
            json!({
                "version": 1,
                "cases": [{
                    "id": "reasoning",
                    "response": {"reasoning": "I should call a tool."}
                }]
            })
            .to_string(),
        )
        .unwrap();
        let catalog = ResponseCatalog::from_path(
            catalog_file.path(),
            ModelOutputProfile::Qwen35,
            tokenizer.clone(),
        )
        .unwrap();
        let prompt_token_ids = tokenizer.encode("<think>").unwrap().token_ids().to_vec();

        let compiled = catalog.resolve("reasoning", &prompt_token_ids).unwrap();
        let decoded = tokenizer.decode(&compiled.token_ids, false).unwrap();

        assert_eq!(decoded.as_str(), "I should call a tool.</think>");
    }

    #[tokio::test]
    async fn semantic_tool_calls_round_trip_through_registered_parsers() {
        let response = response();
        let tools = [ToolDefinition {
            name: "get_weather".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                }
            })),
            strict: None,
        }];
        for profile in [
            ModelOutputProfile::KimiK3,
            ModelOutputProfile::DeepseekV4,
            ModelOutputProfile::Qwen35,
            ModelOutputProfile::Glm52,
            ModelOutputProfile::GptOss,
        ] {
            let text =
                native_text(render_semantic(profile, &response, PromptFraming::None).unwrap());
            let mut reasoning_parser =
                ReasoningParserType::get_reasoning_parser_from_name(profile.reasoning_parser());
            let parsed_reasoning = reasoning_parser.detect_and_parse_reasoning(&text, &[]);
            assert_eq!(parsed_reasoning.reasoning_text, "check weather");
            let (calls, content) = dynamo_parsers::tool_calling::detect_and_parse_tool_call(
                &parsed_reasoning.normal_text,
                Some(profile.tool_call_parser()),
                Some(&tools),
            )
            .await
            .unwrap();
            assert_eq!(calls.len(), 1, "profile={profile}, text={text:?}");
            assert_eq!(calls[0].function.name, "get_weather");
            let arguments: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
            assert_eq!(arguments, json!({"city": "Seattle", "days": 2}));
            assert!(
                content.as_deref().unwrap_or_default().is_empty(),
                "profile={profile}, content={content:?}"
            );
        }
    }

    #[tokio::test]
    async fn semantic_profiles_round_trip_parallel_nested_unicode_and_empty_arguments() {
        let response = SemanticResponse {
            reasoning: None,
            content: None,
            tool_calls: vec![
                SemanticToolCall {
                    name: "lookup".to_string(),
                    arguments: json!({
                        "query": "東京",
                        "nested": {"ok": true},
                        "items": [1, "二"]
                    }),
                },
                SemanticToolCall {
                    name: "ping".to_string(),
                    arguments: json!({}),
                },
            ],
        };
        let tools = [
            ToolDefinition {
                name: "lookup".to_string(),
                parameters: Some(json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "nested": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}}
                        },
                        "items": {"type": "array", "items": {}}
                    }
                })),
                strict: None,
            },
            ToolDefinition {
                name: "ping".to_string(),
                parameters: Some(json!({"type": "object", "properties": {}})),
                strict: None,
            },
        ];

        for profile in [
            ModelOutputProfile::KimiK3,
            ModelOutputProfile::DeepseekV4,
            ModelOutputProfile::Qwen35,
            ModelOutputProfile::Glm52,
            ModelOutputProfile::GptOss,
        ] {
            let text =
                native_text(render_semantic(profile, &response, PromptFraming::None).unwrap());
            let (calls, content) = dynamo_parsers::tool_calling::detect_and_parse_tool_call(
                &text,
                Some(profile.tool_call_parser()),
                Some(&tools),
            )
            .await
            .unwrap();
            assert_eq!(calls.len(), 2, "profile={profile}, text={text:?}");
            assert_eq!(calls[0].function.name, "lookup");
            assert_eq!(calls[1].function.name, "ping");
            assert_eq!(
                serde_json::from_str::<Value>(&calls[0].function.arguments).unwrap(),
                json!({
                    "query": "東京",
                    "nested": {"ok": true},
                    "items": [1, "二"]
                })
            );
            assert_eq!(
                serde_json::from_str::<Value>(&calls[1].function.arguments).unwrap(),
                json!({})
            );
            assert!(content.as_deref().unwrap_or_default().is_empty());
        }
    }

    #[tokio::test]
    async fn semantic_direct_content_round_trips_without_native_markers() {
        let response = SemanticResponse {
            reasoning: None,
            content: Some("héllo 世界".to_string()),
            tool_calls: Vec::new(),
        };

        for profile in [
            ModelOutputProfile::KimiK3,
            ModelOutputProfile::DeepseekV4,
            ModelOutputProfile::Qwen35,
            ModelOutputProfile::Glm52,
            ModelOutputProfile::GptOss,
        ] {
            let text =
                native_text(render_semantic(profile, &response, PromptFraming::None).unwrap());
            let mut reasoning_parser =
                ReasoningParserType::get_reasoning_parser_from_name(profile.reasoning_parser());
            let parsed_reasoning = reasoning_parser.detect_and_parse_reasoning(&text, &[]);
            assert!(parsed_reasoning.reasoning_text.is_empty());
            let content = if profile == ModelOutputProfile::KimiK3 {
                let (calls, content) = dynamo_parsers::tool_calling::detect_and_parse_tool_call(
                    &parsed_reasoning.normal_text,
                    Some(profile.tool_call_parser()),
                    None,
                )
                .await
                .unwrap();
                assert!(calls.is_empty());
                content.unwrap_or_default()
            } else {
                parsed_reasoning.normal_text
            };
            assert_eq!(content, "héllo 世界", "profile={profile}");
        }
    }

    #[test]
    fn harmony_rejects_an_incompatible_model_tokenizer() {
        let tokenizer = Tokenizer::from_file(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/sample-models/TinyLlama_v1.1/tokenizer.json"
        ))
        .unwrap();
        assert!(encode_harmony_compatible(&tokenizer, HARMONY_COMPATIBILITY_PROBE).is_err());
    }
}
