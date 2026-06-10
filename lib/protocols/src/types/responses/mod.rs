// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Dynamo owns the Responses-API input-side type chain. Upstream async-openai
// is the source for everything else (output-side types, streaming events,
// individual tool-call payloads, etc.).
//
// The input chain is owned because upstream marks fields as required that
// real-world clients (OpenAI Agents SDK, Codex, etc.) routinely omit when
// round-tripping a prior assistant turn as input:
//   - `OutputMessage.id` / `.status` — omitted when echoing a previous output
//   - `OutputTextContent.annotations` — omitted when the part carried none
// Upstream is slow to relax these (see e.g. 64bit/async-openai#535 for the
// sibling `ReasoningItem.id` fix, still open at time of writing); OpenAI's own
// hosted API accepts the relaxed shapes on input regardless.
//
// This mirrors the pattern in `crate::types::chat` where Dynamo owns the
// request types it needs to extend or relax while re-exporting the rest of
// upstream's type library verbatim.
//
// Naming: the relaxed assistant-input message is `InputOutputMessage` (and
// `InputOutputMessageContent` / `InputOutputTextContent` for its content
// parts) to avoid colliding with upstream's `OutputMessage`, which remains the
// canonical type for *output-side* response construction (`OutputItem`,
// `Response.output`). `MessageItem`, `Item`, `InputItem`, `InputParam`, and
// `CreateResponse` are input-only and shadow upstream's same-named types
// without conflict.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// Re-export all upstream response types (shared structures like ResponseUsage,
// tool-call item types, streaming events, etc.). The types we own below
// shadow their upstream counterparts where no dual-side conflict exists.
pub use async_openai::types::responses::*;

// Re-export upstream's pre-shadow `InputContent` under an explicit alias.
// Needed because `FunctionCallOutput::Content` and `EasyInputContent::ContentList`
// are non-owned upstream types that carry upstream's original `InputContent`
// inline, so downstream consumers occasionally need to name it alongside the
// Dynamo-owned shadow defined further down this module.
pub use async_openai::types::responses::InputContent as UpstreamInputContent;

// Re-export from parent module for backward compat.
pub use crate::types::ImageDetail;
pub use crate::types::ReasoningEffort;
pub use crate::types::ResponseFormatJsonSchema;

// Backward-compatible type aliases for Dynamo consumer code migration.
pub type Input = InputParam;
pub type PromptConfig = Prompt;
pub type TextConfig = ResponseTextParam;
pub type TextResponseFormat = TextResponseFormatConfiguration;

/// Stream of response events.
pub type ResponseStream = std::pin::Pin<
    Box<dyn futures::Stream<Item = Result<ResponseStreamEvent, crate::error::OpenAIError>> + Send>,
>;

/// Fields on upstream `Response` that the OpenResponses spec requires as
/// `T | null` but async-openai declares as `Option<T>` with
/// `skip_serializing_if = Option::is_none` — meaning `None` disappears from
/// the wire shape, where the spec wants an explicit `null`.
///
/// Colocated here (next to the upstream `Response` re-export) rather than in
/// `lib/llm/src/protocols/openai/responses/mod.rs` so that when upstream's
/// `Response` gains a new nullable-required field, the reviewer editing this
/// module is looking directly at the authoritative list. Keep sorted
/// alphabetically; entries must match serde field names on `Response` exactly.
///
/// Any field we unconditionally populate ourselves during response
/// construction (e.g. `metadata`, `parallel_tool_calls`, `temperature`,
/// `text`, `tool_choice`, `tools`, `top_p`, `top_logprobs`, `truncation`,
/// `service_tier`, `background`) is deliberately absent — it's always
/// present on the wire, so listing it here would be noise.
pub const SPEC_NULLABLE_REQUIRED_RESPONSE_FIELDS: &[&str] = &[
    "billing",
    "completed_at",
    "conversation",
    "error",
    "incomplete_details",
    "instructions",
    "max_output_tokens",
    "max_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "prompt_cache_retention",
    "reasoning",
    "safety_identifier",
    "usage",
];

// ---------------------------------------------------------------------------
// Input-side assistant message (relaxed vs upstream OutputMessage)
// ---------------------------------------------------------------------------

/// Deserialize `null` or a missing field as the default empty `Vec`. Plain
/// `#[serde(default)]` only fires when the field is absent; explicit `null`
/// would otherwise fail `Vec::deserialize`. Clients (notably some Agents SDK
/// variants) have been observed to send `"annotations": null`, so treat
/// omission and explicit null the same.
fn deserialize_null_as_empty_vec<'de, T, D>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    T: Deserialize<'de>,
    D: serde::Deserializer<'de>,
{
    Option::<Vec<T>>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Deserialize `null` or a missing field as `T::default()`. Scalar counterpart
/// to `deserialize_null_as_empty_vec` — plain `#[serde(default)]` rejects
/// explicit `null` because serde tries to deserialize the null into `T` and
/// fails. Real clients emit `null` for unset enum-ish fields (e.g. OpenAI
/// Agents SDK sending `"detail": null` on `input_image` parts).
fn deserialize_null_as_default<'de, T, D>(deserializer: D) -> Result<T, D::Error>
where
    T: Deserialize<'de> + Default,
    D: serde::Deserializer<'de>,
{
    Option::<T>::deserialize(deserializer).map(Option::unwrap_or_default)
}

/// Relaxed counterpart to upstream `OutputTextContent` for input-side content.
/// `annotations` tolerates both missing and explicit `null`; upstream requires
/// it to be a present non-null array.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputOutputTextContent {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_vec")]
    pub annotations: Vec<Annotation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<LogProb>>,
    pub text: String,
}

/// Content parts of a prior assistant message presented as input.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputOutputMessageContent {
    OutputText(InputOutputTextContent),
    Refusal(RefusalContent),
}

/// An assistant message echoed back as input for a subsequent turn. Relaxed
/// compared to upstream `OutputMessage`: `id`, `status`, and `content` are all
/// optional. Some clients send a bare assistant shell (`{"type":"message",
/// "role":"assistant"}`) with no `content` at all, usually on pure tool-call
/// turns; treat absent `content` as an empty vec, same way we treat a missing
/// `id`/`status`.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputOutputMessage {
    #[serde(default, deserialize_with = "deserialize_null_as_empty_vec")]
    pub content: Vec<InputOutputMessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub role: AssistantRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<MessagePhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

// ---------------------------------------------------------------------------
// Input-side image / content / message (shadow upstream, relaxed shapes)
// ---------------------------------------------------------------------------

/// Relaxed counterpart to upstream `InputImageContent`. `detail` defaults to
/// `ImageDetail::Auto` when the client omits it — OpenAI's hosted API and the
/// OpenResponses spec both accept this shape, but upstream's struct marks
/// `detail` as required.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct InputImageContent {
    #[serde(default, deserialize_with = "deserialize_null_as_default")]
    pub detail: ImageDetail,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
}

/// Parts of an input message: text, image, or file. Mirrors upstream
/// `InputContent` but routes `InputImage` through the Dynamo-owned relaxed
/// `InputImageContent` above.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContent {
    InputText(InputTextContent),
    InputImage(InputImageContent),
    InputFile(InputFileContent),
}

/// User / system / developer input message. Shadows upstream `InputMessage`
/// so we can route through the Dynamo-owned `InputContent` chain.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct InputMessage {
    pub content: Vec<InputContent>,
    pub role: InputRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<OutputStatus>,
}

/// Content for `EasyInputMessage`. Shadows upstream's same-named enum so the
/// `ContentList` arm carries Dynamo's relaxed `InputContent` (with optional
/// `detail` on `InputImageContent`) instead of upstream's strict variant.
///
/// Without this shadow, the `InputItem::EasyMessage` fallback in the untagged
/// `InputItem` enum is the only path that still routes through upstream's
/// strict types — so any spec-compliant client that omits `type: "message"`
/// on a multimodal message (the documented default) fails with
/// "data did not match any variant of untagged enum InputItem". See issue
/// #9468.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum EasyInputContent {
    /// Plain-text content. Tried first so `"content": "hi"` short-circuits.
    Text(String),
    /// Structured content list (text/image/file parts).
    ContentList(Vec<InputContent>),
}

impl Default for EasyInputContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// A simplified message input — the spec-default shape when a client omits the
/// `type` discriminator. Shadows upstream `EasyInputMessage` so the `content`
/// field routes through Dynamo's relaxed `EasyInputContent` (and transitively
/// the relaxed `InputContent` / `InputImageContent`). Field set is identical to
/// upstream for drop-in compatibility with construction sites in lib/llm.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct EasyInputMessage {
    /// Type discriminator. Optional with default `MessageType::Message` —
    /// matches the OpenAI Responses spec and `openai-python`'s
    /// `EasyInputMessageParam` (`type: Literal["message"]`, non-Required).
    #[serde(default)]
    pub r#type: MessageType,
    pub role: Role,
    pub content: EasyInputContent,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<MessagePhase>,
}

// ---------------------------------------------------------------------------
// Input-side Item / Message / InputItem / InputParam (shadow upstream)
// ---------------------------------------------------------------------------

/// Message item within `Item`. Untagged; disambiguated by the `role` field:
/// the `Output` variant requires `role: "assistant"` (via `AssistantRole`,
/// which is a single-variant enum) and `Input` requires `role` in
/// `"user" | "system" | "developer"` (via `InputRole`). A payload with an
/// unknown role (e.g. `"tool"`) or a missing `role` produces the generic
/// untagged-enum error — callers are expected to send a valid role. If you
/// see the "data did not match any variant of untagged enum" failure on this
/// type, it is almost always a role mismatch.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum MessageItem {
    /// Prior assistant output echoed back (role: assistant). Tried first — its
    /// `role` constraint excludes user/system/developer inputs.
    Output(InputOutputMessage),
    /// User / system / developer input message.
    Input(InputMessage),
}

/// Structured input/output item, discriminated by `type`. Mirrors upstream
/// `Item` variant-for-variant; only `Message` uses a Dynamo-owned type.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Item {
    Message(MessageItem),
    FileSearchCall(FileSearchToolCall),
    ComputerCall(ComputerToolCall),
    ComputerCallOutput(ComputerCallOutputItemParam),
    WebSearchCall(WebSearchToolCall),
    FunctionCall(FunctionToolCall),
    FunctionCallOutput(FunctionCallOutputItemParam),
    ToolSearchCall(ToolSearchCallItemParam),
    ToolSearchOutput(ToolSearchOutputItemParam),
    Reasoning(ReasoningItem),
    Compaction(CompactionSummaryItemParam),
    ImageGenerationCall(ImageGenToolCall),
    CodeInterpreterCall(CodeInterpreterToolCall),
    LocalShellCall(LocalShellToolCall),
    LocalShellCallOutput(LocalShellToolCallOutput),
    ShellCall(FunctionShellCallItemParam),
    ShellCallOutput(FunctionShellCallOutputItemParam),
    ApplyPatchCall(ApplyPatchToolCallItemParam),
    ApplyPatchCallOutput(ApplyPatchToolCallOutputItemParam),
    McpListTools(MCPListTools),
    McpApprovalRequest(MCPApprovalRequest),
    McpApprovalResponse(MCPApprovalResponse),
    McpCall(MCPToolCall),
    CustomToolCallOutput(CustomToolCallOutput),
    CustomToolCall(CustomToolCall),
}

/// Single input item. Untagged; order matters (most specific first).
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputItem {
    ItemReference(ItemReference),
    Item(Item),
    EasyMessage(EasyInputMessage),
}

/// Input to a `POST /v1/responses` request.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum InputParam {
    Text(String),
    Items(Vec<InputItem>),
}

impl Default for InputParam {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

// ---------------------------------------------------------------------------
// CreateResponse (owned, uses Dynamo-owned InputParam)
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/responses`. Mirrors upstream `CreateResponse`
/// field-for-field but uses Dynamo-owned `InputParam`, which transitively
/// accepts the relaxed input shapes described in this module's header. All
/// other fields reference upstream types verbatim.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Default)]
pub struct CreateResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<ConversationParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeEnum>>,
    pub input: InputParam,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Prompt>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_retention: Option<PromptCacheRetention>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_identifier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ResponseStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponseTextParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoiceParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_assistant_message_without_id_or_status() {
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
                assert!(out.id.is_none());
                assert!(out.status.is_none());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn input_image_without_detail_defaults_to_auto() {
        let json = serde_json::json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg"
        });
        let content: InputContent = serde_json::from_value(json).unwrap();
        match content {
            InputContent::InputImage(img) => assert_eq!(img.detail, ImageDetail::Auto),
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[test]
    fn input_image_with_explicit_null_detail_defaults_to_auto() {
        let json = serde_json::json!({
            "type": "input_image",
            "image_url": "https://example.com/cat.jpg",
            "detail": null
        });
        let content: InputContent = serde_json::from_value(json).unwrap();
        match content {
            InputContent::InputImage(img) => assert_eq!(img.detail, ImageDetail::Auto),
            other => panic!("expected InputImage, got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_without_content_field_deserializes() {
        // Bare assistant shell — no `content` field at all. Seen in real
        // Codex/Agents-SDK traffic on pure tool-call turns. `#[serde(default)]`
        // on `content` must accept omission and yield an empty vec.
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.role, AssistantRole::Assistant);
                assert!(out.content.is_empty());
                assert!(out.id.is_none());
                assert!(out.status.is_none());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn assistant_message_with_explicit_null_content_deserializes() {
        // Mirrors the `annotations: null` case: some serializers emit JSON null
        // for absent fields instead of omitting them. `Vec::deserialize` rejects
        // null, so `content` also needs `deserialize_null_as_empty_vec`.
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": null
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert!(out.content.is_empty());
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn mcp_call_item_deserializes() {
        // Guards against Item variant drift vs upstream — MCP item types were
        // added after the initial owned `Item` chain landed.
        let json = serde_json::json!({
            "type": "mcp_call",
            "id": "mcp_1",
            "server_label": "srv",
            "name": "t",
            "arguments": "{}"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::Item(Item::McpCall(_))));
    }

    #[test]
    fn strict_assistant_message_still_deserializes() {
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "status": "completed",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert_eq!(out.id.as_deref(), Some("msg_1"));
                assert_eq!(out.status, Some(OutputStatus::Completed));
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn user_message_routes_to_input_variant() {
        let json = serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(
            item,
            InputItem::Item(Item::Message(MessageItem::Input(_)))
        ));
    }

    #[test]
    fn function_call_item_still_deserializes() {
        let json = serde_json::json!({
            "type": "function_call",
            "call_id": "c",
            "name": "f",
            "arguments": "{}"
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::Item(Item::FunctionCall(_))));
    }

    #[test]
    fn easy_message_string_content_routes_to_easymessage() {
        let json = serde_json::json!({"role": "assistant", "content": "x"});
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::EasyMessage(_)));
    }

    #[test]
    fn output_text_without_annotations_defaults_empty() {
        let json = serde_json::json!({"type": "output_text", "text": "hi"});
        let part: InputOutputMessageContent = serde_json::from_value(json).unwrap();
        match part {
            InputOutputMessageContent::OutputText(t) => {
                assert!(t.annotations.is_empty());
            }
            _ => panic!("expected OutputText"),
        }
    }

    #[test]
    fn output_text_with_explicit_null_annotations_deserializes_as_empty() {
        // Some clients serialize absent fields as JSON null instead of omitting
        // them. `Vec::deserialize` would reject null; the custom deserializer
        // treats explicit null identically to a missing field.
        let json = serde_json::json!({"type": "output_text", "text": "hi", "annotations": null});
        let part: InputOutputMessageContent = serde_json::from_value(json).unwrap();
        match part {
            InputOutputMessageContent::OutputText(t) => {
                assert!(t.annotations.is_empty());
            }
            _ => panic!("expected OutputText"),
        }
    }

    #[test]
    fn assistant_message_with_explicit_null_id_and_status_deserializes() {
        // `Option<T>` natively accepts null as `None`, so these explicit-null
        // fields should flow through without a custom deserializer. This test
        // pins that behavior against accidental regressions (e.g. if someone
        // switches the field type away from `Option<_>`).
        let json = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "id": null,
            "status": null,
            "content": [{"type": "output_text", "text": "hi", "annotations": null}]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Output(out))) => {
                assert!(out.id.is_none());
                assert!(out.status.is_none());
                assert_eq!(out.content.len(), 1);
            }
            other => panic!("expected Item::Message(Output), got {other:?}"),
        }
    }

    #[test]
    fn create_response_roundtrip_with_relaxed_input() {
        let body = serde_json::json!({
            "model": "m",
            "input": [
                {"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": "hi"}
                ]},
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "\n\n"}
                ]},
                {"type": "function_call_output", "call_id": "c", "output": "x"}
            ]
        });

        let req: CreateResponse = serde_json::from_value(body).unwrap();
        let items = match &req.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 4);
        assert!(matches!(
            items[2],
            InputItem::Item(Item::Message(MessageItem::Output(_)))
        ));
    }

    // ---- EasyInputMessage / multimodal-without-`type` regression coverage ----
    // See issue #9468. Before the EasyInputMessage/EasyInputContent shadow
    // landed, the `InputItem::EasyMessage` fallback still routed through
    // upstream's strict `InputImageContent` (required `detail`), so any
    // multimodal message that omitted the spec-default `type: "message"` would
    // fail with "data did not match any variant of untagged enum InputItem".

    #[test]
    fn easy_message_multimodal_without_type_routes_to_easymessage() {
        // AIPerf's pre-PR-931 payload shape: no top-level `type`, content is a
        // list containing an `input_image` part with no `detail`.
        let json = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::User);
                assert_eq!(easy.r#type, MessageType::Message);
                match easy.content {
                    EasyInputContent::ContentList(parts) => {
                        assert_eq!(parts.len(), 1);
                        match &parts[0] {
                            InputContent::InputImage(img) => {
                                assert_eq!(img.detail, ImageDetail::Auto);
                                assert_eq!(
                                    img.image_url.as_deref(),
                                    Some("data:image/png;base64,abc")
                                );
                            }
                            other => panic!("expected InputImage, got {other:?}"),
                        }
                    }
                    other => panic!("expected ContentList, got {other:?}"),
                }
            }
            other => panic!("expected EasyMessage, got {other:?}"),
        }
    }

    #[test]
    fn easy_message_multimodal_with_explicit_null_detail() {
        // Same shape as above but with `detail: null` — exercises the
        // null-as-default path on the relaxed `InputImageContent` reached via
        // the EasyMessage variant.
        let json = serde_json::json!({
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": null}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        assert!(matches!(item, InputItem::EasyMessage(_)));
    }

    #[test]
    fn easy_message_assistant_multimodal_without_type() {
        // Mixed-turn shape AIPerf emits when the prior assistant turn carried
        // structured (non-string) content: role=assistant, content list, no
        // top-level `type`.
        let json = serde_json::json!({
            "role": "assistant",
            "content": [
                {"type": "input_text", "text": "ok"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::Assistant);
            }
            other => panic!("expected EasyMessage(assistant), got {other:?}"),
        }
    }

    #[test]
    fn easy_message_text_only_without_type_unchanged() {
        // Regression guard: the pre-existing text-only path was already
        // working (no multimodal content -> never hit upstream's strict
        // `InputImageContent`). Pin it so a future glob-shadow change can't
        // break it.
        let json = serde_json::json!({"role": "user", "content": "Hello"});
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::EasyMessage(easy) => {
                assert_eq!(easy.role, Role::User);
                assert!(matches!(easy.content, EasyInputContent::Text(ref s) if s == "Hello"));
            }
            other => panic!("expected EasyMessage(Text), got {other:?}"),
        }
    }

    #[test]
    fn easy_message_with_explicit_type_still_routes_to_item_message() {
        // AIPerf's post-PR-931 payload (with `type: "message"`) should still
        // hit the structured `Item::Message` path first — proving the existing
        // strict path didn't regress when EasyMessage was shadowed.
        let json = serde_json::json!({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,abc"}
            ]
        });
        let item: InputItem = serde_json::from_value(json).unwrap();
        match item {
            InputItem::Item(Item::Message(MessageItem::Input(msg))) => {
                assert_eq!(msg.role, InputRole::User);
                assert_eq!(msg.content.len(), 1);
            }
            other => panic!("expected Item::Message(Input), got {other:?}"),
        }
    }

    #[test]
    fn create_response_roundtrip_aiperf_pre_pr931_payload() {
        // End-to-end shape: the exact request body AIPerf was emitting before
        // PR-931 for a multi-turn multimodal conversation. Mirrors what the
        // HTTP frontend receives. Must deserialize without error and preserve
        // turn ordering.
        let body = serde_json::json!({
            "model": "Qwen/Qwen2-VL-2B-Instruct",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "ok"}]
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Now describe a different one."}]
                }
            ]
        });
        let req: CreateResponse = serde_json::from_value(body).unwrap();
        let items = match &req.input {
            InputParam::Items(items) => items,
            _ => panic!("expected Items"),
        };
        assert_eq!(items.len(), 3);
        // All three turns must land as EasyMessage (no top-level `type`).
        for (idx, item) in items.iter().enumerate() {
            assert!(
                matches!(item, InputItem::EasyMessage(_)),
                "turn {idx} did not route to EasyMessage: {item:?}",
            );
        }
    }
}
