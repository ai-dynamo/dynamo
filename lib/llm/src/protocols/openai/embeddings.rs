// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize, Serializer, de::IgnoredAny};
use serde_json::{Map, Value};
use utoipa::ToSchema;
use validator::Validate;

mod aggregator;
mod nvext;

pub use nvext::{NvExt, NvExtProvider};

/// `/v1/embeddings` request.
///
/// Deserialized through `RequestFields` (see `messages`) and
/// serialized by hand, so a chat-style request reaches the worker without the
/// placeholder `input`. The field attributes still describe the OpenAPI schema.
#[derive(ToSchema, Deserialize, Validate, Debug, Clone)]
#[serde(try_from = "RequestFields")]
pub struct NvCreateEmbeddingRequest {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateEmbeddingRequest,

    /// Whether raw-text inputs should include model-declared special tokens.
    ///
    /// This vLLM-compatible extension is optional. When omitted, raw text
    /// includes model-declared special tokens, matching vLLM's pooling
    /// default. It is ignored when the caller supplies token IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_special_tokens: Option<bool>,

    /// vLLM tokenizer option for raw-text embedding requests. vLLM accepts
    /// -1 as the sentinel for truncating to the model's maximum length.
    /// This integration currently follows vLLM's right-truncation mode and
    /// keeps the first N tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate_prompt_tokens: Option<i64>,

    /// vLLM's chat-style embedding input: one conversation, or a batch of
    /// conversations, whose messages can carry text and media content parts
    /// (`image_url`, `video_url`, `audio_url`, `input_audio`). The worker
    /// renders each conversation with the model's chat template and
    /// multimodal processor and returns one embedding per conversation.
    ///
    /// vLLM also accepts the same conversations in place of `input`; both
    /// forms are normalized to `messages`. When set, `inner.input` is an empty
    /// placeholder that is never serialized.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<EmbeddingMessages>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// One message of a chat-style embedding request. Roles and content parts are
/// left to the backend's chat template (embedding templates use roles such as
/// `query` and `document`), so only the presence of a string `role` is checked.
pub type EmbeddingChatMessage = Map<String, Value>;

/// Chat-style embedding input, as accepted by vLLM's `/v1/embeddings`.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingMessages {
    /// One conversation, embedded as one vector.
    #[schema(value_type = Vec<Object>)]
    Conversation(Vec<EmbeddingChatMessage>),
    /// Several conversations, one vector each, in order.
    #[schema(value_type = Vec<Vec<Object>>)]
    Batch(Vec<Vec<EmbeddingChatMessage>>),
}

impl EmbeddingMessages {
    /// Every conversation is non-empty and every message has a string `role`.
    fn is_well_formed(&self) -> bool {
        fn is_conversation(messages: &[EmbeddingChatMessage]) -> bool {
            !messages.is_empty()
                && messages
                    .iter()
                    .all(|message| message.get("role").is_some_and(Value::is_string))
        }
        match self {
            Self::Conversation(messages) => is_conversation(messages),
            Self::Batch(conversations) => {
                !conversations.is_empty()
                    && conversations
                        .iter()
                        .all(|conversation| is_conversation(conversation))
            }
        }
    }
}

const INPUT_SHAPE_ERROR: &str = "`input` must be a string, a list of strings, a list of token \
                                 IDs, a list of token-ID lists, or chat-style messages in which \
                                 every message has a string `role`";
const MESSAGES_SHAPE_ERROR: &str = "`messages` must be a non-empty list of chat messages, or a \
                                    non-empty list of such lists, and every message needs a \
                                    string `role`";

/// `input` as sent: OpenAI's text and token-ID forms (in `EmbeddingInput`'s
/// variant order), or vLLM's chat-style conversations. Parsed directly, so
/// text and token-ID requests skip building a generic JSON tree.
#[derive(Deserialize)]
#[serde(untagged)]
enum InputField {
    String(String),
    StringArray(Vec<String>),
    IntegerArray(Vec<u32>),
    ArrayOfIntegerArray(Vec<Vec<u32>>),
    Messages(EmbeddingMessages),
    /// Any other shape, rejected with [`INPUT_SHAPE_ERROR`].
    Invalid(IgnoredAny),
}

/// `messages` as sent; any other shape is rejected with [`MESSAGES_SHAPE_ERROR`].
#[derive(Deserialize)]
#[serde(untagged)]
enum MessagesField {
    Messages(EmbeddingMessages),
    Invalid(IgnoredAny),
}

/// The request's fields as sent, before chat-style `input` moves to `messages`.
/// A `null` field counts as absent, as in vLLM.
#[derive(Deserialize)]
struct RequestFields {
    model: String,
    input: Option<InputField>,
    messages: Option<MessagesField>,
    encoding_format: Option<dynamo_protocols::types::EncodingFormat>,
    user: Option<String>,
    dimensions: Option<u32>,
    add_special_tokens: Option<bool>,
    truncate_prompt_tokens: Option<i64>,
    nvext: Option<NvExt>,
    // vLLM chat-request options the worker does not apply. They are read
    // only so that a non-default value is rejected rather than ignored.
    add_generation_prompt: Option<bool>,
    continue_final_message: Option<bool>,
    chat_template_kwargs: Option<Map<String, Value>>,
}

impl TryFrom<RequestFields> for NvCreateEmbeddingRequest {
    type Error = &'static str;

    fn try_from(fields: RequestFields) -> Result<Self, Self::Error> {
        use dynamo_protocols::types::EmbeddingInput;

        // vLLM renders embedding conversations with these at their defaults.
        // Anything else would embed a different prompt than `vllm serve` does.
        if fields.add_generation_prompt == Some(true) {
            return Err("`add_generation_prompt: true` is not supported for embeddings");
        }
        if fields.continue_final_message == Some(true) {
            return Err("`continue_final_message: true` is not supported for embeddings");
        }
        if fields
            .chat_template_kwargs
            .as_ref()
            .is_some_and(|kwargs| !kwargs.is_empty())
        {
            return Err("`chat_template_kwargs` is not supported for embeddings");
        }

        let messages = match fields.messages {
            None => None,
            Some(MessagesField::Messages(messages)) if messages.is_well_formed() => Some(messages),
            Some(_) => return Err(MESSAGES_SHAPE_ERROR),
        };
        let (input, messages) = match (fields.input, messages) {
            (Some(InputField::Messages(_)), Some(_)) => {
                return Err("pass the conversation in either `messages` or `input`, not both");
            }
            (Some(_), Some(_)) => {
                return Err("pass either `input` or chat-style `messages`, not both");
            }
            (None, None) => return Err("missing field `input`"),
            // The OpenAI request type requires `input`; a chat-style request
            // gets an empty placeholder that is never serialized.
            (None, Some(messages)) => (EmbeddingInput::StringArray(vec![]), Some(messages)),
            (Some(input), None) => match input {
                InputField::String(text) => (EmbeddingInput::String(text), None),
                InputField::StringArray(texts) => (EmbeddingInput::StringArray(texts), None),
                InputField::IntegerArray(tokens) => (EmbeddingInput::IntegerArray(tokens), None),
                InputField::ArrayOfIntegerArray(tokens) => {
                    (EmbeddingInput::ArrayOfIntegerArray(tokens), None)
                }
                InputField::Messages(messages) if messages.is_well_formed() => {
                    (EmbeddingInput::StringArray(vec![]), Some(messages))
                }
                InputField::Messages(_) | InputField::Invalid(_) => {
                    return Err(INPUT_SHAPE_ERROR);
                }
            },
        };
        Ok(Self {
            inner: dynamo_protocols::types::CreateEmbeddingRequest {
                model: fields.model,
                input,
                encoding_format: fields.encoding_format,
                user: fields.user,
                dimensions: fields.dimensions,
            },
            add_special_tokens: fields.add_special_tokens,
            truncate_prompt_tokens: fields.truncate_prompt_tokens,
            messages,
            nvext: fields.nvext,
        })
    }
}

/// Serialized form of a text or token-ID request.
#[derive(Serialize)]
struct InputRequestRef<'a> {
    #[serde(flatten)]
    inner: &'a dynamo_protocols::types::CreateEmbeddingRequest,
    #[serde(skip_serializing_if = "Option::is_none")]
    add_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate_prompt_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nvext: Option<&'a NvExt>,
}

/// Serialized form of a chat-style request: vLLM's request body, with
/// `messages` in place of the placeholder `input`.
#[derive(Serialize)]
struct ChatRequestRef<'a> {
    model: &'a str,
    messages: &'a EmbeddingMessages,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<&'a dynamo_protocols::types::EncodingFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    add_special_tokens: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate_prompt_tokens: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nvext: Option<&'a NvExt>,
}

impl Serialize for NvCreateEmbeddingRequest {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Destructured without `..`, so a new field fails to compile here
        // until it is serialized (or deliberately skipped).
        let Self {
            inner,
            add_special_tokens,
            truncate_prompt_tokens,
            messages,
            nvext,
        } = self;
        let Some(messages) = messages else {
            return InputRequestRef {
                inner,
                add_special_tokens: *add_special_tokens,
                truncate_prompt_tokens: *truncate_prompt_tokens,
                nvext: nvext.as_ref(),
            }
            .serialize(serializer);
        };
        let dynamo_protocols::types::CreateEmbeddingRequest {
            model,
            // An empty placeholder on chat-style requests; see `messages`.
            input: _,
            encoding_format,
            user,
            dimensions,
        } = inner;
        ChatRequestRef {
            model,
            messages,
            encoding_format: encoding_format.as_ref(),
            user: user.as_deref(),
            dimensions: *dimensions,
            add_special_tokens: *add_special_tokens,
            truncate_prompt_tokens: *truncate_prompt_tokens,
            nvext: nvext.as_ref(),
        }
        .serialize(serializer)
    }
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI unary chat completion response, embedded
///   using `serde(flatten)`.
#[derive(ToSchema, Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateEmbeddingResponse {
    #[serde(flatten)]
    #[schema(value_type = Object)]
    pub inner: dynamo_protocols::types::CreateEmbeddingResponse,
}

impl NvCreateEmbeddingResponse {
    pub fn empty() -> Self {
        Self {
            inner: dynamo_protocols::types::CreateEmbeddingResponse {
                object: "list".to_string(),
                model: "embedding".to_string(),
                data: vec![],
                usage: dynamo_protocols::types::EmbeddingUsage {
                    prompt_tokens: 0,
                    total_tokens: 0,
                },
            },
        }
    }
}

/// Implements `NvExtProvider` for `NvCreateEmbeddingRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateEmbeddingRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }
}

/// Implements `AnnotationsProvider` for `NvCreateEmbeddingRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateEmbeddingRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::{EmbeddingMessages, NvCreateEmbeddingRequest};
    use serde_json::json;

    fn image_conversation() -> serde_json::Value {
        json!([{
            "role": "document",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "a chart"}
            ]
        }])
    }

    fn parse(value: serde_json::Value) -> Result<NvCreateEmbeddingRequest, serde_json::Error> {
        serde_json::from_value(value)
    }

    #[test]
    fn messages_forward_as_vllm_chat_request() {
        let request = parse(json!({
            "model": "m",
            "messages": image_conversation(),
            "dimensions": 256,
            "nvext": {"annotations": ["token_ids"]}
        }))
        .unwrap();
        assert!(matches!(
            request.messages,
            Some(EmbeddingMessages::Conversation(ref messages)) if messages.len() == 1
        ));

        // The worker receives vLLM's request body: `messages`, no placeholder `input`.
        assert_eq!(
            serde_json::to_value(&request).unwrap(),
            json!({
                "model": "m",
                "messages": image_conversation(),
                "dimensions": 256,
                "nvext": {"annotations": ["token_ids"]}
            })
        );
    }

    #[test]
    fn chat_shaped_input_moves_to_messages() {
        let single = parse(json!({"model": "m", "input": image_conversation()})).unwrap();
        assert!(matches!(
            single.messages,
            Some(EmbeddingMessages::Conversation(_))
        ));

        let batch = parse(json!({
            "model": "m",
            "input": [image_conversation(), [{"role": "query", "content": "chart"}]]
        }))
        .unwrap();
        assert!(matches!(
            batch.messages,
            Some(EmbeddingMessages::Batch(ref conversations)) if conversations.len() == 2
        ));
        let wire = serde_json::to_value(&batch).unwrap();
        assert!(wire.get("input").is_none());
        assert_eq!(wire["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn text_and_token_inputs_are_unchanged() {
        for input in [
            json!("hello"),
            json!(["a", "b"]),
            json!([1, 2]),
            json!([[1], [2]]),
        ] {
            let body = json!({"model": "m", "input": input});
            let request = parse(body.clone()).unwrap();
            assert!(request.messages.is_none());
            assert_eq!(serde_json::to_value(&request).unwrap(), body);
        }
    }

    #[test]
    fn null_messages_is_omitted() {
        let request = parse(json!({"model": "m", "input": "hello", "messages": null})).unwrap();
        assert!(request.messages.is_none());
    }

    #[test]
    fn null_input_with_messages_is_a_chat_request() {
        // As in vLLM, `null` means the field was not sent.
        let request = parse(json!({
            "model": "m",
            "input": null,
            "messages": image_conversation()
        }))
        .unwrap();
        assert!(request.messages.is_some());
        assert!(
            serde_json::to_value(&request)
                .unwrap()
                .get("input")
                .is_none()
        );
    }

    #[test]
    fn default_chat_options_are_accepted_and_not_forwarded() {
        // vLLM's defaults for embedding conversations; sending them changes nothing.
        for options in [
            json!({"add_generation_prompt": false, "continue_final_message": false, "chat_template_kwargs": {}}),
            json!({"add_generation_prompt": null, "continue_final_message": null, "chat_template_kwargs": null}),
        ] {
            let mut body = json!({"model": "m", "messages": image_conversation()});
            body.as_object_mut()
                .unwrap()
                .extend(options.as_object().unwrap().clone());
            let request = parse(body).unwrap();
            assert_eq!(
                serde_json::to_value(&request).unwrap(),
                json!({"model": "m", "messages": image_conversation()})
            );
        }
    }

    #[test]
    fn invalid_chat_input_is_rejected() {
        let cases = [
            (
                json!({"model": "m", "input": "hello", "messages": image_conversation()}),
                "not both",
            ),
            (
                json!({"model": "m", "input": image_conversation(), "messages": image_conversation()}),
                "not both",
            ),
            (json!({"model": "m", "messages": []}), "non-empty"),
            (json!({"model": "m", "messages": [[]]}), "non-empty"),
            (
                json!({"model": "m", "messages": [{"content": "no role"}]}),
                "string `role`",
            ),
            (json!({"model": "m", "messages": "hello"}), "non-empty"),
            (
                json!({"model": "m", "input": [{"text": "no role"}]}),
                "string `role`",
            ),
            (
                json!({"model": "m", "input": {"text": "hello"}}),
                "`input` must be",
            ),
            (json!({"model": "m"}), "missing field `input`"),
            (
                json!({"model": "m", "input": null}),
                "missing field `input`",
            ),
            (json!({"input": "hello"}), "missing field `model`"),
            (
                json!({"model": "m", "messages": image_conversation(), "add_generation_prompt": true}),
                "`add_generation_prompt: true` is not supported",
            ),
            (
                json!({"model": "m", "messages": image_conversation(), "continue_final_message": true}),
                "`continue_final_message: true` is not supported",
            ),
            (
                json!({"model": "m", "messages": image_conversation(), "chat_template_kwargs": {"enable_thinking": false}}),
                "`chat_template_kwargs` is not supported",
            ),
        ];
        for (body, expected) in cases {
            let err = parse(body.clone()).unwrap_err().to_string();
            assert!(err.contains(expected), "{body}: {err}");
        }
    }

    #[test]
    fn truncate_prompt_tokens_round_trips() {
        for truncate_prompt_tokens in [2048, -1] {
            let request: NvCreateEmbeddingRequest = serde_json::from_value(json!({
                "model": "test-model",
                "input": "hello",
                "truncate_prompt_tokens": truncate_prompt_tokens
            }))
            .unwrap();

            assert_eq!(request.truncate_prompt_tokens, Some(truncate_prompt_tokens));

            let value = serde_json::to_value(request).unwrap();
            assert_eq!(value["truncate_prompt_tokens"], truncate_prompt_tokens);
        }
    }

    #[test]
    fn omitted_truncate_prompt_tokens_is_not_serialized() {
        let request: NvCreateEmbeddingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello"
        }))
        .unwrap();

        assert_eq!(request.truncate_prompt_tokens, None);

        let value = serde_json::to_value(request).unwrap();
        assert!(value.get("truncate_prompt_tokens").is_none());
    }

    #[test]
    fn add_special_tokens_round_trips_and_is_optional() {
        for expected in [true, false] {
            let request: NvCreateEmbeddingRequest = serde_json::from_value(json!({
                "model": "test-model",
                "input": "hello",
                "add_special_tokens": expected
            }))
            .unwrap();
            assert_eq!(request.add_special_tokens, Some(expected));
            assert_eq!(
                serde_json::to_value(request).unwrap()["add_special_tokens"],
                expected
            );
        }

        let request: NvCreateEmbeddingRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello"
        }))
        .unwrap();
        assert_eq!(request.add_special_tokens, None);
        assert!(
            serde_json::to_value(request)
                .unwrap()
                .get("add_special_tokens")
                .is_none()
        );
    }
}
