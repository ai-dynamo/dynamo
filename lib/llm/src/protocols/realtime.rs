// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Experimental realtime text options. Workers own `nvext` validation and
//! translation; the frontend must carry it across both transport boundaries.

use dynamo_protocols::types::realtime as openai;
use openai::EventType;
use serde::{Deserialize, Deserializer, Serialize, de::Error};
use serde_json::{Map, Value};

#[derive(Debug, Serialize, Deserialize)]
pub struct WithNvext<T> {
    #[serde(flatten)]
    pub inner: T,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<Map<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SessionUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    pub session: WithNvext<openai::Session>,
}

/// Response overrides are optional. In particular, omitted `input` inherits
/// history, whereas an explicit empty array clears it for this response.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ResponseOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<openai::ResponseAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation: Option<openai::Conversation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Vec<openai::RealtimeConversationItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<openai::MaxOutputTokens>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Map<String, Value>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<openai::ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<openai::RealtimeTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<openai::RealtimeReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<Map<String, Value>>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TextClientEvent {
    #[serde(rename = "session.update")]
    SessionUpdate(SessionUpdate),
    #[serde(rename = "response.create")]
    ResponseCreate {
        #[serde(skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        response: Option<Box<ResponseOptions>>,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum DynamoRealtimeClientEvent {
    Text(TextClientEvent),
    OpenAI(Box<openai::RealtimeClientEvent>),
    Extension(openai::RealtimeClientEventExtension),
}

impl<'de> Deserialize<'de> for DynamoRealtimeClientEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.get("type").and_then(Value::as_str) {
            Some("session.update" | "response.create") => {
                // The pinned upstream function type omits strict. Reject unknown
                // function constraints instead of silently dropping them in serde.
                let options = value.get("session").or_else(|| value.get("response"));
                if let Some(tools) = options
                    .and_then(|o| o.get("tools"))
                    .and_then(Value::as_array)
                {
                    for tool in tools {
                        if tool.get("type").and_then(Value::as_str) == Some("function")
                            && tool.as_object().is_some_and(|o| {
                                o.keys().any(|k| {
                                    !matches!(
                                        k.as_str(),
                                        "type" | "name" | "description" | "parameters"
                                    )
                                })
                            })
                        {
                            return Err(D::Error::custom(
                                "unsupported realtime function tool field",
                            ));
                        }
                    }
                }
                serde_json::from_value(value)
                    .map(Self::Text)
                    .map_err(D::Error::custom)
            }
            Some("input_text.append" | "input_text.commit" | "input_text.clear") => {
                serde_json::from_value(value)
                    .map(Self::Extension)
                    .map_err(D::Error::custom)
            }
            _ => serde_json::from_value(value)
                .map(Self::OpenAI)
                .map_err(D::Error::custom),
        }
    }
}

impl EventType for DynamoRealtimeClientEvent {
    fn event_type(&self) -> &'static str {
        match self {
            Self::Text(TextClientEvent::SessionUpdate(_)) => "session.update",
            Self::Text(TextClientEvent::ResponseCreate { .. }) => "response.create",
            Self::OpenAI(event) => event.event_type(),
            Self::Extension(event) => event.event_type(),
        }
    }
}

impl From<openai::RealtimeClientEvent> for DynamoRealtimeClientEvent {
    fn from(value: openai::RealtimeClientEvent) -> Self {
        match value {
            openai::RealtimeClientEvent::SessionUpdate(update) => {
                Self::Text(TextClientEvent::SessionUpdate(SessionUpdate {
                    event_id: update.event_id,
                    session: WithNvext {
                        inner: update.session,
                        nvext: None,
                    },
                }))
            }
            other => Self::OpenAI(Box::new(other)),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum TextServerEvent {
    #[serde(rename = "session.updated")]
    SessionUpdated {
        event_id: String,
        session: WithNvext<openai::Session>,
    },
    #[serde(rename = "response.done")]
    ResponseDone {
        event_id: String,
        response: Box<WithNvext<openai::RealtimeResponse>>,
    },
    #[serde(rename = "response.nvext.reasoning.delta")]
    ReasoningDelta {
        event_id: String,
        response_id: String,
        delta: String,
    },
    #[serde(rename = "response.nvext.reasoning.done")]
    ReasoningDone {
        event_id: String,
        response_id: String,
        text: String,
    },
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum RealtimeServerEvent {
    Text(TextServerEvent),
    OpenAI(Box<openai::RealtimeServerEvent>),
}

impl<'de> Deserialize<'de> for RealtimeServerEvent {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(deserializer)?;
        match value.get("type").and_then(Value::as_str) {
            Some(
                "session.updated"
                | "response.done"
                | "response.nvext.reasoning.delta"
                | "response.nvext.reasoning.done",
            ) => serde_json::from_value(value)
                .map(Self::Text)
                .map_err(D::Error::custom),
            _ => serde_json::from_value(value)
                .map(Self::OpenAI)
                .map_err(D::Error::custom),
        }
    }
}

impl From<openai::RealtimeServerEvent> for RealtimeServerEvent {
    fn from(value: openai::RealtimeServerEvent) -> Self {
        Self::OpenAI(Box::new(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_request_round_trip_preserves_overrides_and_omission() {
        for value in [
            json!({"type":"session.update", "session": {
                "type":"realtime", "model":"test", "tools":[{
                    "type":"function", "name":"weather", "description":"Weather", "parameters":{"type":"object"}
                }], "tool_choice":{"type":"function","name":"weather"}, "parallel_tool_calls":false,
                "nvext":{"sampling":{"temperature":0.0,"seed":0},"chat_template_kwargs":{"enable_thinking":false}}
            }}),
            json!({"type":"response.create","response":{"conversation":"none","tool_choice":"none"}}),
            json!({"type":"response.create","response":{"input":[],"instructions":"","parallel_tool_calls":false}}),
            json!({"type":"response.create"}),
            json!({"type":"input_text.append","text":"Hello"}),
        ] {
            let event: DynamoRealtimeClientEvent = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(event).unwrap(), value);
        }
    }

    #[test]
    fn server_extensions_are_not_stripped_or_silently_downgraded() {
        for value in [
            json!({"type":"session.updated","event_id":"e1","session":{"type":"realtime","nvext":{"capabilities":["tools","reasoning"]}}}),
            json!({"type":"response.nvext.reasoning.delta","event_id":"e2","response_id":"r1","delta":"Think"}),
            json!({"type":"response.nvext.reasoning.done","event_id":"e3","response_id":"r1","text":"Think"}),
        ] {
            let event: RealtimeServerEvent = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(serde_json::to_value(event).unwrap(), value);
        }
        assert!(
            serde_json::from_value::<RealtimeServerEvent>(json!({
                "type":"session.updated","event_id":"e1","session":{"type":"realtime","nvext":false}
            }))
            .is_err()
        );
    }

    #[test]
    fn legacy_audio_session_round_trips() {
        let value = json!({"type":"session.update","session":{"type":"transcription","audio":{"input":{
            "format":{"type":"audio/pcm","rate":24000},"transcription":{"model":"whisper-1"}
        }}}});
        let event: DynamoRealtimeClientEvent = serde_json::from_value(value.clone()).unwrap();
        let old: openai::DynamoRealtimeClientEvent = serde_json::from_value(value).unwrap();
        assert_eq!(
            serde_json::to_value(event).unwrap(),
            serde_json::to_value(old).unwrap()
        );
    }
}
