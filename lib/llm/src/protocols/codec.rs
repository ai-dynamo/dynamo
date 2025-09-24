// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A module for parsing Server-Sent Events (SSE) streams according to the SSE specification.
//!
//! This module provides `SseLineCodec<T>`, a codec for decoding SSE streams into typed messages.
//! It handles parsing of `id`, `event`, `data`, and comments, and attempts to deserialize
//! the `data` field into the specified type `T`.
//!

// TODO: Determine if we should use an External EventSource crate. There appear to be several
// potential candidates.

use std::{io::Cursor, pin::Pin};

use bytes::BytesMut;
use futures::Stream;
use serde::Deserialize;
use tokio_util::codec::{Decoder, FramedRead, LinesCodec};

use super::Annotated;

/// An error that occurs when decoding an SSE stream.
#[derive(Debug, thiserror::Error)]
pub enum SseCodecError {
    #[error("SseLineCodec decode error: {0}")]
    DecodeError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// A codec for decoding SSE streams into `Message<T>` instances.
///
/// This codec parses SSE streams according to the SSE specification and attempts to deserialize
/// the `data` field into the specified type `T`.
///
/// # Type Parameters
///
/// * `T` - The type to deserialize the `data` field into.
pub struct SseLineCodec {
    lines_codec: LinesCodec,
    data_buffer: String,
    event_type_buffer: String,
    last_event_id_buffer: String,
    comments_buffer: Vec<String>,
}

/// Represents a parsed SSE message.
///
/// The `Message` struct contains optional fields for `id`, `event`, `data`, and a vector of `comments`.
///
/// # Type Parameters
///
/// * `T` - The type to deserialize the `data` field into.
#[derive(Debug)]
pub struct Message {
    pub id: Option<String>,
    pub event: Option<String>,
    pub data: Option<String>,
    pub comments: Option<Vec<String>>,
}

impl Message {
    /// Deserializes the `data` field into the specified type `T`.
    ///
    /// # Errors
    ///
    /// Returns an error if the `data` field is empty or if deserialization fails.
    pub fn decode_data<T>(&self) -> Result<T, SseCodecError>
    where
        T: for<'de> Deserialize<'de>,
    {
        serde_json::from_str(self.data.as_ref().ok_or(SseCodecError::DecodeError(
            "no data: message to decode".to_string(),
        ))?)
        .map_err(|e| SseCodecError::DecodeError(format!("failed to deserialized data: {}", e)))
    }
}

impl<T> TryFrom<Message> for Annotated<T>
where
    T: for<'de> Deserialize<'de>,
{
    type Error = String;

    fn try_from(value: Message) -> Result<Annotated<T>, Self::Error> {
        // determine if the message had an error
        if let Some(event) = value.event.as_ref()
            && event == "error"
        {
            let message = match &value.comments {
                Some(comments) => comments.join("\n"),
                None => "`event: error` detected, but no error message found".to_string(),
            };
            return Err(message);
        }

        // try to deserialize the data to T

        let data: Option<T> = match &value.data {
            Some(_) => value.decode_data().map_err(|e| e.to_string())?,
            None => None,
        };

        Ok(Annotated {
            data,
            id: value.id,
            event: value.event,
            comment: value.comments,
        })
    }
}

impl SseLineCodec {
    /// Creates a new `SseLineCodec<T>`.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for SseLineCodec {
    fn default() -> Self {
        Self {
            lines_codec: LinesCodec::new(),
            data_buffer: String::new(),
            event_type_buffer: String::new(),
            last_event_id_buffer: String::new(),
            comments_buffer: Vec::new(),
        }
    }
}

impl Decoder for SseLineCodec {
    type Item = Message;
    type Error = SseCodecError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        loop {
            match self
                .lines_codec
                .decode(src)
                .map_err(|e| SseCodecError::DecodeError(e.to_string()))?
            {
                Some(line) => {
                    let line = line.trim_end_matches(&['\r', '\n'][..]);
                    if line.is_empty() {
                        // End of event; dispatch
                        if !self.data_buffer.is_empty()
                            || !self.event_type_buffer.is_empty()
                            || !self.last_event_id_buffer.is_empty()
                            || !self.comments_buffer.is_empty()
                        {
                            // Remove the last '\n' if present in data_buffer
                            if self.data_buffer.ends_with('\n') {
                                self.data_buffer.pop();
                            }

                            let data = if !self.data_buffer.is_empty() {
                                Some(std::mem::take(&mut self.data_buffer))
                            } else {
                                None
                            };

                            let message = Message {
                                id: if self.last_event_id_buffer.is_empty() {
                                    None
                                } else {
                                    Some(std::mem::take(&mut self.last_event_id_buffer))
                                },
                                event: if self.event_type_buffer.is_empty() {
                                    None
                                } else {
                                    Some(std::mem::take(&mut self.event_type_buffer))
                                },
                                data,
                                comments: if self.comments_buffer.is_empty() {
                                    None
                                } else {
                                    Some(std::mem::take(&mut self.comments_buffer))
                                },
                            };
                            // No need to clear the buffers; they've been replaced with empty values
                            return Ok(Some(message));
                        } else {
                            // No data to dispatch; continue
                            continue;
                        }
                    } else if let Some(comment) = line.strip_prefix(':') {
                        self.comments_buffer.push(comment.trim().into());
                    } else {
                        let (field_name, field_value) = if let Some(idx) = line.find(':') {
                            let (name, value) = line.split_at(idx);
                            let value = value[1..].trim_start_matches(' ');
                            (name, value)
                        } else {
                            (line, "")
                        };

                        match field_name {
                            "event" => {
                                self.event_type_buffer = field_value.to_string();
                            }
                            "data" => {
                                if field_value != "[DONE]" {
                                    if !self.data_buffer.is_empty() {
                                        self.data_buffer.push('\n');
                                    }
                                    self.data_buffer.push_str(field_value);
                                }
                            }
                            "id" => {
                                if !field_value.contains('\0') {
                                    self.last_event_id_buffer = field_value.to_string();
                                }
                            }
                            "retry" => {
                                // For simplicity, we'll ignore retry in this implementation
                            }
                            _ => {
                                // Ignore unknown fields
                            }
                        }
                    }
                }
                None => {
                    // No more data available at the moment
                    return Ok(None);
                }
            }
        }
    }

    fn decode_eof(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        // Attempt to process any remaining data
        let result = self.decode(src)?;
        if result.is_some() {
            return Ok(result);
        }
        // If there's no data left to process, return None
        if self.data_buffer.is_empty()
            && self.event_type_buffer.is_empty()
            && self.last_event_id_buffer.is_empty()
            && self.comments_buffer.is_empty()
        {
            Ok(None)
        } else {
            // Dispatch any remaining data as an event
            if self.data_buffer.ends_with('\n') {
                self.data_buffer.pop();
            }

            let data = if !self.data_buffer.is_empty() {
                Some(std::mem::take(&mut self.data_buffer))
            } else {
                None
            };

            let message = Message {
                id: if self.last_event_id_buffer.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut self.last_event_id_buffer))
                },
                event: if self.event_type_buffer.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut self.event_type_buffer))
                },
                data,
                comments: if self.comments_buffer.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut self.comments_buffer))
                },
            };
            // No need to clear the buffers; they've been replaced with empty values
            Ok(Some(message))
        }
    }
}

/// Creates a stream of `Message` instances from a text stream of SSE events.
pub fn create_message_stream(
    text: &str,
) -> Pin<Box<dyn Stream<Item = Result<Message, SseCodecError>> + Send + Sync>> {
    let cursor = Cursor::new(text.to_string());
    let framed = FramedRead::new(cursor, SseLineCodec::new());
    Box::pin(framed)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use futures::stream::StreamExt;
    use tokio_util::codec::FramedRead;

    use super::*;

    #[derive(Deserialize, Debug, PartialEq)]
    struct TestData {
        message: String,
    }

    #[tokio::test]
    async fn test_message_with_all_fields() {
        let sample_data = r#"id: 123
event: test
data: {"message": "Hello World"}
: This is a comment

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(Ok(message)) = framed.next().await {
            assert_eq!(message.id, Some("123".to_string()));
            assert_eq!(message.event, Some("test".to_string()));
            assert_eq!(
                message.comments,
                Some(vec!["This is a comment".to_string()])
            );
            let data: TestData = message.decode_data().unwrap();
            assert_eq!(data.message, "Hello World".to_string());
        } else {
            panic!("Expected a message");
        }
    }

    #[tokio::test]
    async fn test_message_with_only_data() {
        let sample_data = r#"data: {"message": "Just some data"}

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(Ok(message)) = framed.next().await {
            assert!(message.id.is_none());
            assert!(message.event.is_none());
            assert!(message.comments.is_none());
            let data: TestData = message.decode_data().unwrap();
            assert_eq!(data.message, "Just some data".to_string());
        } else {
            panic!("Expected a message");
        }
    }

    #[tokio::test]
    async fn test_message_with_only_comment() {
        let sample_data = r#": This is a comment

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(Ok(message)) = framed.next().await {
            assert!(message.id.is_none());
            assert!(message.event.is_none());
            assert!(message.data.is_none());
            assert_eq!(
                message.comments,
                Some(vec!["This is a comment".to_string()])
            );
        } else {
            panic!("Expected a message");
        }
    }

    #[tokio::test]
    async fn test_message_with_multiple_comments() {
        let sample_data = r#": First comment
: Second comment

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(Ok(message)) = framed.next().await {
            assert!(message.id.is_none());
            assert!(message.event.is_none());
            assert!(message.data.is_none());
            assert_eq!(
                message.comments,
                Some(vec![
                    "First comment".to_string(),
                    "Second comment".to_string()
                ])
            );
        } else {
            panic!("Expected a message");
        }
    }

    #[tokio::test]
    async fn test_message_with_partial_fields() {
        let sample_data = r#"id: 456
data: {"message": "Partial data"}

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(Ok(message)) = framed.next().await {
            assert_eq!(message.id, Some("456".to_string()));
            assert!(message.event.is_none());
            assert!(message.comments.is_none());
            let data: TestData = message.decode_data().unwrap();
            assert_eq!(data.message, "Partial data".to_string());
        } else {
            panic!("Expected a message");
        }
    }

    #[tokio::test]
    async fn test_message_with_invalid_json_data() {
        let sample_data = r#"data: {"message": "Invalid JSON

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(result) = framed.next().await {
            match result {
                Ok(message) => {
                    // got a message, but it has invalid json
                    let data = message.decode_data::<TestData>();
                    assert!(data.is_err(), "Expected an error; got {:?}", data);
                }
                _ => panic!("Expected a message"),
            }
        } else {
            panic!("Expected an error");
        }
    }

    #[tokio::test]
    async fn test_message_with_missing_data_field() {
        let sample_data = r#"id: 789
event: test_event

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(Ok(message)) = framed.next().await {
            assert_eq!(message.id, Some("789".to_string()));
            assert_eq!(message.event, Some("test_event".to_string()));
            assert!(message.data.is_none());
            assert!(message.comments.is_none());
        } else {
            panic!("Expected a message");
        }
    }

    #[tokio::test]
    async fn test_message_with_empty_data_field() {
        let sample_data = r#"data:

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(result) = framed.next().await {
            match result {
                Ok(_) => {
                    panic!("Expected no message");
                }
                Err(e) => panic!("Unexpected error: {}", e),
            }
        } else {
            // no message is emitted
        }
    }

    #[tokio::test]
    async fn test_message_with_multiple_data_lines() {
        let sample_data = r#"data: {"message": "Line1"}
data: {"message": "Line2"}

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(result) = framed.next().await {
            match result {
                Ok(message) => {
                    // got a message with data, but the data is junk
                    let data = message.decode_data::<TestData>();
                    assert!(data.is_err(), "Expected an error; got {:?}", data);
                }
                _ => panic!("Expected a message"),
            }
        } else {
            panic!("Expected an error");
        }
    }

    #[tokio::test]
    async fn test_message_with_unrecognized_field() {
        let sample_data = r#"unknown: value
data: {"message": "Hello"}

"#;
        let cursor = Cursor::new(sample_data);
        let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        if let Some(Ok(message)) = framed.next().await {
            // Unrecognized fields are ignored
            assert!(message.id.is_none());
            assert!(message.event.is_none());
            assert!(message.comments.is_none());
            let data: TestData = message.decode_data().unwrap();
            assert_eq!(data.message, "Hello".to_string());
        } else {
            panic!("Expected a message");
        }
    }

    // data recorded on 2024-09-30 from
    // + curl https://integrate.api.nvidia.com/v1/chat/completions -H 'Content-Type: application/json' \
    //     -H 'Authorization: Bearer nvapi-<redacted>' -d '{
    //     "model": "mistralai/mixtral-8x22b-instruct-v0.1",
    //     "messages": [{"role":"user","content":"Write a limerick about the wonders of GPU computing."}],
    //     "temperature": 0.5,
    //     "top_p": 1,
    //     "max_tokens": 64,
    //     "stream": true
    //   }'
    const SAMPLE_CHAT_DATA: &str = r#"
data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":"assistant","content":null},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"A"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" GPU"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" so"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" swift"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" and"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" so"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" clever"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":","},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"In"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" comput"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"ations"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" it"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"'"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"s"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" quite"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" the"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" ende"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"avor"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"."},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"With"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" its"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" thousands"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" of"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" co"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"res"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":","},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"On"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" complex"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" tasks"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" it"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" ro"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"ars"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":","},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"\n"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"S"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"olving"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" problems"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" like"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" never"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":","},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":" forever"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":"!"},"logprobs":null,"finish_reason":null}]}

data: {"id":"chat-e135180178ae4fe6a7a301aa65aaeea5","object":"chat.completion.chunk","created":1727750141,"model":"mistralai/mixtral-8x22b-instruct-v0.1","choices":[{"index":0,"delta":{"role":null,"content":""},"logprobs":null,"finish_reason":"stop","stop_reason":null}]}

data: [DONE]

"#;

    #[tokio::test]
    async fn test_openai_chat_stream() {
        use crate::protocols::openai::chat_completions::NvCreateChatCompletionStreamResponse;

        // let cursor = Cursor::new(SAMPLE_CHAT_DATA);
        // let mut framed = FramedRead::new(cursor, SseLineCodec::new());

        let mut stream = create_message_stream(SAMPLE_CHAT_DATA);

        let mut counter = 0;

        loop {
            match stream.next().await {
                Some(Ok(message)) => {
                    let delta: NvCreateChatCompletionStreamResponse =
                        serde_json::from_str(&message.data.unwrap()).unwrap();
                    counter += 1;
                    println!("counter: {}", counter);
                    println!("delta: {:?}", delta);
                }
                Some(Err(e)) => {
                    panic!("Error: {:?}", e);
                }
                None => {
                    break;
                }
            }
        }

        assert_eq!(counter, 47);
    }

    #[test]
    fn test_successful_conversion() {
        let message = Message {
            id: Some("123".to_string()),
            event: Some("update".to_string()),
            data: Some(r#"{"message": "Hello World"}"#.to_string()),
            comments: Some(vec!["Some comment".to_string()]),
        };

        let annotated: Annotated<TestData> = message.try_into().unwrap();

        assert_eq!(annotated.id, Some("123".to_string()));
        assert_eq!(annotated.event, Some("update".to_string()));
        assert_eq!(annotated.comment, Some(vec!["Some comment".to_string()]));
        assert_eq!(
            annotated.data,
            Some(TestData {
                message: "Hello World".to_string()
            })
        );
    }

    #[test]
    fn test_error_event_with_comments() {
        let message = Message {
            id: Some("456".to_string()),
            event: Some("error".to_string()),
            data: Some("Error data".to_string()),
            comments: Some(vec!["An error occurred".to_string()]),
        };

        let result: Result<Annotated<TestData>, _> = message.try_into();

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "An error occurred".to_string());
    }

    #[test]
    fn test_error_event_without_comments() {
        let message = Message {
            id: Some("789".to_string()),
            event: Some("error".to_string()),
            data: Some("Error data".to_string()),
            comments: None,
        };

        let result: Result<Annotated<TestData>, _> = message.try_into();

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_json_data() {
        let message = Message {
            id: None,
            event: Some("update".to_string()),
            data: Some("Invalid JSON".to_string()),
            comments: None,
        };

        let result: Result<Annotated<TestData>, _> = message.try_into();

        assert!(result.is_err());
    }

    #[test]
    fn test_missing_data_field() {
        let message = Message {
            id: None,
            event: Some("update".to_string()),
            data: None,
            comments: None,
        };

        let result: Result<Annotated<TestData>, _> = message.try_into();

        assert!(result.is_ok());
        let annotated = result.unwrap();
        assert!(annotated.data.is_none());
        assert_eq!(annotated.event, Some("update".to_string()));
    }
}
