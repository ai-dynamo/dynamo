// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType, PublicDetails};

pub(crate) use super::openai::backend_http_error_class;

mod blocklist;

#[cfg(test)]
use blocklist::has_compiler_location;
use blocklist::{ErrorFacts, is_blocked};

/// Decoded context-length bounds shared by protocol renderers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ContextLengthDetails {
    pub(crate) limit: u64,
    pub(crate) actual: Option<u64>,
    pub(crate) prompt_tokens: Option<u64>,
}

impl ContextLengthDetails {
    pub(crate) fn from_public_details(details: &PublicDetails) -> Option<Self> {
        match details {
            PublicDetails::ContextLength {
                limit,
                actual,
                prompt_tokens,
            } => Some(Self {
                limit: *limit,
                actual: *actual,
                prompt_tokens: *prompt_tokens,
            }),
            _ => None,
        }
    }
}

fn context_length_details(error: &DynamoError) -> Option<ContextLengthDetails> {
    ContextLengthDetails::from_public_details(error.public_details()?)
}

/// Protocols whose clients classify overflows by their own substrings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Protocol {
    OpenAi,
    Anthropic,
}

/// Client-visible fields selected together by the frontend disclosure policy.
pub(crate) struct PublicErrorBody<'a> {
    pub(crate) message: Option<String>,
    pub(crate) details: Option<&'a PublicDetails>,
    pub(crate) context_length: Option<ContextLengthDetails>,
    /// Let renderers retain their selected status's fallback when content is masked.
    pub(crate) blocked: bool,
}

/// Locate the outermost typed error for body-policy traversal.
pub(crate) fn first_dynamo_error<'a>(
    error: &'a (dyn std::error::Error + 'static),
) -> Option<&'a DynamoError> {
    let mut current = Some(error);
    while let Some(error) = current {
        if let Some(error) = error.downcast_ref::<DynamoError>() {
            return Some(error);
        }
        current = error.source();
    }
    None
}

/// Select response content once, independently of status, retry and accounting.
/// Shared across protocols: forward captured content unless the frontend blocklist masks it.
pub(crate) fn public_error_body(error: &DynamoError, protocol: Protocol) -> PublicErrorBody<'_> {
    let mut public_message = None;
    let mut upstream_message = None;
    let mut details = None;
    let mut rendered_context_length = None;
    let mut facts = ErrorFacts::default();
    let mut current: Option<&(dyn std::error::Error + 'static)> = Some(error);
    while let Some(cause) = current {
        if let Some(source) = cause.downcast_ref::<DynamoError>() {
            // Deserialization retains diagnostic causes even after demoting an
            // invalid identity. They must not restore outward-facing content.
            if source.reason().as_str() == "runtime.invalid_error" {
                facts.invalid_metadata = true;
                break;
            }
            if public_message.is_none() {
                public_message = source.public_message().map(str::to_owned);
                if public_message.is_none() {
                    public_message = match protocol {
                        Protocol::OpenAi => context_length_public_message(source),
                        Protocol::Anthropic => anthropic_context_length_public_message(source),
                    };
                    if public_message.is_some() {
                        rendered_context_length = context_length_details(source);
                    }
                }
            }
            if details.is_none() {
                details = source
                    .public_details()
                    .filter(|details| !matches!(details, PublicDetails::Message { .. }));
            }
            if let Some(upstream) = source.upstream_error()
                && !upstream.message().trim().is_empty()
            {
                upstream_message.get_or_insert_with(|| upstream.message().to_owned());
            }
            facts.record(source);
        }
        current = cause.source();
    }
    // Authored replacements win; facts from every cause still drive blocking.
    let candidate = public_message.or(upstream_message);
    if is_blocked(&facts, candidate.as_deref()) {
        let message = match dynamo_error::http::http_action_for_error(error) {
            dynamo_error::http::ClientErrorAction::Respond { public_message, .. } => public_message,
            dynamo_error::http::ClientErrorAction::NoDelivery => "Request cancelled",
        };
        return PublicErrorBody {
            message: Some(message.to_string()),
            details: None,
            context_length: None,
            blocked: true,
        };
    }
    PublicErrorBody {
        message: candidate,
        details,
        context_length: rendered_context_length,
        blocked: false,
    }
}

/// Return only the message selected by the shared body policy.
pub(crate) fn public_message_for(error: &DynamoError, protocol: Protocol) -> Option<String> {
    public_error_body(error, protocol).message
}

/// Render the client-visible message for a context-length rejection.
///
/// The preprocessor producers capture [`PublicDetails::ContextLength`] instead
/// of a public message, so the protocol boundary owns the wording. The sentence
/// must contain `maximum context length is` so clients that classify overflows by
/// substring (LiteLLM's `is_error_str_context_window_exceeded`) recognize the
/// rejection; the class-level fallback ("Invalid request") does not.
///
/// Do not move this wording into the producers with
/// [`DynamoError::public_message`]: that setter and `.public_details(...)` assign
/// the same `public` field (last writer wins), so the structured `limit`/`actual`
/// would be erased and this helper would no longer see them.
///
/// Returns `None` when the error is not a context-length rejection or the
/// producer did not record the actual count, in which case the caller keeps its
/// class-level fallback.
pub(crate) fn context_length_public_message(error: &DynamoError) -> Option<String> {
    openai_context_length_message(context_length_details(error)?)
}

/// Render the OpenAI context-length sentence from structured bounds.
///
/// The prompt and the requested output are named separately when both are
/// known: `actual` includes the requested output, so calling it the size of the
/// messages would blame the prompt for a budget the caller can still lower. A
/// prompt that fills the limit is reported as leaving no room for output rather
/// than as an excess, because lowering the output budget cannot help.
pub(crate) fn openai_context_length_message(details: ContextLengthDetails) -> Option<String> {
    let actual = details.actual.filter(|actual| *actual >= details.limit)?;
    let limit = details.limit;

    if let Some(prompt_tokens) = details.prompt_tokens {
        if prompt_tokens >= limit {
            return Some(if prompt_tokens == limit {
                format!(
                    "This model's maximum context length is {limit} tokens. However, your \
                     messages resulted in {prompt_tokens} tokens, which leaves no room for \
                     output. Please reduce the length of the messages.",
                )
            } else {
                format!(
                    "This model's maximum context length is {limit} tokens. However, your \
                     messages resulted in {prompt_tokens} tokens. Please reduce the length of \
                     the messages.",
                )
            });
        }

        let output = requested_output_phrase(actual.saturating_sub(prompt_tokens));
        return Some(if actual == limit {
            format!(
                "This model's maximum context length is {limit} tokens. However, your request \
                 would use {prompt_tokens} prompt tokens plus {output}, which leaves no room for \
                 output. Please reduce the length of the messages or the requested output.",
            )
        } else {
            format!(
                "This model's maximum context length is {limit} tokens. However, your request \
                 would use {prompt_tokens} prompt tokens plus {output} ({actual} tokens total). \
                 Please reduce the length of the messages or the requested output.",
            )
        });
    }

    // Without a prompt count, the total is all that can be reported.
    Some(if actual == limit {
        format!(
            "This model's maximum context length is {limit} tokens. However, your request \
             resulted in {actual} tokens, which leaves no room for output. Please reduce the \
             length of the messages.",
        )
    } else {
        format!(
            "This model's maximum context length is {limit} tokens. However, your request \
             resulted in {actual} tokens. Please reduce the length of the messages or the \
             requested output.",
        )
    })
}

/// Anthropic's wording for a context-length rejection.
///
/// Anthropic clients classify prompt overflows by `prompt is too long`, while
/// LiteLLM's generic classifier looks for `this model's maximum context length
/// is`. Prompt-only messages carry both. A prompt exactly at the limit is
/// reported as leaving no room for output rather than as an excess. A request
/// rejected only because prompt tokens plus the requested output exceed the
/// limit is described as a request total, without `prompt is too long`, so
/// clients do not compact history when lowering `max_tokens` is sufficient.
///
/// Returns `None` when the error is not a context-length rejection, the
/// producer did not record the actual count, or the recorded count is below the
/// limit (which neither producer should produce), in which case the caller
/// keeps its class-level fallback.
pub(crate) fn anthropic_context_length_public_message(error: &DynamoError) -> Option<String> {
    anthropic_context_length_message(context_length_details(error)?)
}

/// Render Anthropic's context-length sentence from structured bounds.
///
/// Returns `None` for a count below the limit or a missing actual count, which
/// neither producer should emit; callers keep their class-level fallback instead
/// of claiming an overflow that did not happen.
pub(crate) fn anthropic_context_length_message(details: ContextLengthDetails) -> Option<String> {
    let actual = details.actual?;
    if actual < details.limit {
        return None;
    }

    if let Some(prompt_tokens) = details
        .prompt_tokens
        .filter(|count| *count >= details.limit)
    {
        return Some(if prompt_tokens == details.limit {
            format!(
                "prompt is too long: this model's maximum context length is {limit} tokens and \
                 the prompt contains {prompt_tokens} tokens, which leaves no room for output. \
                 Please reduce the length of the messages.",
                limit = details.limit,
            )
        } else {
            format!(
                "prompt is too long: this model's maximum context length is {limit} tokens, but \
                 the prompt contains {prompt_tokens} tokens. Please reduce the length of the \
                 messages.",
                limit = details.limit,
            )
        });
    }

    let budget = match details.prompt_tokens {
        Some(prompt_tokens) => format!(
            "the request would use {prompt_tokens} prompt tokens plus {output} ({actual} tokens \
             total)",
            output = requested_output_phrase(actual.saturating_sub(prompt_tokens)),
        ),
        None => format!("the request would use {actual} tokens"),
    };
    Some(format!(
        "this model's maximum context length is {limit} tokens. However, {budget}. Please \
         reduce the length of the messages or the requested output.",
        limit = details.limit,
    ))
}

/// Name the output part of a combined budget.
///
/// A single token is what the preprocessor assumes when `max_tokens` is omitted,
/// so it must not read as a value the client asked for. The wording is still
/// accurate when the client explicitly requested one token.
fn requested_output_phrase(output_tokens: u64) -> String {
    if output_tokens == 1 {
        "at least one output token".to_string()
    } else {
        format!("{output_tokens} requested output tokens")
    }
}

/// Construct a typed invalid-argument error for validation performed at an
/// HTTP protocol adapter boundary.
pub(crate) fn invalid_argument(message: impl Into<String>) -> DynamoError {
    let message = message.into();
    DynamoError::builder()
        .error_type(DynamoErrorType::InvalidArgument)
        .message(message.clone())
        .public_message(message)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessor::media::{
        IMAGE_FETCH_FORBIDDEN_MESSAGE, IMAGE_FETCH_NOT_FOUND_MESSAGE, PUBLIC_MEDIA_MESSAGES,
    };
    use dynamo_runtime::error::{BackendError, ErrorClass, UpstreamError};

    #[test]
    fn received_messages_forward_unless_a_public_replacement_exists() {
        let upstream = UpstreamError::new("Unknown engine rejection: reduce batch size")
            .error_type("engine.NewError");
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Unknown))
            .upstream_error(upstream.clone())
            .diagnostic("private RPC wrapper")
            .build();
        assert_eq!(
            public_message_for(&error, Protocol::OpenAi).as_deref(),
            Some(upstream.message())
        );
        for message in ["", " \n\t"] {
            let empty = DynamoError::builder()
                .upstream_error(UpstreamError::new(message))
                .build();
            assert_eq!(public_message_for(&empty, Protocol::OpenAi), None);
        }
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(upstream.clone())
            .public_message("A safe authored rejection")
            .build();
        assert_eq!(
            public_message_for(&error, Protocol::Anthropic).as_deref(),
            Some("A safe authored rejection")
        );
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(upstream)
            .public_details(PublicDetails::ContextLength {
                limit: 10,
                actual: Some(20),
                prompt_tokens: None,
            })
            .build();
        let body = public_error_body(&error, Protocol::OpenAi);
        assert!(
            body.message
                .unwrap()
                .contains("maximum context length is 10 tokens")
        );
        assert!(body.details.is_some());
        let overridden = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .public_message("Authored replacement")
            .cause(error)
            .build();
        let body = public_error_body(&overridden, Protocol::OpenAi);
        assert_eq!(body.message.as_deref(), Some("Authored replacement"));
        assert_eq!(body.context_length, None);
    }

    #[test]
    fn malformed_received_message_decodes_and_masks_all_response_content() {
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new("received rejection"))
            .public_details(PublicDetails::ContextLength {
                limit: 10,
                actual: Some(20),
                prompt_tokens: None,
            })
            .build();
        let valid_wire = serde_json::to_value(error).unwrap();
        for message in [None, Some(serde_json::Value::Null), Some(7.into())] {
            let mut wire = valid_wire.clone();
            let upstream = wire["upstream_error"].as_object_mut().unwrap();
            match message {
                Some(message) => {
                    upstream.insert("message".to_string(), message);
                }
                None => {
                    upstream.remove("message");
                }
            }
            let received: DynamoError = serde_json::from_value(wire).unwrap();
            assert_eq!(received.class(), ErrorClass::InvalidRequest);
            for protocol in [Protocol::OpenAi, Protocol::Anthropic] {
                let body = public_error_body(&received, protocol);
                assert!(body.blocked);
                assert_eq!(body.message.as_deref(), Some("Invalid request"));
                assert!(body.details.is_none());
                assert!(body.context_length.is_none());
            }
        }
    }

    #[test]
    fn rules_mask_all_response_content_and_public_copies() {
        let cases = [
            UpstreamError::new("http://private-host/path: refused").family("media_fetch"),
            UpstreamError::new("socket 10.0.0.1 failed").family("transport"),
            UpstreamError::new("handler has no attribute secret").family("python_programming"),
            UpstreamError::new("wrapper [12:34:56] /project/cpp/schema.cc:83: source detail"),
            UpstreamError::new(
                "Guided decoding error: /project/cpp/json_schema_converter.cc:83: Unsupported type",
            ),
            UpstreamError::new("malformed source metadata").error_type("x".repeat(129)),
        ];
        for upstream in cases {
            for public_copy in [false, true] {
                let mut builder = DynamoError::builder()
                    .class(ErrorClass::InvalidRequest)
                    .upstream_error(upstream.clone());
                builder = if public_copy {
                    builder.public_message(upstream.message())
                } else {
                    builder.public_details(PublicDetails::ContextLength {
                        limit: 10,
                        actual: Some(20),
                        prompt_tokens: None,
                    })
                };
                let error = builder.build();
                let body = public_error_body(&error, Protocol::OpenAi);
                // Compiler locations are checked on the selected candidate;
                // authored context wording safely replaces a raw compiler log.
                if !public_copy && has_compiler_location(upstream.message()) {
                    assert!(body.message.unwrap().contains("maximum context length"));
                    continue;
                }
                assert_eq!(body.message.as_deref(), Some("Invalid request"));
                assert!(body.details.is_none());
            }
        }
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::Backend(BackendError::Disconnected))
            .upstream_error(UpstreamError::new("RPC socket detail"))
            .build();
        assert_eq!(
            public_message_for(&error, Protocol::OpenAi).as_deref(),
            Some("Service temporarily unavailable")
        );
    }

    #[test]
    fn actual_http_responses_do_not_acquire_legacy_transport_obfuscation() {
        for status in [502, 503, 504] {
            for explicit_transport in [false, true] {
                let mut upstream = UpstreamError::new("engine is temporarily busy")
                    .error_type("sglang.http.Response")
                    .http_status(status);
                if explicit_transport {
                    upstream = upstream.family("transport");
                }
                let error = DynamoError::builder()
                    .error_type(DynamoErrorType::Backend(BackendError::CannotConnect))
                    .upstream_error(upstream)
                    .public_details(PublicDetails::RateLimit {
                        limit: Some(10),
                        remaining: Some(0),
                    })
                    .build();
                for protocol in [Protocol::OpenAi, Protocol::Anthropic] {
                    let body = public_error_body(&error, protocol);
                    assert_eq!(
                        body.message.as_deref(),
                        Some(if explicit_transport {
                            "Service temporarily unavailable"
                        } else {
                            "engine is temporarily busy"
                        })
                    );
                    assert_eq!(body.details.is_none(), explicit_transport);
                }
            }
        }
    }

    #[test]
    fn media_rule_preserves_only_existing_fixed_replacements() {
        for message in [
            "Unsupported video media",
            "Image exceeds the decoded pixel limit",
            IMAGE_FETCH_FORBIDDEN_MESSAGE,
            IMAGE_FETCH_NOT_FOUND_MESSAGE,
        ] {
            let error = DynamoError::builder()
                .class(ErrorClass::InvalidRequest)
                .upstream_error(UpstreamError::new("unsafe remote reason").family("media_fetch"))
                .public_message(message)
                .build();
            assert_eq!(
                public_message_for(&error, Protocol::OpenAi).as_deref(),
                Some(message)
            );
        }
        let error = DynamoError::builder()
            .class(ErrorClass::InvalidRequest)
            .upstream_error(UpstreamError::new("fetch URL leaked").family("media_fetch"))
            .public_message(format!("{IMAGE_FETCH_FORBIDDEN_MESSAGE} private-host"))
            .build();
        assert_eq!(
            public_message_for(&error, Protocol::OpenAi).as_deref(),
            Some("Invalid request")
        );
    }

    #[test]
    fn requests_failures_mask_urls_and_keep_only_fixed_media_replacements() {
        let received =
            "403 Client Error for url: https://PRIVATE_FETCH_URL/image?token=PRIVATE_TOKEN";
        for kind in [
            "requests.exceptions.HTTPError",
            "requests.exceptions.ConnectionError",
            "requests.exceptions.Timeout",
        ] {
            for replacement in [None, Some(received), Some("An authored fetch replacement")] {
                let mut builder = DynamoError::builder()
                    .class(ErrorClass::InvalidRequest)
                    .upstream_error(
                        UpstreamError::new(received)
                            .error_type(kind)
                            .http_status(403),
                    )
                    .public_details(PublicDetails::ContextLength {
                        limit: 10,
                        actual: Some(20),
                        prompt_tokens: None,
                    });
                if let Some(message) = replacement {
                    builder = builder.public_message(message);
                }
                let error = builder.build();
                for protocol in [Protocol::OpenAi, Protocol::Anthropic] {
                    let body = public_error_body(&error, protocol);
                    assert_eq!(body.message.as_deref(), Some("Invalid request"));
                    assert!(body.details.is_none());
                    assert!(body.context_length.is_none());
                }
            }
        }
        for replacement in PUBLIC_MEDIA_MESSAGES {
            let error = DynamoError::builder()
                .class(ErrorClass::InvalidRequest)
                .upstream_error(
                    UpstreamError::new(received).error_type("requests.exceptions.HTTPError"),
                )
                .public_message(*replacement)
                .build();
            for protocol in [Protocol::OpenAi, Protocol::Anthropic] {
                assert_eq!(
                    public_message_for(&error, protocol).as_deref(),
                    Some(*replacement)
                );
            }
        }
    }

    #[test]
    fn pil_image_decode_addresses_mask_only_the_known_selected_diagnostic() {
        let received =
            "Could not decode image: cannot identify image file <_io.BytesIO object at 0xfc123a>";
        for public_copy in [false, true] {
            let mut builder = DynamoError::builder()
                .class(ErrorClass::InvalidRequest)
                .upstream_error(UpstreamError::new(received).error_type("builtins.ValueError"))
                .public_details(PublicDetails::SizeLimit {
                    limit: 10,
                    actual: Some(20),
                });
            if public_copy {
                builder = builder.public_message(received);
            }
            let error = builder.build();
            for protocol in [Protocol::OpenAi, Protocol::Anthropic] {
                let body = public_error_body(&error, protocol);
                assert_eq!(body.message.as_deref(), Some("Invalid request"));
                assert!(body.details.is_none());
                assert!(body.context_length.is_none());
            }
        }
        for selected in [
            "Image could not be decoded; send a supported image.",
            "Could not decode image: cannot identify image file <_io.BytesIO object at 0xnot_hex>",
            "Could not decode image: cannot identify image file <_io.BytesIO object at 0x>",
            "Could not decode image: cannot identify image file <_io.BytesIO object at 0xfc123a",
            "User supplied <_io.BytesIO object at 0xfc123a> as text",
            "Could not decode image: use /models/input.png",
        ] {
            for authored_override in [false, true] {
                let mut builder = DynamoError::builder()
                    .class(ErrorClass::InvalidRequest)
                    .upstream_error(
                        UpstreamError::new(if authored_override {
                            received
                        } else {
                            selected
                        })
                        .error_type("builtins.ValueError"),
                    );
                if authored_override {
                    builder = builder.public_message(selected);
                }
                let error = builder.build();
                for protocol in [Protocol::OpenAi, Protocol::Anthropic] {
                    assert_eq!(
                        public_message_for(&error, protocol).as_deref(),
                        Some(selected)
                    );
                }
            }
        }
    }

    #[test]
    fn context_length_messages_refuse_a_count_below_the_limit() {
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::InvalidArgument)
            .public_details(PublicDetails::ContextLength {
                limit: 10,
                actual: Some(5),
                prompt_tokens: Some(5),
            })
            .build();

        assert_eq!(context_length_public_message(&error), None);
        assert_eq!(anthropic_context_length_public_message(&error), None);
    }

    #[test]
    fn context_length_message_at_the_limit_reports_no_room_for_output() {
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::InvalidArgument)
            .public_details(PublicDetails::ContextLength {
                limit: 64,
                actual: Some(64),
                prompt_tokens: Some(64),
            })
            .build();

        assert_eq!(
            context_length_public_message(&error).as_deref(),
            Some(
                "This model's maximum context length is 64 tokens. However, your messages \
                 resulted in 64 tokens, which leaves no room for output. Please reduce the \
                 length of the messages."
            )
        );
    }

    /// `actual` includes the output budget, so the assumed single output token
    /// must not read as a value the client requested. The prompt is one token
    /// under the limit, so the single output token exactly fills the budget.
    #[test]
    fn context_length_message_single_output_token_is_not_an_explicit_request() {
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::InvalidArgument)
            .public_details(PublicDetails::ContextLength {
                limit: 8192,
                actual: Some(8192),
                prompt_tokens: Some(8191),
            })
            .build();

        assert_eq!(
            context_length_public_message(&error).as_deref(),
            Some(
                "This model's maximum context length is 8192 tokens. However, your request \
                 would use 8191 prompt tokens plus at least one output token, which leaves no \
                 room for output. Please reduce the length of the messages or the requested \
                 output."
            )
        );
        let anthropic = anthropic_context_length_public_message(&error).expect("anthropic wording");
        assert!(
            anthropic.contains("at least one output token"),
            "got: {anthropic}"
        );
        assert!(
            !anthropic.contains("1 requested output token"),
            "omitted max_tokens must not be reported as requested, got: {anthropic}"
        );
    }

    #[test]
    fn public_message_for_prefers_an_explicit_public_message() {
        // Only the explicit public message: `.public_details(...)` and
        // `.public_message(...)` write the same field, last writer wins, so
        // combining them here would erase the message under test.
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::InvalidArgument)
            .public_message("explicit")
            .build();

        assert_eq!(
            public_message_for(&error, Protocol::OpenAi).as_deref(),
            Some("explicit")
        );
        assert_eq!(
            public_message_for(&error, Protocol::Anthropic).as_deref(),
            Some("explicit")
        );
    }

    #[test]
    fn public_message_for_keeps_the_class_fallback_for_other_classes() {
        let error = DynamoError::builder()
            .error_type(DynamoErrorType::InvalidArgument)
            .build();

        assert_eq!(public_message_for(&error, Protocol::OpenAi), None);
        assert_eq!(public_message_for(&error, Protocol::Anthropic), None);
    }
}
