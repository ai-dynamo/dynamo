// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::error::{DynamoError, ErrorType as DynamoErrorType, PublicDetails};

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

/// Render the client-visible message for a semantic error.
///
/// An explicit public message wins. Otherwise a context-length rejection is
/// rendered in the protocol's own wording; every other class keeps the
/// class-level fallback supplied by the caller.
pub(crate) fn public_message_for(error: &DynamoError, protocol: Protocol) -> Option<String> {
    error
        .public_message()
        .map(str::to_owned)
        .or_else(|| match protocol {
            Protocol::OpenAi => context_length_public_message(error),
            Protocol::Anthropic => anthropic_context_length_public_message(error),
        })
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
