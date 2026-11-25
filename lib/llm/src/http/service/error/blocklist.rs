// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frontend disclosure rules shared by every public protocol renderer.

use dynamo_runtime::error::DynamoError;

use crate::preprocessor::media::PUBLIC_MEDIA_MESSAGES;

/// Categories whose received content must use a generic client message.
#[derive(Clone, Copy)]
enum BlockRule {
    InvalidErrorMetadata,
    TransportErrors,
    PythonProgrammingErrors,
    PrivateMediaFetchDetails,
    XGrammarCompilerLocations,
    PilObjectAddresses,
}

/// Active blocklist; other received messages are forwarded by default.
const PUBLIC_ERROR_BLOCKLIST: &[BlockRule] = &[
    BlockRule::InvalidErrorMetadata,
    BlockRule::TransportErrors,
    BlockRule::PythonProgrammingErrors,
    BlockRule::PrivateMediaFetchDetails,
    BlockRule::XGrammarCompilerLocations,
    BlockRule::PilObjectAddresses,
];

/// Source facts collected across causes before choosing public response text.
#[derive(Default)]
pub(super) struct ErrorFacts {
    pub(super) invalid_metadata: bool,
    transport: bool,
    python_programming: bool,
    media_fetch: bool,
}

impl ErrorFacts {
    /// Record provenance without deciding which message the client receives.
    pub(super) fn record(&mut self, error: &DynamoError) {
        let upstream = error.upstream_error();
        let has_http_response = upstream
            .and_then(|error| error.http_status_value())
            .is_some();
        // Legacy transport reasons apply only when no HTTP response was received.
        self.transport |= !has_http_response
            && matches!(
                error.reason().as_str(),
                "backend.cannot_connect"
                    | "backend.disconnected"
                    | "backend.connection_timeout"
                    | "backend.stream_incomplete"
                    | "transport.cannot_connect"
                    | "transport.disconnected"
                    | "transport.connection_timeout"
            );
        if let Some(upstream) = upstream {
            self.invalid_metadata |= !upstream.metadata_valid();
            self.transport |= upstream.family_name() == Some("transport");
            self.python_programming |= upstream.family_name() == Some("python_programming")
                || matches!(
                    upstream.error_type_name(),
                    Some("builtins.TypeError" | "builtins.AttributeError" | "builtins.KeyError")
                );
            self.media_fetch |= upstream.family_name() == Some("media_fetch")
                || upstream.error_type_name().is_some_and(|name| {
                    name.starts_with("dynamo.common.http.")
                        || name.starts_with("requests.exceptions.")
                });
        }
    }
}

impl BlockRule {
    /// Keep category matching separate from protocol wording and HTTP status.
    fn matches(self, facts: &ErrorFacts, message: Option<&str>) -> bool {
        match self {
            Self::InvalidErrorMetadata => facts.invalid_metadata,
            Self::TransportErrors => facts.transport,
            Self::PythonProgrammingErrors => facts.python_programming,
            Self::PrivateMediaFetchDetails => {
                facts.media_fetch && !message.is_some_and(is_fixed_media_message)
            }
            Self::XGrammarCompilerLocations => message.is_some_and(has_compiler_location),
            Self::PilObjectAddresses => message.is_some_and(has_image_decode_object_address),
        }
    }
}

/// Apply the active rules to the selected message and all captured source facts.
pub(super) fn is_blocked(facts: &ErrorFacts, message: Option<&str>) -> bool {
    PUBLIC_ERROR_BLOCKLIST
        .iter()
        .any(|rule| rule.matches(facts, message))
}

/// Accept only exact, already-public media replacements.
fn is_fixed_media_message(message: &str) -> bool {
    PUBLIC_MEDIA_MESSAGES.contains(&message)
}

// PIL includes a process address in this exact BytesIO diagnostic. Inspect a
// bounded hexadecimal pointer suffix, without masking ordinary image/path text.
fn has_image_decode_object_address(message: &str) -> bool {
    const PREFIX: &str = "cannot identify image file <_io.BytesIO object at 0x";
    message.match_indices(PREFIX).any(|(start, _)| {
        let address = &message.as_bytes()[start + PREFIX.len()..];
        let digits = address
            .iter()
            .take(2 * std::mem::size_of::<usize>())
            .take_while(|byte| byte.is_ascii_hexdigit())
            .count();
        digits > 0 && address.get(digits) == Some(&b'>')
    })
}

// XGrammar can embed this prefix inside a wrapper message. Match bounded text
// rather than treating arbitrary filesystem-looking user input as a diagnostic.
pub(super) fn has_compiler_location(message: &str) -> bool {
    for filename in ["json_schema_converter.cc:", "regex_converter.cc:"] {
        if message.match_indices(filename).any(|(start, _)| {
            let rest = &message[start + filename.len()..];
            rest.split_once(':').is_some_and(|(line, _)| {
                !line.is_empty() && line.bytes().all(|c| c.is_ascii_digit())
            })
        }) {
            return true;
        }
    }
    message.match_indices('[').any(|(start, _)| {
        let bytes = &message.as_bytes()[start..];
        if bytes.len() < 11
            || bytes[3] != b':'
            || bytes[6] != b':'
            || bytes[9] != b']'
            || bytes[10] != b' '
            || ![1, 2, 4, 5, 7, 8]
                .into_iter()
                .all(|i| bytes[i].is_ascii_digit())
        {
            return false;
        }
        let location = message[start + 11..]
            .split_whitespace()
            .next()
            .unwrap_or("");
        location
            .strip_suffix(':')
            .and_then(|s| s.rsplit_once(':'))
            .is_some_and(|(path, line)| {
                !path.is_empty() && !line.is_empty() && line.bytes().all(|c| c.is_ascii_digit())
            })
    })
}
