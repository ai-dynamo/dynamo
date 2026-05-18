// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::super::ToolDefinition;
use super::config::JsonParserConfig;
use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use openai_harmony::chat::{Content::Text, Role};
use openai_harmony::{HarmonyEncoding, HarmonyEncodingName, load_harmony_encoding};
use regex::{Captures, Regex};
use serde_json::Value;
use std::sync::OnceLock;

static COMMENTARY_BLOCK_REGEX: OnceLock<Regex> = OnceLock::new();
static COMMENTARY_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static COMMENTARY_HEADER_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static ANALYSIS_BLOCK_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static MESSAGE_CALL_CLEANUP_REGEX: OnceLock<Regex> = OnceLock::new();
static SPECIAL_TOKEN_REGEX: OnceLock<Regex> = OnceLock::new();

/// Regex fallback used only when `openai_harmony`'s tokenizer rejects the
/// input — alternative on this path is silent-drop. Worst case is missing
/// a call, never fabricating one: Harmony tool calls must end with `<|call|>`.
fn commentary_block_regex() -> &'static Regex {
    COMMENTARY_BLOCK_REGEX.get_or_init(|| {
        // Name is `[\w.\-]+` (alphanumeric / dot / hyphen / underscore).
        // Between name and `<|message|>` we tolerate optional
        // `<|constrain|>json` and whitespace by using non-greedy `.*?`.
        // Args must end at `<|call|>`, the required Harmony tool-call stop token.
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary to=functions\.(?P<name>[\w.\-]+).*?<\|message\|>(?P<args>.*?)<\|call\|>",
        )
        .expect("commentary block regex")
    })
}

fn commentary_block_cleanup_regex() -> &'static Regex {
    COMMENTARY_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary(?:\s+to=functions\.(?P<name>[\w.\-]+))?.*?<\|message\|>.*?(?:<\|call\|>|\z)",
        )
        .expect("commentary block cleanup regex")
    })
}

fn commentary_header_cleanup_regex() -> &'static Regex {
    COMMENTARY_HEADER_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(
            r"(?s)(?:<\|start\|>assistant)?<\|channel\|>commentary(?:\s+to=functions\.(?P<name>[\w.\-]+))?.*\z",
        )
        .expect("commentary header cleanup regex")
    })
}

fn analysis_block_cleanup_regex() -> &'static Regex {
    ANALYSIS_BLOCK_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(r"(?s)(?:<\|start\|>assistant)?<\|channel\|>analysis<\|message\|>(?P<body>.*?)(?:<\|end\|>|\z)")
            .expect("analysis block cleanup regex")
    })
}

fn message_call_cleanup_regex() -> &'static Regex {
    MESSAGE_CALL_CLEANUP_REGEX.get_or_init(|| {
        Regex::new(r"(?s)<\|message\|>.*?(?:<\|call\|>|\z)").expect("message call cleanup regex")
    })
}

fn special_token_regex() -> &'static Regex {
    SPECIAL_TOKEN_REGEX.get_or_init(|| {
        Regex::new(r"<\|(?:start|channel|constrain|message|call|end|return)\|>")
            .expect("special token cleanup regex")
    })
}

fn push_unique(items: &mut Vec<String>, item: String) {
    if !items.iter().any(|existing| existing == &item) {
        items.push(item);
    }
}

fn record_special_tokens(text: &str, items: &mut Vec<String>) {
    for m in special_token_regex().find_iter(text) {
        push_unique(items, format!("special_token:{}", m.as_str()));
    }
}

fn strip_harmony_protocol_from_normal_text(text: &str, reason: &'static str) -> String {
    let mut stripped = Vec::new();

    let cleaned = commentary_block_cleanup_regex()
        .replace_all(text, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            let item = match caps.name("name").map(|m| m.as_str()) {
                Some(name) => format!("commentary_tool_call:functions.{name}"),
                None => "commentary_tool_call:missing_recipient".to_string(),
            };
            push_unique(&mut stripped, item);
            ""
        })
        .into_owned();

    let cleaned = commentary_header_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            let item = match caps.name("name").map(|m| m.as_str()) {
                Some(name) => format!("commentary_tool_call_without_message:functions.{name}"),
                None => "commentary_tool_call_without_message:missing_recipient".to_string(),
            };
            push_unique(&mut stripped, item);
            ""
        })
        .into_owned();

    let cleaned = analysis_block_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            push_unique(&mut stripped, "analysis_envelope".to_string());
            ""
        })
        .into_owned();

    let cleaned = message_call_cleanup_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            record_special_tokens(&caps[0], &mut stripped);
            push_unique(&mut stripped, "message_call_payload".to_string());
            ""
        })
        .into_owned();

    let cleaned = special_token_regex()
        .replace_all(&cleaned, |caps: &Captures<'_>| {
            push_unique(&mut stripped, format!("special_token:{}", &caps[0]));
            ""
        })
        .into_owned();

    if stripped.is_empty() {
        return text.to_string();
    }

    let cleaned = cleaned.trim().to_string();

    tracing::warn!(
        family = "harmony",
        reason,
        stripped = ?stripped,
        original_len = text.len(),
        cleaned_len = cleaned.len(),
        "stripped harmony protocol content from normal_text"
    );

    cleaned
}

fn normal_text_after_parse_failure(text: &str, reason: &'static str) -> String {
    let cleaned = strip_harmony_protocol_from_normal_text(text, reason);
    if cleaned == text && !text.trim().is_empty() {
        tracing::warn!(
            family = "harmony",
            reason,
            original_len = text.len(),
            "dropped bare text without a Harmony final/commentary message"
        );
        String::new()
    } else {
        cleaned
    }
}

fn serialize_harmony_arguments(raw_args: &str) -> String {
    let trimmed = raw_args.trim();
    match serde_json::from_str::<Value>(trimmed) {
        Ok(value) => serde_json::to_string(&value).unwrap_or_else(|_| trimmed.to_string()),
        Err(_) => trimmed.to_string(),
    }
}

/// Extract calls via regex when harmony's strict tokenizer rejects the input
/// (truncated JSON, multiple back-to-back commentary blocks, etc.).
/// Returns (calls, residual_text) where residual_text is everything not
/// consumed by a matched commentary block — preserved so non-tool user-visible
/// spans aren't dropped.
fn extract_calls_via_regex(text: &str) -> (Vec<ToolCallResponse>, String) {
    let mut out = Vec::new();
    let mut residual = String::new();
    let mut cursor = 0;
    for cap in commentary_block_regex().captures_iter(text) {
        let m = cap.get(0).expect("regex match has full span");
        residual.push_str(&text[cursor..m.start()]);
        cursor = m.end();

        let name = cap.name("name").map(|x| x.as_str()).unwrap_or("");
        let raw_args = cap.name("args").map(|x| x.as_str().trim()).unwrap_or("{}");
        if name.is_empty() {
            continue;
        }
        let args_json = serialize_harmony_arguments(raw_args);
        let call_idx = out.len() + 1;
        out.push(ToolCallResponse {
            id: format!("call-{call_idx}"),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                arguments: args_json,
            },
        });
    }
    residual.push_str(&text[cursor..]);
    (out, residual.trim().to_string())
}

static GLOBAL_HARMONY_GPTOSS_ENCODING: tokio::sync::OnceCell<
    Result<HarmonyEncoding, anyhow::Error>,
> = tokio::sync::OnceCell::const_new();

pub async fn get_harmony_encoding() -> &'static Result<HarmonyEncoding, anyhow::Error> {
    GLOBAL_HARMONY_GPTOSS_ENCODING
        .get_or_init(|| async {
            tokio::task::spawn_blocking(|| {
                load_harmony_encoding(HarmonyEncodingName::HarmonyGptOss)
            })
            .await
            .map_err(anyhow::Error::msg)
            .flatten()
        })
        .await
}

/// Parse tool calls from a complete Harmony Format text chunk using direct token parsing.
///
/// This function is optimized for parsing complete text chunks where the entire content
/// is available at once. It uses `parse_messages_from_completion_tokens` to directly
/// parse all tokens into Harmony Format messages, then extracts tool calls from messages
/// with the "commentary" channel and "functions.*" recipients.
///
/// This function doesn't perform start token detection
/// or token-by-token streaming, making it more efficient for complete chunks.
///
/// # Arguments
/// * `text` - The full Harmony-format string to be parsed, including required Harmony stop tokens.
///   Example:
///   `<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>`
/// * `_config` - Parser configuration (currently unused but kept for API consistency)
///
/// # Returns
/// * `Ok((tool_calls, normal_text))` - Tuple containing extracted tool calls and any normal text
/// * `Err(e)` - If parsing fails due to encoding or tokenization errors
pub async fn parse_tool_calls_harmony_complete(
    text: &str,
    _config: &JsonParserConfig,
    _tools: Option<&[ToolDefinition]>,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let enc = match get_harmony_encoding().await.as_ref() {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!("Failed to load harmony encoding: {e}. Tool calls will not be parsed.");
            let normal_text = normal_text_after_parse_failure(text, "harmony_encoding_unavailable");
            return Ok((vec![], Some(normal_text)));
        }
    };

    // // Encode the text into tokens using harmony encoding
    let tokens: Vec<u32> = enc.tokenizer().encode_with_special_tokens(text);
    let messages = match enc.parse_messages_from_completion_tokens(tokens, Some(Role::Assistant)) {
        Ok(messages) => messages,
        Err(e) => {
            tracing::debug!(
                "Failed to parse messages from completion tokens: {e}. Falling back to regex extraction."
            );
            // Recovery: harmony rejects parallel commentary blocks even when
            // every call is explicitly closed. Only EOF/truncated recovery is
            // gated, so streaming jails do not synthesize incomplete calls.
            // Harmony differs from generic JSON/XML recovery: a tool call is
            // complete only once `<|call|>` is present. Do not synthesize a
            // call from EOF; that would diverge from vLLM/openai_harmony.
            let (calls, residual) = extract_calls_via_regex(text);
            if !calls.is_empty() {
                let normal_text =
                    strip_harmony_protocol_from_normal_text(&residual, "regex_recovery_residual");
                return Ok((calls, Some(normal_text)));
            }
            let normal_text =
                normal_text_after_parse_failure(text, "parse_failed_no_recovered_calls");
            return Ok((vec![], Some(normal_text)));
        }
    };

    let mut res = Vec::with_capacity(messages.len());
    let mut call_idx = 0; // Index of the tool call
    let mut final_text: Option<String> = None;
    let mut commentary_text: Option<String> = None;
    let has_tool_call_stop = text.contains("<|call|>");

    for message in messages.iter() {
        if message.author.role != Role::Assistant {
            continue;
        }

        let channel = message.channel.as_deref();
        let recipient = message.recipient.as_deref().unwrap_or_default();

        // Handle commentary channel
        if channel == Some("commentary") && recipient.starts_with("functions.") {
            if !has_tool_call_stop {
                continue;
            }

            let Some(fname) = message
                .recipient
                .as_ref()
                .and_then(|r| r.split('.').nth(1))
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
            else {
                continue;
            };

            let Some(args_json) = message.content.first().and_then(|content| match content {
                Text(text) => Some(serialize_harmony_arguments(&text.text)),
                _ => None,
            }) else {
                continue;
            };

            call_idx += 1;
            res.push(ToolCallResponse {
                id: format!("call-{}", call_idx),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: fname.to_string(),
                    arguments: args_json,
                },
            });
        } else if channel == Some("final") {
            if let Some(Text(text)) = message.content.first() {
                final_text = Some(text.text.clone());
            }
        } else if channel == Some("commentary")
            && recipient.is_empty()
            && let Some(Text(text)) = message.content.first()
        {
            commentary_text = Some(text.text.clone());
        }
    }
    let normal_text = final_text
        .filter(|text| !text.is_empty())
        .or_else(|| commentary_text.filter(|text| !text.is_empty()))
        .unwrap_or_default();
    Ok((res, Some(normal_text)))
}

pub fn detect_tool_call_start_harmony(
    chunk: &str,
    config: &JsonParserConfig,
    strict: bool,
) -> bool {
    let trimmed = chunk.trim();
    if trimmed.is_empty() {
        return false;
    }

    if strict {
        // Check for complete start tokens first
        let has_complete_token = config
            .tool_call_start_tokens
            .iter()
            .any(|token| !token.is_empty() && trimmed.contains(token));

        if has_complete_token {
            return true;
        }

        // Check for partial start tokens (streaming scenario)
        // This handles cases where start tokens are split across multiple chunks
        config.tool_call_start_tokens.iter().any(|token| {
            if token.is_empty() {
                return false;
            }
            // Check if the chunk could be a prefix of this start token
            // Handle Unicode character boundaries properly
            for i in 1..=token.chars().count() {
                if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                    let prefix_str = &prefix[..prefix.len()];
                    if trimmed == prefix_str || trimmed.ends_with(prefix_str) {
                        return true;
                    }
                }
            }
            false
        })
    } else {
        // Non-strict mode: check complete tokens and some heuristics
        let has_complete_token = config
            .tool_call_start_tokens
            .iter()
            .any(|token| !token.is_empty() && trimmed.contains(token));

        if has_complete_token {
            return true;
        }

        // Check for partial start tokens or known patterns
        let has_partial_token = config.tool_call_start_tokens.iter().any(|token| {
            if token.is_empty() {
                return false;
            }
            // Check if the chunk could be a prefix of this start token
            // Handle Unicode character boundaries properly
            for i in 1..=token.chars().count() {
                if let Some(prefix) = token.chars().take(i).collect::<String>().get(..) {
                    let prefix_str = &prefix[..prefix.len()];
                    if trimmed == prefix_str || trimmed.ends_with(prefix_str) {
                        return true;
                    }
                }
            }
            false
        });

        has_partial_token || trimmed.contains("<|channel|>")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[tokio::test] // PARSER.batch.1, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_complete_basic() {
        let text = r#"<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"format":"celsius","location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["format"], "celsius");
    }

    #[tokio::test] // PARSER.batch.4, PARSER.harmony.2
    async fn test_parse_tools_harmony_without_start_token() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 0);
    }

    #[tokio::test] // PARSER.batch.7, PARSER.batch.8, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_with_multi_args() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco", "unit":"fahrenheit"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
        assert_eq!(args["unit"], "fahrenheit");
    }

    #[tokio::test] // PARSER.batch.8, PARSER.batch.8, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_with_normal_text() {
        let text = r#"<|channel|>analysis<|message|>Need to use function get_current_weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"San Francisco"}<|call|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "get_current_weather");
        assert_eq!(args["location"], "San Francisco");
    }

    #[tokio::test] // PARSER.batch.3 — gpt-oss
    async fn test_parse_harmony_bare_text_without_final_message_is_dropped() {
        let text = "Hello, how can I help you today?";
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert!(tool_calls.is_empty());
        assert_eq!(normal_content, Some("".to_string()));
    }

    #[tokio::test] // PARSER.batch.3 — gpt-oss
    async fn test_parse_harmony_final_message_returns_normal_text() {
        let text = r#"<|channel|>final<|message|>Hello, how can I help you today?<|return|>"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert!(tool_calls.is_empty());
        assert_eq!(
            normal_content,
            Some("Hello, how can I help you today?".to_string())
        );
    }

    // Harmony's strict tokenizer rejects two back-to-back commentary
    // blocks, so EOF recovery falls back to regex extraction and pins that
    // both calls are surfaced without leaking the raw envelopes.
    #[tokio::test] // PARSER.batch.2 — gpt-oss
    async fn test_parse_harmony_multiple_calls_recovers() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x":1}<|call|><|start|>assistant<|channel|>commentary to=functions.b <|constrain|>json<|message|>{"y":2}<|call|>"#;
        let (tool_calls, _normal) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(tool_calls.len(), 2);
        let (n0, a0) = extract_name_and_args(tool_calls[0].clone());
        let (n1, a1) = extract_name_and_args(tool_calls[1].clone());
        assert_eq!(n0, "a");
        assert_eq!(a0["x"], 1);
        assert_eq!(n1, "b");
        assert_eq!(a1["y"], 2);
    }

    #[tokio::test] // PARSER.batch.4 — gpt-oss
    async fn test_parse_harmony_malformed_json_preserves_raw_arguments() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>not json at all<|call|>"#;
        let (tool_calls, _normal) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, "not json at all");
    }

    #[tokio::test] // PARSER.batch.4 — gpt-oss
    async fn test_parse_harmony_unterminated_json_preserves_raw_arguments() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC<|call|>"#;
        let (tool_calls, _normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(tool_calls[0].function.arguments, r#"{"location":"NYC"#);
    }

    // Bare-envelope PARSER.batch.5: no preceding `analysis` block, no `<|call|>`
    // at the end. Harmony requires the explicit call stop token, so fallback
    // must not accept EOS as a synthetic close.
    #[tokio::test] // PARSER.batch.5 — gpt-oss
    async fn test_parse_harmony_bare_envelope_no_call_token_drops() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"NYC"}"#;
        let (tool_calls, normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert!(tool_calls.is_empty());
        assert_eq!(normal, Some("".to_string()));
    }

    // The regex fallback must preserve any non-tool spans (prose before
    // the call, suffix after `<|call|>`) as `normal_text`, not zero them.
    #[tokio::test]
    async fn test_parse_harmony_regex_fallback_preserves_residual_text() {
        let text = r#"PREFIX <|start|>assistant<|channel|>commentary to=functions.a <|constrain|>json<|message|>{"x":1}<|call|> SUFFIX"#;
        let (tool_calls, normal) = parse_tool_calls_harmony_complete(
            text,
            &JsonParserConfig {
                allow_eof_recovery: true,
                ..Default::default()
            },
            None,
        )
        .await
        .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let normal = normal.unwrap_or_default();
        assert!(
            normal.contains("PREFIX"),
            "normal must keep prefix: {normal:?}"
        );
        assert!(
            normal.contains("SUFFIX"),
            "normal must keep suffix: {normal:?}"
        );
        assert!(
            !normal.contains("<|start|>"),
            "normal must not leak harmony start token: {normal:?}"
        );
        assert!(
            !normal.contains("assistant"),
            "normal must not leak harmony assistant marker: {normal:?}"
        );
    }

    #[tokio::test]
    async fn test_parse_harmony_parse_failure_strips_protocol_from_normal_text() {
        let text = r#"Before <|start|>assistant<|channel|>commentary <|constrain|>json<|message|>{"location":"NYC"<|call|> After"#;
        let (tool_calls, normal) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert!(tool_calls.is_empty());
        let normal = normal.unwrap_or_default();
        assert_eq!(normal, "Before  After");
        for token in [
            "<|start|>",
            "<|channel|>",
            "<|constrain|>",
            "<|message|>",
            "<|call|>",
        ] {
            assert!(
                !normal.contains(token),
                "normal_text must not leak {token}: {normal:?}"
            );
        }
        assert!(
            !normal.contains("commentary"),
            "normal_text must not leak tool-call metadata: {normal:?}"
        );
    }

    #[tokio::test] // PARSER.batch.4, PARSER.batch.5, PARSER.harmony.2
    async fn test_parse_tool_calls_harmony_without_call_token() {
        let text = r#"<|channel|>analysis<|message|>We need to call get_weather function. The user asks "What's the weather like in San Francisco in Celsius?" So location: "San Francisco, CA" unit: "celsius". Let's call function.<|end|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"San Francisco, CA","unit":"celsius"}"#;
        let (tool_calls, normal_content) =
            parse_tool_calls_harmony_complete(text, &Default::default(), None)
                .await
                .unwrap();
        assert_eq!(normal_content, Some("".to_string()));
        assert!(tool_calls.is_empty());
    }

    /// Parser-level invariant: the harmony parser is byte-stable — it
    /// doesn't see `finish_reason` and produces the same output regardless
    /// of the upstream stream-end reason. Real PIPELINE.finish_reason coverage (stop /
    /// tool_calls / length mapping) lives in
    /// `lib/llm/tests/test_streaming_tool_parsers.rs` and belongs in the
    /// cross-parser finish_reason mapping work-item (tracked separately).
    #[tokio::test]
    async fn test_harmony_parser_output_independent_of_upstream_finish() {
        let text = r#"<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"NYC"}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
    }

    /// PARSER.batch.6 — empty args. A no-arg harmony call (`{}`) must still surface
    /// the function name.
    #[tokio::test] // PARSER.batch.6 — gpt-oss
    async fn test_parse_harmony_empty_args() {
        let text = r#"<|channel|>commentary to=functions.current_time <|constrain|>json<|message|>{}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(tool_calls.len(), 1);
        let (name, args) = extract_name_and_args(tool_calls[0].clone());
        assert_eq!(name, "current_time");
        assert_eq!(args, serde_json::json!({}));
    }

    /// PARSER.batch.9 — empty / null content variants. Truly-empty (zero bytes)
    /// and whitespace-only inputs must yield no tool calls. Unlike the
    /// XML/JSON parsers (which trim whitespace down to `Some("")`), the
    /// harmony parser passes the input verbatim through to normal_text —
    /// pin that distinction here.
    #[tokio::test] // PARSER.batch.9 — gpt-oss
    async fn test_parse_harmony_empty_and_whitespace_inputs() {
        for input in &["", " ", "\n", "\t\n  \t"] {
            let (tool_calls, normal) =
                parse_tool_calls_harmony_complete(input, &Default::default(), None)
                    .await
                    .unwrap();
            assert!(
                tool_calls.is_empty(),
                "Empty/whitespace input must yield no calls (input={:?})",
                input
            );
            assert_eq!(
                normal.as_deref(),
                Some(*input),
                "harmony passes empty/whitespace input verbatim to normal_text (input={:?})",
                input
            );
        }
    }

    /// PARSER.batch.10 — duplicate calls (same function name twice). Two
    /// back-to-back commentary blocks for the same function. Pin
    /// parser-level behavior — both calls returned with distinct ids
    /// and distinct args.
    #[tokio::test] // PARSER.batch.10 — gpt-oss
    async fn test_parse_harmony_duplicate_calls_same_name() {
        let text = r#"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"NYC"}<|call|><|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"LA"}<|call|>"#;
        let (tool_calls, _) = parse_tool_calls_harmony_complete(text, &Default::default(), None)
            .await
            .unwrap();
        assert_eq!(
            tool_calls.len(),
            2,
            "Both duplicate-name calls must be returned"
        );
        assert_ne!(
            tool_calls[0].id, tool_calls[1].id,
            "Duplicate calls must have distinct ids"
        );
        let (name0, args0) = extract_name_and_args(tool_calls[0].clone());
        let (name1, args1) = extract_name_and_args(tool_calls[1].clone());
        assert_eq!(name0, "get_weather");
        assert_eq!(name1, "get_weather");
        assert_eq!(args0["city"], "NYC");
        assert_eq!(args1["city"], "LA");
    }
}

#[cfg(test)]
mod detect_parser_tests {
    use super::*;

    #[test] // helper
    fn test_detect_tool_call_start_harmony_chunk_with_tool_call_start_token() {
        let text = r#"<|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_harmony(text, &config, false);
        assert!(result);
    }

    #[test] // helper
    fn test_detect_tool_call_start_harmony_chunk_without_tool_call_start_token() {
        // This is a warkaround for now. Right now everything is treated as tool call start token.
        // We need to improve this in the future.
        let text = r#"<|channel|>commentary to=functions.get_current_weather"#;
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };
        let result = detect_tool_call_start_harmony(text, &config, false);
        assert!(result);
    }

    #[test] // helper, PARSER.stream.3
    fn test_detect_tool_call_start_harmony_partial_tokens() {
        // Test partial token detection for streaming scenarios
        let config = JsonParserConfig {
            tool_call_start_tokens: vec!["<|start|>assistant<|channel|>commentary".to_string()],
            tool_call_end_tokens: vec!["<|call|>".to_string()],
            ..Default::default()
        };

        // Test various partial prefixes in strict mode
        assert!(
            detect_tool_call_start_harmony("<", &config, true),
            "'<' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|", &config, true),
            "'<|' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|start|>", &config, true),
            "'<|start|>' should be detected as potential start"
        );
        assert!(
            detect_tool_call_start_harmony("<|start|>assistant", &config, true),
            "'<|start|>assistant' should be detected as potential start"
        );

        // Test that unrelated text is not detected in strict mode
        assert!(
            !detect_tool_call_start_harmony("hello world", &config, true),
            "'hello world' should not be detected in strict mode"
        );
        assert!(
            !detect_tool_call_start_harmony("xyz", &config, true),
            "'xyz' should not be detected in strict mode"
        );
    }
}
