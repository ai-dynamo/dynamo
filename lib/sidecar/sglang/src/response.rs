// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared parsing for SGLang terminal responses.

use std::collections::HashSet;

use dynamo_backend_common::{CompletionUsage, DynamoError, StopReason, usage};
use serde_json::{Map, Value};

use crate::client;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FinishKind {
    Stop,
    Length,
    Eos,
    Cancelled,
    Abort,
    ContentFilter,
    Error,
}

pub(crate) struct ParsedFinish<'a> {
    pub(crate) kind: FinishKind,
    pub(crate) finish_type: &'a str,
    raw: &'a Value,
}

pub(crate) enum NumericStopReason<'a> {
    Any,
    Requested(&'a HashSet<i64>),
}

impl ParsedFinish<'_> {
    pub(crate) fn stop_reason(&self, numeric: NumericStopReason<'_>) -> Option<StopReason> {
        self.raw.get("matched").and_then(|matched| match matched {
            Value::String(value) => Some(StopReason::String(value.clone())),
            Value::Number(value) => value
                .as_i64()
                .filter(|token| match numeric {
                    NumericStopReason::Any => true,
                    NumericStopReason::Requested(tokens) => tokens.contains(token),
                })
                .map(StopReason::Int),
            _ => None,
        })
    }

    pub(crate) fn failure(&self) -> DynamoError {
        let message = self
            .raw
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("SGLang generation failed");
        let status_code = self.raw.get("status_code").and_then(Value::as_i64);
        let err_type = self.raw.get("err_type").and_then(Value::as_str);
        let detail = format!(
            "SGLang generation {}: {message} (status_code={}, err_type={})",
            self.finish_type,
            status_code
                .map(|value| value.to_string())
                .unwrap_or_else(|| "unknown".to_string()),
            err_type.unwrap_or("unknown")
        );
        if matches!(status_code, Some(400..=499)) {
            client::invalid_arg(detail)
        } else {
            client::protocol_error(detail)
        }
    }

    pub(crate) fn message(&self) -> String {
        self.raw
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("SGLang generation failed")
            .to_string()
    }
}

pub(crate) fn parse_finish(finish: &Value) -> Result<ParsedFinish<'_>, DynamoError> {
    let finish_type = finish
        .get("type")
        .and_then(Value::as_str)
        .or_else(|| finish.as_str())
        .ok_or_else(|| client::protocol_error("SGLang finish_reason is missing a type"))?;
    let kind = match finish_type {
        "stop" => FinishKind::Stop,
        "length" => FinishKind::Length,
        "eos" => FinishKind::Eos,
        "cancelled" => FinishKind::Cancelled,
        kind if kind.starts_with("abort") => FinishKind::Abort,
        "content_filter" => FinishKind::ContentFilter,
        "error" => FinishKind::Error,
        other => {
            return Err(client::protocol_error(format!(
                "SGLang returned unsupported finish_reason type `{other}`"
            )));
        }
    };
    Ok(ParsedFinish {
        kind,
        finish_type,
        raw: finish,
    })
}

pub(crate) fn usage_from_http_meta(meta: &Map<String, Value>) -> Option<CompletionUsage> {
    let prompt_tokens = meta
        .get("prompt_tokens")
        .and_then(Value::as_u64)
        .and_then(|tokens| u32::try_from(tokens).ok())?;
    let completion_tokens = meta
        .get("completion_tokens")
        .and_then(Value::as_u64)
        .and_then(|tokens| u32::try_from(tokens).ok())?;
    let mut completion_usage = usage(prompt_tokens, completion_tokens);
    if let Some(cached_tokens) = meta
        .get("cached_tokens")
        .and_then(Value::as_u64)
        .and_then(|tokens| u32::try_from(tokens).ok())
        .filter(|tokens| *tokens > 0)
    {
        completion_usage.prompt_tokens_details = Some(Default::default());
        if let Some(details) = completion_usage.prompt_tokens_details.as_mut() {
            details.cached_tokens = Some(cached_tokens);
        }
    }
    Some(completion_usage)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use dynamo_backend_common::StopReason;
    use serde_json::json;

    use super::{FinishKind, NumericStopReason, parse_finish};

    #[test]
    fn parses_every_known_finish_reason() {
        for (kind, expected) in [
            ("stop", FinishKind::Stop),
            ("length", FinishKind::Length),
            ("eos", FinishKind::Eos),
            ("cancelled", FinishKind::Cancelled),
            ("abort", FinishKind::Abort),
            ("abort_request", FinishKind::Abort),
            ("content_filter", FinishKind::ContentFilter),
            ("error", FinishKind::Error),
        ] {
            assert_eq!(parse_finish(&json!({"type": kind})).unwrap().kind, expected);
        }
    }

    #[test]
    fn rejects_missing_and_unknown_finish_reasons() {
        assert!(parse_finish(&json!({})).is_err());
        assert!(parse_finish(&json!({"type": "mystery"})).is_err());
    }

    #[test]
    fn filters_internal_numeric_stop_reasons() {
        let finish = json!({"type": "stop", "matched": 17});
        let parsed = parse_finish(&finish).unwrap();
        assert_eq!(
            parsed.stop_reason(NumericStopReason::Any),
            Some(StopReason::Int(17))
        );
        assert_eq!(
            parsed.stop_reason(NumericStopReason::Requested(&HashSet::from([19]))),
            None
        );
    }
}
