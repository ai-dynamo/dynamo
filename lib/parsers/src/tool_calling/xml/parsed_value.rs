// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use serde_json::Value;
use serde_json::value::RawValue;

pub(super) enum ParsedValue {
    Json(Value),
    RawNumber(Box<RawValue>),
}

impl From<Value> for ParsedValue {
    fn from(value: Value) -> Self {
        Self::Json(value)
    }
}

impl Serialize for ParsedValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Json(value) => value.serialize(serializer),
            Self::RawNumber(raw) => raw.serialize(serializer),
        }
    }
}

pub(super) fn is_integer_literal(value: &str) -> bool {
    let value = value.strip_prefix('-').unwrap_or(value);
    !value.is_empty() && value.bytes().all(|b| b.is_ascii_digit())
}

pub(super) fn raw_number_literal(value: &str) -> Option<ParsedValue> {
    RawValue::from_string(value.to_string())
        .ok()
        .map(ParsedValue::RawNumber)
}

pub(super) fn coerce_integer_literal(value: &str) -> Option<ParsedValue> {
    if !is_integer_literal(value) {
        return None;
    }

    if let Ok(n) = value.parse::<i64>() {
        return Some(Value::Number(n.into()).into());
    }
    Some(raw_number_literal(value).unwrap_or_else(|| Value::String(value.to_string()).into()))
}
