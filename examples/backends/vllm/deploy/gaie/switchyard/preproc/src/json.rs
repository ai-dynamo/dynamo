// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Reject ambiguous duplicate object keys before extracting routing signals.
use std::{collections::BTreeMap, fmt};

use serde::{
    Deserialize, Deserializer,
    de::{MapAccess, SeqAccess, Visitor},
};
use serde_json::value::{RawValue, to_raw_value};
use serde_json::{Map, Number, Value};

struct UniqueValue(Value);
impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(UniqueVisitor)
    }
}

struct UniqueVisitor;
impl<'de> Visitor<'de> for UniqueVisitor {
    type Value = UniqueValue;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("JSON without duplicate object keys")
    }
    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Bool(v)))
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(v.into())))
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Number(v.into())))
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Self::Value, E> {
        Number::from_f64(v)
            .map(|n| UniqueValue(Value::Number(n)))
            .ok_or_else(|| E::custom("invalid number"))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(v.to_owned())))
    }
    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::String(v)))
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Self::Value, E> {
        Ok(UniqueValue(Value::Null))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(UniqueValue(value)) = seq.next_element()? {
            values.push(value);
        }
        Ok(UniqueValue(Value::Array(values)))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut values = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom("duplicate JSON object key"));
            }
            let UniqueValue(value) = map.next_value()?;
            values.insert(key, value);
        }
        Ok(UniqueValue(Value::Object(values)))
    }
}

pub fn parse(body: &[u8]) -> Result<Value, serde_json::Error> {
    serde_json::from_slice::<UniqueValue>(body).map(|value| value.0)
}

/// Rewrite a validated object without rounding numbers or reserializing nested values.
pub fn replace_model(body: &[u8], model: &str) -> Result<Vec<u8>, serde_json::Error> {
    let mut fields: BTreeMap<String, &RawValue> = serde_json::from_slice(body)?;
    let model = to_raw_value(model)?;
    fields.insert("model".to_owned(), &model);
    serde_json::to_vec(&fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_duplicates_at_any_depth() {
        assert!(parse(br#"{"model":"auto","model":"other"}"#).is_err());
        assert!(parse(br#"{"model":"auto","mo\u0064el":"other"}"#).is_err());
        assert!(parse(br#"{"messages":[{"role":"user","role":"assistant"}]}"#).is_err());
        assert_eq!(
            parse(br#"{"large":18446744073709551615,"values":[null,true,1.5]}"#).unwrap()["large"]
                .as_u64(),
            Some(u64::MAX)
        );
    }
}
