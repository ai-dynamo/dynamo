// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Adapted from mistral.rs
//
// MIT License
//
// Copyright (c) 2025 Eric Buehler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use akin::akin;
use anyhow::ensure;
use anyhow::Result;
use candle_core::quantized::gguf_file;
use std::collections::HashMap;
use tracing::warn;

use crate::gguf::Content;

pub trait ModelConfigLike {
    fn max_seq_len(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_attn_heads(&self) -> usize;
    fn k_head_dim(&self) -> usize;
    fn v_head_dim(&self) -> usize;
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct ContentConfig {
    max_seq_len: usize,
    hidden_size: usize,
    num_attn_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    key_length: Option<usize>,
    value_length: Option<usize>,
}

#[allow(clippy::cast_possible_truncation)]
impl From<&Content> for ContentConfig {
    fn from(value: &Content) -> Self {
        let metadata = value.get_metadata();
        let arch = metadata["general.architecture"].to_string().unwrap();
        Self {
            max_seq_len: metadata[&format!("{arch}.context_length")]
                .to_u64()
                .unwrap() as usize,
            hidden_size: metadata[&format!("{arch}.embedding_length")]
                .to_u64()
                .unwrap() as usize,
            num_attn_heads: metadata[&format!("{arch}.attention.head_count")]
                .to_u64()
                .unwrap() as usize,
            num_kv_heads: metadata[&format!("{arch}.attention.head_count_kv")]
                .to_u64()
                .unwrap() as usize,
            num_layers: metadata[&format!("{arch}.block_count")].to_u64().unwrap() as usize,
            key_length: metadata
                .get(&format!("{arch}.attention.key_length"))
                .map(|x| x.to_u64().unwrap() as usize),
            value_length: metadata
                .get(&format!("{arch}.attention.value_length"))
                .map(|x| x.to_u64().unwrap() as usize),
        }
    }
}

impl ModelConfigLike for ContentConfig {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn k_head_dim(&self) -> usize {
        self.key_length
            .unwrap_or(self.hidden_size / self.num_attn_heads)
    }
    fn v_head_dim(&self) -> usize {
        self.value_length
            .unwrap_or(self.hidden_size / self.num_attn_heads)
    }
}

pub struct ContentMetadata<'a> {
    pub path_prefix: &'a str,
    pub metadata: &'a HashMap<String, gguf_file::Value>,
}

impl ContentMetadata<'_> {
    // Retrieve a prop the struct needs by querying the metadata content:
    pub fn get_value<T: TryFromValue>(&self, field_name: &str) -> Result<T, anyhow::Error> {
        let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);
        let value = self.metadata.get(&prop_key).cloned();

        // Unwrap the inner value of the `Value` enum via trait method,
        // otherwise format error with prop key as context:
        value
            .try_value_into()
            .or_else(|e| anyhow::bail!("`{prop_key}` `{e}`"))
    }

    // Fail early - Catch all missing mandatory keys upfront:
    pub fn has_required_keys(&self, fields: &[&str]) -> Result<()> {
        let mut all_props_are_present = true;

        for field_name in fields {
            let prop_key = format!("{prefix}.{field_name}", prefix = self.path_prefix);

            if !self.metadata.contains_key(&prop_key) {
                all_props_are_present = false;
                warn!("Expected GGUF metadata to have key: `{prop_key}`");
            }
        }

        ensure!(all_props_are_present, "Tokenizer is missing required props");
        Ok(())
    }
}

// These traits below are a workaround for converting candles GGUF `Value` enum type wrapper.
// A better upstream approach would instead be to provide serialize/deserialize support?
pub trait TryFromValue {
    fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

// Value wrapped types, each has a different conversion method:
// NOTE: Type conversion methods internally bail with "not a <into type> <input value>"
// https://docs.rs/candle-core/latest/candle_core/quantized/gguf_file/enum.Value.html#variants
akin! {
    let &types = [String, bool, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64];
    let &to_type = [
        value.to_string().cloned(),
        value.to_bool(),
        value.to_f32(),
        value.to_f64(),
        value.to_i8(),
        value.to_i16(),
        value.to_i32(),
        value.to_i64(),
        value.to_u8(),
        value.to_u16(),
        value.to_u32(),
        value.to_u64(),
    ];

    impl TryFromValue for *types {
        fn try_from_value(value: gguf_file::Value) -> Result<Self, candle_core::Error> {
            *to_type.or_else(|_| candle_core::bail!("value is not a `*types`"))
        }
    }
}

// Vec<Value> to Vec<T> from above types:
impl<T: TryFromValue> TryFromValue for Vec<T> {
    fn try_from_value(value_vec: gguf_file::Value) -> Result<Self, candle_core::Error> {
        value_vec
            .to_vec()
            .or_else(|_| candle_core::bail!("value is not a `Vec`"))?
            .clone()
            .into_iter()
            .map(|item| T::try_from_value(item))
            .collect()
    }
}

pub trait TryValueInto<T>: Sized {
    fn try_value_into(self) -> Result<T, candle_core::Error>;
}

impl<T: TryFromValue> TryValueInto<T> for gguf_file::Value {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        T::try_from_value(self)
    }
}

impl<T: TryFromValue> TryValueInto<T> for Option<gguf_file::Value> {
    fn try_value_into(self) -> Result<T, candle_core::Error> {
        match self {
            Some(value) => value.try_value_into(),
            None => candle_core::bail!("Expected `Option<gguf_file::Value>` to contain a value"),
        }
    }
}
