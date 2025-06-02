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

use std::collections::HashMap;

use anyhow::Context;
use candle_core::{
    quantized::gguf_file::{self, Value},
    Result,
};
use tracing::info;

use super::GGUFArchitecture;

// Internal invariant: contents and readers must be paired.
/// This abstracts the files for a GGUF model and enables multiple files to be used.
pub struct Content {
    _contents: Vec<gguf_file::Content>,
    arch: GGUFArchitecture,
    all_metadata: HashMap<String, Value>,
}

impl Content {
    /// Create a `Content` from a set of file readers.
    pub fn from_readers<R: std::io::Seek + std::io::Read>(readers: &mut [&mut R]) -> Result<Self> {
        let mut contents = Vec::new();
        let n_readers = readers.len();
        for reader in readers.iter_mut() {
            contents.push(gguf_file::Content::read(reader)?);
        }
        let n_splits = contents
            .iter()
            .filter_map(|ct| {
                ct.metadata
                    .get("split.count")
                    .map(|val| val.to_u64().unwrap())
            })
            .fold(Vec::new(), |mut accum, x| {
                if !accum.contains(&x) {
                    accum.push(x);
                }
                accum
            });
        if n_splits.len() > 1 {
            candle_core::bail!("GGUF files have differing `split.count` values: {n_splits:?}. Perhaps the GGUF files do not match?");
        }
        #[allow(clippy::cast_possible_truncation)]
        if !n_splits.is_empty() && n_readers != n_splits[0] as usize {
            candle_core::bail!(
                "Number of GGUF files does not match the number of splits, expected {} files.",
                n_splits[0]
            );
        } else if n_splits.len() == 1 {
            info!("GGUF file has been split into {} shards", n_splits[0]);
        }

        let mut arch = None;
        for ct in &contents {
            if !ct.metadata.contains_key("general.architecture") {
                continue;
            }

            arch = Some(
                ct.metadata["general.architecture"]
                    .to_string()
                    .context("Model metadata should have declared an architecture")
                    .and_then(GGUFArchitecture::from_value)
                    .unwrap(),
            );
        }
        let arch = arch.expect("GGUF files must specify `general.architecture`");

        let mut all_metadata = HashMap::new();
        for content in &contents {
            all_metadata.extend(content.metadata.clone())
        }

        Ok(Self {
            _contents: contents,
            arch,
            all_metadata,
        })
    }

    pub fn arch(&self) -> GGUFArchitecture {
        self.arch
    }

    /// Get all metadatas
    pub fn get_metadata(&self) -> &HashMap<String, Value> {
        &self.all_metadata
    }
}
