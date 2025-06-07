// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use super::*;
use std::sync::{Arc, Mutex};

use dynamo_llm as llm_rs;

use llm_rs::tokens as rs_tokens;
use rs_tokens::{
    TokenBlock as RsTokenBlock, TokenBlockSequence as RsTokenBlockSequence, Tokens as RsTokens,
};

/// Python binding for `dynamo_llm::tokens::TokenBlock`.
#[pyclass(name = "TokenBlock")]
#[derive(Clone)]
pub(crate) struct TokenBlock {
    inner: RsTokenBlock,
}

#[pymethods]
impl TokenBlock {
    /// Returns the slice of tokens contained in this block.
    #[getter]
    fn tokens(&self) -> Vec<u32> {
        self.inner.tokens().as_ref().to_vec()
    }

    /// Returns the salt hash used for this block.
    #[getter]
    fn salt_hash(&self) -> u64 {
        self.inner.salt_hash()
    }

    /// Returns the hash of only the tokens within this block.
    #[getter]
    fn block_hash(&self) -> u64 {
        self.inner.block_hash()
    }

    /// Returns the sequence-aware hash for this block.
    #[getter]
    fn sequence_hash(&self) -> u64 {
        self.inner.sequence_hash()
    }

    /// Returns the parent sequence hash, if any.
    #[getter]
    fn parent_sequence_hash(&self) -> Option<u64> {
        self.inner.parent_sequence_hash()
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenBlock(tokens={:?}, sequence_hash={}, block_hash={}, salt_hash={}, parent_sequence_hash={:?})",
            self.inner.tokens(),
            self.inner.sequence_hash(),
            self.inner.block_hash(),
            self.inner.salt_hash(),
            self.inner.parent_sequence_hash()
        )
    }
}

impl From<RsTokenBlock> for TokenBlock {
    fn from(inner: RsTokenBlock) -> Self {
        Self { inner }
    }
}

/// Python binding for `dynamo_llm::tokens::TokenBlockSequence`.
#[pyclass(name = "TokenBlockSequence")]
pub(crate) struct TokenBlockSequence {
    inner: Arc<Mutex<RsTokenBlockSequence>>,
}

#[pymethods]
impl TokenBlockSequence {
    /// Create a new `TokenBlockSequence`.
    ///
    /// Parameters
    /// ----------
    /// tokens : List[int], optional
    ///     Initial tokens to populate the sequence. Defaults to an empty list.
    /// block_size : int
    ///     Size of each token block (must be > 0).
    /// salt_hash : int, optional
    ///     Optional salt hash (defaults to 0).
    #[new]
    #[pyo3(signature = (tokens, block_size, salt_hash = None))]
    fn new(tokens: Vec<u32>, block_size: usize, salt_hash: Option<u64>) -> PyResult<Self> {
        let rs_tokens: RsTokens = tokens.into();
        let seq = RsTokenBlockSequence::new(rs_tokens, block_size, salt_hash);
        Ok(Self {
            inner: Arc::new(Mutex::new(seq)),
        })
    }

    /// Append a single token to the sequence. Returns the index of the block that was completed
    /// by this append, or `None` if no block was completed.
    fn append(&self, token: u32) -> PyResult<Option<usize>> {
        let mut guard = self.inner.lock().unwrap();
        let res = guard.append(token).map_err(to_pyerr)?;
        Ok(res)
    }

    /// Extend the sequence with a list of tokens. Returns `(start_idx, end_idx)` if one or more
    /// blocks were completed, otherwise `None`.
    fn extend(&self, tokens: Vec<u32>) -> PyResult<Option<(usize, usize)>> {
        let mut guard = self.inner.lock().unwrap();
        let range_opt = guard.extend(tokens.into()).map_err(to_pyerr)?;
        Ok(range_opt.map(|r| (r.start, r.end)))
    }

    /// Truncate the sequence to `len` tokens.
    fn truncate(&self, len: usize) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard.truncate(len).map_err(to_pyerr)
    }

    /// Remove `count` tokens from the end of the sequence.
    fn unwind(&self, count: usize) -> PyResult<()> {
        let mut guard = self.inner.lock().unwrap();
        guard.unwind(count).map_err(to_pyerr)
    }

    /// Pop and return the last token, or `None` if the sequence is empty.
    fn pop(&self) -> Option<u32> {
        let mut guard = self.inner.lock().unwrap();
        guard.pop()
    }

    /// Total number of tokens currently in the sequence.
    #[getter]
    fn total_tokens(&self) -> usize {
        let guard = self.inner.lock().unwrap();
        guard.total_tokens()
    }

    /// Return the salt hash for the sequence.
    #[getter]
    fn salt_hash(&self) -> u64 {
        let guard = self.inner.lock().unwrap();
        guard.salt_hash()
    }

    /// Return the list of completed blocks.
    fn blocks(&self) -> Vec<TokenBlock> {
        let guard = self.inner.lock().unwrap();
        guard
            .blocks()
            .iter()
            .cloned()
            .map(TokenBlock::from)
            .collect()
    }

    /// Return the tokens currently in the partial (in-progress) block.
    fn current_tokens(&self) -> Vec<u32> {
        let guard = self.inner.lock().unwrap();
        guard.current_block().tokens().as_ref().to_vec()
    }

    fn __repr__(&self) -> String {
        let guard = self.inner.lock().unwrap();
        format!(
            "TokenBlockSequence(total_tokens={}, num_blocks={}, salt_hash={})",
            guard.total_tokens(),
            guard.blocks().len(),
            guard.salt_hash(),
        )
    }
}
