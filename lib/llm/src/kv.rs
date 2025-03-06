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

pub mod manager;
pub mod reserved;
pub mod reuse;
pub mod sequence;
pub mod storage;

use reserved::*;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{atomic::AtomicU64, Arc, RwLock},
};

use async_trait::async_trait;
use derive_getters::Dissolve;
use dynemo_runtime::{
    raise,
    utils::pool::{PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem},
    Result,
};

use crate::tokens::{PartialTokenBlock, SequenceHash, TokenBlock, Tokens};

use tracing as log;

pub type UniqueBlock = PoolItem<KvBlock>;
pub type SharedBlock = SharedPoolItem<KvBlock>;

#[derive(Default)]
pub struct KvBlock {
    token_block: TokenBlock,
    priority: u32,
    return_tick: u64,
}

impl KvBlock {
    /// Creates a new KvBlock with the given token block
    pub fn new(token_block: TokenBlock) -> Self {
        Self {
            token_block,
            priority: 0,
            return_tick: 0,
            // storage: None,
        }
    }

    /// Updates the token block
    pub fn update_token_block(&mut self, token_block: TokenBlock) {
        self.token_block = token_block;
    }

    /// Resets the block to its initial state
    pub(crate) fn reset(&mut self) {
        self.token_block = TokenBlock::default();
        self.priority = 0;
        self.return_tick = 0;
        // self.storage = None;
        // self.storage_state = StorageState::Absent;
    }
}

impl Returnable for KvBlock {
    fn on_return(&mut self) {}
}

pub struct KvBlockConfig {}
