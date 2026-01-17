// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod layer;
pub mod manager;
pub mod reserved;
pub mod reuse;
pub mod sequence;
pub mod storage;

// #[cfg(feature = "cuda_kv")]
// pub mod storage;

use reserved::*;

use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{atomic::AtomicU64, Arc, RwLock},
};

use async_trait::async_trait;
use derive_getters::Dissolve;
use dynamo_runtime::{
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

// pub struct KvStorage {
//     data: u64,
//     size: usize,

//     layer_idx: usize,
//     block_idx: usize,

//     /// The layout of the tensor
//     layout: layer::KvLayer,
// }

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
