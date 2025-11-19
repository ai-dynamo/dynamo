// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    num::NonZero,
};

use anyhow::Result;
use dynamo_tokens::compute_hash_v2;
use serde::{Deserialize, Serialize};

use crate::logical::{BlockId, BlockMetadata, ImmutableBlock, MutableBlock};

/// Status of a request's finished state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FinishedStatus {
    Finished,
    Pending,
}

/// Result returned from [super::LeaderRuntime::get_num_new_matched_tokens]
pub enum MatchResult {
    /// The connector is still evaluating the request.
    Evaluating,

    /// The connector has determined that there are no matches for the request.
    NoMatches,

    /// The connector has matched some tokens for the request.
    Matched(NonZero<usize>),
}

use super::G1;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent, bound = "")]
pub struct BlocksView<T: BlockMetadata> {
    block_ids: Vec<BlockId>,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<T>,
}

impl<T: BlockMetadata> Default for BlocksView<T> {
    fn default() -> Self {
        Self {
            block_ids: Vec::new(),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: BlockMetadata> From<Vec<BlockId>> for BlocksView<T> {
    fn from(block_ids: Vec<BlockId>) -> Self {
        Self {
            block_ids,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: BlockMetadata> From<BlocksView<T>> for Vec<BlockId> {
    fn from(view: BlocksView<T>) -> Self {
        view.block_ids
    }
}

impl<T: BlockMetadata> std::ops::Deref for BlocksView<T> {
    type Target = [BlockId];

    fn deref(&self) -> &Self::Target {
        &self.block_ids
    }
}

impl<T: BlockMetadata> AsRef<[BlockId]> for BlocksView<T> {
    fn as_ref(&self) -> &[BlockId] {
        &self.block_ids
    }
}

impl<T: BlockMetadata> IntoIterator for BlocksView<T> {
    type Item = BlockId;
    type IntoIter = std::vec::IntoIter<BlockId>;

    fn into_iter(self) -> Self::IntoIter {
        self.block_ids.into_iter()
    }
}

impl<T: BlockMetadata> BlocksView<T> {
    pub fn is_empty(&self) -> bool {
        self.block_ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.block_ids.len()
    }

    pub fn as_ref(&self) -> &[BlockId] {
        &self.block_ids
    }

    pub fn extend(&mut self, other: BlocksView<T>) {
        self.block_ids.extend(other.block_ids)
    }
}

#[derive(Debug)]
pub enum Blocks<T: BlockMetadata> {
    View(BlocksView<T>),
    OwnedMutable(Vec<MutableBlock<T>>),
    OwnedImmutable(Vec<ImmutableBlock<T>>),
    Empty,
}

impl<T: BlockMetadata> Default for Blocks<T> {
    fn default() -> Self {
        Blocks::<T>::Empty
    }
}

impl<T: BlockMetadata> Blocks<T> {
    pub fn from_external(ids: Vec<BlockId>) -> Self {
        Blocks::<T>::View(BlocksView::from(ids))
    }

    pub fn from_mutable(blocks: Vec<MutableBlock<T>>) -> Self {
        Blocks::OwnedMutable(blocks)
    }

    pub fn from_immutable(blocks: Vec<ImmutableBlock<T>>) -> Self {
        Blocks::OwnedImmutable(blocks)
    }

    pub fn is_external(&self) -> bool {
        matches!(self, Blocks::View(_))
    }

    pub fn is_mutable(&self) -> bool {
        matches!(self, Blocks::OwnedMutable(_))
    }

    pub fn is_immutable(&self) -> bool {
        matches!(self, Blocks::OwnedImmutable(_))
    }

    pub fn take_view(self) -> BlocksView<T> {
        match self {
            Blocks::View(view) => view,
            _ => panic!("Expected External blocks"),
        }
    }

    pub fn take_mutable(self) -> Vec<MutableBlock<T>> {
        match self {
            Blocks::OwnedMutable(blocks) => blocks,
            _ => panic!("Expected OwnedMutable blocks"),
        }
    }

    pub fn take_immutable(self) -> Vec<ImmutableBlock<T>> {
        match self {
            Blocks::OwnedImmutable(blocks) => blocks,
            _ => panic!("Expected OwnedImmutable blocks"),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Blocks::View(view) => view.block_ids.is_empty(),
            Blocks::OwnedMutable(blocks) => blocks.is_empty(),
            Blocks::OwnedImmutable(blocks) => blocks.is_empty(),
            Blocks::Empty => true,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Blocks::View(view) => view.block_ids.len(),
            Blocks::OwnedMutable(blocks) => blocks.len(),
            Blocks::OwnedImmutable(blocks) => blocks.len(),
            Blocks::Empty => 0,
        }
    }

    pub fn block_ids(&self) -> Vec<BlockId> {
        match self {
            Blocks::View(view) => view.block_ids.clone(),
            Blocks::OwnedMutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            Blocks::OwnedImmutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            Blocks::Empty => Vec::new(),
        }
    }

    pub fn extend(&mut self, others: Blocks<T>) -> Result<()> {
        match (self, others) {
            // Empty can be replaced with any type
            (this @ Blocks::Empty, others) => {
                *this = others;
                Ok(())
            }
            // External can only extend with External
            (Blocks::View(view), Blocks::View(other_view)) => {
                view.block_ids.extend(other_view.block_ids);
                Ok(())
            }
            // Mismatched types - return error
            (this, others) => Err(anyhow::anyhow!(
                "Cannot extend Blocks: type mismatch between {:?} and {:?}",
                std::mem::discriminant(this),
                std::mem::discriminant(&others)
            )),
        }
    }
}

pub trait GetBlockIds {
    fn block_ids(&self) -> Vec<BlockId>;
}

impl<T: BlockMetadata> GetBlockIds for Blocks<T> {
    fn block_ids(&self) -> Vec<BlockId> {
        match self {
            Blocks::View(view) => view.block_ids.clone(),
            Blocks::OwnedMutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            Blocks::OwnedImmutable(blocks) => blocks.iter().map(|b| b.block_id()).collect(),
            Blocks::Empty => Vec::new(),
        }
    }
}

pub trait AsBlocksView<T: BlockMetadata> {
    /// Returns a view of the block_id as a Blocks<T> enum using the Blocks::External variant.
    /// This is useful to keep the typing `T` when dealing with raw block_ids.
    fn as_blocks_view(&self) -> BlocksView<T>;
}

impl<T: BlockMetadata> AsBlocksView<T> for Vec<MutableBlock<T>> {
    fn as_blocks_view(&self) -> BlocksView<T> {
        BlocksView::from(self.iter().map(|b| b.block_id()).collect::<Vec<_>>())
    }
}

impl<T: BlockMetadata> AsBlocksView<T> for &[MutableBlock<T>] {
    fn as_blocks_view(&self) -> BlocksView<T> {
        BlocksView::from(self.iter().map(|b| b.block_id()).collect::<Vec<_>>())
    }
}

impl<T: BlockMetadata> AsBlocksView<T> for Vec<ImmutableBlock<T>> {
    fn as_blocks_view(&self) -> BlocksView<T> {
        BlocksView::from(self.iter().map(|b| b.block_id()).collect::<Vec<_>>())
    }
}

impl<T: BlockMetadata> AsBlocksView<T> for &[ImmutableBlock<T>] {
    fn as_blocks_view(&self) -> BlocksView<T> {
        BlocksView::from(self.iter().map(|b| b.block_id()).collect::<Vec<_>>())
    }
}

/// Minimal representation of a scheduler slot request.
#[derive(Clone, Debug)]
pub struct Request {
    pub request_id: String,
    pub lora_name: Option<String>,
    pub salt_hash: u64,
    pub max_tokens: usize,
}

impl Request {
    pub fn new(
        request_id: impl Into<String>,
        lora_name: Option<String>,
        salt: Option<String>,
        max_tokens: usize,
    ) -> Self {
        // Pack any data that needs to be included in the salt hash into [`SaltPayload`]
        #[derive(Serialize)]
        struct SaltPayload<'a> {
            #[serde(skip_serializing_if = "Option::is_none")]
            salt: Option<&'a str>,
            #[serde(skip_serializing_if = "Option::is_none")]
            lora_name: Option<&'a str>,
        }

        let request_id = request_id.into();
        let payload = SaltPayload {
            salt: salt.as_deref(),
            lora_name: lora_name.as_deref(),
        };
        let salt_bytes = serde_json::to_vec(&payload).expect("failed to serialize salt payload");
        let salt_hash = compute_hash_v2(&salt_bytes, 0);

        Self {
            request_id,
            lora_name,
            salt_hash,
            max_tokens,
        }
    }
}

#[derive(Debug)]
pub struct NewRequestData {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub blocks: Blocks<G1>,
    pub num_computed_tokens: usize,
}

#[derive(Debug)]
pub struct CachedRequestData {
    pub request_id: String,
    pub resumed_from_preemption: bool,
    pub new_token_ids: Vec<u32>,
    pub new_blocks: Blocks<G1>,
    pub num_computed_tokens: usize,
}

#[derive(Debug, Default)]
pub struct SchedulerOutput {
    pub(crate) new_requests: Vec<NewRequestData>,
    pub(crate) cached_requests: Vec<CachedRequestData>,
    pub(crate) num_scheduled_tokens: HashMap<String, usize>,
}

impl SchedulerOutput {
    pub fn new() -> Self {
        Self::default()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_new_request(
        &mut self,
        request_id: impl Into<String>,
        prompt_token_ids: Vec<u32>,
        blocks: Blocks<G1>,
        num_computed_tokens: usize,
    ) {
        self.new_requests.push(NewRequestData {
            request_id: request_id.into(),
            prompt_token_ids,
            blocks,
            num_computed_tokens,
        });
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_cached_request(
        &mut self,
        request_id: impl Into<String>,
        resumed_from_preemption: bool,
        new_token_ids: Vec<u32>,
        new_blocks: Blocks<G1>,
        num_computed_tokens: usize,
    ) {
        self.cached_requests.push(CachedRequestData {
            request_id: request_id.into(),
            resumed_from_preemption,
            new_token_ids,
            new_blocks,
            num_computed_tokens,
        });
    }

    pub fn set_num_scheduled_tokens(&mut self, counts: HashMap<String, usize>) {
        self.num_scheduled_tokens = counts;
    }

    pub fn new_requests(&self) -> &[NewRequestData] {
        &self.new_requests
    }

    pub fn cached_requests(&self) -> &[CachedRequestData] {
        &self.cached_requests
    }
}

pub struct KVConnectorOutput {
    pub finished_sending: Option<HashSet<String>>,
    pub finished_recving: Option<HashSet<String>>,
}
