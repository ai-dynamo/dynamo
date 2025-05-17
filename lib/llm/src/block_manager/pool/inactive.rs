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

use crate::block_manager::block::BlockState;

use super::*;
use tracing::instrument;

#[derive(Default)]
pub struct InactiveBlockPool<S: Storage, M: BlockMetadata> {
    // Direct lookup by sequence_hash
    lookup_map: HashMap<SequenceHash, Block<S, M>>,

    // Ordered by timestamp (oldest first)
    priority_set: BTreeSet<PriorityKey<M>>,

    // Fully Uninitialized
    uninitialized_set: VecDeque<Block<S, M>>,

    // Return Tick
    return_tick: u64,

    // Total blocks
    total_blocks: u64,
}

impl<S: Storage, M: BlockMetadata> InactiveBlockPool<S, M> {
    /// Creates a new, empty [`InactiveBlockPool`].
    ///
    /// # Returns
    ///
    /// A new instance of [`InactiveBlockPool`].
    pub(crate) fn new() -> Self {
        Self {
            lookup_map: HashMap::new(),
            priority_set: BTreeSet::new(),
            uninitialized_set: VecDeque::new(),
            return_tick: 0,
            total_blocks: 0,
        }
    }

    /// Returns the total number of blocks managed by this pool (both available and acquired).
    ///
    /// # Returns
    ///
    /// The total block count as a [`u64`].
    pub fn total_blocks(&self) -> u64 {
        self.total_blocks
    }

    /// Returns the number of blocks currently available in the pool.
    ///
    /// This is calculated dynamically based on the blocks in the [`uninitialized_set`]
    /// and the [`lookup_map`].
    ///
    /// # Returns
    ///
    /// The available block count as a [`u64`].
    pub fn available_blocks(&self) -> u64 {
        self.uninitialized_set.len() as u64 + self.lookup_map.len() as u64
    }

    /// Inserts a block into the pool using its sequence hash for potential reuse.
    ///
    /// If an entry with the same priority key already exists in the [`priority_set`],
    /// the block is reset and moved to the [`uninitialized_set`].
    /// If an entry with the same sequence hash already exists in the [`lookup_map`]
    /// (but not the priority set - indicating an inconsistency), the block is reset
    /// and moved to the [`uninitialized_set`].
    /// Otherwise, the block is added to both the [`lookup_map`] and the [`priority_set`].
    ///
    /// # Arguments
    ///
    /// * `block` - The block to insert ([`Block<T, M>`]).
    /// * `sequence_hash` - The sequence hash associated with the block's content ([`SequenceHash`]).
    #[instrument(level = "trace", skip(self, block), fields(sequence_hash = ?sequence_hash))]
    fn insert_with_sequence_hash(&mut self, block: Block<S, M>, sequence_hash: SequenceHash) {
        let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
        if self.priority_set.contains(&priority_key) {
            tracing::trace!("multiple entries with the same priority key, resetting block and inserting into uninitialized set");
            let mut block = block;
            block.reset();
            self.uninitialized_set.push_back(block);
        } else if let std::collections::hash_map::Entry::Vacant(e) =
            self.lookup_map.entry(sequence_hash)
        {
            tracing::trace!("inserting block to map and priority set");
            self.priority_set.insert(priority_key);
            e.insert(block);
        } else {
            tracing::trace!("multiple entries in lookup map with the same sequence hash, inserting into uninitialized set");
            let mut block = block;
            block.reset();
            self.uninitialized_set.push_back(block);
        }
    }

    /// Internal helper to insert a block into the appropriate internal collection
    /// based on its current state.
    ///
    /// - [`BlockState::Reset`], [`BlockState::Partial`], [`BlockState::Complete`] states result in the block being reset and added
    ///   to the `uninitialized_set`.
    /// - [`BlockState::Registered`] state results in the block being added via [`insert_with_sequence_hash`].
    ///
    /// # Arguments
    ///
    /// * `block` - The block to insert ([`Block<S, M>`]).
    #[instrument(level = "trace", skip(self, block), fields(block_state = ?block.state()))]
    fn insert(&mut self, block: Block<S, M>) {
        tracing::trace!("Inserting block into available pool");

        // If we already have an entry for this sequence hash or the block is reset,
        // we need to move it to the uninitialized set
        match block.state() {
            BlockState::Reset => {
                self.uninitialized_set.push_back(block);
            }
            BlockState::Partial(_) => {
                let mut block = block;
                block.reset();
                self.uninitialized_set.push_back(block);
            }
            BlockState::Complete(_) => {
                let mut block = block;
                block.reset();
                self.uninitialized_set.push_back(block);
            }
            BlockState::Registered(state, _) => {
                let sequence_hash = state.sequence_hash();
                self.insert_with_sequence_hash(block, sequence_hash);
            }
        }
    }

    /// Adds multiple blocks to the pool.
    ///
    /// Each block is reset before being inserted. The total block count is updated.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A vector of blocks ([`Block<T, M>`]) to add.
    #[instrument(level = "debug", skip(self, blocks))]
    pub fn add_blocks(&mut self, blocks: Vec<Block<S, M>>) {
        let count = blocks.len();
        tracing::debug!(count, "Adding blocks to pool");

        for (i, mut block) in blocks.into_iter().enumerate() {
            tracing::trace!(current = i + 1, total = count, "Processing block");
            block.reset();
            self.insert(block);
        }

        self.total_blocks += count as u64;
    }

    /// Adds multiple blocks to the pool.
    ///
    /// The state of the blocks are not reset.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A vector of blocks ([`Block<T, M>`]) to add.
    #[instrument(level = "debug", skip(self, blocks))]
    pub fn add_blocks_with_state(&mut self, blocks: Vec<Block<S, M>>) {
        let count = blocks.len();
        tracing::debug!(count, "Adding blocks to pool");
        self.total_blocks += count as u64;
        // self.available_blocks += count as u64;
        self.return_blocks(blocks);
    }

    /// Returns a single block to the pool.
    ///
    /// Increments the internal return tick, updates the block's metadata,
    /// and inserts the block back into the appropriate internal collection.
    ///
    /// # Arguments
    ///
    /// * `block` - The block ([`Block<S, M>`]) to return.
    #[instrument(level = "debug", skip(self, block))]
    pub fn return_block(&mut self, mut block: Block<S, M>) {
        // increment the return tick
        self.return_tick += 1;

        // update the metadata
        block.metadata_on_returned(self.return_tick);

        // insert the block into the pool
        self.insert(block);

        // self.available_blocks += 1;
    }

    /// Returns multiple blocks to the pool.
    ///
    /// Iterates through the blocks in reverse order (tail to head) and calls
    /// `return_block` for each one.
    ///
    /// # Arguments
    ///
    /// * `blocks` - A vector of blocks ([`Block<T, M>`]) to return.
    #[instrument(level = "debug", skip(self, blocks))]
    pub fn return_blocks(&mut self, blocks: Vec<Block<S, M>>) {
        let count = blocks.len();
        tracing::debug!(count, "Returning blocks to pool");
        // return the block to the pool from tail to head
        for (i, block) in blocks.into_iter().rev().enumerate() {
            tracing::trace!(current = i + 1, total = count, "Returning block");
            // Note: return_block has its own instrumentation
            self.return_block(block);
        }
    }

    /// Attempts to remove and return a block associated with the given sequence hash
    /// from the [`lookup_map`] and [`priority_set`].
    ///
    /// # Arguments
    ///
    /// * `sequence_hash` - The sequence hash ([`SequenceHash`]) of the block to take.
    ///
    /// # Returns
    ///
    /// An [`Option<Block<S, M>>`] containing the block if found, otherwise `None`.
    #[instrument(level = "trace", skip(self), fields(sequence_hash = ?sequence_hash))]
    fn take_with_sequence_hash(&mut self, sequence_hash: SequenceHash) -> Option<Block<S, M>> {
        match self.lookup_map.remove(&sequence_hash) {
            Some(block) => {
                // Remove from priority set
                let priority_key = PriorityKey::new(block.metadata().clone(), sequence_hash);
                self.priority_set.remove(&priority_key);
                Some(block)
            }
            None => None,
        }
    }

    /// Attempts to find and take a block matching the given sequence hash.
    ///
    /// This is a convenience wrapper around `take_with_sequence_hash`.
    ///
    /// # Arguments
    ///
    /// * `sequence_hash` - The sequence hash ([`SequenceHash`]) to match.
    ///
    /// # Returns
    ///
    /// An [`Option<Block<S, M>>`] containing the block if found, otherwise `None`.
    #[instrument(level = "debug", skip(self), fields(sequence_hash = ?sequence_hash))]
    pub fn match_sequence_hash(&mut self, sequence_hash: SequenceHash) -> Option<Block<S, M>> {
        self.take_with_sequence_hash(sequence_hash)
    }

    /// Attempts to find and take multiple blocks matching a sequence of hashes.
    ///
    /// Iterates through the provided hashes and takes blocks using `take_with_sequence_hash`.
    /// Stops if a hash is not found.
    ///
    /// # Arguments
    ///
    /// * `sequence_hashes` - A vector of sequence hashes ([`SequenceHash`]) to match.
    ///
    /// # Returns
    ///
    /// A vector containing the blocks ([`Block<T, M>`]) that were successfully matched and taken.
    /// The vector may be shorter than `sequence_hashes` if not all hashes were found.
    #[instrument(level = "debug", skip(self, sequence_hashes), fields(num_hashes = sequence_hashes.len()))]
    pub fn match_sequence_hashes(
        &mut self,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Vec<Block<S, M>> {
        let total_hashes = sequence_hashes.len();
        let mut matched_blocks = Vec::with_capacity(total_hashes);

        for (i, hash) in sequence_hashes.into_iter().enumerate() {
            tracing::trace!(current = i + 1, total = total_hashes, sequence_hash = ?hash, "Attempting to match sequence hash");
            // Note: take_with_sequence_hash has its own instrumentation
            if let Some(block) = self.take_with_sequence_hash(hash) {
                tracing::trace!(current = i + 1, total = total_hashes, sequence_hash = ?hash, "Matched sequence hash");
                matched_blocks.push(block);
            } else {
                tracing::trace!(current = i + 1, total = total_hashes, sequence_hash = ?hash, "Sequence hash not found, stopping match");
                break;
            }
        }

        matched_blocks
    }

    /// Attempts to find and take multiple blocks matching a sequence of `TokenBlock`s.
    ///
    /// Extracts sequence hashes from the [`TokenBlock`]s and calls [`take_with_sequence_hash`].
    /// Stops if a hash is not found.
    ///
    /// # Arguments
    ///
    /// * `token_blocks` - A slice of [`TokenBlock`]s to match.
    ///
    /// # Returns
    ///
    /// A vector containing the blocks ([`Block<T, M>`]) that were successfully matched and taken.
    /// The vector may be shorter than `token_blocks` if not all corresponding hashes were found.
    #[instrument(level = "debug", skip(self, token_blocks), fields(num_token_blocks = token_blocks.len()))]
    pub fn match_token_blocks(&mut self, token_blocks: &[TokenBlock]) -> Vec<Block<S, M>> {
        let total_blocks = token_blocks.len();
        let mut matched_blocks = Vec::with_capacity(total_blocks);

        tracing::debug!("Attempting to match {} token blocks", total_blocks);

        for (i, token_block) in token_blocks.iter().enumerate() {
            let sequence_hash = token_block.sequence_hash();
            tracing::trace!(sequence_hash = ?sequence_hash, "Attempting to match token block hash {}/{}", i + 1, total_blocks);
            if let Some(block) = self.take_with_sequence_hash(sequence_hash) {
                tracing::trace!(sequence_hash = ?sequence_hash, "Matched token block hash");
                matched_blocks.push(block);
            } else {
                tracing::trace!(sequence_hash = ?sequence_hash, "Token block hash not found, stopping match");
                break;
            }
        }

        tracing::debug!(
            "Matched {} of {} token blocks",
            matched_blocks.len(),
            total_blocks
        );

        matched_blocks
    }

    /// Acquires a single free block from the pool.
    ///
    /// Prioritizes blocks from the [`uninitialized_set`] first, then takes the
    /// lowest priority block from the [`priority_set`] (and [`lookup_map`]).
    /// If a block is taken from the priority set, it is reset.
    ///
    /// # Returns
    ///
    /// An [`Option<Block<T, M>>`] containing a free block if available, otherwise `None`.
    ///
    /// # Panics
    ///
    /// This function can panic if there is an inconsistency between the [`priority_set`]
    /// and [`lookup_map`] (i.e., a key exists in the set but not the map). This indicates
    /// a bug in the pool's internal logic.
    #[instrument(level = "debug", skip(self))]
    pub fn acquire_free_block(&mut self) -> Option<Block<S, M>> {
        // First try uninitialized blocks - these are often part of sequences
        // that have been arranged in the correct order
        if let Some(mut block) = self.uninitialized_set.pop_front() {
            tracing::trace!("Acquired uninitialized block");
            self.return_tick += 1;
            block.metadata_on_acquired(self.return_tick);
            return Some(block);
        }

        // if we have blocks in the priority set, pop the first (it's sorted by priority)
        // a fatal error will occur if the block is not found in the lookup map
        if let Some(key) = self.priority_set.pop_first() {
            tracing::trace!("Acquired priority/registered block map; resetting block");
            match self.lookup_map.remove(&key.sequence_hash()) {
                Some(mut block) => {
                    block.reset();
                    self.return_tick += 1;
                    block.metadata_on_acquired(self.return_tick);
                    Some(block)
                }
                None => {
                    panic!(
                        "Block from priority set not found in lookup map! Inconsistency detected."
                    );
                }
            }
        } else {
            // No blocks available in either set
            None
        }
    }

    /// Acquires a specified number of free blocks from the pool.
    ///
    /// Checks if enough blocks are available and then calls [`acquire_free_block`] repeatedly.
    ///
    /// # Arguments
    ///
    /// * `count` - The number of free blocks to acquire.
    ///
    /// # Returns
    ///
    /// A [`Result`] containing:
    /// - `Ok(Vec<Block<T, M>>)`: A vector of the acquired blocks if successful.
    /// - `Err(BlockPoolError::InsufficientBlocksAvailable)`: If the requested number
    ///   of blocks is not available, or if an inconsistency occurred during acquisition.
    ///
    /// # Panics
    ///
    /// This function can panic if [`acquire_free_block`] panics due to internal inconsistencies.
    #[instrument(level = "debug", skip(self))]
    pub fn acquire_free_blocks(
        &mut self,
        count: usize,
    ) -> Result<Vec<Block<S, M>>, BlockPoolError> {
        if count == 0 {
            return Ok(Vec::new());
        }

        let mut blocks = Vec::with_capacity(count);

        let available_now = self.uninitialized_set.len() + self.lookup_map.len();
        tracing::debug!(
            available_now,
            requested = count,
            "Attempting to acquire free blocks"
        );

        if count > available_now {
            tracing::debug!(
                available_now,
                requested = count,
                "Insufficient blocks available"
            );
            return Err(BlockPoolError::NotEnoughBlocksAvailable(
                count,
                available_now,
            ));
        }

        for i in 0..count {
            tracing::trace!(current = i + 1, total = count, "Acquiring free block");
            // Directly call the logic in acquire_free_block
            // Note: acquire_free_block has its own instrumentation
            if let Some(block) = self.acquire_free_block() {
                blocks.push(block);
            } else {
                // This should not happen if the initial check passed and there are no concurrent modifications.
                // If it does, it indicates an inconsistency or a logic error.
                tracing::error!(
                    requested = count,
                    acquired = blocks.len(),
                    available_at_start = available_now,
                    current_available = self.uninitialized_set.len() + self.lookup_map.len(),
                    "Insufficient blocks during acquisition loop despite initial check."
                );
                // Return the blocks acquired so far, or handle as an error.
                // For now, we break and return what we have, but decrementing 'available_blocks'
                // needs to account for the actual number acquired.
                // Consider returning an error or panicking in debug.
                break;
            }
        }

        let acquired_count = blocks.len();
        tracing::debug!(
            acquired_count,
            requested = count,
            "Finished acquiring blocks"
        );

        // Check if we got the requested number of blocks
        if acquired_count != count {
            // This path is taken if the loop broke early due to unexpected `None` from acquire_free_block
            // Return an error indicating partial success or failure
            // Depending on the desired behavior, you might return the partial list
            // or a more specific error.
            // For consistency with the original check, let's return an error if count wasn't met.
            return Err(BlockPoolError::NotEnoughBlocksAvailable(
                count,
                blocks.len(),
            ));
        }

        Ok(blocks)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::{
        block_manager::{
            block::{registry::BlockRegistry, state::CompleteState, Blocks, PrivateBlockExt},
            events::NullEventManager,
            layout::{BlockLayout, FullyContiguous, LayoutConfigBuilder},
            storage::tests::{NullDeviceAllocator, NullDeviceStorage},
        },
        tokens::{Token, Tokens},
    };

    use super::*;

    use std::sync::Mutex;

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
    pub struct TestMetadata {
        priority: u32,
        returned_tick: u64,
        acquired_tick: u64,
    }

    impl BlockMetadata for TestMetadata {
        fn on_acquired(&mut self, tick: u64) {
            self.acquired_tick = tick;
        }

        fn on_returned(&mut self, tick: u64) {
            self.returned_tick = tick;
        }

        fn reset_metadata(&mut self) {
            self.priority = 0;
        }
    }

    type TestPriorityKey = PriorityKey<TestMetadata>;

    fn make_priority_key(
        priority: u32,
        returned_tick: u64,
        sequence_hash: SequenceHash,
    ) -> TestPriorityKey {
        TestPriorityKey::new(
            TestMetadata {
                priority,
                returned_tick,
                acquired_tick: 0,
            },
            sequence_hash,
        )
    }

    #[test]
    fn test_priority_key_ord() {
        let mut map = BTreeSet::new();

        let hash1 = SequenceHash::from(1u64);
        let hash2 = SequenceHash::from(2u64);
        let hash3 = SequenceHash::from(3u64);

        map.insert(make_priority_key(0, 2, hash1));
        map.insert(make_priority_key(1, 1, hash2));
        map.insert(make_priority_key(0, 3, hash3));

        // Test popping from the map to verify ordering
        let first_key = map.pop_first().unwrap();
        assert_eq!(first_key.metadata().priority, 0);
        assert_eq!(first_key.metadata().returned_tick, 2);
        assert_eq!(first_key.sequence_hash(), hash1);

        let second_key = map.pop_first().unwrap();
        assert_eq!(second_key.metadata().priority, 0);
        assert_eq!(second_key.metadata().returned_tick, 3);
        assert_eq!(second_key.sequence_hash(), hash3);

        let third_key = map.pop_first().unwrap();
        assert_eq!(third_key.metadata().priority, 1);
        assert_eq!(third_key.metadata().returned_tick, 1);
        assert_eq!(third_key.sequence_hash(), hash2);

        // Map should now be empty
        assert!(map.is_empty());
    }

    // Helper function to create a sequence of tokens
    pub fn create_token_sequence(values: &[u32]) -> Tokens {
        let tokens: Vec<Token> = values.iter().map(|&v| Token::from(v)).collect();
        Tokens::from(tokens)
    }

    /// Creates a block collection with the given number of blocks.
    pub fn create_block_collection(
        num_blocks: usize,
    ) -> Blocks<impl BlockLayout<StorageType = NullDeviceStorage>, TestMetadata> {
        let config = LayoutConfigBuilder::default()
            .num_blocks(num_blocks)
            .num_layers(61)
            .page_size(16)
            .inner_dim(576)
            .build()
            .unwrap();

        let layout = FullyContiguous::allocate(config, &NullDeviceAllocator)
            .expect("Failed to allocate layout/storage");

        Blocks::<_, TestMetadata>::new(layout, 42, 0).unwrap()
    }

    /// Creates a vector of Blocks from a token sequence and block size.
    /// Each block is initialized to the Complete state and then Registered.
    pub fn create_blocks(
        tokens: Tokens,
        block_size: usize,
    ) -> Vec<Block<NullDeviceStorage, TestMetadata>> {
        let (token_blocks, _partial_token_block) =
            tokens.into_sequence(block_size, None).into_parts();
        let num_blocks = token_blocks.len();

        if num_blocks == 0 {
            return Vec::new();
        }

        let mut blocks = create_block_collection(num_blocks).into_blocks().unwrap();

        let event_manager = NullEventManager::new();
        let mut registry = BlockRegistry::new(
            event_manager,
            GlobalRegistry::new(Mutex::new(HashMap::new())),
        );

        // Iterate through the generated TokenBlocks and the template Blocks,
        // setting the state and registering each one.
        for (block, token_block) in blocks.iter_mut().zip(token_blocks.into_iter()) {
            assert!(block.state().is_reset()); // Start with empty blocks
            block.update_state(BlockState::Complete(CompleteState::new(token_block)));
            block
                .register(&mut registry)
                .expect("Failed to register block in test helper");
            assert!(block.state().is_registered()); // Ensure registration worked
        }

        blocks
    }

    pub fn create_block_pool(
        num_blocks: usize,
    ) -> InactiveBlockPool<NullDeviceStorage, TestMetadata> {
        let mut pool = InactiveBlockPool::new();
        let blocks = create_block_collection(num_blocks).into_blocks().unwrap();
        pool.add_blocks(blocks);

        pool
    }

    pub fn acquire_blocks(
        tokens: Tokens,
        block_size: usize,
        pool: &mut InactiveBlockPool<NullDeviceStorage, TestMetadata>,
    ) -> (Vec<Block<NullDeviceStorage, TestMetadata>>, usize) {
        let (mut token_blocks, _partial_token_block) =
            tokens.into_sequence(block_size, None).into_parts();

        let total_complete_blocks = token_blocks.len();

        // this will match the token_blocks to any matching blocks in the inactive pool
        // these blocks have the same sequence hash as the token_blocks, thus no updates are needed
        let mut matched_blocks = pool.match_token_blocks(&token_blocks);
        let matched_block_count = matched_blocks.len();

        let event_manager = NullEventManager::new();
        let mut registry = BlockRegistry::new(event_manager, Arc::new(Mutex::new(HashMap::new())));

        // all matched blocks should be in the complete or registered state
        for block in &mut matched_blocks {
            assert!(block.state().is_registered());
        }

        // drain the matched blocks from the token_blocks
        token_blocks.drain(0..matched_block_count);

        assert_eq!(
            token_blocks.len() + matched_blocks.len(),
            total_complete_blocks
        );

        // try to acquire the remaining blocks
        let mut unmatched_blocks = pool.acquire_free_blocks(token_blocks.len()).unwrap();

        assert_eq!(unmatched_blocks.len(), token_blocks.len());

        for unmatched in &unmatched_blocks {
            assert!(unmatched.state().is_reset());
        }

        for (unmatched, token_block) in unmatched_blocks.iter_mut().zip(token_blocks.into_iter()) {
            assert!(unmatched.state().is_reset());
            unmatched.update_state(BlockState::Complete(CompleteState::new(token_block)));
            unmatched.register(&mut registry).unwrap();
            assert!(unmatched.state().is_registered());
        }

        let mut blocks = matched_blocks;
        blocks.extend(unmatched_blocks);
        (blocks, matched_block_count)
    }

    #[test]
    fn test_block_pool_lifecycle() {
        dynamo_runtime::logging::init();

        const PAGE_SIZE: usize = 2;

        let mut pool = create_block_pool(10);
        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let blocks = pool.acquire_free_blocks(10).unwrap();
        assert_eq!(blocks.len(), 10);
        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 0);

        pool.return_blocks(blocks);

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let tokens = create_token_sequence(&[1, 2, 3, 4]);

        let (blocks, matched_block_count) = acquire_blocks(tokens.clone(), PAGE_SIZE, &mut pool);
        assert_eq!(blocks.len(), 2);
        assert_eq!(matched_block_count, 0);
        assert_eq!(pool.available_blocks(), 8);

        pool.return_blocks(blocks);

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let (blocks, matched_block_count) = acquire_blocks(tokens.clone(), PAGE_SIZE, &mut pool);
        assert_eq!(blocks.len(), 2);
        assert_eq!(matched_block_count, 2);
        assert_eq!(pool.available_blocks(), 8);

        pool.return_blocks(blocks);

        assert_eq!(pool.total_blocks(), 10);
        assert_eq!(pool.available_blocks(), 10);

        let blocks = pool.acquire_free_blocks(10).unwrap();
        for block in &blocks {
            assert!(block.state().is_reset());
        }
    }

    #[test]
    fn test_basic_sequence_matching() {
        let mut pool = InactiveBlockPool::new();

        // Create a sequence of 4 tokens split into blocks of 2
        let sequence = create_token_sequence(&[1, 2, 3, 4]);
        let blocks = create_blocks(sequence, 2);
        assert_eq!(blocks.len(), 2);

        // Match the blocks in sequence
        let hashes: Vec<_> = blocks
            .iter()
            .map(|b| {
                b.sequence_hash()
                    .expect("Block should have a sequence hash in this test")
            })
            .collect();

        // Insert blocks into pool
        pool.add_blocks_with_state(blocks);

        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.available_blocks(), 2);

        // Match the blocks in sequence
        let matched = pool.match_sequence_hashes(hashes.clone());
        assert_eq!(matched.len(), 2);

        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.available_blocks(), 0);

        // Validate the blocks are in the correct order and match the sequence hashes
        assert_eq!(matched[0].sequence_hash().unwrap(), hashes[0]);
        assert_eq!(matched[1].sequence_hash().unwrap(), hashes[1]);

        // Return blocks in reverse order (tail to root)
        pool.return_blocks(matched);

        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.available_blocks(), 2);
    }
}
