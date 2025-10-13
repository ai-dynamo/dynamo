// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # KV Block Available Pool
//!
//! The Available Pool manages KV blocks that are not actively in use but retain their previous state.
//!
//! ## Key Features:
//!
//! - **State Preservation**: Blocks in the pool maintain their previous state and can be reused.
//!
//! - **Priority-Based FIFO**: Blocks are returned in first-in, first-out order within their priority levels.
//!   Lower priority values are processed first, allowing important blocks to be retained longer.
//!
//! - **State Matching**: Blocks can be matched against their previous state instead of being taken randomly,
//!   enabling efficient reuse of blocks with specific sequence hashes.
//!
//! - **Priority Management**: Priorities can be applied to blocks based on their sequence hash,
//!   requiring some external knowledge of the block's characteristics.
//!
//! - **State Management**: Blocks can have their states wiped clean/reset individually or in groups.
//!   The entire pool can also be reset as needed.
//!
//! - **Synchronization**: Fence operations ensure all higher priority operations have completed
//!   before proceeding. Note that this is not a true fence - higher priority operations issued
//!   after the fence will still be processed before the fence completes.

use std::sync::atomic::Ordering;

use dynamo_runtime::utils::pool::ReturnHandle;
use tokio::{
    sync::{mpsc, oneshot},
    task::JoinHandle,
};

use super::*;

pub struct AvailableBlocks {
    match_tx: mpsc::UnboundedSender<MatchRequest>,
    control_tx: mpsc::UnboundedSender<ControlRequest>,
    fence_tx: mpsc::UnboundedSender<oneshot::Sender<()>>,
    return_handle: Arc<ReturnHandleImpl>,
    total_blocks: Arc<AtomicU64>,
    available_blocks: Arc<AtomicU64>,
    join_handle: JoinHandle<()>,
}

impl AvailableBlocks {
    pub fn total_blocks(&self) -> u64 {
        self.total_blocks.load(Ordering::SeqCst)
    }

    pub fn available_blocks(&self) -> u64 {
        self.available_blocks.load(Ordering::SeqCst)
    }

    pub fn is_active(&self) -> bool {
        !self.join_handle.is_finished()
    }

    pub async fn match_blocks(&self, hashes: Vec<SequenceHash>) -> Result<Vec<PoolItem<KvBlock>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::MatchMultiple(MatchMultiple {
                hashes,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send match request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    pub async fn match_token_blocks(
        &self,
        token_blocks: &[TokenBlock],
    ) -> Result<Vec<PoolItem<KvBlock>>> {
        let hashes: Vec<u64> = token_blocks.iter().map(|b| b.sequence_hash()).collect();
        self.match_blocks(hashes).await
    }

    pub async fn take_blocks(&self, count: u32) -> Result<Vec<PoolItem<KvBlock>>> {
        let (tx, rx) = oneshot::channel();
        if self
            .match_tx
            .send(MatchRequest::Take(Take {
                count,
                return_handle: self.return_handle.clone(),
                tx,
            }))
            .is_err()
        {
            raise!("failed to send take request; channel closed");
        }

        let matched_blocks = rx.await?;
        Ok(matched_blocks)
    }

    pub async fn insert(&self, block: KvBlock) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::Insert(InsertControl { block, tx }))
            .is_err()
        {
            raise!("failed to send insert request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn update_single(&self, update: UpdateBlock) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::UpdateSingle(UpdateSingleControl {
                update,
                tx,
            }))
            .is_err()
        {
            raise!("failed to send update single request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn update_multiple(&self, updates: Vec<UpdateBlock>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::UpdateMultiple(UpdateMultipleControl {
                updates,
                tx,
            }))
            .is_err()
        {
            raise!("failed to send update multiple request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn reset(&self, sequence_hashes: Vec<SequenceHash>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::Reset(ResetControl {
                sequence_hashes,
                tx,
            }))
            .is_err()
        {
            raise!("failed to send reset request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn reset_all(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self
            .control_tx
            .send(ControlRequest::ResetAll(ResetAllControl { tx }))
            .is_err()
        {
            raise!("failed to send reset all request; channel closed");
        }
        rx.await?;
        Ok(())
    }

    pub async fn fence(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        if self.fence_tx.send(tx).is_err() {
            raise!("failed to send fence request; channel closed");
        }
        rx.await?;
        Ok(())
    }
}

struct ReturnHandleImpl {
    return_tx: mpsc::UnboundedSender<PoolValue<KvBlock>>,
}

impl ReturnHandle<KvBlock> for ReturnHandleImpl {
    fn return_to_pool(&self, value: PoolValue<KvBlock>) {
        if self.return_tx.send(value).is_err() {
            log::trace!("Failed to return block to pool");
        }
    }
}

impl AvailableBlocks {
    pub async fn new() -> Self {
        let (match_tx, match_rx) = mpsc::unbounded_channel();
        let (return_tx, return_rx) = mpsc::unbounded_channel();
        let (control_tx, control_rx) = mpsc::unbounded_channel();
        let (fence_tx, fence_rx) = mpsc::unbounded_channel();

        let total_blocks = Arc::new(AtomicU64::new(0));
        let available_blocks = Arc::new(AtomicU64::new(0));

        let return_tx_clone = return_tx.clone();
        let return_handle = Arc::new(ReturnHandleImpl {
            return_tx: return_tx_clone,
        });

        let join_handle = tokio::spawn(progress_engine(
            match_rx,
            return_rx,
            control_rx,
            fence_rx,
            total_blocks.clone(),
            available_blocks.clone(),
        ));

        Self {
            match_tx,
            control_tx,
            fence_tx,
            return_handle,
            total_blocks,
            available_blocks,
            join_handle,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PriorityKey {
    priority: u32,
    return_tick: u64,
    sequence_hash: SequenceHash,
}

// customize ord and partial ord for to store first by priority (lowest to highest), then by return_tick (lowest to highest)
impl PartialOrd for PriorityKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            .then(self.return_tick.cmp(&other.return_tick))
    }
}

impl From<&KvBlock> for PriorityKey {
    fn from(block: &KvBlock) -> Self {
        Self {
            priority: block.priority,
            return_tick: block.return_tick,
            sequence_hash: block.token_block.sequence_hash(),
        }
    }
}

#[derive(Default)]
struct AvailableBlocksState {
    // Direct lookup by sequence_hash
    lookup_map: HashMap<SequenceHash, PoolValue<KvBlock>>,

    // // Ordered by timestamp (oldest first)
    priority_set: BTreeMap<PriorityKey, SequenceHash>,

    // Fully Uninitialized
    uninitialized_set: VecDeque<PoolValue<KvBlock>>,

    // Return Tick
    return_tick: u64,

    // Total blocks
    total_blocks: Arc<AtomicU64>,

    // Available blocks
    available_blocks: Arc<AtomicU64>,
}

impl AvailableBlocksState {
    fn new(total_blocks: Arc<AtomicU64>, available_blocks: Arc<AtomicU64>) -> Self {
        Self {
            lookup_map: HashMap::new(),
            priority_set: BTreeMap::new(),
            uninitialized_set: VecDeque::new(),
            return_tick: 0,
            total_blocks,
            available_blocks,
        }
    }
    // Insert an item with a given key and sequence_hash
    fn insert(&mut self, block: PoolValue<KvBlock>) {
        let sequence_hash = block.token_block.sequence_hash();
        log::debug!(sequence_hash, "inserting block into available blocks");

        // If we already have an entry for this sequence hash, we need to move it to the uninitialized set
        // the lookup map has only one entry per sequence hash
        if self.lookup_map.contains_key(&sequence_hash) || sequence_hash == 0u64 {
            log::debug!(sequence_hash, "inserted block to uninitialized set");
            self.uninitialized_set.push_back(block);
            return;
        }

        // Insert into timestamp set
        let key = PriorityKey::from(&*block);
        let check_multiple_entries = self.priority_set.insert(key, sequence_hash);
        assert!(
            check_multiple_entries.is_none(),
            "fatal error: multiple entries for the same sequence hash in timestamp set"
        );

        // Add to the lookup map
        let check_multiple_entries = self.lookup_map.insert(sequence_hash, block);
        assert!(
            check_multiple_entries.is_none(),
            "fatal error: multiple entries for the same sequence hash in lookup map"
        );
    }

    fn take_with_sequence_hash(
        &mut self,
        sequence_hash: SequenceHash,
    ) -> Option<PoolValue<KvBlock>> {
        match self.lookup_map.remove(&sequence_hash) {
            Some(block) => {
                // Remove from timestamp set
                self.priority_set.remove(&PriorityKey::from(&*block));
                Some(block)
            }
            None => None,
        }
    }

    fn match_hashes(
        &mut self,
        hashes: Vec<SequenceHash>,
        return_handle: Arc<ReturnHandleImpl>,
    ) -> Vec<PoolItem<KvBlock>> {
        let mut matched_blocks = Vec::with_capacity(hashes.len());

        for hash in hashes {
            if let Some(block) = self.take_with_sequence_hash(hash) {
                matched_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        self.available_blocks
            .fetch_sub(matched_blocks.len() as u64, Ordering::SeqCst);

        matched_blocks
    }

    fn handle_match_single(&mut self, match_single: MatchSingle) {
        let (hash, return_handle, rx) = match_single.dissolve();

        let matched_blocks = self.match_hashes(vec![hash], return_handle);
        let optional_single = matched_blocks.into_iter().next();

        // Send the result back through the channel
        if rx.send(optional_single).is_err() {
            log::trace!("Failed to send matched block to requester");
        }
    }

    fn handle_match_multiple(&mut self, match_multiple: MatchMultiple) {
        let (hashes, return_handle, rx) = match_multiple.dissolve();

        let matched_blocks = self.match_hashes(hashes, return_handle);

        // Send the matched blocks back through the channel
        if rx.send(matched_blocks).is_err() {
            log::trace!("Failed to send matched blocks to requester");
        }
    }

    fn take(&mut self) -> Option<PoolValue<KvBlock>> {
        // First try uninitialized blocks - these are often part of sequences
        // that have been arranged in the correct order
        if let Some(block) = self.uninitialized_set.pop_front() {
            return Some(block);
        }

        // if we have blocks in the priority set, pop the first (it's sorted by priority)
        // a fatal error will occur if the block is not found in the lookup map
        if let Some((_key, sequence_hash)) = self.priority_set.pop_first() {
            let block = match self.lookup_map.remove(&sequence_hash) {
                Some(block) => block,
                None => {
                    panic!("block from priority set not found in lookup map");
                }
            };

            return Some(block);
        }

        None
    }

    fn handle_take(&mut self, take: Take) {
        let (count, return_handle, tx) = take.dissolve();

        let mut taken_blocks = Vec::with_capacity(count as usize);

        for _ in 0..count {
            if let Some(block) = self.take() {
                taken_blocks.push(self.create_pool_item(block, return_handle.clone()));
            } else {
                break;
            }
        }

        self.available_blocks.fetch_sub(
            taken_blocks.len() as u64,
            std::sync::atomic::Ordering::SeqCst,
        );

        // Send the result back through the channel
        if tx.send(taken_blocks).is_err() {
            log::trace!("Failed to send matched blocks to requester");
        }
    }

    fn handle_match_request(&mut self, match_request: MatchRequest) {
        match match_request {
            MatchRequest::MatchSingle(match_single) => self.handle_match_single(match_single),
            MatchRequest::MatchMultiple(match_multiple) => {
                self.handle_match_multiple(match_multiple)
            }
            MatchRequest::Take(take) => self.handle_take(take),
        }
    }

    fn handle_control_request(&mut self, control_request: ControlRequest) {
        match control_request {
            ControlRequest::Insert(insert) => {
                let (block, tx) = insert.dissolve();
                self.handle_insert(block);
                if tx.send(()).is_err() {
                    log::trace!("Failed to send insert ack; receiver dropped");
                }
            }
            ControlRequest::UpdateSingle(update_single) => {
                let (update, tx) = update_single.dissolve();
                self.handle_update_single(update);
                if tx.send(()).is_err() {
                    log::trace!("Failed to send update single ack; receiver dropped");
                }
            }
            ControlRequest::UpdateMultiple(update_multiple) => {
                let (updates, tx) = update_multiple.dissolve();
                self.handle_update_multiple(updates);
                if tx.send(()).is_err() {
                    log::trace!("Failed to send update multiple ack; receiver dropped");
                }
            }
            ControlRequest::Reset(reset) => {
                let (sequence_hashes, tx) = reset.dissolve();
                self.handle_reset(sequence_hashes);
                if tx.send(()).is_err() {
                    log::trace!("Failed to send reset ack; receiver dropped");
                }
            }
            ControlRequest::ResetAll(reset_all) => {
                let tx = reset_all.dissolve();
                self.handle_reset_all();
                if tx.send(()).is_err() {
                    log::trace!("Failed to send reset all ack; receiver dropped");
                }
            }
        }
    }
    fn handle_insert(&mut self, block: KvBlock) {
        self.available_blocks
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.total_blocks
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.return_tick += 1;

        // update the return tick
        let mut block = block;
        block.return_tick = self.return_tick;

        self.insert(PoolValue::Direct(block));
    }
    fn handle_return(&mut self, block: PoolValue<KvBlock>) {
        self.available_blocks
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.return_tick += 1;

        // update the return tick
        let mut block = block;
        block.return_tick = self.return_tick;

        self.insert(block);
    }
    fn handle_update_single(&mut self, update: UpdateBlock) {
        self.update_block(vec![update]);
    }

    fn handle_update_multiple(&mut self, updates: Vec<UpdateBlock>) {
        self.update_block(updates);
    }

    fn update_block(&mut self, updates: Vec<UpdateBlock>) {
        for update in updates {
            if let Some(mut block) = self.take_with_sequence_hash(update.hash) {
                if let Some(priority) = update.priority {
                    block.priority = priority;
                }

                // if let Some(deadline) = update.deadline {
                //     block.set_deadline(deadline);
                // }

                self.insert(block);
            }
        }
    }

    fn handle_reset(&mut self, sequence_hashes: Vec<SequenceHash>) {
        for hash in sequence_hashes {
            if let Some(mut block) = self.take_with_sequence_hash(hash) {
                block.reset();
                self.insert(block);
            }
        }
    }

    fn handle_reset_all(&mut self) {
        // for all blocks in the priority set, reset them
        while let Some((_key, sequence_hash)) = self.priority_set.pop_first() {
            if let Some(mut block) = self.lookup_map.remove(&sequence_hash) {
                block.reset();
                self.insert(block);
            } else {
                panic!("block from priority set not found in lookup map");
            }
        }
    }
}

#[async_trait]
impl PoolExt<KvBlock> for AvailableBlocksState {}

#[derive(Dissolve)]
pub struct MatchSingle {
    hash: SequenceHash,
    return_handle: Arc<ReturnHandleImpl>,
    tx: oneshot::Sender<Option<UniqueBlock>>,
}

#[derive(Dissolve)]
pub struct MatchMultiple {
    hashes: Vec<SequenceHash>,
    return_handle: Arc<ReturnHandleImpl>,
    tx: oneshot::Sender<Vec<UniqueBlock>>,
}

#[derive(Dissolve)]
pub struct Take {
    count: u32,
    return_handle: Arc<ReturnHandleImpl>,
    tx: oneshot::Sender<Vec<UniqueBlock>>,
}

pub enum MatchRequest {
    MatchSingle(MatchSingle),
    MatchMultiple(MatchMultiple),
    Take(Take),
}

pub struct UpdateBlock {
    hash: SequenceHash,
    priority: Option<u32>,
}

#[derive(Dissolve)]
pub struct InsertControl {
    block: KvBlock,
    tx: oneshot::Sender<()>,
}

#[derive(Dissolve)]
pub struct UpdateSingleControl {
    update: UpdateBlock,
    tx: oneshot::Sender<()>,
}

#[derive(Dissolve)]
pub struct UpdateMultipleControl {
    updates: Vec<UpdateBlock>,
    tx: oneshot::Sender<()>,
}

#[derive(Dissolve)]
pub struct ResetControl {
    sequence_hashes: Vec<SequenceHash>,
    tx: oneshot::Sender<()>,
}

#[derive(Dissolve)]
pub struct ResetAllControl {
    tx: oneshot::Sender<()>,
}

pub enum ControlRequest {
    Insert(InsertControl),
    UpdateSingle(UpdateSingleControl),
    UpdateMultiple(UpdateMultipleControl),
    Reset(ResetControl),
    ResetAll(ResetAllControl),
}

pub async fn progress_engine(
    match_rx: mpsc::UnboundedReceiver<MatchRequest>,
    return_rx: mpsc::UnboundedReceiver<PoolValue<KvBlock>>,
    ctrl_rx: mpsc::UnboundedReceiver<ControlRequest>,
    fence_rx: mpsc::UnboundedReceiver<oneshot::Sender<()>>,
    total_blocks: Arc<AtomicU64>,
    available_blocks: Arc<AtomicU64>,
) {
    let mut match_rx = match_rx;
    let mut return_rx = return_rx;
    let mut ctrl_rx = ctrl_rx;
    let mut fence_rx = fence_rx;

    let mut state = AvailableBlocksState::new(total_blocks, available_blocks);

    loop {
        tokio::select! {
            biased;

            Some(match_req) = match_rx.recv(), if !match_rx.is_closed() => {
                state.handle_match_request(match_req);
            }

            Some(block) = return_rx.recv(), if !return_rx.is_closed() => {
                state.handle_return(block);
            }

            Some(req) = ctrl_rx.recv(), if !ctrl_rx.is_closed() => {
                state.handle_control_request(req);
            }

            Some(tx) = fence_rx.recv() => {
                if tx.send(()).is_err() {
                    log::trace!("Failed to send fence ack; receiver dropped");
                }
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::tokens::Token;

    use super::*;

    #[test]
    fn test_priority_key_ord() {
        let mut map = BTreeMap::new();
        let hash1 = SequenceHash::from(1u64);
        let hash2 = SequenceHash::from(2u64);
        let hash3 = SequenceHash::from(3u64);

        map.insert(
            PriorityKey {
                priority: 0,
                return_tick: 1,
                sequence_hash: hash1,
            },
            "value1",
        );
        map.insert(
            PriorityKey {
                priority: 1,
                return_tick: 0,
                sequence_hash: hash2,
            },
            "value2",
        );
        map.insert(
            PriorityKey {
                priority: 0,
                return_tick: 2,
                sequence_hash: hash3,
            },
            "value3",
        );

        let keys: Vec<_> = map.keys().collect();

        // Priority is the primary sort key (0 before 1)
        assert_eq!(keys[0].priority, 0);
        assert_eq!(keys[1].priority, 0);
        assert_eq!(keys[2].priority, 1);

        // For same priority, return_tick is the secondary sort key
        assert_eq!(keys[0].return_tick, 1);
        assert_eq!(keys[1].return_tick, 2);

        // Test popping from the map to verify ordering
        let (first_key, first_value) = map.pop_first().unwrap();
        assert_eq!(first_key.priority, 0);
        assert_eq!(first_key.return_tick, 1);
        assert_eq!(first_key.sequence_hash, hash1);
        assert_eq!(first_value, "value1");

        let (second_key, second_value) = map.pop_first().unwrap();
        assert_eq!(second_key.priority, 0);
        assert_eq!(second_key.return_tick, 2);
        assert_eq!(second_key.sequence_hash, hash3);
        assert_eq!(second_value, "value3");

        let (third_key, third_value) = map.pop_first().unwrap();
        assert_eq!(third_key.priority, 1);
        assert_eq!(third_key.return_tick, 0);
        assert_eq!(third_key.sequence_hash, hash2);
        assert_eq!(third_value, "value2");

        // Map should now be empty
        assert!(map.is_empty());
    }

    // Helper function to create a sequence of tokens
    pub fn create_token_sequence(values: &[u32]) -> Tokens {
        let tokens: Vec<Token> = values.iter().map(|&v| Token::from(v)).collect();
        Tokens::from(tokens)
    }

    // Helper to create blocks from a sequence with given size
    pub fn create_blocks(sequence: Tokens, block_size: usize) -> Vec<KvBlock> {
        let (blocks, _) = sequence.into_sequence(block_size).into_parts();
        blocks
            .into_iter()
            .map(|token_block| KvBlock {
                token_block,
                ..Default::default()
            })
            .collect()
    }

    #[tokio::test]
    async fn test_basic_sequence_matching() {
        let pool = AvailableBlocks::new().await;

        // Create a sequence of 4 tokens split into blocks of 2
        let sequence = create_token_sequence(&[1, 2, 3, 4]);
        let blocks = create_blocks(sequence, 2);
        assert_eq!(blocks.len(), 2);

        // Match the blocks in sequence
        let hashes: Vec<_> = blocks
            .iter()
            .map(|b| b.token_block.sequence_hash())
            .collect();

        // Insert blocks into pool
        for block in blocks {
            pool.insert(block).await.unwrap();
        }

        pool.fence().await.unwrap();

        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.available_blocks(), 2);

        // Match the blocks in sequence
        let matched = pool.match_blocks(hashes.clone()).await.unwrap();
        assert_eq!(matched.len(), 2);

        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.available_blocks(), 0);

        // Validate the blocks are in the correct order and match the sequence hashes
        assert_eq!(matched[0].token_block.sequence_hash(), hashes[0]);
        assert_eq!(matched[1].token_block.sequence_hash(), hashes[1]);

        // Return blocks in reverse order (tail to root)
        for block in matched.into_iter().rev() {
            drop(block); // This will trigger return_to_pool
        }

        pool.fence().await.unwrap();

        assert_eq!(pool.total_blocks(), 2);
        assert_eq!(pool.available_blocks(), 2);
    }

    #[tokio::test]
    async fn test_equal_priority_taking() {
        let pool = AvailableBlocks::new().await;

        // Create two sequences with different priorities
        let seq1 = create_token_sequence(&[1, 2, 3, 4]);
        let seq2 = create_token_sequence(&[5, 6, 7, 8]);

        let mut blocks1 = create_blocks(seq1, 2);
        let mut blocks2 = create_blocks(seq2, 2);

        for block in blocks1.iter_mut() {
            block.priority = 1;
        }
        for block in blocks2.iter_mut() {
            block.priority = 1;
        }

        // If priorities were equal, first in, first out would apply

        // Insert Sequence 2
        for block in blocks2.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        // Insert Sequence 1
        for block in blocks1.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        pool.fence().await.unwrap();

        let blocks = pool.take_blocks(4).await.unwrap();
        assert_eq!(blocks.len(), 4);

        // Validate the blocks are in the correct order
        assert_eq!(blocks[0].token_block.tokens()[0], 7);
        assert_eq!(blocks[1].token_block.tokens()[0], 5);
        assert_eq!(blocks[2].token_block.tokens()[0], 3);
        assert_eq!(blocks[3].token_block.tokens()[0], 1);
    }

    #[tokio::test]
    async fn test_priority_taking() {
        let pool = AvailableBlocks::new().await;

        // Create two sequences with different priorities
        let seq1 = create_token_sequence(&[1, 2, 3, 4]);
        let seq2 = create_token_sequence(&[5, 6, 7, 8]);

        let mut blocks1 = create_blocks(seq1, 2);
        let mut blocks2 = create_blocks(seq2, 2);

        for block in blocks1.iter_mut() {
            block.priority = 1;
        }
        for block in blocks2.iter_mut() {
            block.priority = 2;
        }

        // If priorities were equal, first in, first out would apply
        // but here we have a higher priority block first (which are taken last)
        // returned first, but lower priority blocks inserted after
        // we expect the lower priority blocks to be taken first

        // Insert Sequence 2
        for block in blocks2.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        // Insert Sequence 1
        for block in blocks1.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        pool.fence().await.unwrap();

        let blocks = pool.take_blocks(4).await.unwrap();
        assert_eq!(blocks.len(), 4);

        // Validate the blocks are in the correct order
        assert_eq!(blocks[0].token_block.tokens()[0], 3);
        assert_eq!(blocks[1].token_block.tokens()[0], 1);
        assert_eq!(blocks[2].token_block.tokens()[0], 7);
        assert_eq!(blocks[3].token_block.tokens()[0], 5);
    }

    #[tokio::test]
    async fn test_priority_taking_after_update() {
        let pool = AvailableBlocks::new().await;

        // Create two sequences with different priorities
        let seq1 = create_token_sequence(&[1, 2, 3, 4]);
        let seq2 = create_token_sequence(&[5, 6, 7, 8]);

        let mut blocks1 = create_blocks(seq1, 2);
        let mut blocks2 = create_blocks(seq2, 2);

        for block in blocks1.iter_mut() {
            block.priority = 1;
        }
        for block in blocks2.iter_mut() {
            block.priority = 1;
        }

        // record hash of blocks 2
        // insert blocks 2, then blocks 1
        // update priority of blocks 2 to 2 using the update api
        // pull 4 blocks and test order

        let block_hashes = blocks2
            .iter()
            .map(|b| b.token_block.sequence_hash())
            .collect::<Vec<_>>();

        // Insert Sequence 2
        for block in blocks2.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        // Insert Sequence 1
        for block in blocks1.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        pool.fence().await.unwrap();

        // Update priority of blocks 2 to 2
        pool.update_multiple(
            block_hashes
                .into_iter()
                .map(|h| UpdateBlock {
                    hash: h,
                    priority: Some(2),
                })
                .collect(),
        )
        .await
        .unwrap();

        pool.fence().await.unwrap();

        let blocks = pool.take_blocks(4).await.unwrap();
        assert_eq!(blocks.len(), 4);

        // Validate the blocks are in the correct order
        assert_eq!(blocks[0].token_block.tokens()[0], 3);
        assert_eq!(blocks[1].token_block.tokens()[0], 1);
        assert_eq!(blocks[2].token_block.tokens()[0], 7);
        assert_eq!(blocks[3].token_block.tokens()[0], 5);
    }

    #[tokio::test]
    async fn test_reset_all() {
        let pool = AvailableBlocks::new().await;

        // Create two sequences with different priorities
        let seq1 = create_token_sequence(&[1, 2, 3, 4]);
        let seq2 = create_token_sequence(&[5, 6, 7, 8]);

        let mut blocks1 = create_blocks(seq1, 2);
        let mut blocks2 = create_blocks(seq2, 2);

        for block in blocks1.iter_mut() {
            block.priority = 1;
        }

        for block in blocks2.iter_mut() {
            block.priority = 1;
        }

        // record hash of blocks 2
        let block_hashes = blocks2
            .iter()
            .map(|b| b.token_block.sequence_hash())
            .collect::<Vec<_>>();

        // Insert Sequence 2
        for block in blocks2.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        // Insert Sequence 1
        for block in blocks1.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        // Reset All
        pool.reset_all().await.unwrap();
        pool.fence().await.unwrap();

        // Try to match from block 2 hashes, expect no matches
        let matched = pool.match_blocks(block_hashes).await.unwrap();
        assert_eq!(matched.len(), 0);
    }

    #[tokio::test]
    async fn test_reset_block2() {
        let pool = AvailableBlocks::new().await;

        // Create two sequences with different priorities
        let seq1 = create_token_sequence(&[1, 2, 3, 4]);
        let seq2 = create_token_sequence(&[5, 6, 7, 8]);

        let mut blocks1 = create_blocks(seq1, 2);
        let mut blocks2 = create_blocks(seq2, 2);

        for block in blocks1.iter_mut() {
            block.priority = 1;
        }

        for block in blocks2.iter_mut() {
            block.priority = 1;
        }

        // record hash of blocks 2
        let block2_hashes = blocks2
            .iter()
            .map(|b| b.token_block.sequence_hash())
            .collect::<Vec<_>>();

        let block1_hashes = blocks1
            .iter()
            .map(|b| b.token_block.sequence_hash())
            .collect::<Vec<_>>();

        // Insert Sequence 2
        for block in blocks2.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        // Insert Sequence 1
        for block in blocks1.into_iter().rev() {
            pool.insert(block).await.unwrap();
        }

        // Reset Block 2
        pool.reset(block2_hashes.clone()).await.unwrap();
        pool.fence().await.unwrap();

        // Try to match from block 2 hashes, expect no matches
        let matched = pool.match_blocks(block2_hashes).await.unwrap();
        assert_eq!(matched.len(), 0);

        let matched = pool.match_blocks(block1_hashes).await.unwrap();
        assert_eq!(matched.len(), 2);
    }
}
