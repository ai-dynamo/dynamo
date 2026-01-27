// // SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// // SPDX-License-Identifier: Apache-2.0
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// // http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// //! Prototype KV Manager
// //!
// //! The KV Manager will be linked to three components:
// //! - ForwardPassTask / Scheduler
// //!   - On each forward pass, any slot that has completed a block will:
// //!     - Add the block to the Persistence Engine
// //!     - Acquire a new block to continue generating
// //! - Persistence Engine
// //!   - Will perform copies from GPU memory to CPU memory and possibly CPU memory
// //!     to some global flash storage
// //! - Prefill Descriptor Manager
// //!   - New request that require prefill offload, will acquire leases on any shared
// //!     blocks and any "net new" blocks that need to be populated from the prefill
// //!     instance.
// //!

// use async_trait::async_trait;
// use bytemuck::cast_slice;
// use rayon::prelude::*;
// use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, VecDeque};
// use std::sync::Arc;
// use tokio::{
//     sync::{Mutex, Notify},
//     time::Instant,
// };
// use triton_distributed_llm::kv_router::indexer::compute_block_hash;
// use triton_distributed_llm::kv_router::protocols::LocalBlockHash;
// use dynamo_runtime::utils::pool::{
//     Pool, PoolExt, PoolItem, PoolValue, Returnable, SharedPoolItem,
// };

// pub trait Storage {}

// pub type BlockHash = u64;
// pub type SequenceHash = u64;
// pub type Token = u32;

// pub struct Tokens(Vec<Token>);

// pub struct TokenBlock {
//     tokens: Tokens,
//     sequence_hash: SequenceHash,
//     block_hash: LocalBlockHash,
//     sequence_position: u32,
//     priority: Option<u8>,
//     reserved_deadline: Option<Instant>,
// }

// impl Tokens {
//     pub fn blocks(&self, block_size: usize) -> Vec<TokenBlock> {
//         // split the tokens into blocks of the given size
//         // todo: determine how and when to parallelize the block creation
//         //       we can hash the local chunks in parallel
//         // Use rayon's parallel iterator to process chunks in parallel
//         self.0
//             .chunks_exact(block_size)
//             .par_iter()
//             .map(|chunk| TokenBlock {
//                 tokens: Tokens(chunk.to_vec()),
//                 sequence_hash: 0,
//                 block_hash: compute_block_hash(cast_slice(chunk)),
//                 sequence_position: 0,
//                 priority: None,
//                 reserved_deadline: None,
//             })
//             .collect()
//     }
// }

// pub struct KvBlock<T: Storage> {
//     sequence_hash: SequenceHash,
//     block_hash: BlockHash,
//     sequence_position: u32,
//     reserved_deadline: Option<Instant>,
//     storage: Arc<T>,
// }

// pub struct SampleKvStorage {}
// impl Storage for SampleKvStorage {}

// pub type Block = KvBlock<SampleKvStorage>;

// impl Returnable for Block {}

// pub type UniqueBlock = PoolItem<Block, Pool<Block>>;
// pub type SharedBlock = SharedPoolItem<Block, Pool<Block>>;

// /// A wrapper around a time-critical item that will determine the amount of elapsed/walltime
// /// since the item was created. The `deadline` is optional and if not set, the item will be
// /// considered to have no time constraints. If the `deadline` is set, the item will be will
// /// increment a [prometheus::Counter] if the deadline is exceeded.
// ///
// /// In this manner, we can monitor the time-criticality of the item and take action if it is
// /// taking too long to process.
// // pub struct TimeCritical<T> {
// //     // pub timestamp: Instant,
// //     // pub item: T,
// //     // pub deadline: Option<Instant>,
// // }

// pub struct Sequence {
//     tokens: Vec<u32>,
//     shared_blocks: Vec<SharedBlock>,
//     current_block: UniqueBlock,
// }

// /// Adapt the KvIndexer to hold Block information
// pub struct DeviceRadixTree {}

// /// Adapt the KvIndexer to hold Block information
// pub struct HostRadixTree {}

// /// Owner of the radix trees and the block pool
// pub struct KvBlockManager {}

// /// The [Scheduler] is responsible for determining which [Sequence] objects should be
// /// scheduled for the next forward pass.
// ///
// /// The [Scheduler] will prepare a [Sequence] object for each request and pass that [Sequence]
// /// to either the [ForwardPassEngine] or the [PrefillHandler] depending the size of the
// /// ISL and "net-new" tokens that need to be prefilled to the [Sequence].
// ///
// /// The [Scheduler] has have multiple [Sequences][Sequence] offloaded to the [PrefillHandler];
// /// however, some care needs to be taken that that value is not "too large" as the blocks
// /// held by the [Sequence] can not be reused or repurposed by eviction.
// pub struct Scheduler {
//     // slots: BTreeMap<u64, Sequence>,
//     // pending: VecDeque<Sequencd>,
// }

// /// The [ForwardPassEngine] is responsible for scheduling the forward pass of the model.
// /// It will receive requests from the scheduler that will have the set of SharedBlocks that
// /// associated with the current request tied to a Sequence object.
// ///
// /// The [ForwardPassEngine] appends new tokens to the current block of the [Sequence]. When
// /// the current block is full, it is converted to an immutable [SharedBlock] and a copy/clone
// /// is passed to the [PersistenceEngine] via an mpsc::Sender<TimeCritical<SharedBlock>>.
// ///
// /// The [ForwardPassEngine] should spawn async tasks per forward pass to evaluate the potential
// /// of each [Sequence] and determine how many blocks it could return to the [FreePool] if it was
// /// evicted.
// ///
// /// We only want to evict a [Sequence] if it can free enough blocks to be worth the overhead of
// /// evicting it and most critically, that we have persisted all evicted blocks in host memory.
// /// This will avoid the need to re-prefill the blocks when the sequence is rescheduled.
// ///
// /// The [ForwardPassEngine] should also evaluate the potential of each [Sequence] to be
// /// prefilled and if so, it will return a [PrefillHandler] to the caller.
// pub struct ForwardPassEngine {
//     // scheduler: Scheduler,
//     // kv_manager: KvBlockManager,
//     // persistence_engine: PersistenceEngine,
// }

// /// The [PersistenceEngine] is responsible for copying blocks from GPU memory to
// /// to either host memory or some form of persistent storage.
// ///
// /// The [PersistenceEngine] will have a mpsc receiver of SharedBlock. Each block can
// /// be handled independently and freed after the copy is complete.
// ///
// /// We must time each SharedBlock as it enters the channel, so perhaps we wrap the incoming
// /// SharedBlock in a timestamped context.
// ///
// /// Holding SharedBlocks forbids their reuse, so we need to carefully and accurately monitor
// /// the state of this engine so it is not starving the ForwardPass [Scheduler] of free blocks.
// pub struct PersistenceEngine {}

// /// The [PrefillHandler] is responsible for acquiring blocks from the [KvBlockManager] for a
// /// given request. The input sequence length will be evaluated and two sets of blocks will be
// /// returned to the caller:
// ///   - Vec<SharedBlock>
// ///   - Vec<UniqueBlock>
// ///
// /// The `Vec<SharedBlock>` are the blocks that matched inflight radix tree. By acquiring a
// /// [SharedBlock], this ensure that the blocks cannot be returned to the [FreePool].
// ///
// /// The `Vec<UniqueBlock>` are the new blocks that are not present in the inflight radix tree
// /// which need to be prefilled. The decision to prefill locally via chunking of to offload to
// /// dedicated prefill workers can be made once the target destinations for the KV are determined.
// pub struct PrefillHandler {}

// /// The [MigrationEngine] is responsible for migrating blocks from one physical machine to another.
// /// In an ideal world, this transfer is over NVLink or ConnectX InfiniBand; however, any reasonably
// /// fast transfer will suffice.
// ///
// /// The [MigrationEngine] spawns tasks that operate in two paradigms:
// /// - RDMA Passive Source: The task will acquire [SharedBlocks][SharedBlock] from the [KvBlockManager]
// ///   and hold them until a RDMA GET COMPLETION notification is received. Essentially, the task which
// ///   holds the [SharedBlocks][SharedBlock] is simply responsible for ensuring the memory is pinned
// ///   and not returned to the [FreePool] over the duration of the RDMA GET.
// /// - RDMA Active Puller: The task will receive a set of [SharedBlocks][SharedBlock]. The block list
// ///   is a set of block_ids and a remote target. The task will initiate the RDMA GETs via the NIXL
// ///   library and then wait for completion. Upon completion, and event or active message event will
// ///   be triggered on each RDMA Passive Source to trigger task completion and resource dereferencing.
// ///
// pub struct MigrationEngine {}

// // when in a hashset, PriorityBlockReference must be unique by block_id and sorted by:
// // - priority (lowest to highest)
// // - sequence_id (highest to lowest)
// //
// // - all lower priority items must be evicted before higher priority items
// // - all items with the same priority must be evicted in sequence_id order with the highest sequence
// //   position evicted first
// //
// // when a sequences must have priorities that are ordered, you can not have a block with a lower sequence_id
// // and a lower priority.  the same is true for deadlines.
// #[derive(Debug, Clone, Eq)]
// struct PriorityBlockReference {
//     block_id: SequenceHash,
//     sequence_position: u32,
//     priority: u8,
// }

// struct TimeAwareBlockReference {
//     block_id: SequenceHash,
//     sequence_position: u32,
//     evict_deadline: Instant,
//     priority: u8,
// }

// impl PartialEq for PriorityBlockReference {
//     fn eq(&self, other: &Self) -> bool {
//         self.block_id == other.block_id
//     }
// }

// impl std::hash::Hash for PriorityBlockReference {
//     fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
//         self.block_id.hash(state);
//     }
// }

// // Example usage:
// // let priority_set: HashSet<PriorityBlockReference> = HashSet::new();
// //
// // // To get items in sequence_id order:
// // let mut sorted_refs: Vec<&PriorityBlockReference> = priority_set.iter().collect();
// // sorted_refs.sort_by(|a, b| a.sequence_id.cmp(&b.sequence_id));

// // A key that defines the ordering
// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
// struct PriorityKey {
//     // For PriorityReference
//     priority: u8,
//     sequence_position: u32,
//     // Unique identifier to break ties and ensure uniqueness
//     block_hash: BlockHash,
// }

// impl PriorityKey {
//     fn new_priority(block: &Block, priority: u8) -> Self {
//         Self {
//             priority,
//             sequence_position: block.sequence_position,
//             block_hash: block.block_hash,
//         }
//     }
// }

// // A key that defines deadline-based ordering
// //
// // Sort by deadline, then priority, then sequence_position, then sequence_hash
// #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
// struct DeadlineKey {
//     deadline: Instant,
//     priority: u8,
//     sequence_position: u32,
//     sequence_hash: SequenceHash,
// }

// impl DeadlineKey {
//     fn new_deadline(block: &Block, priority: u8) -> Self {
//         Self {
//             deadline: block
//                 .reserved_deadline
//                 .unwrap_or_else(|| Instant::now() + std::time::Duration::from_secs(u64::MAX)),
//             priority,
//             sequence_position: block.sequence_position,
//             blocksequence_hash_hash: block.sequence_hash,
//         }
//     }
// }

// // Define a struct that combines ordered access with direct lookup
// #[derive(Default)]
// pub struct OrderedLookupSet {
//     // Direct lookup by sequence_hash
//     lookup_map: HashMap<SequenceHash, PoolValue<Block>>,

//     // Ordered by priority
//     priority_set: BTreeMap<PriorityKey, SequenceHash>,

//     // Ordered by deadline
//     deadline_set: BTreeMap<DeadlineKey, SequenceHash>,
// }

// impl<T> OrderedLookupSet {
//     // Insert an item with a given key and sequence_hash
//     pub fn insert(&mut self, key: OrderKey, sequence_hash: SequenceHash, item: T) {
//         // Add to the ordered set
//         self.ordered_set.insert(key.clone(), item);

//         // Add to the lookup map
//         self.lookup_map.entry(sequence_hash).or_default().push(key);
//     }

//     // Remove an item by its key
//     pub fn remove_by_key(&mut self, key: &OrderKey) -> Option<T> {
//         self.ordered_set.remove(key)
//     }

//     // Remove an item by sequence_hash and block_hash
//     pub fn remove_by_hash(
//         &mut self,
//         sequence_hash: SequenceHash,
//         block_hash: BlockHash,
//     ) -> Option<T> {
//         // Find the key in the lookup map
//         if let Some(keys) = self.lookup_map.get_mut(&sequence_hash) {
//             // Find the key with the matching block_hash
//             if let Some(pos) = keys.iter().position(|k| k.block_hash == block_hash) {
//                 // Remove the key from the lookup map
//                 let key = keys.remove(pos);

//                 // If this was the last key for this sequence_hash, remove the entry
//                 if keys.is_empty() {
//                     self.lookup_map.remove(&sequence_hash);
//                 }

//                 // Remove and return the item from the ordered set
//                 return self.ordered_set.remove(&key);
//             }
//         }
//         None
//     }

//     // Pop the highest priority item (first in order)
//     pub fn pop_first(&mut self) -> Option<(OrderKey, T)> {
//         if let Some((key, item)) = self.ordered_set.first_key_value() {
//             let key_clone = key.clone();
//             let sequence_hash = self.get_sequence_hash(&key_clone)?;

//             // Remove from the ordered set
//             let item = self.ordered_set.remove(&key_clone)?;

//             // Remove from the lookup map
//             if let Some(keys) = self.lookup_map.get_mut(&sequence_hash) {
//                 if let Some(pos) = keys.iter().position(|k| k == &key_clone) {
//                     keys.remove(pos);

//                     // If this was the last key for this sequence_hash, remove the entry
//                     if keys.is_empty() {
//                         self.lookup_map.remove(&sequence_hash);
//                     }
//                 }
//             }

//             Some((key_clone, item))
//         } else {
//             None
//         }
//     }

//     // Helper method to find the sequence_hash for a key
//     fn get_sequence_hash(&self, key: &OrderKey) -> Option<SequenceHash> {
//         for (hash, keys) in &self.lookup_map {
//             if keys.iter().any(|k| k == key) {
//                 return Some(*hash);
//             }
//         }
//         None
//     }

//     // Get all items for a sequence_hash
//     pub fn get_by_sequence_hash(&self, sequence_hash: SequenceHash) -> Vec<&T> {
//         if let Some(keys) = self.lookup_map.get(&sequence_hash) {
//             keys.iter()
//                 .filter_map(|key| self.ordered_set.get(key))
//                 .collect()
//         } else {
//             Vec::new()
//         }
//     }
// }

// // Now update the AvailableBlocks implementation
// #[derive(Debug, Clone, Default)]
// pub struct AvailableBlocks {
//     // Map from sequence_hash to blocks
//     sequence_map: BTreeMap<SequenceHash, Vec<UniqueBlock>>,
//     // Ordered by priority with lookup by sequence_hash
//     priority_set: OrderedLookupSet<UniqueBlock>,
//     // Ordered by deadline with lookup by sequence_hash
//     deadline_set: OrderedLookupSet<UniqueBlock>,
// }

// impl AvailableBlocks {
//     // Add a block to the available blocks
//     pub fn add_block(&mut self, block: UniqueBlock) {
//         let block_ref = &*block; // Deref to get the Block
//         let sequence_hash = block_ref.sequence_hash;
//         let priority = calculate_priority(block_ref);

//         // Create keys for our sets
//         let priority_key = OrderKey::new_priority(block_ref, priority);
//         let deadline_key = DeadlineKey::new_deadline(block_ref, priority);

//         // Add to the sequence map
//         self.sequence_map
//             .entry(sequence_hash)
//             .or_default()
//             .push(block.clone());

//         // Add to our sets
//         self.priority_set
//             .insert(priority_key, sequence_hash, block.clone());
//         // For deadline_set, we'd need a similar implementation with DeadlineKey
//         // self.deadline_set.insert(deadline_key, sequence_hash, block);
//     }

//     // Get the highest priority block
//     pub fn pop_highest_priority(&mut self) -> Option<UniqueBlock> {
//         if let Some((key, block)) = self.priority_set.pop_first() {
//             // Remove from sequence map
//             if let Some(blocks) = self.sequence_map.get_mut(&block.sequence_hash) {
//                 if let Some(pos) = blocks.iter().position(|b| b.block_hash == key.block_hash) {
//                     blocks.remove(pos);
//                 }
//             }

//             // Remove from deadline set
//             // self.deadline_set.remove_by_hash(block.sequence_hash, key.block_hash);

//             Some(block)
//         } else {
//             None
//         }
//     }

//     // Get all blocks for a sequence
//     pub fn get_blocks_by_sequence(&self, sequence_hash: SequenceHash) -> Vec<&UniqueBlock> {
//         self.priority_set.get_by_sequence_hash(sequence_hash)
//     }
// }

// // Helper function to calculate priority based on block details
// fn calculate_priority(block: &Block) -> u8 {
//     // Implement your priority calculation logic here
//     0 // Default priority
// }

// async fn available_block_progress_engine(
//     request_rx: Receiver<BlockRequest>,
//     return_rx: Receiver<B>,
// ) {
//     let available_blocks = AvailableBlocks::default();
// }
