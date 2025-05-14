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

use crate::mocker::evictor::LRUEvictor;
use crate::mocker::protocols::{MoveBlock, UniqueBlock};
use std::collections::HashMap;
use std::panic;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio::time::Instant;

/// Mock implementation of worker for testing and simulation
pub struct KvManager {
    pub max_capacity: usize,
    pub block_size: usize,
    pub active_blocks: Arc<Mutex<HashMap<UniqueBlock, usize>>>,
    pub inactive_blocks: Arc<Mutex<LRUEvictor<UniqueBlock>>>,
    pub event_tx: mpsc::Sender<MoveBlock>,
    event_loop_handle: JoinHandle<()>,
}

impl KvManager {
    /// Create a new MockWorkers instance
    pub fn new(max_capacity: usize, block_size: usize) -> Self {
        let active_blocks = Arc::new(Mutex::new(HashMap::new()));

        // Initialize LRUEvictor
        let evictor = LRUEvictor::new();
        // No initial blocks, start with empty inactive_blocks
        let inactive_blocks = Arc::new(Mutex::new(evictor));

        // Create mpsc channel for MoveBlock events
        let (event_tx, event_rx) = mpsc::channel::<MoveBlock>(1024);

        // Initialize start_time
        let start_time = Instant::now();

        // Clone Arc references for the event loop
        let active_blocks_clone = active_blocks.clone();
        let inactive_blocks_clone = inactive_blocks.clone();

        // Spawn the event loop
        let event_loop_handle = tokio::spawn(async move {
            let mut event_rx = event_rx;

            while let Some(event) = event_rx.recv().await {
                // Process event based on block_type
                match event {
                    MoveBlock::Use(hashes, _) => {
                        // Always lock active first, then inactive to maintain consistent order
                        let mut active = active_blocks_clone.lock().await;
                        let mut inactive = inactive_blocks_clone.lock().await;

                        for hash in hashes {
                            // First check if it already exists in active blocks
                            if let Some(ref_count) = active.get_mut(&hash) {
                                // Block already active, just increment reference count
                                *ref_count += 1;
                                continue;
                            }

                            // Then check if it exists in inactive and move it to active if found
                            if inactive.remove(&hash) {
                                // Insert into active with reference count 1
                                active.insert(hash, 1);
                                continue;
                            }

                            // Get counts for capacity check (we already have both locks)
                            let active_count = active.len();
                            let inactive_count = inactive.num_objects();

                            // If at max capacity, evict the oldest entry from inactive blocks
                            if active_count + inactive_count >= max_capacity
                                && inactive.evict().is_none()
                            {
                                panic!("max capacity reached and no free blocks left");
                            }

                            // Now insert the new block in active blocks with reference count 1
                            active.insert(hash, 1);
                        }
                    }
                    MoveBlock::Destroy(hashes) => {
                        let mut active = active_blocks_clone.lock().await;
                        // Loop in inverse direction
                        for hash in hashes.into_iter().rev() {
                            active.remove(&hash);
                        }
                    }
                    MoveBlock::Deref(hashes) => {
                        // Always lock active first, then inactive if needed
                        let mut active = active_blocks_clone.lock().await;
                        let mut inactive = inactive_blocks_clone.lock().await;

                        // Loop in inverse direction
                        for hash in hashes.into_iter().rev() {
                            // Decrement reference count and check if we need to move to inactive
                            if let Some(ref_count) = active.get_mut(&hash) {
                                *ref_count -= 1;

                                // If reference count reaches zero, remove from active and move to inactive
                                if *ref_count == 0 {
                                    active.remove(&hash);

                                    // Use monotonic time instead of system time
                                    inactive.insert(hash, start_time.elapsed().as_secs_f64());
                                }
                            }
                        }
                    }
                    MoveBlock::Promote(uuid, hash) => {
                        let mut active = active_blocks_clone.lock().await;

                        let uuid_block = UniqueBlock::PartialBlock(uuid);
                        let hash_block = UniqueBlock::FullBlock(hash);

                        // Check if the UUID block exists in active blocks
                        if let Some(ref_count) = active.remove(&uuid_block) {
                            // Replace with hash block, keeping the same reference count
                            active.insert(hash_block, ref_count);
                        }
                    }
                }
            }
        });

        KvManager {
            max_capacity,
            block_size,
            active_blocks,
            inactive_blocks,
            event_tx,
            event_loop_handle,
        }
    }

    /// Get a clone of the event sender to send MoveBlock events
    pub fn get_event_sender(&self) -> mpsc::Sender<MoveBlock> {
        self.event_tx.clone()
    }

    /// Get the current capacity (active blocks + inactive blocks)
    pub async fn current_capacity(&self) -> usize {
        let active = self.active_blocks.lock().await.len();
        let inactive = self.inactive_blocks.lock().await.num_objects();
        active + inactive
    }

    /// Get the keys of inactive blocks
    pub async fn get_inactive_blocks(&self) -> Vec<UniqueBlock> {
        let inactive = self.inactive_blocks.lock().await;
        inactive.free_table.keys().cloned().collect()
    }

    /// Get the keys of active blocks
    pub async fn get_active_blocks(&self) -> Vec<UniqueBlock> {
        let active = self.active_blocks.lock().await;
        active.keys().cloned().collect()
    }
}

impl Drop for KvManager {
    fn drop(&mut self) {
        // Abort the event loop task when MockWorkers is dropped
        self.event_loop_handle.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;

    #[test]
    fn test_panic_on_max_capacity() {
        // Create a runtime for async testing
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Create a MockWorker with 10 blocks capacity
            let worker = KvManager::new(10, 16);
            let event_tx = worker.get_event_sender();

            // Helper function to use multiple blocks
            async fn use_blocks(event_tx: &mpsc::Sender<MoveBlock>, ids: Vec<u64>) {
                let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
                event_tx.send(MoveBlock::Use(blocks, None)).await.unwrap();
            }

            // First use 10 blocks (0 to 9) in a batch
            use_blocks(&event_tx, (0..10).collect()).await;

            // Process events to ensure they are processed
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

            // Verify we are at capacity
            assert_eq!(worker.current_capacity().await, 10);

            // The 11th block should cause a panic
            let result = panic::catch_unwind(|| {
                rt.block_on(async {
                    use_blocks(&event_tx, vec![10]).await;
                    // Give time for the event to be processed
                    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                });
            });

            // Verify that a panic occurred
            assert!(
                result.is_err(),
                "Expected a panic when exceeding max capacity"
            );
        });
    }

    #[test]
    fn test_block_lifecycle_stringent() {
        // Create a runtime for async testing
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            // Create a MockWorker with 7 blocks capacity
            let worker = KvManager::new(10, 16);
            let event_tx = worker.get_event_sender();

            // Helper function to use multiple blocks
            async fn use_blocks(event_tx: &mpsc::Sender<MoveBlock>, ids: Vec<u64>) {
                let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
                event_tx.send(MoveBlock::Use(blocks, None)).await.unwrap();
            }

            // Helper function to destroy multiple blocks
            async fn destroy_blocks(event_tx: &mpsc::Sender<MoveBlock>, ids: Vec<u64>) {
                let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
                event_tx.send(MoveBlock::Destroy(blocks)).await.unwrap();
            }

            // Helper function to deref multiple blocks
            async fn deref_blocks(event_tx: &mpsc::Sender<MoveBlock>, ids: Vec<u64>) {
                let blocks = ids.into_iter().map(UniqueBlock::FullBlock).collect();
                event_tx.send(MoveBlock::Deref(blocks)).await.unwrap();
            }

            // Helper function to check if active blocks contain expected blocks with expected ref counts
            async fn assert_active_blocks(worker: &KvManager, expected_blocks: &[(u64, usize)]) {
                let active_blocks_lock = worker.active_blocks.lock().await;

                assert_eq!(
                    active_blocks_lock.len(),
                    expected_blocks.len(),
                    "Active blocks count doesn't match expected"
                );

                for &(id, ref_count) in expected_blocks {
                    let block = UniqueBlock::FullBlock(id);
                    assert!(
                        active_blocks_lock.contains_key(&block),
                        "Block {} not found in active blocks",
                        id
                    );
                    assert_eq!(
                        active_blocks_lock.get(&block),
                        Some(&ref_count),
                        "Block {} has wrong reference count",
                        id
                    );
                }
            }

            // Helper function to check if inactive blocks contain expected blocks
            async fn assert_inactive_blocks(
                worker: &KvManager,
                expected_size: usize,
                expected_blocks: &[u64],
            ) {
                let inactive_blocks = worker.get_inactive_blocks().await;
                let inactive_blocks_count = worker.inactive_blocks.lock().await.num_objects();

                assert_eq!(
                    inactive_blocks_count, expected_size,
                    "Inactive blocks count doesn't match expected"
                );

                for &id in expected_blocks {
                    let block = UniqueBlock::FullBlock(id);
                    assert!(
                        inactive_blocks.contains(&block),
                        "Block {} not found in inactive blocks",
                        id
                    );
                }
            }

            // Helper function to process events
            async fn process_events() {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }

            // First use blocks 0, 1, 2, 3, 4 in a batch
            use_blocks(&event_tx, (0..5).collect()).await;
            process_events().await;

            // Then use blocks 0, 1, 5, 6 in a batch
            use_blocks(&event_tx, vec![0, 1, 5, 6]).await;
            process_events().await;

            // Check that the blocks 0 and 1 are in active blocks, both with reference counts of 2
            assert_active_blocks(
                &worker,
                &[(0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1)],
            )
            .await;

            // Now destroy block 4
            destroy_blocks(&event_tx, vec![4]).await;
            process_events().await;

            // And deref blocks 3, 2, 1, 0 in this order as a batch
            deref_blocks(&event_tx, vec![0, 1, 2, 3]).await;
            process_events().await;

            // Check that the inactive_blocks is size 2 (via num_objects) and contains 3 and 2
            assert_inactive_blocks(&worker, 2, &[3, 2]).await;
            assert_active_blocks(&worker, &[(0, 1), (1, 1), (5, 1), (6, 1)]).await;

            // Now destroy block 6
            destroy_blocks(&event_tx, vec![6]).await;
            process_events().await;

            // And deref blocks 5, 1, 0 as a batch
            deref_blocks(&event_tx, vec![0, 1, 5]).await;
            process_events().await;

            // Check that the inactive_blocks is size 5, and contains 0, 1, 2, 3, 5
            assert_inactive_blocks(&worker, 5, &[0, 1, 2, 3, 5]).await;
            assert_active_blocks(&worker, &[]).await;

            // Now use 0, 1, 2, 7, 8, 9 as a batch
            use_blocks(&event_tx, vec![0, 1, 2, 7, 8, 9]).await;
            process_events().await;

            // Check that the inactive_blocks is size 2, and contains 3 and 5
            assert_inactive_blocks(&worker, 2, &[3, 5]).await;
            assert_active_blocks(&worker, &[(0, 1), (1, 1), (2, 1), (7, 1), (8, 1), (9, 1)]).await;

            // Now use blocks 10, 11, 12 as a batch
            use_blocks(&event_tx, vec![10, 11, 12]).await;
            process_events().await;

            // Check that the inactive_blocks is size 1 and contains only 5
            assert_inactive_blocks(&worker, 1, &[5]).await;
        });
    }
}
