// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::kv_router::indexer::RouterEvent;
use crate::recorder::Recorder;

// Type alias for backward compatibility
pub type KvRecorder = Recorder<RouterEvent>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::indexer::KvIndexer;
    use crate::kv_router::indexer::KvIndexerMetrics;
    use crate::kv_router::protocols::*;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio::fs;
    use tokio_util::sync::CancellationToken;

    fn make_blocks(hashes: Vec<u64>) -> Vec<KvCacheStoredBlockData> {
        hashes
            .iter()
            .map(|i| KvCacheStoredBlockData {
                tokens_hash: LocalBlockHash(*i),
                block_hash: ExternalSequenceBlockHash(*i * 100),
                mm_extra_info: None,
            })
            .collect()
    }

    fn add_blocks(
        hashes: Vec<u64>,
        parent_hash: Option<ExternalSequenceBlockHash>,
    ) -> KvCacheEventData {
        KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            blocks: make_blocks(hashes),
        })
    }

    fn create_store_event(
        worker_id: WorkerId,
        event_id: u64,
        hashes: Vec<u64>,
        parent: Option<ExternalSequenceBlockHash>,
    ) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: add_blocks(hashes, parent),
                dp_rank: 0,
            },
        )
    }

    fn create_remove_event(worker_id: WorkerId, event_id: u64, hashes: Vec<u64>) -> RouterEvent {
        RouterEvent::new(
            worker_id,
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes
                        .iter()
                        .map(|i| ExternalSequenceBlockHash(*i * 100))
                        .collect(),
                }),
                dp_rank: 0,
            },
        )
    }

    #[tokio::test]
    async fn test_recorder_streams_events_to_file() {
        // Create a temporary directory for output files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("kv_events.jsonl");

        // Part 1: Record events to a file
        let token = CancellationToken::new();
        let recorder = KvRecorder::new(token.clone(), &file_path, None, None, None)
            .await
            .unwrap();
        let event_tx = recorder.event_sender();

        // Create first event from worker 1 using helper function
        let event1 = create_store_event(1, 42, vec![1, 2, 3], None);

        // Create second event from worker 2 using helper function
        let event2 = create_remove_event(1, 43, vec![2, 3]);

        // Send both events one after another
        event_tx.send(event1).await.unwrap();
        event_tx.send(event2).await.unwrap();

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Check that both events were recorded
        assert_eq!(recorder.event_count().await, 2);

        // Force shutdown to flush file
        recorder.shutdown();
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Read the file and verify content
        let content = fs::read_to_string(&file_path).await.unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Print the content of the JSONL file
        println!("JSONL file content:");
        for (i, line) in lines.iter().enumerate() {
            println!("Line {}: {}", i + 1, line);
        }

        assert_eq!(lines.len(), 2, "Expected 2 lines in the file");

        // Part 2: Now create a KvIndexer and load the events from the file
        let indexer_token = CancellationToken::new();
        let kv_block_size = 32; // Default block size for testing
        let kv_indexer_metrics = KvIndexerMetrics::new_unregistered();
        let indexer = KvIndexer::new(
            indexer_token.clone(),
            kv_block_size,
            kv_indexer_metrics.into(),
        );
        let indexer_event_tx = indexer.event_sender();

        // Use the send_events method to load events from file to indexer
        let count = KvRecorder::send_events(&file_path, &indexer_event_tx, false, None, None)
            .await
            .unwrap();
        assert_eq!(count, 2, "Expected to send 2 events from file to indexer");
    }
}
