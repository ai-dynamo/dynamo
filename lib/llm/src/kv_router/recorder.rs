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

use crate::kv_router::indexer::RouterEvent;
use crate::recorder::Recorder;

// Type alias for backward compatibility
pub type KvRecorder = Recorder<RouterEvent>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::indexer::WorkerId;
    use crate::kv_router::protocols::*;
    use std::time::Duration;
    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;

    fn make_blocks(hashes: Vec<u64>) -> Vec<KvCacheStoredBlockData> {
        hashes
            .iter()
            .map(|i| KvCacheStoredBlockData {
                tokens_hash: LocalBlockHash(*i),
                block_hash: ExternalSequenceBlockHash(*i * 100),
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
            },
        )
    }

    #[tokio::test]
    async fn test_recorder_records_events() {
        let token = CancellationToken::new();
        let recorder = KvRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Create first event from worker 1 using helper function
        let event1 = create_store_event(1, 42, vec![1, 2, 3], None);

        // Create second event from worker 2 using helper function
        let event2 = create_store_event(2, 43, vec![1, 4, 5], None);

        // Send both events one after another
        event_tx.send(event1).await.unwrap();
        event_tx.send(event2).await.unwrap();

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Check that both events were recorded
        assert_eq!(recorder.event_count().await, 2);

        // Clean up
        recorder.shutdown();
    }

    #[tokio::test]
    async fn test_jsonl_roundtrip_preserves_events() {
        let token = CancellationToken::new();
        let recorder = KvRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Create and send the first two events
        let store_event1 = create_store_event(1, 42, vec![1, 2, 3], None);
        let remove_event1 = create_remove_event(2, 43, vec![3]);

        event_tx.send(store_event1.clone()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        event_tx.send(remove_event1.clone()).await.unwrap();

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Create a temporary directory for output files
        let dir = tempdir().unwrap();
        let file_path1 = dir.path().join("events1.jsonl");

        // Dump first two events to file1
        recorder.to_jsonl(&file_path1, None).await.unwrap();

        // Read the content of the first file
        let content1 = std::fs::read_to_string(&file_path1).unwrap();
        println!("JSONL content of file1:\n{}", content1);

        // Create a new recorder with appropriate time offset based on the loaded file
        let new_recorder = KvRecorder::from_jsonl(&file_path1, token.clone(), None)
            .await
            .unwrap();

        // Verify the new recorder has 2 events
        assert_eq!(
            new_recorder.event_count().await,
            2,
            "Expected to load 2 events"
        );

        // Check that time offset is correctly set
        let first_events = new_recorder.get_events().await;
        let max_timestamp = first_events.iter().map(|(ts, _)| ts).max().unwrap();
        assert_eq!(
            new_recorder.time_offset(),
            *max_timestamp,
            "Time offset should match the last timestamp from the file"
        );

        // Get event sender for the new recorder
        let new_event_tx = new_recorder.event_sender();

        // Sleep a bit longer before sending new events to ensure timestamps are clearly different
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Create and send two more events to the new recorder
        let store_event2 = create_store_event(3, 44, vec![1, 4, 5], None);
        let remove_event2 = create_remove_event(4, 45, vec![4, 5]);

        new_event_tx.send(store_event2.clone()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        new_event_tx.send(remove_event2.clone()).await.unwrap();

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Verify that the new recorder now has 4 events
        assert_eq!(
            new_recorder.event_count().await,
            4,
            "Expected 4 events in the new recorder"
        );

        // Create a second file for output
        let file_path2 = dir.path().join("events2.jsonl");

        // Dump all 4 events to file2
        new_recorder.to_jsonl(&file_path2, None).await.unwrap();

        // Read the content of the second file
        let content2 = std::fs::read_to_string(&file_path2).unwrap();
        println!("JSONL content of file2:\n{}", content2);

        // Split both contents into lines
        let lines1: Vec<&str> = content1.lines().collect();
        let lines2: Vec<&str> = content2.lines().collect();

        // Verify we have the expected number of lines
        assert_eq!(lines1.len(), 2, "Expected 2 lines in file1");
        assert_eq!(lines2.len(), 4, "Expected 4 lines in file2");

        // Verify the first two lines of both files are exactly the same
        assert_eq!(
            lines1[0], lines2[0],
            "First line should be identical in both files"
        );
        assert_eq!(
            lines1[1], lines2[1],
            "Second line should be identical in both files"
        );

        // Verify that all timestamps in new_recorder are sorted
        let all_events = new_recorder.get_events().await;
        for i in 1..all_events.len() {
            assert!(
                all_events[i-1].0 <= all_events[i].0,
                "Events should be in timestamp order: event at index {} has timestamp {} which should be before or equal to event at index {} with timestamp {}",
                i-1, all_events[i-1].0, i, all_events[i].0
            );
        }

        // Clean up
        recorder.shutdown();
        new_recorder.shutdown();
    }

    #[ignore]
    #[tokio::test]
    async fn test_write_to_actual_file() {
        // Use the manifest directory (project root) as a reference point
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        // Create the output path in the same directory as the source file
        let output_path = manifest_dir.join("src/kv_router/test_recorder_events.jsonl");

        println!("Will write to: {}", output_path.display());

        // Create a recorder and send events
        let token = CancellationToken::new();
        let recorder = KvRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Create and send two events
        let store_event = create_store_event(1, 42, vec![1, 2, 3], None);
        let remove_event = create_remove_event(2, 43, vec![3]);

        event_tx.send(store_event).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        event_tx.send(remove_event).await.unwrap();

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Write events to the actual file
        recorder.to_jsonl(&output_path, None).await.unwrap();

        // Verify the file exists
        assert!(
            output_path.exists(),
            "JSONL file should exist at {:?}",
            output_path
        );
        println!("Successfully wrote events to {:?}", output_path);

        // Clean up
        recorder.shutdown();

        // Note: We intentionally don't delete the file so it can be manually inspected
    }

    #[ignore]
    #[tokio::test]
    async fn load_test_100k_events() {
        // Create a cancellation token for the recorder
        let token = CancellationToken::new();
        let recorder = KvRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Define number of events to generate
        const NUM_EVENTS: usize = 100_000;
        println!("Generating {} events...", NUM_EVENTS);

        // Use 10 different worker IDs to simulate distributed system
        const NUM_WORKERS: u64 = 10;

        // Create and send 100k store events with varying data
        for i in 0..NUM_EVENTS {
            // Use modulo to distribute events across different workers
            let worker_id = (i as u64) % NUM_WORKERS;

            // Create event with event ID matching the loop counter
            // Include a small number of block hashes to keep memory usage reasonable
            let event = create_store_event(
                worker_id as i64,
                i as u64,
                vec![
                    i as u64 % 1000,
                    (i as u64 + 1) % 1000,
                    (i as u64 + 2) % 1000,
                ],
                None,
            );

            // Send the event
            event_tx.send(event).await.unwrap();

            // Print progress every 10,000 events
            if i > 0 && i % 10_000 == 0 {
                println!("Sent {} events...", i);
            }
        }

        // Allow time for the recorder to process all events
        // Might need to adjust this based on system performance
        println!("Waiting for events to be processed...");
        tokio::time::sleep(Duration::from_millis(1000)).await;

        // Verify that all events were recorded
        let count = recorder.event_count().await;
        println!("Recorded event count: {}", count);
        assert_eq!(count, NUM_EVENTS, "Expected exactly {} events", NUM_EVENTS);

        // Optionally, create a temp file and verify serialization works
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("load_test_events.jsonl");

        println!("Writing events to temporary file: {}", file_path.display());
        recorder.to_jsonl(&file_path, None).await.unwrap();

        // Verify the file exists
        assert!(
            file_path.exists(),
            "JSONL file should exist at {:?}",
            file_path
        );

        // Clean up
        recorder.shutdown();
        println!("Load test completed successfully");
    }
}
