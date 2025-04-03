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

#![allow(dead_code)]

use crate::kv_router::indexer::RouterEvent;
use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing as log;

/// Record entry that will be serialized to JSONL
#[derive(Serialize, Deserialize)]
struct RecordEntry {
    /// Time in milliseconds since UNIX epoch
    timestamp: u64,
    /// The router event
    router_event: RouterEvent,
}

/// A recorder for KvRouter events
pub struct KvRecorder {
    /// The recorded events - now storing SystemTime for absolute timestamps
    events: Arc<Mutex<Vec<(SystemTime, RouterEvent)>>>,
    /// A sender for RouterEvents that can be cloned and shared with producers
    event_tx: mpsc::Sender<RouterEvent>,
    /// A cancellation token for managing shutdown
    cancel: CancellationToken,
}

impl KvRecorder {
    /// Create a new KvRecorder
    ///
    /// ### Arguments
    ///
    /// * `token` - A cancellation token for managing shutdown
    ///
    /// ### Returns
    ///
    /// A new KvRecorder instance
    pub fn new(token: CancellationToken) -> Self {
        let (event_tx, mut event_rx) = mpsc::channel::<RouterEvent>(2048);
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let cancel_clone = token.clone();
        
        // Spawn a task to receive and record events
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    biased;
                    
                    _ = cancel_clone.cancelled() => {
                        log::debug!("KvRecorder task shutting down");
                        return;
                    }
                    
                    Some(event) = event_rx.recv() => {
                        // Record the event with current timestamp
                        let mut events = events_clone.lock().unwrap();
                        events.push((SystemTime::now(), event));
                    }
                }
            }
        });
        
        Self {
            events,
            event_tx,
            cancel: token,
        }
    }
    
    /// Get a sender that can be used to send events to the recorder
    pub fn event_sender(&self) -> mpsc::Sender<RouterEvent> {
        self.event_tx.clone()
    }
    
    /// Dump recorded events to a JSONL file
    ///
    /// ### Arguments
    ///
    /// * `filename` - Path to the JSONL file to write
    /// * `num_events` - Optional limit on the number of events to write
    ///
    /// ### Returns
    ///
    /// A Result indicating success or failure
    pub fn to_jsonl<P: AsRef<Path>>(&self, filename: P, num_events: Option<usize>) -> io::Result<()> {
        let events = self.events.lock().unwrap();
        
        if events.is_empty() {
            log::warn!("No events to dump");
            return Ok(());
        }
        
        // Store the display name before using filename with File::create
        let display_name = filename.as_ref().display().to_string();
        
        let file = File::create(filename)?;
        let mut writer = BufWriter::new(file);
        
        // Determine how many events to write
        let event_count = match num_events {
            Some(limit) => events.len().min(limit),
            None => events.len(),
        };
        
        for (timestamp, event) in events.iter().take(event_count) {
            // Convert SystemTime to milliseconds since UNIX epoch
            let millis = timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| Duration::from_secs(0))
                .as_millis() as u64;
            
            let entry = RecordEntry {
                timestamp: millis,
                router_event: event.clone(),
            };
            
            // Serialize and write the entry
            serde_json::to_writer(&mut writer, &entry)?;
            writeln!(writer)?;
        }
        
        writer.flush()?;
        log::info!("Dumped {} events to {}", event_count, display_name);
        
        Ok(())
    }
    
    /// Read events from a JSONL file
    ///
    /// ### Arguments
    ///
    /// * `filename` - Path to the JSONL file to read
    /// * `num_lines` - Optional limit on the number of lines to read
    ///
    /// ### Returns
    ///
    /// A Result with the count of records loaded or an error
    pub fn from_jsonl<P: AsRef<Path>>(&self, filename: P, num_events: Option<usize>) -> io::Result<usize> {
        // Store the display name before using filename
        let display_name = filename.as_ref().display().to_string();
        
        // Check if file exists
        if !filename.as_ref().exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound, 
                format!("File not found: {}", display_name)
            ));
        }
        
        // Open the file for reading
        let file = File::open(filename)?;
        let reader = io::BufReader::new(file);
        let mut lines = io::BufRead::lines(reader);
        
        let mut events = self.events.lock().unwrap();
        let mut count = 0;
        
        // Process each line
        while let Some(line) = lines.next() {
            // Check if we've reached the requested number of lines
            if let Some(limit) = num_events {
                if count >= limit {
                    break;
                }
            }
            
            // Parse line
            let line = line?;
            let record: RecordEntry = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(e) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("Failed to parse JSON on line {}: {}", count + 1, e)
                    ));
                }
            };
            
            // Convert timestamp from milliseconds back to SystemTime
            let timestamp = UNIX_EPOCH + Duration::from_millis(record.timestamp);
            
            // Store the event with the original timestamp
            events.push((timestamp, record.router_event));
            count += 1;
        }
        
        log::info!("Loaded {} events from {}", count, display_name);
        Ok(count)
    }
    
    /// Get the count of recorded events
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }
    
    /// Clear all recorded events
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }
    
    /// Shutdown the recorder
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_router::protocols::*;
    use crate::kv_router::indexer::WorkerId;
    use tempfile::tempdir;

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
        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id,
                data: add_blocks(hashes, parent),
            },
        }
    }

    fn create_remove_event(worker_id: WorkerId, event_id: u64, hashes: Vec<u64>) -> RouterEvent {
        RouterEvent {
            worker_id,
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes
                        .iter()
                        .map(|i| ExternalSequenceBlockHash(*i * 100))
                        .collect(),
                }),
            },
        }
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
        assert_eq!(recorder.event_count(), 2);
        
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
        recorder.to_jsonl(&file_path1, None).unwrap();
        
        // Read the content of the first file
        let content1 = std::fs::read_to_string(&file_path1).unwrap();
        println!("JSONL content of file1:\n{}", content1);
        
        // Create a new recorder and load events from file1
        let new_recorder = KvRecorder::new(token.clone());
        let count = new_recorder.from_jsonl(&file_path1, None).unwrap();
        assert_eq!(count, 2, "Expected to load 2 events");
        
        // Get event sender for the new recorder
        let new_event_tx = new_recorder.event_sender();
        
        // Create and send two more events to the new recorder
        let store_event2 = create_store_event(3, 44, vec![1, 4, 5], None);
        let remove_event2 = create_remove_event(4, 45, vec![4, 5]);
        
        new_event_tx.send(store_event2.clone()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        new_event_tx.send(remove_event2.clone()).await.unwrap();
        
        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        // Verify that the new recorder now has 4 events
        assert_eq!(new_recorder.event_count(), 4, "Expected 4 events in the new recorder");
        
        // Create a second file for output
        let file_path2 = dir.path().join("events2.jsonl");
        
        // Dump all 4 events to file2
        new_recorder.to_jsonl(&file_path2, None).unwrap();
        
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
        assert_eq!(lines1[0], lines2[0], "First line should be identical in both files");
        assert_eq!(lines1[1], lines2[1], "Second line should be identical in both files");
        
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
        let output_path = manifest_dir
            .join("src/kv_router/test_recorder_events.jsonl");

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
        recorder.to_jsonl(&output_path, None).unwrap();
        
        // Verify the file exists
        assert!(output_path.exists(), "JSONL file should exist at {:?}", output_path);
        println!("Successfully wrote events to {:?}", output_path);
        
        // Clean up
        recorder.shutdown();
        
        // Note: We intentionally don't delete the file so it can be manually inspected
    }
}