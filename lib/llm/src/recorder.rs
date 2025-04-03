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

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex};
use tokio_util::sync::CancellationToken;
use tracing as log;

/// Record entry that will be serialized to JSONL
#[derive(Serialize, Deserialize)]
struct RecordEntry<T>
where
    T: Clone,
{
    /// Time in milliseconds since recording started
    timestamp: u64,
    /// The event
    event: T,
}

/// A generic recorder for events
pub struct Recorder<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static,
{
    /// The time offset in milliseconds to apply to new events
    time_offset: u64,
    /// The recorded events - now storing elapsed milliseconds since start
    events: Arc<Mutex<Vec<(u64, T)>>>,
    /// A sender for events that can be cloned and shared with producers
    event_tx: mpsc::Sender<T>,
    /// A cancellation token for managing shutdown
    cancel: CancellationToken,
}

impl<T> Recorder<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static,
{
    /// Create a new Recorder without a time offset
    ///
    /// ### Arguments
    ///
    /// * `token` - A cancellation token for managing shutdown
    ///
    /// ### Returns
    ///
    /// A new Recorder instance
    pub fn new(token: CancellationToken) -> Self {
        Self::new_with_offset(token, 0)
    }

    /// Create a new Recorder with a specific time offset
    ///
    /// ### Arguments
    ///
    /// * `token` - A cancellation token for managing shutdown
    /// * `time_offset` - The time offset in milliseconds to apply to new events
    ///
    /// ### Returns
    ///
    /// A new Recorder instance with the specified time offset
    pub fn new_with_offset(token: CancellationToken, time_offset: u64) -> Self {
        let (event_tx, mut event_rx) = mpsc::channel::<T>(2048);
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let cancel_clone = token.clone();
        let start_time = Instant::now();

        // Spawn a task to receive and record events
        tokio::spawn(async move {
            let start_time = start_time;
            let time_offset = time_offset;
            loop {
                tokio::select! {
                    biased;

                    _ = cancel_clone.cancelled() => {
                        log::debug!("Recorder task shutting down");
                        return;
                    }

                    Some(event) = event_rx.recv() => {
                        // Record the event with elapsed time since start in milliseconds, plus the offset
                        let elapsed_ms = start_time.elapsed().as_millis() as u64;
                        let timestamp = elapsed_ms + time_offset;
                        let mut events = events_clone.lock().await;
                        events.push((timestamp, event));
                    }
                }
            }
        });

        Self {
            time_offset,
            events,
            event_tx,
            cancel: token,
        }
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
    pub async fn to_jsonl<P: AsRef<Path>>(
        &self,
        filename: P,
        num_events: Option<usize>,
    ) -> io::Result<()> {
        let events = self.events.lock().await;

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
            let entry = RecordEntry {
                timestamp: *timestamp,
                event: event.clone(),
            };

            // Serialize and write the entry
            serde_json::to_writer(&mut writer, &entry)?;
            writeln!(writer)?;
        }

        writer.flush()?;
        log::info!("Dumped {} events to {}", event_count, display_name);

        Ok(())
    }

    /// Read events from a JSONL file and create a new recorder with appropriate time offset
    ///
    /// ### Arguments
    ///
    /// * `filename` - Path to the JSONL file to read
    /// * `token` - A cancellation token for the new recorder
    /// * `num_events` - Optional limit on the number of events to read
    ///
    /// ### Returns
    ///
    /// A Result with a new Recorder that contains the loaded events and appropriate time offset
    pub async fn from_jsonl<P: AsRef<Path>>(
        filename: P,
        token: CancellationToken,
        num_events: Option<usize>,
    ) -> io::Result<Self> {
        // Store the display name before using filename
        let display_name = filename.as_ref().display().to_string();

        // Check if file exists
        if !filename.as_ref().exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", display_name),
            ));
        }

        // Open the file for reading
        let file = File::open(&filename)?;
        let reader = BufReader::new(file);
        let lines = io::BufRead::lines(reader);

        let mut events = Vec::new();
        let mut count = 0;
        let mut last_timestamp = 0;

        // Process each line
        for line in lines {
            // Check if we've reached the requested number of lines
            if let Some(limit) = num_events {
                if count >= limit {
                    break;
                }
            }

            // Parse line
            let line = line?;
            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            // Try to parse the JSON, skip on error but log a warning
            let record: RecordEntry<T> = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(e) => {
                    log::warn!(
                        "Failed to parse JSON on line {}: {}. Skipping.",
                        count + 1,
                        e
                    );
                    continue;
                }
            };

            // Track the timestamp
            if record.timestamp > last_timestamp {
                last_timestamp = record.timestamp;
            }

            // Store the event with its relative timestamp
            events.push((record.timestamp, record.event));
            count += 1;
        }

        // Create a new recorder with the appropriate time offset
        let recorder = Self::new_with_offset(token, last_timestamp);

        // Add all events to the new recorder's event list
        if !events.is_empty() {
            let mut recorder_events = recorder.events.lock().await;
            recorder_events.extend(events);
        }

        log::info!("Loaded {} events from {}", count, display_name);
        log::info!(
            "New recorder created with time offset of {} ms",
            last_timestamp
        );
        Ok(recorder)
    }

    /// Get a sender that can be used to send events to the recorder
    pub fn event_sender(&self) -> mpsc::Sender<T> {
        self.event_tx.clone()
    }

    /// Get the count of recorded events
    pub async fn event_count(&self) -> usize {
        self.events.lock().await.len()
    }

    /// Clear all recorded events
    pub async fn clear(&self) {
        self.events.lock().await.clear();
    }

    /// Shutdown the recorder
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    /// Get the current time offset
    pub fn time_offset(&self) -> u64 {
        self.time_offset
    }

    /// Get all recorded events as a vector of (timestamp, event) tuples
    pub async fn get_events(&self) -> Vec<(u64, T)> {
        let events = self.events.lock().await;
        events.clone()
    }

    /// Send recorded events to the provided sender
    ///
    /// ### Arguments
    ///
    /// * `event_tx` - A sender for events
    ///
    /// ### Returns
    ///
    /// A Result indicating success or failure
    pub async fn send_events(
        &self,
        event_tx: &mpsc::Sender<T>,
    ) -> Result<usize, mpsc::error::SendError<T>> {
        let events = self.events.lock().await;

        if events.is_empty() {
            log::warn!("No events to send");
            return Ok(0);
        }

        // Assert that events are weakly sorted by timestamp (non-decreasing order)
        for i in 1..events.len() {
            assert!(
                events[i-1].0 <= events[i].0,
                "Events are not in timestamp order: event at index {} has timestamp {} which is after event at index {} with timestamp {}",
                i-1, events[i-1].0, i, events[i].0
            );
        }

        let mut count = 0;

        // Send each event in the order they were recorded
        for (_, event) in events.iter() {
            event_tx.send(event.clone()).await?;
            count += 1;
        }

        log::info!("Sent {} events", count);
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;

    // Simple event types for testing
    type IntRecorder = Recorder<i32>;
    type StringRecorder = Recorder<String>;

    // More complex event type
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestEvent {
        id: u64,
        name: String,
        value: f64,
    }

    type EventRecorder = Recorder<TestEvent>;

    #[tokio::test]
    async fn test_recorder_records_events() {
        let token = CancellationToken::new();
        let recorder = IntRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Create and send two integer events
        event_tx.send(42).await.unwrap();
        event_tx.send(43).await.unwrap();

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
        let recorder = StringRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Create and send two string events
        let event1 = "Hello".to_string();
        let event2 = "World".to_string();

        event_tx.send(event1.clone()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        event_tx.send(event2.clone()).await.unwrap();

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Create a temporary directory for output files
        let dir = tempdir().unwrap();
        let file_path1 = dir.path().join("events1.jsonl");

        // Dump events to file
        recorder.to_jsonl(&file_path1, None).await.unwrap();

        // Read the content of the file
        let content1 = std::fs::read_to_string(&file_path1).unwrap();
        println!("JSONL content of file1:\n{}", content1);

        // Create a new recorder with appropriate time offset based on the loaded file
        let new_recorder = StringRecorder::from_jsonl(&file_path1, token.clone(), None)
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

        // Sleep a bit before sending new events
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Send two more events to the new recorder
        let event3 = "Testing".to_string();
        let event4 = "Recorder".to_string();

        new_event_tx.send(event3.clone()).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        new_event_tx.send(event4.clone()).await.unwrap();

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
        let output_path = manifest_dir.join("src/test_recorder_events.jsonl");

        println!("Will write to: {}", output_path.display());

        // Create a recorder and send events with complex event type
        let token = CancellationToken::new();
        let recorder = EventRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Create and send two events
        let event1 = TestEvent {
            id: 1,
            name: "Event 1".to_string(),
            value: 3.14,
        };
        let event2 = TestEvent {
            id: 2,
            name: "Event 2".to_string(),
            value: 2.71,
        };

        event_tx.send(event1).await.unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        event_tx.send(event2).await.unwrap();

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
        let recorder = IntRecorder::new(token.clone());
        let event_tx = recorder.event_sender();

        // Define number of events to generate
        const NUM_EVENTS: usize = 100_000;
        println!("Generating {} events...", NUM_EVENTS);

        // Create and send 100k integer events
        for i in 0..NUM_EVENTS {
            event_tx.send(i as i32).await.unwrap();

            // Print progress every 10,000 events
            if i > 0 && i % 10_000 == 0 {
                println!("Sent {} events...", i);
            }
        }

        // Allow time for the recorder to process all events
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
