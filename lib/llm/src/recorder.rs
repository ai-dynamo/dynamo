// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::fs::{self, File, OpenOptions};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::{Mutex, mpsc};
use tokio_util::sync::CancellationToken;

/// Record entry that will be serialized to JSONL
#[derive(Serialize, Deserialize)]
struct RecordEntry<T>
where
    T: Clone,
{
    timestamp: u64,
    event: T,
}

/// A generic recorder for events that streams directly to a JSONL file
#[derive(Debug)]
pub struct Recorder<T> {
    /// A sender for events that can be cloned and shared with producers
    event_tx: mpsc::Sender<T>,
    /// A cancellation token for managing shutdown
    cancel: CancellationToken,
    /// Counter for the number of events written
    event_count: Arc<Mutex<usize>>,
    /// Time when the first event was received
    first_event_time: Arc<Mutex<Option<Instant>>>,
}

impl<T> Recorder<T>
where
    T: Serialize + for<'de> Deserialize<'de> + Clone + Send + Sync + 'static,
{
    /// Create a new Recorder that streams events directly to a JSONL file
    ///
    /// ### Arguments
    ///
    /// * `token` - A cancellation token for managing shutdown
    /// * `output_path` - Path to the JSONL file to write events to
    /// * `max_lines_per_file` - Maximum number of lines per file before rotating to a new file.
    ///   If None, no rotation will occur.
    /// * `max_count` - Maximum number of events to record before shutting down.
    ///   If None, no limit will be applied.
    /// * `max_time` - Maximum duration in seconds to record before shutting down.
    ///   If None, no time limit will be applied.
    ///
    /// ### Returns
    ///
    /// A Result with a new Recorder that streams events to the specified file
    pub async fn new<P: AsRef<Path>>(
        token: CancellationToken,
        output_path: P,
        max_lines_per_file: Option<usize>,
        max_count: Option<usize>,
        max_time: Option<f64>,
    ) -> io::Result<Self> {
        let (event_tx, mut event_rx) = mpsc::channel::<T>(2048);
        let event_count = Arc::new(Mutex::new(0));
        let event_count_clone = event_count.clone();
        let cancel_clone = token.clone();
        let start_time = Instant::now();
        let first_event_time = Arc::new(Mutex::new(None));
        let first_event_time_clone = first_event_time.clone();

        // Ensure the directory exists
        if let Some(parent) = output_path.as_ref().parent()
            && !parent.exists()
        {
            fs::create_dir_all(parent).await?;
        }

        // Create the file for writing
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&output_path)
            .await?;

        let file_path = output_path.as_ref().to_path_buf();

        // Spawn a task to receive events and write them to the file
        tokio::spawn(async move {
            let start_time = start_time;
            let mut writer = BufWriter::with_capacity(32768, file);
            let mut line_count = 0;
            let mut file_index = 0;
            let base_path = file_path.clone();

            // Set up max time deadline if specified
            let max_time_deadline = max_time.map(|secs| {
                let duration = Duration::from_secs_f64(secs);
                start_time + duration
            });

            loop {
                // Check time limit if set
                if let Some(deadline) = max_time_deadline
                    && Instant::now() >= deadline
                {
                    tracing::info!("Recorder reached max time limit, shutting down");
                    // Flush and cancel
                    if let Err(e) = writer.flush().await {
                        tracing::error!("Failed to flush on time limit shutdown: {}", e);
                    }
                    cancel_clone.cancel();
                    return;
                }

                tokio::select! {
                    biased;

                    _ = cancel_clone.cancelled() => {
                        // Flush any pending writes before shutting down
                        if let Err(e) = writer.flush().await {
                            tracing::error!("Failed to flush on shutdown: {}", e);
                        }

                        tracing::debug!("Recorder task shutting down");
                        return;
                    }

                    Some(event) = event_rx.recv() => {
                        // Update first_event_time if this is the first event
                        {
                            let mut first_time = first_event_time_clone.lock().await;
                            if first_time.is_none() {
                                *first_time = Some(Instant::now());
                            }
                        }

                        // Calculate elapsed time in milliseconds
                        let elapsed_ms = start_time.elapsed().as_millis() as u64;

                        // Create the record entry
                        let entry = RecordEntry {
                            timestamp: elapsed_ms,
                            event,
                        };

                        // Serialize to JSON string
                        let json = match serde_json::to_string(&entry) {
                            Ok(json) => json,
                            Err(e) => {
                                tracing::error!("Failed to serialize event: {}", e);
                                continue;
                            }
                        };

                        // Write JSON line
                        if let Err(e) = writer.write_all(json.as_bytes()).await {
                            tracing::error!("Failed to write event: {}", e);
                            continue;
                        }

                        // Add a newline
                        if let Err(e) = writer.write_all(b"\n").await {
                            tracing::error!("Failed to write newline: {}", e);
                            continue;
                        }

                        // Increment line count
                        line_count += 1;

                        // Check if we need to rotate to a new file
                        if let Some(max_lines) = max_lines_per_file
                            && line_count >= max_lines {
                                // Flush the current file
                                if let Err(e) = writer.flush().await {
                                    tracing::error!("Failed to flush file before rotation: {}", e);
                                }

                                // Create new filename with suffix
                                file_index += 1;
                                let new_path = create_rotated_path(&base_path, file_index);

                                // Open new file
                                match OpenOptions::new()
                                    .create(true)
                                    .write(true)
                                    .truncate(true)
                                    .open(&new_path)
                                    .await
                                {
                                    Ok(new_file) => {
                                        writer = BufWriter::with_capacity(32768, new_file);
                                        line_count = 0;
                                        tracing::info!("Rotated to new file: {}", new_path.display());
                                    },
                                    Err(e) => {
                                        tracing::error!("Failed to open rotated file {}: {}", new_path.display(), e);
                                        // Continue with the existing file if rotation fails
                                    }
                                }
                            }

                        // Update event count
                        let mut count = event_count_clone.lock().await;
                        *count += 1;

                        // Check if we've reached the maximum count
                        if let Some(max) = max_count
                            && *count >= max {
                                tracing::info!("Recorder reached max event count ({}), shutting down", max);
                                // Flush buffer before shutting down
                                if let Err(e) = writer.flush().await {
                                    tracing::error!("Failed to flush on count limit shutdown: {}", e);
                                }
                                // Drop the lock before cancelling
                                drop(count);
                                cancel_clone.cancel();
                                return;
                            }
                    }
                }
            }
        });

        Ok(Self {
            event_tx,
            cancel: token,
            event_count,
            first_event_time,
        })
    }

    /// Get a sender that can be used to send events to the recorder
    pub fn event_sender(&self) -> mpsc::Sender<T> {
        self.event_tx.clone()
    }

    /// Get the count of recorded events
    pub async fn event_count(&self) -> usize {
        *self.event_count.lock().await
    }

    /// Get the elapsed time since the first event was received
    ///
    /// Returns a Result with the elapsed time or an error if no events have been received yet
    pub async fn elapsed_time(&self) -> io::Result<Duration> {
        let first_time = self.first_event_time.lock().await;
        match *first_time {
            Some(time) => Ok(time.elapsed()),
            None => Err(io::Error::other("No events received yet")),
        }
    }

    /// Shutdown the recorder
    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    /// Send events from a JSONL file to the provided event sender
    ///
    /// ### Arguments
    ///
    /// * `filename` - Path to the JSONL file to read events from
    /// * `event_tx` - A sender for events
    /// * `timed` - If true, events will be sent according to their recorded timestamps.
    ///   If false, events will be sent as fast as possible without delay.
    /// * `max_count` - Maximum number of events to send before stopping. If None, all events will be sent.
    /// * `max_time` - Maximum duration in seconds to send events before stopping. If None, no time limit.
    ///
    /// ### Returns
    ///
    /// A Result indicating success or failure with the number of events sent
    pub async fn send_events<P: AsRef<Path>>(
        filename: P,
        event_tx: &mpsc::Sender<T>,
        timed: bool,
        max_count: Option<usize>,
        max_time: Option<f64>,
    ) -> io::Result<usize> {
        // Store the display name before using filename
        let display_name = filename.as_ref().display().to_string();

        // Check if file exists
        if !filename.as_ref().exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", display_name),
            ));
        }

        // Set up start time and deadline if max_time is specified
        let start_time = Instant::now();
        let deadline = max_time.map(|secs| start_time + Duration::from_secs_f64(secs));

        // Open the file for reading using tokio's async file I/O
        let file = File::open(&filename).await?;
        let reader = BufReader::with_capacity(32768, file);
        let mut lines = reader.lines();

        let mut count = 0;
        let mut line_number = 0;
        let mut prev_timestamp: Option<u64> = None;

        // Read and send events line by line
        while let Some(line) = lines.next_line().await? {
            // Check if we've reached the maximum count
            if let Some(max) = max_count
                && count >= max
            {
                tracing::info!("Reached maximum event count ({}), stopping", max);
                break;
            }

            // Check if we've exceeded the time limit
            if let Some(end_time) = deadline
                && Instant::now() >= end_time
            {
                tracing::info!("Reached maximum time limit, stopping");
                break;
            }

            line_number += 1;

            // Skip empty lines
            if line.trim().is_empty() {
                continue;
            }

            // Try to parse the JSON
            let record: RecordEntry<T> = match serde_json::from_str(&line) {
                Ok(record) => record,
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse JSON on line {}: {}. Skipping.",
                        line_number,
                        e
                    );
                    continue;
                }
            };

            let timestamp = record.timestamp;
            let event = record.event;

            // Handle timing if needed
            if timed
                && let Some(prev) = prev_timestamp
                && timestamp > prev
            {
                let wait_time = timestamp - prev;
                tokio::time::sleep(Duration::from_millis(wait_time)).await;
            }

            // Send the event
            event_tx
                .send(event)
                .await
                .map_err(|e| io::Error::other(format!("Failed to send event: {e}")))?;

            // Update previous timestamp and count
            prev_timestamp = Some(timestamp);
            count += 1;
        }

        if count == 0 {
            tracing::warn!("No events to send from file: {}", display_name);
        } else {
            tracing::info!("Sent {} events from {}", count, display_name);
        }

        Ok(count)
    }
}

impl<T> Drop for Recorder<T> {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}

/// Helper function to create a rotated file path with an index suffix
fn create_rotated_path(base_path: &Path, index: usize) -> PathBuf {
    let path_str = base_path.to_string_lossy();

    if let Some(ext_pos) = path_str.rfind('.') {
        // If there's an extension, insert the index before it
        let (file_path, extension) = path_str.split_at(ext_pos);
        PathBuf::from(format!("{}{}{}", file_path, index, extension))
    } else {
        // If there's no extension, just append the index
        PathBuf::from(format!("{}{}", path_str, index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::tempdir;

    // Type alias for the TestEvent recorder
    type TestEventRecorder = Recorder<TestEvent>;

    // More complex event type
    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestEvent {
        id: u64,
        name: String,
        values: Vec<i32>,
    }

    impl TestEvent {
        // Helper method to generate a random test event
        fn new(id: u64) -> Self {
            // Generate a random number of values between 1 and 100
            let num_values = rand::random_range(1..=100);

            // Generate random values (integers between -100 and 100)
            let values = (0..num_values)
                .map(|_| rand::random_range(-100..=100))
                .collect();

            // Create a name based on the ID
            let name = format!("event_{}", id);

            TestEvent { id, name, values }
        }

        // Helper method to generate a vector of random events
        fn generate_events(count: usize) -> Vec<Self> {
            (0..count).map(|i| Self::new(i as u64)).collect()
        }
    }

    #[tokio::test]
    async fn test_recorder_streams_events_to_file() {
        // Create a temporary directory for output files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("events.jsonl");

        let token = CancellationToken::new();
        let recorder = TestEventRecorder::new(token.clone(), &file_path, None, None, None)
            .await
            .unwrap();
        let event_tx = recorder.event_sender();

        // Create test events using generate_events
        let events = TestEvent::generate_events(2);
        let event1 = events[0].clone();
        let event2 = events[1].clone();

        // Wait some time before the first event
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Send the events
        for event in &events {
            event_tx.send(event.clone()).await.unwrap();
        }

        // Allow some time for processing
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Check that both events were recorded
        assert_eq!(recorder.event_count().await, 2);

        // Check that the elapsed time is between 5 and 15 milliseconds
        let elapsed_ms = recorder.elapsed_time().await.unwrap().as_millis();
        if !(5..=15).contains(&elapsed_ms) {
            println!("Actual elapsed time: {} ms", elapsed_ms);
            assert!((5..=15).contains(&elapsed_ms));
        }

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

        // Parse the lines to verify events
        let entry1: RecordEntry<TestEvent> = serde_json::from_str(lines[0]).unwrap();
        let entry2: RecordEntry<TestEvent> = serde_json::from_str(lines[1]).unwrap();

        assert_eq!(entry1.event, event1);
        assert_eq!(entry2.event, event2);
        assert!(entry2.timestamp >= entry1.timestamp);
    }

    #[ignore]
    #[tokio::test]
    async fn load_test_100k_events() {
        // Create a temporary directory for output files
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("events.jsonl");

        // Create a cancellation token for the recorder
        let token = CancellationToken::new();

        // Set max lines per file to 10,000 (should create 10 files total)
        const MAX_LINES_PER_FILE: usize = 10_000;
        let recorder = TestEventRecorder::new(
            token.clone(),
            &file_path,
            Some(MAX_LINES_PER_FILE),
            None,
            None,
        )
        .await
        .unwrap();
        let event_tx = recorder.event_sender();

        // Define number of events to generate
        const NUM_EVENTS: usize = 100_000;
        println!("Generating {} events...", NUM_EVENTS);

        // Generate events using the helper method
        let events = TestEvent::generate_events(NUM_EVENTS);

        // Send events with progress reporting
        for (i, event) in events.iter().enumerate() {
            event_tx.send(event.clone()).await.unwrap();

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
        assert_eq!(count, NUM_EVENTS);

        // Force a clean shutdown to flush all pending writes
        recorder.shutdown();
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check for the existence of all expected files
        let base_file = file_path.clone();
        let mut found_files = Vec::new();

        // Check base file
        if base_file.exists() {
            found_files.push(base_file.clone());
        }

        // Check rotated files (1-9)
        for i in 1..=9 {
            let rotated_path = create_rotated_path(&base_file, i);
            if rotated_path.exists() {
                found_files.push(rotated_path);
            }
        }

        // Check that we have exactly 10 files
        assert_eq!(
            found_files.len(),
            10,
            "Expected 10 files due to rotation with 10k events each"
        );

        // Add more stringent check for each file size
        for (i, file_path) in found_files.iter().enumerate() {
            let content = fs::read_to_string(file_path).await.unwrap();
            let line_count = content.lines().count();

            if i < found_files.len() - 1 {
                // All files except the last one should have exactly MAX_LINES_PER_FILE lines
                assert_eq!(
                    line_count,
                    MAX_LINES_PER_FILE,
                    "File {} should contain exactly {} lines",
                    file_path.display(),
                    MAX_LINES_PER_FILE
                );
            } else {
                // The last file might have fewer lines
                assert!(
                    line_count <= MAX_LINES_PER_FILE,
                    "Last file should contain at most {} lines",
                    MAX_LINES_PER_FILE
                );
            }
        }

        // Count total lines across all files
        let mut total_lines = 0;

        // Check that timestamps are weakly sorted within each file
        for (i, file_path) in found_files.iter().enumerate() {
            println!("Checking file {}: {}", i, file_path.display());

            // Count lines in the file
            let content = fs::read_to_string(file_path).await.unwrap();
            let line_count = content.lines().count();

            // Should have MAX_LINES_PER_FILE lines in each file (except maybe the last one)
            if i < found_files.len() - 1 {
                assert_eq!(
                    line_count, MAX_LINES_PER_FILE,
                    "Each file except possibly the last should have exactly MAX_LINES_PER_FILE lines"
                );
            }

            total_lines += line_count;

            // Check that timestamps are weakly sorted within each file
            let file = File::open(file_path).await.unwrap();
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            let mut prev_timestamp: Option<u64> = None;
            let mut line_number = 0;
            let mut unsorted_count = 0;

            // Check timestamps in the file without loading everything into memory
            while let Some(line) = lines.next_line().await.unwrap() {
                line_number += 1;
                let entry: RecordEntry<TestEvent> = serde_json::from_str(&line).unwrap();

                if let Some(prev) = prev_timestamp
                    && entry.timestamp < prev
                {
                    unsorted_count += 1;
                    if unsorted_count <= 5 {
                        // Only log first 5 violations to avoid spam
                        println!(
                            "Timestamp order violation in file {} at line {}: {} < {}",
                            file_path.display(),
                            line_number,
                            entry.timestamp,
                            prev
                        );
                    }
                }

                prev_timestamp = Some(entry.timestamp);
            }

            assert_eq!(
                unsorted_count, 0,
                "Timestamps should be weakly sorted within each file"
            );
        }

        assert_eq!(
            total_lines, NUM_EVENTS,
            "Total lines across all files should match NUM_EVENTS"
        );

        println!("Load test with file rotation completed successfully");
    }
}
