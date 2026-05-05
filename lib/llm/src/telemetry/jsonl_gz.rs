// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic rotating gzip JSONL sink shared by audit and agent trace.
//!
//! Records published on the caller's bus are forwarded into an internal mpsc,
//! batched into uncompressed bytes, and appended as gzip members. Segments roll
//! when uncompressed bytes or record-line thresholds are exceeded.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context as _, anyhow};
use flate2::{Compression, write::GzEncoder};
use serde::Serialize;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub struct JsonlGzipSinkOptions {
    pub buffer_bytes: usize,
    pub flush_interval: Duration,
    pub roll_uncompressed_bytes: u64,
    pub roll_lines: Option<u64>,
}

impl Default for JsonlGzipSinkOptions {
    fn default() -> Self {
        Self {
            buffer_bytes: 1024 * 1024,
            flush_interval: Duration::from_millis(1000),
            roll_uncompressed_bytes: 256 * 1024 * 1024,
            roll_lines: None,
        }
    }
}

/// Channel-backed handle for a rotating gzip JSONL sink. Drop cancels the
/// writer task; remaining records are flushed before exit.
pub struct JsonlGzipWriter<T> {
    tx: mpsc::Sender<T>,
    shutdown: CancellationToken,
}

#[derive(Serialize)]
struct GzipEntry<'a, T: Serialize> {
    timestamp: u64,
    event: &'a T,
}

impl<T> JsonlGzipWriter<T>
where
    T: Serialize + Send + Sync + 'static,
{
    pub async fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let shutdown = CancellationToken::new();
        let (tx, rx) = mpsc::channel::<T>(2048);
        let mut writer = GzipBatchWriter::new(path.clone(), options)
            .with_context(|| format!("opening gzip jsonl sink at {path}"))?;
        let worker_shutdown = shutdown.clone();

        tokio::spawn(async move {
            run_gzip_writer(rx, &mut writer, worker_shutdown).await;
        });

        Ok(Self { tx, shutdown })
    }

    /// Forward a record to the writer task. Returns `Err` if the worker has
    /// shut down.
    pub async fn send(&self, rec: T) -> Result<(), mpsc::error::SendError<T>> {
        self.tx.send(rec).await
    }
}

impl<T> Drop for JsonlGzipWriter<T> {
    fn drop(&mut self) {
        self.shutdown.cancel();
    }
}

struct GzipBatchWriter<T: Serialize> {
    base_path: PathBuf,
    current_index: u64,
    start_time: Instant,
    batch: Vec<u8>,
    segment_uncompressed_bytes: u64,
    segment_lines: u64,
    options: JsonlGzipSinkOptions,
    _marker: std::marker::PhantomData<fn(T)>,
}

impl<T: Serialize> GzipBatchWriter<T> {
    fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let base_path = PathBuf::from(path);
        if let Some(parent) = base_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating gzip jsonl directory {}", parent.display()))?;
        }

        let current_index = next_segment_index(&base_path)?;
        Ok(Self {
            base_path,
            current_index,
            start_time: Instant::now(),
            batch: Vec::with_capacity(options.buffer_bytes.max(1)),
            segment_uncompressed_bytes: 0,
            segment_lines: 0,
            options,
            _marker: std::marker::PhantomData,
        })
    }

    async fn push(&mut self, rec: &T) -> anyhow::Result<()> {
        let entry = GzipEntry {
            timestamp: self.start_time.elapsed().as_millis() as u64,
            event: rec,
        };
        let mut line = serde_json::to_vec(&entry).context("serializing gzip jsonl record")?;
        line.push(b'\n');

        let line_len = line.len() as u64;
        if self.should_roll_before(line_len) {
            self.flush_batch().await?;
            self.roll_segment();
        }

        self.batch.extend_from_slice(&line);
        self.segment_uncompressed_bytes = self.segment_uncompressed_bytes.saturating_add(line_len);
        self.segment_lines = self.segment_lines.saturating_add(1);

        if self.batch.len() >= self.options.buffer_bytes.max(1) {
            self.flush_batch().await?;
        }

        Ok(())
    }

    fn should_roll_before(&self, next_line_len: u64) -> bool {
        if self.segment_lines == 0 {
            return false;
        }

        if self
            .segment_uncompressed_bytes
            .saturating_add(next_line_len)
            > self.options.roll_uncompressed_bytes.max(1)
        {
            return true;
        }

        self.options
            .roll_lines
            .is_some_and(|limit| self.segment_lines >= limit)
    }

    fn roll_segment(&mut self) {
        self.current_index = self.current_index.saturating_add(1);
        self.segment_uncompressed_bytes = 0;
        self.segment_lines = 0;
    }

    async fn flush_batch(&mut self) -> anyhow::Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }

        let path = segment_path(&self.base_path, self.current_index);
        let batch = std::mem::take(&mut self.batch);

        tokio::task::spawn_blocking(move || write_gzip_member(path, batch))
            .await
            .context("gzip jsonl writer task panicked")??;

        Ok(())
    }
}

async fn run_gzip_writer<T: Serialize>(
    mut rx: mpsc::Receiver<T>,
    writer: &mut GzipBatchWriter<T>,
    shutdown: CancellationToken,
) {
    let mut flush_tick =
        tokio::time::interval(writer.options.flush_interval.max(Duration::from_millis(1)));
    flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                while let Ok(rec) = rx.try_recv() {
                    if let Err(err) = writer.push(&rec).await {
                        tracing::warn!("gzip jsonl sink dropped record during shutdown: {err}");
                    }
                }
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("gzip jsonl sink failed final flush: {err}");
                }
                return;
            }
            _ = flush_tick.tick() => {
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("gzip jsonl sink failed flush: {err}");
                }
            }
            msg = rx.recv() => {
                match msg {
                    Some(rec) => {
                        if let Err(err) = writer.push(&rec).await {
                            tracing::warn!("gzip jsonl sink dropped record: {err}");
                        }
                    }
                    None => {
                        if let Err(err) = writer.flush_batch().await {
                            tracing::warn!("gzip jsonl sink failed final flush: {err}");
                        }
                        return;
                    }
                }
            }
        }
    }
}

fn write_gzip_member(path: PathBuf, batch: Vec<u8>) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating gzip jsonl directory {}", parent.display()))?;
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening gzip jsonl segment {}", path.display()))?;
    let writer = BufWriter::new(file);
    let mut encoder = GzEncoder::new(writer, Compression::default());
    encoder
        .write_all(&batch)
        .with_context(|| format!("compressing gzip jsonl segment {}", path.display()))?;
    let mut writer = encoder
        .finish()
        .with_context(|| format!("finishing gzip jsonl segment {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing gzip jsonl segment {}", path.display()))?;
    Ok(())
}

pub fn segment_path(base_path: &Path, index: u64) -> PathBuf {
    let raw = base_path.to_string_lossy();
    let prefix = raw
        .strip_suffix(".jsonl.gz")
        .or_else(|| raw.strip_suffix(".jsonl"))
        .unwrap_or(&raw);
    PathBuf::from(format!("{prefix}.{index:06}.jsonl.gz"))
}

fn next_segment_index(base_path: &Path) -> anyhow::Result<u64> {
    for index in 0..u64::MAX {
        if !segment_path(base_path, index).try_exists()? {
            return Ok(index);
        }
    }
    Err(anyhow!(
        "no available gzip jsonl segment index for {}",
        base_path.display()
    ))
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::MultiGzDecoder;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use super::*;

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    struct TestRecord {
        id: u64,
        name: String,
    }

    fn read_gzip_jsonl(path: &Path) -> String {
        let bytes = std::fs::read(path).expect("gzip segment readable");
        let mut decoder = MultiGzDecoder::new(bytes.as_slice());
        let mut content = String::new();
        decoder
            .read_to_string(&mut content)
            .expect("gzip segment decompresses");
        content
    }

    #[tokio::test]
    async fn appends_gzip_members() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
            },
        )
        .await
        .expect("writer should start");

        writer
            .send(TestRecord {
                id: 1,
                name: "first".to_string(),
            })
            .await
            .unwrap();
        writer
            .send(TestRecord {
                id: 2,
                name: "second".to_string(),
            })
            .await
            .unwrap();

        let segment = segment_path(&path, 0);
        let mut content = String::new();
        for _ in 0..100 {
            if segment.exists() {
                content = read_gzip_jsonl(&segment);
                if content.matches("\"name\":").count() == 2 {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(content.contains("\"name\":\"first\""));
        assert!(content.contains("\"name\":\"second\""));
    }

    #[tokio::test]
    async fn rolls_segments_on_line_threshold() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: Some(1),
            },
        )
        .await
        .expect("writer should start");

        writer
            .send(TestRecord {
                id: 1,
                name: "first".to_string(),
            })
            .await
            .unwrap();
        writer
            .send(TestRecord {
                id: 2,
                name: "second".to_string(),
            })
            .await
            .unwrap();

        let first = segment_path(&path, 0);
        let second = segment_path(&path, 1);
        let mut first_content = String::new();
        let mut second_content = String::new();
        for _ in 0..100 {
            if first.exists() && second.exists() {
                first_content = read_gzip_jsonl(&first);
                second_content = read_gzip_jsonl(&second);
                if first_content.contains("\"name\":\"first\"")
                    && second_content.contains("\"name\":\"second\"")
                {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        assert!(first_content.contains("\"name\":\"first\""));
        assert!(second_content.contains("\"name\":\"second\""));
    }
}
