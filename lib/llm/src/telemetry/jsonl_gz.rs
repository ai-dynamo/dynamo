// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic rotating gzip JSONL sink shared by audit and request trace.
//!
//! Records published on the caller's bus are forwarded into an internal mpsc,
//! batched into uncompressed bytes, and appended as gzip members. Segments roll
//! when uncompressed bytes, record-line, or time thresholds are exceeded.
//!
//! The destination of each segment is abstracted behind the [`SegmentSink`]
//! trait. Two implementations are provided:
//!
//! - [`FileSegmentSink`] — appends each gzip batch to a numbered local file.
//! - (in `s3_segment_sink.rs`) `S3SegmentSink` — buffers a segment in memory
//!   and uploads it as one S3 object on close.
//!
//! Callers that just want disk output can keep using [`JsonlGzipWriter::new`];
//! callers that want a custom destination use [`JsonlGzipWriter::with_segment_sink`].

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context as _, anyhow};
use async_trait::async_trait;
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
    /// If set, roll the current segment after this much wall-clock time has
    /// elapsed since it was opened, even if size and line thresholds have
    /// not been hit. Useful for cloud destinations where stale data should
    /// not sit in memory during quiet periods.
    pub roll_interval: Option<Duration>,
    /// Capacity of the internal mpsc channel between the sink's `emit()`
    /// calls and the background writer task. Default 2048.
    pub channel_capacity: usize,
}

impl Default for JsonlGzipSinkOptions {
    fn default() -> Self {
        Self {
            buffer_bytes: 1024 * 1024,
            flush_interval: Duration::from_millis(1000),
            roll_uncompressed_bytes: 256 * 1024 * 1024,
            roll_lines: None,
            roll_interval: None,
            channel_capacity: 2048,
        }
    }
}

/// Destination for rotated gzip-compressed JSONL segments.
///
/// `append_to_segment` is called whenever the writer flushes a buffered
/// batch — multiple appends per segment are normal, and each `gz_bytes`
/// payload is a complete gzip member that can be concatenated with previous
/// appends to form the full segment.
///
/// `close_segment` is called exactly once per segment, after the final
/// append for that `seq`, before the writer advances to the next segment.
/// File-backed destinations may treat this as a no-op; cloud destinations
/// (S3, GCS) typically perform the actual upload here.
///
/// Both methods are expected to handle their own retries; on terminal
/// failure they should `tracing::warn!` and return `Ok(())` rather than
/// propagate, so a single bad segment does not kill the writer task.
/// (Returning `Err` is allowed but currently logged-and-ignored by the
/// writer.)
#[async_trait]
pub trait SegmentSink: Send + Sync + 'static {
    async fn append_to_segment(&self, seq: u64, gz_bytes: Vec<u8>) -> anyhow::Result<()>;
    async fn close_segment(&self, seq: u64) -> anyhow::Result<()>;
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
    /// Construct a writer that persists segments to numbered local files
    /// under `path` (e.g. `path.000000.jsonl.gz`, `path.000001.jsonl.gz`).
    pub async fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        let segment_sink = Arc::new(FileSegmentSink::new(&path)?) as Arc<dyn SegmentSink>;
        Self::with_segment_sink(segment_sink, options)
    }

    /// Construct a writer that hands each rotated segment to the supplied
    /// [`SegmentSink`]. Use this for cloud destinations such as S3.
    pub fn with_segment_sink(
        segment_sink: Arc<dyn SegmentSink>,
        options: JsonlGzipSinkOptions,
    ) -> anyhow::Result<Self> {
        let shutdown = CancellationToken::new();
        let (tx, rx) = mpsc::channel::<T>(options.channel_capacity.max(1));
        let mut writer = GzipBatchWriter::new(segment_sink, options);
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

/// File-backed [`SegmentSink`]. Appends each gzip member to a numbered
/// segment file beside `base_path`, mirroring the original (pre-trait)
/// behavior of this module.
pub struct FileSegmentSink {
    base_path: PathBuf,
    starting_index: u64,
}

impl FileSegmentSink {
    pub fn new(path: &str) -> anyhow::Result<Self> {
        let base_path = PathBuf::from(path);
        if let Some(parent) = base_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating gzip jsonl directory {}", parent.display()))?;
        }

        let starting_index = next_segment_index(&base_path)?;
        Ok(Self {
            base_path,
            starting_index,
        })
    }

    /// Index of the first segment this sink will use. The writer adds this
    /// to its own per-process counter so existing on-disk segments are not
    /// overwritten when a process restarts.
    pub fn starting_index(&self) -> u64 {
        self.starting_index
    }
}

#[async_trait]
impl SegmentSink for FileSegmentSink {
    async fn append_to_segment(&self, seq: u64, gz_bytes: Vec<u8>) -> anyhow::Result<()> {
        let path = segment_path(&self.base_path, self.starting_index.saturating_add(seq));
        tokio::task::spawn_blocking(move || write_gzip_member(path, gz_bytes))
            .await
            .context("gzip jsonl writer task panicked")??;
        Ok(())
    }

    async fn close_segment(&self, _seq: u64) -> anyhow::Result<()> {
        // File-backed segments are written as we go; nothing to finalize.
        Ok(())
    }
}

struct GzipBatchWriter<T: Serialize> {
    segment_sink: Arc<dyn SegmentSink>,
    current_seq: u64,
    start_time: Instant,
    segment_opened_at: tokio::time::Instant,
    batch: Vec<u8>,
    segment_uncompressed_bytes: u64,
    segment_lines: u64,
    options: JsonlGzipSinkOptions,
    _marker: std::marker::PhantomData<fn(T)>,
}

impl<T: Serialize> GzipBatchWriter<T> {
    fn new(segment_sink: Arc<dyn SegmentSink>, options: JsonlGzipSinkOptions) -> Self {
        Self {
            segment_sink,
            current_seq: 0,
            start_time: Instant::now(),
            segment_opened_at: tokio::time::Instant::now(),
            batch: Vec::with_capacity(options.buffer_bytes.max(1)),
            segment_uncompressed_bytes: 0,
            segment_lines: 0,
            options,
            _marker: std::marker::PhantomData,
        }
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
            self.roll_segment().await?;
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

    fn time_to_roll(&self) -> bool {
        if self.segment_lines == 0 {
            return false;
        }
        self.options
            .roll_interval
            .is_some_and(|interval| self.segment_opened_at.elapsed() >= interval)
    }

    async fn roll_segment(&mut self) -> anyhow::Result<()> {
        let closing_seq = self.current_seq;
        if let Err(err) = self.segment_sink.close_segment(closing_seq).await {
            tracing::warn!("gzip jsonl close_segment seq={closing_seq} failed: {err}");
        }
        self.current_seq = self.current_seq.saturating_add(1);
        self.segment_opened_at = tokio::time::Instant::now();
        self.segment_uncompressed_bytes = 0;
        self.segment_lines = 0;
        Ok(())
    }

    async fn flush_batch(&mut self) -> anyhow::Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }

        let batch = std::mem::take(&mut self.batch);
        let gz_bytes = tokio::task::spawn_blocking(move || compress_member(batch))
            .await
            .context("gzip jsonl compress task panicked")?
            .context("compressing gzip jsonl batch")?;
        let seq = self.current_seq;

        if let Err(err) = self.segment_sink.append_to_segment(seq, gz_bytes).await {
            tracing::warn!("gzip jsonl append_to_segment seq={seq} failed: {err}");
        }

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

    // Tick more often than `roll_interval` so the rotation check is
    // responsive but cheap. If `roll_interval` is None, this tick is
    // effectively unused (still cheap).
    let roll_check_interval = writer
        .options
        .roll_interval
        .map(|d| (d / 4).max(Duration::from_millis(50)))
        .unwrap_or_else(|| Duration::from_secs(60));
    let mut roll_tick = tokio::time::interval(roll_check_interval);
    roll_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

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
                let final_seq = writer.current_seq;
                if let Err(err) = writer.segment_sink.close_segment(final_seq).await {
                    tracing::warn!(
                        "gzip jsonl close_segment seq={final_seq} during shutdown failed: {err}"
                    );
                }
                return;
            }
            _ = flush_tick.tick() => {
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("gzip jsonl sink failed flush: {err}");
                }
            }
            _ = roll_tick.tick() => {
                if writer.time_to_roll() {
                    if let Err(err) = writer.flush_batch().await {
                        tracing::warn!("gzip jsonl sink failed pre-roll flush: {err}");
                    }
                    if let Err(err) = writer.roll_segment().await {
                        tracing::warn!("gzip jsonl sink failed time-based roll: {err}");
                    }
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
                        let final_seq = writer.current_seq;
                        if let Err(err) = writer.segment_sink.close_segment(final_seq).await {
                            tracing::warn!(
                                "gzip jsonl close_segment seq={final_seq} on rx-close failed: {err}"
                            );
                        }
                        return;
                    }
                }
            }
        }
    }
}

/// Compress a batch of bytes into a complete (self-contained) gzip member.
fn compress_member(batch: Vec<u8>) -> anyhow::Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::with_capacity(batch.len() / 2), Compression::default());
    encoder.write_all(&batch).context("gzip write_all")?;
    let gz_bytes = encoder.finish().context("gzip finish")?;
    Ok(gz_bytes)
}

fn write_gzip_member(path: PathBuf, gz_bytes: Vec<u8>) -> anyhow::Result<()> {
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
    let mut writer = BufWriter::new(file);
    writer
        .write_all(&gz_bytes)
        .with_context(|| format!("appending gzip jsonl segment {}", path.display()))?;
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
    use std::sync::Mutex;

    use flate2::read::MultiGzDecoder;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use super::*;

    #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
    struct TestRecord {
        id: u64,
        name: String,
    }

    #[derive(Default)]
    struct CollectingSegmentSink {
        appends: Mutex<Vec<(u64, Vec<u8>)>>,
        closes: Mutex<Vec<u64>>,
    }

    #[async_trait::async_trait]
    impl SegmentSink for CollectingSegmentSink {
        async fn append_to_segment(&self, seq: u64, gz_bytes: Vec<u8>) -> anyhow::Result<()> {
            self.appends.lock().unwrap().push((seq, gz_bytes));
            Ok(())
        }

        async fn close_segment(&self, seq: u64) -> anyhow::Result<()> {
            self.closes.lock().unwrap().push(seq);
            Ok(())
        }
    }

    fn decompress(gz_bytes: &[u8]) -> String {
        let mut decoder = MultiGzDecoder::new(gz_bytes);
        let mut content = String::new();
        decoder
            .read_to_string(&mut content)
            .expect("gzip member decompresses");
        content
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
                roll_interval: None,
                channel_capacity: 2048,
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
                roll_interval: None,
                channel_capacity: 2048,
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

    #[tokio::test]
    async fn segment_sink_receives_appended_records() {
        let collector = Arc::new(CollectingSegmentSink::default());
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::with_segment_sink(
            collector.clone() as Arc<dyn SegmentSink>,
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_millis(20),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                roll_interval: None,
                channel_capacity: 2048,
            },
        )
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

        // Wait for the writer's flush_interval to push at least one append.
        for _ in 0..100 {
            if !collector.appends.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        // Drop the writer to trigger the shutdown drain (flushes remaining).
        drop(writer);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let appends = collector.appends.lock().unwrap().clone();
        assert!(!appends.is_empty(), "expected at least one append");

        // All appends are for seq 0 (no rotation triggered).
        assert!(appends.iter().all(|(seq, _)| *seq == 0));

        // Concatenate all gzip members and decompress.
        let mut all_bytes = Vec::new();
        for (_, bytes) in appends {
            all_bytes.extend_from_slice(&bytes);
        }
        let content = decompress(&all_bytes);
        assert!(content.contains("\"name\":\"first\""));
        assert!(content.contains("\"name\":\"second\""));
    }

    #[tokio::test]
    async fn segment_sink_closes_old_segment_on_roll() {
        let collector = Arc::new(CollectingSegmentSink::default());
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::with_segment_sink(
            collector.clone() as Arc<dyn SegmentSink>,
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_millis(20),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: Some(1),
                roll_interval: None,
                channel_capacity: 2048,
            },
        )
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

        // Wait for both records to land + the rotation close to fire.
        for _ in 0..100 {
            {
                let closes = collector.closes.lock().unwrap();
                if !closes.is_empty() {
                    break;
                }
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }

        drop(writer);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let closes = collector.closes.lock().unwrap().clone();
        assert!(closes.contains(&0), "seq 0 should be closed when rolling");

        // Records should appear in their own seq groups.
        let appends = collector.appends.lock().unwrap().clone();
        let seq0: Vec<_> = appends.iter().filter(|(s, _)| *s == 0).collect();
        let seq1: Vec<_> = appends.iter().filter(|(s, _)| *s == 1).collect();
        assert!(!seq0.is_empty(), "seq 0 has appends");
        assert!(!seq1.is_empty(), "seq 1 has appends");
    }

    #[tokio::test]
    async fn segment_sink_rotates_on_time_interval() {
        let collector = Arc::new(CollectingSegmentSink::default());
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::with_segment_sink(
            collector.clone() as Arc<dyn SegmentSink>,
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_millis(20),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                roll_interval: Some(Duration::from_millis(100)),
                channel_capacity: 2048,
            },
        )
        .expect("writer should start");

        writer
            .send(TestRecord {
                id: 1,
                name: "first".to_string(),
            })
            .await
            .unwrap();

        // Wait long enough for the 100ms roll_interval to fire after the
        // first append. We expect seq 0 to be closed and a fresh segment
        // available for subsequent records.
        tokio::time::sleep(Duration::from_millis(300)).await;

        writer
            .send(TestRecord {
                id: 2,
                name: "second".to_string(),
            })
            .await
            .unwrap();

        drop(writer);
        tokio::time::sleep(Duration::from_millis(100)).await;

        let closes = collector.closes.lock().unwrap().clone();
        assert!(
            closes.contains(&0),
            "time-based rotation should close seq 0 (closes={closes:?})"
        );
    }
}
