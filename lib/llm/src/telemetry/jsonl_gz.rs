// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic rotating gzip JSONL sink shared by audit and request trace.
//!
//! Records published on the caller's bus are forwarded into an internal mpsc,
//! batched into uncompressed bytes, and appended as gzip members. Segments roll
//! when uncompressed bytes or record-line thresholds are exceeded.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

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
    /// Maximum number of segments to retain for this exact output prefix.
    /// `None` disables count-based retention.
    pub max_segments: Option<usize>,
    /// Total compressed file bytes for this prefix, including the active segment.
    /// The active segment is never removed, so this is a soft limit.
    pub max_bytes: Option<u64>,
    /// Expire segments by their last modification time, including idle segments.
    pub max_age: Option<Duration>,
}

impl Default for JsonlGzipSinkOptions {
    fn default() -> Self {
        Self {
            buffer_bytes: 1024 * 1024,
            flush_interval: Duration::from_millis(1000),
            roll_uncompressed_bytes: 256 * 1024 * 1024,
            roll_lines: None,
            max_segments: None,
            max_bytes: None,
            max_age: None,
        }
    }
}

/// Channel-backed handle for a rotating gzip JSONL sink.
///
/// Drop requests writer shutdown, but cannot wait for its final flush. Call
/// [`Self::shutdown`] or [`Self::close`] when completion must be awaited.
pub struct JsonlGzipWriter<T> {
    tx: Option<mpsc::Sender<T>>,
    shutdown: CancellationToken,
    worker: Option<tokio::task::JoinHandle<anyhow::Result<()>>>,
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
        let display_path = path.clone();
        let mut writer =
            tokio::task::spawn_blocking(move || GzipBatchWriter::<T>::new(path, options))
                .await
                .context("gzip jsonl sink initializer panicked")?
                .with_context(|| format!("opening gzip jsonl sink at {display_path}"))?;
        let worker_shutdown = shutdown.clone();

        let worker =
            tokio::spawn(async move { run_gzip_writer(rx, &mut writer, worker_shutdown).await });

        Ok(Self {
            tx: Some(tx),
            shutdown,
            worker: Some(worker),
        })
    }

    /// Forward a record to the writer task. Returns `Err` if the worker has
    /// shut down.
    pub async fn send(&self, rec: T) -> Result<(), mpsc::error::SendError<T>> {
        match &self.tx {
            Some(tx) => tx.send(rec).await,
            None => Err(mpsc::error::SendError(rec)),
        }
    }

    /// A cloned handle to the writer's input channel, for enqueuing records
    /// concurrently without holding a lock on the writer. `None` after shutdown.
    /// Sending on the clone fails once the writer closes admission for shutdown.
    pub fn sender(&self) -> Option<mpsc::Sender<T>> {
        self.tx.clone()
    }

    /// Drain all accepted records, flush the active segment, and wait for the
    /// writer task to exit. Cancelling this wait leaves the task available for a
    /// subsequent call to await.
    pub async fn shutdown(&mut self) -> anyhow::Result<()> {
        self.tx.take();
        self.shutdown.cancel();
        if let Some(worker) = self.worker.as_mut() {
            // Keep the handle until completion so cancellation of this await
            // does not prevent a subsequent shutdown from joining the worker.
            let result = worker.await;
            self.worker.take();
            result.context("gzip jsonl writer task panicked")??;
        }
        Ok(())
    }

    /// Consuming convenience wrapper around [`Self::shutdown`].
    pub async fn close(mut self) -> anyhow::Result<()> {
        self.shutdown().await
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
    active_file: Option<File>,
    prune_pending: bool,
    segment_uncompressed_bytes: u64,
    segment_lines: u64,
    options: JsonlGzipSinkOptions,
    last_write_time: Option<SystemTime>,
    // Retention requires exclusive ownership of the prefix. Other writers take
    // a shared lock, so a cleaner cannot unlink another live writer's segment.
    prefix_lock: Arc<File>,
    _marker: std::marker::PhantomData<fn(T)>,
}

impl<T: Serialize> GzipBatchWriter<T> {
    fn new(path: String, options: JsonlGzipSinkOptions) -> anyhow::Result<Self> {
        if options.max_segments == Some(0) {
            return Err(anyhow!("gzip jsonl max_segments must be positive"));
        }
        if options.max_bytes == Some(0) || options.max_age == Some(Duration::ZERO) {
            return Err(anyhow!("gzip jsonl retention limits must be positive"));
        }

        let base_path = PathBuf::from(path);
        if let Some(parent) = base_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("creating gzip jsonl directory {}", parent.display()))?;
        }

        let prefix_lock = lock_prefix(
            &base_path,
            options.max_bytes.is_some() || options.max_age.is_some(),
        )?;
        let current_index = next_segment_index(&base_path)?;
        Ok(Self {
            base_path,
            current_index,
            start_time: Instant::now(),
            batch: Vec::with_capacity(options.buffer_bytes.max(1)),
            active_file: None,
            prune_pending: true,
            segment_uncompressed_bytes: 0,
            segment_lines: 0,
            options,
            last_write_time: None,
            prefix_lock: Arc::new(prefix_lock),
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
        self.active_file = None;
        self.prune_pending = true;
        self.segment_uncompressed_bytes = 0;
        self.segment_lines = 0;
    }

    async fn flush_batch(&mut self) -> anyhow::Result<()> {
        if self.batch.is_empty() {
            return Ok(());
        }

        let batch = std::mem::take(&mut self.batch);
        let base_path = self.base_path.clone();
        let current_index = self.current_index;
        let max_segments = self.options.max_segments;
        let max_bytes = self.options.max_bytes;
        let max_age = self.options.max_age;
        let active_file = self.active_file.take();
        let prune_pending = self.prune_pending;
        let prefix_lock = self.prefix_lock.clone();

        let (active_file, current_index, result) = tokio::task::spawn_blocking(move || {
            // A blocking write outlives cancellation of the async writer.
            let _prefix_lock = prefix_lock;
            let (mut active_file, current_index, path) = match active_file {
                Some(file) => (file, current_index, segment_path(&base_path, current_index)),
                None => match create_available_segment(&base_path, current_index) {
                    Ok(segment) => segment,
                    Err(err) => return (None, current_index, Err(err)),
                },
            };

            let result = write_gzip_member(&mut active_file, &path, batch).map(|()| {
                if prune_pending
                    && let Some(max_segments) = max_segments
                    && let Err(err) = prune_segments(&base_path, max_segments)
                {
                    tracing::warn!("gzip jsonl sink failed to prune old segments: {err}");
                }
                if prune_pending
                    && (max_bytes.is_some() || max_age.is_some())
                    && let Err(err) = prune_retained_segments(
                        &base_path,
                        Some(current_index),
                        max_bytes,
                        max_age,
                        SystemTime::now(),
                    )
                {
                    tracing::warn!("gzip jsonl sink failed retention cleanup: {err}");
                }
            });
            (Some(active_file), current_index, result)
        })
        .await
        .context("gzip jsonl writer task panicked")?;

        self.active_file = active_file;
        self.current_index = current_index;
        if result.is_ok() {
            self.prune_pending = false;
            self.last_write_time = Some(SystemTime::now());
        }
        result?;

        Ok(())
    }

    async fn maintain_retention(&mut self, now: SystemTime) -> anyhow::Result<()> {
        // Called after flushing. Close an idle segment before it expires, so
        // retention still progresses when no more requests arrive.
        if let (Some(max_age), Some(last_write)) = (self.options.max_age, self.last_write_time)
            && now
                .duration_since(last_write)
                .is_ok_and(|age| age >= max_age)
            && self.batch.is_empty()
            && self.active_file.is_some()
        {
            self.roll_segment();
        }
        let base_path = self.base_path.clone();
        let active_index = self.active_file.as_ref().map(|_| self.current_index);
        let max_bytes = self.options.max_bytes;
        let max_age = self.options.max_age;
        let prefix_lock = self.prefix_lock.clone();
        tokio::task::spawn_blocking(move || {
            let _prefix_lock = prefix_lock;
            prune_retained_segments(&base_path, active_index, max_bytes, max_age, now)
        })
        .await
        .context("gzip jsonl retention task panicked")?
    }
}

async fn run_gzip_writer<T: Serialize>(
    mut rx: mpsc::Receiver<T>,
    writer: &mut GzipBatchWriter<T>,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let mut flush_tick =
        tokio::time::interval(writer.options.flush_interval.max(Duration::from_millis(1)));
    flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let retention_enabled = writer.options.max_bytes.is_some() || writer.options.max_age.is_some();
    let mut retention_tick = tokio::time::interval(Duration::from_secs(60));
    retention_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut first_error = None;

    loop {
        tokio::select! {
            biased;
            _ = shutdown.cancelled() => {
                rx.close();
                // Outstanding permits may still publish after close. Waiting
                // for None includes them while rejecting new sends.
                while let Some(rec) = rx.recv().await {
                    if let Err(err) = writer.push(&rec).await {
                        tracing::warn!("gzip jsonl sink dropped record during shutdown: {err}");
                        first_error.get_or_insert(err);
                    }
                }
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("gzip jsonl sink failed final flush: {err}");
                    first_error.get_or_insert(err);
                }
                return first_error.map_or(Ok(()), Err);
            }
            _ = flush_tick.tick() => {
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("gzip jsonl sink failed flush: {err}");
                    first_error.get_or_insert(err);
                }
            }
            _ = retention_tick.tick(), if retention_enabled => {
                if let Err(err) = writer.flush_batch().await {
                    tracing::warn!("gzip jsonl sink failed flush before retention: {err}");
                    first_error.get_or_insert(err);
                }
                if let Err(err) = writer.maintain_retention(SystemTime::now()).await {
                    tracing::warn!("gzip jsonl sink failed retention cleanup: {err}");
                }
            }
            msg = rx.recv() => {
                match msg {
                    Some(rec) => {
                        if let Err(err) = writer.push(&rec).await {
                            tracing::warn!("gzip jsonl sink dropped record: {err}");
                            first_error.get_or_insert(err);
                        }
                    }
                    None => {
                        if let Err(err) = writer.flush_batch().await {
                            tracing::warn!("gzip jsonl sink failed final flush: {err}");
                            first_error.get_or_insert(err);
                        }
                        return first_error.map_or(Ok(()), Err);
                    }
                }
            }
        }
    }
}

fn ensure_segment_parent(path: &Path) -> anyhow::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating gzip jsonl directory {}", parent.display()))?;
    }
    Ok(())
}

fn lock_prefix(base_path: &Path, exclusive: bool) -> anyhow::Result<File> {
    let path = segment_path(base_path, 0).with_extension("lock");
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(exclusive);
    // Existing shared locks only need read access, including when a previous
    // writer created the lock file under a different UID.
    let file = match options.open(&path) {
        Ok(file) => Ok(file),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            match std::fs::OpenOptions::new()
                .create_new(true)
                .read(true)
                .write(true)
                .open(&path)
            {
                // Another writer may create the lock between open and create.
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    options.open(&path)
                }
                result => result,
            }
        }
        Err(error) => Err(error),
    }
    .with_context(|| format!("opening gzip jsonl prefix lock {}", path.display()))?;
    let result = if exclusive {
        file.try_lock()
    } else {
        file.try_lock_shared()
    };
    check_prefix_lock_result(result, exclusive).with_context(|| format!(
        "locking gzip jsonl prefix {}; retention requires a separate output prefix for each writer",
        base_path.display()
    ))?;
    // Keep this file on disk: removing it would let new writers lock a different
    // inode while an existing writer still owns this one.
    Ok(file)
}

fn check_prefix_lock_result(
    result: Result<(), std::fs::TryLockError>,
    exclusive: bool,
) -> Result<(), std::fs::TryLockError> {
    match result {
        Err(std::fs::TryLockError::Error(error))
            if !exclusive && error.kind() == std::io::ErrorKind::Unsupported =>
        {
            tracing::warn!(%error, "gzip jsonl shared prefix lock unsupported; continuing without a lock");
            Ok(())
        }
        result => result,
    }
}

fn prune_retained_segments(
    base_path: &Path,
    active_index: Option<u64>,
    max_bytes: Option<u64>,
    max_age: Option<Duration>,
    now: SystemTime,
) -> anyhow::Result<()> {
    let mut segments = Vec::new();
    let mut total_bytes = 0_u64;
    for (index, path) in matching_segments(base_path)?.prunable {
        let metadata = std::fs::symlink_metadata(&path)
            .with_context(|| format!("reading retained segment {}", path.display()))?;
        if !metadata.is_file() {
            continue;
        }
        let is_active = Some(index) == active_index;
        if !is_active
            && let Some(limit) = max_age
            && now
                .duration_since(metadata.modified()?)
                .is_ok_and(|age| age >= limit)
        {
            std::fs::remove_file(&path)
                .with_context(|| format!("expiring gzip jsonl segment {}", path.display()))?;
            continue;
        }
        total_bytes = total_bytes.saturating_add(metadata.len());
        if !is_active {
            segments.push((path, metadata.len()));
        }
    }
    // Only unexpired closed segments remain, ordered by index across restarts.
    if let Some(limit) = max_bytes {
        for (path, bytes) in segments {
            if total_bytes <= limit {
                break;
            }
            std::fs::remove_file(&path)
                .with_context(|| format!("pruning gzip jsonl segment {}", path.display()))?;
            total_bytes = total_bytes.saturating_sub(bytes);
        }
    }
    Ok(())
}

fn create_available_segment(
    base_path: &Path,
    initial_index: u64,
) -> anyhow::Result<(File, u64, PathBuf)> {
    let initial_path = segment_path(base_path, initial_index);
    // Check the parent outside the collision loop. An `AlreadyExists` error
    // from directory creation is not evidence that a segment index is taken.
    ensure_segment_parent(&initial_path)?;

    let mut index = initial_index;
    loop {
        let path = segment_path(base_path, index);
        // `create_new` is atomic and fails for every existing filesystem
        // object, including symlinks. Keeping this handle for the segment's
        // lifetime also prevents later flushes from following a replacement
        // symlink.
        match std::fs::OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(&path)
        {
            Ok(file) => return Ok((file, index, path)),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                index = index.checked_add(1).ok_or_else(|| {
                    anyhow!(
                        "no available gzip jsonl segment index for {}",
                        base_path.display()
                    )
                })?;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("creating gzip jsonl segment {}", path.display()));
            }
        }
    }
}

fn write_gzip_member(file: &mut File, path: &Path, batch: Vec<u8>) -> anyhow::Result<()> {
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
    match matching_segments(base_path)?.occupied.last() {
        Some((index, _)) => index.checked_add(1).ok_or_else(|| {
            anyhow!(
                "no available gzip jsonl segment index for {}",
                base_path.display()
            )
        }),
        None => Ok(0),
    }
}

fn prune_segments(base_path: &Path, max_segments: usize) -> anyhow::Result<()> {
    prune_segments_with(base_path, max_segments, |path| {
        std::fs::remove_file(path)
            .with_context(|| format!("pruning gzip jsonl segment {}", path.display()))
    })
}

fn prune_segments_with(
    base_path: &Path,
    max_segments: usize,
    mut remove: impl FnMut(&Path) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let segments = matching_segments(base_path)?.prunable;
    let remove_count = segments.len().saturating_sub(max_segments);
    for (_, path) in segments.into_iter().take(remove_count) {
        remove(&path)?;
    }
    Ok(())
}

struct MatchingSegments {
    occupied: Vec<(u64, PathBuf)>,
    prunable: Vec<(u64, PathBuf)>,
}

fn matching_segments(base_path: &Path) -> anyhow::Result<MatchingSegments> {
    let first_segment = segment_path(base_path, 0);
    let parent = first_segment
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if !parent.try_exists()? {
        return Ok(MatchingSegments {
            occupied: Vec::new(),
            prunable: Vec::new(),
        });
    }

    let mut occupied = Vec::new();
    let mut prunable = Vec::new();
    for entry in std::fs::read_dir(parent)
        .with_context(|| format!("listing gzip jsonl directory {}", parent.display()))?
    {
        let entry = entry.with_context(|| {
            format!("reading gzip jsonl directory entry in {}", parent.display())
        })?;
        let Some(index) = segment_index(base_path, &entry.path()) else {
            continue;
        };
        let path = entry.path();
        let file_type = entry
            .file_type()
            .with_context(|| format!("reading file type for {}", path.display()))?;
        occupied.push((index, path.clone()));
        if file_type.is_file() {
            prunable.push((index, path));
        }
    }
    occupied.sort_unstable_by_key(|(index, _)| *index);
    prunable.sort_unstable_by_key(|(index, _)| *index);
    Ok(MatchingSegments { occupied, prunable })
}

fn segment_index(base_path: &Path, candidate: &Path) -> Option<u64> {
    let name = candidate.file_name()?.to_str()?;
    let without_suffix = name.strip_suffix(".jsonl.gz")?;
    let (_, index_text) = without_suffix.rsplit_once('.')?;
    if index_text.len() < 6 || !index_text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let index = index_text.parse::<u64>().ok()?;
    (segment_path(base_path, index).file_name()? == candidate.file_name()?).then_some(index)
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

    #[test]
    fn byte_retention_counts_active_bytes_and_removes_oldest_closed_segments() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace");
        for (index, size) in [4, 6, 7, 10].into_iter().enumerate() {
            std::fs::write(segment_path(&path, index as u64), vec![0; size]).unwrap();
        }
        let unrelated = dir.path().join("trace_other.000000.jsonl.gz");
        let backup = dir.path().join("trace.000000.jsonl.gz.bak");
        std::fs::write(&unrelated, [0; 100]).unwrap();
        std::fs::write(&backup, [0; 100]).unwrap();
        std::fs::create_dir(segment_path(&path, 4)).unwrap();

        prune_retained_segments(&path, Some(3), Some(20), None, SystemTime::now()).unwrap();
        assert!(!segment_path(&path, 0).exists());
        assert!(!segment_path(&path, 1).exists());
        assert!(segment_path(&path, 2).exists());
        assert!(segment_path(&path, 3).exists());
        assert!(segment_path(&path, 4).is_dir());
        assert!(unrelated.exists());
        assert!(backup.exists());

        // A single active segment may exceed the budget; it is never unlinked.
        prune_retained_segments(&path, Some(3), Some(1), None, SystemTime::now()).unwrap();
        assert!(!segment_path(&path, 2).exists());
        assert_eq!(std::fs::metadata(segment_path(&path, 3)).unwrap().len(), 10);
    }

    fn set_modified(path: &Path, time: SystemTime) {
        File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(time))
            .unwrap();
    }

    #[test]
    fn age_and_byte_retention_apply_together_using_modification_time() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace");
        let now = SystemTime::now();
        for index in 0..4 {
            let segment = segment_path(&path, index);
            std::fs::write(&segment, [0; 10]).unwrap();
            set_modified(&segment, now);
        }
        // Expire the newer index first, then enforce the byte limit by index.
        set_modified(&segment_path(&path, 2), now - Duration::from_secs(100));
        prune_retained_segments(&path, Some(3), Some(20), Some(Duration::from_secs(60)), now)
            .unwrap();
        assert!(!segment_path(&path, 0).exists());
        assert!(segment_path(&path, 1).exists());
        assert!(!segment_path(&path, 2).exists());
        assert!(segment_path(&path, 3).exists());
    }

    #[tokio::test]
    async fn idle_segment_expires_and_record_clock_survives_rotation() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace");
        let mut writer = GzipBatchWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                max_age: Some(Duration::from_secs(1)),
                ..Default::default()
            },
        )
        .unwrap();
        writer.start_time = Instant::now() - Duration::from_secs(10);
        let record = TestRecord {
            id: 1,
            name: "before".into(),
        };
        writer.push(&record).await.unwrap();
        assert!(segment_path(&path, 0).exists());
        writer
            .maintain_retention(SystemTime::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert!(!segment_path(&path, 0).exists());
        assert!(writer.active_file.is_none());

        writer
            .push(&TestRecord {
                id: 2,
                name: "after".into(),
            })
            .await
            .unwrap();
        let row: serde_json::Value =
            serde_json::from_str(&read_gzip_jsonl(&segment_path(&path, 1))).unwrap();
        assert_eq!(row["event"]["id"], 2);
        assert!(row["timestamp"].as_u64().unwrap() >= 10_000);
    }

    #[tokio::test]
    async fn restart_cleans_expired_segments_without_new_records() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace");
        let old = segment_path(&path, 7);
        std::fs::write(&old, b"old segment").unwrap();
        set_modified(&old, SystemTime::now() - Duration::from_secs(100));
        let writer = JsonlGzipWriter::<TestRecord>::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                max_age: Some(Duration::from_secs(60)),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while old.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("retention must run without incoming records");
        writer.close().await.unwrap();
    }

    #[test]
    fn retention_requires_exclusive_prefix_ownership() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace");
        let retained = JsonlGzipSinkOptions {
            max_bytes: Some(100),
            ..Default::default()
        };
        let first =
            GzipBatchWriter::<TestRecord>::new(path.display().to_string(), retained.clone())
                .unwrap();
        for options in [JsonlGzipSinkOptions::default(), retained.clone()] {
            assert!(GzipBatchWriter::<TestRecord>::new(
                format!("{}.jsonl.gz", path.display()), options,
            ).is_err());
        }
        drop(first);
        let shared = GzipBatchWriter::<TestRecord>::new(
            path.display().to_string(),
            JsonlGzipSinkOptions::default(),
        )
        .unwrap();
        assert!(
            GzipBatchWriter::<TestRecord>::new(path.display().to_string(), retained.clone())
                .is_err()
        );
        drop(shared);
        assert!(GzipBatchWriter::<TestRecord>::new(path.display().to_string(), retained).is_ok());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn readonly_existing_lock_allows_recording_and_preserves_exclusion() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempdir().unwrap();
        let path = dir.path().join("trace");
        let lock_path = segment_path(&path, 0).with_extension("lock");
        let exclusive_owner = File::options()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&lock_path)
            .unwrap();
        std::fs::set_permissions(&lock_path, std::fs::Permissions::from_mode(0o444)).unwrap();

        let mut shared = lock_prefix(&path, false).unwrap();
        // Check the descriptor too, so this regression test still detects an
        // unnecessary write-capable open when run as root.
        assert!(shared.write_all(b"must not write").is_err());
        drop(shared);

        let writer =
            JsonlGzipWriter::new(path.display().to_string(), JsonlGzipSinkOptions::default())
                .await
                .unwrap();
        assert!(matches!(
            exclusive_owner.try_lock(),
            Err(std::fs::TryLockError::WouldBlock)
        ));
        writer
            .send(TestRecord {
                id: 1,
                name: "recorded".into(),
            })
            .await
            .unwrap();
        writer.close().await.unwrap();
        let row: serde_json::Value =
            serde_json::from_str(&read_gzip_jsonl(&segment_path(&path, 0))).unwrap();
        assert_eq!(row["event"]["id"], 1);

        exclusive_owner.try_lock().unwrap();
        assert!(lock_prefix(&path, false).is_err());
        assert!(std::fs::read(&lock_path).unwrap().is_empty());
    }

    #[test]
    fn only_unsupported_shared_lock_errors_are_tolerated() {
        use std::fs::TryLockError;
        use std::io::{Error, ErrorKind};

        for exclusive in [false, true] {
            assert!(check_prefix_lock_result(Ok(()), exclusive).is_ok());
            assert_eq!(
                check_prefix_lock_result(
                    Err(TryLockError::Error(ErrorKind::Unsupported.into())),
                    exclusive
                )
                .is_ok(),
                !exclusive,
            );
            assert!(matches!(
                check_prefix_lock_result(Err(TryLockError::WouldBlock), exclusive),
                Err(TryLockError::WouldBlock),
            ));
            for kind in [ErrorKind::PermissionDenied, ErrorKind::Other] {
                let result = check_prefix_lock_result(
                    Err(TryLockError::Error(Error::from(kind))),
                    exclusive,
                );
                assert!(matches!(result, Err(TryLockError::Error(error)) if error.kind() == kind));
            }
        }
        #[cfg(unix)]
        {
            // ENOLCK is resource exhaustion, not evidence that locking is unsupported.
            let result = check_prefix_lock_result(
                Err(TryLockError::Error(Error::from_raw_os_error(libc::ENOLCK))),
                false,
            );
            assert!(
                matches!(result, Err(TryLockError::Error(error)) if error.raw_os_error() == Some(libc::ENOLCK))
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn byte_retention_does_not_follow_symlinks() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("trace");
        let target = dir.path().join("unrelated");
        std::fs::write(&target, b"keep").unwrap();
        std::os::unix::fs::symlink(&target, segment_path(&path, 0)).unwrap();
        std::fs::write(segment_path(&path, 1), b"remove").unwrap();
        prune_retained_segments(&path, None, Some(1), None, SystemTime::now()).unwrap();
        assert_eq!(std::fs::read(target).unwrap(), b"keep");
        assert!(
            std::fs::symlink_metadata(segment_path(&path, 0))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(!segment_path(&path, 1).exists());
    }

    #[tokio::test]
    async fn shutdown_drains_reserved_records_after_cancelled_await() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("shutdown_reserved");
        let mut writer = JsonlGzipWriter::<TestRecord>::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                flush_interval: Duration::from_secs(60),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        let sender = writer.sender().unwrap();
        let permit = sender.clone().reserve_owned().await.unwrap();
        sender
            .send(TestRecord {
                id: 1,
                name: "queued".into(),
            })
            .await
            .unwrap();

        {
            let shutdown = writer.shutdown();
            tokio::pin!(shutdown);
            tokio::select! {
                result = &mut shutdown => panic!("shutdown abandoned an outstanding permit: {result:?}"),
                _ = tokio::time::timeout(Duration::from_secs(5), sender.closed()) => {
                    assert!(sender.is_closed(), "shutdown must close admission");
                }
            }
            assert!(futures::poll!(&mut shutdown).is_pending());
        }
        assert!(
            sender
                .send(TestRecord {
                    id: 3,
                    name: "rejected".into(),
                })
                .await
                .is_err()
        );

        let shutdown = writer.shutdown();
        tokio::pin!(shutdown);
        assert!(futures::poll!(&mut shutdown).is_pending());
        permit.send(TestRecord {
            id: 2,
            name: "reserved".into(),
        });
        tokio::time::timeout(Duration::from_secs(5), shutdown)
            .await
            .unwrap()
            .unwrap();

        let content = read_gzip_jsonl(&segment_path(&path, 0));
        let ids: Vec<u64> = content
            .lines()
            .map(|line| {
                serde_json::from_str::<serde_json::Value>(line).unwrap()["event"]["id"]
                    .as_u64()
                    .unwrap()
            })
            .collect();
        assert_eq!(ids, [1, 2]);
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
                max_segments: None,
                ..Default::default()
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
        writer.close().await.expect("writer should close cleanly");

        let segment = segment_path(&path, 0);
        let content = read_gzip_jsonl(&segment);
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
                max_segments: None,
                ..Default::default()
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
        writer.close().await.expect("writer should close cleanly");

        let first = segment_path(&path, 0);
        let second = segment_path(&path, 1);
        let first_content = read_gzip_jsonl(&first);
        let second_content = read_gzip_jsonl(&second);

        assert!(first_content.contains("\"name\":\"first\""));
        assert!(second_content.contains("\"name\":\"second\""));
    }

    #[tokio::test]
    async fn restart_uses_one_more_than_highest_existing_index() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        std::fs::write(segment_path(&path, 0), b"old zero").unwrap();
        std::fs::write(segment_path(&path, 2), b"old two").unwrap();

        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: None,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        writer
            .send(TestRecord {
                id: 3,
                name: "after restart".to_string(),
            })
            .await
            .unwrap();
        writer.close().await.unwrap();

        assert!(!segment_path(&path, 1).exists());
        assert!(segment_path(&path, 0).exists());
        assert!(segment_path(&path, 2).exists());
        assert!(segment_path(&path, 3).exists());
        assert!(read_gzip_jsonl(&segment_path(&path, 3)).contains("after restart"));
    }

    #[tokio::test]
    async fn overlapping_writers_claim_distinct_segments_on_first_flush() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let options = JsonlGzipSinkOptions {
            buffer_bytes: 1024,
            flush_interval: Duration::from_secs(60),
            roll_uncompressed_bytes: 1024 * 1024,
            roll_lines: None,
            max_segments: None,
            ..Default::default()
        };
        let first = JsonlGzipWriter::<TestRecord>::new(path.display().to_string(), options.clone())
            .await
            .unwrap();
        let second = JsonlGzipWriter::<TestRecord>::new(path.display().to_string(), options)
            .await
            .unwrap();

        first
            .send(TestRecord {
                id: 1,
                name: "first writer".to_string(),
            })
            .await
            .unwrap();
        first.close().await.unwrap();
        second
            .send(TestRecord {
                id: 2,
                name: "second writer".to_string(),
            })
            .await
            .unwrap();
        second.close().await.unwrap();

        assert!(read_gzip_jsonl(&segment_path(&path, 0)).contains("first writer"));
        assert!(read_gzip_jsonl(&segment_path(&path, 1)).contains("second writer"));
    }

    #[tokio::test]
    async fn retention_prunes_only_exact_prefix_after_new_segment_is_written() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let unrelated = dir.path().join("test_trace_other.000000.jsonl.gz");
        let backup = dir.path().join("test_trace.000000.jsonl.gz.bak");
        std::fs::write(segment_path(&path, 0), b"old zero").unwrap();
        std::fs::write(segment_path(&path, 1), b"old one").unwrap();
        std::fs::write(&unrelated, b"unrelated").unwrap();
        std::fs::write(&backup, b"backup").unwrap();

        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: Some(2),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        writer
            .send(TestRecord {
                id: 2,
                name: "replacement".to_string(),
            })
            .await
            .unwrap();
        writer.close().await.unwrap();

        assert!(!segment_path(&path, 0).exists());
        assert!(segment_path(&path, 1).exists());
        assert!(segment_path(&path, 2).exists());
        assert!(unrelated.exists());
        assert!(backup.exists());
    }

    #[tokio::test]
    async fn repeated_flush_of_active_segment_does_not_rescan_retention() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let oldest = segment_path(&path, 0);
        std::fs::write(&oldest, b"old zero").unwrap();
        std::fs::write(segment_path(&path, 1), b"old one").unwrap();

        let mut writer = GzipBatchWriter::<TestRecord>::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        writer
            .push(&TestRecord {
                id: 2,
                name: "first flush".to_string(),
            })
            .await
            .unwrap();
        writer.flush_batch().await.unwrap();

        assert!(!oldest.exists());
        std::fs::write(&oldest, b"created after the segment was opened").unwrap();

        writer
            .push(&TestRecord {
                id: 3,
                name: "second flush".to_string(),
            })
            .await
            .unwrap();
        writer.flush_batch().await.unwrap();

        assert!(oldest.exists());
        let content = read_gzip_jsonl(&segment_path(&path, 2));
        assert!(content.contains("first flush"));
        assert!(content.contains("second flush"));
    }

    #[tokio::test]
    async fn retention_of_one_keeps_only_active_segment() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: Some(1),
                max_segments: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        for id in 0..3 {
            writer
                .send(TestRecord {
                    id,
                    name: format!("record {id}"),
                })
                .await
                .unwrap();
        }
        writer.close().await.unwrap();

        assert!(!segment_path(&path, 0).exists());
        assert!(!segment_path(&path, 1).exists());
        assert!(segment_path(&path, 2).exists());
        assert!(read_gzip_jsonl(&segment_path(&path, 2)).contains("record 2"));
    }

    #[tokio::test]
    async fn retention_keeps_four_newest_segments() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: Some(1),
                max_segments: Some(4),
                ..Default::default()
            },
        )
        .await
        .unwrap();

        for id in 0..6 {
            writer
                .send(TestRecord {
                    id,
                    name: format!("record {id}"),
                })
                .await
                .unwrap();
        }
        writer.close().await.unwrap();

        assert!(!segment_path(&path, 0).exists());
        assert!(!segment_path(&path, 1).exists());
        for index in 2..6 {
            assert!(segment_path(&path, index).exists());
        }
    }

    #[tokio::test]
    async fn oversized_record_occupies_a_segment_by_itself() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024 * 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 32,
                roll_lines: None,
                max_segments: None,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        writer
            .send(TestRecord {
                id: 0,
                name: "x".repeat(1024),
            })
            .await
            .unwrap();
        writer
            .send(TestRecord {
                id: 1,
                name: "next".to_string(),
            })
            .await
            .unwrap();
        writer.close().await.unwrap();

        let first = read_gzip_jsonl(&segment_path(&path, 0));
        let second = read_gzip_jsonl(&segment_path(&path, 1));
        assert_eq!(first.matches("\"name\":").count(), 1);
        assert!(first.contains(&"x".repeat(1024)));
        assert_eq!(second.matches("\"name\":").count(), 1);
        assert!(second.contains("\"name\":\"next\""));
    }

    #[tokio::test]
    async fn occupied_directory_uses_next_index_and_is_not_pruned() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let old = segment_path(&path, 0);
        let occupied = segment_path(&path, 1);
        std::fs::write(&old, b"old").unwrap();
        std::fs::create_dir(&occupied).unwrap();

        let writer: JsonlGzipWriter<TestRecord> = JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();
        writer
            .send(TestRecord {
                id: 1,
                name: "cannot write".to_string(),
            })
            .await
            .unwrap();
        writer.close().await.unwrap();

        assert!(!old.exists());
        assert!(occupied.is_dir());
        assert!(read_gzip_jsonl(&segment_path(&path, 2)).contains("cannot write"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn segment_creation_skips_symlink_created_after_allocation() {
        use std::os::unix::fs::symlink;

        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let target = dir.path().join("target");
        let original = b"must remain unchanged";
        std::fs::write(&target, original).unwrap();

        let mut writer = GzipBatchWriter::<TestRecord>::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1024,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        symlink(&target, segment_path(&path, 0)).unwrap();

        writer
            .push(&TestRecord {
                id: 1,
                name: "must not reach target".to_string(),
            })
            .await
            .unwrap();
        writer.flush_batch().await.unwrap();

        assert_eq!(std::fs::read(&target).unwrap(), original);
        assert!(
            std::fs::symlink_metadata(segment_path(&path, 0))
                .unwrap()
                .file_type()
                .is_symlink()
        );
        assert!(read_gzip_jsonl(&segment_path(&path, 1)).contains("must not reach target"));
    }

    #[test]
    fn rolls_only_after_uncompressed_limit_would_be_exceeded() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let mut writer = GzipBatchWriter::<TestRecord>::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                roll_uncompressed_bytes: 100,
                ..Default::default()
            },
        )
        .unwrap();
        writer.segment_lines = 1;
        writer.segment_uncompressed_bytes = 50;

        assert!(!writer.should_roll_before(50));
        assert!(writer.should_roll_before(51));
    }

    #[test]
    fn prune_failure_leaves_new_segment_and_returns_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        for index in 0..3 {
            std::fs::write(segment_path(&path, index), format!("segment {index}")).unwrap();
        }

        let mut attempted = Vec::new();
        let result = prune_segments_with(&path, 2, |candidate| {
            attempted.push(candidate.to_path_buf());
            Err(anyhow!("injected prune failure"))
        });

        assert!(result.is_err());
        assert_eq!(attempted, vec![segment_path(&path, 0)]);
        assert!(segment_path(&path, 0).exists());
        assert!(segment_path(&path, 2).exists());
    }

    #[tokio::test]
    async fn rejects_zero_segment_retention() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_trace");
        let result = JsonlGzipWriter::<TestRecord>::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                max_segments: Some(0),
                ..Default::default()
            },
        )
        .await;

        assert!(result.is_err());
    }
}
