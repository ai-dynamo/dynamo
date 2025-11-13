// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp;
use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::fs::OpenOptions;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::time::SystemTime;
use std::{collections::HashMap, pin::Pin};

use anyhow::Context as _;
use async_trait::async_trait;
use futures::StreamExt;
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use parking_lot::Mutex;

use crate::storage::key_value_store::KeyValue;

use super::{Key, KeyValueBucket, KeyValueStore, StoreError, StoreOutcome, WatchEvent};

/// How long until a key expires. We keep the keys alive by touching the files.
/// 10s is the same as our etcd lease expiry.
const DEFAULT_TTL: Duration = Duration::from_secs(10);

/// Don't do keep-alive any more often than this. Limits the disk write load.
const MIN_KEEP_ALIVE: Duration = Duration::from_secs(1);

/// Treat as a singleton
#[derive(Clone)]
pub struct FileStore {
    root: PathBuf,
    connection_id: u64,
    /// Directories we may have created files in, for shutdown cleanup and keep-alive.
    /// Arc so that we only ever have one map here after clone.
    active_dirs: Arc<Mutex<HashMap<PathBuf, Directory>>>,
}

impl FileStore {
    pub(super) fn new<P: Into<PathBuf>>(root_dir: P) -> Self {
        let fs = FileStore {
            root: root_dir.into(),
            connection_id: rand::random::<u64>(),
            active_dirs: Arc::new(Mutex::new(HashMap::new())),
        };
        let c = fs.clone();
        thread::spawn(move || c.expiry_thread());
        fs
    }

    /// Keep our files alive and delete expired keys. Does not return.
    /// We run this in a real thread so it doesn't get delayed by tokio runtime load.
    /// It doesn't need any cleanup so we don't use cancellation token.
    fn expiry_thread(&self) -> ! {
        loop {
            let ttl = self.shortest_ttl();
            let keep_alive_interval = cmp::max(ttl / 3, MIN_KEEP_ALIVE);

            thread::sleep(keep_alive_interval);

            self.keep_alive();
            if let Err(err) = self.delete_expired_files() {
                tracing::error!(error = %err, "FileStore delete_expired_files");
            }
        }
    }

    /// The shortest TTL of any directory we are using.
    fn shortest_ttl(&self) -> Duration {
        let mut ttl = DEFAULT_TTL;
        let active_dirs = self.active_dirs.lock().clone();
        for (_, dir) in active_dirs {
            ttl = cmp::min(ttl, dir.ttl);
        }
        tracing::trace!("FileStore expiry shortest ttl {ttl:?}");
        ttl
    }

    fn keep_alive(&self) {
        let active_dirs = self.active_dirs.lock().clone();
        for (_, dir) in active_dirs {
            dir.keep_alive();
        }
    }

    fn delete_expired_files(&self) -> anyhow::Result<()> {
        let active_dirs = self.active_dirs.lock().clone();
        for (path, dir) in active_dirs {
            dir.delete_expired_files()
                .with_context(|| path.display().to_string())?;
        }
        Ok(())
    }
}

#[async_trait]
impl KeyValueStore for FileStore {
    type Bucket = Directory;

    /// A "bucket" is a directory
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        ttl: Option<Duration>,
    ) -> Result<Self::Bucket, StoreError> {
        let p = self.root.join(bucket_name);
        if let Some(dir) = self.active_dirs.lock().get(&p) {
            return Ok(dir.clone());
        };

        if p.exists() {
            // Get
            if !p.is_dir() {
                return Err(StoreError::FilesystemError(
                    "Bucket name is not a directory".to_string(),
                ));
            }
        } else {
            // Create
            fs::create_dir_all(&p).map_err(to_fs_err)?;
        }
        let dir = Directory::new(self.root.clone(), p.clone(), ttl.unwrap_or(DEFAULT_TTL));
        self.active_dirs.lock().insert(p, dir.clone());
        Ok(dir)
    }

    /// A "bucket" is a directory
    async fn get_bucket(&self, bucket_name: &str) -> Result<Option<Self::Bucket>, StoreError> {
        let p = self.root.join(bucket_name);
        if let Some(dir) = self.active_dirs.lock().get(&p) {
            return Ok(Some(dir.clone()));
        };

        if !p.exists() {
            return Ok(None);
        }
        if !p.is_dir() {
            return Err(StoreError::FilesystemError(
                "Bucket name is not a directory".to_string(),
            ));
        }
        // The filesystem itself doesn't store the TTL so for now default it
        let dir = Directory::new(self.root.clone(), p.clone(), DEFAULT_TTL);
        self.active_dirs.lock().insert(p, dir.clone());
        Ok(Some(dir))
    }

    fn connection_id(&self) -> u64 {
        self.connection_id
    }

    // This cannot be a Drop imp because DistributedRuntime is cloned various places including
    // Python. Drop doesn't get called.
    fn shutdown(&self) {
        for (_, mut dir) in self.active_dirs.lock().drain() {
            if let Err(err) = dir.delete_owned_files() {
                tracing::error!(error = %err, %dir, "Failed shutdown delete of owned files");
            }
        }
    }
}

#[derive(Clone)]
pub struct Directory {
    root: PathBuf,
    p: PathBuf,
    ttl: Duration,
    /// These are the files we created and hence must delete on shutdown
    owned_files: Arc<Mutex<HashSet<PathBuf>>>,
}

impl Directory {
    fn new(root: PathBuf, p: PathBuf, ttl: Duration) -> Self {
        // Canonicalize root to handle symlinks (e.g., /var -> /private/var on macOS)
        let canonical_root = root.canonicalize().unwrap_or_else(|_| root.clone());
        if ttl < MIN_KEEP_ALIVE {
            let h_ttl = humantime::format_duration(ttl);
            tracing::warn!(path = %p.display(), ttl = %h_ttl, "ttl is too short, increasing to {}", humantime::format_duration(MIN_KEEP_ALIVE));
        }
        let ttl = cmp::max(ttl, MIN_KEEP_ALIVE);
        Directory {
            root: canonical_root,
            p,
            ttl,
            owned_files: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// touch the files we own so they don't get deleted by a different FileStore
    fn keep_alive(&self) {
        let owned_files = self.owned_files.lock().clone();
        for path in owned_files {
            let file = match OpenOptions::new().write(true).open(&path) {
                Ok(f) => f,
                Err(err) => {
                    tracing::error!(path = %path.display(), error = %err, "FileStore::keep_alive failed opening owned file");
                    continue;
                }
            };
            if let Err(err) = file.set_modified(SystemTime::now()) {
                tracing::error!(path = %path.display(), error = %err, "FileStore::keep_alive failed set_modified on owned file");
                continue;
            }
            tracing::trace!("FileStore keep_alive set {}", path.display());
        }
    }

    /// Remove any files not touched for longer than TTL.
    /// This looks at all files in the directory to catch orphaned files from processes that didn't stop cleanly.
    /// Returns an error if we cannot open the directory. Errors inside the directory are logged
    /// but non-fatal.
    fn delete_expired_files(&self) -> anyhow::Result<()> {
        let deadline = SystemTime::now() - self.ttl;
        let dirname = self.p.display().to_string();
        for entry in fs::read_dir(&self.p).with_context(|| dirname.clone())? {
            let entry = match entry {
                Ok(p) => p,
                Err(err) => {
                    tracing::warn!(dir = dirname, error = %err, "File store could read directory contents");
                    continue;
                }
            };
            if !entry.file_type().map(|f| f.is_file()).unwrap_or(false) {
                tracing::warn!(dir = dirname, entry = %entry.path().display(), "File store directory should only contain files");
                continue;
            }
            let ctx = entry.path().display().to_string();
            let metadata = match entry.metadata() {
                Ok(m) => m,
                Err(err) => {
                    tracing::warn!(path = %ctx, error = %err, "Failed fetching metadata");
                    continue;
                }
            };
            let last_modified = match metadata.modified() {
                Ok(lm) => lm,
                Err(err) => {
                    // We should only get an error on platforms with no mtime, which we don't
                    // support anyway.
                    tracing::warn!(path = %ctx, error = %err, "Failed reading mtime");
                    continue;
                }
            };
            if last_modified < deadline {
                tracing::info!(path = ctx, ?last_modified, "Expired");
                if let Err(err) = fs::remove_file(entry.path()) {
                    tracing::warn!(path = %ctx, error = %err, "Failed removing");
                }
            }
        }
        Ok(())
    }

    fn delete_owned_files(&mut self) -> anyhow::Result<()> {
        let mut errs = Vec::new();
        for p in self.owned_files.lock().drain() {
            if let Err(err) = fs::remove_file(&p) {
                errs.push(format!("{}: {err}", p.display()));
            }
        }
        if !errs.is_empty() {
            anyhow::bail!(errs.join(", "));
        }
        Ok(())
    }
}

impl fmt::Display for Directory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.p.display())
    }
}

#[async_trait]
impl KeyValueBucket for Directory {
    /// Write a file to the directory
    async fn insert(
        &self,
        key: &Key,
        value: bytes::Bytes,
        _revision: u64, // Not used. Maybe put in file name?
    ) -> Result<StoreOutcome, StoreError> {
        let safe_key = Key::new(key.as_ref()); // because of from_raw
        let full_path = self.p.join(safe_key.as_ref());
        self.owned_files.lock().insert(full_path.clone());
        let str_path = full_path.display().to_string();
        fs::write(&full_path, &value)
            .context(str_path)
            .map_err(a_to_fs_err)?;
        Ok(StoreOutcome::Created(0))
    }

    /// Read a file from the directory
    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StoreError> {
        let safe_key = Key::new(key.as_ref()); // because of from_raw
        let full_path = self.p.join(safe_key.as_ref());
        if !full_path.exists() {
            return Ok(None);
        }
        let str_path = full_path.display().to_string();
        let data: bytes::Bytes = fs::read(&full_path)
            .context(str_path)
            .map_err(a_to_fs_err)?
            .into();
        Ok(Some(data))
    }

    /// Delete a file from the directory
    async fn delete(&self, key: &Key) -> Result<(), StoreError> {
        let safe_key = Key::new(key.as_ref()); // because of from_raw
        let full_path = self.p.join(safe_key.as_ref());
        let str_path = full_path.display().to_string();
        if !full_path.exists() {
            return Err(StoreError::MissingKey(str_path));
        }

        self.owned_files.lock().remove(&full_path);

        fs::remove_file(&full_path)
            .context(str_path)
            .map_err(a_to_fs_err)
    }

    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = WatchEvent> + Send + 'life0>>, StoreError> {
        let (tx, mut rx) = tokio::sync::mpsc::channel(128);

        let mut watcher = RecommendedWatcher::new(
            move |res: Result<Event, notify::Error>| {
                if let Err(err) = tx.blocking_send(res) {
                    tracing::error!(error = %err, "Failed to send file watch event");
                }
            },
            Config::default(),
        )
        .map_err(to_fs_err)?;

        watcher
            .watch(&self.p, RecursiveMode::NonRecursive)
            .map_err(to_fs_err)?;

        let dir = self.p.clone();
        let root = self.root.clone();

        Ok(Box::pin(async_stream::stream! {
            // Keep watcher alive for the duration of the stream
            let _watcher = watcher;

            while let Some(event_result) = rx.recv().await {
                let event = match event_result {
                    Ok(event) => event,
                    Err(err) => {
                        tracing::error!(error = %err, "Failed receiving file watch event");
                        continue;
                    }
                };

                for item_path in event.paths {
                    // Skip if the event is for the directory itself
                    if item_path == dir {
                        tracing::warn!("Unexpected event on the directory itself");
                        continue;
                    }

                    // Canonicalize paths to handle symlinks (e.g., /var -> /private/var on macOS)
                    let canonical_item_path = match item_path.canonicalize() {
                        Ok(p) => p,
                        Err(err) => {
                            tracing::warn!(error = %err, item = %item_path.display(), "Failed to canonicalize path. Using original path.");
                            item_path.clone()
                        }
                    };

                    let key = match canonical_item_path.strip_prefix(&root) {
                        Ok(stripped) => stripped.display().to_string().replace("_", "/"),
                        Err(err) => {
                            // Possibly this should be a panic.
                            // A key cannot be outside the file store root.
                            tracing::error!(
                                error = %err,
                                item_path = %canonical_item_path.display(),
                                root = %root.display(),
                                "Item in file store is not prefixed with file store root. Should be impossible. Ignoring invalid key.");
                            continue;
                        }
                    };

                    match event.kind {
                        EventKind::Create(_) | EventKind::Modify(_) => {
                            let data: bytes::Bytes = match fs::read(&item_path) {
                                Ok(data) => data.into(),
                                Err(err) => {
                                    tracing::warn!(error = %err, item = %item_path.display(), "Failed reading event item. Skipping.");
                                    continue;
                                }
                            };
                            let item = KeyValue::new(key, data);
                            yield WatchEvent::Put(item);
                        }
                        EventKind::Remove(_) => {
                            yield WatchEvent::Delete(Key::from_raw(key));
                        }
                        event_type => {
                            tracing::debug!(?event_type, dir = %dir.display(), "Ignoring event type");
                            continue;
                        }
                    }
                }
            }
        }))
    }

    async fn entries(&self) -> Result<HashMap<String, bytes::Bytes>, StoreError> {
        let contents = fs::read_dir(&self.p)
            .with_context(|| self.p.display().to_string())
            .map_err(a_to_fs_err)?;
        let mut out = HashMap::new();
        for entry in contents {
            let entry = entry.map_err(to_fs_err)?;
            if !entry.path().is_file() {
                tracing::warn!(
                    path = %entry.path().display(),
                    "Unexpected entry, directory should only contain files."
                );
                continue;
            }

            // Canonicalize paths to handle symlinks (e.g., /var -> /private/var on macOS)
            let canonical_entry_path = match entry.path().canonicalize() {
                Ok(p) => p,
                Err(err) => {
                    tracing::warn!(error = %err, path = %entry.path().display(), "Failed to canonicalize path. Using original path.");
                    entry.path()
                }
            };

            let key = match canonical_entry_path.strip_prefix(&self.root) {
                Ok(p) => p.to_string_lossy().to_string().replace("_", "/"),
                Err(err) => {
                    tracing::error!(
                        error = %err,
                        path = %canonical_entry_path.display(),
                        root = %self.root.display(),
                        "FileStore path not in root. Should be impossible. Skipping entry."
                    );
                    continue;
                }
            };
            let data: bytes::Bytes = fs::read(entry.path())
                .with_context(|| self.p.display().to_string())
                .map_err(a_to_fs_err)?
                .into();
            out.insert(key, data);
        }
        Ok(out)
    }
}

// For anyhow preserve the context
fn a_to_fs_err(err: anyhow::Error) -> StoreError {
    StoreError::FilesystemError(format!("{err:#}"))
}

fn to_fs_err<E: std::error::Error>(err: E) -> StoreError {
    StoreError::FilesystemError(err.to_string())
}
