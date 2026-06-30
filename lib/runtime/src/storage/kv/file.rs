// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cmp;
use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
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
use notify::{Config, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher, event};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::{Bucket, Key, KeyValue, Store, StoreError, StoreOutcome, WatchEvent};

/// How long until a key expires. We keep the keys alive by touching the files.
/// 10s is the same as our etcd lease expiry.
const DEFAULT_TTL: Duration = Duration::from_secs(10);

/// Don't do keep-alive any more often than this. Limits the disk write load.
const MIN_KEEP_ALIVE: Duration = Duration::from_secs(1);

/// Prefix for temporary files used in atomic writes.
/// Files with this prefix are ignored by the watcher.
const TEMP_FILE_PREFIX: &str = ".tmp_";
const TEMP_FILE_CREATE_ATTEMPTS: usize = 16;

/// Treat as a singleton
#[derive(Clone)]
pub struct FileStore {
    cancel_token: CancellationToken,
    root: PathBuf,
    connection_id: u64,
    /// Directories we may have created files in, for shutdown cleanup and keep-alive.
    /// Arc so that we only ever have one map here after clone.
    active_dirs: Arc<Mutex<HashMap<PathBuf, Directory>>>,
}

impl FileStore {
    pub(super) fn new<P: Into<PathBuf>>(cancel_token: CancellationToken, root_dir: P) -> Self {
        let fs = FileStore {
            cancel_token,
            root: root_dir.into(),
            connection_id: rand::random::<u64>(),
            active_dirs: Arc::new(Mutex::new(HashMap::new())),
        };
        let c = fs.clone();
        thread::spawn(move || c.expiry_thread());
        fs
    }

    /// Keep our files alive and delete expired keys.
    ///
    /// Does not return until cancellation token cancelled. On shutdown the process will
    /// often exit before we detect cancellation. That's fine.
    /// We run this in a real thread so it doesn't get delayed by tokio runtime under heavy load.
    fn expiry_thread(&self) {
        loop {
            let ttl = self.shortest_ttl();
            let keep_alive_interval = cmp::max(ttl / 3, MIN_KEEP_ALIVE);

            // Check before and after the sleep
            if self.cancel_token.is_cancelled() {
                break;
            }

            thread::sleep(keep_alive_interval);

            if self.cancel_token.is_cancelled() {
                break;
            }

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
impl Store for FileStore {
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
        // Keep watched paths and event paths in the same form across symlinked roots.
        let canonical_root = root.canonicalize().unwrap_or_else(|_| root.clone());
        let canonical_path = p.canonicalize().unwrap_or_else(|_| p.clone());
        if ttl < MIN_KEEP_ALIVE {
            let h_ttl = humantime::format_duration(ttl);
            tracing::warn!(path = %p.display(), ttl = %h_ttl, "ttl is too short, increasing to {}", humantime::format_duration(MIN_KEEP_ALIVE));
        }
        let ttl = cmp::max(ttl, MIN_KEEP_ALIVE);
        Directory {
            root: canonical_root,
            p: canonical_path,
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

    fn write_temp_file(&self, value: &[u8]) -> Result<PathBuf, StoreError> {
        for _ in 0..TEMP_FILE_CREATE_ATTEMPTS {
            let temp_name = format!("{TEMP_FILE_PREFIX}{:016x}", rand::random::<u64>());
            let temp_path = self.p.join(&temp_name);
            if write_temp_file_at(&temp_path, value)? {
                return Ok(temp_path);
            }
        }

        Err(StoreError::FilesystemError(format!(
            "failed to create unique FileStore temp file in {} after {TEMP_FILE_CREATE_ATTEMPTS} attempts",
            self.p.display()
        )))
    }
}

impl fmt::Display for Directory {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.p.display())
    }
}

#[async_trait]
impl Bucket for Directory {
    /// Write a file to the directory by publishing a completed temp file.
    /// This ensures watchers never see a partially written file.
    /// Revision-zero inserts provide create-if-absent publication for this
    /// FileStore path, but not leases, fencing, crash durability, or strict
    /// runtime-wide cardinality guarantees.
    async fn insert(
        &self,
        key: &Key,
        value: bytes::Bytes,
        revision: u64,
    ) -> Result<StoreOutcome, StoreError> {
        let safe_key = key.url_safe();
        let full_path = self.p.join(safe_key.as_ref());
        let str_path = full_path.display().to_string();

        let temp_path = self.write_temp_file(&value)?;

        if revision == 0 {
            // No-clobber publish for revision-zero inserts: the link fails if another
            // writer already created the key, and readers never see a partial target file.
            match fs::hard_link(&temp_path, &full_path) {
                Ok(()) => {
                    if let Err(err) = fs::remove_file(&temp_path) {
                        tracing::warn!(
                            path = %temp_path.display(),
                            error = %err,
                            "Failed to remove FileStore temp file after create-if-absent publish"
                        );
                    }
                    self.owned_files.lock().insert(full_path.clone());
                    return Ok(StoreOutcome::Created(0));
                }
                Err(err) if err.kind() == ErrorKind::AlreadyExists => {
                    if let Err(remove_err) = fs::remove_file(&temp_path) {
                        tracing::warn!(
                            path = %temp_path.display(),
                            error = %remove_err,
                            "Failed to remove unused FileStore temp file after create-if-absent conflict"
                        );
                    }
                    return Ok(StoreOutcome::Exists(0));
                }
                Err(err) => {
                    if let Err(remove_err) = fs::remove_file(&temp_path) {
                        tracing::warn!(
                            path = %temp_path.display(),
                            error = %remove_err,
                            "Failed to remove unused FileStore temp file after create-if-absent error"
                        );
                    }
                    return Err(to_fs_err(err));
                }
            }
        }

        // Atomic rename to target path
        fs::rename(&temp_path, &full_path)
            .with_context(|| format!("renaming {} to {}", temp_path.display(), str_path))
            .map_err(a_to_fs_err)?;

        self.owned_files.lock().insert(full_path.clone());
        Ok(StoreOutcome::Created(revision))
    }

    /// Read a file from the directory
    async fn get(&self, key: &Key) -> Result<Option<bytes::Bytes>, StoreError> {
        let safe_key = key.url_safe();
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
        let safe_key = key.url_safe();
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

                    let canonical_item_path = canonicalize_event_path(&item_path);

                    let key = match canonical_item_path.strip_prefix(&root) {
                        Ok(stripped) => Key::from_url_safe(&stripped.display().to_string()),
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

                    // Skip temp files used for atomic writes
                    if item_path.file_name()
                        .map(|n| n.to_string_lossy().starts_with(TEMP_FILE_PREFIX))
                        .unwrap_or(false)
                    {
                        continue;
                    }

                    match event.kind {
                        // Handle file creation, modification, and rename-to (from atomic writes)
                        EventKind::Create(event::CreateKind::File)
                        | EventKind::Modify(event::ModifyKind::Data(event::DataChange::Content))
                        | EventKind::Modify(event::ModifyKind::Name(event::RenameMode::To)) => {
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
                            yield WatchEvent::Delete(key);
                        }
                        _ => {
                            // These happen every time the keep-alive updates last modified time
                            continue;
                        }
                    }
                }
            }
        }))
    }

    async fn entries(&self) -> Result<HashMap<Key, bytes::Bytes>, StoreError> {
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

            // Skip temp files used for atomic writes
            if entry
                .file_name()
                .to_string_lossy()
                .starts_with(TEMP_FILE_PREFIX)
            {
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
                Ok(p) => Key::from_url_safe(&p.to_string_lossy()),
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

fn write_temp_file_at(temp_path: &Path, value: &[u8]) -> Result<bool, StoreError> {
    let mut file = match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp_path)
    {
        Ok(file) => file,
        Err(err) if err.kind() == ErrorKind::AlreadyExists => return Ok(false),
        Err(err) => {
            let err = anyhow::Error::new(err)
                .context(format!("creating temp file {}", temp_path.display()));
            return Err(a_to_fs_err(err));
        }
    };

    if let Err(err) = file.write_all(value) {
        if let Err(remove_err) = fs::remove_file(temp_path) {
            tracing::warn!(
                path = %temp_path.display(),
                error = %remove_err,
                "Failed to remove FileStore temp file after write error"
            );
        }
        let err =
            anyhow::Error::new(err).context(format!("writing temp file {}", temp_path.display()));
        return Err(a_to_fs_err(err));
    }

    Ok(true)
}

fn canonicalize_event_path(path: &Path) -> PathBuf {
    if let Ok(canonical_path) = path.canonicalize() {
        return canonical_path;
    }
    let (Some(parent), Some(file_name)) = (path.parent(), path.file_name()) else {
        return path.to_path_buf();
    };
    let Ok(canonical_parent) = parent.canonicalize() else {
        return path.to_path_buf();
    };
    canonical_parent.join(file_name)
}

// For anyhow preserve the context
fn a_to_fs_err(err: anyhow::Error) -> StoreError {
    StoreError::FilesystemError(format!("{err:#}"))
}

fn to_fs_err<E: std::error::Error>(err: E) -> StoreError {
    StoreError::FilesystemError(err.to_string())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::time::Duration;

    use futures::StreamExt;
    use tokio_util::sync::CancellationToken;

    use crate::storage::kv::{Bucket as _, FileStore, Key, Store as _, StoreOutcome};

    #[test]
    fn deleted_event_path_canonicalizes_existing_parent() {
        let t = tempfile::tempdir().unwrap();
        let canonical_root = t.path().join("canonical");
        let bucket = canonical_root.join("v1/claims");
        fs::create_dir_all(&bucket).unwrap();
        let linked_root = t.path().join("linked");
        symlink(&canonical_root, &linked_root).unwrap();

        assert_eq!(
            super::canonicalize_event_path(&linked_root.join("v1/claims/deleted")),
            canonical_root
                .canonicalize()
                .unwrap()
                .join("v1/claims/deleted")
        );
    }

    #[tokio::test]
    async fn external_delete_is_observed_under_noncanonical_root() {
        let t = tempfile::tempdir().unwrap();
        let watcher_cancel = CancellationToken::new();
        let creator_cancel = CancellationToken::new();
        let watcher_store = FileStore::new(watcher_cancel.clone(), t.path());
        let creator_store = FileStore::new(creator_cancel.clone(), t.path());
        let watcher_bucket = watcher_store
            .get_or_create_bucket("v1/claims", None)
            .await
            .unwrap();
        let creator_bucket = creator_store
            .get_or_create_bucket("v1/claims", None)
            .await
            .unwrap();
        let mut events = watcher_bucket.watch().await.unwrap();
        let key = Key::new("scope/session".to_string());

        creator_bucket
            .insert(&key, "value".into(), 0)
            .await
            .unwrap();
        loop {
            let event = tokio::time::timeout(Duration::from_secs(2), events.next())
                .await
                .expect("FileStore watcher did not observe claim creation")
                .expect("FileStore watcher ended after claim creation");
            if matches!(event, super::WatchEvent::Put(ref item) if item.key_str() == "v1/claims/scope/session")
            {
                break;
            }
        }

        creator_bucket.delete(&key).await.unwrap();
        let event = tokio::time::timeout(Duration::from_secs(2), events.next())
            .await
            .expect("FileStore watcher did not observe claim deletion")
            .expect("FileStore watcher ended after claim deletion");
        assert!(
            matches!(event, super::WatchEvent::Delete(ref deleted) if deleted == &Key::new("v1/claims/scope/session".to_string()))
        );

        watcher_cancel.cancel();
        creator_cancel.cancel();
    }

    #[tokio::test]
    async fn test_entries_full_path() {
        let t = tempfile::tempdir().unwrap();

        let cancel_token = CancellationToken::new();
        let m = FileStore::new(cancel_token.clone(), t.path());
        let bucket = m.get_or_create_bucket("v1/tests", None).await.unwrap();
        let _ = bucket
            .insert(&Key::new("key1/multi/part".to_string()), "value1".into(), 0)
            .await
            .unwrap();
        let _ = bucket
            .insert(&Key::new("key2".to_string()), "value2".into(), 0)
            .await
            .unwrap();
        let entries = bucket.entries().await.unwrap();
        let keys: HashSet<Key> = entries.into_keys().collect();
        cancel_token.cancel(); // stop the background thread

        assert!(keys.contains(&Key::new("v1/tests/key1/multi/part".to_string())));
        assert!(keys.contains(&Key::new("v1/tests/key2".to_string())));
    }

    #[test]
    fn test_temp_file_creation_does_not_overwrite_existing_path() {
        let t = tempfile::tempdir().unwrap();
        let temp_path = t.path().join(".tmp_existing");

        fs::write(&temp_path, b"sentinel").unwrap();
        let created = super::write_temp_file_at(&temp_path, b"new").unwrap();

        assert!(!created);
        assert_eq!(fs::read(&temp_path).unwrap(), b"sentinel");
    }

    #[tokio::test]
    async fn test_insert_revision_zero_is_create_if_absent() {
        let t = tempfile::tempdir().unwrap();

        let cancel_token = CancellationToken::new();
        let m = FileStore::new(cancel_token.clone(), t.path());
        let bucket = m.get_or_create_bucket("v1/tests", None).await.unwrap();
        let key = Key::new("singleton".to_string());

        let first = bucket.insert(&key, "winner".into(), 0).await.unwrap();
        let second = bucket.insert(&key, "loser".into(), 0).await.unwrap();
        let value = bucket.get(&key).await.unwrap().unwrap();
        cancel_token.cancel();

        assert_eq!(first, StoreOutcome::Created(0));
        assert_eq!(second, StoreOutcome::Exists(0));
        assert_eq!(value.as_ref(), b"winner");
    }

    #[tokio::test]
    async fn test_insert_nonzero_revision_overwrites() {
        let t = tempfile::tempdir().unwrap();

        let cancel_token = CancellationToken::new();
        let m = FileStore::new(cancel_token.clone(), t.path());
        let bucket = m.get_or_create_bucket("v1/tests", None).await.unwrap();
        let key = Key::new("existing".to_string());

        bucket.insert(&key, "old".into(), 0).await.unwrap();
        let outcome = bucket.insert(&key, "new".into(), 1).await.unwrap();
        let value = bucket.get(&key).await.unwrap().unwrap();
        cancel_token.cancel();

        assert_eq!(outcome, StoreOutcome::Created(1));
        assert_eq!(value.as_ref(), b"new");
    }

    #[tokio::test]
    async fn test_concurrent_insert_revision_zero_has_one_winner() {
        let t = tempfile::tempdir().unwrap();
        let root = t.path().to_path_buf();
        let key = Key::new("singleton".to_string());

        let mut tasks = Vec::new();
        for index in 0..16 {
            let root = root.clone();
            let key = key.clone();
            tasks.push(tokio::spawn(async move {
                let cancel_token = CancellationToken::new();
                let store = FileStore::new(cancel_token.clone(), root);
                let bucket = store.get_or_create_bucket("v1/claims", None).await.unwrap();
                let value = format!("value-{index}");
                let outcome = bucket.insert(&key, value.clone().into(), 0).await.unwrap();
                let stored = bucket.get(&key).await.unwrap().unwrap();
                cancel_token.cancel();
                (outcome, String::from_utf8(stored.to_vec()).unwrap(), value)
            }));
        }

        let mut created_values = Vec::new();
        let mut observed_values = HashSet::new();
        for task in tasks {
            let (outcome, stored, attempted) = task.await.unwrap();
            observed_values.insert(stored);
            if outcome == StoreOutcome::Created(0) {
                created_values.push(attempted);
            } else {
                assert_eq!(outcome, StoreOutcome::Exists(0));
            }
        }

        assert_eq!(created_values.len(), 1);
        assert_eq!(observed_values.len(), 1);
        assert_eq!(
            observed_values.into_iter().next().unwrap(),
            created_values.pop().unwrap()
        );
    }
}
