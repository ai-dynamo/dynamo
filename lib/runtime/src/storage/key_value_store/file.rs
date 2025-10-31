// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::OsString;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::{collections::HashMap, pin::Pin};

use anyhow::Context as _;
use async_trait::async_trait;
use futures::StreamExt;
use inotify::{Event, EventMask, EventStream, Inotify, WatchMask};

use crate::storage::key_value_store::KeyValue;

use super::{Key, KeyValueBucket, KeyValueStore, StoreError, StoreOutcome, WatchEvent};

// Number of bytes of a buffer big enough to hold the biggest inotify event.
// 1024 is what the inotify crate's tests use.
const EVENT_SIZE: usize = 1024;

#[derive(Clone)]
pub struct FileStore {
    connection_id: u64,
}

impl Default for FileStore {
    fn default() -> Self {
        Self::new()
    }
}

impl FileStore {
    pub fn new() -> Self {
        FileStore {
            connection_id: rand::random::<u64>(),
        }
    }
}

#[async_trait]
impl KeyValueStore for FileStore {
    type Bucket = Directory;

    /// A "bucket" is a directory
    async fn get_or_create_bucket(
        &self,
        bucket_name: &str,
        _ttl: Option<Duration>, // TODO ttl not used yet
    ) -> Result<Self::Bucket, StoreError> {
        let p = PathBuf::from(bucket_name);
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
        Ok(Directory::new(p))
    }

    /// A "bucket" is a directory
    async fn get_bucket(&self, bucket_name: &str) -> Result<Option<Self::Bucket>, StoreError> {
        let p = PathBuf::from(bucket_name);
        if !p.exists() {
            return Err(StoreError::MissingBucket(bucket_name.to_string()));
        }
        if !p.is_dir() {
            return Err(StoreError::FilesystemError(
                "Bucket name is not a directory".to_string(),
            ));
        }
        Ok(Some(Directory::new(p)))
    }

    fn connection_id(&self) -> u64 {
        self.connection_id
    }
}

#[derive(Clone)]
pub struct Directory {
    p: PathBuf,
    event_buffer: Vec<u8>,
}

impl Directory {
    pub fn new(p: PathBuf) -> Self {
        Directory {
            p,
            event_buffer: Vec::with_capacity(EVENT_SIZE),
        }
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
        fs::remove_file(&full_path)
            .context(str_path)
            .map_err(a_to_fs_err)
    }

    async fn watch(
        &self,
    ) -> Result<Pin<Box<dyn futures::Stream<Item = WatchEvent> + Send + 'life0>>, StoreError> {
        let inotify = Inotify::init().map_err(to_fs_err)?;
        inotify
            .watches()
            .add(
                &self.p,
                WatchMask::MODIFY | WatchMask::CREATE | WatchMask::DELETE,
            )
            .map_err(to_fs_err)?;

        let dir = self.p.clone();
        Ok(Box::pin(async_stream::stream! {
            let mut buffer = [0; 1024];
            let mut events = match inotify.into_event_stream(&mut buffer) {
                Ok(events) => events,
                Err(err) => {
                    tracing::error!(error = %err, "Failed getting event stream from inotify");
                    return;
                }
            };
            while let Some(Ok(event)) = events.next().await {
                let Some(name) = event.name else {
                    tracing::warn!("Unexpected event on the directory itself");
                    return;
                };
                let item_path = dir.join(name);
                let data: bytes::Bytes = match fs::read(&item_path) {
                    Ok(data) => data.into(),
                    Err(err) => {
                        tracing::warn!(error = %err, item = %item_path.display(), "Failed reading event item. Skipping.");
                        return;
                    }
                };
                let item = KeyValue::new(item_path.display().to_string(), data);
                match event.mask {
                    EventMask::MODIFY | EventMask::CREATE => {
                        yield WatchEvent::Put(item);
                    }
                    EventMask::DELETE => {
                        yield WatchEvent::Delete(item);
                    }
                    event_type => {
                        tracing::warn!(?event_type, dir = %dir.display(), "Unexpected event type");
                        continue;
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
            let key = entry.file_name().to_string_lossy().to_string();
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
