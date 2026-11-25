// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Recovery rounds: request ids, and the record written before a round is
//! sent.
//!
//! vLLM never deletes a round's check-in keys, so a reused request id could
//! see stale check-ins. Ids are therefore `<world>-e<epoch>-r<seq>-<8 hex>`:
//! the epoch grows every time the controller starts, the sequence number
//! every round, and the random suffix guards against a lost state directory.

use std::fs::{self, File};
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::decision::StoreAddr;
use crate::protocol::Instruction;

/// What the controller is about to send, persisted before the first order
/// leaves. A restarted controller reads the last record back: its
/// `survivors` are the world's members (unless `escalated`), its `store` says
/// whether later rounds must carry a store address, and `next_store_port`
/// keeps store ports from repeating.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundRecord {
    pub world: String,
    pub epoch: u64,
    pub seq: u64,
    pub request_id: String,
    pub instruction: Instruction,
    pub params: Value,
    pub removed: Vec<u32>,
    pub survivors: Vec<u32>,
    pub store: Option<StoreAddr>,
    /// RFC 3339, UTC.
    pub sent_at: String,
    /// RFC 3339, UTC: the controller gives up on the round after this.
    pub not_after: String,
    /// Port of the next store rebuild after this round.
    #[serde(default)]
    pub next_store_port: u32,
    /// The controller escalated after this round was sent (in this incident
    /// or a later one), so the supervisor restarts the whole world and this
    /// round's survivors and store no longer describe it.
    #[serde(default)]
    pub escalated: bool,
}

/// A request id that no other round of any controller epoch uses.
pub fn request_id(world: &str, epoch: u64, seq: u64) -> String {
    format!("{world}-e{epoch}-r{seq}-{:08x}", rand::random::<u32>())
}

/// RFC 3339 UTC timestamp with millisecond precision.
pub fn timestamp(time: SystemTime) -> String {
    humantime::format_rfc3339_millis(time).to_string()
}

/// Durable controller state: the epoch and the last round.
pub trait RoundStore: Send + Sync {
    /// Starts a controller epoch: one greater than any epoch seen before.
    fn begin_epoch(&self) -> Result<u64>;
    /// Persists `record`. Called before any order of the round is sent; a
    /// round that cannot be recorded is not sent.
    fn save(&self, record: &RoundRecord) -> Result<()>;
    /// The last saved round.
    fn last(&self) -> Result<Option<RoundRecord>>;
}

/// [`RoundStore`] in a directory, one world per file pair:
/// `<world>.epoch.json` and `<world>.round.json`. Each file is replaced
/// atomically (temporary file, fsync, rename, fsync of the directory). The
/// directory is created on the first write, so reading (as `--dry-run` does)
/// changes nothing on disk.
#[derive(Clone, Debug)]
pub struct FileRoundStore {
    dir: PathBuf,
    world: String,
}

#[derive(Serialize, Deserialize)]
struct EpochFile {
    epoch: u64,
}

impl FileRoundStore {
    pub fn new(dir: impl Into<PathBuf>, world: &str) -> Self {
        Self {
            dir: dir.into(),
            world: world.to_string(),
        }
    }

    pub fn epoch_path(&self) -> PathBuf {
        self.dir.join(format!("{}.epoch.json", self.world))
    }

    pub fn round_path(&self) -> PathBuf {
        self.dir.join(format!("{}.round.json", self.world))
    }
}

impl RoundStore for FileRoundStore {
    fn begin_epoch(&self) -> Result<u64> {
        let from_epoch = read_json::<EpochFile>(&self.epoch_path())?.map_or(0, |file| file.epoch);
        let from_round = self.last()?.map_or(0, |record| record.epoch);
        let epoch = from_epoch
            .max(from_round)
            .checked_add(1)
            .with_context(|| format!("the epoch in {} cannot grow", self.dir.display()))?;
        write_json_atomic(&self.epoch_path(), &EpochFile { epoch })?;
        Ok(epoch)
    }

    fn save(&self, record: &RoundRecord) -> Result<()> {
        write_json_atomic(&self.round_path(), record)
    }

    fn last(&self) -> Result<Option<RoundRecord>> {
        read_json(&self.round_path())
    }
}

/// [`RoundStore`] in memory, for `--dry-run` and tests.
#[derive(Debug, Default)]
pub struct MemoryRoundStore {
    state: Mutex<(u64, Option<RoundRecord>)>,
}

impl RoundStore for MemoryRoundStore {
    fn begin_epoch(&self) -> Result<u64> {
        let mut state = self.lock()?;
        state.0 += 1;
        Ok(state.0)
    }

    fn save(&self, record: &RoundRecord) -> Result<()> {
        self.lock()?.1 = Some(record.clone());
        Ok(())
    }

    fn last(&self) -> Result<Option<RoundRecord>> {
        Ok(self.lock()?.1.clone())
    }
}

impl MemoryRoundStore {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, (u64, Option<RoundRecord>)>> {
        self.state
            .lock()
            .map_err(|_| anyhow::anyhow!("round store lock poisoned"))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(error).with_context(|| format!("cannot read {}", path.display()));
        }
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("{} is not valid controller state", path.display()))
}

fn write_json_atomic<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let dir = path
        .parent()
        .with_context(|| format!("{} has no parent directory", path.display()))?;
    fs::create_dir_all(dir)
        .with_context(|| format!("cannot create state directory {}", dir.display()))?;
    let name = path
        .file_name()
        .with_context(|| format!("{} has no file name", path.display()))?;
    let mut tmp_name = name.to_os_string();
    tmp_name.push(format!(".tmp.{}", std::process::id()));
    let tmp = dir.join(tmp_name);

    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let mut file =
        File::create(&tmp).with_context(|| format!("cannot create {}", tmp.display()))?;
    file.write_all(&bytes)
        .and_then(|()| file.sync_all())
        .with_context(|| format!("cannot write {}", tmp.display()))?;
    drop(file);
    fs::rename(&tmp, path)
        .with_context(|| format!("cannot rename {} to {}", tmp.display(), path.display()))?;
    File::open(dir)
        .and_then(|dir| dir.sync_all())
        .with_context(|| format!("cannot sync directory {}", dir.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(epoch: u64, seq: u64) -> RoundRecord {
        RoundRecord {
            world: "w".to_string(),
            epoch,
            seq,
            request_id: request_id("w", epoch, seq),
            instruction: Instruction::ScaleDown,
            params: serde_json::json!({"removed_dp_ranks": [1]}),
            removed: vec![1],
            survivors: vec![0, 2],
            store: None,
            sent_at: timestamp(SystemTime::now()),
            not_after: timestamp(SystemTime::now()),
            next_store_port: 29650,
            escalated: false,
        }
    }

    #[test]
    fn request_ids_name_world_epoch_and_sequence() {
        let id = request_id("world-a", 3, 12);
        let suffix = id.strip_prefix("world-a-e3-r12-").expect(&id);
        assert_eq!(suffix.len(), 8);
        assert!(suffix.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn epochs_grow_across_restarts_and_rounds() {
        let parent = tempfile::tempdir().unwrap();
        let dir = parent.path().join("state");
        let store = FileRoundStore::new(&dir, "w");
        assert_eq!(store.last().unwrap(), None);
        assert!(!dir.exists(), "reading must not create the directory");
        assert_eq!(store.begin_epoch().unwrap(), 1);
        assert_eq!(store.begin_epoch().unwrap(), 2);

        // A round record from a later epoch (say, restored from elsewhere)
        // still moves the next epoch past it.
        let saved = record(7, 1);
        store.save(&saved).unwrap();
        let reopened = FileRoundStore::new(&dir, "w");
        assert_eq!(reopened.last().unwrap(), Some(saved));
        assert_eq!(reopened.begin_epoch().unwrap(), 8);

        // Only the two state files remain: no temporary files.
        let mut names: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["w.epoch.json", "w.round.json"]);
    }

    #[test]
    fn records_without_the_newer_fields_still_load() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileRoundStore::new(dir.path(), "w");
        let mut value = serde_json::to_value(record(1, 1)).unwrap();
        let object = value.as_object_mut().unwrap();
        object.remove("next_store_port");
        object.remove("escalated");
        fs::write(store.round_path(), value.to_string()).unwrap();
        let loaded = store.last().unwrap().unwrap();
        assert_eq!(loaded.next_store_port, 0);
        assert!(!loaded.escalated);
    }

    #[test]
    fn corrupt_state_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileRoundStore::new(dir.path(), "w");
        fs::write(store.round_path(), b"{not json").unwrap();
        let error = store.begin_epoch().unwrap_err();
        assert!(format!("{error:#}").contains("not valid controller state"));
    }

    #[test]
    fn memory_store_counts_epochs() {
        let store = MemoryRoundStore::default();
        assert_eq!(store.begin_epoch().unwrap(), 1);
        assert_eq!(store.begin_epoch().unwrap(), 2);
        store.save(&record(2, 1)).unwrap();
        assert_eq!(store.last().unwrap().map(|r| r.seq), Some(1));
    }
}
