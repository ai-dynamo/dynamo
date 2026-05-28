// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Peer discovery for the velo-based standalone indexer.
//!
//! The indexer service publishes its `PeerInfo` as a JSON snapshot
//! (`kv-indexer.json`) to a well-known filesystem directory.  Workers and
//! routers read this file via [`connect_to_indexer`] and call
//! `messenger.register_peer(peer_info)` to open a connection.
//!
//! **Note:** this is *not* velo's built-in `FilesystemPeerDiscovery`.  The
//! on-disk format is our own [`IndexerPeerSnapshot`] JSON and is incompatible
//! with velo's `PeerRegistry` file format.  Do not pass `discovery_dir` to
//! `FilesystemPeerDiscovery::new` on either end.
//!
//! ## Discovery flow
//!
//! ```text
//!   indexer process                    worker process
//!   ───────────────────                ──────────────────────────────────────
//!   IndexerDiscovery::publish()  →     (kv-indexer.json written to dir)
//!                                      connect_to_indexer(&messenger, dir)
//!                                        └─ reads kv-indexer.json
//!                                        └─ messenger.register_peer(peer_info)
//!                                      ──► messenger can now address indexer
//! ```
//!
//! ## Out-of-band registration (no shared filesystem)
//!
//! Call [`IndexerDiscovery::peer_info_bytes`] to serialise the indexer's
//! `PeerInfo` to MessagePack bytes.  Ship those bytes via an env var or config
//! file, then call `messenger.register_peer(rmp_serde::from_slice(&bytes)?)` on
//! the worker side.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use velo::{InstanceId, Messenger, PeerInfo};

/// A serialisable snapshot of the indexer's velo peer information.
///
/// Workers and routers receive this (via a config file, env var, or filesystem
/// peer discovery) and call `messenger.register_peer(peer_info)` to connect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexerPeerSnapshot {
    /// The velo `WorkerId` of the indexer
    /// (`messenger.instance_id().worker_id().as_u64()`).
    ///
    /// This is a diagnostic/display field used for logging and introspection.
    /// The authoritative `InstanceId` for addressing the indexer is obtained by
    /// calling [`IndexerPeerSnapshot::peer_info`] and then `peer_info.instance_id()`.
    pub worker_id: u64,
    /// Serialised [`PeerInfo`] bytes (MessagePack).
    pub peer_info_bytes: Vec<u8>,
}

impl IndexerPeerSnapshot {
    fn from_messenger(messenger: &Messenger) -> Result<Self> {
        let peer_info = messenger.peer_info();
        let peer_info_bytes = rmp_serde::to_vec(&peer_info)
            .context("failed to serialise PeerInfo to MessagePack")?;
        Ok(Self {
            worker_id: messenger.instance_id().worker_id().as_u64(),
            peer_info_bytes,
        })
    }

    /// Deserialise the embedded `PeerInfo`.
    pub fn peer_info(&self) -> Result<PeerInfo> {
        rmp_serde::from_slice(&self.peer_info_bytes).context("failed to deserialise PeerInfo")
    }
}

/// Filesystem path inside a discovery directory where the indexer writes its
/// `PeerInfo` snapshot.
const INDEXER_PEER_FILE: &str = "kv-indexer.json";

/// Manages publication and revocation of the indexer's velo peer info.
///
/// Drop this handle to remove the published peer-info file.
pub struct IndexerDiscovery {
    peer_file: PathBuf,
    messenger: Arc<Messenger>,
}

impl IndexerDiscovery {
    /// Publish the messenger's `PeerInfo` to `discovery_dir/kv-indexer.json`.
    ///
    /// Workers should call [`connect_to_indexer`] (not velo's built-in
    /// `FilesystemPeerDiscovery`) to read this file and register the indexer
    /// peer.  The on-disk format is [`IndexerPeerSnapshot`] JSON, which is
    /// incompatible with velo's `PeerRegistry` format.
    ///
    /// `discovery_dir` is created if it does not exist.
    pub fn publish(messenger: Arc<Messenger>, discovery_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(discovery_dir).with_context(|| {
            format!(
                "failed to create peer-discovery directory `{}`",
                discovery_dir.display()
            )
        })?;

        let peer_file = discovery_dir.join(INDEXER_PEER_FILE);
        let snapshot = IndexerPeerSnapshot::from_messenger(&messenger)?;
        let json =
            serde_json::to_vec_pretty(&snapshot).context("failed to serialise peer snapshot")?;

        // Write atomically via a sibling temp file so readers never observe a
        // partial document (std::fs::write truncates then writes; a reader that
        // opens the file between those two steps sees an empty or partial JSON).
        let tmp_file = discovery_dir.join(format!(".{INDEXER_PEER_FILE}.tmp"));
        std::fs::write(&tmp_file, &json).with_context(|| {
            format!(
                "failed to write peer snapshot to tmp file `{}`",
                tmp_file.display()
            )
        })?;
        std::fs::rename(&tmp_file, &peer_file).with_context(|| {
            format!(
                "failed to rename tmp peer snapshot to `{}`",
                peer_file.display()
            )
        })?;

        tracing::info!(
            path      = %peer_file.display(),
            worker_id = snapshot.worker_id,
            "Published indexer PeerInfo to filesystem discovery"
        );

        Ok(Self {
            peer_file,
            messenger,
        })
    }

    /// Read the published peer snapshot back (useful for testing / introspection).
    pub fn snapshot(&self) -> Result<IndexerPeerSnapshot> {
        IndexerPeerSnapshot::from_messenger(&self.messenger)
    }

    /// The velo `InstanceId` of this indexer.
    pub fn instance_id(&self) -> InstanceId {
        self.messenger.instance_id()
    }

    /// Serialise the indexer's `PeerInfo` to bytes so it can be shared
    /// out-of-band (env var, config file, …).
    pub fn peer_info_bytes(&self) -> Result<Vec<u8>> {
        let peer_info = self.messenger.peer_info();
        rmp_serde::to_vec(&peer_info).context("failed to serialise PeerInfo")
    }
}

impl Drop for IndexerDiscovery {
    fn drop(&mut self) {
        if self.peer_file.exists() {
            if let Err(e) = std::fs::remove_file(&self.peer_file) {
                tracing::warn!(
                    path = %self.peer_file.display(),
                    error = %e,
                    "Failed to remove peer-discovery file on drop"
                );
            } else {
                tracing::debug!(
                    path = %self.peer_file.display(),
                    "Removed peer-discovery file"
                );
            }
        }
    }
}

// ── Client helper ──────────────────────────────────────────────────────────

/// Read the indexer's [`IndexerPeerSnapshot`] from `discovery_dir`.
///
/// Returns `None` if the file does not exist yet (indexer not started).
///
/// Reads atomically: the file is opened directly and `ErrorKind::NotFound`
/// is mapped to `Ok(None)`.  An intervening `exists()` + `read()` sequence
/// would race with the indexer dropping or atomically replacing the file
/// during startup/shutdown and return a spurious error.
pub fn read_peer_snapshot(discovery_dir: &Path) -> Result<Option<IndexerPeerSnapshot>> {
    let path = discovery_dir.join(INDEXER_PEER_FILE);
    let json = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(anyhow::Error::new(e).context(format!(
                "failed to read peer snapshot from `{}`",
                path.display()
            )))
        }
    };
    let snapshot: IndexerPeerSnapshot =
        serde_json::from_slice(&json).context("failed to parse peer snapshot JSON")?;
    Ok(Some(snapshot))
}

/// Resolve the indexer's `PeerInfo` from `discovery_dir` and register it with
/// `messenger`.
///
/// Returns the indexer's [`InstanceId`] so the caller can address it.
///
/// Returns `None` if the indexer's peer file is not present yet.
pub fn connect_to_indexer(
    messenger: &Arc<Messenger>,
    discovery_dir: &Path,
) -> Result<Option<InstanceId>> {
    let Some(snapshot) = read_peer_snapshot(discovery_dir)? else {
        return Ok(None);
    };
    let peer_info = snapshot.peer_info()?;
    let instance_id = peer_info.instance_id();
    messenger
        .register_peer(peer_info)
        .context("failed to register indexer peer")?;
    tracing::info!(
        ?instance_id,
        discovery_dir = %discovery_dir.display(),
        "Registered indexer peer from filesystem discovery"
    );
    Ok(Some(instance_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    async fn make_messenger() -> Arc<Messenger> {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let transport: Arc<dyn velo::backend::Transport> = Arc::new(
            velo::backend::tcp::TcpTransportBuilder::new()
                .from_listener(listener)
                .unwrap()
                .build()
                .unwrap(),
        );
        Messenger::builder()
            .add_transport(transport)
            .build()
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn publish_writes_peer_file_and_drop_removes_it() {
        let dir = tempfile::tempdir().unwrap();
        let messenger = make_messenger().await;

        let discovery = IndexerDiscovery::publish(messenger.clone(), dir.path()).unwrap();
        let peer_file = dir.path().join(INDEXER_PEER_FILE);

        assert!(peer_file.exists(), "peer file should exist after publish");

        let snapshot = read_peer_snapshot(dir.path()).unwrap().unwrap();
        assert_eq!(
            snapshot.worker_id,
            messenger.instance_id().worker_id().as_u64(),
            "worker_id in snapshot should match messenger"
        );

        // PeerInfo should round-trip without error.
        let _peer_info = snapshot.peer_info().unwrap();

        drop(discovery);
        assert!(
            !peer_file.exists(),
            "peer file should be removed when IndexerDiscovery is dropped"
        );
    }

    #[tokio::test]
    async fn connect_to_indexer_registers_peer() {
        let dir = tempfile::tempdir().unwrap();
        let indexer_messenger = make_messenger().await;
        let client_messenger = make_messenger().await;

        let _discovery = IndexerDiscovery::publish(indexer_messenger.clone(), dir.path()).unwrap();

        let instance_id = connect_to_indexer(&client_messenger, dir.path())
            .unwrap()
            .expect("should find indexer");

        assert_eq!(
            instance_id,
            indexer_messenger.instance_id(),
            "resolved instance_id should match the indexer's"
        );
    }

    #[tokio::test]
    async fn connect_to_indexer_returns_none_when_not_published() {
        let dir = tempfile::tempdir().unwrap();
        let client_messenger = make_messenger().await;

        let result = connect_to_indexer(&client_messenger, dir.path()).unwrap();
        assert!(result.is_none(), "should return None before indexer starts");
    }
}
