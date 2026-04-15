// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use tokio::sync::mpsc;

use super::session::SessionId;
use super::types::StagingMode;

/// Status of an onboarding operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OnboardingStatus {
    /// Searching for blocks (local or remote).
    Searching,

    /// Holding blocks without staging (StagingMode::Hold).
    /// Provides location breakdown for cost analysis.
    /// - `local_g2`: number of blocks in local G2 (ready to use)
    /// - `local_g3`: number of blocks in local G3 (needs local staging)
    /// - `remote_g2`: number of blocks in remote G2 (needs RDMA pull)
    /// - `remote_g3`: number of blocks in remote G3 (needs remote staging + RDMA)
    /// - `pending_g4`: number of blocks with G4 load in progress
    /// - `loaded_g4`: number of blocks successfully loaded from G4 (included in local_g2)
    /// - `failed_g4`: number of blocks that failed to load from G4
    Holding {
        local_g2: usize,
        local_g3: usize,
        remote_g2: usize,
        remote_g3: usize,
        pending_g4: usize,
        loaded_g4: usize,
        failed_g4: usize,
    },

    /// Preparing: staging G3→G2 (StagingMode::Prepare or Full).
    /// - `matched`: total number of blocks matched during search
    /// - `staging_local`: number of local G3→G2 transfers in progress
    /// - `staging_remote`: number of remote G3→G2 transfers in progress
    Preparing {
        matched: usize,
        staging_local: usize,
        staging_remote: usize,
    },

    /// Prepared: all blocks in G2, session still alive (StagingMode::Prepare).
    /// - `local_g2`: number of blocks in local G2
    /// - `remote_g2`: number of blocks in remote G2 instances
    Prepared { local_g2: usize, remote_g2: usize },

    /// Staging: full mode with RDMA pulls (StagingMode::Full).
    /// - `matched`: total number of blocks matched
    /// - `staging_local`: local G3→G2 in progress
    /// - `staging_remote`: remote G3→G2 in progress
    /// - `pulling`: remote G2→local G2 (RDMA) in progress
    Staging {
        matched: usize,
        staging_local: usize,
        staging_remote: usize,
        pulling: usize,
    },

    /// Operation complete - all blocks are in initiator's G2 (StagingMode::Full).
    /// Or terminal state for Hold/Prepare modes.
    /// - `matched`: total number of blocks in local G2
    Complete { matched_blocks: usize },
}

/// Control commands for managing live sessions.
#[derive(Debug)]
pub(crate) enum SessionControl {
    /// Trigger prepare operation (Hold → Prepare): stage all G3→G2
    Prepare,

    /// Trigger pull operation (Prepare → Full): RDMA pull remote G2→local G2
    Pull,

    /// Cancel session and release all blocks
    Cancel,

    /// Shutdown session (normal completion)
    Shutdown,
}

/// Handle to a live onboarding session for deferred operations.
///
/// Only available for StagingMode::Hold and StagingMode::Prepare.
#[derive(Debug)]
pub struct SessionHandle {
    session_id: SessionId,
    mode: StagingMode,
    control_tx: mpsc::Sender<SessionControl>,
}

impl SessionHandle {
    pub(crate) fn new(
        session_id: SessionId,
        mode: StagingMode,
        control_tx: mpsc::Sender<SessionControl>,
    ) -> Self {
        Self {
            session_id,
            mode,
            control_tx,
        }
    }

    /// Get the session ID.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the current staging mode.
    pub fn mode(&self) -> StagingMode {
        self.mode
    }

    /// Trigger G3→G2 staging on all instances (Hold → Prepare).
    ///
    /// The server validates that the session is in Hold mode before processing.
    /// After this completes, the session transitions to Prepare mode internally.
    pub async fn prepare(&self) -> Result<()> {
        self.control_tx
            .send(SessionControl::Prepare)
            .await
            .map_err(|_| anyhow::anyhow!("session task has exited"))
    }

    /// Trigger RDMA pull from remote G2→local G2 (Prepare → Complete).
    ///
    /// The server validates that the session is in Prepare mode before processing.
    /// After this completes, the session transitions to Complete status.
    pub async fn pull(&self) -> Result<()> {
        self.control_tx
            .send(SessionControl::Pull)
            .await
            .map_err(|_| anyhow::anyhow!("session task has exited"))
    }

    /// Cancel session and release all held blocks.
    pub async fn cancel(&self) -> Result<()> {
        self.control_tx
            .send(SessionControl::Cancel)
            .await
            .map_err(|_| anyhow::anyhow!("session task has exited"))
    }

    /// Shutdown session (used internally).
    #[expect(dead_code)]
    pub(crate) async fn shutdown(&self) -> Result<()> {
        self.control_tx
            .send(SessionControl::Shutdown)
            .await
            .map_err(|_| anyhow::anyhow!("session task has exited"))
    }
}
