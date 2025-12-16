// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod accessor;
mod instance;
mod nova;
pub mod session;
mod state;

pub use accessor::{BlockAccessor, PolicyContext, TieredBlock};
pub use instance::InstanceLeader;
pub use nova::NovaLeaderService;
pub use session::{
    ControllableSession, ControllableSessionOptions, ControllableSessionResult, G2BlockInfo,
    G3BlockInfo, InitiatorSession, RemoteSessionHandle, RemoteSessionMessage, RemoteSessionPhase,
    RemoteSessionState, ResponderSession, SessionId,
};
pub use state::{LeaderState, RemoteLeaderInfo, route_local_to_remote};

use anyhow::Result;
use futures::future::{BoxFuture, Either, Ready, ready};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, mpsc, watch};

use std::sync::Arc;

use crate::{
    logical::blocks::ImmutableBlock,
    v2::{G2, SequenceHash},
};

/// Leader trait for distributed block onboarding operations.
pub trait Leader: Send + Sync {
    /// Find matching blocks with default options.
    fn find_matches(&self, sequence_hashes: &[SequenceHash]) -> Result<FindMatchesResult> {
        self.find_matches_with_options(sequence_hashes, FindMatchesOptions::default())
    }

    /// Find matching blocks with custom options.
    fn find_matches_with_options(
        &self,
        sequence_hashes: &[SequenceHash],
        options: FindMatchesOptions,
    ) -> Result<FindMatchesResult>;
}

/// Staging mode for matched blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StagingMode {
    /// Hold blocks in their current tiers (G2 and G3) without staging.
    /// Session stays alive for future operations.
    /// Blocks remain on their original instances (local or remote).
    Hold,

    /// Stage all G3→G2 on local and remote instances.
    /// No RDMA pulls from remote instances.
    /// Remote blocks stay in remote G2.
    /// Session stays alive for future operations.
    Prepare,

    /// Full staging: G3→G2 everywhere, then RDMA pull remote G2→local G2.
    /// Session completes after all blocks are in local G2.
    #[default]
    Full,
}

/// Options for find_matches operation.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct FindMatchesOptions {
    /// Whether to search remote instances in addition to local search.
    /// Default: false (local only)
    pub search_remote: bool,

    /// Staging mode controlling how blocks are staged and session lifecycle.
    /// Default: StagingMode::Full
    pub staging_mode: StagingMode,
}

/// Result of a find_matches operation.
///
/// This enum has two variants:
/// - `Ready`: Immediate result when no async work is needed (local search with Hold mode)
/// - `AsyncSession`: When staging or remote search is required
pub enum FindMatchesResult {
    /// Immediate result - blocks are held in place without staging.
    ///
    /// Returned when `search_remote == false` AND `staging_mode == Hold`.
    /// Blocks remain in their original tiers (G2 or G3) on the local instance.
    Ready(ReadyResult),

    /// Async session for staging and/or remote search.
    ///
    /// Returned when:
    /// - `search_remote == true` (remote searching enabled)
    /// - OR `staging_mode` is `Prepare` or `Full` (local/remote staging)
    AsyncSession(AsyncSessionResult),
}

/// Immediate result containing matched blocks held directly.
///
/// No session is created - blocks are owned directly by this struct (RAII).
/// Dropping this struct will release the block references.
pub struct ReadyResult {
    /// G2 blocks held directly via RAII
    blocks: Vec<ImmutableBlock<G2>>,
}

impl ReadyResult {
    /// Create a new ready result with G2 blocks.
    pub fn new(blocks: Vec<ImmutableBlock<G2>>) -> Self {
        Self { blocks }
    }

    /// Number of G2 blocks held.
    pub fn g2_count(&self) -> usize {
        self.blocks.len()
    }

    /// Take ownership of the G2 blocks.
    ///
    /// After calling this, the ReadyResult will be empty.
    pub fn take_g2_blocks(&mut self) -> Vec<ImmutableBlock<G2>> {
        std::mem::take(&mut self.blocks)
    }

    /// Get a reference to the G2 blocks.
    pub fn blocks(&self) -> &[ImmutableBlock<G2>] {
        &self.blocks
    }
}

/// Async session result for staging and/or remote search operations.
pub struct AsyncSessionResult {
    session_id: SessionId,
    status_rx: watch::Receiver<OnboardingStatus>,
    blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
    session_handle: Option<SessionHandle>,
}

impl AsyncSessionResult {
    /// Create a new async session result.
    pub fn new(
        session_id: SessionId,
        status_rx: watch::Receiver<OnboardingStatus>,
        blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
        session_handle: Option<SessionHandle>,
    ) -> Self {
        Self {
            session_id,
            status_rx,
            blocks,
            session_handle,
        }
    }

    /// Get the session ID for this onboarding operation.
    pub fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Get the current status of the onboarding operation.
    pub fn status(&self) -> OnboardingStatus {
        self.status_rx.borrow().clone()
    }

    /// Get session handle for deferred operations (Hold/Prepare modes only).
    ///
    /// Returns None for StagingMode::Full.
    pub fn session_handle(&self) -> Option<&SessionHandle> {
        self.session_handle.as_ref()
    }

    /// Non-blocking check if blocks are available.
    ///
    /// Returns Some(count) if blocks are available, None if still in progress.
    /// Use wait_for_completion() to take ownership of blocks.
    pub fn get_blocks_count(&self) -> Option<usize> {
        self.blocks.try_lock().ok()?.as_ref().map(|v| v.len())
    }

    /// Wait for the operation to complete and return the matched blocks.
    ///
    /// For StagingMode::Full, waits for Complete status.
    /// For Hold/Prepare modes, waits for terminal state (Holding/Prepared/Complete).
    ///
    /// This method returns a future that can be used with tokio::select!.
    pub fn wait_for_completion(&self) -> BoxFuture<'static, Result<()>> {
        let mut status_rx = self.status_rx.clone();
        Box::pin(async move {
            // Wait for terminal status
            status_rx
                .wait_for(|status| {
                    matches!(
                        status,
                        OnboardingStatus::Complete { .. }
                            | OnboardingStatus::Holding { .. }
                            | OnboardingStatus::Prepared { .. }
                    )
                })
                .await
                .map_err(|e| anyhow::anyhow!("failed to wait for completion: {e}"))?;

            Ok(())
        })
    }
}

impl FindMatchesResult {
    /// Check if this is a ready (immediate) result.
    pub fn is_ready(&self) -> bool {
        matches!(self, FindMatchesResult::Ready(_))
    }

    /// Check if this is an async session result.
    pub fn is_async(&self) -> bool {
        matches!(self, FindMatchesResult::AsyncSession(_))
    }

    /// Get the ready result, if this is a Ready variant.
    pub fn as_ready(&self) -> Option<&ReadyResult> {
        match self {
            FindMatchesResult::Ready(r) => Some(r),
            FindMatchesResult::AsyncSession(_) => None,
        }
    }

    /// Get the ready result mutably, if this is a Ready variant.
    pub fn as_ready_mut(&mut self) -> Option<&mut ReadyResult> {
        match self {
            FindMatchesResult::Ready(r) => Some(r),
            FindMatchesResult::AsyncSession(_) => None,
        }
    }

    /// Get the async session result, if this is an AsyncSession variant.
    pub fn as_async(&self) -> Option<&AsyncSessionResult> {
        match self {
            FindMatchesResult::Ready(_) => None,
            FindMatchesResult::AsyncSession(a) => Some(a),
        }
    }

    /// Get the async session result mutably, if this is an AsyncSession variant.
    pub fn as_async_mut(&mut self) -> Option<&mut AsyncSessionResult> {
        match self {
            FindMatchesResult::Ready(_) => None,
            FindMatchesResult::AsyncSession(a) => Some(a),
        }
    }

    /// Get the number of G2 blocks available or matched.
    ///
    /// For Ready: returns the count of blocks held.
    /// For AsyncSession: returns the count if blocks are available, 0 otherwise.
    pub fn g2_count(&self) -> usize {
        match self {
            FindMatchesResult::Ready(r) => r.g2_count(),
            FindMatchesResult::AsyncSession(a) => a.get_blocks_count().unwrap_or(0),
        }
    }

    /// Take ownership of G2 blocks if available.
    ///
    /// For Ready: always succeeds, returns the blocks.
    /// For AsyncSession: returns Some if blocks are available and lock succeeds.
    pub fn take_g2_blocks(&mut self) -> Option<Vec<ImmutableBlock<G2>>> {
        match self {
            FindMatchesResult::Ready(r) => Some(r.take_g2_blocks()),
            FindMatchesResult::AsyncSession(a) => a.blocks.try_lock().ok()?.take(),
        }
    }

    pub fn session_id(&self) -> Option<SessionId> {
        match self {
            FindMatchesResult::Ready(_) => None,
            FindMatchesResult::AsyncSession(a) => Some(a.session_id()),
        }
    }

    /// Wait for the operation to complete.
    ///
    /// For Ready variant: returns immediately with Ok(()).
    /// For AsyncSession variant: waits for terminal status (Complete/Holding/Prepared).
    ///
    /// Returns an Either future that can be used with tokio::select!.
    pub fn wait_for_completion(&self) -> Either<Ready<Result<()>>, BoxFuture<'static, Result<()>>> {
        match self {
            FindMatchesResult::Ready(_) => Either::Left(ready(Ok(()))),
            FindMatchesResult::AsyncSession(async_session) => {
                Either::Right(async_session.wait_for_completion())
            }
        }
    }
}

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
