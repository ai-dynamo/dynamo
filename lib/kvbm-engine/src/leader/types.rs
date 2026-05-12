// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use futures::future::{BoxFuture, Either, Ready, ready};
use serde::{Deserialize, Serialize};
use tokio::sync::{Mutex, watch};

use std::sync::Arc;

use crate::{G2, G3};
use kvbm_logical::blocks::ImmutableBlock;

use super::onboarding::{OnboardingStatus, SessionHandle};
use super::session::SessionId;

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
#[derive(Debug)]
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
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchBreakdown {
    pub host_blocks: usize,
    pub disk_blocks: usize,
    pub object_blocks: usize,
}

#[derive(Debug)]
pub struct ReadyResult {
    /// G2 blocks held directly via RAII
    blocks: Vec<ImmutableBlock<G2>>,
    /// G3 blocks held directly via RAII (host-bypass mode).
    ///
    /// In standard mode this is always empty — disk hits become G2 blocks
    /// after `stage_g3_to_g2` runs in the AsyncSession path. Populated only
    /// when host-bypass is in effect and disk hits are returned for direct
    /// G3→G1 onboarding.
    g3_blocks: Vec<ImmutableBlock<G3>>,
    breakdown: MatchBreakdown,
}

impl ReadyResult {
    /// Create a new ready result with G2 blocks (standard mode).
    pub fn new(blocks: Vec<ImmutableBlock<G2>>, breakdown: MatchBreakdown) -> Self {
        Self {
            blocks,
            g3_blocks: Vec::new(),
            breakdown,
        }
    }

    /// Create a new ready result with G2 + G3 blocks (host-bypass mode).
    pub fn new_with_g3(
        g2_blocks: Vec<ImmutableBlock<G2>>,
        g3_blocks: Vec<ImmutableBlock<G3>>,
        breakdown: MatchBreakdown,
    ) -> Self {
        Self {
            blocks: g2_blocks,
            g3_blocks,
            breakdown,
        }
    }

    /// Number of G2 blocks held.
    pub fn g2_count(&self) -> usize {
        self.blocks.len()
    }

    /// Number of G3 blocks held (host-bypass mode).
    pub fn g3_count(&self) -> usize {
        self.g3_blocks.len()
    }

    /// Total matched blocks across all tiers (G2 + G3).
    pub fn total_count(&self) -> usize {
        self.g2_count() + self.g3_count()
    }

    /// Take ownership of the G2 blocks.
    ///
    /// After calling this, the ReadyResult's G2 list will be empty.
    pub fn take_g2_blocks(&mut self) -> Vec<ImmutableBlock<G2>> {
        std::mem::take(&mut self.blocks)
    }

    /// Take ownership of the G3 blocks (host-bypass mode).
    ///
    /// After calling this, the ReadyResult's G3 list will be empty.
    pub fn take_g3_blocks(&mut self) -> Vec<ImmutableBlock<G3>> {
        std::mem::take(&mut self.g3_blocks)
    }

    /// Get a reference to the G2 blocks.
    pub fn blocks(&self) -> &[ImmutableBlock<G2>] {
        &self.blocks
    }

    /// Get a reference to the G3 blocks (host-bypass mode).
    pub fn g3_blocks(&self) -> &[ImmutableBlock<G3>] {
        &self.g3_blocks
    }

    pub fn match_breakdown(&self) -> MatchBreakdown {
        self.breakdown
    }
}

/// Async session result for staging and/or remote search operations.
#[derive(Debug)]
pub struct AsyncSessionResult {
    session_id: SessionId,
    status_rx: watch::Receiver<OnboardingStatus>,
    blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
    match_breakdown: Arc<Mutex<MatchBreakdown>>,
    session_handle: Option<SessionHandle>,
}

impl AsyncSessionResult {
    /// Create a new async session result.
    pub fn new(
        session_id: SessionId,
        status_rx: watch::Receiver<OnboardingStatus>,
        blocks: Arc<Mutex<Option<Vec<ImmutableBlock<G2>>>>>,
        match_breakdown: Arc<Mutex<MatchBreakdown>>,
        session_handle: Option<SessionHandle>,
    ) -> Self {
        Self {
            session_id,
            status_rx,
            blocks,
            match_breakdown,
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

    pub fn match_breakdown(&self) -> MatchBreakdown {
        self.match_breakdown
            .try_lock()
            .map(|v| *v)
            .unwrap_or_default()
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

    /// Take ownership of G3 blocks if available (host-bypass mode).
    ///
    /// Only Ready results carry G3 blocks; AsyncSession returns `None` because
    /// remote-search / staging paths don't currently populate the G3 holder.
    pub fn take_g3_blocks(&mut self) -> Option<Vec<ImmutableBlock<G3>>> {
        match self {
            FindMatchesResult::Ready(r) => Some(r.take_g3_blocks()),
            FindMatchesResult::AsyncSession(_) => None,
        }
    }

    pub fn match_breakdown(&self) -> MatchBreakdown {
        match self {
            FindMatchesResult::Ready(r) => r.match_breakdown(),
            FindMatchesResult::AsyncSession(a) => a.match_breakdown(),
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
