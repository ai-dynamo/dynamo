// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
