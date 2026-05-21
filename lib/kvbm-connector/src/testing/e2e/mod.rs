// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for connector instances.
//!
//! This module provides E2E tests using TestConnectorCluster to test
//! multi-instance scenarios like bidirectional transfers.

// These E2E tests build TestConnectorClusters (NIXL/UCX + CUDA required), so
// they are gated behind `testing-nixl` and skipped by the CPU pre-merge job.
#[cfg(all(test, feature = "testing-nixl"))]
mod find_blocks;

#[cfg(all(test, feature = "s3"))]
mod s3_object;
