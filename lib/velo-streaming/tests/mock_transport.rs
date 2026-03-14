// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for MockFrameTransport (TEST-11, TEST-13).
//!
//! These tests validate the trait boundary and core state-machine assertions
//! against an in-memory transport — no network, no Messenger required.

mod common;

use std::sync::Arc;
use velo_common::WorkerId;
use velo_streaming::AnchorManager;

use common::MockFrameTransport;

run_transport_tests!(mock_suite, {
    Arc::new(AnchorManager::new(
        WorkerId::from_u64(1),
        Arc::new(MockFrameTransport::new()),
    ))
});
