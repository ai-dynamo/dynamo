// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Velo-based transport runtime for the standalone KV cache indexer.
//!
//! This module provides the process shell and custom filesystem peer
//! discovery.  Query and ingest handlers are intentional no-op stubs
//! that land in follow-up MRs.
//!
//! The existing `indexer-runtime` (HTTP + ZMQ) path is untouched.

pub mod discovery;
pub mod query_engine;
pub mod subscriber;
