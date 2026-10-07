// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY (campaign e2e-indexer-contention-20261006).
//!
//! Phantom KV-event publishers and a side query driver that load one serving indexer
//! (`python -m dynamo.router --serve-indexer`) through its production interfaces: direct-ZMQ
//! KV event envelopes (accepted via the `DYN_EXPERIMENT_STATIC_KV_SOURCES` patch) and the
//! `kv_indexer_query` request-plane endpoint. See `README.md`.

pub mod plan;
pub mod stream;

#[cfg(feature = "bins")]
pub mod pipeline;
#[cfg(feature = "bins")]
pub mod stats;
