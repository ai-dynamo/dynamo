// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! KV events in, prefix knowledge out: the Dynamo router's KV indexer stage.
//!
//! Lands in slices. This slice holds the cuckoo-filter pool index (`cuckoo`), the hash
//! lifecycle delegate (`delegate`), and the indexer counters (`metrics`). The trees, the
//! lower tiers, and the ingest backend follow. Until then `lib/kv-router` re-exports every
//! item here at its old path.
//!
//! Depends only on `dynamo-router-protocol`, `dynamo-kv-events`, and `dynamo-kv-hashing`.
//! The `export` feature (Prometheus counter vectors) is transitional: the design replaces it
//! with atomics and a `stats()` snapshot in the metrics behavior change.

pub mod cuckoo;
pub mod delegate;
pub mod metrics;

pub use delegate::{HashLifecycle, KvIndexerDelegate};
