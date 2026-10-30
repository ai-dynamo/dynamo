// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV indexer stage, re-exported at its old paths. The index structures, their
//! traits, counters, residency projection, and test helpers live in `dynamo-kv-indexer`;
//! this module keeps the request-plane DTOs (`api`), the lower tiers, and the local
//! indexer until their own slices move.

mod api;
#[cfg(test)]
mod delegate_tests;
mod local;
#[cfg(test)]
mod local_tests;
mod lower_tier;
mod lower_tier_indexers;
#[cfg(feature = "runtime-protocols")]
mod runtime_metrics;

pub use dynamo_kv_indexer::*;

pub use api::*;
pub use local::*;
pub use lower_tier::*;
pub use lower_tier_indexers::*;
#[cfg(feature = "runtime-protocols")]
pub use runtime_metrics::KvIndexerMetricsExt;

// Kernel hook; lives with the scheduler, reachable here for existing callers.
pub use crate::scheduling::shared_cache::SharedKvCache;
