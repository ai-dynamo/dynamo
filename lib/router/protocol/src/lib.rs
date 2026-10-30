// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The identities and worker-facing vocabulary every Dynamo router crate names.
//!
//! A type enters this crate only if it is serialized across a process boundary to an
//! engine, a worker, or a sidecar, or is an identity every crate names. Nothing here
//! runs: no tasks, no IO, no traits with behavior, no hashing. Dependencies are exactly
//! `serde`, `serde_json`, and `thiserror`, and the layering gate checks that by equality.
//!
//! Deliberately elsewhere: KV event types and their engine codec (`dynamo-kv-events`),
//! the hash functions (`dynamo-kv-hashing`), the scheduling request, eligibility, and
//! host hooks (`dynamo-scheduler`), candidate rows (`dynamo-worker-selection`), and
//! queue rejection (`dynamo-admission`).

pub mod attempt;
pub mod block_hash;
pub mod identity;
pub mod kv_hints;
pub mod load;
pub mod metadata;
pub mod mm;
pub mod residency;
pub mod session;
pub mod worker;
pub mod worker_type;

pub use worker_type::WorkerType;
