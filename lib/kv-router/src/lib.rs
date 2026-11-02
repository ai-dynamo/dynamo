// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The composition crate of the router: it assembles the scheduler kernel and the stage
//! crates into a running router and is the one crate plugins and standalone consumers name.
//!
//! - [`services::selection`]: `SelectionService` and `SelectionServiceBuilder`, the composed
//!   router an embedding host or the standalone service runs.
//! - [`plugins`]: the plugin registry (`RouterPluginRegistry`, `RouterPlugins`) and the
//!   worker-selection and request-classifier contracts; [`plugins::prelude`] is the single
//!   import a statically linked plugin needs.
//! - [`config`]: the flat `KvRouterConfig` with its CLI, environment, and policy-file parsers,
//!   converted once into the kernel's validated configuration; [`scheduling::policy_config`]
//!   is the YAML policy file and its resolver.
//! - [`tracking_hash`]: the tracking-hash key file read and the context built from the flat
//!   configuration.
//! - Adapters behind features: the HTTP servers (`http`), the ZMQ KV-event listener and its
//!   registry (`ingest-zmq`), the replica mesh (`replica-sync-zmq`), the indexer peer-recovery
//!   client (`peer-recovery`), and Prometheus exposition (`metrics`), under [`services`].
//!
//! Hosts import the owning crate's `host` module (`dynamo_scheduler::host`,
//! `dynamo_kv_indexer::host`, `dynamo_slot_tracker::host`, `dynamo_session_affinity::host`,
//! `dynamo_worker_selection::host`, `dynamo_admission::host`) for stage and kernel items; see
//! the README's migration map for the paths this crate no longer re-exports.

pub mod plugins;
pub mod scheduling;
pub mod services;
pub mod tracking_hash;

pub use scheduling::config;

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;
