// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! System-level distributed profiler for Dynamo inference benchmarking.
//!
//! `sysprofile` captures per-request traces across all Dynamo components
//! (frontend, router, engine, KVBM) during benchmark runs, merges them
//! into a single time-aligned Perfetto trace, and produces an HTML report
//! identifying system-level bottlenecks.
//!
//! # Gating
//!
//! | `DYN_SYSPROFILE_ENABLE` | Effect |
//! |-------------------------|--------|
//! | unset / `0`             | All hooks compile to no-ops; zero overhead |
//! | `1` / `true`            | Traces written to `DYN_SYSPROFILE_DIR` |
//!
//! # Usage
//!
//! ```rust,ignore
//! // Function-call form (RAII guard)
//! let _r = sysprofile::range("dynamo.frontend.recv");
//!
//! // With traceparent correlation
//! let _r = sysprofile::range_with("dynamo.prefill.compute", traceparent);
//! ```

pub mod config;
pub mod perfetto;
pub mod range;
pub mod writer;

pub use config::SysprofileConfig;
pub use range::{RangeGuard, range, range_with};
