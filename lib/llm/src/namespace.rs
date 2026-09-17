// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Namespace filtering for model discovery.
//!
//! The definitions live in `dynamo-runtime` so that crates below `dynamo-llm` in the
//! dependency graph — `dynamo-rl`, in particular — can scope their own discovery the
//! same way model discovery does. They are re-exported here so `crate::namespace::…`
//! keeps working throughout this crate.

pub use dynamo_runtime::namespace::{GLOBAL_NAMESPACE, NamespaceFilter, is_global_namespace};
