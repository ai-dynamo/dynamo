// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The KV indexer stage, re-exported at its old paths. Everything lives in
//! `dynamo-kv-indexer`; this module adds the runtime-backed counter constructor and the
//! kernel's shared-cache hook for existing callers.

pub use dynamo_kv_indexer::*;

// Kernel hook; lives with the scheduler, reachable here for existing callers.
pub use crate::scheduling::shared_cache::SharedKvCache;
