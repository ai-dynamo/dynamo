// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-role compatibility exports.
//!
//! The canonical type lives in `dynamo-kv-router` so policy providers and all router hosts can
//! share it without introducing a dependency cycle.

pub use dynamo_kv_router::worker_type::{ParseWorkerTypeError, WorkerType};
