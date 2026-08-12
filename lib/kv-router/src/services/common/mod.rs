// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(any(
    feature = "standalone-indexer",
    feature = "standalone-slot-tracker",
    feature = "standalone-selection"
))]
pub(crate) mod zmq;

#[cfg(any(feature = "standalone-slot-tracker", feature = "standalone-selection"))]
pub(crate) mod replica_sync;

#[cfg(feature = "standalone-slot-tracker")]
pub(crate) mod replica_sync_http;
