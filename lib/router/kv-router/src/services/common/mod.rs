// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(any(feature = "ingest-zmq", feature = "replica-sync-zmq"))]
pub(crate) mod zmq;

#[cfg(all(feature = "http", feature = "replica-sync-zmq"))]
pub(crate) mod http;

// The HTTP routes use the rest of the mesh API.
#[cfg(feature = "replica-sync-zmq")]
#[cfg_attr(not(feature = "http"), allow(dead_code))]
pub(crate) mod replica_sync;

#[cfg(all(feature = "http", feature = "replica-sync-zmq"))]
pub(crate) mod replica_sync_http;
