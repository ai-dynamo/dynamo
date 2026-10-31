// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(any(feature = "ingest-zmq", feature = "replica-sync-zmq"))]
pub(crate) use dynamo_zmq as zmq;

#[cfg(all(feature = "http", feature = "replica-sync-zmq"))]
pub(crate) mod http;

#[cfg(feature = "replica-sync-zmq")]
#[cfg_attr(
    not(any(
        feature = "http",
        all(feature = "ingest-zmq", feature = "peer-recovery")
    )),
    allow(dead_code)
)]
pub(crate) mod replica_sync;

#[cfg(all(feature = "http", feature = "replica-sync-zmq"))]
pub(crate) mod replica_sync_http;
