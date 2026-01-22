// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;
use futures::future::{Either, Ready, ready};
use serde::{Deserialize, Serialize};
use std::{
    pin::Pin,
    task::{Context, Poll},
};

use dynamo_nova::events::LocalEventWaiter;

pub use crate::{
    distributed::worker::{ImportMetadataResponseAwaiter, SerializedResponseAwaiter},
    physical::manager::{LayoutHandle, SerializedLayout},
    v2::{BlockId, SequenceHash, logical::LogicalLayoutHandle},
};

pub struct SerializedLayoutResponse {
    awaiter: Either<Ready<Result<SerializedLayout>>, SerializedResponseAwaiter>,
}

impl SerializedLayoutResponse {
    pub fn ready(layout: SerializedLayout) -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(layout))),
        }
    }

    pub fn from_boxed(awaiter: SerializedResponseAwaiter) -> Self {
        Self {
            awaiter: Either::Right(awaiter),
        }
    }

    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }
}

impl std::future::IntoFuture for SerializedLayoutResponse {
    type Output = Result<SerializedLayout>;
    type IntoFuture = Either<Ready<Result<SerializedLayout>>, SerializedResponseAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

pub struct ImportMetadataResponse {
    awaiter: Either<Ready<Result<Vec<LayoutHandle>>>, ImportMetadataResponseAwaiter>,
}

impl ImportMetadataResponse {
    pub fn ready(handles: Vec<LayoutHandle>) -> Self {
        Self {
            awaiter: Either::Left(ready(Ok(handles))),
        }
    }

    pub fn from_boxed(awaiter: ImportMetadataResponseAwaiter) -> Self {
        Self {
            awaiter: Either::Right(awaiter),
        }
    }

    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, Either::Right(_))
    }
}

impl std::future::IntoFuture for ImportMetadataResponse {
    type Output = Result<Vec<LayoutHandle>>;
    type IntoFuture = Either<Ready<Result<Vec<LayoutHandle>>>, ImportMetadataResponseAwaiter>;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

/// Response type for `connect_remote` operations.
///
/// This type represents the completion state of a remote metadata import
/// with handle mapping storage. Like other response types, it can be awaited.
///
/// For direct workers, this is typically ready immediately.
/// For replicated workers, this aggregates multiple underlying imports.
pub struct ConnectRemoteResponse {
    awaiter: ConnectRemoteAwaiter,
}

pub enum ConnectRemoteAwaiter {
    Ready(Ready<Result<()>>),
    Event(LocalEventWaiter),
}

impl std::future::Future for ConnectRemoteAwaiter {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Ready(ready) => Pin::new(ready).poll(cx),
            Self::Event(waiter) => Pin::new(waiter).poll(cx),
        }
    }
}

impl ConnectRemoteResponse {
    /// Create a response that is already completed.
    ///
    /// This is used when the connect operation completes synchronously,
    /// such as for DirectWorker with local metadata import.
    pub fn ready() -> Self {
        Self {
            awaiter: ConnectRemoteAwaiter::Ready(ready(Ok(()))),
        }
    }

    /// Create a response from an event waiter.
    ///
    /// This is used when the connect operation requires waiting for
    /// multiple underlying operations to complete (e.g., ReplicatedWorker).
    pub fn from_awaiter(awaiter: LocalEventWaiter) -> Self {
        Self {
            awaiter: ConnectRemoteAwaiter::Event(awaiter),
        }
    }

    /// Check if the response can yield the current task.
    pub fn could_yield(&self) -> bool {
        matches!(self.awaiter, ConnectRemoteAwaiter::Event(_))
    }
}

impl std::future::IntoFuture for ConnectRemoteResponse {
    type Output = Result<()>;
    type IntoFuture = ConnectRemoteAwaiter;

    fn into_future(self) -> Self::IntoFuture {
        self.awaiter
    }
}

/// Remote descriptor for transfer operations.
#[derive(Serialize, Deserialize, Clone)]
pub enum RemoteDescriptor {
    Layout {
        handle: LayoutHandle,
        block_ids: Vec<BlockId>,
    },
    Object {
        keys: Vec<SequenceHash>,
    },
}

/// Configuration sent from leader to workers for G2/G3/G4 layout creation.
///
/// This message is sent via Nova RPC during Phase 3 coordination.
/// Workers use this to create additional cache tiers beyond G1 (GPU KV).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderLayoutConfig {
    /// Leader provided rank of this worker
    ///
    /// The Connector framework provides us with an ordered list of workers. To ensure
    /// leaders and workers are all in-sync on this information, the leader will send
    /// each worker the rank provided by the Connector framework.
    pub rank: usize,

    /// Number of host/pinned blocks for G2 tier.
    pub host_block_count: usize,

    /// Number of disk blocks for G3 tier (None = no disk tier).
    pub disk_block_count: Option<usize>,

    /// Object storage configuration for G4 tier (None = no object tier).
    ///
    /// When present, workers should instantiate object clients for storing
    /// blocks in external object storage (S3/MinIO).
    #[serde(default)]
    pub object: Option<dynamo_kvbm_config::ObjectConfig>,

    /// Parallelism mode for this worker.
    ///
    /// When `ReplicatedData` and rank > 0, the worker skips G2/G3 creation
    /// since only rank 0 has host/disk storage in replicated mode.
    #[serde(default)]
    pub parallelism: dynamo_kvbm_config::ParallelismMode,
}

/// Worker's response after configuring additional layouts (G2, G3).
///
/// Returned in response to a `LeaderLayoutConfig` request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerLayoutResponse {
    /// Full exported metadata including all registered layouts (G1, G2, G3).
    pub metadata: SerializedLayout,

    /// Which logical layouts were successfully created in this operation.
    pub created_layouts: Vec<LogicalLayoutHandle>,
}
